"""Generate high-performance precomputed demonstration cache for Streamlit GUI.

Runs the validated VIO + ML pipeline and associates ground truth,
saving results to artifacts/demo_trajectory_cache.npz.
"""

from pathlib import Path
import sys
import time
import numpy as np

# Ensure src is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from geonav.config.settings import ExtrinsicsConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.types import TrajectoryPoint
from geonav.hardware.health import HealthMonitor, HealthState
from geonav.ml.features import FeatureExtractor
from geonav.ml.inference import MLVelocityCorrector
from geonav.state import NavigationState
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


def generate_cache() -> Path:
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    artifacts_dir = repo_root / "artifacts"
    weights_path = artifacts_dir / "phase7_edge_mlp_weights.json"
    scaler_path = artifacts_dir / "phase7_scaler.json"
    output_path = artifacts_dir / "demo_trajectory_cache.npz"

    print(f"Loading dataset from {dataset_path}...")
    dataset = EurocDataset(dataset_path)

    vio_config = VIOConfig()
    if dataset.camera_extrinsics is not None:
        T_BS = dataset.camera_extrinsics
        R_BS, t_BS = T_BS[:3, :3], T_BS[:3, 3]
        vio_config.extrinsics = ExtrinsicsConfig(
            rotation_matrix=(
                (float(R_BS[0, 0]), float(R_BS[0, 1]), float(R_BS[0, 2])),
                (float(R_BS[1, 0]), float(R_BS[1, 1]), float(R_BS[1, 2])),
                (float(R_BS[2, 0]), float(R_BS[2, 1]), float(R_BS[2, 2])),
            ),
            translation=(float(t_BS[0]), float(t_BS[1]), float(t_BS[2])),
        )

    pipeline = VIOPipeline(config=vio_config)
    corrector = MLVelocityCorrector(
        weights_path=weights_path,
        scaler_path=scaler_path,
        max_correction_mps=3.0,
        min_confidence=0.20,
    )
    health_monitor = HealthMonitor()
    extractor = FeatureExtractor()

    print("Synchronizing streams...")
    synced = list(
        synchronize_streams(
            (img.to_camera_frame() for img in dataset.images()),
            (s.to_imu_sample() for s in dataset.imu()),
        )
    )

    n_frames = len(synced)
    print(f"Processing {n_frames} frames through VIO + ML...")

    timestamps = np.zeros(n_frames, dtype=np.float64)
    baseline_pos = np.zeros((n_frames, 3), dtype=np.float64)
    baseline_vel = np.zeros((n_frames, 3), dtype=np.float64)
    baseline_ori = np.zeros((n_frames, 4), dtype=np.float64)

    hybrid_pos = np.zeros((n_frames, 3), dtype=np.float64)
    hybrid_vel = np.zeros((n_frames, 3), dtype=np.float64)
    hybrid_ori = np.zeros((n_frames, 4), dtype=np.float64)

    delta_v = np.zeros((n_frames, 3), dtype=np.float64)
    confidence = np.zeros(n_frames, dtype=np.float64)
    num_tracked = np.zeros(n_frames, dtype=np.int32)
    num_inliers = np.zeros(n_frames, dtype=np.int32)
    health_states = []

    prev_hybrid: NavigationState = None
    t0 = time.time()

    for idx, m in enumerate(synced):
        t_prev = pipeline._current_state.timestamp if pipeline._current_state is not None else m.camera_frame.timestamp
        t_curr = m.camera_frame.timestamp
        dt = max(t_curr - t_prev, 1e-4) if pipeline._is_initialized else 0.05
        timestamps[idx] = t_curr

        # Classical step
        raw_state = pipeline.process_measurement(m)
        baseline_pos[idx] = raw_state.position
        baseline_vel[idx] = raw_state.velocity
        baseline_ori[idx] = raw_state.orientation

        # Tracked features & inliers
        if pipeline._last_visual_result is not None:
            n_track = pipeline._last_visual_result.num_tracked
            n_inl = pipeline._last_visual_result.num_inliers
            vis_ok = pipeline._last_visual_result.success
        else:
            n_track = 0
            n_inl = 0
            vis_ok = False

        num_tracked[idx] = n_track
        num_inliers[idx] = n_inl

        # ML correction
        feat = extractor.extract(
            visual_result=pipeline._last_visual_result,
            pts_prev=pipeline._prev_pts,
            pts_curr=None,
            imu_samples=m.imu_samples,
            current_state=raw_state,
            dt=dt,
        )
        dv, conf = corrector.predict(feat)
        delta_v[idx] = dv
        confidence[idx] = conf

        corrected = corrector.correct_state(prev_hybrid, raw_state, feat, dt)
        hybrid_pos[idx] = corrected.position
        hybrid_vel[idx] = corrected.velocity
        hybrid_ori[idx] = corrected.orientation
        prev_hybrid = corrected

        # Health update
        inlier_ratio = (n_inl / n_track) if n_track > 0 else 0.0
        health_info = health_monitor.update(
            timestamp=t_curr,
            vio_initialized=pipeline._is_initialized,
            tracked_features=n_track,
            inlier_ratio=inlier_ratio,
            visual_success=vis_ok,
            imu_available=len(m.imu_samples) > 0,
            ml_confidence=float(conf),
            position=raw_state.position,
            velocity=raw_state.velocity,
            orientation=raw_state.orientation,
        )
        health_states.append(health_info.state.value)

    elapsed = time.time() - t0
    print(f"Pipeline executed in {elapsed:.2f}s ({n_frames/elapsed:.1f} FPS)")

    # Ground truth association
    print("Associating with Ground Truth...")
    gt_raw = [
        TrajectoryPoint(
            s.timestamp,
            np.array(s.position, dtype=np.float64),
            np.array(s.orientation, dtype=np.float64),
            np.array(s.velocity, dtype=np.float64),
        )
        for s in dataset.ground_truth()
    ]
    p_gt0 = gt_raw[0].position
    gt_pts = [
        TrajectoryPoint(g.timestamp, g.position - p_gt0, g.orientation, g.velocity)
        for g in gt_raw
    ]
    hyb_pts = [
        TrajectoryPoint(timestamps[i], hybrid_pos[i], hybrid_ori[i], hybrid_vel[i])
        for i in range(n_frames)
    ]

    matched_pairs, _, _ = associate_timestamps(hyb_pts, gt_pts, max_time_diff_s=0.010)
    matched_gt_dict = {pair.timestamp_est: pair.gt_point for pair in matched_pairs}

    gt_pos = np.full((n_frames, 3), np.nan, dtype=np.float64)
    gt_vel = np.full((n_frames, 3), np.nan, dtype=np.float64)
    pos_errors = np.full(n_frames, np.nan, dtype=np.float64)

    for i in range(n_frames):
        ts = timestamps[i]
        if ts in matched_gt_dict:
            g = matched_gt_dict[ts]
            gt_pos[i] = g.position
            gt_vel[i] = g.velocity
            pos_errors[i] = np.linalg.norm(hybrid_pos[i] - g.position)

    print(f"Saving demonstration cache to {output_path}...")
    np.savez_compressed(
        output_path,
        timestamps=timestamps,
        baseline_pos=baseline_pos,
        baseline_vel=baseline_vel,
        baseline_ori=baseline_ori,
        hybrid_pos=hybrid_pos,
        hybrid_vel=hybrid_vel,
        hybrid_ori=hybrid_ori,
        delta_v=delta_v,
        confidence=confidence,
        num_tracked=num_tracked,
        num_inliers=num_inliers,
        health_states=np.array(health_states, dtype=object),
        gt_pos=gt_pos,
        gt_vel=gt_vel,
        pos_errors=pos_errors,
    )
    print(f"Cache saved successfully ({output_path.stat().st_size / 1024:.1f} KB).")
    return output_path


if __name__ == "__main__":
    generate_cache()
