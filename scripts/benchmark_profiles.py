"""Phase 8 edge profile benchmark — compare DEVELOPMENT vs EDGE accuracy/performance.

Runs the full VIO+ML pipeline under each operating profile and reports:
- FPS, mean latency, P95 latency
- Raw ATE RMSE, RPE, velocity RMSE
- Tracking failure count
- Memory usage

Usage:
    python scripts/benchmark_profiles.py

IMPORTANT: Development-machine software benchmark.
Do NOT report results as embedded or UAV onboard hardware performance.
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geonav.config.settings import (
    ExtrinsicsConfig, OperatingProfile, VIOConfig, get_settings_for_profile,
)
from geonav.datasets.euroc import EurocDataset
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.types import TrajectoryPoint
from geonav.ml.features import FeatureExtractor
from geonav.ml.inference import MLVelocityCorrector
from geonav.state import NavigationState
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


def _path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def _ate_rmse(est: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


def _vel_rmse(est_v: np.ndarray, gt_v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((est_v - gt_v) ** 2, axis=1))))


def _rpe_time(timestamps: np.ndarray, est: np.ndarray, gt: np.ndarray, delta_s: float) -> float:
    errs = []
    n = len(timestamps)
    j = 0
    for i in range(n):
        while j < n - 1 and timestamps[j] - timestamps[i] < delta_s:
            j += 1
        if timestamps[j] - timestamps[i] >= delta_s * 0.9:
            errs.append(np.linalg.norm((est[j] - est[i]) - (gt[j] - gt[i])))
    return float(np.sqrt(np.mean(np.array(errs) ** 2))) if errs else float("nan")


def run_profile(
    profile: OperatingProfile,
    dataset_path: Path,
    corrector: MLVelocityCorrector,
    gt_pts: List[TrajectoryPoint],
) -> Dict:
    """Run VIO+ML under the given profile and compute accuracy + performance metrics."""
    settings = get_settings_for_profile(profile)
    dataset = EurocDataset(dataset_path)

    vio_config = settings.vio
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
    extractor = FeatureExtractor()
    synced = list(synchronize_streams(
        (img.to_camera_frame() for img in dataset.images()),
        (s.to_imu_sample() for s in dataset.imu()),
    ))

    profile_cfg = settings.profile_config
    hybrid_states: List[NavigationState] = []
    prev_hybrid: Optional[NavigationState] = None
    frame_latencies: List[float] = []
    tracking_failures = 0
    frame_idx = 0

    tracemalloc.start()

    t_run_start = time.perf_counter()
    for m in synced:
        t_frame = time.perf_counter()

        t_prev = pipeline._current_state.timestamp if pipeline._current_state is not None else m.camera_frame.timestamp
        t_curr = m.camera_frame.timestamp
        dt = max(t_curr - t_prev, 1e-4) if pipeline._is_initialized else 0.05

        raw_state = pipeline.process_measurement(m)

        # ML inference every ml_inference_every_n frames
        if pipeline._is_initialized and frame_idx % profile_cfg.ml_inference_every_n == 0:
            feat = extractor.extract(
                visual_result=pipeline._last_visual_result,
                pts_prev=pipeline._prev_pts,
                pts_curr=None,
                imu_samples=m.imu_samples,
                current_state=raw_state,
                dt=dt,
            )
            corrected = corrector.correct_state(prev_hybrid, raw_state, feat, dt)
            prev_hybrid = corrected
        else:
            corrected = raw_state

        hybrid_states.append(corrected)

        if pipeline._last_visual_result is not None and not pipeline._last_visual_result.success:
            tracking_failures += 1

        frame_latencies.append((time.perf_counter() - t_frame) * 1e3)
        frame_idx += 1

    total_runtime = time.perf_counter() - t_run_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fps = len(hybrid_states) / total_runtime if total_runtime > 0 else 0.0
    lat_arr = np.array(frame_latencies)

    # Accuracy evaluation
    hyb_pts = [
        TrajectoryPoint(s.timestamp, np.array(s.position), np.array(s.orientation), np.array(s.velocity))
        for s in hybrid_states
    ]
    pairs, _, _ = associate_timestamps(hyb_pts, gt_pts, max_time_diff_s=0.01)
    est_pos = np.array([p.est_point.position for p in pairs])
    gt_pos  = np.array([p.gt_point.position  for p in pairs])
    est_vel = np.array([p.est_point.velocity  for p in pairs])
    gt_vel  = np.array([p.gt_point.velocity   for p in pairs])
    ts      = np.array([p.timestamp_est        for p in pairs])

    ate_rmse = _ate_rmse(est_pos, gt_pos)
    vel_rmse = _vel_rmse(est_vel, gt_vel)
    path_est = _path_length(est_pos)
    path_gt  = _path_length(gt_pos)
    rpe_1s   = _rpe_time(ts, est_pos, gt_pos, 1.0)
    rpe_5s   = _rpe_time(ts, est_pos, gt_pos, 5.0)

    return {
        "profile": profile.value,
        "benchmark_type": "Development-machine software benchmark",
        "frames_processed": len(hybrid_states),
        "total_runtime_s": round(total_runtime, 2),
        "fps": round(fps, 1),
        "mean_latency_ms": round(float(np.mean(lat_arr)), 3),
        "median_latency_ms": round(float(np.median(lat_arr)), 3),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "peak_memory_mb": round(peak / (1024 * 1024), 1),
        "tracking_failures": tracking_failures,
        "raw_ate_rmse_m": round(ate_rmse, 3),
        "velocity_rmse_mps": round(vel_rmse, 3),
        "estimated_path_m": round(path_est, 3),
        "gt_path_m": round(path_gt, 3),
        "scale_ratio": round(path_est / path_gt if path_gt > 0 else float("nan"), 3),
        "rpe_1s_rmse_m": round(rpe_1s, 3),
        "rpe_5s_rmse_m": round(rpe_5s, 3),
        "profile_config": {
            "max_features": settings.profile_config.max_features,
            "ml_inference_every_n": settings.profile_config.ml_inference_every_n,
            "mapping_enabled": settings.profile_config.mapping_enabled,
            "keyframe_interval_frames": settings.profile_config.keyframe_interval_frames,
        },
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    corrector = MLVelocityCorrector(
        weights_path=artifacts_dir / "phase7_edge_mlp_weights.json",
        scaler_path=artifacts_dir / "phase7_scaler.json",
        max_correction_mps=3.0,
        min_confidence=0.20,
    )

    # Load GT once
    dataset_gt = EurocDataset(dataset_path)
    gt_raw = [
        TrajectoryPoint(
            s.timestamp, np.array(s.position, dtype=np.float64),
            np.array(s.orientation, dtype=np.float64),
            np.array(s.velocity, dtype=np.float64),
        )
        for s in dataset_gt.ground_truth()
    ]
    p_gt0 = gt_raw[0].position.copy()
    gt_pts = [
        TrajectoryPoint(g.timestamp, g.position - p_gt0, g.orientation, g.velocity)
        for g in gt_raw
    ]

    print("=" * 70)
    print("GEONAV-AI -- PHASE 8 PROFILE BENCHMARK")
    print("(Development-machine software benchmark)")
    print("=" * 70)

    results = {}
    for profile in [OperatingProfile.DEVELOPMENT, OperatingProfile.BALANCED, OperatingProfile.EDGE]:
        print(f"\nRunning {profile.value} profile...")
        r = run_profile(profile, dataset_path, corrector, gt_pts)
        results[profile.value] = r
        print(f"  FPS: {r['fps']:.1f} | ATE RMSE: {r['raw_ate_rmse_m']:.3f} m | "
              f"Vel RMSE: {r['velocity_rmse_mps']:.3f} m/s | "
              f"Failures: {r['tracking_failures']} | "
              f"Memory: {r['peak_memory_mb']:.1f} MB")

    print("\n" + "=" * 70)
    print(f"{'Metric':<35} {'DEVELOPMENT':>14} {'BALANCED':>14} {'EDGE':>14}")
    print("-" * 77)
    metrics_to_show = [
        ("FPS", "fps"),
        ("Mean Latency (ms)", "mean_latency_ms"),
        ("P95 Latency (ms)", "p95_latency_ms"),
        ("Raw ATE RMSE (m)", "raw_ate_rmse_m"),
        ("RPE 1s RMSE (m)", "rpe_1s_rmse_m"),
        ("RPE 5s RMSE (m)", "rpe_5s_rmse_m"),
        ("Velocity RMSE (m/s)", "velocity_rmse_mps"),
        ("Tracking Failures", "tracking_failures"),
        ("Peak Memory (MB)", "peak_memory_mb"),
    ]
    for label, key in metrics_to_show:
        vals = [results[p.value][key] for p in [OperatingProfile.DEVELOPMENT, OperatingProfile.BALANCED, OperatingProfile.EDGE]]
        print(f"{label:<35} {vals[0]:>14.3f} {vals[1]:>14.3f} {vals[2]:>14.3f}")
    print("=" * 70)

    # Export JSON
    out_json = artifacts_dir / "phase8_performance_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExported comparison to: {out_json}")

    # Generate comparison plot
    profiles = ["DEVELOPMENT", "BALANCED", "EDGE"]
    fps_vals = [results[p]["fps"] for p in profiles]
    ate_vals = [results[p]["raw_ate_rmse_m"] for p in profiles]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = range(len(profiles))
    bars_fps = axes[0].bar(x, fps_vals, color=["#2196F3", "#FF9800", "#4CAF50"], edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(profiles)
    axes[0].set_ylabel("Frames Per Second")
    axes[0].set_title("Processing FPS by Profile")
    axes[0].bar_label(bars_fps, fmt="%.1f", padding=2)
    axes[0].grid(True, alpha=0.3, axis="y")

    bars_ate = axes[1].bar(x, ate_vals, color=["#2196F3", "#FF9800", "#4CAF50"], edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(profiles)
    axes[1].set_ylabel("Raw ATE RMSE [m]")
    axes[1].set_title("Accuracy (Raw ATE RMSE) by Profile")
    axes[1].bar_label(bars_ate, fmt="%.3f", padding=2)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("GEONAV-AI Phase 8: FPS vs Accuracy Trade-off\n(Development-machine software benchmark)",
                 fontsize=11)
    plt.tight_layout()
    plot_path = artifacts_dir / "phase8_performance_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Generated comparison plot: {plot_path}")
    print("\n[DONE] Phase 8 benchmark complete.")


if __name__ == "__main__":
    main()
