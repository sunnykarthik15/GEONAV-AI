"""Independent Phase 7 Metric Verification Script.

Re-derives all key metrics independently from stored VIO trajectories
and compares against phase7_ml_metrics.json. Reports agreement within
documented numerical tolerance.

Usage:
    python scripts/verify_phase7_metrics.py

This script re-runs the VIO + hybrid pipeline and computes metrics
using an independent implementation, without calling TrajectoryEvaluator.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from geonav.config.settings import ExtrinsicsConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.types import TrajectoryPoint
from geonav.ml.features import FeatureExtractor
from geonav.ml.inference import MLVelocityCorrector
from geonav.state import NavigationState
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


# ── Tolerances ──────────────────────────────────────────────────────────────
ATE_TOL_M = 0.5          # Acceptable ATE RMSE difference (m)
VEL_TOL_MPS = 0.05       # Acceptable velocity RMSE difference (m/s)
SCALE_TOL = 0.05         # Acceptable scale ratio difference
PATH_TOL_M = 2.0         # Acceptable path length difference (m)


# ── Independent metric implementations ─────────────────────────────────────

def _path_length(positions: np.ndarray) -> float:
    """Compute total arc length of a 3D position trajectory."""
    if len(positions) < 2:
        return 0.0
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _ate_metrics(est: np.ndarray, gt: np.ndarray) -> dict:
    """Independently compute ATE statistics from matched position arrays."""
    errors = np.linalg.norm(est - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "final": float(errors[-1]),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def _velocity_rmse(est_vel: np.ndarray, gt_vel: np.ndarray) -> dict:
    """Independently compute velocity error metrics."""
    errs = est_vel - gt_vel
    norms = np.linalg.norm(errs, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(norms**2))),
        "x_rmse": float(np.sqrt(np.mean(errs[:, 0]**2))),
        "y_rmse": float(np.sqrt(np.mean(errs[:, 1]**2))),
        "z_rmse": float(np.sqrt(np.mean(errs[:, 2]**2))),
    }


def _rpe(est: np.ndarray, gt: np.ndarray, delta_frames: int) -> dict:
    """Compute Relative Pose Error over delta_frames step."""
    n = len(est) - delta_frames
    if n <= 0:
        return {"rmse": float("nan"), "mean": float("nan"), "num_pairs": 0}
    errs = []
    for i in range(n):
        d_est = est[i + delta_frames] - est[i]
        d_gt = gt[i + delta_frames] - gt[i]
        errs.append(np.linalg.norm(d_est - d_gt))
    errs = np.array(errs)
    return {
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "mean": float(np.mean(errs)),
        "num_pairs": n,
    }


def _rpe_time(timestamps: np.ndarray, est: np.ndarray, gt: np.ndarray,
              delta_s: float) -> dict:
    """Compute RPE for a time interval delta_s."""
    errs = []
    n = len(timestamps)
    j = 0
    for i in range(n):
        while j < n - 1 and timestamps[j] - timestamps[i] < delta_s:
            j += 1
        if timestamps[j] - timestamps[i] >= delta_s * 0.9:
            d_est = est[j] - est[i]
            d_gt = gt[j] - gt[i]
            errs.append(np.linalg.norm(d_est - d_gt))
    if not errs:
        return {"rmse": float("nan"), "mean": float("nan"), "num_pairs": 0}
    errs = np.array(errs)
    return {
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "mean": float(np.mean(errs)),
        "num_pairs": len(errs),
    }


# ── Pipeline runner ──────────────────────────────────────────────────────────

def _run_pipeline(dataset_path: Path, corrector: MLVelocityCorrector):
    """Run both baseline and hybrid VIO, return trajectory lists."""
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
    synced = list(synchronize_streams(
        (img.to_camera_frame() for img in dataset.images()),
        (s.to_imu_sample() for s in dataset.imu()),
    ))

    extractor = FeatureExtractor()
    baseline_states: List[NavigationState] = []
    hybrid_states: List[NavigationState] = []
    prev_hybrid: Optional[NavigationState] = None

    for m in synced:
        t_prev = pipeline._current_state.timestamp if pipeline._current_state is not None else m.camera_frame.timestamp
        t_curr = m.camera_frame.timestamp
        dt = max(t_curr - t_prev, 1e-4) if pipeline._is_initialized else 0.05

        raw_state = pipeline.process_measurement(m)
        baseline_states.append(raw_state)

        feat = extractor.extract(
            visual_result=pipeline._last_visual_result,
            pts_prev=pipeline._prev_pts,
            pts_curr=None,
            imu_samples=m.imu_samples,
            current_state=raw_state,
            dt=dt,
        )
        corrected = corrector.correct_state(prev_hybrid, raw_state, feat, dt)
        hybrid_states.append(corrected)
        prev_hybrid = corrected

    return baseline_states, hybrid_states


# ── Main verification ────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    artifacts_dir = repo_root / "artifacts"
    weights_path = artifacts_dir / "phase7_edge_mlp_weights.json"
    scaler_path = artifacts_dir / "phase7_scaler.json"
    stored_metrics_path = artifacts_dir / "phase7_ml_metrics.json"

    print("=" * 70)
    print("GEONAV-AI — PHASE 7 INDEPENDENT METRIC VERIFICATION")
    print("=" * 70)

    if not stored_metrics_path.is_file():
        print(f"ERROR: Stored metrics not found at {stored_metrics_path}")
        print("Run 'python src/geonav/ml/evaluate.py' first.")
        sys.exit(1)

    with open(stored_metrics_path) as f:
        stored = json.load(f)

    print(f"\nLoaded stored metrics from: {stored_metrics_path}")
    print(f"Running independent pipeline on: {dataset_path}\n")

    corrector = MLVelocityCorrector(
        weights_path=weights_path,
        scaler_path=scaler_path,
        max_correction_mps=3.0,
        min_confidence=0.20,
    )

    t0 = time.perf_counter()
    baseline_states, hybrid_states = _run_pipeline(dataset_path, corrector)
    elapsed = time.perf_counter() - t0
    fps = len(baseline_states) / elapsed
    print(f"Ran {len(baseline_states)} frames in {elapsed:.2f}s ({fps:.1f} FPS)")

    # Load GT
    dataset = EurocDataset(dataset_path)
    gt_raw = [
        TrajectoryPoint(
            s.timestamp,
            np.array(s.position, dtype=np.float64),
            np.array(s.orientation, dtype=np.float64),
            np.array(s.velocity, dtype=np.float64),
        )
        for s in dataset.ground_truth()
    ]
    p_gt0 = gt_raw[0].position.copy()
    gt_pts = [
        TrajectoryPoint(g.timestamp, g.position - p_gt0, g.orientation, g.velocity)
        for g in gt_raw
    ]

    # Associate timestamps
    base_pts = [
        TrajectoryPoint(s.timestamp, np.array(s.position), np.array(s.orientation), np.array(s.velocity))
        for s in baseline_states
    ]
    hyb_pts = [
        TrajectoryPoint(s.timestamp, np.array(s.position), np.array(s.orientation), np.array(s.velocity))
        for s in hybrid_states
    ]

    base_pairs, _, _ = associate_timestamps(base_pts, gt_pts, max_time_diff_s=0.01)
    hyb_pairs, _, _ = associate_timestamps(hyb_pts, gt_pts, max_time_diff_s=0.01)

    print(f"Matched pairs — Baseline: {len(base_pairs)}, Hybrid: {len(hyb_pairs)}")

    # Extract arrays
    base_est = np.array([p.est_point.position for p in base_pairs])
    base_gt  = np.array([p.gt_point.position  for p in base_pairs])
    hyb_est  = np.array([p.est_point.position  for p in hyb_pairs])
    hyb_gt   = np.array([p.gt_point.position   for p in hyb_pairs])

    base_v_est = np.array([p.est_point.velocity for p in base_pairs])
    base_v_gt  = np.array([p.gt_point.velocity  for p in base_pairs])
    hyb_v_est  = np.array([p.est_point.velocity  for p in hyb_pairs])
    hyb_v_gt   = np.array([p.gt_point.velocity   for p in hyb_pairs])

    ts_base = np.array([p.timestamp_est for p in base_pairs])
    ts_hyb  = np.array([p.timestamp_est for p in hyb_pairs])

    # Independently computed metrics
    base_ate  = _ate_metrics(base_est, base_gt)
    hyb_ate   = _ate_metrics(hyb_est, hyb_gt)
    base_vel  = _velocity_rmse(base_v_est, base_v_gt)
    hyb_vel   = _velocity_rmse(hyb_v_est, hyb_v_gt)

    # Scale / path length
    base_path = _path_length(base_est)
    hyb_path  = _path_length(hyb_est)
    gt_path   = _path_length(base_gt)  # GT path from baseline association

    base_scale = base_path / gt_path if gt_path > 0 else float("nan")
    hyb_scale  = hyb_path  / gt_path if gt_path > 0 else float("nan")

    # RPE
    rpe_1f   = {"base": _rpe(base_est, base_gt, 1),   "hyb": _rpe(hyb_est, hyb_gt, 1)}
    rpe_1s   = {"base": _rpe_time(ts_base, base_est, base_gt, 1.0),
                "hyb":  _rpe_time(ts_hyb,  hyb_est,  hyb_gt,  1.0)}
    rpe_2s   = {"base": _rpe_time(ts_base, base_est, base_gt, 2.0),
                "hyb":  _rpe_time(ts_hyb,  hyb_est,  hyb_gt,  2.0)}
    rpe_5s   = {"base": _rpe_time(ts_base, base_est, base_gt, 5.0),
                "hyb":  _rpe_time(ts_hyb,  hyb_est,  hyb_gt,  5.0)}

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INDEPENDENTLY COMPUTED METRICS:")
    print("=" * 70)
    print(f"{'Metric':<40} {'Baseline':>12} {'Hybrid':>12}")
    print("-" * 66)
    print(f"{'ATE RMSE [m]':<40} {base_ate['rmse']:>12.3f} {hyb_ate['rmse']:>12.3f}")
    print(f"{'ATE Mean [m]':<40} {base_ate['mean']:>12.3f} {hyb_ate['mean']:>12.3f}")
    print(f"{'ATE Median [m]':<40} {base_ate['median']:>12.3f} {hyb_ate['median']:>12.3f}")
    print(f"{'Final Position Error [m]':<40} {base_ate['final']:>12.3f} {hyb_ate['final']:>12.3f}")
    print(f"{'Velocity RMSE [m/s]':<40} {base_vel['rmse']:>12.3f} {hyb_vel['rmse']:>12.3f}")
    print(f"{'Vx RMSE [m/s]':<40} {base_vel['x_rmse']:>12.3f} {hyb_vel['x_rmse']:>12.3f}")
    print(f"{'Vy RMSE [m/s]':<40} {base_vel['y_rmse']:>12.3f} {hyb_vel['y_rmse']:>12.3f}")
    print(f"{'Vz RMSE [m/s]':<40} {base_vel['z_rmse']:>12.3f} {hyb_vel['z_rmse']:>12.3f}")
    print(f"{'Estimated Path Length [m]':<40} {base_path:>12.3f} {hyb_path:>12.3f}")
    print(f"{'GT Path Length [m]':<40} {gt_path:>12.3f} {'(same)':>12}")
    print(f"{'Scale Ratio (path_est/path_GT)':<40} {base_scale:>12.3f} {hyb_scale:>12.3f}")
    print(f"{'RPE 1-frame RMSE [m]':<40} {rpe_1f['base']['rmse']:>12.3f} {rpe_1f['hyb']['rmse']:>12.3f}")
    print(f"{'RPE 1s RMSE [m]':<40} {rpe_1s['base']['rmse']:>12.3f} {rpe_1s['hyb']['rmse']:>12.3f}")
    print(f"{'RPE 2s RMSE [m]':<40} {rpe_2s['base']['rmse']:>12.3f} {rpe_2s['hyb']['rmse']:>12.3f}")
    print(f"{'RPE 5s RMSE [m]':<40} {rpe_5s['base']['rmse']:>12.3f} {rpe_5s['hyb']['rmse']:>12.3f}")
    print("=" * 70)

    # Scale ratio note
    print("\nNOTE: Scale ratio = path_length(est) / path_length(GT).")
    print("This is a path-length ratio, NOT a formal metric-scale estimate.")
    print("A ratio close to 1.0 means similar total displacement, not correct trajectory shape.")

    # ── Compare against stored metrics ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON vs STORED METRICS (phase7_ml_metrics.json):")
    print("=" * 70)

    stored_base_ate = stored["baseline_metrics"]["raw_ate_rmse_m"]
    stored_hyb_ate  = stored["hybrid_metrics"]["raw_ate_rmse_m"]
    stored_base_vel = stored["baseline_metrics"]["velocity_rmse_mps"]
    stored_hyb_vel  = stored["hybrid_metrics"]["velocity_rmse_mps"]
    stored_base_scale = stored["baseline_metrics"]["scale_ratio"]
    stored_hyb_scale  = stored["hybrid_metrics"]["scale_ratio"]

    checks = [
        ("Baseline ATE RMSE [m]",   base_ate['rmse'], stored_base_ate, ATE_TOL_M),
        ("Hybrid ATE RMSE [m]",     hyb_ate['rmse'],  stored_hyb_ate,  ATE_TOL_M),
        ("Baseline Vel RMSE [m/s]", base_vel['rmse'], stored_base_vel, VEL_TOL_MPS),
        ("Hybrid Vel RMSE [m/s]",   hyb_vel['rmse'],  stored_hyb_vel,  VEL_TOL_MPS),
        ("Baseline Scale Ratio",    base_scale,       stored_base_scale, SCALE_TOL),
        ("Hybrid Scale Ratio",      hyb_scale,        stored_hyb_scale,  SCALE_TOL),
    ]

    all_pass = True
    for name, computed, stored_val, tol in checks:
        diff = abs(computed - stored_val)
        status = "[PASS]" if diff <= tol else "[FAIL]"
        if diff > tol:
            all_pass = False
        print(f"  {status}  {name}: computed={computed:.4f}, stored={stored_val:.4f}, "
              f"diff={diff:.4f} (tol={tol:.4f})")

    print("=" * 70)

    # ── Export independent verification results ───────────────────────────────
    verification = {
        "status": "PASS" if all_pass else "FAIL",
        "dataset": "EuRoC MH_01_easy",
        "frames_processed": len(baseline_states),
        "fps": round(fps, 1),
        "gt_path_length_m": round(gt_path, 3),
        "independent_baseline": {
            "ate_rmse_m": round(base_ate["rmse"], 4),
            "ate_mean_m": round(base_ate["mean"], 4),
            "ate_median_m": round(base_ate["median"], 4),
            "final_position_error_m": round(base_ate["final"], 4),
            "velocity_rmse_mps": round(base_vel["rmse"], 4),
            "vx_rmse_mps": round(base_vel["x_rmse"], 4),
            "vy_rmse_mps": round(base_vel["y_rmse"], 4),
            "vz_rmse_mps": round(base_vel["z_rmse"], 4),
            "path_length_m": round(base_path, 3),
            "scale_ratio": round(base_scale, 4),
            "rpe_1frame_rmse_m": round(rpe_1f["base"]["rmse"], 4),
            "rpe_1s_rmse_m": round(rpe_1s["base"]["rmse"], 4),
            "rpe_2s_rmse_m": round(rpe_2s["base"]["rmse"], 4),
            "rpe_5s_rmse_m": round(rpe_5s["base"]["rmse"], 4),
        },
        "independent_hybrid": {
            "ate_rmse_m": round(hyb_ate["rmse"], 4),
            "ate_mean_m": round(hyb_ate["mean"], 4),
            "ate_median_m": round(hyb_ate["median"], 4),
            "final_position_error_m": round(hyb_ate["final"], 4),
            "velocity_rmse_mps": round(hyb_vel["rmse"], 4),
            "vx_rmse_mps": round(hyb_vel["x_rmse"], 4),
            "vy_rmse_mps": round(hyb_vel["y_rmse"], 4),
            "vz_rmse_mps": round(hyb_vel["z_rmse"], 4),
            "path_length_m": round(hyb_path, 3),
            "scale_ratio": round(hyb_scale, 4),
            "rpe_1frame_rmse_m": round(rpe_1f["hyb"]["rmse"], 4),
            "rpe_1s_rmse_m": round(rpe_1s["hyb"]["rmse"], 4),
            "rpe_2s_rmse_m": round(rpe_2s["hyb"]["rmse"], 4),
            "rpe_5s_rmse_m": round(rpe_5s["hyb"]["rmse"], 4),
        },
        "scale_ratio_note": (
            "Scale ratio = path_length(estimated) / path_length(GT). "
            "This is a path-length ratio, NOT a formal metric-scale estimate. "
            "A value near 1.0 indicates similar total displacement magnitude, "
            "not that the trajectory shape is correct."
        ),
        "temporal_split_note": (
            "The 70/30 split is a chronologically held-out temporal segment "
            "from the same MH_01_easy sequence. It is NOT an unseen-flight test."
        ),
        "ate_type": "RAW ATE (no SE3/Sim3 alignment applied to primary metrics)",
    }

    out_path = artifacts_dir / "phase7_independent_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)
    print(f"\nExported independent verification to: {out_path}")

    if all_pass:
        print("\n[PASS] PHASE 7 INDEPENDENT METRIC VERIFICATION -- PASS")
    else:
        print("\n[FAIL] PHASE 7 INDEPENDENT METRIC VERIFICATION -- FAIL")
        print("   Discrepancies exceed tolerance. Investigate before declaring Phase 7 complete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
