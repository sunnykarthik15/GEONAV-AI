"""Evaluation and ablation benchmark script for GEONAV-AI Phase 7 ML."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from geonav.config.settings import ExtrinsicsConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.evaluator import TrajectoryEvaluator
from geonav.evaluation.types import TrajectoryPoint
from geonav.ml.features import FeatureExtractor
from geonav.ml.inference import MLVelocityCorrector
from geonav.state import NavigationState
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


def run_hybrid_vio(
    dataset_path: Path,
    corrector: MLVelocityCorrector,
) -> Tuple[List[NavigationState], List[NavigationState], float]:
    """Execute both classical baseline VIO and hybrid VIO + ML correction.

    Returns:
        Tuple: (baseline_states, hybrid_states, total_runtime_s)
    """
    dataset = EurocDataset(dataset_path)

    vio_config = VIOConfig()
    if dataset.camera_extrinsics is not None:
        T_BS = dataset.camera_extrinsics
        R_BS = T_BS[:3, :3]
        t_BS = T_BS[:3, 3]
        vio_config.extrinsics = ExtrinsicsConfig(
            rotation_matrix=(
                (float(R_BS[0, 0]), float(R_BS[0, 1]), float(R_BS[0, 2])),
                (float(R_BS[1, 0]), float(R_BS[1, 1]), float(R_BS[1, 2])),
                (float(R_BS[2, 0]), float(R_BS[2, 1]), float(R_BS[2, 2])),
            ),
            translation=(float(t_BS[0]), float(t_BS[1]), float(t_BS[2])),
        )

    pipeline = VIOPipeline(config=vio_config)
    synced = list(
        synchronize_streams(
            (img.to_camera_frame() for img in dataset.images()),
            (s.to_imu_sample() for s in dataset.imu()),
        )
    )

    extractor = FeatureExtractor()
    baseline_states: List[NavigationState] = []
    hybrid_states: List[NavigationState] = []
    prev_hybrid_state: Optional[NavigationState] = None

    t_start = time.time()
    for m in synced:
        t_prev = pipeline._current_state.timestamp if pipeline._current_state is not None else m.camera_frame.timestamp
        t_curr = m.camera_frame.timestamp
        dt = max(t_curr - t_prev, 1e-4) if pipeline._is_initialized else 0.05

        # 1. Classical step
        raw_state = pipeline.process_measurement(m)
        baseline_states.append(raw_state)

        # 2. Extract causal inference features
        feat = extractor.extract(
            visual_result=pipeline._last_visual_result,
            pts_prev=pipeline._prev_pts,
            pts_curr=None,
            imu_samples=m.imu_samples,
            current_state=raw_state,
            dt=dt,
        )

        # 3. Apply learned velocity correction
        corrected_state = corrector.correct_state(
            prev_corrected_state=prev_hybrid_state,
            raw_current_state=raw_state,
            features=feat,
            dt=dt,
        )
        hybrid_states.append(corrected_state)
        prev_hybrid_state = corrected_state

    total_runtime = time.time() - t_start
    return baseline_states, hybrid_states, total_runtime


def main() -> None:
    """Run full ablation evaluation and generate artifacts."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    weights_path = artifacts_dir / "phase7_edge_mlp_weights.json"
    scaler_path = artifacts_dir / "phase7_scaler.json"

    print("=" * 70)
    print("GEONAV-AI — PHASE 7 ML ABLATION EVALUATION")
    print("=" * 70)

    if not weights_path.is_file() or not scaler_path.is_file():
        print("Model weights or scaler not found. Running training first...")
        from geonav.ml.train import main as train_main
        train_main()

    corrector = MLVelocityCorrector(
        weights_path=weights_path,
        scaler_path=scaler_path,
        max_correction_mps=3.0,
        min_confidence=0.20,
    )

    print(f"\nRunning Baseline VIO and Hybrid VIO + ML on {dataset_path}...")
    baseline_states, hybrid_states, runtime_s = run_hybrid_vio(dataset_path, corrector)
    fps = len(baseline_states) / runtime_s
    print(f"Finished {len(baseline_states)} frames in {runtime_s:.2f} s ({fps:.1f} FPS).")

    # Load Ground Truth
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
    p_gt0 = gt_raw[0].position
    gt_pts = [
        TrajectoryPoint(
            g.timestamp,
            g.position - p_gt0,
            g.orientation,
            g.velocity,
        )
        for g in gt_raw
    ]

    base_pts = [
        TrajectoryPoint(s.timestamp, np.array(s.position), np.array(s.orientation), np.array(s.velocity))
        for s in baseline_states
    ]
    hyb_pts = [
        TrajectoryPoint(s.timestamp, np.array(s.position), np.array(s.orientation), np.array(s.velocity))
        for s in hybrid_states
    ]

    # Evaluate Baseline
    eval_base = TrajectoryEvaluator("EuRoC MH_01_easy (Baseline Classical VIO)")
    metrics_base = eval_base.evaluate(base_pts, gt_pts)

    # Evaluate Hybrid
    eval_hyb = TrajectoryEvaluator("EuRoC MH_01_easy (Hybrid VIO + ML)")
    metrics_hyb = eval_hyb.evaluate(hyb_pts, gt_pts)

    # Chronological Evaluation Split (last 30% of matched pairs)
    n_pairs = len(eval_base.matched_pairs)
    n_train = int(n_pairs * 0.70)

    base_eval_pairs = eval_base.matched_pairs[n_train:]
    hyb_eval_pairs = eval_hyb.matched_pairs[n_train:]

    from geonav.evaluation.trajectory import compute_ate, compute_velocity_errors
    base_eval_ate = compute_ate(
        np.array([p.est_point.position for p in base_eval_pairs]),
        np.array([p.gt_point.position for p in base_eval_pairs]),
    )
    hyb_eval_ate = compute_ate(
        np.array([p.est_point.position for p in hyb_eval_pairs]),
        np.array([p.gt_point.position for p in hyb_eval_pairs]),
    )

    base_eval_vel = compute_velocity_errors(
        np.array([p.est_point.velocity for p in base_eval_pairs]),
        np.array([p.gt_point.velocity for p in base_eval_pairs]),
    )
    hyb_eval_vel = compute_velocity_errors(
        np.array([p.est_point.velocity for p in hyb_eval_pairs]),
        np.array([p.gt_point.velocity for p in hyb_eval_pairs]),
    )

    print("\n" + "=" * 70)
    print("PHASE 7 ABLATION STUDY RESULTS:")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Baseline VIO':<15} | {'Hybrid VIO + ML':<15} | {'Delta':<12}")
    print("-" * 75)

    def row(name: str, v_base: float, v_hyb: float, unit: str = ""):
        delta = v_hyb - v_base
        pct = (delta / max(abs(v_base), 1e-6)) * 100
        sign = "+" if delta > 0 else ""
        print(f"{name:<35} | {v_base:>10.3f} {unit:<4} | {v_hyb:>10.3f} {unit:<4} | {sign}{delta:>6.3f} ({sign}{pct:.1f}%)")

    row("Full Trajectory Raw ATE RMSE", metrics_base.raw_ate.rmse, metrics_hyb.raw_ate.rmse, "m")
    row("Full Trajectory Mean Position Error", metrics_base.raw_ate.mean, metrics_hyb.raw_ate.mean, "m")
    row("Final Position Error", metrics_base.raw_ate.final_error, metrics_hyb.raw_ate.final_error, "m")
    row("Velocity RMSE (Overall)", metrics_base.velocity_error.rmse, metrics_hyb.velocity_error.rmse, "m/s")
    row("  Vx RMSE (Forward)", metrics_base.velocity_error.x_rmse, metrics_hyb.velocity_error.x_rmse, "m/s")
    row("  Vy RMSE (Lateral)", metrics_base.velocity_error.y_rmse, metrics_hyb.velocity_error.y_rmse, "m/s")
    row("  Vz RMSE (Vertical)", metrics_base.velocity_error.z_rmse, metrics_hyb.velocity_error.z_rmse, "m/s")
    row("Estimated Path Length", metrics_base.scale_analysis.estimated_path_length, metrics_hyb.scale_analysis.estimated_path_length, "m")
    row("Scale Ratio (Est / GT)", metrics_base.scale_analysis.scale_ratio, metrics_hyb.scale_analysis.scale_ratio, "")

    print("-" * 75)
    print("CHRONOLOGICAL EVALUATION SPLIT ONLY (Last 30% Unseen Time Horizon):")
    row("Eval-Split Raw ATE RMSE", base_eval_ate.rmse, hyb_eval_ate.rmse, "m")
    row("Eval-Split Mean Position Error", base_eval_ate.mean, hyb_eval_ate.mean, "m")
    row("Eval-Split Velocity RMSE", base_eval_vel.rmse, hyb_eval_vel.rmse, "m/s")
    print("=" * 70)

    # Export JSON Metrics
    metrics_export = {
        "dataset": "EuRoC MH_01_easy",
        "model_architecture": "EdgeMLP (18 -> 32 -> 16 -> 3 + confidence head)",
        "model_size_kb": round(float(weights_path.stat().st_size) / 1024.0, 2),
        "inference_engine": "NumpyEdgeMLP (zero-dependency pure NumPy)",
        "processing_fps": round(fps, 1),
        "total_runtime_s": round(runtime_s, 2),
        "split_policy": "Chronological (70% train, 30% evaluation, no shuffling)",
        "baseline_metrics": {
            "raw_ate_rmse_m": metrics_base.raw_ate.rmse,
            "final_position_error_m": metrics_base.raw_ate.final_error,
            "velocity_rmse_mps": metrics_base.velocity_error.rmse,
            "axis_errors": metrics_base.axis_errors.to_dict(),
            "scale_ratio": metrics_base.scale_analysis.scale_ratio,
        },
        "hybrid_metrics": {
            "raw_ate_rmse_m": metrics_hyb.raw_ate.rmse,
            "final_position_error_m": metrics_hyb.raw_ate.final_error,
            "velocity_rmse_mps": metrics_hyb.velocity_error.rmse,
            "axis_errors": metrics_hyb.axis_errors.to_dict(),
            "scale_ratio": metrics_hyb.scale_analysis.scale_ratio,
        },
        "eval_split_unseen_horizon": {
            "sample_count": len(base_eval_pairs),
            "baseline_ate_rmse_m": base_eval_ate.rmse,
            "hybrid_ate_rmse_m": hyb_eval_ate.rmse,
            "baseline_velocity_rmse_mps": base_eval_vel.rmse,
            "hybrid_velocity_rmse_mps": hyb_eval_vel.rmse,
        },
    }

    json_path = artifacts_dir / "phase7_ml_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"\nExported Phase 7 metrics to: {json_path}")

    # Generate Comparison Plot
    plot_path = artifacts_dir / "phase7_ml_comparison.png"
    times = [p.timestamp_est - eval_base.matched_pairs[0].timestamp_est for p in eval_base.matched_pairs]

    base_errs = [
        np.linalg.norm(p.est_point.position - p.gt_point.position) for p in eval_base.matched_pairs
    ]
    hyb_errs = [
        np.linalg.norm(p.est_point.position - p.gt_point.position) for p in eval_hyb.matched_pairs
    ]

    fig, axs = plt.subplots(2, 1, figsize=(11, 8))

    # Top: Position Error Over Time
    axs[0].plot(times, base_errs, "r-", linewidth=1.5, label=f"Baseline Classical VIO (RMSE: {metrics_base.raw_ate.rmse:.2f} m)")
    axs[0].plot(times, hyb_errs, "b-", linewidth=1.5, label=f"Hybrid VIO + ML (RMSE: {metrics_hyb.raw_ate.rmse:.2f} m)")
    split_t = times[n_train]
    axs[0].axvline(x=split_t, color="k", linestyle="--", label=f"Train/Eval Split Horizon ({split_t:.1f} s)")
    axs[0].set_ylabel("Position Error [m]")
    axs[0].set_title("GEONAV-AI Phase 7: Trajectory Position Error Comparison (Baseline vs Hybrid)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper left")

    # Bottom: Velocity Error Over Time
    base_v_errs = [
        np.linalg.norm(p.est_point.velocity - p.gt_point.velocity) for p in eval_base.matched_pairs
    ]
    hyb_v_errs = [
        np.linalg.norm(p.est_point.velocity - p.gt_point.velocity) for p in eval_hyb.matched_pairs
    ]
    axs[1].plot(times, base_v_errs, "r-", alpha=0.7, label=f"Baseline Velocity Error (RMSE: {metrics_base.velocity_error.rmse:.2f} m/s)")
    axs[1].plot(times, hyb_v_errs, "b-", alpha=0.7, label=f"Hybrid Velocity Error (RMSE: {metrics_hyb.velocity_error.rmse:.2f} m/s)")
    axs[1].axvline(x=split_t, color="k", linestyle="--")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity Error [m/s]")
    axs[1].set_title("Velocity Magnitude Error Comparison")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Generated comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
