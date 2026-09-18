"""Script for GEONAV-AI Phase 6 Ground-Truth Evaluation on EuRoC MH_01_easy."""

import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

# Ensure src directory is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from geonav.config.settings import ExtrinsicsConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.evaluation.evaluator import TrajectoryEvaluator
from geonav.evaluation.types import TrajectoryPoint
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


def main() -> None:
    """Execute complete ground-truth evaluation workflow on EuRoC MH_01_easy."""
    print("=" * 75)
    print("GEONAV-AI — PHASE 6 GROUND-TRUTH EVALUATION")
    print("=" * 75)

    seq_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    if not seq_path.is_dir():
        print(f"Error: EuRoC dataset sequence not found at: {seq_path}")
        return

    print(f"Loading real EuRoC dataset from: {seq_path}")
    dataset = EurocDataset(seq_path)

    if dataset.num_ground_truth == 0:
        print(f"Error: No ground-truth states found in {seq_path / 'mav0' / 'state_groundtruth_estimate0'}")
        return

    print(f"Loaded {dataset.num_images} camera frames and {dataset.num_ground_truth} ground-truth states.")

    # Prepare ground-truth trajectory points
    gt_points = []
    for s in dataset.ground_truth():
        gt_points.append(
            TrajectoryPoint(
                timestamp=s.timestamp,
                position=np.array(s.position, dtype=np.float64),
                orientation=np.array(s.orientation, dtype=np.float64),
                velocity=np.array(s.velocity, dtype=np.float64),
            )
        )
    print(f"Prepared {len(gt_points)} ground-truth trajectory points.")

    # Configure VIO with dataset extrinsics
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

    # Run VIO to obtain estimated trajectory
    print("\nRunning VIO estimator across synchronized sensor streams...")
    pipeline = VIOPipeline(config=vio_config)
    synced_stream = list(
        synchronize_streams(
            (img.to_camera_frame() for img in dataset.images()),
            (s.to_imu_sample() for s in dataset.imu()),
        )
    )

    t_start = time.time()
    for measurement in synced_stream:
        pipeline.process_measurement(measurement)
    vio_runtime = time.time() - t_start
    print(f"VIO finished in {vio_runtime:.2f} s ({len(synced_stream) / vio_runtime:.1f} FPS).")

    est_trajectory = pipeline.get_trajectory()
    print(f"Obtained {len(est_trajectory)} estimated navigation states.")

    # In EuRoC, the ground-truth origin is arbitrary in the hall, while VIO starts at [0, 0, 0].
    # For meaningful raw trajectory comparison without artificial offset, we align the GT origin
    # to start at [0, 0, 0] relative to the first ground-truth sample:
    p_gt0 = gt_points[0].position
    gt_points_origin = []
    for g in gt_points:
        gt_points_origin.append(
            TrajectoryPoint(
                timestamp=g.timestamp,
                position=g.position - p_gt0,
                orientation=g.orientation,
                velocity=g.velocity,
            )
        )

    est_points = []
    for st in est_trajectory:
        est_points.append(
            TrajectoryPoint(
                timestamp=st.timestamp,
                position=np.array(st.position, dtype=np.float64),
                orientation=np.array(st.orientation, dtype=np.float64),
                velocity=np.array(st.velocity, dtype=np.float64),
            )
        )

    # Load Phase 5 mapping statistics if available
    mapping_stats = None
    map_stats_file = repo_root / "artifacts" / "phase5_mapping_statistics.json"
    if map_stats_file.is_file():
        try:
            with open(map_stats_file, mode="r", encoding="utf-8") as f:
                mapping_stats = json.load(f)
        except Exception:
            pass

    # Execute Evaluation
    print("\nExecuting trajectory evaluation and error analysis...")
    evaluator = TrajectoryEvaluator(dataset_name="EuRoC MH_01_easy", max_time_diff_s=0.01)
    metrics = evaluator.evaluate(est_points, gt_points_origin, mapping_stats=mapping_stats)

    print("\n" + "=" * 75)
    print("PHASE 6 GROUND-TRUTH EVALUATION SUMMARY:")
    print("=" * 75)
    print(f"  Dataset:                         {metrics.dataset}")
    print(f"  Total Camera Frames / Est Poses: {metrics.estimated_pose_count}")
    print(f"  Total Ground-Truth States:       {metrics.ground_truth_pose_count}")
    print(f"  Matched Timestamp Pairs:         {metrics.matched_pose_count}")
    print(f"  Unmatched Estimated Poses:       {metrics.unmatched_estimated_count}")
    print(f"  Unmatched Ground-Truth States:   {metrics.unmatched_groundtruth_count}")
    print(f"  Timestamp Match Threshold:       {metrics.time_association_threshold_s * 1000:.1f} ms")

    print("\n  [RAW ABSOLUTE TRAJECTORY ERROR — ATE]")
    print(f"    RMSE:                          {metrics.raw_ate.rmse:.3f} m")
    print(f"    Mean Error:                    {metrics.raw_ate.mean:.3f} m")
    print(f"    Median Error:                  {metrics.raw_ate.median:.3f} m")
    print(f"    Std Deviation:                 {metrics.raw_ate.std:.3f} m")
    print(f"    Min Error:                     {metrics.raw_ate.min:.3f} m")
    print(f"    Max Error:                     {metrics.raw_ate.max:.3f} m")
    print(f"    Final Position Error:          {metrics.raw_ate.final_error:.3f} m")

    print("\n  [AXIS-SPECIFIC ERROR BREAKDOWN]")
    print(f"    X RMSE (North / Forward):      {metrics.axis_errors.x_rmse:.3f} m (MAE: {metrics.axis_errors.x_mae:.3f} m)")
    print(f"    Y RMSE (East / Right):         {metrics.axis_errors.y_rmse:.3f} m (MAE: {metrics.axis_errors.y_mae:.3f} m)")
    print(f"    Z RMSE (Down / Vertical):      {metrics.axis_errors.z_rmse:.3f} m (MAE: {metrics.axis_errors.z_mae:.3f} m)")

    print("\n  [RELATIVE POSE ERROR — RPE]")
    for name, rpe_val in metrics.rpe.items():
        print(f"    Interval '{name}' ({rpe_val.num_pairs} pairs):")
        print(f"      Translation RMSE:            {rpe_val.trans_rmse:.4f} m (Mean: {rpe_val.trans_mean:.4f} m, Max: {rpe_val.trans_max:.4f} m)")
        print(f"      Rotation RMSE:               {rpe_val.rot_rmse_deg:.3f} deg (Mean: {rpe_val.rot_mean_deg:.3f} deg, Max: {rpe_val.rot_max_deg:.3f} deg)")

    print("\n  [ORIENTATION ERROR]")
    print(f"    Orientation RMSE:              {metrics.orientation_error.rmse_deg:.2f} deg")
    print(f"    Mean Orientation Error:        {metrics.orientation_error.mean_deg:.2f} deg")
    print(f"    Median Orientation Error:      {metrics.orientation_error.median_deg:.2f} deg")
    print(f"    Max Orientation Error:         {metrics.orientation_error.max_deg:.2f} deg")

    if metrics.velocity_error is not None:
        print("\n  [VELOCITY ERROR]")
        print(f"    Velocity RMSE:                 {metrics.velocity_error.rmse:.3f} m/s")
        print(f"    Axis RMSE (Vx, Vy, Vz):        ({metrics.velocity_error.x_rmse:.3f}, {metrics.velocity_error.y_rmse:.3f}, {metrics.velocity_error.z_rmse:.3f}) m/s")
        print(f"    Final Velocity Error:          {metrics.velocity_error.final_error:.3f} m/s")

    print("\n  [SCALE ANALYSIS]")
    print(f"    Estimated Path Length:         {metrics.scale_analysis.estimated_path_length:.2f} m")
    print(f"    Ground-Truth Path Length:      {metrics.scale_analysis.groundtruth_path_length:.2f} m")
    print(f"    Scale Ratio (Est / GT):        {metrics.scale_analysis.scale_ratio:.3f}")

    if metrics.se3_alignment is not None:
        print("\n  [EVALUATION-ONLY SE(3) RIGID ALIGNMENT]")
        print(f"    SE(3) Aligned ATE RMSE:        {metrics.se3_alignment.aligned_ate.rmse:.3f} m")
        print(f"    SE(3) Aligned Mean Error:      {metrics.se3_alignment.aligned_ate.mean:.3f} m")
        print(f"    SE(3) Translation Vector:      {metrics.se3_alignment.translation}")

    if metrics.sim3_alignment is not None:
        print("\n  [EVALUATION-ONLY SIM(3) SIMILARITY ALIGNMENT]")
        print(f"    Sim(3) Aligned ATE RMSE:       {metrics.sim3_alignment.aligned_ate.rmse:.3f} m")
        print(f"    Sim(3) Aligned Mean Error:     {metrics.sim3_alignment.aligned_ate.mean:.3f} m")
        print(f"    Optimal Fitted Scale:          {metrics.sim3_alignment.scale:.3f}")
        print(f"    Sim(3) Translation Vector:     {metrics.sim3_alignment.translation}")

    print("=" * 75)

    # Export Artifacts
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "phase6_trajectory_errors.csv"
    evaluator.export_csv(csv_path)
    print(f"Exported trajectory errors CSV: {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

    json_path = artifacts_dir / "phase6_evaluation_metrics.json"
    evaluator.export_json(json_path)
    print(f"Exported evaluation metrics JSON: {json_path}")

    plot_paths = evaluator.generate_plots(artifacts_dir)
    for k, p in plot_paths.items():
        print(f"Generated plot '{k}': {p}")

    # Copy to IDE artifacts folder
    ide_artifacts_dir = Path(r"C:\Users\karth\.gemini\antigravity-ide\brain\7acff3b5-f467-419a-a2b7-3f422b71c4fe")
    if ide_artifacts_dir.is_dir():
        for p in [csv_path, json_path] + list(plot_paths.values()):
            try:
                shutil.copy2(p, ide_artifacts_dir / p.name)
            except Exception:
                pass


if __name__ == "__main__":
    main()
