"""High-level trajectory evaluator orchestrating error calculation, alignment, and export."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geonav.evaluation.alignment import align_trajectory_points, umeyama_alignment
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.trajectory import (
    compute_ate,
    compute_axis_errors,
    compute_orientation_errors,
    compute_rpe,
    compute_scale_analysis,
    compute_velocity_errors,
)
from geonav.evaluation.types import (
    AlignmentResult,
    AssociatedPair,
    EvaluationMetrics,
    TrajectoryPoint,
)
from geonav.vio.geometry import quaternion_to_rotation_matrix


class TrajectoryEvaluator:
    """Computes comprehensive trajectory evaluation metrics between estimated and ground-truth poses."""

    def __init__(
        self,
        dataset_name: str = "EuRoC MH_01_easy",
        max_time_diff_s: float = 0.01,
    ) -> None:
        """Initialize trajectory evaluator.

        Args:
            dataset_name: Identifier of the dataset being evaluated.
            max_time_diff_s: Maximum allowable time threshold for timestamp matching.
        """
        self.dataset_name = dataset_name
        self.max_time_diff_s = max_time_diff_s

        self.matched_pairs: List[AssociatedPair] = []
        self.metrics: Optional[EvaluationMetrics] = None

        self._aligned_se3_pts: Optional[np.ndarray] = None
        self._aligned_sim3_pts: Optional[np.ndarray] = None

    def evaluate(
        self,
        est_points: Sequence[TrajectoryPoint],
        gt_points: Sequence[TrajectoryPoint],
        mapping_stats: Optional[Dict[str, Any]] = None,
    ) -> EvaluationMetrics:
        """Execute complete evaluation workflow across matched trajectory points.

        Args:
            est_points: Estimated trajectory points.
            gt_points: Ground-truth trajectory points.
            mapping_stats: Optional Phase 5 mapping statistics dictionary.

        Returns:
            EvaluationMetrics: Aggregated results.
        """
        # 1. Timestamp Association
        pairs, un_est, un_gt = associate_timestamps(
            est_points,
            gt_points,
            max_time_diff_s=self.max_time_diff_s,
        )
        self.matched_pairs = pairs

        if not pairs:
            raise ValueError("Zero trajectory points matched within timestamp threshold")

        # 2. Extract arrays
        pts_est = np.array([p.est_point.position for p in pairs], dtype=np.float64)
        pts_gt = np.array([p.gt_point.position for p in pairs], dtype=np.float64)

        quats_est = np.array([p.est_point.orientation for p in pairs], dtype=np.float64)
        quats_gt = np.array([p.gt_point.orientation for p in pairs], dtype=np.float64)

        est_has_vel = all(p.est_point.velocity is not None for p in pairs)
        gt_has_vel = all(p.gt_point.velocity is not None for p in pairs)

        # 3. Raw Absolute Trajectory Error (ATE)
        raw_ate = compute_ate(pts_est, pts_gt)

        # 4. Axis-wise Error
        axis_errors = compute_axis_errors(pts_est, pts_gt)

        # 5. Relative Pose Error (RPE)
        est_matched_pts = [p.est_point for p in pairs]
        gt_matched_pts = [p.gt_point for p in pairs]

        rpe_dict = {
            "1_frame": compute_rpe(est_matched_pts, gt_matched_pts, interval_frames=1, interval_name="1_frame"),
            "1_second": compute_rpe(est_matched_pts, gt_matched_pts, interval_time_s=1.0, interval_name="1_second"),
            "2_seconds": compute_rpe(est_matched_pts, gt_matched_pts, interval_time_s=2.0, interval_name="2_seconds"),
            "5_seconds": compute_rpe(est_matched_pts, gt_matched_pts, interval_time_s=5.0, interval_name="5_seconds"),
        }

        # 6. Orientation Error
        orient_err = compute_orientation_errors(quats_est, quats_gt)

        # 7. Scale Analysis
        scale_res = compute_scale_analysis(pts_est, pts_gt)

        # 8. Velocity Error (if available)
        vel_res = None
        if est_has_vel and gt_has_vel:
            vel_est = np.array([p.est_point.velocity for p in pairs], dtype=np.float64)
            vel_gt = np.array([p.gt_point.velocity for p in pairs], dtype=np.float64)
            vel_res = compute_velocity_errors(vel_est, vel_gt)

        # 9. Evaluation-only Rigid SE(3) Alignment
        s_se3, R_se3, t_se3 = umeyama_alignment(pts_est, pts_gt, with_scale=False)
        self._aligned_se3_pts = align_trajectory_points(pts_est, s_se3, R_se3, t_se3)
        ate_se3 = compute_ate(self._aligned_se3_pts, pts_gt)
        se3_res = AlignmentResult("SE(3)", s_se3, R_se3, t_se3, ate_se3)

        # 10. Evaluation-only Similarity Sim(3) Alignment
        s_sim3, R_sim3, t_sim3 = umeyama_alignment(pts_est, pts_gt, with_scale=True)
        self._aligned_sim3_pts = align_trajectory_points(pts_est, s_sim3, R_sim3, t_sim3)
        ate_sim3 = compute_ate(self._aligned_sim3_pts, pts_gt)
        sim3_res = AlignmentResult("Sim(3)", s_sim3, R_sim3, t_sim3, ate_sim3)

        metrics = EvaluationMetrics(
            dataset=self.dataset_name,
            estimated_pose_count=len(est_points),
            ground_truth_pose_count=len(gt_points),
            matched_pose_count=len(pairs),
            unmatched_estimated_count=un_est,
            unmatched_groundtruth_count=un_gt,
            time_association_threshold_s=self.max_time_diff_s,
            raw_ate=raw_ate,
            axis_errors=axis_errors,
            rpe=rpe_dict,
            orientation_error=orient_err,
            scale_analysis=scale_res,
            velocity_error=vel_res,
            se3_alignment=se3_res,
            sim3_alignment=sim3_res,
            mapping_statistics=mapping_stats,
        )

        self.metrics = metrics
        return metrics

    def export_csv(self, file_path: Union[str, Path]) -> Path:
        """Export per-frame trajectory errors to a CSV file.

        Args:
            file_path: Destination path.

        Returns:
            Path: Resolved path.
        """
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "estimated_x",
            "estimated_y",
            "estimated_z",
            "groundtruth_x",
            "groundtruth_y",
            "groundtruth_z",
            "error_x",
            "error_y",
            "error_z",
            "position_error",
            "orientation_error_deg",
        ]

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for pair in self.matched_pairs:
                pe = pair.est_point.position
                pg = pair.gt_point.position
                diff = pe - pg
                pos_err = float(np.linalg.norm(diff))

                Re = quaternion_to_rotation_matrix(pair.est_point.orientation)
                Rg = quaternion_to_rotation_matrix(pair.gt_point.orientation)
                R_err = Rg.T @ Re
                cos_ang = min(max((np.trace(R_err) - 1.0) * 0.5, -1.0), 1.0)
                rot_err_deg = math.degrees(math.acos(cos_ang))

                writer.writerow(
                    {
                        "timestamp": f"{pair.timestamp_est:.6f}",
                        "estimated_x": f"{pe[0]:.6f}",
                        "estimated_y": f"{pe[1]:.6f}",
                        "estimated_z": f"{pe[2]:.6f}",
                        "groundtruth_x": f"{pg[0]:.6f}",
                        "groundtruth_y": f"{pg[1]:.6f}",
                        "groundtruth_z": f"{pg[2]:.6f}",
                        "error_x": f"{diff[0]:.6f}",
                        "error_y": f"{diff[1]:.6f}",
                        "error_z": f"{diff[2]:.6f}",
                        "position_error": f"{pos_err:.6f}",
                        "orientation_error_deg": f"{rot_err_deg:.4f}",
                    }
                )

        return path

    def export_json(self, file_path: Union[str, Path]) -> Path:
        """Export comprehensive metrics to a machine-readable JSON file.

        Args:
            file_path: Destination path.

        Returns:
            Path: Resolved path.
        """
        if self.metrics is None:
            raise ValueError("Cannot export JSON before calling evaluate()")

        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        return path

    def generate_plots(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate evaluation plots: position error vs time, 3D trajectories, and axis-wise errors.

        Args:
            output_dir: Destination folder.

        Returns:
            Dict[str, Path]: Mapping of plot identifiers to generated file paths.
        """
        if not self.matched_pairs or self.metrics is None:
            raise ValueError("Cannot generate plots before calling evaluate()")

        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        times = np.array([p.timestamp_est for p in self.matched_pairs], dtype=np.float64)
        t_rel = times - times[0]

        pe = np.array([p.est_point.position for p in self.matched_pairs], dtype=np.float64)
        pg = np.array([p.gt_point.position for p in self.matched_pairs], dtype=np.float64)
        diffs = pe - pg
        pos_errors = np.linalg.norm(diffs, axis=1)

        plot_paths: Dict[str, Path] = {}

        # 1. Position Error vs Time
        p_err_path = out / "phase6_position_error.png"
        plt.figure(figsize=(10, 5))
        plt.plot(t_rel, pos_errors, "b-", linewidth=1.5, label="Raw Position Error")
        plt.axhline(
            y=self.metrics.raw_ate.rmse,
            color="r",
            linestyle="--",
            label=f"Raw ATE RMSE: {self.metrics.raw_ate.rmse:.2f} m",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        plt.title(
            f"GEONAV-AI Phase 6: Position Error Over Time ({self.dataset_name})\n"
            f"RMSE: {self.metrics.raw_ate.rmse:.2f} m | Final Error: {self.metrics.raw_ate.final_error:.2f} m"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(p_err_path, dpi=150)
        plt.close()
        plot_paths["position_error"] = p_err_path

        # 2. 3D Trajectory Comparison (Raw VIO, Ground Truth, Aligned SE(3))
        p_traj_path = out / "phase6_trajectory_comparison.png"
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pe[:, 0], pe[:, 1], pe[:, 2], "b-", linewidth=1.5, label="Raw Estimated VIO Trajectory")
        ax.plot(pg[:, 0], pg[:, 1], pg[:, 2], "g--", linewidth=1.5, label="Ground Truth Trajectory")

        if self._aligned_se3_pts is not None:
            ax.plot(
                self._aligned_se3_pts[:, 0],
                self._aligned_se3_pts[:, 1],
                self._aligned_se3_pts[:, 2],
                "m:",
                linewidth=1.8,
                label=f"Evaluation-Only Aligned Estimate (SE(3) RMSE: {self.metrics.se3_alignment.aligned_ate.rmse:.2f} m)",
            )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Trajectory Comparison: Raw vs Ground Truth vs Aligned ({self.dataset_name})")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(p_traj_path, dpi=150)
        plt.close()
        plot_paths["trajectory_comparison"] = p_traj_path

        # 3. Axis-wise Error Over Time (X, Y, Z components)
        p_axis_path = out / "phase6_axis_errors.png"
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(t_rel, diffs[:, 0], "r-", label=f"X Error (RMSE: {self.metrics.axis_errors.x_rmse:.2f} m)")
        axs[0].set_ylabel("X Error [m]")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="upper right")

        axs[1].plot(t_rel, diffs[:, 1], "g-", label=f"Y Error (RMSE: {self.metrics.axis_errors.y_rmse:.2f} m)")
        axs[1].set_ylabel("Y Error [m]")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="upper right")

        axs[2].plot(t_rel, diffs[:, 2], "b-", label=f"Z Error (RMSE: {self.metrics.axis_errors.z_rmse:.2f} m)")
        axs[2].set_ylabel("Z Error [m]")
        axs[2].set_xlabel("Time [s]")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc="upper right")

        fig.suptitle(f"GEONAV-AI Phase 6: Axis-Specific Error Breakdown ({self.dataset_name})")
        plt.tight_layout()
        plt.savefig(p_axis_path, dpi=150)
        plt.close()
        plot_paths["axis_errors"] = p_axis_path

        return plot_paths
