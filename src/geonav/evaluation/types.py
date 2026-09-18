"""Data types and result structures for trajectory and ground-truth evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrajectoryPoint:
    """Represents a single pose and kinematic state along a trajectory."""

    timestamp: float
    position: np.ndarray
    orientation: np.ndarray  # Unit quaternion [qw, qx, qy, qz]
    velocity: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate shapes and finite values."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError(f"Position must have shape (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Orientation quaternion must have shape (4,), got {self.orientation.shape}")
        if not np.all(np.isfinite(self.position)):
            raise ValueError(f"Position contains non-finite values: {self.position}")
        if not np.all(np.isfinite(self.orientation)):
            raise ValueError(f"Orientation contains non-finite values: {self.orientation}")

        # Normalize quaternion
        q_norm = np.linalg.norm(self.orientation)
        if q_norm < 1e-6:
            raise ValueError(f"Near-zero quaternion norm: {q_norm}")
        self.orientation /= q_norm

        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float64)
            if self.velocity.shape != (3,):
                raise ValueError(f"Velocity must have shape (3,), got {self.velocity.shape}")
            if not np.all(np.isfinite(self.velocity)):
                raise ValueError(f"Velocity contains non-finite values: {self.velocity}")


@dataclass
class AssociatedPair:
    """Pair of timestamp-associated estimated and ground-truth trajectory points."""

    timestamp_est: float
    timestamp_gt: float
    time_diff: float
    est_point: TrajectoryPoint
    gt_point: TrajectoryPoint


@dataclass
class ATEResult:
    """Absolute Trajectory Error metrics."""

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    final_error: float
    final_estimated_position: Tuple[float, float, float]
    final_groundtruth_position: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse_m": float(self.rmse),
            "mean_m": float(self.mean),
            "median_m": float(self.median),
            "std_m": float(self.std),
            "min_m": float(self.min),
            "max_m": float(self.max),
            "final_error_m": float(self.final_error),
            "final_estimated_position_m": list(self.final_estimated_position),
            "final_groundtruth_position_m": list(self.final_groundtruth_position),
        }


@dataclass
class AxisErrors:
    """Axis-wise positional error metrics."""

    x_rmse: float
    y_rmse: float
    z_rmse: float
    x_mae: float
    y_mae: float
    z_mae: float
    x_max: float
    y_max: float
    z_max: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x_rmse_m": float(self.x_rmse),
            "y_rmse_m": float(self.y_rmse),
            "z_rmse_m": float(self.z_rmse),
            "x_mae_m": float(self.x_mae),
            "y_mae_m": float(self.y_mae),
            "z_mae_m": float(self.z_mae),
            "x_max_m": float(self.x_max),
            "y_max_m": float(self.y_max),
            "z_max_m": float(self.z_max),
        }


@dataclass
class RPEIntervalResult:
    """Relative Pose Error metrics for a specific interval."""

    interval_name: str
    delta_frames: Optional[int]
    delta_time_s: Optional[float]
    num_pairs: int
    trans_rmse: float
    trans_mean: float
    trans_median: float
    trans_max: float
    rot_rmse_deg: float
    rot_mean_deg: float
    rot_median_deg: float
    rot_max_deg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interval_name": self.interval_name,
            "delta_frames": self.delta_frames,
            "delta_time_s": self.delta_time_s,
            "num_pairs": self.num_pairs,
            "trans_rmse_m": float(self.trans_rmse),
            "trans_mean_m": float(self.trans_mean),
            "trans_median_m": float(self.trans_median),
            "trans_max_m": float(self.trans_max),
            "rot_rmse_deg": float(self.rot_rmse_deg),
            "rot_mean_deg": float(self.rot_mean_deg),
            "rot_median_deg": float(self.rot_median_deg),
            "rot_max_deg": float(self.rot_max_deg),
        }


@dataclass
class OrientationErrorResult:
    """Absolute orientation error metrics."""

    rmse_deg: float
    mean_deg: float
    median_deg: float
    max_deg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse_deg": float(self.rmse_deg),
            "mean_deg": float(self.mean_deg),
            "median_deg": float(self.median_deg),
            "max_deg": float(self.max_deg),
        }


@dataclass
class VelocityErrorResult:
    """Linear velocity error metrics."""

    rmse: float
    x_rmse: float
    y_rmse: float
    z_rmse: float
    mean: float
    max: float
    final_error: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse_mps": float(self.rmse),
            "x_rmse_mps": float(self.x_rmse),
            "y_rmse_mps": float(self.y_rmse),
            "z_rmse_mps": float(self.z_rmse),
            "mean_mps": float(self.mean),
            "max_mps": float(self.max),
            "final_error_mps": float(self.final_error),
        }


@dataclass
class ScaleAnalysisResult:
    """Trajectory path length and scale ratio metrics."""

    estimated_path_length: float
    groundtruth_path_length: float
    scale_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimated_path_length_m": float(self.estimated_path_length),
            "groundtruth_path_length_m": float(self.groundtruth_path_length),
            "scale_ratio": float(self.scale_ratio),
        }


@dataclass
class AlignmentResult:
    """SE(3) or Sim(3) trajectory alignment results."""

    method: str
    scale: float
    rotation_matrix: np.ndarray
    translation: np.ndarray
    aligned_ate: ATEResult

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "scale": float(self.scale),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_m": [float(x) for x in self.translation],
            "aligned_ate": self.aligned_ate.to_dict(),
        }


@dataclass
class EvaluationMetrics:
    """Root aggregated ground-truth evaluation metrics."""

    dataset: str
    estimated_pose_count: int
    ground_truth_pose_count: int
    matched_pose_count: int
    unmatched_estimated_count: int
    unmatched_groundtruth_count: int
    time_association_threshold_s: float
    raw_ate: ATEResult
    axis_errors: AxisErrors
    rpe: Dict[str, RPEIntervalResult]
    orientation_error: OrientationErrorResult
    scale_analysis: ScaleAnalysisResult
    velocity_error: Optional[VelocityErrorResult] = None
    aligned_orientation_error: Optional[OrientationErrorResult] = None
    se3_alignment: Optional[AlignmentResult] = None
    sim3_alignment: Optional[AlignmentResult] = None
    mapping_statistics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to a JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "dataset": self.dataset,
            "estimated_pose_count": self.estimated_pose_count,
            "ground_truth_pose_count": self.ground_truth_pose_count,
            "matched_pose_count": self.matched_pose_count,
            "unmatched_estimated_count": self.unmatched_estimated_count,
            "unmatched_groundtruth_count": self.unmatched_groundtruth_count,
            "time_association_threshold_s": self.time_association_threshold_s,
            "raw_ate": self.raw_ate.to_dict(),
            "axis_errors": self.axis_errors.to_dict(),
            "rpe": {k: v.to_dict() for k, v in self.rpe.items()},
            "orientation_error": self.orientation_error.to_dict(),
            "scale_analysis": self.scale_analysis.to_dict(),
        }
        if self.aligned_orientation_error is not None:
            d["aligned_orientation_error"] = self.aligned_orientation_error.to_dict()
        if self.velocity_error is not None:
            d["velocity_error"] = self.velocity_error.to_dict()
        if self.se3_alignment is not None:
            d["se3_alignment"] = self.se3_alignment.to_dict()
        if self.sim3_alignment is not None:
            d["sim3_alignment"] = self.sim3_alignment.to_dict()
        if self.mapping_statistics is not None:
            d["mapping_statistics"] = self.mapping_statistics
        return d
