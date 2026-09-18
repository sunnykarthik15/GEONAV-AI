"""Evaluation module: Quantitative ground-truth evaluation and trajectory benchmarking."""

from geonav.evaluation.alignment import align_trajectory_points, umeyama_alignment
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.evaluator import TrajectoryEvaluator
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
    ATEResult,
    AxisErrors,
    EvaluationMetrics,
    OrientationErrorResult,
    RPEIntervalResult,
    ScaleAnalysisResult,
    TrajectoryPoint,
    VelocityErrorResult,
)

__all__ = [
    "AlignmentResult",
    "AssociatedPair",
    "ATEResult",
    "AxisErrors",
    "EvaluationMetrics",
    "OrientationErrorResult",
    "RPEIntervalResult",
    "ScaleAnalysisResult",
    "TrajectoryEvaluator",
    "TrajectoryPoint",
    "VelocityErrorResult",
    "align_trajectory_points",
    "associate_timestamps",
    "compute_ate",
    "compute_axis_errors",
    "compute_orientation_errors",
    "compute_rpe",
    "compute_scale_analysis",
    "compute_velocity_errors",
    "umeyama_alignment",
]
