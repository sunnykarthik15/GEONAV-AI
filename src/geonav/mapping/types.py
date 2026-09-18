"""Data types and representations for 3D point-cloud mapping."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TriangulationStatus(Enum):
    """Result status for two-view landmark triangulation."""

    SUCCESS = "SUCCESS"
    REJECTED_PARALLAX = "REJECTED_PARALLAX"
    REJECTED_NEGATIVE_DEPTH = "REJECTED_NEGATIVE_DEPTH"
    REJECTED_DEPTH_RANGE = "REJECTED_DEPTH_RANGE"
    REJECTED_REPROJECTION = "REJECTED_REPROJECTION"
    REJECTED_NUMERICAL = "REJECTED_NUMERICAL"


@dataclass
class Landmark:
    """Represents a triangulated 3D landmark in world navigation coordinates.

    Attributes:
        id: Unique landmark identifier.
        position_world: 3D coordinates (X, Y, Z) in world frame (meters).
        observation_count: Number of camera frames observing this landmark.
        first_timestamp: Timestamp of the initial observation (seconds).
        last_timestamp: Timestamp of the most recent observation (seconds).
        reprojection_error: Mean reprojection error across observations in pixels.
        color: Optional RGB color tuple (R, G, B) with values in [0, 255].
    """

    id: int
    position_world: np.ndarray
    observation_count: int = 2
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    reprojection_error: float = 0.0
    color: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        """Validate landmark coordinates."""
        arr = np.asarray(self.position_world, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Landmark position must have shape (3,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Landmark position contains non-finite values: {arr}")
        self.position_world = arr


@dataclass
class Keyframe:
    """Represents a reference keyframe used for multi-view triangulation.

    Attributes:
        id: Unique keyframe identifier.
        timestamp: Image capture timestamp (seconds).
        R_wc: 3x3 rotation matrix transforming vectors from camera to world frame.
        p_wc: 3D camera optical center position in world frame (meters).
        P_norm: 3x4 normalized camera projection matrix [R_cw | t_cw].
        feature_ids: 1D array of persistent feature track IDs (N,).
        feature_points: 2D pixel coordinates in image plane (N, 2).
        feature_points_norm: 2D undistorted normalized ray coordinates (N, 2).
        frame_image: Optional grayscale or RGB image array.
    """

    id: int
    timestamp: float
    R_wc: np.ndarray
    p_wc: np.ndarray
    P_norm: np.ndarray
    feature_ids: np.ndarray
    feature_points: np.ndarray
    feature_points_norm: np.ndarray
    frame_image: Optional[np.ndarray] = None


@dataclass
class MappingStatistics:
    """Quantitative performance and diagnostics metrics for the mapping run."""

    total_frames: int = 0
    keyframes_count: int = 0
    total_landmarks: int = 0
    triangulation_attempts: int = 0
    triangulation_successes: int = 0
    rejected_parallax: int = 0
    rejected_negative_depth: int = 0
    rejected_depth_range: int = 0
    rejected_reprojection: int = 0
    rejected_numerical: int = 0
    mean_reprojection_error: float = 0.0
    median_reprojection_error: float = 0.0
    max_reprojection_error: float = 0.0
    min_depth: float = 0.0
    max_depth: float = 0.0
    spatial_extent: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    runtime_s: float = 0.0
    fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a serializable dictionary."""
        return {
            "total_frames": self.total_frames,
            "keyframes_count": self.keyframes_count,
            "total_landmarks": self.total_landmarks,
            "triangulation_attempts": self.triangulation_attempts,
            "triangulation_successes": self.triangulation_successes,
            "rejected_parallax": self.rejected_parallax,
            "rejected_negative_depth": self.rejected_negative_depth,
            "rejected_depth_range": self.rejected_depth_range,
            "rejected_reprojection": self.rejected_reprojection,
            "rejected_numerical": self.rejected_numerical,
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "median_reprojection_error": float(self.median_reprojection_error),
            "max_reprojection_error": float(self.max_reprojection_error),
            "min_depth": float(self.min_depth),
            "max_depth": float(self.max_depth),
            "spatial_extent": {
                k: [float(v[0]), float(v[1])] for k, v in self.spatial_extent.items()
            },
            "runtime_s": float(self.runtime_s),
            "fps": float(self.fps),
        }
