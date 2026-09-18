"""Data types and status enums for Visual-Inertial Odometry."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class VIOStatus(Enum):
    """Execution status of the Visual-Inertial Odometry estimator."""

    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZED = "INITIALIZED"
    TRACKING_OK = "TRACKING_OK"
    DEGRADED_IMU_ONLY = "DEGRADED_IMU_ONLY"
    FAILED = "FAILED"


@dataclass
class VisualTrackingResult:
    """Represents the output of the visual front-end for a camera interval.

    Attributes:
        success: True if visual relative motion was successfully estimated.
        num_tracked: Number of feature points successfully tracked between frames.
        num_inliers: Number of geometric inliers passing Essential matrix RANSAC.
        R_rel: 3x3 relative rotation matrix from previous to current frame, or None.
        t_rel: 3-element unit translation direction vector from previous to current frame, or None.
        points_prev: Tracked keypoint pixel coordinates in previous frame (N, 2).
        points_curr: Tracked keypoint pixel coordinates in current frame (N, 2).
        status_message: Human-readable explanation of tracking status or failure reason.
    """

    success: bool
    num_tracked: int
    num_inliers: int
    R_rel: Optional[np.ndarray] = None
    t_rel: Optional[np.ndarray] = None
    points_prev: Optional[np.ndarray] = None
    points_curr: Optional[np.ndarray] = None
    status_message: str = ""


@dataclass
class IMUPropagationResult:
    """Represents the integrated inertial state increment over a time interval.

    Attributes:
        success: True if propagation succeeded across valid IMU measurements.
        delta_p: Integrated 3D position displacement vector in world frame (meters).
        delta_v: Integrated 3D linear velocity change vector in world frame (m/s).
        delta_q: Integrated relative rotation unit quaternion (qw, qx, qy, qz).
        dt: Total elapsed duration of the integrated IMU samples (seconds).
        status_message: Description of propagation result or failure reason.
    """

    success: bool
    delta_p: np.ndarray
    delta_v: np.ndarray
    delta_q: np.ndarray
    dt: float
    status_message: str = ""
