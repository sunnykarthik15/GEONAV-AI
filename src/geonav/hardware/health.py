"""Health monitoring state machine for GEONAV-AI pipeline.

Tracks the health state of all pipeline subsystems and provides
a unified health indicator with defined state transitions.

Health states:
    INITIALIZING → TRACKING: Once VIO is initialized with sufficient features.
    TRACKING → DEGRADED:     When visual tracking fails (IMU-only fallback).
    DEGRADED → TRACKING:     When visual tracking recovers.
    DEGRADED → LOST:         When IMU-only propagation also fails or exceeds duration limit.
    LOST → INITIALIZING:     On explicit reset or re-initialization.
    Any → LOST:              On numerical failure (NaN/Inf in position/velocity).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class HealthState(Enum):
    """Pipeline operational health states.

    Transition conditions:
        INITIALIZING:
            Entry: Always on construction or reset.
            Exit → TRACKING: VIO initialized AND tracked_features >= min_features.

        TRACKING:
            Entry: From INITIALIZING or DEGRADED when visual tracking is active.
            Exit → DEGRADED: visual_success = False for current frame.
            Exit → LOST: numerical_failure = True.

        DEGRADED:
            Entry: From TRACKING when visual tracking fails.
            Represents: IMU dead-reckoning only; position error accumulating.
            Exit → TRACKING: visual_success = True (tracking recovered).
            Exit → LOST: consecutive_degraded_frames > degraded_timeout_frames
                         OR numerical_failure = True.

        LOST:
            Entry: From any state on numerical failure, or from DEGRADED after timeout.
            Represents: Estimate is unreliable; reset required.
            Exit → INITIALIZING: On explicit reset() call.
    """

    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


@dataclass
class SystemHealth:
    """Snapshot of pipeline health indicators at a single timestep.

    Attributes:
        state: Current HealthState.
        vio_initialized: Whether VIO has completed initialization.
        tracked_features: Number of features tracked in the current frame.
        inlier_ratio: Fraction of features passing RANSAC.
        visual_success: Whether visual motion estimation succeeded.
        imu_available: Whether IMU data was available for this step.
        ml_confidence: ML correction confidence [0.0, 1.0].
        timestamp_valid: Whether the current frame timestamp is monotonically increasing.
        numerically_valid: Whether position/velocity/orientation are all finite.
        consecutive_degraded_frames: Counter of consecutive frames without visual tracking.
        message: Human-readable health summary.
    """

    state: HealthState = HealthState.INITIALIZING
    vio_initialized: bool = False
    tracked_features: int = 0
    inlier_ratio: float = 0.0
    visual_success: bool = False
    imu_available: bool = True
    ml_confidence: float = 0.0
    timestamp_valid: bool = True
    numerically_valid: bool = True
    consecutive_degraded_frames: int = 0
    message: str = "Initializing"


class HealthMonitor:
    """Pipeline health state machine with defined transition conditions.

    Updates health state on each frame based on measured indicators.
    Provides a summary SystemHealth snapshot after each update.
    """

    def __init__(
        self,
        min_features_for_tracking: int = 10,
        min_inlier_ratio_for_tracking: float = 0.3,
        degraded_timeout_frames: int = 50,
    ) -> None:
        """Initialize health monitor.

        Args:
            min_features_for_tracking: Minimum tracked features to remain in TRACKING state.
            min_inlier_ratio_for_tracking: Minimum RANSAC inlier ratio for valid visual update.
            degraded_timeout_frames: Max consecutive DEGRADED frames before transitioning to LOST.
        """
        self.min_features = min_features_for_tracking
        self.min_inlier_ratio = min_inlier_ratio_for_tracking
        self.degraded_timeout = degraded_timeout_frames

        self._state: HealthState = HealthState.INITIALIZING
        self._consecutive_degraded: int = 0
        self._vio_initialized: bool = False
        self._last_timestamp: Optional[float] = None

    @property
    def state(self) -> HealthState:
        """Current health state."""
        return self._state

    def reset(self) -> None:
        """Reset monitor to INITIALIZING state."""
        self._state = HealthState.INITIALIZING
        self._consecutive_degraded = 0
        self._vio_initialized = False
        self._last_timestamp = None

    def update(
        self,
        *,
        timestamp: float,
        vio_initialized: bool,
        tracked_features: int,
        inlier_ratio: float,
        visual_success: bool,
        imu_available: bool,
        ml_confidence: float,
        position: Optional[tuple] = None,
        velocity: Optional[tuple] = None,
        orientation: Optional[tuple] = None,
    ) -> SystemHealth:
        """Update health state based on current-frame indicators.

        Args:
            timestamp: Current frame timestamp in seconds.
            vio_initialized: Whether VIO pipeline is initialized.
            tracked_features: Number of tracked keypoints.
            inlier_ratio: Fraction of RANSAC inliers.
            visual_success: Whether visual relative motion was estimated.
            imu_available: Whether IMU data is available.
            ml_confidence: ML confidence value [0.0, 1.0].
            position: Current position (x, y, z) for numerical validation.
            velocity: Current velocity (vx, vy, vz) for numerical validation.
            orientation: Current orientation [qw, qx, qy, qz] for numerical validation.

        Returns:
            SystemHealth: Snapshot of current health state.
        """
        self._vio_initialized = vio_initialized

        # Numerical validity check
        numerically_valid = True
        if position is not None:
            numerically_valid &= bool(np.all(np.isfinite(np.array(position))))
        if velocity is not None:
            numerically_valid &= bool(np.all(np.isfinite(np.array(velocity))))
        if orientation is not None:
            numerically_valid &= bool(np.all(np.isfinite(np.array(orientation))))

        # Timestamp monotonicity check
        timestamp_valid = True
        if self._last_timestamp is not None:
            timestamp_valid = float(timestamp) > self._last_timestamp
        self._last_timestamp = float(timestamp)

        # State machine transitions
        prev_state = self._state

        if not numerically_valid:
            # Numerical failure: always transition to LOST from any state
            self._state = HealthState.LOST
            self._consecutive_degraded = 0
            message = "LOST: Numerical failure (NaN/Inf in navigation state)"

        elif self._state == HealthState.INITIALIZING:
            if vio_initialized and tracked_features >= self.min_features:
                self._state = HealthState.TRACKING
                message = "TRACKING: VIO initialized, visual tracking active"
            else:
                message = f"INITIALIZING: waiting for VIO + features ({tracked_features}/{self.min_features})"

        elif self._state == HealthState.TRACKING:
            if not visual_success or tracked_features < self.min_features:
                self._state = HealthState.DEGRADED
                self._consecutive_degraded = 1
                message = f"DEGRADED: Visual tracking failed (feats={tracked_features}, success={visual_success})"
            else:
                message = f"TRACKING: feats={tracked_features}, inlier_ratio={inlier_ratio:.2f}"

        elif self._state == HealthState.DEGRADED:
            if visual_success and tracked_features >= self.min_features:
                self._state = HealthState.TRACKING
                self._consecutive_degraded = 0
                message = f"TRACKING: Visual tracking recovered (feats={tracked_features})"
            elif self._consecutive_degraded >= self.degraded_timeout:
                self._state = HealthState.LOST
                self._consecutive_degraded = 0
                message = f"LOST: Degraded for {self._consecutive_degraded} consecutive frames"
            else:
                self._consecutive_degraded += 1
                message = (f"DEGRADED: IMU-only ({self._consecutive_degraded}/"
                           f"{self.degraded_timeout} frames, feats={tracked_features})")

        elif self._state == HealthState.LOST:
            message = "LOST: Reset required"

        else:
            message = f"UNKNOWN state: {self._state}"

        return SystemHealth(
            state=self._state,
            vio_initialized=vio_initialized,
            tracked_features=tracked_features,
            inlier_ratio=inlier_ratio,
            visual_success=visual_success,
            imu_available=imu_available,
            ml_confidence=ml_confidence,
            timestamp_valid=timestamp_valid,
            numerically_valid=numerically_valid,
            consecutive_degraded_frames=self._consecutive_degraded,
            message=message,
        )
