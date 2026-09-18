"""Navigation message and output implementations for GEONAV-AI.

Defines the NavigationMessage — the canonical typed output of the full
GEONAV-AI navigation pipeline — and provides a ConsoleNavigationOutput
implementation for logging/debugging.

All units are explicit and documented on each field.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class TrackingStatus(Enum):
    """Visual-inertial tracking quality indicator.

    States:
        INITIALIZING:   Pipeline is collecting initial sensor data; state estimate not yet valid.
        TRACKING:       Normal operation; visual and inertial data fused successfully.
        DEGRADED:       Visual tracking failed; falling back to IMU dead-reckoning only.
        LOST:           Navigation estimate is unreliable; reset required.
    """

    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


@dataclass
class NavigationMessage:
    """Canonical typed navigation output of the GEONAV-AI pipeline.

    All fields use SI units as documented below. Consumers must not
    assume any other unit convention.

    Attributes:
        timestamp:          State epoch in seconds (float, monotonically increasing).
        position:           3D position in meters [x, y, z], local inertial NED frame.
        velocity:           3D velocity in m/s [vx, vy, vz], local inertial NED frame.
        orientation:        Unit quaternion [qw, qx, qy, qz] (body orientation w.r.t. world).
        tracking_status:    Current VIO tracking quality (TrackingStatus enum).
        ml_confidence:      ML correction confidence in [0.0, 1.0] (0.0 if ML disabled/unloaded).
        tracked_features:   Number of features tracked in the current frame (0 if unavailable).
        inlier_ratio:       RANSAC inlier ratio in [0.0, 1.0] (0.0 if unavailable).
        imu_available:      True if IMU data was present for this estimate.
        is_numerically_valid: True if position/velocity/orientation are all finite.
        sequence_index:     Frame/measurement index (0-based).
        health_flags:       Dictionary of boolean health indicators for subsystem diagnostics.
    """

    timestamp: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # [qw, qx, qy, qz]
    tracking_status: TrackingStatus = TrackingStatus.INITIALIZING
    ml_confidence: float = 0.0
    tracked_features: int = 0
    inlier_ratio: float = 0.0
    imu_available: bool = True
    is_numerically_valid: bool = True
    sequence_index: int = 0
    health_flags: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate numerical validity of position, velocity, orientation."""
        p = np.array(self.position, dtype=np.float64)
        v = np.array(self.velocity, dtype=np.float64)
        q = np.array(self.orientation, dtype=np.float64)
        self.is_numerically_valid = (
            bool(np.all(np.isfinite(p)))
            and bool(np.all(np.isfinite(v)))
            and bool(np.all(np.isfinite(q)))
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "timestamp_s": float(self.timestamp),
            "position_m": list(self.position),
            "velocity_mps": list(self.velocity),
            "orientation_qwxyz": list(self.orientation),
            "tracking_status": self.tracking_status.value,
            "ml_confidence": float(self.ml_confidence),
            "tracked_features": int(self.tracked_features),
            "inlier_ratio": float(self.inlier_ratio),
            "imu_available": bool(self.imu_available),
            "is_numerically_valid": bool(self.is_numerically_valid),
            "sequence_index": int(self.sequence_index),
            "health_flags": self.health_flags,
        }


class ConsoleNavigationOutput:
    """NavigationOutput implementation that prints messages to stdout.

    Useful for debugging, logging, and offline validation.
    """

    def __init__(self, verbose: bool = False, every_n: int = 100) -> None:
        """Initialize console output.

        Args:
            verbose: If True, print every message. If False, print every every_n messages.
            every_n: Interval between printed messages when verbose=False.
        """
        self.verbose = verbose
        self.every_n = every_n
        self._count = 0

    def publish(self, message: NavigationMessage) -> None:
        """Print navigation message to stdout at configured interval.

        Args:
            message: NavigationMessage to publish.
        """
        self._count += 1
        if self.verbose or self._count % self.every_n == 0:
            p = message.position
            v = message.velocity
            print(
                f"[NAV #{message.sequence_index:05d}] "
                f"t={message.timestamp:.3f}s | "
                f"pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m | "
                f"vel=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}) m/s | "
                f"status={message.tracking_status.value} | "
                f"conf={message.ml_confidence:.2f} | "
                f"feats={message.tracked_features} | "
                f"valid={message.is_numerically_valid}"
            )

    def close(self) -> None:
        """No-op: console output has no resources to release."""
        pass
