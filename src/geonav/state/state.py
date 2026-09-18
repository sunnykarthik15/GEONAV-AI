"""Navigation state representation for vehicle pose and motion."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class NavigationState:
    """Represents vehicle navigation state (pose and kinematics) at a specific timestamp.

    Attributes:
        timestamp: Monotonic epoch/system timestamp in seconds (must be non-negative).
        position: 3D position vector (x, y, z) in meters in the navigation frame.
        velocity: 3D linear velocity vector (vx, vy, vz) in m/s in the navigation frame.
        orientation: Quaternion representation (qw, qx, qy, qz) representing rotation.
    """

    timestamp: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a float or integer")
        if math.isnan(self.timestamp) or math.isinf(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"timestamp must be a finite non-negative number, got {self.timestamp}")

        if not isinstance(self.position, Sequence) or len(self.position) != 3:
            raise ValueError(f"position must contain exactly 3 values, got {self.position}")
        for val in self.position:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"position values must be finite numbers, got {val}")

        if not isinstance(self.velocity, Sequence) or len(self.velocity) != 3:
            raise ValueError(f"velocity must contain exactly 3 values, got {self.velocity}")
        for val in self.velocity:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"velocity values must be finite numbers, got {val}")

        if not isinstance(self.orientation, Sequence) or len(self.orientation) != 4:
            raise ValueError(f"orientation quaternion must contain exactly 4 values, got {self.orientation}")
        for val in self.orientation:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"orientation values must be finite numbers, got {val}")

        self.position = (
            float(self.position[0]),
            float(self.position[1]),
            float(self.position[2]),
        )
        self.velocity = (
            float(self.velocity[0]),
            float(self.velocity[1]),
            float(self.velocity[2]),
        )
        self.orientation = (
            float(self.orientation[0]),
            float(self.orientation[1]),
            float(self.orientation[2]),
            float(self.orientation[3]),
        )
