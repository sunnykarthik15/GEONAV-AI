"""Inertial Measurement Unit (IMU) sensor interface and sample data representation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple


@dataclass
class IMUSample:
    """Represents a single 6-DOF IMU reading.

    Attributes:
        timestamp: Monotonic sample timestamp in seconds (must be non-negative).
        linear_acceleration: 3-axis acceleration vector (ax, ay, az) in m/s^2.
        angular_velocity: 3-axis angular rates vector (wx, wy, wz) in rad/s.
    """

    timestamp: float
    linear_acceleration: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a float or integer")
        if math.isnan(self.timestamp) or math.isinf(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"timestamp must be a finite non-negative number, got {self.timestamp}")

        if not isinstance(self.linear_acceleration, Sequence) or len(self.linear_acceleration) != 3:
            raise ValueError(f"linear_acceleration must contain exactly 3 values, got {self.linear_acceleration}")
        for val in self.linear_acceleration:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"linear_acceleration values must be finite numbers, got {val}")

        if not isinstance(self.angular_velocity, Sequence) or len(self.angular_velocity) != 3:
            raise ValueError(f"angular_velocity must contain exactly 3 values, got {self.angular_velocity}")
        for val in self.angular_velocity:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"angular_velocity values must be finite numbers, got {val}")

        self.linear_acceleration = (
            float(self.linear_acceleration[0]),
            float(self.linear_acceleration[1]),
            float(self.linear_acceleration[2]),
        )
        self.angular_velocity = (
            float(self.angular_velocity[0]),
            float(self.angular_velocity[1]),
            float(self.angular_velocity[2]),
        )


class IMUSensor(ABC):
    """Abstract base class representing an IMU sensor data source."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check whether the IMU sensor is currently connected."""
        raise NotImplementedError

    @abstractmethod
    def read_sample(self) -> Optional[IMUSample]:
        """Read and return the latest IMU sample if available."""
        raise NotImplementedError


class DatasetIMUSensor(IMUSensor):
    """IMU sensor adapter providing samples sequentially from a dataset loader."""

    def __init__(self, loader: Any) -> None:
        """Initialize adapter with a dataset IMU loader or iterable of DatasetIMUSample.

        Args:
            loader: Iterable or sequence of DatasetIMUSample instances.
        """
        self._loader = loader
        self._iterator: Optional[Any] = None
        self._connected: bool = True

    def is_connected(self) -> bool:
        """Check whether sensor is connected."""
        return self._connected

    def connect(self) -> None:
        """Connect sensor and initialize iterator."""
        self._connected = True
        self._iterator = iter(self._loader)

    def disconnect(self) -> None:
        """Disconnect sensor."""
        self._connected = False

    def read_sample(self) -> Optional[IMUSample]:
        """Read the next sequential IMU sample from the dataset.

        Returns:
            Optional[IMUSample]: The next IMU sample, or None if exhausted/disconnected.
        """
        if not self._connected:
            return None
        if self._iterator is None:
            self._iterator = iter(self._loader)
        try:
            item = next(self._iterator)
            if hasattr(item, "to_imu_sample"):
                return item.to_imu_sample()
            if isinstance(item, IMUSample):
                return item
            raise TypeError(f"Unexpected item type from IMU dataset loader: {type(item)}")
        except StopIteration:
            return None

