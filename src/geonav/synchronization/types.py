"""Data types for camera-IMU synchronized measurements."""

from dataclasses import dataclass
from typing import List, Optional

from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample


@dataclass
class SynchronizedMeasurement:
    """Represents a synchronized measurement package for Visual-Inertial Odometry.

    Associates a current camera frame with the sequence of high-rate IMU samples
    spanning the inter-frame time interval (T_previous < t_IMU <= T_current).

    Attributes:
        camera_frame: The camera frame at the current timestamp (T_current).
        imu_samples: Chronological list of IMU samples falling within the interval.
        start_timestamp: Previous camera frame timestamp (T_previous), or None if this is the initial frame.
        end_timestamp: Current camera frame timestamp (T_current).
    """

    camera_frame: CameraFrame
    imu_samples: List[IMUSample]
    start_timestamp: Optional[float]
    end_timestamp: float

    @property
    def is_first_frame(self) -> bool:
        """Return True if this package represents the first camera frame in the sequence."""
        return self.start_timestamp is None

    @property
    def num_imu_samples(self) -> int:
        """Return the number of IMU samples associated with this interval."""
        return len(self.imu_samples)

    @property
    def duration_s(self) -> Optional[float]:
        """Return duration of the camera interval in seconds, or None for the initial frame."""
        if self.start_timestamp is None:
            return None
        return self.end_timestamp - self.start_timestamp
