"""Sensor synchronization interface, buffering, and deterministic temporal windowing."""

from collections.abc import Iterable, Iterator, Sequence
import math
from typing import List, Optional

from geonav.config.settings import SyncConfig
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.synchronization.types import SynchronizedMeasurement


def validate_camera_frames(frames: Sequence[CameraFrame]) -> None:
    """Validate that camera frames are valid and strictly increasing in time.

    Args:
        frames: Sequence of CameraFrame objects.

    Raises:
        TypeError: If an element is not a CameraFrame.
        ValueError: If timestamps are non-finite, negative, or not strictly increasing.
    """
    for i, frame in enumerate(frames):
        if not isinstance(frame, CameraFrame):
            raise TypeError(f"Expected CameraFrame instance at index {i}, got {type(frame)}")
        ts = frame.timestamp
        if math.isnan(ts) or math.isinf(ts) or ts < 0:
            raise ValueError(f"Camera frame at index {i} has invalid timestamp: {ts}")
        if i > 0 and ts <= frames[i - 1].timestamp:
            raise ValueError(
                f"Camera frames must have strictly increasing timestamps: "
                f"frame {i} ({ts}) <= frame {i - 1} ({frames[i - 1].timestamp})"
            )


def validate_imu_samples(samples: Sequence[IMUSample]) -> None:
    """Validate that IMU samples are valid, finite, and chronologically non-decreasing.

    Args:
        samples: Sequence of IMUSample objects.

    Raises:
        TypeError: If an element is not an IMUSample.
        ValueError: If timestamps or values are non-finite, negative, or out of order.
    """
    for k, sample in enumerate(samples):
        if not isinstance(sample, IMUSample):
            raise TypeError(f"Expected IMUSample instance at index {k}, got {type(sample)}")
        ts = sample.timestamp
        if math.isnan(ts) or math.isinf(ts) or ts < 0:
            raise ValueError(f"IMU sample at index {k} has invalid timestamp: {ts}")
        for val in sample.linear_acceleration:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"IMU sample at index {k} has non-finite linear acceleration value: {val}")
        for val in sample.angular_velocity:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"IMU sample at index {k} has non-finite angular velocity value: {val}")
        if k > 0 and ts < samples[k - 1].timestamp:
            raise ValueError(
                f"IMU samples must have chronologically non-decreasing timestamps: "
                f"sample {k} ({ts}) < sample {k - 1} ({samples[k - 1].timestamp})"
            )


def synchronize_streams(
    camera_stream: Iterable[CameraFrame],
    imu_stream: Iterable[IMUSample],
    config: Optional[SyncConfig] = None,
) -> Iterator[SynchronizedMeasurement]:
    """Deterministically synchronize camera frames and IMU samples using a two-pointer window.

    Boundary Policy:
        T_previous < t_IMU <= T_current

        - Exclusive Left (t_IMU > T_previous): Prevents duplicate assignment between intervals.
        - Inclusive Right (t_IMU <= T_current): Deterministically assigns the boundary sample.
        - First Frame (T_0): If include_first_frame is True, emits SynchronizedMeasurement
          with start_timestamp=None, end_timestamp=T_0, and imu_samples=[].
        - IMU samples with t <= T_0 are excluded from inter-frame intervals.
        - IMU samples with t > T_final are excluded from completed intervals.
        - No original IMU samples are mutated or deleted.

    Args:
        camera_stream: Iterable of CameraFrame objects.
        imu_stream: Iterable of IMUSample objects.
        config: Optional SyncConfig configuration.

    Yields:
        SynchronizedMeasurement: Synchronized measurement packages.

    Raises:
        ValueError: If inputs fail numerical/timestamp validation, or if require_imu_for_interval
            is True and an inter-frame interval contains zero IMU samples.
    """
    if config is None:
        config = SyncConfig()

    # Materialize camera frames to validate ordering and bounds
    camera_frames = list(camera_stream)
    if not camera_frames:
        return

    validate_camera_frames(camera_frames)

    # Materialize IMU samples to validate ordering and bounds
    imu_samples = list(imu_stream)
    validate_imu_samples(imu_samples)

    imu_idx = 0
    num_imu = len(imu_samples)

    first_frame = camera_frames[0]
    t_first = first_frame.timestamp

    # Handle initial camera frame
    if config.include_first_frame:
        yield SynchronizedMeasurement(
            camera_frame=first_frame,
            imu_samples=[],
            start_timestamp=None,
            end_timestamp=t_first,
        )

    # Exclude IMU samples occurring before or exactly at the first camera timestamp (t <= T_0)
    # from inter-frame synchronization without mutating the original input sequence.
    while imu_idx < num_imu and imu_samples[imu_idx].timestamp <= t_first:
        imu_idx += 1

    t_prev = t_first

    # Process consecutive camera intervals: (T_previous, T_current]
    for curr_frame in camera_frames[1:]:
        t_curr = curr_frame.timestamp
        interval_imu: List[IMUSample] = []

        # Collect all IMU samples satisfying: T_previous < t_IMU <= T_current
        while imu_idx < num_imu and imu_samples[imu_idx].timestamp <= t_curr:
            interval_imu.append(imu_samples[imu_idx])
            imu_idx += 1

        if config.require_imu_for_interval and len(interval_imu) == 0:
            raise ValueError(
                f"No IMU samples found for camera interval ({t_prev}, {t_curr}] "
                f"with require_imu_for_interval=True"
            )

        yield SynchronizedMeasurement(
            camera_frame=curr_frame,
            imu_samples=interval_imu,
            start_timestamp=t_prev,
            end_timestamp=t_curr,
        )

        t_prev = t_curr


class SensorSynchronizer:
    """Buffers camera frames and IMU samples and performs deterministic temporal alignment.

    Preserves Phase 1 FIFO buffer behaviors while providing Phase 3 synchronization capabilities.
    """

    def __init__(self, max_camera_buffer: int = 100, max_imu_buffer: int = 1000) -> None:
        if max_camera_buffer <= 0 or max_imu_buffer <= 0:
            raise ValueError("Buffer limits must be positive integers")
        self.max_camera_buffer: int = max_camera_buffer
        self.max_imu_buffer: int = max_imu_buffer
        self._camera_buffer: List[CameraFrame] = []
        self._imu_buffer: List[IMUSample] = []

    def add_camera_frame(self, frame: CameraFrame) -> None:
        """Add a camera frame to the FIFO buffer."""
        if not isinstance(frame, CameraFrame):
            raise TypeError("Expected CameraFrame instance")
        if len(self._camera_buffer) >= self.max_camera_buffer:
            self._camera_buffer.pop(0)
        self._camera_buffer.append(frame)

    def add_imu_sample(self, sample: IMUSample) -> None:
        """Add an IMU sample to the FIFO buffer."""
        if not isinstance(sample, IMUSample):
            raise TypeError("Expected IMUSample instance")
        if len(self._imu_buffer) >= self.max_imu_buffer:
            self._imu_buffer.pop(0)
        self._imu_buffer.append(sample)

    def get_camera_buffer(self) -> List[CameraFrame]:
        """Return a copy of the current camera buffer."""
        return list(self._camera_buffer)

    def get_imu_buffer(self) -> List[IMUSample]:
        """Return a copy of the current IMU buffer."""
        return list(self._imu_buffer)

    def clear_buffers(self) -> None:
        """Clear all buffered sensor data."""
        self._camera_buffer.clear()
        self._imu_buffer.clear()

    def synchronize(self, config: Optional[SyncConfig] = None) -> List[SynchronizedMeasurement]:
        """Synchronize currently buffered camera frames and IMU samples.

        Args:
            config: Optional synchronization configuration.

        Returns:
            List[SynchronizedMeasurement]: Synchronized measurement packages.
        """
        return list(synchronize_streams(self._camera_buffer, self._imu_buffer, config=config))
