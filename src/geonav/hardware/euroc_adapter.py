"""EuRoC MAV dataset adapters implementing CameraSource and IMUSource interfaces.

These adapters wrap the existing EurocDataset loader to make it conform
to the CameraSource and IMUSource abstract interfaces, enabling the same
pipeline code to work with both dataset validation and (future) real hardware.
"""

from pathlib import Path
from typing import Iterator, Optional, Union
import numpy as np

from geonav.datasets.euroc import EurocDataset
from geonav.hardware.interfaces import CameraSource, IMUSource
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample


class EurocCameraSource(CameraSource):
    """CameraSource adapter for EuRoC MAV dataset sequences.

    Wraps EurocDataset.images() to implement the CameraSource interface.
    Suitable for offline validation only (not real-time streaming).
    """

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """Initialize the EuRoC camera source.

        Args:
            dataset_path: Path to the EuRoC sequence root directory
                          (e.g., data/raw/euroc/MH_01_easy).
        """
        self._path = Path(dataset_path)
        self._dataset = EurocDataset(self._path)

    def frames(self) -> Iterator[CameraFrame]:
        """Yield camera frames from EuRoC sequence in chronological order.

        Yields:
            CameraFrame: Next camera frame with timestamp and image data.
        """
        for img_record in self._dataset.images():
            yield img_record.to_camera_frame()

    def camera_matrix(self) -> Optional[np.ndarray]:
        """Return the camera intrinsic matrix from EuRoC calibration.

        Returns:
            np.ndarray: 3x3 intrinsic matrix K, or None if unavailable.
        """
        cal = self._dataset.camera_calibration
        if cal is None:
            return None
        try:
            fx = cal.get("fx") or cal.get("focal_length_x")
            fy = cal.get("fy") or cal.get("focal_length_y")
            cx = cal.get("cx") or cal.get("principal_point_x")
            cy = cal.get("cy") or cal.get("principal_point_y")
            if None in (fx, fy, cx, cy):
                return None
            return np.array([
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        except (KeyError, TypeError):
            return None

    def reset(self) -> None:
        """Re-initialize the dataset loader to replay from the beginning."""
        self._dataset = EurocDataset(self._path)

    @property
    def dataset(self) -> EurocDataset:
        """Provide direct access to the underlying EurocDataset instance."""
        return self._dataset


class EurocIMUSource(IMUSource):
    """IMUSource adapter for EuRoC MAV dataset sequences.

    Wraps EurocDataset.imu() to implement the IMUSource interface.
    Suitable for offline validation only (not real-time streaming).
    """

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """Initialize the EuRoC IMU source.

        Args:
            dataset_path: Path to the EuRoC sequence root directory.
        """
        self._path = Path(dataset_path)
        self._dataset = EurocDataset(self._path)

    def samples(self) -> Iterator[IMUSample]:
        """Yield IMU samples from EuRoC sequence in chronological order.

        Yields:
            IMUSample: Next IMU measurement with timestamp, acceleration, and angular velocity.

        Unit conventions:
            - timestamp: seconds
            - linear_acceleration: m/s²
            - angular_velocity: rad/s
        """
        for imu_record in self._dataset.imu():
            yield imu_record.to_imu_sample()

    def reset(self) -> None:
        """Re-initialize the dataset loader to replay from the beginning."""
        self._dataset = EurocDataset(self._path)
