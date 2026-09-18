"""Data types and representations for EuRoC dataset ingestion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample


@dataclass
class DatasetImage:
    """Represents a camera frame entry from a visual-inertial dataset.

    Attributes:
        timestamp_ns: Nanosecond timestamp from dataset index.
        timestamp: Timestamp converted to seconds (timestamp_ns * 1e-9).
        filename: Image filename as recorded in dataset CSV.
        file_path: Resolved path to the image file on disk.
    """

    timestamp_ns: int
    timestamp: float
    filename: str
    file_path: Path

    def load_data(self) -> np.ndarray:
        """Lazily load and return raw pixel data from disk without image processing.

        Returns:
            np.ndarray: Loaded image array.

        Raises:
            FileNotFoundError: If the image file does not exist on disk.
            ValueError: If the image file cannot be decoded.
        """
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Image file not found: {self.file_path}")

        img = cv2.imread(str(self.file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode image file: {self.file_path}")
        return img

    def to_camera_frame(self) -> CameraFrame:
        """Convert to Phase 1 CameraFrame by loading pixel data.

        Returns:
            CameraFrame: Initialized CameraFrame instance.
        """
        frame_data = self.load_data()
        return CameraFrame(timestamp=self.timestamp, frame_data=frame_data)


@dataclass
class DatasetIMUSample:
    """Represents an IMU measurement entry from a visual-inertial dataset.

    EuRoC convention: w_x, w_y, w_z (rad/s), a_x, a_y, a_z (m/s^2).

    Attributes:
        timestamp_ns: Nanosecond timestamp from dataset CSV.
        timestamp: Timestamp converted to seconds (timestamp_ns * 1e-9).
        angular_velocity: 3-axis gyroscope reading (wx, wy, wz) in rad/s.
        linear_acceleration: 3-axis accelerometer reading (ax, ay, az) in m/s^2.
    """

    timestamp_ns: int
    timestamp: float
    angular_velocity: Tuple[float, float, float]
    linear_acceleration: Tuple[float, float, float]

    def to_imu_sample(self) -> IMUSample:
        """Convert to Phase 1 IMUSample.

        Returns:
            IMUSample: Initialized IMUSample instance.
        """
        return IMUSample(
            timestamp=self.timestamp,
            linear_acceleration=self.linear_acceleration,
            angular_velocity=self.angular_velocity,
        )


@dataclass
class DatasetSequence:
    """Metadata describing a visual-inertial dataset sequence.

    Attributes:
        sequence_name: Name or identifier of the dataset sequence.
        path: Root filesystem path of the dataset sequence.
        camera_name: Camera sensor directory name (e.g., 'cam0').
        imu_name: IMU sensor directory name (e.g., 'imu0').
        num_images: Total number of indexed camera images.
        num_imu_samples: Total number of recorded IMU samples.
    """

    sequence_name: str
    path: Path
    camera_name: str
    imu_name: str
    num_images: int
    num_imu_samples: int


@dataclass
class CameraCalibration:
    """Camera intrinsic and extrinsic calibration parameters from sensor.yaml.

    Attributes:
        T_BS: 4x4 rigid transformation matrix from camera to body frame (v_B = R_BS * v_S + p_BS).
        intrinsics: Tuple of 4 camera intrinsic parameters (fu, fv, cu, cv).
        distortion_coefficients: Tuple of lens distortion parameters (e.g. k1, k2, p1, p2).
        camera_model: Camera projection model name (default: 'pinhole').
        distortion_model: Distortion model name (default: 'radial-tangential').
        resolution: (width, height) image resolution in pixels.
        rate_hz: Nominal sensor acquisition rate in Hertz.
    """

    T_BS: np.ndarray
    intrinsics: Optional[Tuple[float, float, float, float]] = None
    distortion_coefficients: Optional[Tuple[float, ...]] = None
    camera_model: str = "pinhole"
    distortion_model: str = "radial-tangential"
    resolution: Optional[Tuple[int, int]] = None
    rate_hz: Optional[float] = None


@dataclass
class GroundTruthState:
    """Represents a ground-truth pose and state estimate from EuRoC dataset.

    Attributes:
        timestamp_ns: Nanosecond timestamp from dataset CSV.
        timestamp: Timestamp in seconds (timestamp_ns * 1e-9).
        position: 3D position (x, y, z) in meters in the reference frame.
        orientation: Unit quaternion (qw, qx, qy, qz) in scalar-first format.
        velocity: 3D linear velocity (vx, vy, vz) in m/s.
        angular_velocity: 3D body angular velocity (wx, wy, wz) in rad/s.
        linear_acceleration: 3D body linear acceleration (ax, ay, az) in m/s^2.
    """

    timestamp_ns: int
    timestamp: float
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    linear_acceleration: Tuple[float, float, float]
