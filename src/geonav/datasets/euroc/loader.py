"""High-level EuRoC visual-inertial dataset interface."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union
import numpy as np

from geonav.datasets.euroc.camera_loader import EurocCameraLoader
from geonav.datasets.euroc.imu_loader import EurocIMULoader
from geonav.datasets.euroc.types import (
    CameraCalibration,
    DatasetImage,
    DatasetIMUSample,
    DatasetSequence,
)


class EurocDataset:
    """High-level reader for an EuRoC-format visual-inertial dataset sequence.

    Automatically resolves both standard 'dataset/mav0/{cam0,imu0}' structure
    and direct sequence-root 'dataset/{cam0,imu0}' structure.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        camera_name: str = "cam0",
        imu_name: str = "imu0",
    ) -> None:
        """Initialize EuRoC dataset reader.

        Args:
            dataset_path: Path to dataset sequence root directory.
            camera_name: Name of camera directory (default: 'cam0').
            imu_name: Name of IMU directory (default: 'imu0').

        Raises:
            FileNotFoundError: If dataset path or required sensor folders are missing.
        """
        self.root_path = Path(dataset_path).resolve()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {self.root_path}")

        self.camera_name = camera_name
        self.imu_name = imu_name

        # Resolve sensor root: check for mav0 subdirectory first, then direct root
        mav0_dir = self.root_path / "mav0"
        if (mav0_dir / self.camera_name).is_dir() and (mav0_dir / self.imu_name).is_dir():
            sensor_root = mav0_dir
        elif (self.root_path / self.camera_name).is_dir() and (self.root_path / self.imu_name).is_dir():
            sensor_root = self.root_path
        else:
            # Provide specific error message pointing out the missing sensor folder
            cam_check_mav0 = mav0_dir / self.camera_name
            cam_check_root = self.root_path / self.camera_name
            if not cam_check_mav0.is_dir() and not cam_check_root.is_dir():
                raise FileNotFoundError(
                    f"Camera folder '{self.camera_name}' not found under {self.root_path} or {mav0_dir}"
                )
            raise FileNotFoundError(
                f"IMU folder '{self.imu_name}' not found under {self.root_path} or {mav0_dir}"
            )

        self.sensor_root = sensor_root
        self.camera = EurocCameraLoader(sensor_root / self.camera_name)
        self.imu_loader = EurocIMULoader(sensor_root / self.imu_name)

        sequence_name = self.root_path.name
        self.metadata = DatasetSequence(
            sequence_name=sequence_name,
            path=self.root_path,
            camera_name=self.camera_name,
            imu_name=self.imu_name,
            num_images=len(self.camera),
            num_imu_samples=len(self.imu_loader),
        )

    def images(self) -> Iterator[DatasetImage]:
        """Iterate over dataset camera images in chronological order."""
        return iter(self.camera)

    def imu(self) -> Iterator[DatasetIMUSample]:
        """Iterate over dataset IMU samples in chronological order."""
        return iter(self.imu_loader)

    @property
    def num_images(self) -> int:
        """Total number of indexed images."""
        return len(self.camera)

    @property
    def num_imu_samples(self) -> int:
        """Total number of recorded IMU samples."""
        return len(self.imu_loader)

    @property
    def camera_extrinsics(self) -> Optional[np.ndarray]:
        """Return 4x4 camera-to-body extrinsic transformation matrix T_BS if loaded."""
        return self.camera.extrinsics_T_BS

    @property
    def camera_calibration(self) -> Optional[CameraCalibration]:
        """Return full parsed camera calibration if available."""
        return self.camera.calibration
