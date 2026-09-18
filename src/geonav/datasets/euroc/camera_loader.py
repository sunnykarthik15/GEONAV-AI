"""EuRoC format camera data loader."""

from collections.abc import Iterator
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import yaml

from geonav.datasets.euroc.types import CameraCalibration, DatasetImage
from geonav.vio.geometry import validate_extrinsics


class EurocCameraLoader:
    """Loads camera metadata and frame references from an EuRoC-format camera directory.

    Maintains lazy image loading so pixel arrays are only loaded into memory when requested.
    Maintains deterministic chronological ordering of frames based on timestamps.
    """

    def __init__(self, camera_dir: Union[Path, str]) -> None:
        """Initialize the camera loader and parse the camera index.

        Args:
            camera_dir: Path to the camera folder (e.g., 'mav0/cam0').

        Raises:
            FileNotFoundError: If camera_dir or data.csv does not exist.
            ValueError: If data.csv contains malformed entries.
        """
        self.camera_dir = Path(camera_dir).resolve()
        if not self.camera_dir.is_dir():
            raise FileNotFoundError(f"Camera directory not found: {self.camera_dir}")

        self.csv_path = self.camera_dir / "data.csv"
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Camera index file 'data.csv' not found in {self.camera_dir}")

        self.data_dir = self.camera_dir / "data"
        self._images: List[DatasetImage] = []
        self._parse_index()

        self.sensor_yaml_path = self.camera_dir / "sensor.yaml"
        self.calibration: Optional[CameraCalibration] = None
        self._parse_calibration()

    def _parse_index(self) -> None:
        """Parse data.csv and index images chronologically."""
        entries: List[DatasetImage] = []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue

                parts = [p.strip() for p in clean_line.split(",")]
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    raise ValueError(
                        f"Malformed camera index row at line {line_idx} in {self.csv_path}: '{clean_line}'"
                    )

                try:
                    ts_ns = int(parts[0])
                except ValueError as err:
                    raise ValueError(
                        f"Invalid timestamp at line {line_idx} in {self.csv_path}: '{parts[0]}'"
                    ) from err

                if ts_ns < 0:
                    raise ValueError(
                        f"Negative timestamp at line {line_idx} in {self.csv_path}: {ts_ns}"
                    )

                filename = parts[1]
                # Resolve file path: check cam0/data/filename first, then cam0/filename
                if self.data_dir.is_dir():
                    file_path = self.data_dir / filename
                else:
                    file_path = self.camera_dir / filename

                ts_sec = float(ts_ns) * 1e-9
                entries.append(
                    DatasetImage(
                        timestamp_ns=ts_ns,
                        timestamp=ts_sec,
                        filename=filename,
                        file_path=file_path,
                    )
                )

        # Ensure deterministic chronological ordering
        entries.sort(key=lambda img: img.timestamp_ns)
        self._images = entries

    def __len__(self) -> int:
        """Return total number of indexed frames."""
        return len(self._images)

    def __getitem__(self, index: int) -> DatasetImage:
        """Get dataset image metadata by index."""
        return self._images[index]

    def __iter__(self) -> Iterator[DatasetImage]:
        """Iterate over dataset images in chronological order."""
        return iter(self._images)

    def get_image(self, index: int) -> DatasetImage:
        """Retrieve dataset image entry at index.

        Args:
            index: Frame index.

        Returns:
            DatasetImage: Metadata for the indexed frame.
        """
        return self._images[index]

    def load_image_data(self, index: int) -> np.ndarray:
        """Lazily load pixel data of image at index.

        Args:
            index: Frame index.

        Returns:
            np.ndarray: Loaded image array.
        """
        return self._images[index].load_data()

    def _parse_calibration(self) -> None:
        """Parse camera calibration from sensor.yaml if present."""
        if not self.sensor_yaml_path.is_file():
            return

        try:
            with open(self.sensor_yaml_path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return

            T_arr = None
            if "T_BS" in data and isinstance(data["T_BS"], dict) and "data" in data["T_BS"]:
                raw_mat = data["T_BS"]["data"]
                if len(raw_mat) == 16:
                    T_arr = np.array(raw_mat, dtype=np.float64).reshape(4, 4)
                    validate_extrinsics(T_arr)

            intrinsics = None
            if "intrinsics" in data and isinstance(data["intrinsics"], (list, tuple)):
                intrinsics = tuple(float(x) for x in data["intrinsics"])

            distortion_coeffs = None
            if "distortion_coefficients" in data and isinstance(data["distortion_coefficients"], (list, tuple)):
                distortion_coeffs = tuple(float(x) for x in data["distortion_coefficients"])

            resolution = None
            if "resolution" in data and isinstance(data["resolution"], (list, tuple)) and len(data["resolution"]) == 2:
                resolution = (int(data["resolution"][0]), int(data["resolution"][1]))

            rate_hz = float(data["rate_hz"]) if "rate_hz" in data else None
            camera_model = str(data.get("camera_model", "pinhole"))
            distortion_model = str(data.get("distortion_model", "radial-tangential"))

            if T_arr is not None:
                self.calibration = CameraCalibration(
                    T_BS=T_arr,
                    intrinsics=intrinsics,
                    distortion_coefficients=distortion_coeffs,
                    camera_model=camera_model,
                    distortion_model=distortion_model,
                    resolution=resolution,
                    rate_hz=rate_hz,
                )
        except Exception as err:
            # If calibration YAML is malformed, log or raise as ValueError
            raise ValueError(f"Failed to parse camera calibration in {self.sensor_yaml_path}: {err}") from err

    @property
    def extrinsics_T_BS(self) -> Optional[np.ndarray]:
        """Return 4x4 camera-to-body extrinsic transformation matrix T_BS if loaded."""
        return self.calibration.T_BS if self.calibration else None
