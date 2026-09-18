"""EuRoC format IMU data loader."""

from collections.abc import Iterator
import math
from pathlib import Path
from typing import List, Union

from geonav.datasets.euroc.types import DatasetIMUSample


class EurocIMULoader:
    """Loads raw IMU measurements from an EuRoC-format IMU directory or CSV file.

    Preserves deterministic chronological ordering and validates data integrity.
    """

    def __init__(self, imu_source: Union[Path, str]) -> None:
        """Initialize the IMU loader and parse IMU measurements.

        Args:
            imu_source: Path to the IMU folder (e.g., 'mav0/imu0') or directly to 'data.csv'.

        Raises:
            FileNotFoundError: If the directory or data.csv cannot be found.
            ValueError: If malformed rows or invalid numerical values are encountered.
        """
        source_path = Path(imu_source).resolve()
        if source_path.is_file():
            self.csv_path = source_path
            self.imu_dir = source_path.parent
        elif source_path.is_dir():
            self.imu_dir = source_path
            self.csv_path = source_path / "data.csv"
            if not self.csv_path.is_file():
                raise FileNotFoundError(f"IMU index file 'data.csv' not found in {self.imu_dir}")
        else:
            raise FileNotFoundError(f"IMU source path not found: {source_path}")

        self._samples: List[DatasetIMUSample] = []
        self._parse_csv()

    def _parse_csv(self) -> None:
        """Parse IMU CSV file rows: timestamp [ns], w_x, w_y, w_z, a_x, a_y, a_z."""
        samples: List[DatasetIMUSample] = []

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue

                parts = [p.strip() for p in clean_line.split(",")]
                if len(parts) != 7:
                    raise ValueError(
                        f"Malformed IMU row at line {line_idx} in {self.csv_path}: "
                        f"expected 7 values, found {len(parts)} ('{clean_line}')"
                    )

                # Parse timestamp
                try:
                    ts_ns = int(parts[0])
                except ValueError as err:
                    raise ValueError(
                        f"Invalid timestamp '{parts[0]}' at line {line_idx} in {self.csv_path}"
                    ) from err

                if ts_ns < 0:
                    raise ValueError(
                        f"Negative timestamp at line {line_idx} in {self.csv_path}: {ts_ns}"
                    )

                # Parse float values for gyroscope (w_x, w_y, w_z) and accelerometer (a_x, a_y, a_z)
                parsed_vals: List[float] = []
                for col_idx, p in enumerate(parts[1:], start=1):
                    try:
                        val = float(p)
                    except ValueError as err:
                        raise ValueError(
                            f"Non-numeric value '{p}' in column {col_idx} at line {line_idx} in {self.csv_path}"
                        ) from err

                    if math.isnan(val) or math.isinf(val):
                        raise ValueError(
                            f"Non-finite numerical value '{p}' in column {col_idx} at line {line_idx} in {self.csv_path}"
                        )
                    parsed_vals.append(val)

                angular_velocity = (parsed_vals[0], parsed_vals[1], parsed_vals[2])
                linear_acceleration = (parsed_vals[3], parsed_vals[4], parsed_vals[5])
                ts_sec = float(ts_ns) * 1e-9

                samples.append(
                    DatasetIMUSample(
                        timestamp_ns=ts_ns,
                        timestamp=ts_sec,
                        angular_velocity=angular_velocity,
                        linear_acceleration=linear_acceleration,
                    )
                )

        # Preserve deterministic chronological ordering
        samples.sort(key=lambda s: s.timestamp_ns)
        self._samples = samples

    def __len__(self) -> int:
        """Return total number of IMU samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> DatasetIMUSample:
        """Retrieve IMU sample by index."""
        return self._samples[index]

    def __iter__(self) -> Iterator[DatasetIMUSample]:
        """Iterate through IMU samples in chronological order."""
        return iter(self._samples)
