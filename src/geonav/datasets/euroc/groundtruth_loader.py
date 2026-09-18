"""EuRoC format ground-truth state loader."""

from collections.abc import Iterator
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from geonav.datasets.euroc.types import GroundTruthState


class EurocGroundTruthLoader:
    """Loads and validates ground-truth pose and state estimates from EuRoC dataset files."""

    def __init__(self, groundtruth_path: Union[Path, str]) -> None:
        """Initialize the ground-truth loader and parse data.csv.

        Args:
            groundtruth_path: Path to data.csv or folder containing data.csv (e.g. state_groundtruth_estimate0).

        Raises:
            FileNotFoundError: If ground-truth file or directory does not exist.
            ValueError: If data.csv contains malformed entries, non-increasing timestamps, or non-finite values.
        """
        p = Path(groundtruth_path).resolve()
        if p.is_dir():
            csv_path = p / "data.csv"
        else:
            csv_path = p

        if not csv_path.is_file():
            raise FileNotFoundError(f"Ground-truth file not found: {csv_path}")

        self.csv_path = csv_path
        self._states: List[GroundTruthState] = []
        self._parse()

    def _parse(self) -> None:
        """Parse ground-truth CSV and index states in chronological order."""
        entries: List[GroundTruthState] = []
        last_ts_ns = -1

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue

                parts = [p.strip() for p in clean_line.split(",")]
                if len(parts) < 11:
                    raise ValueError(
                        f"Malformed ground-truth row at line {line_idx} in {self.csv_path}: "
                        f"expected at least 11 columns, got {len(parts)}"
                    )

                try:
                    ts_ns = int(parts[0])
                    px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
                    qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    vx, vy, vz = float(parts[8]), float(parts[9]), float(parts[10])

                    wx = float(parts[11]) if len(parts) > 11 else 0.0
                    wy = float(parts[12]) if len(parts) > 12 else 0.0
                    wz = float(parts[13]) if len(parts) > 13 else 0.0

                    ax = float(parts[14]) if len(parts) > 14 else 0.0
                    ay = float(parts[15]) if len(parts) > 15 else 0.0
                    az = float(parts[16]) if len(parts) > 16 else 0.0
                except (ValueError, IndexError) as err:
                    raise ValueError(
                        f"Non-numeric value at line {line_idx} in {self.csv_path}: '{clean_line}'"
                    ) from err

                if ts_ns < 0:
                    raise ValueError(f"Negative timestamp at line {line_idx}: {ts_ns}")

                if ts_ns < last_ts_ns:
                    raise ValueError(
                        f"Out-of-order timestamp at line {line_idx}: {ts_ns} < {last_ts_ns}"
                    )
                last_ts_ns = ts_ns

                # Finiteness check
                vals = [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az]
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError(f"Non-finite value detected at line {line_idx}")

                # Normalize quaternion
                q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                if q_norm < 1e-6:
                    raise ValueError(f"Near-zero quaternion norm at line {line_idx}: {q_norm}")
                qw /= q_norm
                qx /= q_norm
                qy /= q_norm
                qz /= q_norm

                # Canonical representation (scalar part non-negative)
                if qw < 0.0:
                    qw, qx, qy, qz = -qw, -qx, -qy, -qz

                ts_sec = float(ts_ns) * 1e-9
                entries.append(
                    GroundTruthState(
                        timestamp_ns=ts_ns,
                        timestamp=ts_sec,
                        position=(px, py, pz),
                        orientation=(qw, qx, qy, qz),
                        velocity=(vx, vy, vz),
                        angular_velocity=(wx, wy, wz),
                        linear_acceleration=(ax, ay, az),
                    )
                )

        self._states = entries

    def __len__(self) -> int:
        """Return total number of ground-truth states."""
        return len(self._states)

    def __getitem__(self, index: int) -> GroundTruthState:
        """Retrieve state by index."""
        return self._states[index]

    def __iter__(self) -> Iterator[GroundTruthState]:
        """Iterate over ground-truth states in chronological order."""
        return iter(self._states)

    def get_timestamps(self) -> np.ndarray:
        """Return 1D array of timestamps in seconds."""
        return np.array([s.timestamp for s in self._states], dtype=np.float64)

    def get_positions(self) -> np.ndarray:
        """Return (N, 3) array of 3D positions in meters."""
        return np.array([s.position for s in self._states], dtype=np.float64)

    def get_orientations(self) -> np.ndarray:
        """Return (N, 4) array of unit quaternions [qw, qx, qy, qz]."""
        return np.array([s.orientation for s in self._states], dtype=np.float64)

    def get_velocities(self) -> np.ndarray:
        """Return (N, 3) array of linear velocities in m/s."""
        return np.array([s.velocity for s in self._states], dtype=np.float64)
