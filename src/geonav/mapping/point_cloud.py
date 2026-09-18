"""Point cloud map representation and PLY export."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from geonav.mapping.types import Landmark


class PointCloudMap:
    """Stores and manages reconstructed 3D landmarks for local mapping.

    Provides point array extraction, bounds computation, and standard PLY export.
    """

    def __init__(self, max_landmarks: int = 20000) -> None:
        """Initialize empty point cloud map.

        Args:
            max_landmarks: Maximum number of landmarks allowed in memory.
        """
        self.max_landmarks = max_landmarks
        self._landmarks: Dict[int, Landmark] = {}

    def add_landmark(self, landmark: Landmark) -> bool:
        """Add a new landmark to the map.

        Args:
            landmark: Reconstructed Landmark instance.

        Returns:
            bool: True if successfully added, False if capacity reached.
        """
        if len(self._landmarks) >= self.max_landmarks:
            return False

        self._landmarks[landmark.id] = landmark
        return True

    def update_landmark(self, landmark_id: int, timestamp: float, new_error: float) -> bool:
        """Update an existing landmark with an additional observation.

        Args:
            landmark_id: ID of the landmark being re-observed.
            timestamp: Timestamp of the new observation.
            new_error: Reprojection error of the new observation.

        Returns:
            bool: True if landmark existed and was updated, False otherwise.
        """
        if landmark_id not in self._landmarks:
            return False

        lm = self._landmarks[landmark_id]
        n = lm.observation_count
        # Running average update of reprojection error
        lm.reprojection_error = (lm.reprojection_error * n + new_error) / (n + 1)
        lm.observation_count = n + 1
        lm.last_timestamp = max(lm.last_timestamp, timestamp)
        return True

    def get_landmark(self, landmark_id: int) -> Optional[Landmark]:
        """Retrieve a specific landmark by ID."""
        return self._landmarks.get(landmark_id)

    def get_landmarks(self) -> List[Landmark]:
        """Return a list of all landmarks currently stored in the map."""
        return list(self._landmarks.values())

    def get_points(self) -> np.ndarray:
        """Extract an (N, 3) NumPy array of 3D landmark world coordinates.

        Returns:
            np.ndarray: Array of shape (N, 3), or empty (0, 3) array if map is empty.
        """
        if not self._landmarks:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([lm.position_world for lm in self._landmarks.values()], dtype=np.float64)

    def get_colors(self) -> Optional[np.ndarray]:
        """Extract an (N, 3) uint8 NumPy array of RGB colors if available.

        Returns:
            Optional[np.ndarray]: Array of shape (N, 3) uint8 or None if no colors exist.
        """
        if not self._landmarks:
            return None

        has_colors = any(lm.color is not None for lm in self._landmarks.values())
        if not has_colors:
            return None

        colors = []
        for lm in self._landmarks.values():
            if lm.color is not None:
                colors.append(lm.color)
            else:
                colors.append((180, 180, 180))  # Default neutral gray
        return np.array(colors, dtype=np.uint8)

    def count(self) -> int:
        """Return total number of landmarks in the map."""
        return len(self._landmarks)

    def __len__(self) -> int:
        """Return total number of landmarks in the map."""
        return len(self._landmarks)

    def clear(self) -> None:
        """Clear all stored landmarks."""
        self._landmarks.clear()

    def reset(self) -> None:
        """Reset map to empty state."""
        self.clear()

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Compute the 3D bounding box of the point cloud.

        Returns:
            Dict[str, Tuple[float, float]]: Min and max for 'x', 'y', and 'z'.
        """
        pts = self.get_points()
        if len(pts) == 0:
            return {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            }

        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        return {
            "x": (float(mins[0]), float(maxs[0])),
            "y": (float(mins[1]), float(maxs[1])),
            "z": (float(mins[2]), float(maxs[2])),
        }

    def export_ply(self, file_path: Union[str, Path]) -> Path:
        """Export the stored point cloud to a standard ASCII PLY file.

        The exported file is compatible with MeshLab, CloudCompare, Blender, and Open3D.

        Args:
            file_path: Destination path for the .ply file.

        Returns:
            Path: Resolved absolute path to the written PLY file.
        """
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        pts = self.get_points()
        colors = self.get_colors()
        num_vertices = len(pts)

        has_color = colors is not None and len(colors) == num_vertices

        header_lines = [
            "ply",
            "format ascii 1.0",
            "comment GEONAV-AI Phase 5 Local Sparse Point Cloud Map",
            f"element vertex {num_vertices}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_color:
            header_lines.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])

        header_lines.append("end_header\n")
        header = "\n".join(header_lines)

        with open(path, mode="w", encoding="ascii") as f:
            f.write(header)
            if has_color and colors is not None:
                for (x, y, z), (r, g, b) in zip(pts, colors):
                    f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                for x, y, z in pts:
                    f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")

        return path
