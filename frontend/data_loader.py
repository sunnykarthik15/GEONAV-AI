"""Data loading and caching utilities for the GEONAV-AI Streamlit frontend."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
import streamlit as st

# Ensure src/ is on sys.path robustly
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geonav.datasets.euroc import EurocDataset
from geonav.vio.visual_frontend import VisualFrontEnd


@st.cache_resource(show_spinner=False)
def get_dataset(dataset_path: Union[str, Path]) -> Optional[EurocDataset]:
    """Load and cache EuRoC dataset reader.

    Returns None if dataset path does not exist.
    """
    p = Path(dataset_path).resolve()
    if not p.is_dir():
        return None
    try:
        return EurocDataset(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_trajectory_cache(cache_path: Union[str, Path]) -> Optional[Dict[str, np.ndarray]]:
    """Load precomputed demonstration trajectory cache (.npz).

    Returns a dictionary of numpy arrays or None if file not found.
    """
    p = Path(cache_path).resolve()
    if not p.is_file():
        return None
    try:
        with np.load(p, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_sparse_map(ply_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse ASCII PLY point cloud file for 3D sparse map visualization.

    Returns:
        Tuple:
            - points: np.ndarray of shape (N, 3) with [X, Y, Z]
            - colors: np.ndarray of shape (N, 3) with RGB values in [0, 255]
            - count: Total number of landmarks
    """
    p = Path(ply_path).resolve()
    if not p.is_file():
        return np.empty((0, 3)), np.empty((0, 3)), 0

    points = []
    colors = []
    in_header = True

    try:
        with open(p, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if in_header:
                    if line == "end_header":
                        in_header = False
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                        if len(parts) >= 6:
                            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                            colors.append([r, g, b])
                        else:
                            colors.append([120, 120, 120])
                    except (ValueError, IndexError):
                        continue

        if points:
            pts_arr = np.asarray(points, dtype=np.float64)
            col_arr = np.asarray(colors, dtype=np.uint8)
            return pts_arr, col_arr, len(pts_arr)
        return np.empty((0, 3)), np.empty((0, 3)), 0
    except Exception:
        return np.empty((0, 3)), np.empty((0, 3)), 0


@st.cache_data(show_spinner=False)
def load_verified_metrics(artifacts_dir: Union[str, Path]) -> Dict[str, dict]:
    """Load verified quantitative evaluation metrics JSON files."""
    ad = Path(artifacts_dir).resolve()
    results = {}

    ml_path = ad / "phase7_ml_metrics.json"
    if ml_path.is_file():
        try:
            with open(ml_path, mode="r", encoding="utf-8") as f:
                results["phase7_ml"] = json.load(f)
        except Exception:
            pass

    eval_path = ad / "phase6_evaluation_metrics.json"
    if eval_path.is_file():
        try:
            with open(eval_path, mode="r", encoding="utf-8") as f:
                results["phase6_eval"] = json.load(f)
        except Exception:
            pass

    perf_path = ad / "phase8_performance_metrics.json"
    if perf_path.is_file():
        try:
            with open(perf_path, mode="r", encoding="utf-8") as f:
                results["phase8_perf"] = json.load(f)
        except Exception:
            pass

    map_path = ad / "phase5_mapping_statistics.json"
    if map_path.is_file():
        try:
            with open(map_path, mode="r", encoding="utf-8") as f:
                results["phase5_map"] = json.load(f)
        except Exception:
            pass

    return results


def quaternion_to_euler_degrees(q: Sequence[float]) -> Tuple[float, float, float]:
    """Convert Hamilton unit quaternion [w, x, y, z] to Euler angles in degrees.

    Convention: Z-Y-X Tait-Bryan (Roll, Pitch, Yaw) for NED navigation frame.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def load_frame_with_overlay(
    dataset: EurocDataset,
    frame_idx: int,
    overlay_features: bool = True,
    max_features: int = 150,
) -> Tuple[np.ndarray, int]:
    """Load real EuRoC camera frame and optionally overlay detected Shi-Tomasi feature points.

    Returns:
        Tuple:
            - RGB image ready for display (np.ndarray uint8 of shape (H, W, 3))
            - Number of detected/displayed feature points
    """
    raw_gray = dataset.camera.load_image_data(frame_idx)
    rgb_img = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2RGB)

    feature_count = 0
    if overlay_features:
        # Detect corner features on the real camera image using OpenCV Shi-Tomasi
        pts = cv2.goodFeaturesToTrack(
            raw_gray,
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=10.0,
            blockSize=3,
        )
        if pts is not None:
            feature_count = len(pts)
            for p in pts:
                x, y = int(p[0, 0]), int(p[0, 1])
                cv2.circle(rgb_img, (x, y), 3, (0, 255, 64), -1)
                cv2.circle(rgb_img, (x, y), 4, (0, 128, 32), 1)

    return rgb_img, feature_count
