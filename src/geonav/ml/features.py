"""Feature extraction module for GEONAV-AI Machine Learning subsystem.

Extracts strictly causal, inference-time features from visual tracking,
IMU measurements, and current classical VIO state. No future frames or ground truth are used.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import numpy as np

from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.state import NavigationState
from geonav.vio.types import VisualTrackingResult


@dataclass
class MLFeatures:
    """Container for the 18-dimensional causal feature vector."""

    tracked_features: float
    inlier_ratio: float
    mean_flow_mag: float
    std_flow_mag: float
    mean_accel: Tuple[float, float, float]
    var_accel: Tuple[float, float, float]
    mean_gyro: Tuple[float, float, float]
    vio_velocity: Tuple[float, float, float]
    vio_v_norm: float
    dt: float

    def to_array(self) -> np.ndarray:
        """Convert features to a 1D NumPy array of shape (18,)."""
        arr = np.array(
            [
                float(self.tracked_features),
                float(self.inlier_ratio),
                float(self.mean_flow_mag),
                float(self.std_flow_mag),
                float(self.mean_accel[0]),
                float(self.mean_accel[1]),
                float(self.mean_accel[2]),
                float(self.var_accel[0]),
                float(self.var_accel[1]),
                float(self.var_accel[2]),
                float(self.mean_gyro[0]),
                float(self.mean_gyro[1]),
                float(self.mean_gyro[2]),
                float(self.vio_velocity[0]),
                float(self.vio_velocity[1]),
                float(self.vio_velocity[2]),
                float(self.vio_v_norm),
                float(self.dt),
            ],
            dtype=np.float32,
        )
        # Scrub any non-finite entries
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
        return arr


class FeatureExtractor:
    """Causal inference-time feature extractor for GEONAV-AI."""

    def __init__(self) -> None:
        """Initialize feature extractor."""
        self.feature_dim = 18

    def extract(
        self,
        visual_result: Optional[VisualTrackingResult],
        pts_prev: Optional[np.ndarray],
        pts_curr: Optional[np.ndarray],
        imu_samples: Optional[Sequence[IMUSample]],
        current_state: NavigationState,
        dt: float,
    ) -> MLFeatures:
        """Extract inference features at current time step.

        Args:
            visual_result: Tracking and motion result from current visual frontend step.
            pts_prev: Previous matched keypoints (N, 2).
            pts_curr: Current matched keypoints (N, 2).
            imu_samples: IMU samples accumulated across the current inter-frame step.
            current_state: Latest navigation state from classical VIO.
            dt: Inter-frame time delta in seconds.

        Returns:
            MLFeatures: Clean 18-element feature container.
        """
        # 1. Visual Tracking Features
        tracked_count = 0.0
        inlier_ratio = 0.0
        mean_flow = 0.0
        std_flow = 0.0

        if visual_result is not None and visual_result.success:
            tracked_count = float(visual_result.num_tracked)
            if visual_result.num_tracked > 0:
                inlier_ratio = float(visual_result.num_inliers) / float(visual_result.num_tracked)

        if (
            pts_prev is not None
            and pts_curr is not None
            and len(pts_prev) > 0
            and len(pts_prev) == len(pts_curr)
        ):
            diffs = pts_curr - pts_prev
            flow_mags = np.linalg.norm(diffs, axis=1)
            if len(flow_mags) > 0:
                mean_flow = float(np.mean(flow_mags))
                std_flow = float(np.std(flow_mags))

        # 2. IMU Kinematic Features
        mean_accel = (0.0, 0.0, 0.0)
        var_accel = (0.0, 0.0, 0.0)
        mean_gyro = (0.0, 0.0, 0.0)

        if imu_samples is not None and len(imu_samples) > 0:
            accels = np.array([s.linear_acceleration for s in imu_samples], dtype=np.float64)
            gyros = np.array([s.angular_velocity for s in imu_samples], dtype=np.float64)

            # Check finiteness
            if np.all(np.isfinite(accels)):
                m_a = np.mean(accels, axis=0)
                v_a = np.var(accels, axis=0)
                mean_accel = (float(m_a[0]), float(m_a[1]), float(m_a[2]))
                var_accel = (float(v_a[0]), float(v_a[1]), float(v_a[2]))

            if np.all(np.isfinite(gyros)):
                m_g = np.mean(gyros, axis=0)
                mean_gyro = (float(m_g[0]), float(m_g[1]), float(m_g[2]))

        # 3. VIO Navigation State Features
        v = np.asarray(current_state.velocity, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            v = np.zeros(3, dtype=np.float64)

        vio_vel = (float(v[0]), float(v[1]), float(v[2]))
        v_norm = float(np.linalg.norm(v))

        # Sanitize dt
        safe_dt = float(np.clip(dt, 1e-4, 1.0))

        return MLFeatures(
            tracked_features=tracked_count,
            inlier_ratio=inlier_ratio,
            mean_flow_mag=mean_flow,
            std_flow_mag=std_flow,
            mean_accel=mean_accel,
            var_accel=var_accel,
            mean_gyro=mean_gyro,
            vio_velocity=vio_vel,
            vio_v_norm=v_norm,
            dt=safe_dt,
        )
