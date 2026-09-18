"""Dataset preparation and normalization module for Phase 7 ML.

Prepares supervised training data offline from EuRoC MH_01_easy by pairing causal
inference-time features with ground-truth velocity residuals (v_GT - v_VIO).
Enforces strict chronological partitioning to avoid temporal data leakage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from geonav.config.settings import ExtrinsicsConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.types import TrajectoryPoint
from geonav.ml.features import FeatureExtractor
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


class FeatureScaler:
    """Z-score feature normalizer fitted strictly on training data."""

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.mean = np.zeros(18, dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = np.ones(18, dtype=np.float32) if std is None else np.asarray(std, dtype=np.float32)

    def fit(self, X: np.ndarray) -> None:
        """Fit mean and std from training feature matrix."""
        self.mean = np.mean(X, axis=0).astype(np.float32)
        s = np.std(X, axis=0).astype(np.float32)
        # Avoid division by zero
        s[s < 1e-4] = 1.0
        self.std = s

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization: (X - mean) / std."""
        X_arr = np.asarray(X, dtype=np.float32)
        normed = (X_arr - self.mean) / self.std
        np.nan_to_num(normed, copy=False, nan=0.0, posinf=5.0, neginf=-5.0)
        return normed

    def to_dict(self) -> Dict[str, list]:
        """Convert scaler to dictionary."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "FeatureScaler":
        """Load scaler from dictionary."""
        return cls(mean=np.array(d["mean"], dtype=np.float32), std=np.array(d["std"], dtype=np.float32))

    def save(self, path: Union[str, Path]) -> None:
        """Save scaler parameters to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load scaler parameters from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def extract_vio_dataset(
    dataset_path: Union[str, Path],
    split_ratio: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureScaler, Dict[str, float]]:
    """Run VIO on EuRoC sequence and extract synchronized causal feature/target pairs.

    Args:
        dataset_path: Path to EuRoC sequence (e.g. data/raw/euroc/MH_01_easy).
        split_ratio: Chronological fraction used for training (default: 0.70).

    Returns:
        Tuple containing:
            - X_train: Normalized training features (N_train, 18).
            - y_train: Training velocity residual targets (N_train, 3).
            - X_eval: Normalized evaluation features (N_eval, 18).
            - y_eval: Evaluation velocity residual targets (N_eval, 3).
            - scaler: Fitted FeatureScaler.
            - split_meta: Chronological boundary timestamps and counts.
    """
    dataset = EurocDataset(dataset_path)
    gt_pts = [
        TrajectoryPoint(
            s.timestamp,
            np.array(s.position, dtype=np.float64),
            np.array(s.orientation, dtype=np.float64),
            np.array(s.velocity, dtype=np.float64),
        )
        for s in dataset.ground_truth()
    ]

    vio_config = VIOConfig()
    if dataset.camera_extrinsics is not None:
        T_BS = dataset.camera_extrinsics
        R_BS = T_BS[:3, :3]
        t_BS = T_BS[:3, 3]
        vio_config.extrinsics = ExtrinsicsConfig(
            rotation_matrix=(
                (float(R_BS[0, 0]), float(R_BS[0, 1]), float(R_BS[0, 2])),
                (float(R_BS[1, 0]), float(R_BS[1, 1]), float(R_BS[1, 2])),
                (float(R_BS[2, 0]), float(R_BS[2, 1]), float(R_BS[2, 2])),
            ),
            translation=(float(t_BS[0]), float(t_BS[1]), float(t_BS[2])),
        )

    pipeline = VIOPipeline(config=vio_config)
    synced = list(
        synchronize_streams(
            (img.to_camera_frame() for img in dataset.images()),
            (s.to_imu_sample() for s in dataset.imu()),
        )
    )

    extractor = FeatureExtractor()
    features_list: List[np.ndarray] = []
    est_points: List[TrajectoryPoint] = []

    # Run VIO while capturing features at each step
    for m in synced:
        t_prev = pipeline._current_state.timestamp if pipeline._current_state is not None else m.camera_frame.timestamp
        t_curr = m.camera_frame.timestamp
        dt = max(t_curr - t_prev, 1e-4) if pipeline._is_initialized else 0.05

        # Process measurement in classical VIO
        state = pipeline.process_measurement(m)

        # Extract causal features from current step
        vis_res = pipeline._last_visual_result
        feat = extractor.extract(
            visual_result=vis_res,
            pts_prev=pipeline._prev_pts,
            pts_curr=None,
            imu_samples=m.imu_samples,
            current_state=state,
            dt=dt,
        )
        features_list.append(feat.to_array())

        est_points.append(
            TrajectoryPoint(
                timestamp=state.timestamp,
                position=np.array(state.position, dtype=np.float64),
                orientation=np.array(state.orientation, dtype=np.float64),
                velocity=np.array(state.velocity, dtype=np.float64),
            )
        )

    # Associate estimated states with ground truth
    pairs, _, _ = associate_timestamps(est_points, gt_pts, max_time_diff_s=0.01)

    # Align features with matched pairs
    matched_features: List[np.ndarray] = []
    matched_residuals: List[np.ndarray] = []
    timestamps: List[float] = []

    pair_idx = 0
    for i, est_pt in enumerate(est_points):
        if pair_idx < len(pairs) and pairs[pair_idx].est_point.timestamp == est_pt.timestamp:
            p = pairs[pair_idx]
            # Velocity residual target: v_GT - v_VIO
            v_err = p.gt_point.velocity - p.est_point.velocity
            matched_features.append(features_list[i])
            matched_residuals.append(v_err.astype(np.float32))
            timestamps.append(est_pt.timestamp)
            pair_idx += 1

    X = np.array(matched_features, dtype=np.float32)
    y = np.array(matched_residuals, dtype=np.float32)

    n_total = len(X)
    n_train = int(n_total * split_ratio)

    # Strict chronological partition
    X_train_raw = X[:n_train]
    y_train = y[:n_train]
    X_eval_raw = X[n_train:]
    y_eval = y[n_train:]

    # Fit scaler strictly on training split
    scaler = FeatureScaler()
    scaler.fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_eval = scaler.transform(X_eval_raw)

    split_meta = {
        "total_samples": float(n_total),
        "train_samples": float(n_train),
        "eval_samples": float(n_total - n_train),
        "split_ratio": float(split_ratio),
        "t_train_start": float(timestamps[0]),
        "t_train_end": float(timestamps[n_train - 1]),
        "t_eval_start": float(timestamps[n_train]),
        "t_eval_end": float(timestamps[-1]),
    }

    return X_train, y_train, X_eval, y_eval, scaler, split_meta
