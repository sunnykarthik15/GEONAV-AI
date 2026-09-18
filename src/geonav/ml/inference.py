"""Inference engine for real-time learned velocity residual correction."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from geonav.ml.dataset import FeatureScaler
from geonav.ml.features import MLFeatures
from geonav.ml.model import NumpyEdgeMLP
from geonav.state import NavigationState


class MLVelocityCorrector:
    """Real-time inference corrector using lightweight EdgeMLP.

    Applies learned velocity corrections to classical VIO states with
    calibrated confidence gating and physical clamping limits.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        max_correction_mps: float = 3.0,
        min_confidence: float = 0.20,
    ) -> None:
        """Initialize inference corrector.

        Args:
            weights_path: Path to phase7_edge_mlp_weights.json.
            scaler_path: Path to phase7_scaler.json.
            max_correction_mps: Physical clamp limit on residual magnitude.
            min_confidence: Threshold below which correction is attenuated.
        """
        self.model = NumpyEdgeMLP()
        self.scaler = FeatureScaler()
        self.max_correction_mps = float(max_correction_mps)
        self.min_confidence = float(min_confidence)
        self.is_loaded = False

        if weights_path is not None and Path(weights_path).is_file():
            self.model.load_from_json(weights_path)
            self.is_loaded = True

        if scaler_path is not None and Path(scaler_path).is_file():
            self.scaler = FeatureScaler.load(scaler_path)

    def predict(self, features: MLFeatures) -> Tuple[np.ndarray, float]:
        """Predict velocity residual and confidence from causal features.

        Args:
            features: 18-element causal feature container.

        Returns:
            Tuple[np.ndarray, float]:
                - delta_v_gated: Gated velocity residual in m/s (shape (3,)).
                - confidence: Model confidence in [0.0, 1.0].
        """
        if not self.is_loaded:
            return np.zeros(3, dtype=np.float64), 0.0

        raw_feat = features.to_array()
        normed = self.scaler.transform(raw_feat)

        delta_v_pred, conf = self.model.forward(normed)
        delta_v = np.asarray(delta_v_pred, dtype=np.float64)

        # Check finiteness
        if not np.all(np.isfinite(delta_v)):
            return np.zeros(3, dtype=np.float64), 0.0

        # Physical clamping (e.g. max ±3.0 m/s)
        delta_v_clamped = np.clip(delta_v, -self.max_correction_mps, self.max_correction_mps)

        # Confidence gating: linear attenuation below min_confidence
        if conf <= self.min_confidence:
            attenuation = 0.0
        elif conf < (self.min_confidence + 0.3):
            attenuation = (conf - self.min_confidence) / 0.3
        else:
            attenuation = 1.0

        delta_v_gated = delta_v_clamped * attenuation
        return delta_v_gated, float(conf)

    def correct_state(
        self,
        prev_corrected_state: Optional[NavigationState],
        raw_current_state: NavigationState,
        features: MLFeatures,
        dt: float,
    ) -> NavigationState:
        """Apply velocity correction and integrate position.

        Args:
            prev_corrected_state: Previously corrected navigation state (or None at start).
            raw_current_state: Current raw state produced by classical VIO.
            features: Causal features extracted at current time step.
            dt: Inter-frame time delta in seconds.

        Returns:
            NavigationState: Corrected navigation state.
        """
        delta_v_gated, conf = self.predict(features)

        v_raw = np.array(raw_current_state.velocity, dtype=np.float64)
        v_corrected = v_raw + delta_v_gated

        if prev_corrected_state is None:
            # First frame: retain raw position
            p_corrected = np.array(raw_current_state.position, dtype=np.float64)
        else:
            # Integrate position using corrected velocity
            p_prev = np.array(prev_corrected_state.position, dtype=np.float64)
            p_corrected = p_prev + v_corrected * dt

        return NavigationState(
            timestamp=raw_current_state.timestamp,
            position=(float(p_corrected[0]), float(p_corrected[1]), float(p_corrected[2])),
            velocity=(float(v_corrected[0]), float(v_corrected[1]), float(v_corrected[2])),
            orientation=raw_current_state.orientation,
        )
