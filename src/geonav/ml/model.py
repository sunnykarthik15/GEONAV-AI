"""Compact Edge MLP model architecture for learned velocity residual and confidence estimation."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


if TORCH_AVAILABLE:
    class EdgeMLP(nn.Module):
        """Lightweight Edge Multi-Layer Perceptron in PyTorch.

        Architecture:
            Input: 18 features
            Hidden 1: Linear(18, 32) + ReLU
            Hidden 2: Linear(32, 16) + ReLU
            Residual Head: Linear(16, 3) -> delta_v [m/s]
            Confidence Head: Linear(16, 1) + Sigmoid -> confidence [0, 1]
        """

        def __init__(self, input_dim: int = 18) -> None:
            super().__init__()
            self.input_dim = input_dim

            self.shared = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            self.residual_head = nn.Linear(16, 3)
            self.confidence_head = nn.Sequential(
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning (delta_v, confidence)."""
            feat = self.shared(x)
            delta_v = self.residual_head(feat)
            confidence = self.confidence_head(feat)
            return delta_v, confidence

        def export_weights_dict(self) -> Dict[str, list]:
            """Export weights to a plain dictionary for standalone NumPy inference."""
            sd = self.state_dict()
            return {k: v.cpu().numpy().tolist() for k, v in sd.items()}

        def load_weights_dict(self, d: Dict[str, list]) -> None:
            """Load weights from a dictionary."""
            sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}
            self.load_state_dict(sd)

else:
    class EdgeMLP:  # type: ignore
        """Placeholder when PyTorch is not installed."""
        def __init__(self, input_dim: int = 18) -> None:
            self.input_dim = input_dim


class NumpyEdgeMLP:
    """Pure NumPy inference engine for EdgeMLP.

    Enables zero-dependency, sub-millisecond edge execution without PyTorch runtime overhead.
    """

    def __init__(self, weights: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Initialize NumPy inference engine with weight matrices."""
        self.weights: Dict[str, np.ndarray] = {}
        if weights is not None:
            self.set_weights(weights)

    def set_weights(self, weights: Dict[str, Union[np.ndarray, list]]) -> None:
        """Set weights dictionary."""
        self.weights = {k: np.array(v, dtype=np.float32) for k, v in weights.items()}

    def load_from_json(self, path: Union[str, Path]) -> None:
        """Load weights from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.set_weights(d)

    def save_to_json(self, path: Union[str, Path]) -> None:
        """Save weights to JSON file."""
        serializable = {k: v.tolist() for k, v in self.weights.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform forward pass on a 1D feature array of shape (18,).

        Returns:
            Tuple[np.ndarray, float]:
                - delta_v: Predicted velocity residual (3,) in m/s.
                - confidence: Predicted confidence scalar in [0.0, 1.0].
        """
        if not self.weights:
            # Fallback if no weights loaded
            return np.zeros(3, dtype=np.float32), 0.0

        x_in = np.asarray(x, dtype=np.float32).reshape(-1)

        # Shared layer 1
        w1 = self.weights["shared.0.weight"]  # shape (32, 18)
        b1 = self.weights["shared.0.bias"]    # shape (32,)
        h1 = np.maximum(0.0, w1 @ x_in + b1)  # ReLU

        # Shared layer 2
        w2 = self.weights["shared.2.weight"]  # shape (16, 32)
        b2 = self.weights["shared.2.bias"]    # shape (16,)
        h2 = np.maximum(0.0, w2 @ h1 + b2)    # ReLU

        # Residual head
        w_res = self.weights["residual_head.weight"]  # shape (3, 16)
        b_res = self.weights["residual_head.bias"]    # shape (3,)
        delta_v = w_res @ h2 + b_res

        # Confidence head
        w_conf = self.weights["confidence_head.0.weight"]  # shape (1, 16)
        b_conf = self.weights["confidence_head.0.bias"]    # shape (1,)
        logit = float((w_conf @ h2 + b_conf)[0])
        # Numerically stable sigmoid
        conf = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))

        return delta_v, float(conf)
