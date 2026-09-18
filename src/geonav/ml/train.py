"""Training script for GEONAV-AI Phase 7 EdgeMLP."""

import os
from pathlib import Path
import random
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from geonav.ml.dataset import FeatureScaler, extract_vio_dataset
from geonav.ml.model import EdgeMLP, NumpyEdgeMLP


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_edge_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_conf: float = 0.05,
) -> Tuple[EdgeMLP, Dict[str, list]]:
    """Train EdgeMLP on training features and evaluate on chronological evaluation split.

    Args:
        X_train: (N_train, 18) normalized training features.
        y_train: (N_train, 3) velocity residual targets.
        X_eval: (N_eval, 18) normalized evaluation features.
        y_eval: (N_eval, 3) velocity residual targets.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        weight_decay: L2 regularization weight.
        lambda_conf: Weight for confidence regularization.

    Returns:
        Tuple[EdgeMLP, Dict[str, list]]: Trained model and training history dictionary.
    """
    set_seed(42)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_eval, dtype=torch.float32)
    y_val_t = torch.tensor(y_eval, dtype=torch.float32)

    model = EdgeMLP(input_dim=18)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }

    print(f"Starting training EdgeMLP ({len(X_train)} train, {len(X_eval)} eval samples, {epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            delta_v_pred, conf_pred = model(bx)

            # Combined loss: MSE on velocity residual + regularization encouraging confidence
            loss_res = mse_loss(delta_v_pred, by)
            loss_conf = torch.mean((1.0 - conf_pred) ** 2)
            total_loss = loss_res + lambda_conf * loss_conf

            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            v_pred_val, _ = model(X_val_t)
            val_loss = float(mse_loss(v_pred_val, y_val_t).item())
            val_mae = float(torch.mean(torch.abs(v_pred_val - y_val_t)).item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

    return model, history


def main() -> None:
    """Run full dataset extraction and model training pipeline."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"

    print("=" * 70)
    print("GEONAV-AI — PHASE 7 ML VELOCITY CORRECTION MODEL TRAINING")
    print("=" * 70)

    if not dataset_path.is_dir():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Extract dataset
    print(f"Extracting VIO features and targets from {dataset_path}...")
    X_train, y_train, X_eval, y_eval, scaler, meta = extract_vio_dataset(dataset_path, split_ratio=0.70)
    print(f"Extracted {int(meta['total_samples'])} matched samples.")
    print(f"  Train: {int(meta['train_samples'])} samples ({meta['t_train_start']:.2f}s to {meta['t_train_end']:.2f}s)")
    print(f"  Eval:  {int(meta['eval_samples'])} samples ({meta['t_eval_start']:.2f}s to {meta['t_eval_end']:.2f}s)")

    # Train model
    model, history = train_edge_mlp(X_train, y_train, X_eval, y_eval, epochs=40, batch_size=32)

    # Export artifacts
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    weights_json_path = artifacts_dir / "phase7_edge_mlp_weights.json"
    weights_dict = model.export_weights_dict()
    numpy_model = NumpyEdgeMLP(weights_dict)
    numpy_model.save_to_json(weights_json_path)
    print(f"Exported NumPy-compatible model weights to: {weights_json_path}")

    scaler_path = artifacts_dir / "phase7_scaler.json"
    scaler.save(scaler_path)
    print(f"Exported feature scaler parameters to: {scaler_path}")

    torch_model_path = artifacts_dir / "phase7_edge_mlp.pt"
    torch.save(model.state_dict(), torch_model_path)
    print(f"Exported PyTorch model checkpoint to: {torch_model_path}")


if __name__ == "__main__":
    main()
