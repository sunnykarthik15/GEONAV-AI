"""Unit tests for GEONAV-AI visual demonstration frontend components and utilities."""

import math
from pathlib import Path
import numpy as np
import pytest

from frontend.data_loader import (
    get_dataset,
    load_frame_with_overlay,
    load_sparse_map,
    load_trajectory_cache,
    load_verified_metrics,
    quaternion_to_euler_degrees,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_quaternion_to_euler_degrees_identity():
    """Identity quaternion [1, 0, 0, 0] should yield zero Roll, Pitch, Yaw."""
    roll, pitch, yaw = quaternion_to_euler_degrees([1.0, 0.0, 0.0, 0.0])
    assert math.isclose(roll, 0.0, abs_tol=1e-5)
    assert math.isclose(pitch, 0.0, abs_tol=1e-5)
    assert math.isclose(yaw, 0.0, abs_tol=1e-5)


def test_quaternion_to_euler_degrees_90deg_yaw():
    """90-degree yaw rotation around Z axis."""
    # q = [cos(45°), 0, 0, sin(45°)]
    val = math.sqrt(0.5)
    roll, pitch, yaw = quaternion_to_euler_degrees([val, 0.0, 0.0, val])
    assert math.isclose(roll, 0.0, abs_tol=1e-5)
    assert math.isclose(pitch, 0.0, abs_tol=1e-5)
    assert math.isclose(yaw, 90.0, abs_tol=1e-4)


def test_load_sparse_map(repo_root):
    """Verify loading precomputed Phase 5 PLY sparse map."""
    ply_path = repo_root / "artifacts" / "phase5_MH_01_easy_map.ply"
    pts, cols, count = load_sparse_map(ply_path)
    assert count == 279
    assert pts.shape == (279, 3)
    assert cols.shape == (279, 3)
    assert np.all(np.isfinite(pts))


def test_load_sparse_map_missing():
    """Missing map file should return empty arrays gracefully without crashing."""
    pts, cols, count = load_sparse_map("non_existent_map.ply")
    assert count == 0
    assert len(pts) == 0


def test_load_trajectory_cache(repo_root):
    """Verify loading demonstration trajectory cache."""
    cache_path = repo_root / "artifacts" / "demo_trajectory_cache.npz"
    cache = load_trajectory_cache(cache_path)
    assert cache is not None
    assert "timestamps" in cache
    assert "hybrid_pos" in cache
    assert "baseline_pos" in cache
    assert "delta_v" in cache
    assert "confidence" in cache
    assert "gt_pos" in cache
    assert "pos_errors" in cache
    assert len(cache["timestamps"]) == 3682
    assert cache["hybrid_pos"].shape == (3682, 3)
    assert cache["delta_v"].shape == (3682, 3)


def test_load_frame_with_overlay(repo_root):
    """Verify real image loading and feature overlay on EuRoC MH_01_easy."""
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    dataset = get_dataset(dataset_path)
    assert dataset is not None

    rgb_img, feat_count = load_frame_with_overlay(dataset, frame_idx=0, overlay_features=True)
    assert rgb_img.shape == (480, 752, 3)
    assert rgb_img.dtype == np.uint8
    assert feat_count > 0


def test_load_verified_metrics(repo_root):
    """Verify loading phase metrics."""
    metrics = load_verified_metrics(repo_root / "artifacts")
    assert "phase7_ml" in metrics
    assert "phase6_eval" in metrics
    assert "phase8_perf" in metrics
    assert "phase5_map" in metrics
