"""Closed-form Umeyama SE(3) and Sim(3) trajectory alignment."""

from typing import Tuple
import numpy as np


def umeyama_alignment(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    with_scale: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute optimal closed-form least-squares SE(3) or Sim(3) alignment via Umeyama (1991).

    Minimizes:
        sum_i || y_i - (s * R * x_i + t) ||^2

    Args:
        pts_src: Source point coordinates of shape (N, 3) (estimated trajectory).
        pts_dst: Destination point coordinates of shape (N, 3) (ground-truth trajectory).
        with_scale: If True, solves Sim(3) with scale s; if False, solves rigid SE(3) (s = 1.0).

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            - s: Scalar scale factor (float).
            - R: 3x3 orthonormal rotation matrix (shape (3, 3)).
            - t: 3D translation vector (shape (3,)).

    Raises:
        ValueError: If input arrays have incorrect shapes, fewer than 3 points, or non-finite values.
    """
    src = np.asarray(pts_src, dtype=np.float64)
    dst = np.asarray(pts_dst, dtype=np.float64)

    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: {src.shape} vs {dst.shape}")
    if len(src.shape) != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) point arrays, got {src.shape}")
    if src.shape[0] < 3:
        raise ValueError(f"At least 3 corresponding points required for alignment, got {src.shape[0]}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("Point arrays contain non-finite values (NaN or Inf)")

    n, m = src.shape

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = float(np.mean(np.sum(src_c**2, axis=1)))
    if var_src < 1e-12:
        raise ValueError("Degenerate source points: spatial variance is near zero")

    # Cross-covariance matrix
    cov = (dst_c.T @ src_c) / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)

    # Disambiguate reflection to guarantee det(R) = +1
    S = np.eye(m, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    if with_scale:
        scale = float((1.0 / var_src) * np.sum(D * np.diag(S)))
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(f"Invalid non-positive scale computed: {scale}")
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def align_trajectory_points(
    points: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply similarity transformation (s * R * p + t) to an array of 3D points.

    Args:
        points: Array of 3D points of shape (N, 3).
        scale: Positive scalar scale factor.
        R: 3x3 orthonormal rotation matrix.
        t: 3D translation vector (3,).

    Returns:
        np.ndarray: Aligned points of shape (N, 3).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return (scale * (R @ pts.T)).T + t
