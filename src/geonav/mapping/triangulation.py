"""Two-view triangulation and landmark validation algorithms."""

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from geonav.config.settings import MappingConfig
from geonav.mapping.types import Keyframe, TriangulationStatus


def triangulate_two_views(
    P0_norm: np.ndarray,
    P1_norm: np.ndarray,
    x0_norm: np.ndarray,
    x1_norm: np.ndarray,
) -> np.ndarray:
    """Triangulate a 3D point from normalized coordinates in two calibrated views via linear DLT.

    Args:
        P0_norm: 3x4 normalized projection matrix [R_cw0 | t_cw0] for view 0.
        P1_norm: 3x4 normalized projection matrix [R_cw1 | t_cw1] for view 1.
        x0_norm: 2-element array (x, y) of normalized coordinates in view 0.
        x1_norm: 2-element array (x, y) of normalized coordinates in view 1.

    Returns:
        np.ndarray: Triangulated 3D point in world coordinates of shape (3,).

    Raises:
        ValueError: If inputs have incorrect shapes, non-finite values, or singular geometry.
    """
    P0 = np.asarray(P0_norm, dtype=np.float64)
    P1 = np.asarray(P1_norm, dtype=np.float64)
    if P0.shape != (3, 4) or P1.shape != (3, 4):
        raise ValueError(f"Projection matrices must have shape (3, 4), got {P0.shape} and {P1.shape}")

    u0 = np.asarray(x0_norm, dtype=np.float64).ravel()
    u1 = np.asarray(x1_norm, dtype=np.float64).ravel()
    if u0.shape != (2,) or u1.shape != (2,):
        raise ValueError(f"Point coordinates must have shape (2,), got {u0.shape} and {u1.shape}")

    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1)) and np.all(np.isfinite(u0)) and np.all(np.isfinite(u1))):
        raise ValueError("Triangulation input contains non-finite values (NaN or Inf)")

    # Construct the 4x4 algebraic linear system A * X = 0
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = u0[0] * P0[2] - P0[0]
    A[1] = u0[1] * P0[2] - P0[1]
    A[2] = u1[0] * P1[2] - P1[0]
    A[3] = u1[1] * P1[2] - P1[1]

    # Solve via Singular Value Decomposition
    _, S, Vt = np.linalg.svd(A)

    # Check for rank deficiency (at least 3 non-zero singular values required for unique 3D point)
    if S[2] < 1e-4:
        raise ValueError("Singular triangulation geometry (insufficient baseline or parallel rays)")

    X_homo = Vt[-1]

    w = X_homo[3]
    if abs(w) < 1e-12 or not math.isfinite(w):
        raise ValueError("Singular triangulation geometry or point at infinity (w near zero)")

    X_3d = X_homo[:3] / w
    if not np.all(np.isfinite(X_3d)):
        raise ValueError("Triangulated coordinates contain non-finite values")

    return X_3d


def compute_parallax_angle_deg(
    C0: np.ndarray,
    C1: np.ndarray,
    X_w: np.ndarray,
) -> float:
    """Compute the optical ray parallax angle (in degrees) between two camera centers and a 3D point.

    Args:
        C0: 3D optical center of camera 0 in world frame (shape (3,)).
        C1: 3D optical center of camera 1 in world frame (shape (3,)).
        X_w: 3D landmark coordinates in world frame (shape (3,)).

    Returns:
        float: Parallax angle in degrees.
    """
    r0 = np.asarray(X_w, dtype=np.float64) - np.asarray(C0, dtype=np.float64)
    r1 = np.asarray(X_w, dtype=np.float64) - np.asarray(C1, dtype=np.float64)

    norm0 = float(np.linalg.norm(r0))
    norm1 = float(np.linalg.norm(r1))

    if norm0 < 1e-6 or norm1 < 1e-6 or not math.isfinite(norm0) or not math.isfinite(norm1):
        return 0.0

    cos_theta = float(np.dot(r0, r1) / (norm0 * norm1))
    cos_theta = min(max(cos_theta, -1.0), 1.0)
    return math.degrees(math.acos(cos_theta))


def project_world_point(
    X_w: np.ndarray,
    R_wc: np.ndarray,
    p_wc: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Project a 3D world landmark into a camera image plane using the calibrated pinhole model.

    Args:
        X_w: 3D coordinates in world frame (3,).
        R_wc: 3x3 camera-to-world rotation matrix.
        p_wc: 3D camera optical center in world frame (3,).
        K: 3x3 camera intrinsic matrix.
        dist_coeffs: Optional lens distortion coefficients (k1, k2, p1, p2).

    Returns:
        Tuple[np.ndarray, float]: (pixel_coordinates (2,), optical_depth_Z).
    """
    R_cw = R_wc.T
    t_cw = -R_cw @ p_wc

    X_cam = R_cw @ X_w + t_cw
    depth_z = float(X_cam[2])

    rvec, _ = cv2.Rodrigues(R_cw)
    dist = dist_coeffs if dist_coeffs is not None else np.zeros(4, dtype=np.float64)

    proj, _ = cv2.projectPoints(
        X_w.reshape(1, 1, 3).astype(np.float64),
        rvec,
        t_cw.reshape(3, 1).astype(np.float64),
        K.astype(np.float64),
        dist.astype(np.float64),
    )

    pixel_pt = proj.reshape(2)
    return pixel_pt, depth_z


def validate_triangulated_point(
    X_w: np.ndarray,
    kf0: Keyframe,
    kf1: Keyframe,
    u0_px: np.ndarray,
    u1_px: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    config: MappingConfig,
) -> Tuple[TriangulationStatus, float]:
    """Validate a triangulated 3D landmark against cheirality, depth, parallax, and reprojection bounds.

    Args:
        X_w: Triangulated 3D position in world frame (3,).
        kf0: Reference keyframe 0.
        kf1: Reference keyframe 1.
        u0_px: Observed pixel coordinates in view 0 (2,).
        u1_px: Observed pixel coordinates in view 1 (2,).
        K: 3x3 camera intrinsic matrix.
        dist_coeffs: Optional distortion coefficients.
        config: Mapping configuration parameters.

    Returns:
        Tuple[TriangulationStatus, float]: (status, mean_reprojection_error_px).
    """
    # 1. Finite coordinates check
    if not np.all(np.isfinite(X_w)):
        return TriangulationStatus.REJECTED_NUMERICAL, 0.0

    # 2. Parallax baseline check
    parallax = compute_parallax_angle_deg(kf0.p_wc, kf1.p_wc, X_w)
    if parallax < config.min_triangulation_angle_deg:
        return TriangulationStatus.REJECTED_PARALLAX, 0.0

    # 3. Projection and depth check in both views
    try:
        p0_proj, z0 = project_world_point(X_w, kf0.R_wc, kf0.p_wc, K, dist_coeffs)
        p1_proj, z1 = project_world_point(X_w, kf1.R_wc, kf1.p_wc, K, dist_coeffs)
    except Exception:
        return TriangulationStatus.REJECTED_NUMERICAL, 0.0

    # Cheirality check (positive depth)
    if z0 <= 0.0 or z1 <= 0.0:
        return TriangulationStatus.REJECTED_NEGATIVE_DEPTH, 0.0

    # Reasonable depth range check
    if z0 < config.min_depth_m or z0 > config.max_depth_m or z1 < config.min_depth_m or z1 > config.max_depth_m:
        return TriangulationStatus.REJECTED_DEPTH_RANGE, 0.0

    # 4. Reprojection error check
    e0 = float(np.linalg.norm(p0_proj - u0_px))
    e1 = float(np.linalg.norm(p1_proj - u1_px))
    mean_err = 0.5 * (e0 + e1)

    if not (math.isfinite(e0) and math.isfinite(e1)):
        return TriangulationStatus.REJECTED_NUMERICAL, 0.0

    if e0 > config.max_reprojection_error_px or e1 > config.max_reprojection_error_px:
        return TriangulationStatus.REJECTED_REPROJECTION, mean_err

    return TriangulationStatus.SUCCESS, mean_err
