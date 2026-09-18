"""Deterministic synthetic and numerical verification tests for Phase 4B geometry gate.

Tests correspond to Phase 4B Verification Gate requirements:
- Section 4: Synthetic rotation tests (Test A Identity, Test B Known rotations, Test C Quaternion consistency)
- Section 5: Synthetic translation-direction test (+X, +Y, +Z)
- Section 6: Forward and inverse transformation consistency
- Section 7: Rotation matrix orthonormality and determinant verification
"""

import math
from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from geonav.vio.geometry import (
    invert_transform,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_direction_camera_to_body,
    transform_point_body_to_camera,
    transform_point_camera_to_body,
    transform_relative_rotation_camera_to_body,
    validate_extrinsics,
    validate_rotation_matrix,
)

# Reference EuRoC MH_01_easy camera-to-body extrinsic calibration T_BS
EUROC_T_BS = np.array([
    [0.0148655429818, -0.9998809296980,  0.00414029679422, -0.0216401454975],
    [0.9995572490080,  0.0149672133247,  0.02571552994800, -0.0646769867680],
    [-0.0257744366974,  0.00375618835797,  0.99966072717800,  0.00981073058949],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def test_synthetic_rotation_test_a_identity():
    """Section 4 - Test A (Identity): If R_c_rel = I then R_b_rel = I."""
    R_bc = EUROC_T_BS[:3, :3]
    R_c_ident = np.eye(3, dtype=np.float64)

    R_b_rel = transform_relative_rotation_camera_to_body(R_c_ident, R_bc)

    # Must be numerically identical to identity matrix
    assert np.allclose(R_b_rel, np.eye(3), atol=1e-12)
    assert np.max(np.abs(R_b_rel - np.eye(3))) < 1e-12


def test_synthetic_rotation_test_b_known_rotations():
    """Section 4 - Test B (Known rotation): 90-deg rotations around X, Y, Z axes."""
    # Simplified orthogonal camera-to-body rotation:
    # Camera X (Right) -> Body Y (Right)
    # Camera Y (Down) -> Body -X (-Up = Down)
    # Camera Z (Forward) -> Body Z (Forward)
    R_bc = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 1. 90-degree rotation around camera X axis
    # In camera frame: Rx(90) = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    Rx_90 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    # Since camera X corresponds to body Y, this must transform to a 90-deg rotation around body Y:
    # Ry(90) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    Ry_90_expected = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    Rb_x = transform_relative_rotation_camera_to_body(Rx_90, R_bc)
    assert np.allclose(Rb_x, Ry_90_expected, atol=1e-12)

    # 2. 90-degree rotation around camera Z axis
    # In camera frame: Rz(90) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    Rz_90 = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Since camera Z corresponds to body Z, this must remain a 90-deg rotation around body Z:
    Rb_z = transform_relative_rotation_camera_to_body(Rz_90, R_bc)
    assert np.allclose(Rb_z, Rz_90, atol=1e-12)

    # 3. 90-degree rotation around camera Y axis
    # In camera frame: Ry(90) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    Ry_90 = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    # Since camera Y corresponds to -body X, a +90 deg rotation around camera Y is a -90 deg rotation around body X:
    # Rx(-90) = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    Rx_neg90_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    Rb_y = transform_relative_rotation_camera_to_body(Ry_90, R_bc)
    assert np.allclose(Rb_y, Rx_neg90_expected, atol=1e-12)


def test_synthetic_rotation_test_c_quaternion_consistency():
    """Section 4 - Test C (Quaternion consistency): Matrix -> Quaternion -> Matrix agreement."""
    R_bc = EUROC_T_BS[:3, :3]

    # Convert rotation matrix to unit quaternion
    q = rotation_matrix_to_quaternion(R_bc)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)

    # Convert back to rotation matrix
    R_reconstructed = quaternion_to_rotation_matrix(q)

    # Maximum element-wise numerical difference must be negligible (< 1e-12)
    max_diff = np.max(np.abs(R_reconstructed - R_bc))
    assert max_diff < 1e-12


def test_synthetic_translation_direction_axes():
    """Section 5 - Translation-direction transformation for camera +X, +Y, +Z axes."""
    R_bc = EUROC_T_BS[:3, :3]

    # Unit direction vectors along camera axes
    d_cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d_cam_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    d_cam_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    d_body_x = transform_direction_camera_to_body(d_cam_x, R_bc)
    d_body_y = transform_direction_camera_to_body(d_cam_y, R_bc)
    d_body_z = transform_direction_camera_to_body(d_cam_z, R_bc)

    # By definition of matrix-vector multiplication, R @ e_i equals column i of R
    assert np.allclose(d_body_x, R_bc[:, 0], atol=1e-12)
    assert np.allclose(d_body_y, R_bc[:, 1], atol=1e-12)
    assert np.allclose(d_body_z, R_bc[:, 2], atol=1e-12)

    # Preserves exact unit length
    assert np.isclose(np.linalg.norm(d_body_x), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(d_body_y), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(d_body_z), 1.0, atol=1e-12)


def test_forward_inverse_transformation_consistency():
    """Section 6 - Verify T_BC^-1 maps body coordinates back to camera coordinates."""
    # Deterministic test point in camera frame
    p_C = np.array([1.23456789, -2.34567891, 3.45678912], dtype=np.float64)

    # Forward: p_B = T_BC(p_C)
    p_B = transform_point_camera_to_body(p_C, EUROC_T_BS)

    # Inverse: p_C_recovered = T_CB(p_B)
    p_C_recovered = transform_point_body_to_camera(p_B, EUROC_T_BS)

    # Difference must be machine-precision zero
    max_err = np.max(np.abs(p_C_recovered - p_C))
    assert max_err < 1e-11

    # Also verify via explicit invert_transform matrix T_CB
    T_CB = invert_transform(EUROC_T_BS)
    p_B_homo = np.append(p_B, 1.0)
    p_C_homo = T_CB @ p_B_homo
    assert np.allclose(p_C_homo[:3], p_C, atol=1e-11)
    assert np.allclose(EUROC_T_BS @ T_CB, np.eye(4), atol=1e-12)


def test_rotation_orthogonality_and_determinant():
    """Section 7 - Verify loaded EuRoC rotation satisfies R.T @ R ~ I and det(R) ~ +1."""
    R_bc, p_bc = validate_extrinsics(EUROC_T_BS)

    # 1. Check orthonormality R.T @ R ~ I
    ortho_diff = np.abs(R_bc.T @ R_bc - np.eye(3))
    max_ortho_diff = np.max(ortho_diff)
    assert max_ortho_diff < 1e-11

    # 2. Check determinant det(R) == +1 (SO(3), no reflection)
    det_R = float(np.linalg.det(R_bc))
    assert math.isclose(det_R, 1.0, abs_tol=1e-11)

    # 3. Validation function accepts matrix
    assert validate_rotation_matrix(R_bc, tolerance=1e-6) is True
