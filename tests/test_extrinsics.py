"""Unit tests for Phase 4B Camera-IMU Extrinsic Calibration."""

import math
from pathlib import Path
import sys
import tempfile

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import yaml

from geonav.config.settings import CameraConfig, ExtrinsicsConfig, Settings, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.datasets.euroc.camera_loader import EurocCameraLoader
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.synchronization.types import SynchronizedMeasurement
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
from geonav.vio.pipeline import VIOPipeline
from geonav.vio.types import VisualTrackingResult


def test_1_identity_extrinsic_transformation():
    """Test 1: Identity transformation leaves points and vectors unchanged."""
    T_ident = np.eye(4, dtype=np.float64)
    p_cam = np.array([1.5, -2.3, 4.8], dtype=np.float64)

    p_body = transform_point_camera_to_body(p_cam, T_ident)
    assert np.allclose(p_body, p_cam, atol=1e-12)

    p_back = transform_point_body_to_camera(p_body, T_ident)
    assert np.allclose(p_back, p_cam, atol=1e-12)

    T_inv = invert_transform(T_ident)
    assert np.allclose(T_inv, T_ident, atol=1e-12)


def test_2_known_pure_rotation():
    """Test 2: Known pure 90-degree rotation correctly transforms coordinate axes."""
    # 90 degrees rotation around Z axis:
    # X_cam -> Y_body, Y_cam -> -X_body, Z_cam -> Z_body
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R

    # Test camera X axis [1, 0, 0] -> [0, 1, 0]
    p_cam_x = np.array([1.0, 0.0, 0.0])
    p_body_x = transform_point_camera_to_body(p_cam_x, T)
    assert np.allclose(p_body_x, [0.0, 1.0, 0.0], atol=1e-12)

    # Test camera Y axis [0, 1, 0] -> [-1, 0, 0]
    p_cam_y = np.array([0.0, 1.0, 0.0])
    p_body_y = transform_point_camera_to_body(p_cam_y, T)
    assert np.allclose(p_body_y, [-1.0, 0.0, 0.0], atol=1e-12)

    # Direction transform also matches
    d_body = transform_direction_camera_to_body(p_cam_x, R)
    assert np.allclose(d_body, [0.0, 1.0, 0.0], atol=1e-12)


def test_3_known_translation():
    """Test 3: Known translation offset adds position correctly."""
    p_bc = np.array([0.15, -0.05, 0.25], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = p_bc

    p_cam = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    p_body = transform_point_camera_to_body(p_cam, T)
    expected_p = p_cam + p_bc
    assert np.allclose(p_body, expected_p, atol=1e-12)

    # Inverting translation
    p_back = transform_point_body_to_camera(p_body, T)
    assert np.allclose(p_back, p_cam, atol=1e-12)


def test_4_correct_inverse_transformation():
    """Test 4: Inverting rigid transformation produces exact mathematical inverse."""
    # Arbitrary rigid transform (rotation by 30 deg around X and 45 deg around Y + translation)
    phi = math.radians(30.0)
    theta = math.radians(45.0)
    Rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    Ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    R_bc = Ry @ Rx
    p_bc = np.array([-0.02, -0.06, 0.01])

    T_bc = np.eye(4, dtype=np.float64)
    T_bc[:3, :3] = R_bc
    T_bc[:3, 3] = p_bc

    T_cb = invert_transform(T_bc)

    # T_bc * T_cb == I and T_cb * T_bc == I
    assert np.allclose(T_bc @ T_cb, np.eye(4), atol=1e-12)
    assert np.allclose(T_cb @ T_bc, np.eye(4), atol=1e-12)

    # Transforming round-trip
    p_orig = np.array([2.5, -1.2, 3.7])
    p_body = transform_point_camera_to_body(p_orig, T_bc)
    p_restored = transform_point_body_to_camera(p_body, T_bc)
    assert np.allclose(p_restored, p_orig, atol=1e-12)


def test_5_rotation_orthonormality():
    """Test 5: validate_rotation_matrix enforces orthonormality and det(R) == +1."""
    # Valid rotation
    phi = math.radians(25.0)
    R_valid = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(phi), -math.sin(phi)],
        [0.0, math.sin(phi), math.cos(phi)],
    ])
    assert validate_rotation_matrix(R_valid) is True

    # Non-orthogonal matrix (sheared)
    R_sheared = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValueError, match="not orthonormal"):
        validate_rotation_matrix(R_sheared)

    # Reflection matrix (det == -1)
    R_reflect = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant is not \\+1"):
        validate_rotation_matrix(R_reflect)


def test_6_quaternion_conversion_consistency():
    """Test 6: Extrinsic rotation converts to/from quaternion consistently."""
    # EuRoC-like camera extrinsic rotation
    T_BS = np.array([
        [0.01486554, -0.99988093, 0.00414030, -0.02164015],
        [0.99955725, 0.01496721, 0.02571553, -0.06467699],
        [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
        [0.0, 0.0, 0.0, 1.0],
    ])
    R_bc, p_bc = validate_extrinsics(T_BS)

    q = rotation_matrix_to_quaternion(R_bc)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)

    R_back = quaternion_to_rotation_matrix(q)
    assert np.allclose(R_back, R_bc, atol=1e-6)


def test_7_camera_to_body_vector_transformation():
    """Test 7: Direction vector transformation preserves unit length."""
    phi = math.radians(45.0)
    R_bc = np.array([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    d_cam = np.array([0.6, 0.8, 0.0])  # norm = 1.0
    d_body = transform_direction_camera_to_body(d_cam, R_bc)
    assert np.isclose(np.linalg.norm(d_body), 1.0, atol=1e-12)

    # Zero norm direction handling
    with pytest.raises(ValueError, match="Direction vector contains non-finite"):
        transform_direction_camera_to_body([float("nan"), 0.0, 1.0], R_bc)


def test_8_body_to_camera_vector_transformation():
    """Test 8: Body to camera point transformation inverts camera to body."""
    T_bc = np.array([
        [0.0, -1.0, 0.0, 0.05],
        [1.0, 0.0, 0.0, -0.10],
        [0.0, 0.0, 1.0, 0.02],
        [0.0, 0.0, 0.0, 1.0],
    ])
    p_body = np.array([1.2, -3.4, 5.6])
    p_cam = transform_point_body_to_camera(p_body, T_bc)
    p_re_body = transform_point_camera_to_body(p_cam, T_bc)
    assert np.allclose(p_re_body, p_body, atol=1e-12)


def test_9_invalid_non_finite_calibration_rejection():
    """Test 9: Invalid or non-finite calibration matrices are rejected."""
    # Matrix with NaN
    T_nan = np.eye(4)
    T_nan[0, 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        validate_extrinsics(T_nan)

    # Matrix with Inf
    T_inf = np.eye(4)
    T_inf[1, 1] = float("inf")
    with pytest.raises(ValueError, match="non-finite values"):
        validate_extrinsics(T_inf)

    # Wrong shape
    with pytest.raises(ValueError, match="must have shape \\(4, 4\\)"):
        validate_extrinsics(np.eye(3))

    # Non-affine bottom row
    T_bad_row = np.eye(4)
    T_bad_row[3, 3] = 2.0
    with pytest.raises(ValueError, match="bottom row must be \\[0, 0, 0, 1\\]"):
        validate_extrinsics(T_bad_row)


def test_10_deterministic_repeated_transformation():
    """Test 10: Repeated transformations yield deterministic, bitwise-identical results."""
    T_bc = np.array([
        [0.01486554, -0.99988093, 0.00414030, -0.02164015],
        [0.99955725, 0.01496721, 0.02571553, -0.06467699],
        [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
        [0.0, 0.0, 0.0, 1.0],
    ])
    p_cam = np.array([0.123456789, -0.987654321, 2.345678901])

    first_res = transform_point_camera_to_body(p_cam, T_bc)
    for _ in range(50):
        res = transform_point_camera_to_body(p_cam, T_bc)
        assert np.array_equal(res, first_res)


def test_11_real_euroc_calibration_parsing():
    """Test 11: Real EuRoC sensor.yaml calibration parsing in EurocDataset."""
    repo_root = Path(__file__).resolve().parent.parent
    euroc_dir = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"

    if euroc_dir.is_dir():
        dataset = EurocDataset(euroc_dir)
        assert dataset.camera_extrinsics is not None
        assert dataset.camera_extrinsics.shape == (4, 4)

        # Check translation matches EuRoC cam0 sensor.yaml exactly
        R, t = validate_extrinsics(dataset.camera_extrinsics)
        assert t[0] == pytest.approx(-0.0216401454975, abs=1e-6)
        assert t[1] == pytest.approx(-0.064676986768, abs=1e-6)
        assert t[2] == pytest.approx(0.00981073058949, abs=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    else:
        # If dataset is not present, test via temporary directory with sensor.yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cam_dir = tmppath / "cam0"
            cam_dir.mkdir()
            (cam_dir / "data.csv").write_text("#timestamp,filename\n1000,1000.png\n", encoding="utf-8")
            (cam_dir / "1000.png").write_bytes(b"dummy")
            yaml_content = {
                "T_BS": {
                    "cols": 4,
                    "rows": 4,
                    "data": [
                        0.0, -1.0, 0.0, -0.02,
                        1.0, 0.0, 0.0, -0.06,
                        0.0, 0.0, 1.0, 0.01,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                }
            }
            with open(cam_dir / "sensor.yaml", "w") as f:
                yaml.dump(yaml_content, f)

            loader = EurocCameraLoader(cam_dir)
            assert loader.extrinsics_T_BS is not None
            assert loader.extrinsics_T_BS.shape == (4, 4)
            assert loader.extrinsics_T_BS[0, 3] == pytest.approx(-0.02, abs=1e-6)


def test_12_vio_visual_measurement_transformed_consistently():
    """Test 12: VIO pipeline integrates R_BC and transforms visual measurements consistently."""
    # Rotate 90 degrees around Z: camera X -> body Y, camera Y -> -body X
    R_bc = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    ext_config = ExtrinsicsConfig(rotation_matrix=R_bc)
    vio_config = VIOConfig(extrinsics=ext_config)

    pipeline = VIOPipeline(config=vio_config)
    assert np.allclose(pipeline._R_bc, np.array(R_bc), atol=1e-6)

    # Relative camera rotation: 10 degrees around camera X
    # In body frame, rotation must become 10 degrees around body Y
    alpha = math.radians(10.0)
    R_c_rel = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(alpha), -math.sin(alpha)],
        [0.0, math.sin(alpha), math.cos(alpha)],
    ])
    R_b_rel = transform_relative_rotation_camera_to_body(R_c_rel, np.array(R_bc))

    # Vector along camera X is [1, 0, 0] -> transforms to body Y [0, 1, 0]
    # R_b_rel applied to body Y should leave it invariant since it's the rotation axis
    assert np.allclose(R_b_rel @ [0, 1, 0], [0, 1, 0], atol=1e-6)

    # Relative translation along camera X [1, 0, 0] transforms to body Y [0, 1, 0]
    d_body = transform_direction_camera_to_body([1.0, 0.0, 0.0], np.array(R_bc))
    assert np.allclose(d_body, [0.0, 1.0, 0.0], atol=1e-6)
