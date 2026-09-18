"""Unit tests for Phase 4 Visual-Inertial Odometry baseline."""

import math
from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pytest

from geonav.config.settings import CameraConfig, Settings, VIOConfig
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.synchronization.types import SynchronizedMeasurement
from geonav.vio.geometry import (
    align_gravity,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_slerp,
    quaternion_to_rotation_matrix,
    rotate_vector,
    rotation_matrix_to_quaternion,
)
from geonav.vio.imu_propagator import IMUPropagator
from geonav.vio.pipeline import VIOPipeline
from geonav.vio.types import VIOStatus
from geonav.vio.visual_frontend import VisualFrontEnd


def _create_synthetic_pattern_image(width: int = 640, height: int = 480, shift_x: int = 0) -> np.ndarray:
    """Create a high-contrast synthetic test image containing geometric shapes."""
    img = np.zeros((height, width), dtype=np.uint8)
    # Draw multiple squares with high contrast
    for y in range(50, height - 50, 60):
        for x in range(50, width - 50, 60):
            pt_x = x + shift_x
            if 0 <= pt_x < width - 30:
                cv2.rectangle(img, (pt_x, y), (pt_x + 30, y + 30), 255, -1)
                cv2.circle(img, (pt_x + 15, y + 15), 5, 0, -1)
    return img


def _make_imu(timestamp: float, ax: float = 0.0, ay: float = 0.0, az: float = -9.81, wx: float = 0.0) -> IMUSample:
    """Helper to create dummy IMUSample."""
    return IMUSample(
        timestamp=timestamp,
        linear_acceleration=(ax, ay, az),
        angular_velocity=(wx, 0.0, 0.0),
    )


def test_1_vio_initialization() -> None:
    """Test 1: Verify initial state is position=[0,0,0], velocity=[0,0,0], orientation=identity."""
    pipeline = VIOPipeline()
    assert pipeline.status == VIOStatus.UNINITIALIZED
    assert pipeline.get_state() is None

    init_frame = CameraFrame(timestamp=10.0, frame_data=_create_synthetic_pattern_image())
    init_state = pipeline.initialize(10.0, init_frame)

    assert pipeline.status == VIOStatus.INITIALIZED
    assert init_state.timestamp == 10.0
    assert init_state.position == (0.0, 0.0, 0.0)
    assert init_state.velocity == (0.0, 0.0, 0.0)
    assert init_state.orientation == (1.0, 0.0, 0.0, 0.0)
    assert len(pipeline.get_trajectory()) == 1


def test_2_valid_imu_propagation() -> None:
    """Test 2: Compare numerical IMU propagation against analytical kinematic integration."""
    # Propagator with g_world = [0, 0, 9.81]
    propagator = IMUPropagator(gravity_magnitude=9.81)

    # Accelerometer measures specific force: ax = 2.0 m/s^2 forward, az = -9.81 (cancels gravity)
    # Over dt = 1.0s, analytical result:
    # a_world = [2.0, 0.0, -9.81] + [0, 0, 9.81] = [2.0, 0.0, 0.0]
    # v(1.0) = v0 + a*t = [2.0, 0.0, 0.0]
    # p(1.0) = p0 + 0.5*a*t^2 = [1.0, 0.0, 0.0]
    sample1 = _make_imu(0.0, ax=2.0, ay=0.0, az=-9.81)
    sample2 = _make_imu(1.0, ax=2.0, ay=0.0, az=-9.81)

    p0 = np.zeros(3)
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    res, (p_final, v_final, q_final) = propagator.propagate_interval(p0, v0, q0, [sample1, sample2])

    assert res.success is True
    assert p_final[0] == pytest.approx(1.0, abs=1e-3)
    assert p_final[1] == pytest.approx(0.0, abs=1e-3)
    assert p_final[2] == pytest.approx(0.0, abs=1e-3)
    assert v_final[0] == pytest.approx(2.0, abs=1e-3)
    assert q_final[0] == pytest.approx(1.0, abs=1e-3)  # No rotation


def test_3_invalid_non_positive_imu_dt_rejection() -> None:
    """Test 3: Reject zero dt, negative dt, or decreasing timestamps in IMU propagation."""
    propagator = IMUPropagator()
    p0 = np.zeros(3)
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    # Zero dt
    s_dup1 = _make_imu(1.0)
    s_dup2 = _make_imu(1.0)
    with pytest.raises(ValueError, match="Non-positive or non-increasing IMU time step"):
        propagator.propagate_interval(p0, v0, q0, [s_dup1, s_dup2])

    # Negative dt
    s_rev1 = _make_imu(2.0)
    s_rev2 = _make_imu(1.5)
    with pytest.raises(ValueError, match="Non-positive or non-increasing IMU time step"):
        propagator.propagate_interval(p0, v0, q0, [s_rev1, s_rev2])


def test_4_feature_extraction_tracking() -> None:
    """Test 4: Extract and track Shi-Tomasi corners on synthetic geometric images."""
    frontend = VisualFrontEnd()
    img1 = _create_synthetic_pattern_image(shift_x=0)
    img2 = _create_synthetic_pattern_image(shift_x=5)  # 5 pixel horizontal translation

    pts1 = frontend.detect_features(img1)
    assert len(pts1) >= 20  # Sufficient corners detected

    tracked_prev, tracked_curr = frontend.track_features(img1, img2, pts1)
    assert len(tracked_curr) >= 15

    # Check that tracked shift is approximately 5 pixels along X
    shifts = tracked_curr - tracked_prev
    median_shift_x = float(np.median(shifts[:, 0]))
    median_shift_y = float(np.median(shifts[:, 1]))
    assert median_shift_x == pytest.approx(5.0, abs=0.5)
    assert median_shift_y == pytest.approx(0.0, abs=0.5)


def test_5_insufficient_features_handling() -> None:
    """Test 5: Uniform/featureless image must produce degraded/failure behavior."""
    frontend = VisualFrontEnd()
    uniform_img = np.full((480, 640), 128, dtype=np.uint8)

    pts = frontend.detect_features(uniform_img)
    assert len(pts) == 0

    # Motion estimation with empty points fails gracefully
    res = frontend.estimate_relative_motion(pts, pts)
    assert res.success is False
    assert "Insufficient" in res.status_message


def test_6_invalid_correspondence_handling() -> None:
    """Test 6: Mismatched point counts or NaN coordinates rejected safely."""
    frontend = VisualFrontEnd()
    pts1 = np.ones((10, 2), dtype=np.float32)
    pts2 = np.ones((20, 2), dtype=np.float32)

    res = frontend.estimate_relative_motion(pts1, pts2)
    assert res.success is False
    assert "Mismatched" in res.status_message

    # NaN coordinates
    pts_nan = np.full((15, 2), np.nan, dtype=np.float32)
    res_nan = frontend.estimate_relative_motion(pts_nan, pts_nan)
    assert res_nan.success is False
    assert "Non-finite" in res_nan.status_message


def test_7_camera_geometry_failure_handling() -> None:
    """Test 7: Collinear/degenerate points must not produce a fabricated pose."""
    frontend = VisualFrontEnd()
    # All points lie exactly on a single straight line: (x, 100)
    collinear_pts = np.zeros((30, 2), dtype=np.float32)
    collinear_pts[:, 0] = np.linspace(50, 550, 30)
    collinear_pts[:, 1] = 100.0  # Zero y-variance

    res = frontend.estimate_relative_motion(collinear_pts, collinear_pts)
    assert res.success is False
    assert "Degenerate" in res.status_message or "failed" in res.status_message


def test_8_valid_visual_relative_motion() -> None:
    """Test 8: Recover known relative rotation from synthetic projected 3D points."""
    frontend = VisualFrontEnd()
    K = frontend.K

    # Generate 50 random 3D landmark points in front of the camera (Z > 0)
    np.random.seed(42)
    pts_3d = np.random.uniform(low=[-2, -2, 3], high=[2, 2, 7], size=(60, 3))

    # Known pure yaw rotation of 5 degrees around optical axis / Y
    angle = math.radians(5.0)
    R_true = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )
    t_true = np.array([0.1, 0.0, 0.0])  # Non-zero translation direction

    # Project into Frame 1 (camera 1 at origin)
    p1_cam = pts_3d
    u1 = (K @ p1_cam.T).T
    pts1_2d = u1[:, :2] / u1[:, 2:3]

    # Project into Frame 2: p2 = R_true * (p1 - t_true)
    p2_cam = (R_true @ (pts_3d - t_true).T).T
    u2 = (K @ p2_cam.T).T
    pts2_2d = u2[:, :2] / u2[:, 2:3]

    res = frontend.estimate_relative_motion(pts1_2d.astype(np.float32), pts2_2d.astype(np.float32))

    assert res.success is True
    assert res.R_rel is not None
    assert res.t_rel is not None
    # Check trace of R_rel matches trace of R_true
    assert np.trace(res.R_rel) == pytest.approx(np.trace(R_true), abs=0.1)


def test_9_coordinate_transformation_correctness() -> None:
    """Test 9: Verify quaternion algebra, vector rotation, and rotation matrix conversion."""
    # 90-degree rotation around Z axis
    q_z90 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    q_z90 = quaternion_normalize(q_z90)

    # Rotate vector [1, 0, 0] by 90 degrees around Z -> should be [0, 1, 0]
    v = np.array([1.0, 0.0, 0.0])
    v_rot = rotate_vector(q_z90, v)
    assert v_rot[0] == pytest.approx(0.0, abs=1e-6)
    assert v_rot[1] == pytest.approx(1.0, abs=1e-6)
    assert v_rot[2] == pytest.approx(0.0, abs=1e-6)

    # Conversion to rotation matrix and back
    R = quaternion_to_rotation_matrix(q_z90)
    q_back = rotation_matrix_to_quaternion(R)
    assert q_back[0] == pytest.approx(q_z90[0], abs=1e-6)
    assert q_back[3] == pytest.approx(q_z90[3], abs=1e-6)

    # Slerp at t=0.5 between identity and q_z90 should yield 45-degree rotation
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    q_mid = quaternion_slerp(q_ident, q_z90, 0.5)
    assert q_mid[0] == pytest.approx(math.cos(math.pi / 8), abs=1e-6)


def test_10_vio_processing_valid_synchronized_measurement() -> None:
    """Test 10: Full VIO pipeline execution on consecutive synthetic measurements."""
    pipeline = VIOPipeline()
    img1 = _create_synthetic_pattern_image(shift_x=0)
    img2 = _create_synthetic_pattern_image(shift_x=4)

    f1 = CameraFrame(timestamp=1.0, frame_data=img1)
    f2 = CameraFrame(timestamp=1.1, frame_data=img2)

    # IMU samples in interval (1.0, 1.1]
    imu_samples = [
        _make_imu(1.02, ax=0.1, az=-9.81),
        _make_imu(1.06, ax=0.1, az=-9.81),
        _make_imu(1.10, ax=0.1, az=-9.81),
    ]

    m1 = SynchronizedMeasurement(
        camera_frame=f1,
        imu_samples=[],
        start_timestamp=None,
        end_timestamp=1.0,
    )
    m2 = SynchronizedMeasurement(
        camera_frame=f2,
        imu_samples=imu_samples,
        start_timestamp=1.0,
        end_timestamp=1.1,
    )

    s1 = pipeline.process_measurement(m1)
    assert pipeline.status == VIOStatus.INITIALIZED
    assert s1.timestamp == 1.0

    s2 = pipeline.process_measurement(m2)
    assert s2 is not None
    assert s2.timestamp == 1.1
    assert pipeline.status in (VIOStatus.TRACKING_OK, VIOStatus.DEGRADED_IMU_ONLY)

    traj = pipeline.get_trajectory()
    assert len(traj) == 2


def test_11_invalid_synchronized_measurement_handling() -> None:
    """Test 11: Reject non-increasing timestamps or invalid measurement types."""
    pipeline = VIOPipeline()
    img = _create_synthetic_pattern_image()
    f1 = CameraFrame(timestamp=2.0, frame_data=img)
    f2_bad = CameraFrame(timestamp=1.5, frame_data=img)  # Non-increasing

    m1 = SynchronizedMeasurement(camera_frame=f1, imu_samples=[], start_timestamp=None, end_timestamp=2.0)
    m2_bad = SynchronizedMeasurement(camera_frame=f2_bad, imu_samples=[], start_timestamp=2.0, end_timestamp=1.5)

    pipeline.process_measurement(m1)
    with pytest.raises(ValueError, match="Non-increasing camera timestamp"):
        pipeline.process_measurement(m2_bad)

    with pytest.raises(TypeError, match="Expected SynchronizedMeasurement"):
        pipeline.process_measurement("not_a_measurement")  # type: ignore


def test_12_deterministic_repeated_execution() -> None:
    """Test 12: Same inputs produce identical outputs across pipeline resets."""
    img1 = _create_synthetic_pattern_image(shift_x=0)
    img2 = _create_synthetic_pattern_image(shift_x=3)
    m1 = SynchronizedMeasurement(CameraFrame(1.0, img1), [], None, 1.0)
    m2 = SynchronizedMeasurement(CameraFrame(1.1, img2), [_make_imu(1.05), _make_imu(1.10)], 1.0, 1.1)

    p1 = VIOPipeline()
    p1.process_measurement(m1)
    s1_run1 = p1.process_measurement(m2)

    p2 = VIOPipeline()
    p2.process_measurement(m1)
    s1_run2 = p2.process_measurement(m2)

    assert s1_run1.position == s1_run2.position
    assert s1_run1.velocity == s1_run2.velocity
    assert s1_run1.orientation == s1_run2.orientation


def test_13_no_nan_inf_in_valid_vio_output() -> None:
    """Test 13: Every component of NavigationState in trajectory is finite."""
    pipeline = VIOPipeline()
    img1 = _create_synthetic_pattern_image(shift_x=0)
    img2 = _create_synthetic_pattern_image(shift_x=2)

    m1 = SynchronizedMeasurement(CameraFrame(1.0, img1), [], None, 1.0)
    m2 = SynchronizedMeasurement(
        CameraFrame(1.1, img2),
        [_make_imu(1.05, ax=0.05, az=-9.81), _make_imu(1.10, ax=0.05, az=-9.81)],
        1.0,
        1.1,
    )

    pipeline.process_measurement(m1)
    pipeline.process_measurement(m2)

    for state in pipeline.get_trajectory():
        assert np.all(np.isfinite(state.position))
        assert np.all(np.isfinite(state.velocity))
        assert np.all(np.isfinite(state.orientation))
        assert math.isfinite(state.timestamp)


def test_14_visual_degradation_fallback_to_imu_only() -> None:
    """Test 14: Force visual failure and verify DEGRADED_IMU_ONLY mode."""
    pipeline = VIOPipeline()
    img_good = _create_synthetic_pattern_image()
    # Uniform featureless image causes tracking to fail
    img_blank = np.full((480, 640), 128, dtype=np.uint8)

    m1 = SynchronizedMeasurement(CameraFrame(1.0, img_good), [], None, 1.0)
    m2 = SynchronizedMeasurement(
        CameraFrame(1.1, img_blank),
        [_make_imu(1.05, ax=0.5, az=-9.81), _make_imu(1.10, ax=0.5, az=-9.81)],
        1.0,
        1.1,
    )

    pipeline.process_measurement(m1)
    s2 = pipeline.process_measurement(m2)

    assert pipeline.status == VIOStatus.DEGRADED_IMU_ONLY
    assert s2 is not None
    # Position must have moved due to IMU forward acceleration
    assert s2.position[0] > 0.0


def test_15_velocity_update_bounded_no_exponential_explosion() -> None:
    """Test 15: Regression test to verify velocity does not exponentially double over consecutive frames."""
    pipeline = VIOPipeline()
    t_start = 1.0
    dt_frame = 0.05  # 20 Hz camera

    # Process 50 consecutive frames with small continuous displacement
    num_frames = 50
    velocities = []

    for i in range(num_frames):
        t_curr = t_start + i * dt_frame
        shift = int(i * 1.5) % 30
        img = _create_synthetic_pattern_image(shift_x=shift)
        frame = CameraFrame(t_curr, img)

        if i == 0:
            m = SynchronizedMeasurement(frame, [], None, t_curr)
        else:
            t_prev = t_curr - dt_frame
            # Constant small motion with gravity compensation
            imu_samples = [
                _make_imu(t_prev + 0.025, ax=0.01, az=-9.81),
                _make_imu(t_curr, ax=0.01, az=-9.81),
            ]
            m = SynchronizedMeasurement(frame, imu_samples, t_prev, t_curr)

        state = pipeline.process_measurement(m)
        if state is not None:
            v_mag = float(np.linalg.norm(state.velocity))
            velocities.append(v_mag)

    # Verify that all velocities are strictly finite
    assert all(math.isfinite(v) for v in velocities)

    # In the original bug: v_k was doubled at every frame, so v_50 would be > 10^12
    # In the fixed implementation: velocity remains strictly bounded and physical
    assert max(velocities) < 10.0, f"Velocity exploded: max speed = {max(velocities)} m/s"


def test_16_gravity_alignment_level_stationary():
    """Test 16: Level stationary IMU produces identity orientation [1, 0, 0, 0]."""
    # Specific force measured at rest with NED level: f_meas = -R^T g = [0, 0, -9.81]
    samples = [_make_imu(timestamp=0.01 * i, ax=0.0, ay=0.0, az=-9.81) for i in range(10)]
    q = align_gravity(samples, min_samples=5)

    assert isinstance(q, np.ndarray)
    assert q.shape == (4,)
    # Level IMU gives w=1, x=0, y=0, z=0
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_17_gravity_alignment_known_roll():
    """Test 17: Stationary IMU with known roll produces correct leveled quaternion."""
    phi = math.radians(30.0)  # +30 deg roll
    g = 9.81
    # Body rolled by phi: R_wb = R_x(phi), f_meas = -R_wb^T @ g_world = [0, -g*sin(phi), -g*cos(phi)]
    f_meas = np.array([0.0, -g * math.sin(phi), -g * math.cos(phi)])
    samples = [_make_imu(0.01 * i, ax=f_meas[0], ay=f_meas[1], az=f_meas[2]) for i in range(10)]

    q = align_gravity(samples, min_samples=5)
    # Expected quaternion for roll phi, pitch 0, yaw 0:
    # q_x = [cos(phi/2), sin(phi/2), 0, 0]
    expected_q = np.array([math.cos(phi / 2.0), math.sin(phi / 2.0), 0.0, 0.0])
    dot = float(np.abs(np.dot(q, expected_q)))
    assert np.isclose(dot, 1.0, atol=1e-5)

    # Check that rotating f_meas to world balances gravity exactly:
    R = quaternion_to_rotation_matrix(q)
    f_world = R @ f_meas
    g_world = np.array([0.0, 0.0, g])
    a_kinematic = f_world + g_world
    assert np.allclose(a_kinematic, np.zeros(3), atol=1e-5)


def test_18_gravity_alignment_known_pitch():
    """Test 18: Stationary IMU with known pitch produces correct leveled quaternion."""
    theta = math.radians(20.0)  # +20 deg pitch
    g = 9.81
    # Body pitched by theta: f_meas = [g*sin(theta), 0, -g*cos(theta)]
    f_meas = np.array([g * math.sin(theta), 0.0, -g * math.cos(theta)])
    samples = [_make_imu(0.01 * i, ax=f_meas[0], ay=f_meas[1], az=f_meas[2]) for i in range(10)]

    q = align_gravity(samples, min_samples=5)
    # Expected quaternion for roll 0, pitch theta, yaw 0:
    # q_y = [cos(theta/2), 0, sin(theta/2), 0]
    expected_q = np.array([math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0])
    dot = float(np.abs(np.dot(q, expected_q)))
    assert np.isclose(dot, 1.0, atol=1e-5)

    # Check gravity balance:
    R = quaternion_to_rotation_matrix(q)
    f_world = R @ f_meas
    g_world = np.array([0.0, 0.0, g])
    a_kinematic = f_world + g_world
    assert np.allclose(a_kinematic, np.zeros(3), atol=1e-5)


def test_19_gravity_alignment_combined_roll_pitch():
    """Test 19: Combined known roll and pitch case correctly restores gravity vector."""
    phi = math.radians(-15.0)
    theta = math.radians(25.0)
    g = 9.81

    # Form rotation R_wb = R_y(theta) * R_x(phi)
    # At rest, f_meas = -R_wb^T * g_world
    q_x = np.array([math.cos(phi / 2.0), math.sin(phi / 2.0), 0.0, 0.0])
    q_y = np.array([math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0])
    q_true = quaternion_multiply(q_y, q_x)
    R_wb = quaternion_to_rotation_matrix(q_true)
    g_world = np.array([0.0, 0.0, g])
    f_meas = -R_wb.T @ g_world

    samples = [_make_imu(0.01 * i, ax=f_meas[0], ay=f_meas[1], az=f_meas[2]) for i in range(10)]
    q_est = align_gravity(samples, min_samples=5)

    dot = float(np.abs(np.dot(q_est, q_true)))
    assert np.isclose(dot, 1.0, atol=1e-5)

    R_est = quaternion_to_rotation_matrix(q_est)
    f_world = R_est @ f_meas
    a_kinematic = f_world + g_world
    assert np.allclose(a_kinematic, np.zeros(3), atol=1e-5)


def test_20_gravity_alignment_deterministic_yaw():
    """Test 20: Yaw remains deterministically initialized to 0.0."""
    test_angles = [
        (math.radians(10), math.radians(20)),
        (math.radians(-35), math.radians(-15)),
        (math.radians(45), math.radians(-30)),
    ]
    g_world = np.array([0.0, 0.0, 9.81])
    for phi, theta in test_angles:
        q_x = np.array([math.cos(phi / 2.0), math.sin(phi / 2.0), 0.0, 0.0])
        q_y = np.array([math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0])
        q_rot = quaternion_multiply(q_y, q_x)
        R = quaternion_to_rotation_matrix(q_rot)
        f_meas = -R.T @ g_world

        samples = [_make_imu(0.01 * i, ax=f_meas[0], ay=f_meas[1], az=f_meas[2]) for i in range(10)]
        q_est = align_gravity(samples, min_samples=5)

        # Yaw from quaternion: psi = arctan2(2(w*z + x*y), 1 - 2(y^2 + z^2))
        w, x, y, z = q_est
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        assert np.isclose(yaw, 0.0, atol=1e-5)


def test_21_gravity_alignment_rejects_nan_inf_samples():
    """Test 21: NaN and Inf accelerometer measurements are safely rejected."""
    valid_sample = (0.0, 0.0, -9.81)
    nan_sample = (float("nan"), 0.0, -9.81)
    inf_sample = (0.0, float("inf"), -9.81)

    # Mix with sufficient valid samples
    samples = [nan_sample, inf_sample] + [valid_sample for _ in range(5)]
    q = align_gravity(samples, min_samples=5)
    assert np.all(np.isfinite(q))
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-5)

    # When ALL samples are invalid, ValueError must be raised
    with pytest.raises(ValueError, match="Insufficient valid finite accelerometer samples"):
        align_gravity([nan_sample, inf_sample], min_samples=1)


def test_22_gravity_alignment_insufficient_samples():
    """Test 22: Insufficient samples raise ValueError, and pipeline falls back safely."""
    # Fewer samples than min_samples
    samples = [_make_imu(0.0, ax=0.0, ay=0.0, az=-9.81)]
    with pytest.raises(ValueError, match="Insufficient valid finite accelerometer samples"):
        align_gravity(samples, min_samples=5)

    with pytest.raises(ValueError, match="Insufficient valid finite accelerometer samples"):
        align_gravity([], min_samples=1)

    # Test pipeline fallback behavior: initialize with empty IMU samples
    pipeline = VIOPipeline()
    state = pipeline.initialize(0.0, initial_imu_samples=[])
    assert state.orientation == (1.0, 0.0, 0.0, 0.0)


def test_23_gravity_alignment_quaternion_normalized():
    """Test 23: Quaternion is strictly normalized to unit length."""
    # Arbitrary non-unit acceleration vector
    samples = [_make_imu(0.01 * i, ax=3.5, ay=-4.2, az=-8.1) for i in range(10)]
    q = align_gravity(samples, min_samples=5)
    norm = np.linalg.norm(q)
    assert np.isclose(norm, 1.0, atol=1e-12)

    # Pipeline initialization also maintains unit quaternion
    pipeline = VIOPipeline()
    state = pipeline.initialize(0.0, initial_imu_samples=samples)
    q_pipe = np.asarray(state.orientation)
    assert np.isclose(np.linalg.norm(q_pipe), 1.0, atol=1e-12)


def test_24_gravity_alignment_compensation_zero_acceleration():
    """Test 24: Gravity compensation produces approximately zero kinematic acceleration at rest."""
    g = 9.81
    f_raw = np.array([8.91, -0.42, -3.34])
    f_normed = f_raw / np.linalg.norm(f_raw) * g

    samples = [_make_imu(0.005 * i, ax=f_normed[0], ay=f_normed[1], az=f_normed[2]) for i in range(20)]
    q_aligned = align_gravity(samples, min_samples=10)

    # Propagate with IMUPropagator
    propagator = IMUPropagator(gravity_magnitude=g)
    p0 = np.zeros(3)
    v0 = np.zeros(3)

    p1, v1, q1 = propagator.integrate_sample_step(p0, v0, q_aligned, samples[0], dt=0.005)

    # Linear kinematic acceleration = (v1 - v0) / dt
    a_kinematic = (v1 - v0) / 0.005
    assert np.allclose(a_kinematic, np.zeros(3), atol=1e-5)
    assert np.allclose(p1, np.zeros(3), atol=1e-6)


