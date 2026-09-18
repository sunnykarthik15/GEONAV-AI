"""Unit tests for Phase 6 Ground-Truth Evaluation Subsystem."""

import math
from pathlib import Path
import sys
import tempfile

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from geonav.evaluation.alignment import align_trajectory_points, umeyama_alignment
from geonav.evaluation.association import associate_timestamps
from geonav.evaluation.evaluator import TrajectoryEvaluator
from geonav.evaluation.trajectory import (
    compute_ate,
    compute_axis_errors,
    compute_orientation_errors,
    compute_rpe,
    compute_scale_analysis,
    compute_velocity_errors,
)
from geonav.evaluation.types import TrajectoryPoint
from geonav.vio.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion


def test_1_timestamp_matching_exact():
    """Test 1: Exact timestamp matching pairs all elements correctly."""
    est_pts = [
        TrajectoryPoint(timestamp=1.0, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=2.0, position=np.array([1, 1, 1]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=3.0, position=np.array([2, 2, 2]), orientation=np.array([1, 0, 0, 0])),
    ]
    gt_pts = [
        TrajectoryPoint(timestamp=1.0, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=2.0, position=np.array([1, 1, 1]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=3.0, position=np.array([2, 2, 2]), orientation=np.array([1, 0, 0, 0])),
    ]

    pairs, un_est, un_gt = associate_timestamps(est_pts, gt_pts, max_time_diff_s=0.01)
    assert len(pairs) == 3
    assert un_est == 0
    assert un_gt == 0
    for p in pairs:
        assert p.time_diff == pytest.approx(0.0, abs=1e-12)


def test_2_timestamp_matching_small_offset():
    """Test 2: Small timestamp offset (< max_time_diff_s) matches successfully."""
    est_pts = [
        TrajectoryPoint(timestamp=1.002, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=2.004, position=np.array([1, 1, 1]), orientation=np.array([1, 0, 0, 0])),
    ]
    gt_pts = [
        TrajectoryPoint(timestamp=1.000, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=2.000, position=np.array([1, 1, 1]), orientation=np.array([1, 0, 0, 0])),
    ]

    pairs, un_est, un_gt = associate_timestamps(est_pts, gt_pts, max_time_diff_s=0.01)
    assert len(pairs) == 2
    assert un_est == 0
    assert un_gt == 0
    assert pairs[0].time_diff == pytest.approx(0.002, abs=1e-6)
    assert pairs[1].time_diff == pytest.approx(0.004, abs=1e-6)


def test_3_timestamp_rejection_beyond_threshold():
    """Test 3: Offsets exceeding threshold are rejected."""
    est_pts = [
        TrajectoryPoint(timestamp=1.025, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=2.001, position=np.array([1, 1, 1]), orientation=np.array([1, 0, 0, 0])),
    ]
    gt_pts = [
        TrajectoryPoint(timestamp=1.000, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0])),
        TrajectoryPoint(timestamp=2.000, position=np.array([1, 1, 1]), orientation=np.array([1, 0, 0, 0])),
    ]

    pairs, un_est, un_gt = associate_timestamps(est_pts, gt_pts, max_time_diff_s=0.01)
    assert len(pairs) == 1
    assert un_est == 1
    assert un_gt == 1
    assert pairs[0].timestamp_est == 2.001


def test_4_zero_trajectory_error():
    """Test 4: Identical trajectories yield exact zero ATE, axis error, and RPE."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.2],
        [2.0, 1.0, 0.4],
        [3.0, 1.5, 0.6],
    ])

    ate = compute_ate(pts, pts)
    assert ate.rmse == pytest.approx(0.0, abs=1e-12)
    assert ate.mean == pytest.approx(0.0, abs=1e-12)
    assert ate.max == pytest.approx(0.0, abs=1e-12)
    assert ate.final_error == pytest.approx(0.0, abs=1e-12)

    axis_err = compute_axis_errors(pts, pts)
    assert axis_err.x_rmse == pytest.approx(0.0, abs=1e-12)
    assert axis_err.y_rmse == pytest.approx(0.0, abs=1e-12)
    assert axis_err.z_rmse == pytest.approx(0.0, abs=1e-12)


def test_5_known_constant_translation_error():
    """Test 5: Constant translation offset [3, 4, 0] yields exactly 5.0m RMSE."""
    pts_gt = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    offset = np.array([3.0, 4.0, 0.0])  # ||offset|| = 5.0
    pts_est = pts_gt + offset

    ate = compute_ate(pts_est, pts_gt)
    assert ate.rmse == pytest.approx(5.0, abs=1e-10)
    assert ate.mean == pytest.approx(5.0, abs=1e-10)
    assert ate.median == pytest.approx(5.0, abs=1e-10)
    assert ate.max == pytest.approx(5.0, abs=1e-10)
    assert ate.final_error == pytest.approx(5.0, abs=1e-10)


def test_6_known_axis_specific_errors():
    """Test 6: Known offsets along individual axes match axis error calculations."""
    pts_gt = np.zeros((10, 3))
    pts_est = pts_gt.copy()
    pts_est[:, 0] = 2.0   # dx = 2.0
    pts_est[:, 1] = -1.5  # dy = -1.5
    pts_est[:, 2] = 4.0   # dz = 4.0

    ax_err = compute_axis_errors(pts_est, pts_gt)
    assert ax_err.x_rmse == pytest.approx(2.0, abs=1e-10)
    assert ax_err.y_rmse == pytest.approx(1.5, abs=1e-10)
    assert ax_err.z_rmse == pytest.approx(4.0, abs=1e-10)
    assert ax_err.x_mae == pytest.approx(2.0, abs=1e-10)
    assert ax_err.y_mae == pytest.approx(1.5, abs=1e-10)
    assert ax_err.z_mae == pytest.approx(4.0, abs=1e-10)


def test_7_known_rotation_error():
    """Test 7: Known 45-degree rotation difference matches orientation error computation."""
    # Rotation of 45 degrees around Z axis
    theta = math.radians(45.0)
    R_rot = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    q_rot = rotation_matrix_to_quaternion(R_rot)
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])

    quats_est = np.tile(q_rot, (5, 1))
    quats_gt = np.tile(q_ident, (5, 1))

    orient_err = compute_orientation_errors(quats_est, quats_gt)
    assert orient_err.mean_deg == pytest.approx(45.0, abs=1e-6)
    assert orient_err.rmse_deg == pytest.approx(45.0, abs=1e-6)
    assert orient_err.max_deg == pytest.approx(45.0, abs=1e-6)


def test_8_ate_rmse_calculation():
    """Test 8: Explicit mathematical verification of RMSE with diverse error magnitudes."""
    pts_gt = np.zeros((4, 3))
    # Errors: [1, 2, 3, 4] meters
    pts_est = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 4.0, 0.0],
    ])

    expected_rmse = math.sqrt((1**2 + 2**2 + 3**2 + 4**2) / 4)  # sqrt(30 / 4) = sqrt(7.5) = 2.7386127875
    ate = compute_ate(pts_est, pts_gt)
    assert ate.rmse == pytest.approx(expected_rmse, abs=1e-10)
    assert ate.min == pytest.approx(1.0, abs=1e-10)
    assert ate.max == pytest.approx(4.0, abs=1e-10)


def test_9_trajectory_length_calculation():
    """Test 9: Trajectory length correctly sums Euclidean step segments."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],   # length 5
        [3.0, 4.0, 12.0],  # length 12
    ])
    gt_pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],   # length 1
    ])

    res = compute_scale_analysis(pts, gt_pts)
    assert res.estimated_path_length == pytest.approx(17.0, abs=1e-10)
    assert res.groundtruth_path_length == pytest.approx(1.0, abs=1e-10)
    assert res.scale_ratio == pytest.approx(17.0, abs=1e-10)


def test_10_known_scale_ratio():
    """Test 10: Scale ratio matches ratio of path lengths."""
    est_pts = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])  # length 4
    gt_pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])   # length 2

    scale_res = compute_scale_analysis(est_pts, gt_pts)
    assert scale_res.scale_ratio == pytest.approx(2.0, abs=1e-10)


def test_11_se3_alignment_umeyama():
    """Test 11: Rigid SE(3) Umeyama alignment recovers rotation and translation perfectly."""
    # 60 deg rotation around Y + translation
    ang = math.radians(60.0)
    R_true = np.array([
        [math.cos(ang), 0.0, math.sin(ang)],
        [0.0, 1.0, 0.0],
        [-math.sin(ang), 0.0, math.cos(ang)],
    ])
    t_true = np.array([5.0, -3.0, 2.0])

    np.random.seed(123)
    pts_est = np.random.uniform(-10, 10, size=(50, 3))
    pts_gt = (R_true @ pts_est.T).T + t_true

    s, R_rec, t_rec = umeyama_alignment(pts_est, pts_gt, with_scale=False)
    assert s == 1.0
    assert np.allclose(R_rec, R_true, atol=1e-10)
    assert np.allclose(t_rec, t_true, atol=1e-10)

    # Aligned ATE must be zero
    pts_aligned = align_trajectory_points(pts_est, s, R_rec, t_rec)
    ate_aligned = compute_ate(pts_aligned, pts_gt)
    assert ate_aligned.rmse < 1e-10


def test_12_sim3_alignment_umeyama():
    """Test 12: Sim(3) Umeyama alignment recovers scale, rotation, and translation."""
    scale_true = 2.45
    ang = math.radians(35.0)
    R_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(ang), -math.sin(ang)],
        [0.0, math.sin(ang), math.cos(ang)],
    ])
    t_true = np.array([-1.2, 4.5, 3.1])

    np.random.seed(456)
    pts_est = np.random.uniform(-5, 5, size=(60, 3))
    pts_gt = scale_true * (R_true @ pts_est.T).T + t_true

    s_rec, R_rec, t_rec = umeyama_alignment(pts_est, pts_gt, with_scale=True)
    assert s_rec == pytest.approx(scale_true, abs=1e-10)
    assert np.allclose(R_rec, R_true, atol=1e-10)
    assert np.allclose(t_rec, t_true, atol=1e-10)

    pts_aligned = align_trajectory_points(pts_est, s_rec, R_rec, t_rec)
    ate_aligned = compute_ate(pts_aligned, pts_gt)
    assert ate_aligned.rmse < 1e-10


def test_13_relative_pose_error_rpe():
    """Test 13: Relative Pose Error computation across 1-frame interval."""
    # Constant velocity trajectory along X: v = 1 m/s, dt = 0.1s => step = 0.1m
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    pts_est = [
        TrajectoryPoint(t, np.array([t * 1.0, 0, 0]), np.array([1, 0, 0, 0])) for t in times
    ]
    # Ground truth moving at v = 1.1 m/s (step = 0.11m)
    pts_gt = [
        TrajectoryPoint(t, np.array([t * 1.1, 0, 0]), np.array([1, 0, 0, 0])) for t in times
    ]

    rpe = compute_rpe(pts_est, pts_gt, interval_frames=1)
    # Relative step error per frame: |0.10 - 0.11| = 0.01 m
    assert rpe.trans_rmse == pytest.approx(0.01, abs=1e-6)
    assert rpe.rot_rmse_deg == pytest.approx(0.0, abs=1e-6)


def test_14_nan_inf_safety():
    """Test 14: Non-finite inputs are rejected by ATE and alignment."""
    valid_pts = np.ones((5, 3))
    nan_pts = valid_pts.copy()
    nan_pts[0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        compute_ate(nan_pts, valid_pts)

    with pytest.raises(ValueError, match="non-finite"):
        umeyama_alignment(nan_pts, valid_pts)


def test_15_invalid_quaternion_handling():
    """Test 15: TrajectoryPoint normalizes quaternions and rejects zero-norm."""
    # Unnormalized quaternion is normalized automatically
    pt = TrajectoryPoint(timestamp=1.0, position=[0, 0, 0], orientation=[2.0, 0, 0, 0])
    assert np.isclose(np.linalg.norm(pt.orientation), 1.0, atol=1e-12)
    assert pt.orientation[0] == pytest.approx(1.0, abs=1e-6)

    # Near-zero norm quaternion is rejected
    with pytest.raises(ValueError, match="Near-zero quaternion norm"):
        TrajectoryPoint(timestamp=1.0, position=[0, 0, 0], orientation=[0.0, 0, 0, 0])


def test_16_umeyama_reflection_handling():
    """Test 16: Umeyama alignment enforces det(R) = +1.0 and does not return a reflection."""
    np.random.seed(789)
    pts_src = np.random.uniform(-5, 5, size=(30, 3))
    # Create target with an explicit reflection across Z (det = -1)
    pts_refl = pts_src.copy()
    pts_refl[:, 2] = -pts_refl[:, 2]

    # Umeyama must return proper rotation in SO(3) with det(R) = +1.0
    s, R, t = umeyama_alignment(pts_src, pts_refl, with_scale=False)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_17_orientation_error_initial_alignment():
    """Test 17: compute_orientation_errors correctly computes raw vs frame-aligned orientation error."""
    # Frame 0: 90 deg yaw offset between est and gt
    # Frame 1: identical relative motion (both rotate 10 deg)
    ang_offset = math.radians(90.0)
    q_offset = [math.cos(ang_offset / 2), 0, 0, math.sin(ang_offset / 2)]

    q_gt0 = [1.0, 0.0, 0.0, 0.0]
    q_est0 = q_offset

    # Frame 1: both rotated by 10 deg pitch
    ang_pitch = math.radians(10.0)
    q_pitch = [math.cos(ang_pitch / 2), 0, math.sin(ang_pitch / 2), 0]
    from geonav.vio.geometry import quaternion_multiply
    q_gt1 = quaternion_multiply(q_gt0, q_pitch)
    q_est1 = quaternion_multiply(q_est0, q_pitch)

    quats_gt = np.array([q_gt0, q_gt1])
    quats_est = np.array([q_est0, q_est1])

    # Raw orientation error should reflect the 90 deg constant offset
    raw_res = compute_orientation_errors(quats_est, quats_gt, align_initial=False)
    assert raw_res.rmse_deg == pytest.approx(90.0, abs=1e-6)

    # Frame-aligned orientation error cancels the constant frame offset and yields 0 drift
    aligned_res = compute_orientation_errors(quats_est, quats_gt, align_initial=True)
    assert aligned_res.rmse_deg == pytest.approx(0.0, abs=1e-6)


def test_18_umeyama_arbitrary_3d_transform():
    phi, theta, psi = 0.35, -0.42, 1.15
    Rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    Ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    Rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    R_true = Rz @ Ry @ Rx
    s_true = 0.732
    t_true = np.array([12.3, -4.5, 6.7])

    np.random.seed(999)
    pts_src = np.random.uniform(-20, 20, size=(100, 3))
    pts_dst = s_true * (R_true @ pts_src.T).T + t_true

    s_rec, R_rec, t_rec = umeyama_alignment(pts_src, pts_dst, with_scale=True)
    assert s_rec == pytest.approx(s_true, abs=1e-10)
    assert np.allclose(R_rec, R_true, atol=1e-10)
    assert np.allclose(t_rec, t_true, atol=1e-10)

    aligned = align_trajectory_points(pts_src, s_rec, R_rec, t_rec)
    ate = compute_ate(aligned, pts_dst)
    assert ate.rmse < 1e-10


def test_19_timestamp_association_boundary_audit():
    """Test 19: Timestamp association rejects frames outside ground-truth temporal coverage."""
    # GT runs from t=1.0 to t=5.0 (step 0.05)
    gt_times = np.arange(1.0, 5.01, 0.05)
    gt_pts = [TrajectoryPoint(t, [t, 0, 0], [1, 0, 0, 0]) for t in gt_times]

    # Estimated poses run from t=0.0 to t=6.0 (has 1s pre-flight and 1s post-flight)
    est_times = np.arange(0.0, 6.01, 0.1)
    est_pts = [TrajectoryPoint(t, [t, 0, 0], [1, 0, 0, 0]) for t in est_times]

    matched, un_est, un_gt = associate_timestamps(est_pts, gt_pts, max_time_diff_s=0.01)
    # Frames with t < 1.0 (t=0.0..0.9 => 10 frames) and t > 5.0 (t=5.1..6.0 => 10 frames) are unmatched
    assert un_est == 20
    assert len(matched) == len(est_pts) - 20
