"""Unit tests for Phase 5 3D point-cloud mapping subsystem."""

import math
from pathlib import Path
import sys
import tempfile

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pytest

from geonav.config.settings import CameraConfig, ExtrinsicsConfig, MappingConfig
from geonav.mapping.mapper import LocalMapper
from geonav.mapping.point_cloud import PointCloudMap
from geonav.mapping.triangulation import (
    compute_parallax_angle_deg,
    project_world_point,
    triangulate_two_views,
    validate_triangulated_point,
)
from geonav.mapping.types import (
    Keyframe,
    Landmark,
    MappingStatistics,
    TriangulationStatus,
)
from geonav.state import NavigationState
from geonav.vio.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from geonav.vio.types import VisualTrackingResult


def test_1_identity_camera_poses_triangulation():
    """Test 1: Linear DLT triangulation with identity poses and parallax baseline."""
    # Camera 0 at origin
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])

    # Camera 1 translated along X by 0.5 m (pure translation baseline)
    C1 = np.array([0.5, 0.0, 0.0])
    R1 = np.eye(3)
    t1 = -R1 @ C1
    P1 = np.hstack([R1, t1.reshape(3, 1)])

    # Known 3D world point in front of both cameras
    X_true = np.array([0.25, -0.10, 3.50])

    # Project to normalized ray coordinates
    x0_norm = X_true[:2] / X_true[2]
    X_cam1 = R1 @ X_true + t1
    x1_norm = X_cam1[:2] / X_cam1[2]

    # Triangulate
    X_rec = triangulate_two_views(P0, P1, x0_norm, x1_norm)

    assert np.allclose(X_rec, X_true, atol=1e-10)
    assert np.max(np.abs(X_rec - X_true)) < 1e-10


def test_2_known_translation_triangulation():
    """Test 2: Known baseline translation along arbitrary direction reconstructs 3D point accurately."""
    # Camera 0 at origin
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])

    # Camera 1 displaced by [0.2, 0.1, 0.05] meters
    C1 = np.array([0.2, 0.1, 0.05])
    R1 = np.eye(3)
    t1 = -R1 @ C1
    P1 = np.hstack([R1, t1.reshape(3, 1)])

    # Multiple 3D test points
    test_points = [
        np.array([0.0, 0.0, 4.0]),
        np.array([-0.5, 0.3, 2.5]),
        np.array([1.2, -0.8, 5.0]),
    ]

    for X_true in test_points:
        x0_norm = X_true[:2] / X_true[2]
        X_cam1 = R1 @ X_true + t1
        x1_norm = X_cam1[:2] / X_cam1[2]

        X_rec = triangulate_two_views(P0, P1, x0_norm, x1_norm)
        assert np.allclose(X_rec, X_true, atol=1e-9)


def test_3_known_rotation_triangulation():
    """Test 3: Triangulation under combined translation baseline and 15-deg relative rotation."""
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])

    # 15 degrees yaw rotation around Y axis + translation
    angle = math.radians(15.0)
    R1 = np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    C1 = np.array([0.4, 0.0, 0.0])
    t1 = -R1 @ C1
    P1 = np.hstack([R1, t1.reshape(3, 1)])

    X_true = np.array([0.5, 0.2, 3.0])

    x0_norm = X_true[:2] / X_true[2]
    X_cam1 = R1 @ X_true + t1
    x1_norm = X_cam1[:2] / X_cam1[2]

    X_rec = triangulate_two_views(P0, P1, x0_norm, x1_norm)
    assert np.allclose(X_rec, X_true, atol=1e-8)


def test_4_reprojection_error_computation():
    """Test 4: Reprojection error computation matches ground truth projections."""
    K = np.array([[450.0, 0.0, 320.0], [0.0, 450.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(4)

    R_wc = np.eye(3)
    p_wc = np.array([0.0, 0.0, 0.0])
    X_w = np.array([0.5, 0.2, 2.5])

    proj_pt, depth = project_world_point(X_w, R_wc, p_wc, K, dist)

    assert depth == pytest.approx(2.5, abs=1e-6)
    expected_u = 450.0 * (0.5 / 2.5) + 320.0  # 410.0
    expected_v = 450.0 * (0.2 / 2.5) + 240.0  # 276.0

    assert proj_pt[0] == pytest.approx(expected_u, abs=1e-4)
    assert proj_pt[1] == pytest.approx(expected_v, abs=1e-4)


def test_5_invalid_negative_depth_rejection():
    """Test 5: Points reconstructed behind camera are rejected with REJECTED_NEGATIVE_DEPTH."""
    config = MappingConfig(min_depth_m=0.2, max_depth_m=40.0)
    K = np.array([[450.0, 0.0, 320.0], [0.0, 450.0, 240.0], [0.0, 0.0, 1.0]])

    kf0 = Keyframe(
        id=0,
        timestamp=0.0,
        R_wc=np.eye(3),
        p_wc=np.zeros(3),
        P_norm=np.hstack([np.eye(3), np.zeros((3, 1))]),
        feature_ids=np.array([1]),
        feature_points=np.array([[320.0, 240.0]]),
        feature_points_norm=np.array([[0.0, 0.0]]),
    )

    kf1 = Keyframe(
        id=1,
        timestamp=0.1,
        R_wc=np.eye(3),
        p_wc=np.array([0.5, 0.0, 0.0]),
        P_norm=np.hstack([np.eye(3), np.array([[-0.5], [0.0], [0.0]])]),
        feature_ids=np.array([1]),
        feature_points=np.array([[300.0, 240.0]]),
        feature_points_norm=np.array([[-0.1, 0.0]]),
    )

    # Point with negative Z (behind camera 0)
    X_behind = np.array([0.25, 0.0, -2.0])

    status, _ = validate_triangulated_point(
        X_behind, kf0, kf1, kf0.feature_points[0], kf1.feature_points[0], K, None, config
    )
    assert status == TriangulationStatus.REJECTED_NEGATIVE_DEPTH


def test_6_nan_inf_numerical_safety():
    """Test 6: Triangulation rejects NaN/Inf inputs gracefully."""
    P_valid = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_nan = P_valid.copy()
    P_nan[0, 0] = float("nan")

    u0 = np.array([0.1, 0.2])
    u1 = np.array([0.15, 0.2])

    with pytest.raises(ValueError, match="non-finite"):
        triangulate_two_views(P_nan, P_valid, u0, u1)

    with pytest.raises(ValueError, match="non-finite"):
        triangulate_two_views(P_valid, P_valid, [float("inf"), 0.0], u1)

    # Singular geometry check (identical projection centers, zero parallax)
    with pytest.raises(ValueError, match="Singular"):
        triangulate_two_views(P_valid, P_valid, [0.0, 0.0], [0.0, 0.0])


def test_7_point_cloud_map_insertion_and_bounds():
    """Test 7: PointCloudMap stores landmarks and calculates correct 3D bounds."""
    pc = PointCloudMap(max_landmarks=100)
    assert pc.count() == 0

    lm1 = Landmark(id=1, position_world=np.array([1.0, 2.0, 3.0]), reprojection_error=0.8, color=(255, 0, 0))
    lm2 = Landmark(id=2, position_world=np.array([-2.0, 0.5, 4.5]), reprojection_error=1.2, color=(0, 255, 0))

    assert pc.add_landmark(lm1) is True
    assert pc.add_landmark(lm2) is True
    assert pc.count() == 2

    pts = pc.get_points()
    assert pts.shape == (2, 3)
    assert np.allclose(pts[0], [1.0, 2.0, 3.0])

    colors = pc.get_colors()
    assert colors is not None
    assert colors.shape == (2, 3)
    assert np.array_equal(colors[0], [255, 0, 0])

    bounds = pc.get_bounds()
    assert bounds["x"] == (-2.0, 1.0)
    assert bounds["y"] == (0.5, 2.0)
    assert bounds["z"] == (3.0, 4.5)


def test_8_point_cloud_reset():
    """Test 8: Reset clears all stored landmarks."""
    pc = PointCloudMap()
    pc.add_landmark(Landmark(id=1, position_world=np.array([0.0, 1.0, 2.0])))
    assert pc.count() == 1

    pc.reset()
    assert pc.count() == 0
    assert len(pc.get_points()) == 0


def test_9_ply_export_validity():
    """Test 9: PointCloudMap exports valid standard ASCII PLY file."""
    pc = PointCloudMap()
    pc.add_landmark(Landmark(id=1, position_world=np.array([1.23, -4.56, 7.89]), color=(200, 100, 50)))
    pc.add_landmark(Landmark(id=2, position_world=np.array([0.12, 0.34, 0.56]), color=(10, 20, 30)))

    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = Path(tmpdir) / "test_map.ply"
        exported = pc.export_ply(ply_path)
        assert exported.is_file()

        content = exported.read_text(encoding="ascii")
        assert "ply" in content
        assert "format ascii 1.0" in content
        assert "element vertex 2" in content
        assert "property float x" in content
        assert "property float y" in content
        assert "property float z" in content
        assert "property uchar red" in content
        assert "end_header" in content
        assert "1.2300 -4.5600 7.8900 200 100 50" in content


def test_10_duplicate_track_association():
    """Test 10: LocalMapper associates continuing feature tracks and avoids duplicate landmarks."""
    map_cfg = MappingConfig(
        keyframe_translation_threshold_m=0.10,
        min_triangulation_angle_deg=0.5,
        min_tracked_features=2,
    )
    cam_cfg = CameraConfig(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    mapper = LocalMapper(mapping_config=map_cfg, camera_config=cam_cfg)

    # Frame 0: Keyframe at origin
    pts_f0 = np.array([[320.0, 240.0], [350.0, 240.0]], dtype=np.float32)
    state0 = NavigationState(timestamp=0.0, position=(0.0, 0.0, 0.0), velocity=(0, 0, 0), orientation=(1, 0, 0, 0))
    res0 = VisualTrackingResult(success=True, num_tracked=2, num_inliers=2, points_prev=pts_f0, points_curr=pts_f0)
    mapper.process_frame(0.0, None, state0, res0)
    assert len(mapper.keyframes) == 1

    # Frame 1: Translated by 0.2m along X -> becomes Keyframe 1 and triggers triangulation
    pts_f1 = np.array([[290.0, 240.0], [320.0, 240.0]], dtype=np.float32)
    state1 = NavigationState(timestamp=0.1, position=(0.2, 0.0, 0.0), velocity=(1, 0, 0), orientation=(1, 0, 0, 0))
    res1 = VisualTrackingResult(success=True, num_tracked=2, num_inliers=2, points_prev=pts_f0, points_curr=pts_f1)
    mapper.process_frame(0.1, None, state1, res1)

    initial_landmark_count = mapper.point_cloud.count()
    assert initial_landmark_count > 0

    # Frame 2: Continuing the exact same tracks further
    pts_f2 = np.array([[260.0, 240.0], [290.0, 240.0]], dtype=np.float32)
    state2 = NavigationState(timestamp=0.2, position=(0.4, 0.0, 0.0), velocity=(1, 0, 0), orientation=(1, 0, 0, 0))
    res2 = VisualTrackingResult(success=True, num_tracked=2, num_inliers=2, points_prev=pts_f1, points_curr=pts_f2)
    mapper.process_frame(0.2, None, state2, res2)

    # Landmark count must NOT duplicate existing landmarks
    assert mapper.point_cloud.count() == initial_landmark_count
    # Observation count should increase
    lms = mapper.get_landmarks()
    assert any(lm.observation_count >= 3 for lm in lms)


def test_11_parallax_angle_filtering():
    """Test 11: Insufficient parallax angle between camera rays is rejected."""
    C0 = np.array([0.0, 0.0, 0.0])
    C1 = np.array([0.001, 0.0, 0.0])  # Sub-millimeter baseline (almost zero parallax)
    X_w = np.array([0.0, 0.0, 10.0])  # Point 10 meters away

    angle = compute_parallax_angle_deg(C0, C1, X_w)
    assert angle < 0.1  # Very small parallax (< 0.1 deg)

    config = MappingConfig(min_triangulation_angle_deg=1.0)
    kf0 = Keyframe(0, 0.0, np.eye(3), C0, np.zeros((3, 4)), np.array([1]), np.zeros((1, 2)), np.zeros((1, 2)))
    kf1 = Keyframe(1, 0.1, np.eye(3), C1, np.zeros((3, 4)), np.array([1]), np.zeros((1, 2)), np.zeros((1, 2)))

    K = np.eye(3)
    status, _ = validate_triangulated_point(X_w, kf0, kf1, np.zeros(2), np.zeros(2), K, None, config)
    assert status == TriangulationStatus.REJECTED_PARALLAX


def test_12_keyframe_selection_thresholds():
    """Test 12: Keyframe selection respects translation and rotation thresholds."""
    map_cfg = MappingConfig(
        keyframe_translation_threshold_m=0.20,
        keyframe_rotation_threshold_deg=10.0,
        min_tracked_features=5,
    )
    mapper = LocalMapper(mapping_config=map_cfg)

    # Initial frame
    st0 = NavigationState(0.0, (0.0, 0.0, 0.0), (0, 0, 0), (1, 0, 0, 0))
    res0 = VisualTrackingResult(True, 10, 10, points_prev=np.zeros((10, 2)), points_curr=np.zeros((10, 2)))
    assert mapper.process_frame(0.0, None, st0, res0) == 0  # First keyframe

    # Small movement: delta_p = 0.05m (< 0.20m) -> should NOT become keyframe
    st1 = NavigationState(0.1, (0.05, 0.0, 0.0), (0, 0, 0), (1, 0, 0, 0))
    res1 = VisualTrackingResult(True, 10, 10, points_prev=np.zeros((10, 2)), points_curr=np.zeros((10, 2)))
    assert mapper.process_frame(0.1, None, st1, res1) is None
    assert len(mapper.keyframes) == 1

    # Large translation: delta_p = 0.25m (>= 0.20m) -> becomes Keyframe 1
    st2 = NavigationState(0.2, (0.25, 0.0, 0.0), (0, 0, 0), (1, 0, 0, 0))
    res2 = VisualTrackingResult(True, 10, 10, points_prev=np.zeros((10, 2)), points_curr=np.zeros((10, 2)))
    assert mapper.process_frame(0.2, None, st2, res2) == 1
    assert len(mapper.keyframes) == 2
