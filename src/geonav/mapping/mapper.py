"""Local sparse 3D point-cloud mapping pipeline."""

import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from geonav.config.settings import CameraConfig, ExtrinsicsConfig, MappingConfig
from geonav.mapping.point_cloud import PointCloudMap
from geonav.mapping.triangulation import (
    compute_parallax_angle_deg,
    project_world_point,
    triangulate_two_views,
    validate_triangulated_point,
)
from geonav.mapping.types import Keyframe, Landmark, MappingStatistics, TriangulationStatus
from geonav.sensors.camera import CameraFrame
from geonav.state import NavigationState
from geonav.vio.geometry import (
    quaternion_to_rotation_matrix,
    validate_extrinsics,
)
from geonav.vio.types import VisualTrackingResult


class LocalMapper:
    """Constructs a local sparse 3D point-cloud map from visual observations and VIO poses."""

    def __init__(
        self,
        mapping_config: Optional[MappingConfig] = None,
        camera_config: Optional[CameraConfig] = None,
        extrinsics_config: Optional[ExtrinsicsConfig] = None,
    ) -> None:
        """Initialize local mapper with calibration and tracking settings.

        Args:
            mapping_config: Mapping configuration parameters.
            camera_config: Camera intrinsic parameters.
            extrinsics_config: Camera-to-body spatial extrinsic parameters.
        """
        self.map_cfg = mapping_config or MappingConfig()
        self.cam_cfg = camera_config or CameraConfig()
        self.ext_cfg = extrinsics_config or ExtrinsicsConfig()

        # Build camera intrinsic matrix K
        fx = self.cam_cfg.fx if self.cam_cfg.fx is not None else float(self.cam_cfg.width)
        fy = self.cam_cfg.fy if self.cam_cfg.fy is not None else float(self.cam_cfg.height)
        cx = self.cam_cfg.cx if self.cam_cfg.cx is not None else (self.cam_cfg.width / 2.0)
        cy = self.cam_cfg.cy if self.cam_cfg.cy is not None else (self.cam_cfg.height / 2.0)

        self.K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dist = self.cam_cfg.distortion_coeffs
        self.dist_coeffs = np.array(dist, dtype=np.float64) if dist is not None else np.zeros(4, dtype=np.float64)

        # Extrinsics R_BC and p_BC (camera to body)
        R_bc_tuple = self.ext_cfg.rotation_matrix
        T_mat = np.eye(4, dtype=np.float64)
        T_mat[:3, :3] = np.array(R_bc_tuple, dtype=np.float64)
        T_mat[:3, 3] = np.array(self.ext_cfg.translation, dtype=np.float64)
        self.R_bc, self.p_bc = validate_extrinsics(T_mat)

        # Point cloud storage
        self.point_cloud = PointCloudMap(max_landmarks=self.map_cfg.max_landmarks)

        # Keyframe history
        self.keyframes: List[Keyframe] = []
        self._next_keyframe_id = 0

        # Feature tracking state
        self._next_track_id = 0
        self._active_track_ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self._active_pts_prev: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._track_initial_observations: Dict[int, Tuple[Keyframe, np.ndarray, np.ndarray]] = {}
        self._track_landmark_ids: Dict[int, int] = {}

        # Diagnostic metrics and statistics
        self.stats = MappingStatistics()
        self._start_time = time.time()
        self._total_processing_time = 0.0
        self._reprojection_errors: List[float] = []
        self._depths: List[float] = []

    def compute_camera_pose(
        self,
        nav_state: NavigationState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute camera pose in world frame from vehicle navigation state and extrinsics.

        Formulas:
            R_WC = R_WB * R_BC
            p_WC = p_WB + R_WB * p_BC
            R_CW = R_WC^T
            t_CW = -R_CW * p_WC
            P_norm = [R_CW | t_CW]

        Args:
            nav_state: Vehicle navigation state (position, orientation).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (R_wc, p_wc, P_norm).
        """
        R_wb = quaternion_to_rotation_matrix(nav_state.orientation)
        p_wb = np.asarray(nav_state.position, dtype=np.float64)

        R_wc = R_wb @ self.R_bc
        p_wc = p_wb + R_wb @ self.p_bc

        R_cw = R_wc.T
        t_cw = -R_cw @ p_wc
        P_norm = np.hstack([R_cw, t_cw.reshape(3, 1)])

        return R_wc, p_wc, P_norm

    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """Undistort 2D pixel coordinates into normalized optical ray coordinates (x/z, y/z).

        Args:
            pts_px: Array of 2D pixel coordinates (N, 2).

        Returns:
            np.ndarray: Array of 2D normalized ray coordinates (N, 2).
        """
        if pts_px is None or len(pts_px) == 0:
            return np.empty((0, 2), dtype=np.float64)

        pts_reshaped = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        norm_pts = cv2.undistortPoints(pts_reshaped, self.K, self.dist_coeffs)
        return norm_pts.reshape(-1, 2).astype(np.float64)

    def _sample_point_color(
        self,
        image: Optional[np.ndarray],
        pt_px: np.ndarray,
    ) -> Optional[Tuple[int, int, int]]:
        """Safely sample RGB color from the image at given pixel coordinates."""
        if not self.map_cfg.extract_rgb or image is None:
            return None

        u, v = int(round(pt_px[0])), int(round(pt_px[1]))
        h, w = image.shape[:2]
        if 0 <= u < w and 0 <= v < h:
            val = image[v, u]
            if isinstance(val, np.ndarray) and len(val) >= 3:
                # Assuming BGR image from OpenCV
                return (int(val[2]), int(val[1]), int(val[0]))
            else:
                gray_val = int(val)
                return (gray_val, gray_val, gray_val)
        return None

    def _is_keyframe(
        self,
        p_wc: np.ndarray,
        R_wc: np.ndarray,
        num_features: int,
    ) -> bool:
        """Evaluate keyframe selection criteria based on displacement, rotation, and intervals."""
        if not self.keyframes:
            return True

        last_kf = self.keyframes[-1]
        delta_p = float(np.linalg.norm(p_wc - last_kf.p_wc))

        # Relative rotation angle between camera poses
        R_rel = R_wc @ last_kf.R_wc.T
        cos_angle = min(max((np.trace(R_rel) - 1.0) * 0.5, -1.0), 1.0)
        delta_deg = math.degrees(math.acos(cos_angle))

        frames_since_last = self.stats.total_frames - (last_kf.id * self.map_cfg.keyframe_interval_frames)

        # Criteria: translation, rotation, or periodic interval with minimal movement
        has_translation = delta_p >= self.map_cfg.keyframe_translation_threshold_m
        has_rotation = delta_deg >= self.map_cfg.keyframe_rotation_threshold_deg
        has_interval = (frames_since_last >= self.map_cfg.keyframe_interval_frames) and (delta_p >= 0.04)

        has_sufficient_features = num_features >= self.map_cfg.min_tracked_features

        return (has_translation or has_rotation or has_interval) and has_sufficient_features

    def process_frame(
        self,
        timestamp: float,
        image: Optional[np.ndarray],
        nav_state: NavigationState,
        visual_result: Optional[VisualTrackingResult] = None,
    ) -> Optional[int]:
        """Process a camera frame and update the 3D landmark map.

        Args:
            timestamp: Frame acquisition timestamp in seconds.
            image: Grayscale or color camera frame array.
            nav_state: Current vehicle state estimated by VIO.
            visual_result: Visual front-end tracking result for this interval.

        Returns:
            Optional[int]: Keyframe ID if a new keyframe was formed, else None.
        """
        t_start = time.time()
        self.stats.total_frames += 1

        if not self.map_cfg.enabled or nav_state is None:
            return None

        # Compute camera pose in world frame
        R_wc, p_wc, P_norm = self.compute_camera_pose(nav_state)

        # Feature correspondences
        pts_curr = visual_result.points_curr if (visual_result and visual_result.points_curr is not None) else None
        pts_prev = visual_result.points_prev if (visual_result and visual_result.points_prev is not None) else None

        num_features = len(pts_curr) if pts_curr is not None else 0

        # Track ID assignment and persistence
        current_track_ids = np.empty((num_features,), dtype=np.int64)

        if num_features > 0 and pts_prev is not None and len(self._active_track_ids) > 0 and len(self._active_pts_prev) > 0:
            # Associate surviving tracks with previous frame features using nearest neighbor matching
            # LK optical flow points that remain identical in prev_pts inherit their track ID
            prev_flat = self._active_pts_prev
            curr_prev_flat = pts_prev

            # Distance matrix between previous active points and current interval's prev points
            dists = np.linalg.norm(curr_prev_flat[:, None, :] - prev_flat[None, :, :], axis=2)
            min_indices = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)

            for i in range(num_features):
                if min_dists[i] < 1.0:  # Matches same physical point in previous frame
                    current_track_ids[i] = self._active_track_ids[min_indices[i]]
                else:
                    current_track_ids[i] = self._next_track_id
                    self._next_track_id += 1
        else:
            for i in range(num_features):
                current_track_ids[i] = self._next_track_id
                self._next_track_id += 1

        # Update landmarks that have already been triangulated
        if num_features > 0 and pts_curr is not None:
            for tid, pt_px in zip(current_track_ids, pts_curr):
                if tid in self._track_landmark_ids:
                    lm_id = self._track_landmark_ids[tid]
                    lm = self.point_cloud.get_landmark(lm_id)
                    if lm is not None:
                        try:
                            pt_proj, z = project_world_point(lm.position_world, R_wc, p_wc, self.K, self.dist_coeffs)
                            if z > 0:
                                err = float(np.linalg.norm(pt_proj - pt_px))
                                if err <= self.map_cfg.max_reprojection_error_px:
                                    self.point_cloud.update_landmark(lm_id, timestamp, err)
                        except Exception:
                            pass

        # Check keyframe criteria
        new_kf_id = None
        if self._is_keyframe(p_wc, R_wc, num_features):
            pts_norm = self.undistort_points(pts_curr)
            kf = Keyframe(
                id=self._next_keyframe_id,
                timestamp=timestamp,
                R_wc=R_wc,
                p_wc=p_wc,
                P_norm=P_norm,
                feature_ids=current_track_ids,
                feature_points=pts_curr if pts_curr is not None else np.empty((0, 2), dtype=np.float32),
                feature_points_norm=pts_norm,
                frame_image=image,
            )
            self.keyframes.append(kf)
            new_kf_id = kf.id
            self._next_keyframe_id += 1
            self.stats.keyframes_count = len(self.keyframes)

            # Register initial observations for new tracks
            if pts_curr is not None:
                for tid, pt_px, pt_n in zip(current_track_ids, pts_curr, pts_norm):
                    if tid not in self._track_initial_observations and tid not in self._track_landmark_ids:
                        self._track_initial_observations[tid] = (kf, pt_px, pt_n)

            # Triangulate tracks across keyframes
            if len(self.keyframes) >= 2:
                self._triangulate_new_landmarks(kf)

        # Update active tracking buffers for next frame
        if num_features > 0 and pts_curr is not None:
            self._active_track_ids = current_track_ids
            self._active_pts_prev = pts_curr
        else:
            self._active_track_ids = np.empty((0,), dtype=np.int64)
            self._active_pts_prev = np.empty((0, 2), dtype=np.float32)

        # Update timing and statistics
        elapsed = time.time() - t_start
        self._total_processing_time += elapsed
        self.stats.runtime_s = self._total_processing_time
        if self._total_processing_time > 0:
            self.stats.fps = self.stats.total_frames / self._total_processing_time

        self.stats.total_landmarks = self.point_cloud.count()
        self.stats.spatial_extent = self.point_cloud.get_bounds()
        if self._reprojection_errors:
            self.stats.mean_reprojection_error = float(np.mean(self._reprojection_errors))
            self.stats.median_reprojection_error = float(np.median(self._reprojection_errors))
            self.stats.max_reprojection_error = float(np.max(self._reprojection_errors))
        if self._depths:
            self.stats.min_depth = float(np.min(self._depths))
            self.stats.max_depth = float(np.max(self._depths))

        return new_kf_id

    def _triangulate_new_landmarks(self, current_kf: Keyframe) -> None:
        """Attempt triangulation for active feature tracks against their initial keyframe."""
        if len(current_kf.feature_ids) == 0:
            return

        for tid, pt_curr_px, pt_curr_norm in zip(
            current_kf.feature_ids,
            current_kf.feature_points,
            current_kf.feature_points_norm,
        ):
            if tid in self._track_landmark_ids:
                continue  # Already triangulated

            if tid not in self._track_initial_observations:
                continue

            ref_kf, pt_ref_px, pt_ref_norm = self._track_initial_observations[tid]
            if ref_kf.id == current_kf.id:
                continue

            self.stats.triangulation_attempts += 1

            # Two-view linear DLT triangulation
            try:
                X_w = triangulate_two_views(
                    ref_kf.P_norm,
                    current_kf.P_norm,
                    pt_ref_norm,
                    pt_curr_norm,
                )
            except Exception:
                self.stats.rejected_numerical += 1
                continue

            # Validation against geometric criteria
            status, err = validate_triangulated_point(
                X_w,
                ref_kf,
                current_kf,
                pt_ref_px,
                pt_curr_px,
                self.K,
                self.dist_coeffs,
                self.map_cfg,
            )

            if status == TriangulationStatus.SUCCESS:
                # Sample RGB color if available
                color = self._sample_point_color(current_kf.frame_image, pt_curr_px)

                # Optical depth in current frame
                _, depth = project_world_point(X_w, current_kf.R_wc, current_kf.p_wc, self.K, self.dist_coeffs)

                landmark = Landmark(
                    id=int(tid),
                    position_world=X_w,
                    observation_count=2,
                    first_timestamp=ref_kf.timestamp,
                    last_timestamp=current_kf.timestamp,
                    reprojection_error=err,
                    color=color,
                )

                if self.point_cloud.add_landmark(landmark):
                    self.stats.triangulation_successes += 1
                    self._track_landmark_ids[tid] = landmark.id
                    self._reprojection_errors.append(err)
                    self._depths.append(depth)
                    # Remove initial observation record once triangulated
                    self._track_initial_observations.pop(tid, None)
            elif status == TriangulationStatus.REJECTED_PARALLAX:
                self.stats.rejected_parallax += 1
            elif status == TriangulationStatus.REJECTED_NEGATIVE_DEPTH:
                self.stats.rejected_negative_depth += 1
            elif status == TriangulationStatus.REJECTED_DEPTH_RANGE:
                self.stats.rejected_depth_range += 1
            elif status == TriangulationStatus.REJECTED_REPROJECTION:
                self.stats.rejected_reprojection += 1
            elif status == TriangulationStatus.REJECTED_NUMERICAL:
                self.stats.rejected_numerical += 1

    def get_landmarks(self) -> List[Landmark]:
        """Retrieve all reconstructed landmarks currently stored in the map."""
        return self.point_cloud.get_landmarks()

    def get_point_cloud(self) -> PointCloudMap:
        """Retrieve the PointCloudMap container object."""
        return self.point_cloud

    def get_statistics(self) -> MappingStatistics:
        """Retrieve current quantitative mapping metrics and diagnostics."""
        return self.stats

    def export_ply(self, path: Union[str, Path]) -> Path:
        """Export the current 3D point cloud map to a standard PLY file."""
        return self.point_cloud.export_ply(path)

    def reset(self) -> None:
        """Reset the local mapper state, clearing all landmarks, keyframes, and tracks."""
        self.point_cloud.clear()
        self.keyframes.clear()
        self._next_keyframe_id = 0
        self._next_track_id = 0
        self._active_track_ids = np.empty((0,), dtype=np.int64)
        self._active_pts_prev = np.empty((0, 2), dtype=np.float32)
        self._track_initial_observations.clear()
        self._track_landmark_ids.clear()
        self.stats = MappingStatistics()
        self._reprojection_errors.clear()
        self._depths.clear()
        self._total_processing_time = 0.0
