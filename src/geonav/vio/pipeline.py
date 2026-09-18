"""Visual-Inertial Odometry (VIO) pipeline implementation."""

from typing import List, Optional, Sequence
import numpy as np

from geonav.config.settings import Settings, VIOConfig
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.state.state import NavigationState
from geonav.synchronization.types import SynchronizedMeasurement
from geonav.vio.geometry import (
    align_gravity,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_slerp,
    rotate_vector,
    rotation_matrix_to_quaternion,
    transform_direction_camera_to_body,
    transform_relative_rotation_camera_to_body,
    validate_rotation_matrix,
)
from geonav.vio.imu_propagator import IMUPropagator
from geonav.vio.types import VIOStatus, VisualTrackingResult
from geonav.vio.visual_frontend import VisualFrontEnd



class VIOPipeline:
    """Estimates vehicle motion trajectory by fusing synchronized visual and inertial measurements.

    Architecture:
        Classical baseline Visual-Inertial Odometry estimator using:
        - Shi-Tomasi feature detection with bidirectional Lucas-Kanade optical flow.
        - Essential matrix 5-point RANSAC for visual relative motion estimation.
        - High-rate Euler/quaternion IMU propagation with gravity compensation.
        - Complementary visual-inertial state correction and direction-constrained motion updates.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[VIOConfig] = None,
    ) -> None:
        """Initialize the VIO pipeline with configuration settings.

        Args:
            settings: Root Settings object.
            config: Optional explicit VIOConfig overriding settings.vio.
        """
        self.settings = settings or Settings()
        self.vio_config = config or self.settings.vio

        self.frontend = VisualFrontEnd(
            camera_config=self.settings.camera,
            vio_config=self.vio_config,
        )
        self.propagator = IMUPropagator(
            gravity_magnitude=self.vio_config.gravity_magnitude,
        )

        # Extrinsic camera-to-body spatial calibration
        extrinsics = self.vio_config.extrinsics
        self._R_bc = np.asarray(extrinsics.rotation_matrix, dtype=np.float64)
        self._p_bc = np.asarray(extrinsics.translation, dtype=np.float64)
        validate_rotation_matrix(self._R_bc)

        self._is_initialized: bool = False
        self._attitude_aligned: bool = False
        self._initial_imu_buffer: List[IMUSample] = []
        self._status: VIOStatus = VIOStatus.UNINITIALIZED
        self._current_state: Optional[NavigationState] = None
        self._trajectory: List[NavigationState] = []

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._last_visual_result: Optional[VisualTrackingResult] = None

    @property
    def status(self) -> VIOStatus:
        """Return the current estimator operational status."""
        return self._status

    def initialize(
        self,
        timestamp: float,
        initial_frame: Optional[CameraFrame] = None,
        initial_imu_samples: Optional[Sequence[IMUSample]] = None,
    ) -> NavigationState:
        """Initialize navigation state and initial visual features.

        Initial State Convention:
            Position: [0, 0, 0] meters.
            Velocity: [0, 0, 0] m/s.
            Orientation: Leveled unit quaternion [w, x, y, z] from stationary accelerometer gravity alignment,
                         or [1, 0, 0, 0] if alignment is disabled or samples unavailable.
            Frame: Local inertial navigation frame (NED).

        Args:
            timestamp: Initial epoch timestamp in seconds.
            initial_frame: Optional CameraFrame for extracting initial visual features.
            initial_imu_samples: Optional initial stationary IMU samples for gravity alignment.

        Returns:
            NavigationState: Initialized vehicle state at T_0.
        """
        initial_p = (0.0, 0.0, 0.0)
        initial_v = (0.0, 0.0, 0.0)
        initial_q = (1.0, 0.0, 0.0, 0.0)

        if (
            self.vio_config.initial_gravity_alignment
            and initial_imu_samples
        ):
            self._initial_imu_buffer.extend(initial_imu_samples)
            if len(self._initial_imu_buffer) >= self.vio_config.initial_alignment_min_samples:
                try:
                    q_arr = align_gravity(
                        self._initial_imu_buffer,
                        min_samples=self.vio_config.initial_alignment_min_samples,
                    )
                    initial_q = (float(q_arr[0]), float(q_arr[1]), float(q_arr[2]), float(q_arr[3]))
                    self._attitude_aligned = True
                except Exception:
                    initial_q = (1.0, 0.0, 0.0, 0.0)

        init_state = NavigationState(
            timestamp=float(timestamp),
            position=initial_p,
            velocity=initial_v,
            orientation=initial_q,
        )

        self._current_state = init_state
        self._trajectory = [init_state]
        self._is_initialized = True
        self._status = VIOStatus.INITIALIZED

        if initial_frame is not None:
            gray = self.frontend.preprocess_image(initial_frame.frame_data)
            self._prev_gray = gray
            self._prev_pts = self.frontend.detect_features(gray)

        return init_state

    def process_measurement(
        self,
        measurement: SynchronizedMeasurement,
    ) -> Optional[NavigationState]:
        """Process a synchronized camera-IMU measurement package to estimate updated vehicle state.

        Args:
            measurement: SynchronizedMeasurement containing camera frame and associated IMU samples.

        Returns:
            Optional[NavigationState]: Updated vehicle state estimate, or None if processing failed.

        Raises:
            TypeError: If measurement is not a SynchronizedMeasurement.
            ValueError: If timestamps are non-monotonic or corrupted.
        """
        if not isinstance(measurement, SynchronizedMeasurement):
            raise TypeError(f"Expected SynchronizedMeasurement instance, got {type(measurement)}")

        t_curr = measurement.end_timestamp

        # 1. First frame initialization
        if measurement.is_first_frame or not self._is_initialized:
            return self.initialize(t_curr, measurement.camera_frame, measurement.imu_samples)

        assert self._current_state is not None
        t_prev = self._current_state.timestamp

        if t_curr <= t_prev:
            raise ValueError(
                f"Non-increasing camera timestamp: current ({t_curr}) <= previous ({t_prev})"
            )

        # If initial attitude has not yet been aligned from stationary IMU and alignment is enabled,
        # accumulate stationary IMU samples and align attitude once sufficient samples are available
        if not self._attitude_aligned and self.vio_config.initial_gravity_alignment:
            if measurement.imu_samples:
                self._initial_imu_buffer.extend(measurement.imu_samples)
            if len(self._initial_imu_buffer) >= self.vio_config.initial_alignment_min_samples:
                try:
                    q_arr = align_gravity(
                        self._initial_imu_buffer,
                        min_samples=self.vio_config.initial_alignment_min_samples,
                    )
                    aligned_q = (float(q_arr[0]), float(q_arr[1]), float(q_arr[2]), float(q_arr[3]))
                    self._current_state = NavigationState(
                        timestamp=self._current_state.timestamp,
                        position=self._current_state.position,
                        velocity=self._current_state.velocity,
                        orientation=aligned_q,
                    )
                    if self._trajectory:
                        self._trajectory[0] = self._current_state
                    self._attitude_aligned = True
                except Exception:
                    pass


        # 2. Visual front-end tracking
        curr_gray = self.frontend.preprocess_image(measurement.camera_frame.frame_data)
        visual_res = VisualTrackingResult(
            success=False,
            num_tracked=0,
            num_inliers=0,
            status_message="Visual tracking not executed",
        )

        pts_tracked_curr = np.empty((0, 2), dtype=np.float32)

        if self._prev_gray is not None:
            if self._prev_pts is None or len(self._prev_pts) < self.frontend.min_features:
                self._prev_pts = self.frontend.detect_features(self._prev_gray)

            pts_prev_matched, pts_curr_matched = self.frontend.track_features(
                self._prev_gray,
                curr_gray,
                self._prev_pts,
            )

            if len(pts_prev_matched) >= self.frontend.min_features:
                visual_res = self.frontend.estimate_relative_motion(
                    pts_prev_matched,
                    pts_curr_matched,
                )
                pts_tracked_curr = pts_curr_matched

        self._last_visual_result = visual_res

        # 3. IMU propagation over the inter-frame interval
        p_prev = np.asarray(self._current_state.position, dtype=np.float64)
        v_prev = np.asarray(self._current_state.velocity, dtype=np.float64)
        q_prev = np.asarray(self._current_state.orientation, dtype=np.float64)

        try:
            imu_res, (p_prop, v_prop, q_prop) = self.propagator.propagate_interval(
                initial_p=p_prev,
                initial_v=v_prev,
                initial_q=q_prev,
                imu_samples=measurement.imu_samples,
            )
        except Exception as err:
            self._status = VIOStatus.FAILED
            raise ValueError(f"IMU propagation error: {err}") from err

        # 4. Fusion and State Update
        delta_t = max(t_curr - t_prev, 1e-4)

        if visual_res.success and visual_res.R_rel is not None and visual_res.t_rel is not None:
            # 1. Transform relative camera rotation to body/IMU frame:
            # Let R_c_rel = R_{C_1, C_0} transform vectors from camera frame 0 to 1.
            # Then the body relative rotation R_{B_1, B_0} is given by similarity transformation:
            # R_b_rel = R_BC * R_c_rel * R_BC^T
            # The body orientation change from t_prev to t_curr is delta_R_b_vis = R_b_rel^T = R_BC * R_c_rel^T * R_BC^T
            R_c_rel = visual_res.R_rel
            R_b_rel = transform_relative_rotation_camera_to_body(R_c_rel, self._R_bc)
            delta_R_b_vis = R_b_rel.T
            delta_q_vis = rotation_matrix_to_quaternion(delta_R_b_vis)

            # Target orientation predicted purely by visual rotation
            q_vis_target = quaternion_multiply(q_prev, delta_q_vis)

            # Complementary fusion of orientation (visual rotation suppresses gyro integration drift)
            alpha_rot = self.vio_config.visual_rotation_weight
            q_fused = quaternion_slerp(q_prop, q_vis_target, alpha_rot)

            # 2. Transform monocular visual translation direction from camera frame to body frame:
            # t_unit_body = R_BC * t_unit_cam
            # Note: For monocular VIO with unknown metric scale, visual translation scale is constrained
            # by IMU displacement norm. The physical lever-arm term (I - R_b_rel) * p_BC is sub-millimeter
            # per frame (<0.6 mm) and requires absolute metric scale, so the direction is transformed via R_BC.
            t_unit_cam = visual_res.t_rel
            t_unit_body = transform_direction_camera_to_body(t_unit_cam, self._R_bc)
            # Rotate visual direction into world frame using previous orientation
            d_unit_world = rotate_vector(q_prev, t_unit_body)

            # IMU displacement increment
            delta_p_imu = p_prop - p_prev

            # Project IMU displacement onto visual motion direction to reduce drift orthogonal to visual ray
            proj_scale = float(np.dot(delta_p_imu, d_unit_world))

            beta_dir = self.vio_config.visual_direction_weight
            if proj_scale > 0.0 and self.vio_config.use_imu_scale:
                delta_p_constrained = (1.0 - beta_dir) * delta_p_imu + beta_dir * (proj_scale * d_unit_world)
            else:
                delta_p_constrained = delta_p_imu

            p_fused = p_prev + delta_p_constrained
            v_fused = delta_p_constrained / delta_t

            self._status = VIOStatus.TRACKING_OK
            new_features_needed = len(pts_tracked_curr) < (self.frontend.max_features // 2)
        else:
            # Degraded mode: fallback to pure IMU dead-reckoning propagation
            p_fused = p_prop
            v_fused = v_prop
            q_fused = q_prop
            self._status = VIOStatus.DEGRADED_IMU_ONLY
            new_features_needed = True

        # Prepare features for the next camera frame
        if new_features_needed:
            self._prev_pts = self.frontend.detect_features(curr_gray)
        else:
            self._prev_pts = pts_tracked_curr

        self._prev_gray = curr_gray

        # Assemble and record new NavigationState
        updated_state = NavigationState(
            timestamp=t_curr,
            position=(float(p_fused[0]), float(p_fused[1]), float(p_fused[2])),
            velocity=(float(v_fused[0]), float(v_fused[1]), float(v_fused[2])),
            orientation=(float(q_fused[0]), float(q_fused[1]), float(q_fused[2]), float(q_fused[3])),
        )

        self._current_state = updated_state
        self._trajectory.append(updated_state)
        return updated_state

    def process_frame(self, frame: CameraFrame) -> Optional[NavigationState]:
        """Phase 1 compatibility adapter: process a single camera frame."""
        if not isinstance(frame, CameraFrame):
            raise TypeError("Expected CameraFrame instance")
        if not self._is_initialized:
            return self.initialize(frame.timestamp, frame)
        # Create minimal SynchronizedMeasurement with empty IMU for backward compatibility
        meas = SynchronizedMeasurement(
            camera_frame=frame,
            imu_samples=[],
            start_timestamp=self._current_state.timestamp if self._current_state else None,
            end_timestamp=frame.timestamp,
        )
        return self.process_measurement(meas)

    def process_imu(self, sample: IMUSample) -> Optional[NavigationState]:
        """Phase 1 compatibility adapter: process a single IMU sample."""
        if not isinstance(sample, IMUSample):
            raise TypeError("Expected IMUSample instance")
        if not self._is_initialized or self._current_state is None:
            return None

        dt = 0.005  # Nominal step
        p = np.asarray(self._current_state.position)
        v = np.asarray(self._current_state.velocity)
        q = np.asarray(self._current_state.orientation)

        p_next, v_next, q_next = self.propagator.integrate_sample_step(p, v, q, sample, dt)
        t_next = self._current_state.timestamp + dt

        new_state = NavigationState(
            timestamp=t_next,
            position=tuple(p_next),
            velocity=tuple(v_next),
            orientation=tuple(q_next),
        )
        self._current_state = new_state
        self._trajectory.append(new_state)
        return new_state

    def get_state(self) -> Optional[NavigationState]:
        """Retrieve the latest estimated vehicle navigation state."""
        return self._current_state

    def get_trajectory(self) -> List[NavigationState]:
        """Retrieve the sequence of estimated navigation states."""
        return list(self._trajectory)

    def reset(self) -> None:
        """Reset internal pipeline state to uninitialized."""
        self._is_initialized = False
        self._attitude_aligned = False
        self._initial_imu_buffer.clear()
        self._status = VIOStatus.UNINITIALIZED
        self._current_state = None
        self._trajectory.clear()
        self._prev_gray = None
        self._prev_pts = None
        self._last_visual_result = None
