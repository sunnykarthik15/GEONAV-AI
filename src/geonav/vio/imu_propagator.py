"""Inertial state propagation and kinematic integration for Visual-Inertial Odometry."""

from collections.abc import Sequence
import math
from typing import Optional, Tuple, Union
import numpy as np

from geonav.sensors.imu import IMUSample
from geonav.vio.geometry import (
    quaternion_multiply,
    quaternion_normalize,
    rotate_vector,
)
from geonav.vio.types import IMUPropagationResult


class IMUPropagator:
    """Propagates vehicle orientation, velocity, and position by integrating high-rate IMU measurements.

    Coordinate & Gravity Convention:
        World frame: NED (North-East-Down).
        Gravity vector: g_world = [0, 0, +g] in NED.
        Accelerometer specific force model:
            a_kinematic = R(q) * a_body + g_world
        When stationary and level, a resting accelerometer on a table measures reaction
        force upward (in body FRD: a_body = [0, 0, -g]), yielding a_kinematic = [0, 0, 0].
    """

    def __init__(
        self,
        gravity_magnitude: float = 9.81,
        gravity_vector: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> None:
        """Initialize IMU propagator with gravity parameters.

        Args:
            gravity_magnitude: Gravitational acceleration in m/s^2.
            gravity_vector: Optional 3D gravity vector in world coordinates (defaults to [0, 0, +g]).
        """
        self.gravity_magnitude = float(gravity_magnitude)
        if gravity_vector is not None:
            self.g_world = np.asarray(gravity_vector, dtype=np.float64)
            if self.g_world.shape != (3,):
                raise ValueError("gravity_vector must have shape (3,)")
        else:
            self.g_world = np.array([0.0, 0.0, self.gravity_magnitude], dtype=np.float64)

    def integrate_sample_step(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        sample: IMUSample,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform a single Euler/kinematic propagation step across interval dt.

        Args:
            p: 3D position vector in world frame (meters).
            v: 3D linear velocity vector in world frame (m/s).
            q: Unit quaternion (qw, qx, qy, qz) representing rotation from body to world frame.
            sample: IMUSample containing angular velocity and linear acceleration.
            dt: Time step in seconds (must be strictly positive).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (p_next, v_next, q_next).

        Raises:
            ValueError: If dt <= 0 or sensor values are non-finite.
        """
        if dt <= 0.0:
            raise ValueError(f"Integration time step dt must be strictly positive, got dt={dt}")
        if math.isnan(dt) or math.isinf(dt):
            raise ValueError(f"Non-finite integration time step dt: {dt}")

        omega = np.asarray(sample.angular_velocity, dtype=np.float64)
        accel_body = np.asarray(sample.linear_acceleration, dtype=np.float64)

        if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(accel_body)):
            raise ValueError("IMU sample contains non-finite angular velocity or acceleration values")

        # 1. Orientation update: compute incremental quaternion delta_q
        omega_norm = np.linalg.norm(omega)
        angle = omega_norm * dt

        if angle > 1e-8:
            axis = omega / omega_norm
            half_angle = angle * 0.5
            delta_q = np.array(
                [math.cos(half_angle), *(axis * math.sin(half_angle))],
                dtype=np.float64,
            )
        else:
            # Numerically stable Taylor expansion for small rotation angles
            delta_q = np.array(
                [1.0, 0.5 * omega[0] * dt, 0.5 * omega[1] * dt, 0.5 * omega[2] * dt],
                dtype=np.float64,
            )

        q_next = quaternion_multiply(q, delta_q)
        q_next = quaternion_normalize(q_next)

        # 2. Acceleration transformation to world frame and gravity compensation
        a_world = rotate_vector(q, accel_body) + self.g_world

        # 3. Kinematic velocity and position integration
        v_next = v + a_world * dt
        p_next = p + v * dt + 0.5 * a_world * (dt**2)

        return p_next, v_next, q_next

    def propagate_interval(
        self,
        initial_p: Union[Sequence[float], np.ndarray],
        initial_v: Union[Sequence[float], np.ndarray],
        initial_q: Union[Sequence[float], np.ndarray],
        imu_samples: Sequence[IMUSample],
    ) -> Tuple[IMUPropagationResult, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Propagate state across a discrete sequence of IMU samples.

        Args:
            initial_p: Initial position vector (3,).
            initial_v: Initial velocity vector (3,).
            initial_q: Initial unit quaternion (4,).
            imu_samples: Sequence of chronological IMUSample objects.

        Returns:
            Tuple[IMUPropagationResult, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
                (propagation_result, (final_p, final_v, final_q)).
        """
        p_curr = np.asarray(initial_p, dtype=np.float64)
        v_curr = np.asarray(initial_v, dtype=np.float64)
        q_curr = quaternion_normalize(initial_q)

        if not imu_samples or len(imu_samples) == 0:
            zero_dp = np.zeros(3, dtype=np.float64)
            zero_dv = np.zeros(3, dtype=np.float64)
            ident_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            res = IMUPropagationResult(
                success=True,
                delta_p=zero_dp,
                delta_v=zero_dv,
                delta_q=ident_q,
                dt=0.0,
                status_message="Zero IMU samples in interval",
            )
            return res, (p_curr, v_curr, q_curr)

        total_dt = 0.0
        p_start = p_curr.copy()
        v_start = v_curr.copy()
        q_start = q_curr.copy()

        # Iterate through consecutive samples to determine dt
        for i in range(len(imu_samples) - 1):
            s_curr = imu_samples[i]
            s_next = imu_samples[i + 1]

            dt = s_next.timestamp - s_curr.timestamp
            if dt <= 0.0:
                raise ValueError(
                    f"Non-positive or non-increasing IMU time step at index {i}: "
                    f"t_curr={s_curr.timestamp}, t_next={s_next.timestamp}, dt={dt}"
                )

            p_curr, v_curr, q_curr = self.integrate_sample_step(
                p_curr, v_curr, q_curr, s_curr, dt
            )
            total_dt += dt

        # For the final sample, if more than 1 sample, assume interval equal to preceding dt
        if len(imu_samples) == 1:
            # Single sample: minimal dt cannot be inferred from next sample
            dt = 0.005  # Nominal ~200Hz single sample step
            p_curr, v_curr, q_curr = self.integrate_sample_step(
                p_curr, v_curr, q_curr, imu_samples[0], dt
            )
            total_dt = dt

        delta_p = p_curr - p_start
        delta_v = v_curr - v_start
        # delta_q = q_start_inv * q_curr
        q_start_inv = np.array([q_start[0], -q_start[1], -q_start[2], -q_start[3]])
        delta_q = quaternion_multiply(q_start_inv, q_curr)

        res = IMUPropagationResult(
            success=True,
            delta_p=delta_p,
            delta_v=delta_v,
            delta_q=delta_q,
            dt=total_dt,
            status_message=f"Propagated across {len(imu_samples)} IMU samples",
        )
        return res, (p_curr, v_curr, q_curr)
