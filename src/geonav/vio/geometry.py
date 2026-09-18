"""Quaternion and 3D spatial rotation utilities for Visual-Inertial Odometry.

Quaternion convention throughout GEONAV-AI:
    q = (qw, qx, qy, qz)
where qw is the scalar (real) component and (qx, qy, qz) is the vector (imaginary) component.
Hamilton product convention is enforced.
"""

import math
from typing import Sequence, Tuple, Union
import numpy as np


def quaternion_normalize(q: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Args:
        q: 4-element sequence or array (qw, qx, qy, qz).

    Returns:
        np.ndarray: Unit-norm quaternion of shape (4,).

    Raises:
        ValueError: If input length is not 4, values are non-finite, or norm is zero.
    """
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Quaternion contains non-finite values: {arr}")

    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize zero-norm quaternion: norm={norm}")

    return arr / norm


def quaternion_multiply(
    q1: Union[Sequence[float], np.ndarray],
    q2: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Compute the Hamilton product of two quaternions: q = q1 ⊗ q2.

    Args:
        q1: Left quaternion (qw1, qx1, qy1, qz1).
        q2: Right quaternion (qw2, qx2, qy2, qz2).

    Returns:
        np.ndarray: Resulting unit-norm quaternion of shape (4,).
    """
    q1_arr = quaternion_normalize(q1)
    q2_arr = quaternion_normalize(q2)

    w1, x1, y1, z1 = q1_arr
    w2, x2, y2, z2 = q2_arr

    qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    prod = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quaternion_normalize(prod)


def quaternion_to_rotation_matrix(q: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert a unit quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix R.

    The rotation matrix transforms vectors from body frame to world frame: v_world = R * v_body.

    Args:
        q: 4-element sequence or array representing a unit quaternion.

    Returns:
        np.ndarray: 3x3 orthogonal rotation matrix.
    """
    q_norm = quaternion_normalize(q)
    w, x, y, z = q_norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_quaternion(R: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    """Convert a 3x3 orthogonal rotation matrix R to a unit quaternion (qw, qx, qy, qz).

    Uses Shepperd's algorithm for numerical stability across all rotation angles.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        np.ndarray: Unit-norm quaternion of shape (4,).

    Raises:
        ValueError: If matrix shape is not (3, 3) or contains non-finite values.
    """
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {R_arr.shape}")
    if not np.all(np.isfinite(R_arr)):
        raise ValueError("Rotation matrix contains non-finite values")

    trace = R_arr[0, 0] + R_arr[1, 1] + R_arr[2, 2]

    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R_arr[2, 1] - R_arr[1, 2]) * s
        qy = (R_arr[0, 2] - R_arr[2, 0]) * s
        qz = (R_arr[1, 0] - R_arr[0, 1]) * s
    elif (R_arr[0, 0] > R_arr[1, 1]) and (R_arr[0, 0] > R_arr[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R_arr[0, 0] - R_arr[1, 1] - R_arr[2, 2])
        qw = (R_arr[2, 1] - R_arr[1, 2]) / s
        qx = 0.25 * s
        qy = (R_arr[0, 1] + R_arr[1, 0]) / s
        qz = (R_arr[0, 2] + R_arr[2, 0]) / s
    elif R_arr[1, 1] > R_arr[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R_arr[1, 1] - R_arr[0, 0] - R_arr[2, 2])
        qw = (R_arr[0, 2] - R_arr[2, 0]) / s
        qx = (R_arr[0, 1] + R_arr[1, 0]) / s
        qy = 0.25 * s
        qz = (R_arr[1, 2] + R_arr[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R_arr[2, 2] - R_arr[0, 0] - R_arr[1, 1])
        qw = (R_arr[1, 0] - R_arr[0, 1]) / s
        qx = (R_arr[0, 2] + R_arr[2, 0]) / s
        qy = (R_arr[1, 2] + R_arr[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # Canonical representation: ensure scalar part is non-negative
    if q[0] < 0.0:
        q = -q
    return quaternion_normalize(q)


def rotate_vector(
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Rotate a 3D vector v by quaternion q: v_rot = R(q) * v.

    Args:
        q: Unit quaternion (qw, qx, qy, qz).
        v: 3D vector (x, y, z).

    Returns:
        np.ndarray: Rotated vector of shape (3,).

    Raises:
        ValueError: If v does not have shape (3,) or contains non-finite values.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    if v_arr.shape != (3,):
        raise ValueError(f"Expected 3-element vector, got shape {v_arr.shape}")
    if not np.all(np.isfinite(v_arr)):
        raise ValueError("Vector contains non-finite values")

    q_norm = quaternion_normalize(q)
    w = q_norm[0]
    u = q_norm[1:]  # vector part

    # Rodrigues / quaternion vector rotation formula:
    # v_rot = v + 2*u x (u x v + w*v)
    uv = np.cross(u, v_arr)
    uuv = np.cross(u, uv)
    return v_arr + 2.0 * (w * uv + uuv)


def quaternion_slerp(
    q0: Union[Sequence[float], np.ndarray],
    q1: Union[Sequence[float], np.ndarray],
    t: float,
) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Args:
        q0: Initial unit quaternion (t=0.0).
        q1: Target unit quaternion (t=1.0).
        t: Interpolation factor in [0.0, 1.0].

    Returns:
        np.ndarray: Interpolated unit quaternion.
    """
    q0_arr = quaternion_normalize(q0)
    q1_arr = quaternion_normalize(q1)

    dot = float(np.dot(q0_arr, q1_arr))

    # If dot is negative, slerp takes the long way around the sphere.
    # Negate one quaternion to take the shortest path.
    if dot < 0.0:
        q1_arr = -q1_arr
        dot = -dot

    # Clamp dot product to prevent NaN from acos
    dot = min(max(dot, -1.0), 1.0)

    if dot > 0.9995:
        # Quaternions are very close; linear interpolation avoids division by zero
        result = q0_arr + t * (q1_arr - q0_arr)
        return quaternion_normalize(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0_arr + s1 * q1_arr
    return quaternion_normalize(result)


def align_gravity(
    accel_samples: Union[Sequence[object], np.ndarray],
    min_samples: int = 1,
) -> np.ndarray:
    """Estimate initial vehicle attitude (roll and pitch) from stationary accelerometer measurements.

    Mathematical Derivation:
        World / Navigation Frame:
            NED convention: X = North, Y = East, Z = Down.
            Gravity vector: g_world = [0, 0, +g]^T where g = 9.81 m/s^2 pointing Down (+Z).

        Body / IMU Frame:
            X = Forward, Y = Right, Z = Down.

        Specific Force Measurement:
            The IMU accelerometer measures specific force f_body:
                f_body = a_kinematic - R(q_{wb})^T * g_world
            At rest, a_kinematic = [0, 0, 0]^T, so:
                f_meas ≈ -R(q_{wb})^T * g_world

        Gravity Direction in Body Coordinates:
            Since g_world = [0, 0, g]^T:
                R(q_{wb})^T * [0, 0, 1]^T = -f_meas / ||f_meas||
            Let u = -f_meas / ||f_meas|| = [u_x, u_y, u_z]^T.
            Here u is the unit vector pointing in the direction of world Down (+Z)
            expressed in the body frame coordinates.

        Euler Angle Parameterization (Z-Y-X Tait-Bryan convention):
            With yaw ψ = 0 (unobservable from gravity alone):
                R_{wb}(roll=φ, pitch=θ, yaw=0) = R_y(θ) * R_x(φ)
            The 3rd column of R_{wb} (which equals R_{wb}^T * [0, 0, 1]^T) is:
                [-sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]^T = [u_x, u_y, u_z]^T
            Therefore:
                sin(θ) = -u_x  ==>  θ = arctan2(-u_x, sqrt(u_y^2 + u_z^2))
                tan(φ) = u_y / u_z  ==>  φ = arctan2(u_y, u_z)
                ψ = 0.0

        Quaternion Representation (Hamilton scalar-first [w, x, y, z]):
            q_{wb} = q_y(θ) ⊗ q_x(φ):
                w = cos(θ/2) * cos(φ/2)
                x = cos(θ/2) * sin(φ/2)
                y = sin(θ/2) * cos(φ/2)
                z = -sin(θ/2) * sin(φ/2)

    Args:
        accel_samples: Sequence of IMUSample objects or 3D acceleration vectors/tuples.
        min_samples: Minimum number of valid samples required (default: 1).

    Returns:
        np.ndarray: Unit quaternion [w, x, y, z] leveling the body frame to the world frame.

    Raises:
        ValueError: If insufficient samples are provided, samples are non-finite,
                    or the mean accelerometer magnitude is near zero.
    """
    clean_accels = []
    if accel_samples is not None:
        for s in accel_samples:
            if hasattr(s, "linear_acceleration"):
                a = s.linear_acceleration
            elif hasattr(s, "__len__") and len(s) == 3:
                a = s
            else:
                continue

            try:
                ax, ay, az = float(a[0]), float(a[1]), float(a[2])
            except (ValueError, TypeError):
                continue

            if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az):
                clean_accels.append((ax, ay, az))

    if len(clean_accels) < min_samples:
        raise ValueError(
            f"Insufficient valid finite accelerometer samples: {len(clean_accels)} < {min_samples}"
        )

    mean_f = np.mean(clean_accels, axis=0)
    norm_f = float(np.linalg.norm(mean_f))
    if norm_f < 1e-4 or not math.isfinite(norm_f):
        raise ValueError(f"Accelerometer norm too close to zero or non-finite: {norm_f}")

    u = -mean_f / norm_f
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])

    theta = math.atan2(-ux, math.sqrt(uy**2 + uz**2))
    phi = math.atan2(uy, uz)

    w = math.cos(theta * 0.5) * math.cos(phi * 0.5)
    x = math.cos(theta * 0.5) * math.sin(phi * 0.5)
    y = math.sin(theta * 0.5) * math.cos(phi * 0.5)
    z = -math.sin(theta * 0.5) * math.sin(phi * 0.5)

    return quaternion_normalize(np.array([w, x, y, z], dtype=np.float64))


def validate_rotation_matrix(
    R: Union[Sequence, np.ndarray],
    tolerance: float = 1e-3,
) -> bool:
    """Validate that R is a real, finite, orthonormal 3x3 rotation matrix with det(R) ~ +1.

    Args:
        R: 3x3 array or sequence.
        tolerance: Numerical tolerance for orthonormality and determinant checks.

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If R is malformed, non-finite, not orthonormal, or has det != +1.
    """
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"Rotation matrix must have shape (3, 3), got {R_arr.shape}")
    if not np.all(np.isfinite(R_arr)):
        raise ValueError("Rotation matrix contains non-finite values (NaN or Inf)")

    # Check orthonormality: R * R^T ~ I
    diff = np.abs(R_arr @ R_arr.T - np.eye(3))
    if np.max(diff) > tolerance:
        raise ValueError(f"Matrix is not orthonormal: max |R*R^T - I| = {np.max(diff):.6e} > {tolerance}")

    # Check determinant ~ +1 (special orthogonal SO(3), no reflection)
    det = float(np.linalg.det(R_arr))
    if abs(det - 1.0) > tolerance:
        raise ValueError(f"Matrix determinant is not +1: det(R) = {det:.6f}")

    return True


def validate_extrinsics(
    T: Union[Sequence, np.ndarray],
    tolerance: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate 4x4 rigid transformation matrix T = [R | t; 0 0 0 1].

    Args:
        T: 4x4 array or sequence.
        tolerance: Numerical tolerance for rotation checks.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (R, t) where R is (3, 3) float64 array and t is (3,) float64 array.

    Raises:
        ValueError: If T is invalid, non-finite, or has non-affine bottom row.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    if T_arr.shape != (4, 4):
        raise ValueError(f"Extrinsic transformation matrix must have shape (4, 4), got {T_arr.shape}")
    if not np.all(np.isfinite(T_arr)):
        raise ValueError("Extrinsic matrix contains non-finite values (NaN or Inf)")

    # Check bottom row is [0, 0, 0, 1]
    if not np.allclose(T_arr[3, :], [0.0, 0.0, 0.0, 1.0], atol=tolerance):
        raise ValueError(f"Extrinsic matrix bottom row must be [0, 0, 0, 1], got {T_arr[3, :]}")

    R = T_arr[:3, :3]
    t = T_arr[:3, 3]
    validate_rotation_matrix(R, tolerance=tolerance)
    return R, t


def invert_transform(T: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Compute the exact inverse of a 4x4 rigid transformation matrix T = [R | p].

    Inverse formula:
        T^(-1) = [R^T | -R^T * p; 0 0 0 1]

    Args:
        T: 4x4 rigid transformation matrix.

    Returns:
        np.ndarray: Inverted 4x4 transformation matrix.
    """
    R, p = validate_extrinsics(T)
    T_inv = np.eye(4, dtype=np.float64)
    R_inv = R.T
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -R_inv @ p
    return T_inv


def transform_point_camera_to_body(
    p_cam: Union[Sequence[float], np.ndarray],
    T_bc: Union[Sequence, np.ndarray],
) -> np.ndarray:
    """Transform a 3D point from Camera coordinates (C) to Body/IMU coordinates (B).

    Formula:
        p_B = R_BC * p_C + p_BC

    Args:
        p_cam: 3D point in camera frame (x, y, z).
        T_bc: 4x4 extrinsic transformation matrix T_BC = [R_BC | p_BC].

    Returns:
        np.ndarray: 3D point in body frame of shape (3,).
    """
    R_bc, p_bc = validate_extrinsics(T_bc)
    p_arr = np.asarray(p_cam, dtype=np.float64)
    if p_arr.shape != (3,):
        raise ValueError(f"Point must have shape (3,), got {p_arr.shape}")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("Point contains non-finite values")
    return R_bc @ p_arr + p_bc


def transform_point_body_to_camera(
    p_body: Union[Sequence[float], np.ndarray],
    T_bc: Union[Sequence, np.ndarray],
) -> np.ndarray:
    """Transform a 3D point from Body/IMU coordinates (B) to Camera coordinates (C).

    Formula:
        p_C = R_BC^T * (p_B - p_BC)

    Args:
        p_body: 3D point in body frame (x, y, z).
        T_bc: 4x4 extrinsic transformation matrix T_BC = [R_BC | p_BC].

    Returns:
        np.ndarray: 3D point in camera frame of shape (3,).
    """
    R_bc, p_bc = validate_extrinsics(T_bc)
    p_arr = np.asarray(p_body, dtype=np.float64)
    if p_arr.shape != (3,):
        raise ValueError(f"Point must have shape (3,), got {p_arr.shape}")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("Point contains non-finite values")
    return R_bc.T @ (p_arr - p_bc)


def transform_direction_camera_to_body(
    d_cam: Union[Sequence[float], np.ndarray],
    R_bc: Union[Sequence, np.ndarray],
) -> np.ndarray:
    """Transform a 3D unit motion direction vector from Camera frame (C) to Body frame (B).

    Formula:
        d_B = R_BC * d_C

    Args:
        d_cam: 3D direction vector in camera frame.
        R_bc: 3x3 rotation matrix R_BC transforming vectors from camera to body.

    Returns:
        np.ndarray: Transformed unit direction vector in body frame of shape (3,).
    """
    validate_rotation_matrix(R_bc)
    d_arr = np.asarray(d_cam, dtype=np.float64)
    if d_arr.shape != (3,):
        raise ValueError(f"Direction vector must have shape (3,), got {d_arr.shape}")
    if not np.all(np.isfinite(d_arr)):
        raise ValueError("Direction vector contains non-finite values")

    d_body = np.asarray(R_bc, dtype=np.float64) @ d_arr
    norm = np.linalg.norm(d_body)
    if norm > 1e-12:
        d_body /= norm
    return d_body


def transform_relative_rotation_camera_to_body(
    R_c_rel: Union[Sequence, np.ndarray],
    R_bc: Union[Sequence, np.ndarray],
) -> np.ndarray:
    """Transform an inter-frame relative camera rotation R_c_rel into the Body frame.

    Derivation:
        Let R_c_rel = R_{C1, C0} transform vectors from camera at t0 (C0) to camera at t1 (C1):
            v_C1 = R_c_rel * v_C0
        Using v_C = R_BC^T * v_B:
            R_BC^T * v_B1 = R_c_rel * (R_BC^T * v_B0)
            v_B1 = (R_BC * R_c_rel * R_BC^T) * v_B0
        Therefore, the relative body rotation R_b_rel = R_{B1, B0} is:
            R_b_rel = R_BC * R_c_rel * R_BC^T

    Args:
        R_c_rel: 3x3 relative rotation matrix in camera frame.
        R_bc: 3x3 extrinsic camera-to-body rotation matrix R_BC.

    Returns:
        np.ndarray: 3x3 relative rotation matrix in body frame of shape (3, 3).
    """
    validate_rotation_matrix(R_c_rel)
    validate_rotation_matrix(R_bc)
    R_c = np.asarray(R_c_rel, dtype=np.float64)
    R_b = np.asarray(R_bc, dtype=np.float64)
    return R_b @ R_c @ R_b.T

