"""Trajectory evaluation algorithms: ATE, Axis Errors, RPE, Orientation, and Scale."""

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np

from geonav.evaluation.types import (
    ATEResult,
    AxisErrors,
    OrientationErrorResult,
    RPEIntervalResult,
    ScaleAnalysisResult,
    TrajectoryPoint,
    VelocityErrorResult,
)
from geonav.vio.geometry import quaternion_to_rotation_matrix


def compute_ate(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> ATEResult:
    """Compute Absolute Trajectory Error (ATE) metrics between aligned/matched position vectors.

    Args:
        est_positions: (N, 3) array of estimated 3D positions in meters.
        gt_positions: (N, 3) array of ground-truth 3D positions in meters.

    Returns:
        ATEResult: Computed RMSE, mean, median, min, max, and final error.

    Raises:
        ValueError: If array shapes mismatch, are empty, or contain non-finite values.
    """
    est = np.asarray(est_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)

    if est.shape != gt.shape:
        raise ValueError(f"Shape mismatch in ATE calculation: {est.shape} vs {gt.shape}")
    if len(est.shape) != 2 or est.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) position arrays, got {est.shape}")
    if len(est) == 0:
        raise ValueError("Cannot compute ATE on empty trajectory arrays")
    if not (np.all(np.isfinite(est)) and np.all(np.isfinite(gt))):
        raise ValueError("Position arrays contain non-finite values")

    errors = np.linalg.norm(est - gt, axis=1)

    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    std_err = float(np.std(errors))
    min_err = float(np.min(errors))
    max_err = float(np.max(errors))
    final_err = float(errors[-1])

    final_est = (float(est[-1, 0]), float(est[-1, 1]), float(est[-1, 2]))
    final_gt = (float(gt[-1, 0]), float(gt[-1, 1]), float(gt[-1, 2]))

    return ATEResult(
        rmse=rmse,
        mean=mean_err,
        median=median_err,
        std=std_err,
        min=min_err,
        max=max_err,
        final_error=final_err,
        final_estimated_position=final_est,
        final_groundtruth_position=final_gt,
    )


def compute_axis_errors(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> AxisErrors:
    """Compute component-wise translational errors along X, Y, and Z axes.

    Args:
        est_positions: (N, 3) array of estimated positions.
        gt_positions: (N, 3) array of ground-truth positions.

    Returns:
        AxisErrors: Component RMSE and Mean Absolute Error (MAE).
    """
    est = np.asarray(est_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)

    diffs = est - gt
    abs_diffs = np.abs(diffs)

    x_rmse = float(np.sqrt(np.mean(diffs[:, 0] ** 2)))
    y_rmse = float(np.sqrt(np.mean(diffs[:, 1] ** 2)))
    z_rmse = float(np.sqrt(np.mean(diffs[:, 2] ** 2)))

    x_mae = float(np.mean(abs_diffs[:, 0]))
    y_mae = float(np.mean(abs_diffs[:, 1]))
    z_mae = float(np.mean(abs_diffs[:, 2]))

    x_max = float(np.max(abs_diffs[:, 0]))
    y_max = float(np.max(abs_diffs[:, 1]))
    z_max = float(np.max(abs_diffs[:, 2]))

    return AxisErrors(
        x_rmse=x_rmse,
        y_rmse=y_rmse,
        z_rmse=z_rmse,
        x_mae=x_mae,
        y_mae=y_mae,
        z_mae=z_mae,
        x_max=x_max,
        y_max=y_max,
        z_max=z_max,
    )


def compute_rpe(
    est_points: Sequence[TrajectoryPoint],
    gt_points: Sequence[TrajectoryPoint],
    interval_frames: Optional[int] = None,
    interval_time_s: Optional[float] = None,
    interval_name: str = "interval",
) -> RPEIntervalResult:
    """Compute Relative Pose Error (RPE) over a specified frame or time interval.

    Args:
        est_points: Sequence of matched estimated TrajectoryPoint objects.
        gt_points: Sequence of matched ground-truth TrajectoryPoint objects.
        interval_frames: Step in indices between compared pairs.
        interval_time_s: Time step in seconds between compared pairs (approximate search).
        interval_name: Human-readable identifier for the interval.

    Returns:
        RPEIntervalResult: Translational and rotational relative error metrics.
    """
    n = len(est_points)
    if n != len(gt_points):
        raise ValueError(f"Trajectory length mismatch: {n} vs {len(gt_points)}")
    if n < 2:
        return RPEIntervalResult(
            interval_name=interval_name,
            delta_frames=interval_frames,
            delta_time_s=interval_time_s,
            num_pairs=0,
            trans_rmse=0.0,
            trans_mean=0.0,
            trans_median=0.0,
            trans_max=0.0,
            rot_rmse_deg=0.0,
            rot_mean_deg=0.0,
            rot_median_deg=0.0,
            rot_max_deg=0.0,
        )

    trans_errors = []
    rot_errors_deg = []

    times = np.array([pt.timestamp for pt in est_points], dtype=np.float64)

    if interval_time_s is not None and interval_time_s > 0.0:
        # Time-based interval pairs
        for i in range(n):
            target_t = times[i] + interval_time_s
            if target_t > times[-1]:
                break
            j = int(np.searchsorted(times, target_t))
            if j < n and abs(times[j] - target_t) < 0.1:  # Within 100ms
                e_pt0, e_pt1 = est_points[i], est_points[j]
                g_pt0, g_pt1 = gt_points[i], gt_points[j]

                # Relative translation
                delta_p_est = e_pt1.position - e_pt0.position
                delta_p_gt = g_pt1.position - g_pt0.position
                trans_errors.append(float(np.linalg.norm(delta_p_est - delta_p_gt)))

                # Relative rotation
                R_e0 = quaternion_to_rotation_matrix(e_pt0.orientation)
                R_e1 = quaternion_to_rotation_matrix(e_pt1.orientation)
                R_g0 = quaternion_to_rotation_matrix(g_pt0.orientation)
                R_g1 = quaternion_to_rotation_matrix(g_pt1.orientation)

                delta_R_est = R_e0.T @ R_e1
                delta_R_gt = R_g0.T @ R_g1

                R_err = delta_R_gt.T @ delta_R_est
                cos_ang = min(max((np.trace(R_err) - 1.0) * 0.5, -1.0), 1.0)
                rot_errors_deg.append(math.degrees(math.acos(cos_ang)))
    else:
        # Frame-based interval pairs
        step = interval_frames if (interval_frames is not None and interval_frames > 0) else 1
        for i in range(n - step):
            j = i + step
            e_pt0, e_pt1 = est_points[i], est_points[j]
            g_pt0, g_pt1 = gt_points[i], gt_points[j]

            delta_p_est = e_pt1.position - e_pt0.position
            delta_p_gt = g_pt1.position - g_pt0.position
            trans_errors.append(float(np.linalg.norm(delta_p_est - delta_p_gt)))

            R_e0 = quaternion_to_rotation_matrix(e_pt0.orientation)
            R_e1 = quaternion_to_rotation_matrix(e_pt1.orientation)
            R_g0 = quaternion_to_rotation_matrix(g_pt0.orientation)
            R_g1 = quaternion_to_rotation_matrix(g_pt1.orientation)

            delta_R_est = R_e0.T @ R_e1
            delta_R_gt = R_g0.T @ R_g1

            R_err = delta_R_gt.T @ delta_R_est
            cos_ang = min(max((np.trace(R_err) - 1.0) * 0.5, -1.0), 1.0)
            rot_errors_deg.append(math.degrees(math.acos(cos_ang)))

    if not trans_errors:
        return RPEIntervalResult(
            interval_name=interval_name,
            delta_frames=interval_frames,
            delta_time_s=interval_time_s,
            num_pairs=0,
            trans_rmse=0.0,
            trans_mean=0.0,
            trans_median=0.0,
            trans_max=0.0,
            rot_rmse_deg=0.0,
            rot_mean_deg=0.0,
            rot_median_deg=0.0,
            rot_max_deg=0.0,
        )

    t_arr = np.array(trans_errors, dtype=np.float64)
    r_arr = np.array(rot_errors_deg, dtype=np.float64)

    return RPEIntervalResult(
        interval_name=interval_name,
        delta_frames=interval_frames,
        delta_time_s=interval_time_s,
        num_pairs=len(t_arr),
        trans_rmse=float(np.sqrt(np.mean(t_arr**2))),
        trans_mean=float(np.mean(t_arr)),
        trans_median=float(np.median(t_arr)),
        trans_max=float(np.max(t_arr)),
        rot_rmse_deg=float(np.sqrt(np.mean(r_arr**2))),
        rot_mean_deg=float(np.mean(r_arr)),
        rot_median_deg=float(np.median(r_arr)),
        rot_max_deg=float(np.max(r_arr)),
    )


def compute_orientation_errors(
    est_orientations: np.ndarray,
    gt_orientations: np.ndarray,
) -> OrientationErrorResult:
    """Compute absolute angular orientation error between estimated and ground-truth quaternions.

    Args:
        est_orientations: (N, 4) array of unit quaternions [qw, qx, qy, qz].
        gt_orientations: (N, 4) array of ground-truth unit quaternions.

    Returns:
        OrientationErrorResult: RMSE, mean, median, and maximum error in degrees.
    """
    est = np.asarray(est_orientations, dtype=np.float64)
    gt = np.asarray(gt_orientations, dtype=np.float64)

    if est.shape != gt.shape:
        raise ValueError(f"Orientation shape mismatch: {est.shape} vs {gt.shape}")

    errors_deg = []
    for q_e, q_g in zip(est, gt):
        R_e = quaternion_to_rotation_matrix(q_e)
        R_g = quaternion_to_rotation_matrix(q_g)

        R_err = R_g.T @ R_e
        cos_ang = min(max((np.trace(R_err) - 1.0) * 0.5, -1.0), 1.0)
        errors_deg.append(math.degrees(math.acos(cos_ang)))

    err_arr = np.array(errors_deg, dtype=np.float64)
    return OrientationErrorResult(
        rmse_deg=float(np.sqrt(np.mean(err_arr**2))),
        mean_deg=float(np.mean(err_arr)),
        median_deg=float(np.median(err_arr)),
        max_deg=float(np.max(err_arr)),
    )


def compute_velocity_errors(
    est_velocities: np.ndarray,
    gt_velocities: np.ndarray,
) -> VelocityErrorResult:
    """Compute linear velocity error metrics.

    Args:
        est_velocities: (N, 3) array of estimated velocities in m/s.
        gt_velocities: (N, 3) array of ground-truth velocities in m/s.

    Returns:
        VelocityErrorResult: Overall RMSE, axis-wise RMSEs, mean, max, and final error.
    """
    est = np.asarray(est_velocities, dtype=np.float64)
    gt = np.asarray(gt_velocities, dtype=np.float64)

    if est.shape != gt.shape:
        raise ValueError(f"Velocity shape mismatch: {est.shape} vs {gt.shape}")

    diffs = est - gt
    norms = np.linalg.norm(diffs, axis=1)

    rmse = float(np.sqrt(np.mean(norms**2)))
    x_rmse = float(np.sqrt(np.mean(diffs[:, 0] ** 2)))
    y_rmse = float(np.sqrt(np.mean(diffs[:, 1] ** 2)))
    z_rmse = float(np.sqrt(np.mean(diffs[:, 2] ** 2)))
    mean_err = float(np.mean(norms))
    max_err = float(np.max(norms))
    final_err = float(norms[-1])

    return VelocityErrorResult(
        rmse=rmse,
        x_rmse=x_rmse,
        y_rmse=y_rmse,
        z_rmse=z_rmse,
        mean=mean_err,
        max=max_err,
        final_error=final_err,
    )


def compute_scale_analysis(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> ScaleAnalysisResult:
    """Calculate cumulative path lengths and descriptive scale ratio.

    Args:
        est_positions: (N, 3) array of estimated positions.
        gt_positions: (M, 3) array of ground-truth positions.

    Returns:
        ScaleAnalysisResult: Estimated length, ground-truth length, and scale ratio.
    """
    est = np.asarray(est_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)

    diffs_est = np.diff(est, axis=0) if len(est) > 1 else np.zeros((1, 3))
    diffs_gt = np.diff(gt, axis=0) if len(gt) > 1 else np.zeros((1, 3))

    len_est = float(np.sum(np.linalg.norm(diffs_est, axis=1)))
    len_gt = float(np.sum(np.linalg.norm(diffs_gt, axis=1)))

    scale_ratio = len_est / len_gt if len_gt > 1e-6 else 1.0

    return ScaleAnalysisResult(
        estimated_path_length=len_est,
        groundtruth_path_length=len_gt,
        scale_ratio=scale_ratio,
    )
