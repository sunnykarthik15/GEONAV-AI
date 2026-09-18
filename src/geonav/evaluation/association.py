"""Deterministic timestamp association between estimated and ground-truth trajectories."""

from typing import List, Sequence, Tuple
import numpy as np

from geonav.evaluation.types import AssociatedPair, TrajectoryPoint


def associate_timestamps(
    est_traj: Sequence[TrajectoryPoint],
    gt_traj: Sequence[TrajectoryPoint],
    max_time_diff_s: float = 0.01,
) -> Tuple[List[AssociatedPair], int, int]:
    """Deterministically associate estimated poses with nearest ground-truth poses.

    Uses binary search nearest-neighbor matching. Rejects associations where the absolute
    time difference exceeds max_time_diff_s.

    Args:
        est_traj: Chronologically ordered sequence of estimated TrajectoryPoint objects.
        gt_traj: Chronologically ordered sequence of ground-truth TrajectoryPoint objects.
        max_time_diff_s: Maximum allowable time difference in seconds for association.

    Returns:
        Tuple[List[AssociatedPair], int, int]:
            - List of matched AssociatedPair objects.
            - Count of unmatched estimated poses.
            - Count of unmatched ground-truth poses.

    Raises:
        ValueError: If max_time_diff_s <= 0 or trajectories contain out-of-order timestamps.
    """
    if max_time_diff_s <= 0.0 or not np.isfinite(max_time_diff_s):
        raise ValueError(f"max_time_diff_s must be strictly positive and finite, got {max_time_diff_s}")

    if not est_traj or not gt_traj:
        return [], len(est_traj), len(gt_traj)

    est_times = np.array([pt.timestamp for pt in est_traj], dtype=np.float64)
    gt_times = np.array([pt.timestamp for pt in gt_traj], dtype=np.float64)

    # Monotonicity checks
    if np.any(np.diff(est_times) < 0.0):
        raise ValueError("Estimated trajectory timestamps must be monotonically non-decreasing")
    if np.any(np.diff(gt_times) < 0.0):
        raise ValueError("Ground-truth trajectory timestamps must be monotonically non-decreasing")

    matched_pairs: List[AssociatedPair] = []
    matched_gt_indices = set()
    unmatched_est_count = 0

    num_gt = len(gt_times)

    for i, est_pt in enumerate(est_traj):
        t_est = est_times[i]

        # Find nearest index in gt_times via searchsorted
        idx = int(np.searchsorted(gt_times, t_est))

        candidates = []
        if idx < num_gt:
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if idx + 1 < num_gt:
            candidates.append(idx + 1)

        best_idx = None
        best_diff = float("inf")

        for c in candidates:
            diff = abs(gt_times[c] - t_est)
            if diff < best_diff:
                best_diff = diff
                best_idx = c

        if best_idx is not None and best_diff <= max_time_diff_s:
            gt_pt = gt_traj[best_idx]
            matched_pairs.append(
                AssociatedPair(
                    timestamp_est=t_est,
                    timestamp_gt=gt_times[best_idx],
                    time_diff=best_diff,
                    est_point=est_pt,
                    gt_point=gt_pt,
                )
            )
            matched_gt_indices.add(best_idx)
        else:
            unmatched_est_count += 1

    unmatched_gt_count = num_gt - len(matched_gt_indices)
    return matched_pairs, unmatched_est_count, unmatched_gt_count
