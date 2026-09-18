# GEONAV-AI — Phase 7 Feature Causality Audit

**Date:** 2026-09-19  
**Source:** `src/geonav/ml/features.py` — `FeatureExtractor.extract()`  
**Status:** ✅ PASS — All 18 features are strictly causal

---

## Overview

The ML model receives an 18-dimensional input vector at each camera frame. Every element must be available at the current inference time step without access to:
- Future camera frames
- Future IMU samples
- Ground-truth position, velocity, or orientation
- Post-processed trajectory statistics
- Absolute wall-clock timestamps

This audit inspects each feature individually.

---

## Feature Vector Definition

```python
# MLFeatures.to_array() output order:
# [0]   tracked_features
# [1]   inlier_ratio
# [2]   mean_flow_mag
# [3]   std_flow_mag
# [4]   mean_accel[0]  (x)
# [5]   mean_accel[1]  (y)
# [6]   mean_accel[2]  (z)
# [7]   var_accel[0]
# [8]   var_accel[1]
# [9]   var_accel[2]
# [10]  mean_gyro[0]
# [11]  mean_gyro[1]
# [12]  mean_gyro[2]
# [13]  vio_velocity[0]
# [14]  vio_velocity[1]
# [15]  vio_velocity[2]
# [16]  vio_v_norm
# [17]  dt
```

---

## Detailed Feature Audit

### Group 1: Visual Tracking Quality (Features 0–3)

**Source:** `VisualTrackingResult` from `VIOPipeline._last_visual_result`, which is set at the end of `process_measurement()` on the **current** frame.

---

#### Feature 0: `tracked_features`

| Property | Value |
|---|---|
| Name | `tracked_features` |
| Source | `visual_result.num_tracked` |
| Math | Count of features tracked from previous to current frame via Lucas-Kanade optical flow |
| Timestamp | Current frame t_k |
| Past data? | Yes — previous frame image is used as reference |
| Current data? | Yes — current frame |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| Post-processed? | ❌ No |
| **Causal?** | ✅ Yes |

---

#### Feature 1: `inlier_ratio`

| Property | Value |
|---|---|
| Name | `inlier_ratio` |
| Source | `num_inliers / num_tracked` |
| Math | Fraction of tracked features passing Essential matrix RANSAC |
| Timestamp | Current frame t_k |
| Past data? | Yes — uses previous frame keypoints |
| Current data? | Yes |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| Post-processed? | ❌ No |
| **Causal?** | ✅ Yes |

---

#### Feature 2: `mean_flow_mag`

| Property | Value |
|---|---|
| Name | `mean_flow_mag` |
| Source | `np.mean(np.linalg.norm(pts_curr - pts_prev, axis=1))` |
| Math | Mean Euclidean pixel displacement magnitude of tracked keypoints from t_{k-1} to t_k |
| Timestamp | Current frame t_k (uses pts from t_{k-1} and t_k) |
| Past data? | Yes — previous keypoint positions |
| Current data? | Yes — current keypoint positions |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| Post-processed? | ❌ No |
| **Causal?** | ✅ Yes |
| Note | Set to 0.0 in `evaluate.py` (pts_curr=None) since prev points are unavailable from external call; does not affect causality |

---

#### Feature 3: `std_flow_mag`

| Property | Value |
|---|---|
| Name | `std_flow_mag` |
| Source | `np.std(np.linalg.norm(pts_curr - pts_prev, axis=1))` |
| Math | Standard deviation of pixel displacement magnitudes |
| Timestamp | Current frame t_k |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| **Causal?** | ✅ Yes |

---

### Group 2: IMU Kinematic Features (Features 4–12)

**Source:** `measurement.imu_samples` — the list of IMU samples accumulated between t_{k-1} and t_k. These samples are associated with the inter-frame interval before the current frame is processed, using the synchronizer boundary policy `exclusive_left_inclusive_right`: `t_{k-1} < t_IMU <= t_k`.

No future IMU samples (t > t_k) are included.

---

#### Features 4–6: `mean_accel` (x, y, z)

| Property | Value |
|---|---|
| Name | `mean_accel[0,1,2]` |
| Source | `np.mean(accels, axis=0)` over inter-frame IMU samples |
| Math | Per-axis mean of raw accelerometer readings (m/s²) over (t_{k-1}, t_k] |
| Timestamp | All samples within current inter-frame interval |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| Gravity-removed? | ❌ No — raw accelerometer readings (gravity + specific force) |
| **Causal?** | ✅ Yes |

---

#### Features 7–9: `var_accel` (x, y, z)

| Property | Value |
|---|---|
| Name | `var_accel[0,1,2]` |
| Source | `np.var(accels, axis=0)` |
| Math | Per-axis variance of accelerometer readings — captures vibration / dynamic maneuver intensity |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| **Causal?** | ✅ Yes |

---

#### Features 10–12: `mean_gyro` (x, y, z)

| Property | Value |
|---|---|
| Name | `mean_gyro[0,1,2]` |
| Source | `np.mean(gyros, axis=0)` |
| Math | Per-axis mean angular velocity (rad/s) over inter-frame interval |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| **Causal?** | ✅ Yes |

---

### Group 3: VIO Navigation State Features (Features 13–16)

**Source:** `current_state` argument — the `NavigationState` returned by `VIOPipeline.process_measurement()` for the **current** frame. This is the classical VIO estimate, not GT.

---

#### Features 13–15: `vio_velocity` (x, y, z)

| Property | Value |
|---|---|
| Name | `vio_velocity[0,1,2]` |
| Source | `current_state.velocity` |
| Math | 3D velocity estimate in world frame (m/s) from classical VIO at t_k |
| Timestamp | Current frame t_k |
| Future data? | ❌ No |
| Uses GT? | ❌ No — this is the VIO estimate, not GT |
| Post-processed? | ❌ No — this is the raw pipeline output before correction |
| **Causal?** | ✅ Yes |

---

#### Feature 16: `vio_v_norm`

| Property | Value |
|---|---|
| Name | `vio_v_norm` |
| Source | `np.linalg.norm(current_state.velocity)` |
| Math | Euclidean speed magnitude from classical VIO (m/s) |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| **Causal?** | ✅ Yes |

---

### Group 4: Timing (Feature 17)

#### Feature 17: `dt`

| Property | Value |
|---|---|
| Name | `dt` |
| Source | `t_curr - t_prev` where `t_prev = _current_state.timestamp`, `t_curr = camera_frame.timestamp` |
| Math | Inter-frame duration in seconds, clamped to [1e-4, 1.0] |
| Timestamp | Derived from consecutive camera frame timestamps |
| Future data? | ❌ No |
| Uses GT? | ❌ No |
| Absolute timestamp? | ❌ No — only the **difference** between consecutive timestamps |
| **Causal?** | ✅ Yes |
| Note | `dt` does NOT include absolute UNIX/GPS time, only the inter-frame interval. It cannot be used to identify specific points in the sequence by absolute time. |

---

## Temporal Bias Assessment (A8)

Feature 17 (`dt`) is the only time-related feature. It is approximately constant (~0.05 s at 20 FPS for MH_01_easy) with small variations from IMU jitter. It does not encode absolute sequence progress.

**However**, a subtle indirect temporal bias is possible: if VIO drift is monotonically increasing, then `vio_velocity` and `vio_v_norm` implicitly encode approximate sequence progress through accumulated error. This is inherent to any drift-correcting system trained on a single sequence and is not caused by feature design.

Diagnostic B (dt-removal test) and Diagnostic C (alternative split) bound this risk — see `PHASE_7_AI_ML_REPORT.md`.

---

## Summary

| Feature Index | Name | Source | Future? | GT? | Causal? |
|---|---|---|---|---|---|
| 0 | `tracked_features` | Visual tracking | ❌ | ❌ | ✅ |
| 1 | `inlier_ratio` | RANSAC | ❌ | ❌ | ✅ |
| 2 | `mean_flow_mag` | Optical flow | ❌ | ❌ | ✅ |
| 3 | `std_flow_mag` | Optical flow | ❌ | ❌ | ✅ |
| 4 | `mean_accel[0]` | IMU | ❌ | ❌ | ✅ |
| 5 | `mean_accel[1]` | IMU | ❌ | ❌ | ✅ |
| 6 | `mean_accel[2]` | IMU | ❌ | ❌ | ✅ |
| 7 | `var_accel[0]` | IMU | ❌ | ❌ | ✅ |
| 8 | `var_accel[1]` | IMU | ❌ | ❌ | ✅ |
| 9 | `var_accel[2]` | IMU | ❌ | ❌ | ✅ |
| 10 | `mean_gyro[0]` | IMU | ❌ | ❌ | ✅ |
| 11 | `mean_gyro[1]` | IMU | ❌ | ❌ | ✅ |
| 12 | `mean_gyro[2]` | IMU | ❌ | ❌ | ✅ |
| 13 | `vio_velocity[0]` | VIO state | ❌ | ❌ | ✅ |
| 14 | `vio_velocity[1]` | VIO state | ❌ | ❌ | ✅ |
| 15 | `vio_velocity[2]` | VIO state | ❌ | ❌ | ✅ |
| 16 | `vio_v_norm` | VIO state | ❌ | ❌ | ✅ |
| 17 | `dt` | Camera timestamps | ❌ | ❌ | ✅ |

**All 18 features are strictly causal. No feature uses future data or ground truth.**

---

## Verdict

> **PHASE 7 FEATURE AUDIT — PASS**
>
> All 18 features are derived exclusively from information available at inference time (current frame t_k and prior data). No feature uses future camera frames, future IMU samples, ground-truth position, ground-truth velocity, ground-truth orientation, or any post-processed trajectory statistics.
