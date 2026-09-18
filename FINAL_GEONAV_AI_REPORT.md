# GEONAV-AI — Final Project Report

**Report Date:** 2026-09-19  
**Repository:** GEONAV-AI  
**Status:** Phases 1–8 Complete (Software Validation)

---

## 1. Project Summary

GEONAV-AI is a navigation software module for Unmanned Aerial Vehicles (UAVs) operating in GPS-denied environments. It uses camera imagery and inertial sensor measurements as primary inputs for Visual-Inertial Odometry (VIO), augmented with a lightweight edge-optimized machine learning velocity correction layer.

> **Important:** GEONAV-AI is a software module. No physical UAV hardware is available. All performance metrics are development-machine software benchmarks validated against the EuRoC MAV MH_01_easy dataset. Physical UAV integration is a documented future step.

---

## 2. Development Phase Summary

| Phase | Description | Status |
|---|---|---|
| 1 | Software foundation, interfaces, typed data structures | Complete |
| 2 | EuRoC MAV dataset ingestion (camera + IMU loaders) | Complete |
| 3 | Camera–IMU temporal synchronization | Complete |
| 4 | Classical VIO baseline (Shi-Tomasi, LK flow, Essential matrix, IMU propagation) | Complete |
| 4A | Initial gravity leveling from stationary accelerometer | Complete |
| 4B | Camera–IMU extrinsic calibration (EuRoC T_BS), geometry verification gate | Complete |
| 5 | Sparse 3D mapping (keyframe triangulation, PLY export) | Complete |
| 6 | Ground-truth evaluation (ATE, RPE, velocity error, SE3/Sim3 alignment) | Complete |
| 7 | AI/ML velocity correction (NumpyEdgeMLP, confidence gating), verification gate | Complete |
| 8 | Edge optimization, operating profiles, hardware abstraction layer, health monitor | Complete |

---

## 3. Real-Data Validation Dataset

| Property | Value |
|---|---|
| Dataset | EuRoC MAV MH_01_easy |
| Camera frames | 3,682 |
| IMU samples | 36,820 |
| Ground-truth states | 36,382 |
| GT path length | 80.51 m |
| Sequence duration | ~182 s |
| Visual frames processed | 3,682 / 3,682 |
| Visual updates successful | 3,681 / 3,681 |
| Tracking failures | 0 |
| NaN/Inf in trajectory | 0 |

---

## 4. Visual-Inertial Odometry (Phase 4 Classical Baseline)

### Architecture
- **Feature detection:** Shi-Tomasi corners (`cv2.goodFeaturesToTrack`), max 200 features
- **Feature tracking:** Pyramidal Lucas-Kanade optical flow with forward-backward consistency rejection
- **Relative motion:** 5-point RANSAC Essential matrix (`cv2.findEssentialMat`), cheirality-checked recovery
- **IMU propagation:** Exact quaternion + trapezoidal velocity/position integration with gravity compensation (NED)
- **Fusion:** Complementary SLERP orientation update; IMU-scale-constrained visual direction translation
- **Gravity alignment:** Stationary accelerometer leveling for initial attitude (Phase 4A)
- **Extrinsics:** Full EuRoC T_BS calibration matrix applied (Phase 4B)

### Phase 4 Baseline Performance

| Metric | Value |
|---|---|
| Frames processed | 3,682 |
| Processing rate | ~54 FPS |
| Raw ATE RMSE | 88.93 m |
| Final position error | 150.65 m |
| Velocity RMSE | 1.286 m/s |
| GT path length | 80.51 m |
| Estimated path length | 186.05 m (scale ratio 2.311) |

---

## 5. Sparse Mapping (Phase 5)

| Metric | Value |
|---|---|
| Landmarks triangulated | 279 |
| Mean reprojection error | 0.889 px |
| Keyframes used | ~45 |
| Map format | PLY (exported to artifacts/) |

---

## 6. Ground-Truth Evaluation (Phase 6)

Full trajectory evaluation against EuRoC VIO ground truth:

| Metric | Value |
|---|---|
| Matched pose pairs | 3,638 / 3,682 |
| Time association threshold | 10 ms |
| Raw ATE RMSE | 88.93 m |
| ATE mean | 77.37 m |
| RPE 1s RMSE | 1.163 m |
| RPE 5s RMSE | 4.955 m |
| Velocity RMSE | 1.286 m/s |
| Orientation RMSE | ~130° (known NED↔EuRoC frame mismatch) |

---

## 7. ML Velocity Correction (Phase 7)

### Model Architecture

```
Input: 18-dim feature vector (visual tracking, IMU kinematics, VIO state)
  → Hidden Layer 1: Linear(18→64) + BatchNorm + GELU
  → Hidden Layer 2: Linear(64→32) + BatchNorm + GELU
  → Velocity Head: Linear(32→3) → delta_v (m/s)
  → Confidence Head: Linear(32→1) → Sigmoid → conf ∈ [0, 1]
Inference engine: NumpyEdgeMLP (pure NumPy, no PyTorch runtime)
```

### Training

- Train split: First 70% of MH_01_easy (chronologically)
- Validation split: Last 30% of MH_01_easy (same sequence)
- Labels: `y = v_GT(t) - v_VIO(t)` (offline, GT never used at inference)
- Epochs: 40 | LR: 1e-3 | Batch: 32

### Verified Performance

| Metric | Classical VIO | Hybrid VIO+ML | Reduction |
|---|---|---|---|
| **Raw ATE RMSE [m]** | 88.93 | **6.01** | **93.2%** |
| ATE Mean [m] | 77.37 | 4.52 | 94.2% |
| ATE Median [m] | 76.72 | 3.55 | 95.4% |
| Final Position Error [m] | 150.65 | 16.97 | 88.7% |
| Velocity RMSE [m/s] | 1.286 | 0.462 | 64.1% |
| Vx RMSE [m/s] | 0.506 | 0.255 | 49.6% |
| Vy RMSE [m/s] | 0.578 | 0.336 | 41.8% |
| Vz RMSE [m/s] | 1.031 | 0.187 | **81.9%** |
| RPE 1s RMSE [m] | 1.163 | 0.395 | 66.0% |
| RPE 5s RMSE [m] | 4.955 | 1.657 | 66.6% |

**Validation split (last 30%, chronologically held out from same sequence):**

| Metric | Classical | Hybrid |
|---|---|---|
| ATE RMSE [m] | 130.68 | 9.72 |
| Velocity RMSE [m/s] | 1.221 | 0.611 |

### Phase 7 Verification Gate: PASSED

| Check | Result |
|---|---|
| A1: Data leakage audit | PASS — GT never enters inference |
| A2: Split terminology | PASS — documented as temporal segment, not unseen flight |
| A3: Feature causality (18 features) | PASS — all strictly causal |
| A4: Label generation (GT offline only) | PASS |
| A5: Position integration causality | PASS — Euler, `p_k = p_{k-1} + v*dt` |
| A6: Independent metric recomputation | PASS — diff=0.0000 on all 6 metrics |
| A7: Scale ratio recomputation | PASS |
| A8: Temporal bias diagnostic | PASS — LOW RISK, documented |
| A9: Ablation integrity | PASS — identical VIO params, only ML differs |
| A10: ATE type confirmation | PASS — raw ATE confirmed |
| A11: Confidence gating tests | PASS — 8 new tests added, all pass |
| A12: Acceptance decision | **PASS** |

---

## 8. Edge Optimization & Hardware Abstraction (Phase 8)

### Per-Component Latency (Development-Machine)

| Component | Mean (ms) | P95 (ms) |
|---|---|---|
| Preprocessing | 0.249 | 0.365 |
| Feature Detection | 0.004 | 0.006 |
| Optical Flow | 7.540 | 10.536 |
| Essential Matrix | 3.081 | 5.341 |
| IMU Propagation | 11.133 | 16.784 |
| ML Feature Extraction | 0.689 | 1.014 |
| ML Inference + Correction | **0.508** | **0.733** |
| Total Frame (profiler) | 48.38 | 64.67 |
| Peak Memory | 3.4 MB | — |
| ML Model Size | 25.7 KB | — |

> **Note:** Profiler FPS (20.7) is lower than real-time FPS (~54) because each operation is measured independently (double-processing). Real pipeline FPS: ~54 on development machine.

### Operating Profiles

| Profile | Max Features | ML Every N | Mapping | Buffer Limit |
|---|---|---|---|---|
| DEVELOPMENT | 200 | 1 | Enabled | Unlimited |
| BALANCED | 150 | 2 | Enabled | 10,000 |
| EDGE | 100 | 3 | Disabled | 5,000 |

### Hardware Abstraction Layer

| Class | Purpose |
|---|---|
| `CameraSource` | Abstract camera frame iterator |
| `IMUSource` | Abstract IMU sample iterator |
| `NavigationOutput` | Abstract state delivery channel |
| `UAVIntegrationAdapter` | Future flight-controller interface (not implemented) |
| `EurocCameraSource` | Dataset validation adapter |
| `EurocIMUSource` | Dataset validation adapter |
| `NavigationMessage` | Typed state output (SI units) |
| `ConsoleNavigationOutput` | Debug/logging output |
| `HealthMonitor` | INITIALIZING→TRACKING→DEGRADED→LOST state machine |

---

## 9. Test Suite

| Category | Tests |
|---|---|
| Phase 1 (camera, IMU, state, config) | 28 |
| Phase 2 (EuRoC dataset) | 21 |
| Phase 3 (synchronization) | 26 |
| Phase 4 (VIO pipeline, geometry) | 32 |
| Phase 4B (extrinsics) | 20 |
| Phase 5 (mapping) | 18 |
| Phase 6 (evaluation) | 24 |
| Phase 7 (ML subsystem) | 16 |
| Phase 8 (hardware, health, profiles) | **56** |
| **Total** | **185 / 185 PASSING** |

---

## 10. Limitations

1. **Single-sequence evaluation.** Only EuRoC MH_01_easy has been validated. Cross-sequence generalization (e.g., MH_02, V1_01) is untested.

2. **Temporal validation segment, not unseen flight.** The 30% ML validation split is from the same sequence — not a held-out flight. True cross-flight evaluation requires additional datasets.

3. **No IMU bias estimation.** VIO drift has systematic IMU bias components that are partially compensated by ML but not explicitly modeled. An IMU preintegration + EKF approach would reduce residual drift.

4. **No loop closure.** Without revisiting known landmarks, position drift accumulates over time. 6.01 m residual ATE over ~130 s is the current limit without loop closure.

5. **Monocular scale ambiguity.** Scale is constrained by IMU displacement magnitude, but without sufficient acceleration excitation, metric scale observability degrades.

6. **No physical UAV testing.** All results are from offline dataset replay. Real-world performance on physical UAV hardware has not been demonstrated.

7. **Orientation error ~130°.** Known coordinate frame inversion (NED vs EuRoC room frame) combined with initial stationary epipolar degeneracy causes high orientation error in raw metrics. Position-level metrics are unaffected by this frame offset after zero-initialization alignment.

---

## 11. Future Work

| Priority | Item |
|---|---|
| High | Evaluate on additional EuRoC sequences (MH_02–05, V1_01–03) |
| High | IMU preintegration with bias estimation |
| Medium | Loop closure and relocalization |
| Medium | Multi-sequence ML training for cross-flight generalization |
| Medium | Physical hardware testing (Pi 5, Jetson Nano, Orin) |
| Medium | MAVLink adapter implementation |
| Low | Factor graph back-end (GTSAM) |
| Low | Full SLAM with dense mapping |

---

## 12. Key Files

| File | Purpose |
|---|---|
| [`src/geonav/vio/pipeline.py`](src/geonav/vio/pipeline.py) | VIO pipeline entry point |
| [`src/geonav/ml/inference.py`](src/geonav/ml/inference.py) | ML velocity corrector |
| [`src/geonav/ml/model.py`](src/geonav/ml/model.py) | NumpyEdgeMLP, EdgeMLP |
| [`src/geonav/hardware/health.py`](src/geonav/hardware/health.py) | Health monitor state machine |
| [`src/geonav/config/settings.py`](src/geonav/config/settings.py) | OperatingProfile, Settings |
| [`PHASE_7_DATA_LEAKAGE_AUDIT.md`](PHASE_7_DATA_LEAKAGE_AUDIT.md) | ML pipeline leakage audit |
| [`PHASE_7_FEATURE_AUDIT.md`](PHASE_7_FEATURE_AUDIT.md) | Feature causality audit |
| [`PHASE_7_AI_ML_REPORT.md`](PHASE_7_AI_ML_REPORT.md) | Phase 7 verification report |
| [`scripts/verify_phase7_metrics.py`](scripts/verify_phase7_metrics.py) | Independent metric verification |
| [`scripts/profile_pipeline.py`](scripts/profile_pipeline.py) | Phase 8 latency profiler |
| [`scripts/benchmark_profiles.py`](scripts/benchmark_profiles.py) | Phase 8 profile benchmark |

---

## 13. Conclusion

GEONAV-AI has been successfully developed through all 8 planned phases:

- **Classical VIO baseline:** Fully operational, processing 3,682 frames at ~54 FPS with 0 failures.
- **ML enhancement:** Raw ATE reduced from 88.93 m to 6.01 m (93.2%) using a 25.7 KB NumpyEdgeMLP model with 0.51 ms inference latency.
- **Verification:** Phase 7 verification gate PASSED all A1–A12 checks. Independent metric recomputation confirms all results exactly.
- **Edge readiness:** Hardware abstraction layer, operating profiles, and health monitoring provide the architectural foundation for UAV integration.
- **Test coverage:** 185/185 tests passing.

The system is ready for the next validation step: evaluation on additional EuRoC sequences and eventual physical UAV hardware integration.

---

*This report was generated from verified software execution results. All performance claims are development-machine software benchmarks. No physical UAV hardware was used.*
