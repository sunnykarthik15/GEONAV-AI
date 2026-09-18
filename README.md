# GEONAV-AI

## Project Definition

> **"GEONAV-AI is a navigation software module intended for integration into UAV platforms. It is not the UAV itself."**

**Current status:** Software-validated on real EuRoC MAV dataset. No physical UAV hardware available. All performance claims are development-machine software benchmarks only.

## Problem

Unmanned Aerial Vehicle (UAV) navigation becomes difficult or unreliable when GPS or other external global positioning signals are degraded, spoofed, or completely unavailable (GPS-denied environments such as indoor facilities, urban canyons, dense forests, or subterranean infrastructure).

## Intended Approach

GEONAV-AI uses onboard camera imagery and high-rate Inertial Measurement Unit (IMU) measurements for Visual-Inertial Odometry (VIO), enhanced with a lightweight edge-optimized ML velocity correction layer.

## System Architecture (Phase 8)

```
Camera Dataset ──> Camera Loader ──> Camera Frames
                                          |
                                     timestamps
                                          v
                                    Synchronizer (Phase 3)
                                          ^
                                     timestamps
                                          |
IMU Dataset ─────> IMU Loader ────> IMU Samples
                                          |
                                          v
                              Synchronized Measurements
                                          |
                                          v
                                  VIO Pipeline (Phase 4)
                                  [Shi-Tomasi + LK Flow]
                                  [Essential Matrix RANSAC]
                                  [IMU Propagation + Fusion]
                                          |
                                          v
                               Camera-IMU Extrinsics (Phase 4B)
                               [EuRoC T_BS calibration]
                                          |
                                          v
                            Mapping Subsystem (Phase 5)
                            [Triangulation + Keyframes]
                                          |
                                          v
                     Ground-Truth Evaluation (Phase 6)
                     [ATE, RPE, Velocity Error, Scale]
                                          |
                                          v
                    ML Velocity Correction (Phase 7)
                    [NumpyEdgeMLP, Confidence Gating]
                                          |
                                          v
             Hardware Abstraction + Health Monitor (Phase 8)
             [CameraSource, IMUSource, NavigationOutput]
             [OperatingProfile: DEVELOPMENT/BALANCED/EDGE]
                                          |
                                          v
                           Navigation State + Trajectory
```

## Validated Performance (EuRoC MH_01_easy)

**All metrics are development-machine software benchmarks. Not embedded/UAV hardware performance.**

| Metric | Classical VIO | ML-Enhanced (Hybrid) | Reduction |
|---|---|---|---|
| Raw ATE RMSE [m] | 88.93 | **6.01** | 93.2% |
| Final Position Error [m] | 150.65 | 16.97 | 88.7% |
| Velocity RMSE [m/s] | 1.286 | 0.462 | 64.1% |
| Vz RMSE [m/s] | 1.031 | 0.187 | 81.9% |
| RPE 1s RMSE [m] | 1.163 | 0.395 | 66.0% |
| RPE 5s RMSE [m] | 4.955 | 1.657 | 66.6% |
| Processing Rate | ~54 FPS | ~54 FPS | — |
| ML Inference Latency | — | 0.51 ms | — |
| ML Model Size | — | 25.7 KB | — |
| Mapping Landmarks | 279 | 279 | — |
| Mean Reprojection Error | 0.889 px | 0.889 px | — |

**Test suite: 185/185 passing**

> Note: The 30% validation split is a chronologically held-out temporal segment from the same MH_01_easy sequence (not an unseen flight). Only one EuRoC sequence is currently available.

---

## Development Phases

### Phase 1: Software Foundation & Interfaces (Complete)
- Established directory structure, clean modular layout, and dependency configuration.
- Defined core typed data representations (`CameraFrame`, `IMUSample`, `NavigationState`).
- Created abstract sensor interfaces (`CameraSensor`, `IMUSensor`).
- Created buffering synchronization interface (`SensorSynchronizer`).
- Created pipeline placeholders (`VIOPipeline`, `Mapper`, `Settings`).

### Phase 2: Visual-Inertial Data Ingestion (Complete)
- Implemented real visual-inertial dataset ingestion using the standard **EuRoC MAV** dataset format as reference.
- `EurocCameraLoader`: Parses `cam0/data.csv`, extracts timestamps, resolves image paths, and lazily loads raw images.
- `EurocIMULoader`: Parses `imu0/data.csv`, extracts nanosecond timestamps, angular rates $(w_x, w_y, w_z)$, and linear accelerations $(a_x, a_y, a_z)$.
- `EurocDataset`: High-level reader automatically resolving standard `mav0/` or direct sequence roots.
- `DatasetCameraSensor` & `DatasetIMUSensor`: Adapters connecting dataset loaders to Phase 1 sensor interfaces.

### Phase 3: Camera–IMU Synchronization & Preprocessing (Complete)
- Implemented deterministic timestamp-based camera–IMU windowing and sensor-data validation.
- `SynchronizedMeasurement`: Encapsulates a camera frame, its preceding camera timestamp ($T_{previous}$), current timestamp ($T_{current}$), and associated IMU samples in $(T_{previous}, T_{current}]$.
- `synchronize_streams()`: High-performance linear two-pointer streaming synchronization.

### Phase 4: Visual-Inertial Odometry (VIO) Baseline (Complete)
- Implemented a classical Visual-Inertial Odometry baseline combining 2D feature tracking, essential-matrix geometry, IMU state propagation, and a loosely coupled state update.
- Classical visual front-end: Shi-Tomasi corner detection, pyramidal Lucas-Kanade optical flow, forward-backward bidirectional consistency checking.
- Geometric motion estimation: Normalized coordinates, RANSAC essential matrix estimation (`cv2.findEssentialMat`), cheirality-checked pose recovery (`cv2.recoverPose`).
- High-rate IMU propagator: Exact trapezoidal/quaternion integration with small-angle stability, body-to-world rotation, and gravity compensation in NED.
- Loosely coupled state fusion: Gyro and visual rotation update via spherical linear interpolation (SLERP), translational direction constraint along visual epipolar ray, and fallback to `DEGRADED_IMU_ONLY` on visual tracking loss.

### Phase 4A: Initial Gravity Leveling (Complete)
- Stationary accelerometer-based gravity alignment for initial orientation estimate.
- Eliminates the pure-zero initial attitude assumption.

### Phase 4B: Camera-IMU Extrinsic Calibration (Complete)
- Integrated real EuRoC T_BS calibration matrix into the VIO pipeline.
- Verified body-frame consistency of relative rotation and translation transforms.
- Geometry verification gate: PASSED.

### Phase 5: Sparse 3D Mapping (Complete)
- Lightweight keyframe-based triangulation subsystem.
- 279 landmarks reconstructed from MH_01_easy with mean reprojection error 0.889 px.
- PLY map export for external visualization.

### Phase 6: Ground-Truth Evaluation (Complete)
- Rigorous trajectory evaluation: ATE, RPE, axis errors, scale analysis, velocity errors.
- SE(3) and Sim(3) alignment implemented.
- Full comparison of estimated vs EuRoC ground-truth trajectory.

### Phase 7: AI/ML Velocity Correction (Complete, Verified)
- `NumpyEdgeMLP` — pure NumPy inference engine, 25.7 KB model, 0.51 ms/frame.
- 18-dimensional causal feature vector (visual, IMU, VIO state).
- Trained on first 70% of MH_01_easy; evaluated on last 30%.
- **Baseline ATE RMSE: 88.93 m → Hybrid ATE RMSE: 6.01 m (raw, −93.2%)**.
- Phase 7 Verification Gate: PASSED (all A1–A12 checks).
- Independent metric verification script confirms all metrics exactly.

### Phase 8: Edge Optimization & UAV Integration Readiness (Complete)
- Hardware abstraction layer: `CameraSource`, `IMUSource`, `NavigationOutput`, `UAVIntegrationAdapter`.
- `EurocCameraSource` / `EurocIMUSource` implement interfaces for dataset validation.
- `HealthMonitor`: INITIALIZING → TRACKING → DEGRADED → LOST state machine.
- `OperatingProfile` enum: DEVELOPMENT / BALANCED / EDGE with configurable `ProfileConfig`.
- Profile benchmark: FPS vs accuracy trade-off across 3 profiles.
- 185/185 tests passing.

---

## Phase 4 — Visual-Inertial Odometry

```
Camera Frame
     │
     ▼
Feature Detection (Shi-Tomasi)
     │
     ▼
Feature Tracking (LK Optical Flow + Forward-Backward Check)
     │
     ▼
Essential Matrix (RANSAC 5-point)
     │
     ▼
Visual Relative Motion (R_vis, t_dir)
             │
             │           IMU Samples in (T_prev, T_curr]
             │                         │
             │                         ▼
             │                  IMU Propagation
             │                 (v_prop, p_prop, q_prop)
             │                         │
             ▼                         ▼
        Baseline VIO State Update / Correction
                         │
                         ▼
        Local Navigation State / Trajectory
```

> **Phase 4 is a classical baseline VIO implementation. It is not SLAM, does not perform global mapping, does not perform loop closure, and does not provide globally drift-free localization.**

### 1. Visual Odometry Front-End
- **Image Preprocessing**: Validates image shape, finite pixels, and converts to single-channel 8-bit grayscale.
- **Feature Detection**: Classical Shi-Tomasi corner detection (`cv2.goodFeaturesToTrack`) with configurable maximum features (default 200), minimum quality (0.01), and minimum pixel spacing (10.0 px).
- **Feature Tracking**: Pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) tracked from previous frame to current frame, verified via backward tracking (current to previous). Points with forward-backward Euclidean deviation $\ge 1.0$ px or outside image bounds are rejected.
- **Essential Matrix & Relative Motion**: Normalizes feature coordinates with camera intrinsics ($f_x, f_y, c_x, c_y$) and distortion parameters ($k_1, k_2, p_1, p_2$). Computes essential matrix using 5-point RANSAC (`cv2.findEssentialMat`) and decomposes it into relative rotation $R_{vis}$ and relative translation unit vector $\hat{\mathbf{t}}_{vis}$ with cheirality positive-depth checking (`cv2.recoverPose`).

### 2. IMU Propagation
- **Chronological Windowing**: Integrates IMU samples in $(T_{previous}, T_{current}]$. Rejects non-positive or non-finite $\Delta t$.
- **Orientation Evolution**: Computes incremental quaternion rotation using small-angle stable formulation for $\|\boldsymbol{\omega}\| \to 0$ and Hamilton multiplication:
  $$q_{k+1} = \text{normalize}(q_k \otimes \Delta q(\boldsymbol{\omega}_k, \Delta t))$$
- **Kinematic Acceleration & Gravity**: Transforms specific force from body frame to world frame and compensates for gravity in NED:
  $$\mathbf{a}_{world} = R(q_k)\mathbf{a}_{body} + \mathbf{g}_{world}$$
- **State Integration**: Numerically integrates acceleration and velocity over the variable $\Delta t$ steps:
  $$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_{world}\Delta t, \quad \mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}\mathbf{a}_{world}\Delta t^2$$

### 3. Visual-Inertial Fusion Rule
The baseline estimator maintains a single state $(\mathbf{p}, \mathbf{v}, q)$ at camera timestamps:
1. **Inertial Propagation**: Propagates previous navigation state through high-rate IMU samples to obtain $(\mathbf{p}_{prop}, \mathbf{v}_{prop}, q_{prop})$.
2. **Visual Target**: When visual tracking succeeds ($\ge \text{min\_features}$ inliers), constructs relative visual rotation $q_{vis\_target} = q_{prev} \otimes q(R_{vis})$.
3. **Orientation Update**: Fuses inertial and visual orientation via complementary SLERP:
   $$q_{updated} = \text{slerp}(q_{prop}, q_{vis\_target}, \alpha), \quad \alpha = 0.5$$
4. **Translation Update**: Monocular vision cannot provide metric scale. Rather than equating metric translation to visual direction times IMU norm (which user guidelines explicitly forbid), the estimator computes IMU displacement $\Delta \mathbf{p}_{imu} = \mathbf{p}_{prop} - \mathbf{p}_{prev}$ and decomposes it along the camera-frame visual direction:
   $$\mathbf{u}_{world} = R(q_{prev}) \hat{\mathbf{t}}_{vis}$$
   The displacement component along $\mathbf{u}_{world}$ is retained, constraining inter-frame drift orthogonal to the epipolar ray.
5. **Velocity Continuity**: Updates velocity consistently with the state transition $\mathbf{v} = (\mathbf{p} - \mathbf{p}_{prev}) / \Delta T$.

### 4. Monocular Scale Ambiguity
Monocular vision recovers relative translation only up to an unknown scale factor:
$$\|\hat{\mathbf{t}}_{vis}\| = 1 \neq \text{metric distance}$$
Essential matrix decomposition provides ray direction only. Metric scale can only be recovered or propagated through IMU integration, which depends strictly on:
- High-quality accelerometer calibration and zero-bias stability
- Sufficient linear acceleration excitation to observe scale
- Accurate camera-to-IMU temporal and spatial calibration
- Correct gravity handling

If these conditions are not met, metric scale cannot be arbitrarily invented. GEONAV-AI documents this explicit limitation.

### 5. Coordinate Frames and Gravity Convention
- **World / Navigation Frame**: North-East-Down (NED):
  - $X$: North
  - $Y$: East
  - $Z$: Down
- **Camera Frame**: Standard optical coordinate frame:
  - $X$: Right
  - $Y$: Down
  - $Z$: Forward along optical axis
- **Body / IMU Frame**: Aircraft body frame:
  - $X$: Forward
  - $Y$: Right
  - $Z$: Down
- **Gravity Direction & Specific Force**:
  - $\mathbf{g}_{world} = [0, 0, +9.81]^T \text{ m/s}^2$ pointing Down.
  - The accelerometer measures specific force $\mathbf{f}_{meas} = \mathbf{a}_{kinematic} - R(q)^T \mathbf{g}_{world}$.
  - When stationary and level ($R = I$), the sensor measures upward normal reaction force $\mathbf{f}_{meas} = [0, 0, -9.81]^T$.
  - Compensated world acceleration is:
    $$\mathbf{a}_{world} = R(q)\mathbf{f}_{meas} + \mathbf{g}_{world} = [0, 0, 0]^T$$
- **Quaternion Format**: Hamilton format $(w, x, y, z)$ with $w$ as real scalar, satisfying $q \otimes q^{-1} = 1$.

### 6. Calibration Assumptions & Extrinsics
- **Camera Intrinsics**: Supported through `CameraConfig` ($f_x, f_y, c_x, c_y$, distortion parameters). Defaults can be overridden with actual sequence calibration (such as EuRoC `cam0/sensor.yaml`).
- **Camera-to-IMU Extrinsics**: Phase 4 baseline operates in the body coordinate system with identity extrinsic rotation $R_{BC} = I$ and translation $\mathbf{p}_{BC} = \mathbf{0}$ by default. A full 6-DoF extrinsic transformation matrix is documented as an explicit architectural hook for subsequent phases.

### 7. Failure Handling & Degraded IMU-Only Mode
- **Tracking Loss**: If tracked geometric inliers drop below `min_features` (default 15), status transitions to `DEGRADED_IMU_ONLY`.
- **IMU Fallback**: In `DEGRADED_IMU_ONLY` mode, state is propagated purely via IMU kinematic integration without fabricating fake visual poses. Visual feature re-detection is triggered immediately on the current frame.
- **Critical Failure**: If IMU timestamps are invalid, non-increasing, non-finite, or $\Delta t \le 0$, the pipeline marks status as `FAILED` and refuses to update state.
- **Determinism**: Identical sensor inputs produce bit-for-bit deterministic outputs. Every numerical state is strictly finite (no silent NaNs or Infs).

---

## Phase 3 Synchronization Concepts

### 1. Why Synchronization is Necessary
Camera frames and IMU measurements operate at vastly different sampling frequencies (e.g. 20–30 Hz for optical imagery vs. 200 Hz for inertial sensors). Their clocks and arrival times are independent. Downstream state estimators (VIO) require knowing exactly which high-rate inertial accelerations and rotations occurred during the exact time interval between consecutive visual frames.

### 2. Why Accurate Timestamps Matter
Inertial measurements must be integrated over time to compute changes in velocity and orientation between visual observations. Even minor microsecond timing mismatches cause integration errors, drift, and scale estimation divergence. Therefore, nanosecond dataset timestamps are converted to floating-point seconds and used deterministically.

### 3. How IMU Samples are Grouped Between Camera Frames
For each consecutive pair of camera frames:
$$\text{Camera}[i-1] \text{ at } T_{previous} \quad \text{and} \quad \text{Camera}[i] \text{ at } T_{current}$$
all IMU samples falling within $(T_{previous}, T_{current}]$ are collected in strict chronological order into a `SynchronizedMeasurement` package.

### 4. Timestamp Boundary Policy
GEONAV-AI enforces the strict mathematical boundary convention:
$$\mathbf{T_{previous} < t_{IMU} \le T_{current}}$$
- **Exclusive Left ($t_{IMU} > T_{previous}$)**: Prevents duplicate assignment of boundary samples that were already assigned to the preceding interval $(T_{prev-1}, T_{previous}]$.
- **Inclusive Right ($t_{IMU} \le T_{current}$)**: Deterministically assigns any IMU sample coinciding with the current frame boundary to the current interval.
- **Coverage Guarantee**: Consecutive camera intervals assign every IMU sample that falls within the camera timeline $(T_0, T_{final}]$ to exactly one interval, with no duplication.
- IMU samples at or before the first camera timestamp ($t \le T_0$) and after the final camera timestamp ($t > T_{final}$) fall outside the inter-frame synchronization timeline and are excluded from completed camera intervals without mutating or deleting input sequences.

### 5. First Camera Frame Handling
The initial camera frame ($T_0$) has no prior interval ($T_{previous}$ does not exist).
- When `include_first_frame = True`, it is emitted with `start_timestamp = None`, `end_timestamp = T_0`, `imu_samples = []`, and `is_first_frame = True`.
- When `include_first_frame = False`, synchronization output begins directly with the first inter-frame interval $(T_0, T_1]$.
- No synthetic or fake IMU intervals are fabricated.

### 6. Missing IMU Data Representation
If zero IMU samples occur within an inter-frame interval:
- With `require_imu_for_interval = False` (default), `imu_samples` is represented as an empty list `[]`.
- **No interpolation, no zero-filling, and no synthetic measurements are ever generated.**
- With `require_imu_for_interval = True`, an inter-frame interval with zero IMU samples raises a `ValueError`.

---

## Current Capability & Limitations

### Current Capability
GEONAV-AI can:
- Read recorded camera frames and IMU measurements from local EuRoC datasets
- Validate sensor readings and timestamp monotonicity
- Deterministically synchronize high-rate IMU measurements with consecutive camera frames into `SynchronizedMeasurement` packages
- Extract and track visual features across frames using Lucas-Kanade optical flow and bidirectional consistency checking
- Estimate relative camera rotation and translation direction using essential matrix RANSAC
- Propagate kinematic state (position, velocity, orientation) using discrete IMU measurements with gravity compensation in NED
- Fuse visual and inertial measurements into a local navigation trajectory via a loosely coupled baseline
- Gracefully handle visual tracking loss by switching to `DEGRADED_IMU_ONLY` propagation and triggering feature re-detection

### Current Limitations
GEONAV-AI Phase 4 is a classical baseline VIO implementation:
- **No SLAM**: Does not perform global bundle adjustment, global point-cloud or mesh mapping, or landmark map persistence.
- **No Loop Closure**: Does not detect previously visited locations or correct accumulated long-term drift.
- **No Globally Drift-Free Localization**: Being dead-reckoning odometry without absolute positioning references (like GPS), drift naturally accumulates over long trajectories.
- **Monocular Scale Observability**: Without sufficient acceleration excitation and high-accuracy IMU calibration, monocular metric scale recovery remains limited.
- **Planar or Degenerate Motion**: Monocular essential matrix estimation degrades under pure rotation or zero parallax.

## What GEONAV-AI Is NOT

To maintain clarity of scope, GEONAV-AI is explicitly **NOT**:

- A drone or physical aerial platform
- A flight controller (it does not command motors, rotors, or flight dynamics)
- A standalone hardware GPS replacement device by itself
- An obstacle avoidance or path planning system
- A complete SLAM system with global mapping and loop closure
- Validated on physical UAV hardware (software-validated on EuRoC dataset only)

## Performance Profiling (Development-Machine Software Benchmark)

Measured on development laptop, Python 3.11, CPU only. Do NOT extrapolate to embedded hardware.

| Component | Mean (ms) | P95 (ms) |
|---|---|---|
| Preprocessing | 0.249 | 0.365 |
| Optical Flow | 7.540 | 10.536 |
| Essential Matrix | 3.081 | 5.341 |
| IMU Propagation | 11.133 | 16.784 |
| ML Inference + Correction | 0.508 | 0.733 |
| **Total Frame** | **~18–25** | ~35 |
| Peak Memory | 3.4 MB | — |
| Real-time FPS (hybrid) | ~54 | — |


## Project Directory Structure

```
GEONAV-AI/
|
+-- src/
|   +-- geonav/
|       +-- __init__.py
|       +-- config/
|       |   +-- settings.py         # Settings, VIOConfig, MappingConfig, OperatingProfile
|       +-- datasets/euroc/          # EurocDataset, camera/IMU loaders, types
|       +-- evaluation/              # ATE, RPE, scale, velocity, SE3/Sim3 alignment
|       +-- hardware/                # Phase 8: HAL interfaces, health monitor, adapters
|       |   +-- interfaces.py        # CameraSource, IMUSource, NavigationOutput, UAVAdapter
|       |   +-- health.py            # HealthMonitor state machine
|       |   +-- nav_output.py        # NavigationMessage (SI-unit typed), ConsoleOutput
|       |   +-- euroc_adapter.py     # EurocCameraSource, EurocIMUSource
|       +-- mapping/                 # Phase 5: keyframe triangulation, PLY export
|       +-- ml/                      # Phase 7: NumpyEdgeMLP, features, inference, train
|       +-- sensors/                 # CameraFrame, IMUSample, sensor interfaces
|       +-- state/                   # NavigationState
|       +-- synchronization/         # SensorSynchronizer, SynchronizedMeasurement
|       +-- vio/                     # Phase 4: pipeline, geometry, frontend, propagator
|
+-- tests/                           # 185 unit tests
|   +-- test_ml.py                   # Phase 7 ML tests (16 tests, incl. A11 gating)
|   +-- test_hardware_interfaces.py  # Phase 8 HAL interface tests
|   +-- test_health_monitoring.py    # Phase 8 health state machine tests
|   +-- test_edge_configuration.py   # Phase 8 profile/config tests
|   +-- [10 other test files]
|
+-- scripts/
|   +-- validate_euroc_vio.py        # Real-data VIO validation
|   +-- verify_phase7_metrics.py     # Independent Phase 7 metric verification
|   +-- profile_pipeline.py          # Phase 8 per-component latency profiler
|   +-- benchmark_profiles.py        # Phase 8 DEVELOPMENT/BALANCED/EDGE benchmark
|
+-- artifacts/                        # Exported metrics, weights, maps, plots
+-- PHASE_7_DATA_LEAKAGE_AUDIT.md
+-- PHASE_7_FEATURE_AUDIT.md
+-- PHASE_7_AI_ML_REPORT.md
+-- PHASE_6_GROUND_TRUTH_EVALUATION_REPORT.md
+-- FINAL_GEONAV_AI_REPORT.md
+-- requirements.txt
+-- README.md
```


## Getting Started

### Prerequisites

- Python 3.11+

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sunnykarthik15/GEONAV-AI.git
cd GEONAV-AI
pip install -r requirements.txt
```

### Running Tests

Execute the complete unit test suite using pytest:

```bash
python -m pytest tests/ -v
```

Expected: **185/185 tests passing**.

### Running Validation Scripts

```bash
# Full real-data VIO validation (Phase 4)
python scripts/validate_euroc_vio.py

# Independent Phase 7 metric verification
python scripts/verify_phase7_metrics.py

# Phase 8 per-component latency profiler
python scripts/profile_pipeline.py

# Phase 8 profile benchmark (DEVELOPMENT vs BALANCED vs EDGE)
python scripts/benchmark_profiles.py
```

### Verifying Imports

To verify package imports across all development phases:

```bash
python -c "import sys; sys.path.insert(0, 'src'); import geonav; from geonav.datasets.euroc import EurocDataset; from geonav.vio import VIOPipeline; from geonav.ml.inference import MLVelocityCorrector; from geonav.hardware import HealthMonitor, EurocCameraSource; from geonav.config.settings import get_settings_for_profile, OperatingProfile; print('All phase imports verified successfully')"
```

### Validating on EuRoC Dataset

To run the real-dataset validation script (requires MH_01_easy at `data/raw/euroc/MH_01_easy`):

```bash
python scripts/validate_euroc_vio.py
```
