# GEONAV-AI — PHASE 7 AI/ML DESIGN DOCUMENT
**AI/ML Suitability Analysis, Architecture Selection, and Engineering Specification for Edge UAV Visual-Inertial Navigation**

---

## 1. CURRENT VIO BASELINE WEAKNESSES

The Phase 6 ground-truth evaluation of the classical dead-reckoning VIO pipeline on EuRoC `MH_01_easy` quantitatively identified the primary performance bottlenecks:

1. **Vertical Drift Dominance**:
   - Raw $Z$-axis RMSE reached **88.621 m** (with terminal error of **150.522 m**), while horizontal tracking remained bounded ($X$ RMSE: 5.766 m, $Y$ RMSE: 4.597 m).
   - In monocular visual odometry without an altimeter, rangefinder, or depth sensor, vertical translation induces uniform optical expansion rather than strong parallax. In the absence of an integrated filter estimating accelerometer biases, open-loop vertical accelerometer integration accumulates error over the 182-second sequence.
2. **Velocity Estimation Drift**:
   - Total linear velocity RMSE is **1.286 m/s** (horizontal $V_x$: 0.506 m/s, $V_y$: 0.578 m/s; vertical $V_z$: 1.031 m/s).
   - Unmodeled IMU accelerometer biases and imperfect gravity removal create persistent velocity offsets that integrate directly into quadratic position error.
3. **Monocular Scale Ambiguity & Metric Scale Dilation**:
   - The estimated cumulative trajectory length was **186.05 m** compared to the ground-truth length of **80.51 m** (scale ratio: $2.311$), heavily dilated along the vertical axis.
   - However, global similarity alignment (Sim(3)) achieved an ATE RMSE of **3.999 m**, demonstrating that the trajectory shape and relative turns estimated by the frontend are topologically sound.
4. **Sensitivity to Epipolar Degeneracy at Near-Zero Velocity**:
   - During stationary and low-parallax periods (e.g., on the launch stand), 5-point essential matrix recovery ($E = [t]_\times R$ with $\|t\| \to 0$) suffers from numerical instability, occasionally producing large instantaneous rotation steps.

---

## 2. CANDIDATE AI/ML IMPROVEMENTS

We evaluate four candidate AI/ML architectures for addressing these weaknesses:

| Candidate Architecture | Description | Pros | Cons / Risks | Edge UAV Viability |
| :--- | :--- | :--- | :--- | :--- |
| **Candidate 1: End-to-End Deep VIO** (e.g., DeepVO, VINet) | End-to-end neural network mapping raw consecutive images + IMU tensors directly to 6-DoF poses. | Eliminates manual feature tracking and camera calibration pipelines. | Complete black box; lacks geometric interpretability; unproven generalization outside training scenes; massive GPU compute requirement (>50 ms/frame). | **UNSUITABLE** (High latency, zero safety guarantees). |
| **Candidate 2: Learned Feature Front-End** (e.g., SuperPoint, SuperGlue) | Deep neural networks for keypoint detection and descriptor matching, feeding into classical geometry. | Superior feature repeatability in low-texture and motion-blurred scenes. | High memory footprint (>150 MB); 30–80 ms per frame on embedded CPUs; does not solve vertical IMU bias integration or scale drift. | **MARGINAL** (Too heavy for CPU-only edge flight). |
| **Candidate 3: Deep Loop Closure / Global Relocalization** (e.g., NetVLAD, HF-Net) | Deep scene descriptor matching to detect loop closures and constrain pose graph drift. | Global drift elimination when revisiting mapped environments. | Requires building and querying large topological feature databases; ineffective in non-revisited exploratory flight paths; does not improve dead-reckoning between loops. | **FUTURE SCOPE** (High memory overhead). |
| **Candidate 4: Hybrid Learned Velocity Residual Correction** (Selected Approach) | Classical VIO computes interpretable geometric visual-inertial state; a compact Edge MLP predicts velocity residuals ($\Delta \mathbf{v}$) and confidence from causal inference-time signals. | Preserves classical geometric interpretability; directly attacks dominant velocity bias; ultra-lightweight (<50 KB, <0.2 ms CPU latency); fails gracefully back to classical VIO if confidence is low. | **HIGHLY SUITABLE** (Ideal for real-time edge execution on resource-constrained UAV flight hardware). |

---

## 3. SELECTION RATIONALE FOR HYBRID VELOCITY RESIDUAL CORRECTION

The **Hybrid Learned Velocity Residual Correction** architecture is selected based on the following engineering criteria:

1. **Safety and Predictability**: The classical VIO pipeline remains the primary, interpretable geometric navigation backbone. The neural network never invents arbitrary global 3D positions; it only predicts incremental velocity corrections bounded by physical flight envelopes.
2. **Directly Addresses the Root Failure Mode**: Position error in VIO is the integral of velocity error:
   $$\mathbf{e}_p(t) = \int_0^t \mathbf{e}_v(\tau) d\tau$$
   By predicting and subtracting the systematic velocity residual $\Delta \mathbf{v}(t) = \mathbf{v}_{GT}(t) - \mathbf{v}_{VIO}(t)$ at each step, velocity error is bounded, which directly halts quadratic position divergence.
3. **Ultra-Low Computational Footprint**: A compact Multi-Layer Perceptron (MLP) with two hidden layers (32 and 16 units) requires only ~1,200 parameters (< 10 KB). Inference executes in $< 0.15$ ms on a single standard CPU thread without requiring GPU acceleration.
4. **Calibrated Confidence Gating**: The model outputs an auxiliary scalar confidence value $c \in [0, 1]$. When visual tracking degrades or input features fall outside normal distributions, $c \to 0$ and the correction gracefully attenuates to the unassisted classical VIO estimate.

---

## 4. ARCHITECTURE OVERVIEW

```
       [ Camera Frames ]              [ IMU Samples ]
               │                             │
               ▼                             ▼
       ┌─────────────────────────────────────────────┐
       │     Classical Geometric VIO Pipeline        │
       │  - KLT Optical Flow Feature Tracking        │
       │  - 5-Point Essential Matrix + RANSAC        │
       │  - Accelerometer Gravity Leveling           │
       │  - High-Rate IMU Kinematic Integration     │
       └──────────────────────┬──────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     Classical VIO State             Causal Feature Vector
      (p_vio, v_vio, q_vio)             x_t ∈ R^18
              │                               │
              │                               ▼
              │                     ┌───────────────────┐
              │                     │ Compact Edge MLP  │
              │                     │ (Input -> 32->16) │
              │                     └─────────┬─────────┘
              │                               │
              │             ┌─────────────────┴─────────────────┐
              │             ▼                                   ▼
              │     Velocity Residual               Prediction Confidence
              │     Δv_pred ∈ R^3                          c ∈ [0, 1]
              │             │                                   │
              └─────────────┼───────────────────────────────────┘
                            ▼
              ┌───────────────────────────┐
              │     Gated State Fusion    │
              │ v_corr = v_vio + c * Δv   │
              │ p_corr = p_prev + v_corr*dt│
              └─────────────┬─────────────┘
                            ▼
                Corrected Navigation State
```

---

## 5. INPUT FEATURE VECTOR SPECIFICATION

All input features are strictly causal, computed exclusively from information available at the current time step $t_k$. No future frames, ground truth, or offline parameters are accessible at inference time:

| Index | Feature Symbol | Description | Physical Motivation |
| :--- | :--- | :--- | :--- |
| **0** | $N_{tracked}$ | Number of KLT features successfully tracked | Indicates visual tracking reliability. |
| **1** | $r_{inlier}$ | RANSAC inlier ratio ($N_{inliers} / N_{tracked}$) | Detects epipolar geometry confidence and outliers. |
| **2** | $\|\bar{\mathbf{d}}_{flow}\|$ | Mean optical flow displacement magnitude (pixels) | Captures visual scene motion rate. |
| **3** | $\sigma_{flow}$ | Standard deviation of optical flow displacement | Measures flow field coherence. |
| **4–6** | $\bar{\mathbf{a}}_{imu}$ | Mean linear acceleration over inter-frame interval $[a_x, a_y, a_z]$ | Captures vehicle specific force and tilt. |
| **7–9** | $\sigma^2_{\mathbf{a}}$ | Variance of acceleration components $[\sigma^2_{ax}, \sigma^2_{ay}, \sigma^2_{az}]$ | Detects dynamic maneuvers vs stationary hover. |
| **10–12**| $\bar{\boldsymbol{\omega}}_{imu}$ | Mean angular velocity $[\omega_x, \omega_y, \omega_z]$ | Captures vehicle turn rate. |
| **13–15**| $\mathbf{v}_{vio}$ | Current uncorrected VIO velocity estimate $[v_x, v_y, v_z]$ | Baseline velocity to be corrected. |
| **16** | $\|\mathbf{v}_{vio}\|$ | Velocity magnitude | Scalar dynamic speed. |
| **17** | $\Delta t$ | Inter-frame time interval (s) | Normalizes temporal integration steps. |

Total feature vector dimensionality: **18 features**.

---

## 6. TARGET LABELS & TRAINING OBJECTIVE

Ground-truth states are used **ONLY OFFLINE** during dataset preparation to generate supervision labels. Ground truth is never accessed during inference or evaluation.

### Target:
$$\Delta \mathbf{v}^* = \mathbf{v}_{GT}(t_k) - \mathbf{v}_{VIO}(t_k) \in \mathbb{R}^3$$

### Loss Function:
The network is trained with a composite loss combining Mean Squared Error (MSE) on the velocity residual with a confidence regularization term:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^B \left( \|\Delta \mathbf{v}^*_i - \Delta \hat{\mathbf{v}}_i\|^2 + \lambda_{reg} (1 - c_i)^2 \right)$$

where $c_i \in [0, 1]$ is bounded via a sigmoid activation, penalizing unnecessary confidence attenuation when errors are small.

---

## 7. DATASET SPLIT & AVOIDING TEMPORAL DATA LEAKAGE

Because the development dataset currently contains sequence `MH_01_easy`, we enforce strict **chronological sequence-level partitioning** to prevent temporal leakage:

- **Training Partition (70%)**: First 2,546 matched frames ($t \in [1403636580.84, 1403636708.20]$ s, duration: $127.36$ s). Includes takeoff, climb, multiple hall loops, and initial translation maneuvers.
- **Evaluation / Validation Partition (30%)**: Final 1,092 matched frames ($t \in [1403636708.25, 1403636762.74]$ s, duration: $54.54$ s). Includes return loops, descent, and touchdown.
- **Leakage Prevention**: No random shuffling across time steps. Input feature scaling (Z-score normalization: $\mu, \sigma$) is fitted **strictly on the training partition** and applied frozen to the evaluation partition.
- **Generalization Boundary Disclosure**: Because only `MH_01_easy` is presently available in the local repository, generalization across unseen trajectories/environments cannot be formally asserted until additional sequences are ingested.

---

## 8. COMPUTATIONAL BUDGET & REAL-TIME EDGE VIABILITY

| Metric | Target Budget | Expected Edge Performance | Status |
| :--- | :--- | :--- | :--- |
| **Model Size** | $< 100$ KB | $\approx 15$ KB (PyTorch/NumPy weights) | Within budget |
| **Inference Latency** | $< 1.0$ ms / frame | $\approx 0.08 - 0.15$ ms / frame (single CPU thread) | Well within budget |
| **Memory Allocation** | $< 2$ MB RAM | $\approx 0.4$ MB | Negligible |
| **Inference Rate** | $\ge 50$ FPS | $> 500$ FPS (standalone inference) | Real-time ready |

---

## 9. RISK MANAGEMENT & DEGRADATION POLICIES

1. **Overfitting to Known Trajectory**: Controlled via aggressive L2 weight regularization ($\lambda = 10^{-4}$), dropout ($p = 0.1$), and early stopping on validation loss.
2. **Prediction Explosion**: Output velocity residuals are clamped to physically reasonable limits:
   $$\Delta \hat{\mathbf{v}}_{clamped} = \text{clip}(\Delta \hat{\mathbf{v}}, -3.0 \text{ m/s}, +3.0 \text{ m/s})$$
3. **Low Confidence Fallback**: When $c < 0.2$, the residual application is linearly attenuated:
   $$\mathbf{v}_{corrected} = \mathbf{v}_{VIO} + \max(0, 2(c - 0.2)) \cdot \Delta \hat{\mathbf{v}}$$
   ensuring the estimator seamlessly defaults to classical VIO under degraded visual conditions.
4. **NaN/Inf Numerical Safety**: All input features are scrubbed with `np.isfinite()`. Any non-finite feature vector immediately triggers a zero-residual bypass.
