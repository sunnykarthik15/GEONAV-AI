# GEONAV-AI — Phase 7 Data Leakage Audit

**Date:** 2026-09-19  
**Auditor:** Phase 7 Verification Gate (automated + manual source inspection)  
**Dataset:** EuRoC MAV MH_01_easy  
**Status:** ✅ PASS — No data leakage detected

---

## Audit Scope

This document performs a rigorous, step-by-step audit of every component in the Phase 7 ML pipeline, tracing the flow of ground-truth (GT) data and future information across:

1. Feature extraction
2. Label generation
3. Train/validation split
4. Model training
5. Inference
6. Velocity correction
7. Position integration
8. Metric computation

**Central Question:**
> Can the model receive any ground-truth information, directly or indirectly, during inference?

**Answer: NO.**  
Evidence is documented below for each stage.

---

## 1. Feature Extraction

**Source:** `src/geonav/ml/features.py` — `FeatureExtractor.extract()`

The feature vector has 18 dimensions derived exclusively from:

| Group | Source | Available at Inference? |
|---|---|---|
| Visual tracking (4) | `VisualTrackingResult` from current frame | ✅ Yes |
| IMU kinematics (9) | `IMUSample` list for current inter-frame interval | ✅ Yes |
| VIO state (4) | `NavigationState` produced by classical VIO at current step | ✅ Yes |
| Time delta (1) | `dt = t_curr - t_prev` (from consecutive camera timestamps) | ✅ Yes |

**GT involvement:** NONE. No ground-truth position, velocity, or orientation enters any of the 18 features. The VIO state used (`vio_velocity`) is the estimate produced by the classical pipeline, not GT.

**Future information:** NONE. All features are derived from data available at the current camera frame timestamp. Optical flow uses the previous frame and current frame only (no lookahead).

---

## 2. Label Generation

**Source:** `src/geonav/ml/dataset.py` — `extract_vio_dataset()`

Training targets are defined as:
```
y_i = v_GT(t_i) - v_VIO(t_i)    [velocity residual, m/s, shape (3,)]
```

GT velocity is sourced from `EurocDataset.ground_truth()` and associated with estimated states via `associate_timestamps()` using a 10 ms time-difference threshold.

**GT involvement during training-label generation:** ✅ Intentional and correct. GT is used **exclusively** for computing the offline training label. This is the standard supervised-learning paradigm.

**GT involvement during inference:** ❌ NONE. The label-generation code path (`dataset.py`) is never called during inference. Inference only calls `FeatureExtractor.extract()` followed by `MLVelocityCorrector.predict()`.

---

## 3. Train / Validation Split

**Source:** `src/geonav/ml/dataset.py` — `extract_vio_dataset()`, lines 180–194

```python
n_train = int(n_total * split_ratio)   # split_ratio = 0.70
X_train_raw = X[:n_train]              # First 70% chronologically
X_eval_raw  = X[n_train:]             # Last 30% chronologically
```

**Split type:** Strict chronological partition.  
**Split ratio:** 70% train / 30% validation.  
**Shuffling:** NONE before splitting (samples are in temporal order from the VIO run).  
**Scaler fitting:** `FeatureScaler` is fitted **exclusively on training data** (`X_train_raw`), then applied to both splits. No validation-data statistics enter the scaler.

### Timestamp Ranges (from training run output)

| Split | Samples | Time Range |
|---|---|---|
| Train | 2,546 | 1,403,636,580.86 s → 1,403,636,708.11 s |
| Validation | 1,092 | 1,403,636,708.16 s → 1,403,636,762.71 s |

**Temporal gap:** 0.05 s between last training sample and first validation sample — no overlap.

### Correct Terminology

> [!IMPORTANT]
> The validation split is a **chronologically held-out temporal segment from the same MH_01_easy sequence**.  
> It is **NOT** an unseen-flight test and should NOT be described as "unseen trajectory generalization."  
> Only one EuRoC sequence (MH_01_easy) is available. True cross-sequence generalization has not been demonstrated and is an explicit limitation of this study.

---

## 4. Model Training

**Source:** `src/geonav/ml/train.py` — `train_edge_mlp()`

Training uses only `X_train` and `y_train`. Validation loss is computed on `X_eval`/`y_eval` for monitoring only (not used for gradient updates). PyTorch DataLoader shuffles within the training set (acceptable — temporal order is not required within the training set for this regression task since features are local/single-step).

**GT involvement:** Only indirectly, through pre-computed `y_train` labels (which were derived from GT velocity). GT is never passed to the model during training.

**Hyperparameter tuning on validation:** The validation set (last 30%) was NOT used to tune hyperparameters. Epochs (40), learning rate (1e-3), and batch size (32) were set before training. No hyperparameter search was performed.

---

## 5. Inference

**Source:** `src/geonav/ml/inference.py` — `MLVelocityCorrector.predict()`

Inference path:
```
MLFeatures (18-dim) 
  → FeatureScaler.transform() [uses scaler trained on training data only]
  → NumpyEdgeMLP.forward()   [pure NumPy, no GT access]
  → (delta_v_pred, confidence)
  → confidence gating
  → delta_v_gated
```

**GT involvement:** NONE. The inference chain accesses only the feature vector (causal, no GT) and pre-loaded model weights (trained offline).

**Future information:** NONE. Features are constructed from past+current frame data only.

**Confidence computation:** Sigmoid of a learned linear head. Output ∈ [0, 1]. No GT enters the confidence computation.

---

## 6. Velocity Correction

**Source:** `src/geonav/ml/inference.py` — `MLVelocityCorrector.correct_state()`, line 107

```python
v_corrected = v_raw + delta_v_gated
```

Where:
- `v_raw` = classical VIO velocity at current step (no GT)
- `delta_v_gated` = ML predicted residual, confidence-gated (no GT)

**GT involvement:** NONE.

---

## 7. Position Integration

**Source:** `src/geonav/ml/inference.py` — `correct_state()`, lines 109–115

```python
if prev_corrected_state is None:
    p_corrected = np.array(raw_current_state.position)  # First frame: raw VIO position
else:
    p_prev = np.array(prev_corrected_state.position)
    p_corrected = p_prev + v_corrected * dt              # Euler integration
```

**Integration method:** Forward Euler: `p_k = p_{k-1} + v_corrected * dt`

**dt source:** `t_curr - t_prev` from consecutive camera frame timestamps.

**Causality:** Integration only uses the *previously corrected* state `p_{k-1}` and current corrected velocity. No future states are accessed. No retroactive trajectory correction occurs.

**GT involvement:** NONE.

**Numerical safety:** `dt` is clamped to `[1e-4, 1.0]` in the feature extractor. An uninitialized `prev_corrected_state=None` falls back to the raw VIO position, preventing error accumulation on the first frame.

---

## 8. Metric Computation

**Source:** `src/geonav/ml/evaluate.py` — `run_hybrid_vio()`, `main()`

GT is loaded **after** the full VIO+ML pipeline run completes, solely for offline evaluation. The evaluation chain:

```
1. Run VIO + ML → collect baseline_states, hybrid_states
2. Load GT from EurocDataset.ground_truth()
3. Associate timestamps (10 ms threshold)
4. Compute ATE, velocity RMSE, RPE via TrajectoryEvaluator
```

GT is never passed into step 1. The call sequence enforces this: `run_hybrid_vio()` completes and returns before GT is loaded.

**ATE reported:** The reported 6.01 m is **raw ATE** (`metrics_hyb.raw_ate.rmse`). It is not SE(3)-aligned or Sim(3)-aligned ATE. This is confirmed by inspecting `evaluate.py` line 204:
```python
row("Full Trajectory Raw ATE RMSE", metrics_base.raw_ate.rmse, metrics_hyb.raw_ate.rmse, "m")
```

---

## Confidence Gating

**Source:** `src/geonav/ml/inference.py` — `predict()`, lines 76–83

```python
if conf <= min_confidence:           # conf <= 0.20 → full attenuation
    attenuation = 0.0
elif conf < (min_confidence + 0.3):  # 0.20 < conf < 0.50 → linear ramp
    attenuation = (conf - min_confidence) / 0.3
else:                                # conf >= 0.50 → full correction
    attenuation = 1.0

delta_v_gated = delta_v_clamped * attenuation
```

- **Mathematical definition:** Piecewise linear gate in [0, 1].
- **Output range:** Correction magnitude ∈ [0, `max_correction_mps`] = [0, 3.0 m/s].
- **GT involvement:** NONE. Confidence is the Sigmoid output of a learned linear projection of the penultimate MLP layer.
- **Low-confidence fallback:** When `conf ≤ 0.20`, the correction is identically zero → pure classical VIO state is returned.

---

## Summary Table

| Stage | GT Used? | Future Data Used? | Causal? |
|---|---|---|---|
| Feature extraction | ❌ No | ❌ No | ✅ Yes |
| Label generation (offline) | ✅ Yes (intended) | ❌ No | N/A |
| Train/val split | N/A | ❌ No (chronological) | ✅ Yes |
| Model training | ❌ No (labels only) | ❌ No | ✅ Yes |
| Inference | ❌ No | ❌ No | ✅ Yes |
| Velocity correction | ❌ No | ❌ No | ✅ Yes |
| Position integration | ❌ No | ❌ No | ✅ Yes |
| Metric computation | ✅ Yes (evaluation only) | ❌ No | N/A |

---

## Limitations

1. **Single-sequence evaluation only.** Cross-sequence generalization (e.g., training on MH_02 and testing on MH_01) has not been demonstrated.

2. **Temporal bias risk.** Because training and validation come from the same sequence, the model may learn trajectory-specific patterns (e.g., the corridor structure of Machine Hall). This is bounded by Diagnostic B (dt-removal) and Diagnostic C (alternative split) in section A8 of the verification gate.

3. **Confidence calibration.** The confidence output is a sigmoid from a learned linear head regularized by `(1 - conf)^2` loss. It is not calibrated via a held-out calibration set. Confidence values should be treated as relative quality indicators, not probability estimates.

---

## Verdict

> **PHASE 7 DATA LEAKAGE AUDIT — PASS**
>
> Ground-truth information does not enter inference, feature extraction, confidence computation, or position integration. GT is used exclusively for offline training-label generation, which is the correct and standard practice for supervised learning. The train/validation split is strictly chronological with no temporal overlap. The feature scaler is fitted on training data only.
