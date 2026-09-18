"""Unit and regression tests for GEONAV-AI Phase 7 ML subsystem."""

import math
import numpy as np
import pytest
import torch

from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.ml.dataset import FeatureScaler
from geonav.ml.features import FeatureExtractor, MLFeatures
from geonav.ml.inference import MLVelocityCorrector
from geonav.ml.model import EdgeMLP, NumpyEdgeMLP
from geonav.state import NavigationState
from geonav.vio.types import VisualTrackingResult


def test_1_feature_extractor_dimensions():
    """Test 1: Feature extractor produces exactly 18-element feature vector."""
    extractor = FeatureExtractor()
    assert extractor.feature_dim == 18

    dummy_state = NavigationState(
        timestamp=1.0,
        position=(0.0, 0.0, 0.0),
        velocity=(1.0, -0.5, 0.2),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    feat = extractor.extract(
        visual_result=None,
        pts_prev=None,
        pts_curr=None,
        imu_samples=None,
        current_state=dummy_state,
        dt=0.05,
    )

    arr = feat.to_array()
    assert arr.shape == (18,)
    assert arr.dtype == np.float32
    assert np.all(np.isfinite(arr))
    assert arr[13] == pytest.approx(1.0, abs=1e-5)   # vx
    assert arr[14] == pytest.approx(-0.5, abs=1e-5)  # vy
    assert arr[15] == pytest.approx(0.2, abs=1e-5)   # vz


def test_2_feature_extractor_nan_inf_safety():
    """Test 2: Non-finite inputs are safely scrubbed to finite values."""
    extractor = FeatureExtractor()

    # Mock IMU sample with non-finite acceleration/gyro
    class MockBadIMU:
        linear_acceleration = (float("nan"), 0.0, 9.8)
        angular_velocity = (0.0, float("inf"), 0.0)

    bad_imu = [MockBadIMU()]
    valid_state = NavigationState(
        timestamp=1.0,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    feat = extractor.extract(
        visual_result=None,
        pts_prev=None,
        pts_curr=None,
        imu_samples=bad_imu,
        current_state=valid_state,
        dt=float("nan"),  # Non-finite dt
    )

    arr = feat.to_array()
    assert arr.shape == (18,)
    assert np.all(np.isfinite(arr))

    # Explicit non-finite values in MLFeatures
    raw_bad_feat = MLFeatures(
        tracked_features=float("nan"),
        inlier_ratio=float("inf"),
        mean_flow_mag=float("-inf"),
        std_flow_mag=0.0,
        mean_accel=(float("nan"), 0.0, 0.0),
        var_accel=(0.0, float("inf"), 0.0),
        mean_gyro=(0.0, 0.0, 0.0),
        vio_velocity=(float("nan"), 0.0, 0.0),
        vio_v_norm=float("inf"),
        dt=0.05,
    )
    arr2 = raw_bad_feat.to_array()
    assert arr2.shape == (18,)
    assert np.all(np.isfinite(arr2))


def test_3_feature_scaler_fit_transform():
    """Test 3: FeatureScaler normalizes training data and handles zero variance."""
    np.random.seed(42)
    X = np.random.normal(loc=5.0, scale=2.0, size=(100, 18)).astype(np.float32)
    X[:, 0] = 3.0  # Constant column (zero variance)

    scaler = FeatureScaler()
    scaler.fit(X)

    X_norm = scaler.transform(X)
    assert X_norm.shape == (100, 18)
    assert np.all(np.isfinite(X_norm))
    # Constant column should be scaled without div by zero
    assert np.all(X_norm[:, 0] == 0.0)
    # Variable column should have approx zero mean and unit variance
    assert np.mean(X_norm[:, 1]) == pytest.approx(0.0, abs=1e-2)
    assert np.std(X_norm[:, 1]) == pytest.approx(1.0, abs=1e-2)


def test_4_edge_mlp_forward_shapes():
    """Test 4: EdgeMLP PyTorch model produces correct output shapes and bounded confidence."""
    model = EdgeMLP(input_dim=18)
    x = torch.randn(8, 18)

    delta_v, conf = model(x)
    assert delta_v.shape == (8, 3)
    assert conf.shape == (8, 1)

    # Sigmoid confidence must be strictly in [0, 1]
    assert torch.all(conf >= 0.0)
    assert torch.all(conf <= 1.0)


def test_5_numpy_edge_mlp_parity():
    """Test 5: NumpyEdgeMLP reproduces PyTorch EdgeMLP outputs to high precision."""
    torch.manual_seed(42)
    torch_model = EdgeMLP(input_dim=18)
    torch_model.eval()

    weights_dict = torch_model.export_weights_dict()
    numpy_model = NumpyEdgeMLP(weights_dict)

    x_np = np.random.normal(0, 1, size=(18,)).astype(np.float32)
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        dv_t, conf_t = torch_model(x_t)

    dv_np, conf_np = numpy_model.forward(x_np)

    assert np.allclose(dv_np, dv_t.squeeze(0).numpy(), atol=1e-5)
    assert conf_np == pytest.approx(float(conf_t.item()), abs=1e-5)


def test_6_ml_velocity_corrector_clamping():
    """Test 6: Velocity corrector clamps extreme predictions to max_correction_mps."""
    corrector = MLVelocityCorrector(max_correction_mps=2.0)
    corrector.is_loaded = True

    # Mock model returning massive delta_v
    class MockModel:
        def forward(self, x):
            return np.array([10.0, -15.0, 20.0], dtype=np.float32), 0.95

    corrector.model = MockModel()

    feat = MLFeatures(
        tracked_features=100, inlier_ratio=0.8, mean_flow_mag=2.0, std_flow_mag=0.5,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.05,
    )

    delta_v_gated, conf = corrector.predict(feat)
    assert delta_v_gated[0] == pytest.approx(2.0, abs=1e-5)
    assert delta_v_gated[1] == pytest.approx(-2.0, abs=1e-5)
    assert delta_v_gated[2] == pytest.approx(2.0, abs=1e-5)
    assert conf == pytest.approx(0.95, abs=1e-5)


def test_7_ml_velocity_corrector_confidence_attenuation():
    """Test 7: Low confidence attenuates velocity correction to zero."""
    corrector = MLVelocityCorrector(min_confidence=0.25)
    corrector.is_loaded = True

    # Mock model returning low confidence
    class MockLowConfModel:
        def forward(self, x):
            return np.array([1.5, -1.0, 0.5], dtype=np.float32), 0.10  # < 0.25

    corrector.model = MockLowConfModel()

    feat = MLFeatures(
        tracked_features=10, inlier_ratio=0.2, mean_flow_mag=0.1, std_flow_mag=0.01,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(0, 0, 0), vio_v_norm=0.0, dt=0.05,
    )

    delta_v_gated, conf = corrector.predict(feat)
    # With confidence 0.10 < 0.25, correction must be zero
    assert np.all(delta_v_gated == 0.0)
    assert conf == pytest.approx(0.10, abs=1e-5)


def test_8_state_correction_integration():
    """Test 8: State correction integrates position correctly with corrected velocity."""
    corrector = MLVelocityCorrector()
    corrector.is_loaded = True

    # Mock model returning constant [0.2, 0.0, -0.5] with full confidence
    class MockConstantModel:
        def forward(self, x):
            return np.array([0.2, 0.0, -0.5], dtype=np.float32), 1.0

    corrector.model = MockConstantModel()

    prev_state = NavigationState(
        timestamp=1.0,
        position=(10.0, 5.0, 2.0),
        velocity=(1.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    raw_curr = NavigationState(
        timestamp=1.1,
        position=(10.1, 5.0, 2.0),
        velocity=(1.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    feat = MLFeatures(
        tracked_features=100, inlier_ratio=0.9, mean_flow_mag=2.0, std_flow_mag=0.5,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.1,
    )

    corr_state = corrector.correct_state(prev_state, raw_curr, feat, dt=0.1)

    # v_corrected = [1.0, 0.0, 0.0] + [0.2, 0.0, -0.5] = [1.2, 0.0, -0.5]
    assert corr_state.velocity[0] == pytest.approx(1.2, abs=1e-5)
    assert corr_state.velocity[1] == pytest.approx(0.0, abs=1e-5)
    assert corr_state.velocity[2] == pytest.approx(-0.5, abs=1e-5)

    # p_corrected = [10.0, 5.0, 2.0] + [1.2, 0.0, -0.5] * 0.1 = [10.12, 5.0, 1.95]
    assert corr_state.position[0] == pytest.approx(10.12, abs=1e-5)
    assert corr_state.position[1] == pytest.approx(5.0, abs=1e-5)
    assert corr_state.position[2] == pytest.approx(1.95, abs=1e-5)


# ── Phase 7 Verification Gate: A11 Confidence Gating Tests ──────────────────

def test_9_unloaded_model_returns_zero_correction():
    """Test 9 (A11): Unloaded corrector returns zero correction and zero confidence."""
    corrector = MLVelocityCorrector()  # No weights path → is_loaded=False
    assert not corrector.is_loaded

    feat = MLFeatures(
        tracked_features=100, inlier_ratio=0.9, mean_flow_mag=2.0, std_flow_mag=0.5,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.05,
    )
    delta_v, conf = corrector.predict(feat)
    assert np.all(delta_v == 0.0), "Unloaded model must return zero correction"
    assert conf == pytest.approx(0.0, abs=1e-9), "Unloaded model must return zero confidence"


def test_10_nan_feature_input_produces_finite_correction():
    """Test 10 (A11): NaN in feature vector is sanitized before inference."""
    corrector = MLVelocityCorrector()
    corrector.is_loaded = True

    class MockFiniteModel:
        def forward(self, x):
            assert np.all(np.isfinite(x)), "Model received non-finite input"
            return np.zeros(3, dtype=np.float32), 0.8

    corrector.model = MockFiniteModel()

    feat = MLFeatures(
        tracked_features=float("nan"),
        inlier_ratio=float("inf"),
        mean_flow_mag=float("-inf"),
        std_flow_mag=0.0,
        mean_accel=(float("nan"), 0.0, 0.0),
        var_accel=(0.0, 0.0, 0.0),
        mean_gyro=(0.0, 0.0, 0.0),
        vio_velocity=(0.0, 0.0, 0.0),
        vio_v_norm=0.0,
        dt=0.05,
    )
    # to_array() sanitizes NaN/Inf; predict() should not crash
    delta_v, conf = corrector.predict(feat)
    assert np.all(np.isfinite(delta_v))
    assert np.isfinite(conf)


def test_11_inf_delta_v_prediction_returns_zero():
    """Test 11 (A11): Non-finite delta_v prediction falls back to zero correction."""
    corrector = MLVelocityCorrector()
    corrector.is_loaded = True

    class MockInfModel:
        def forward(self, x):
            return np.array([float("inf"), float("nan"), -float("inf")], dtype=np.float32), 0.9

    corrector.model = MockInfModel()
    feat = MLFeatures(
        tracked_features=50, inlier_ratio=0.7, mean_flow_mag=1.0, std_flow_mag=0.3,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.05,
    )
    delta_v, conf = corrector.predict(feat)
    assert np.all(delta_v == 0.0), "Non-finite prediction must produce zero correction"
    assert conf == pytest.approx(0.0, abs=1e-9)


def test_12_confidence_boundary_exactly_at_min():
    """Test 12 (A11): Confidence exactly at min_confidence → zero correction."""
    corrector = MLVelocityCorrector(min_confidence=0.20)
    corrector.is_loaded = True

    class MockExactMinConf:
        def forward(self, x):
            return np.array([1.0, 1.0, 1.0], dtype=np.float32), 0.20  # Exactly at threshold

    corrector.model = MockExactMinConf()
    feat = MLFeatures(
        tracked_features=80, inlier_ratio=0.8, mean_flow_mag=1.5, std_flow_mag=0.4,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.05,
    )
    delta_v, conf = corrector.predict(feat)
    # conf == min_confidence → attenuation = 0.0
    assert np.all(delta_v == 0.0), "conf == min_confidence must give zero correction"


def test_13_confidence_mid_ramp_partial_attenuation():
    """Test 13 (A11): Confidence in linear ramp zone yields partial correction."""
    min_conf = 0.20
    corrector = MLVelocityCorrector(min_confidence=min_conf)
    corrector.is_loaded = True

    conf_val = 0.35  # Mid-ramp: 0.20 < 0.35 < 0.50
    predicted_dv = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    class MockMidRampModel:
        def forward(self, x):
            return predicted_dv, conf_val

    corrector.model = MockMidRampModel()
    feat = MLFeatures(
        tracked_features=100, inlier_ratio=0.85, mean_flow_mag=2.0, std_flow_mag=0.5,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.05,
    )
    delta_v, conf = corrector.predict(feat)
    expected_attenuation = (conf_val - min_conf) / 0.3  # = 0.5
    expected_dv_x = float(predicted_dv[0]) * expected_attenuation
    assert delta_v[0] == pytest.approx(expected_dv_x, abs=1e-5)


def test_14_high_confidence_full_correction():
    """Test 14 (A11): Confidence above ramp ceiling applies full unclamped correction."""
    corrector = MLVelocityCorrector(min_confidence=0.20, max_correction_mps=5.0)
    corrector.is_loaded = True

    predicted_dv = np.array([0.5, -0.3, 0.1], dtype=np.float32)

    class MockHighConfModel:
        def forward(self, x):
            return predicted_dv, 0.95  # Well above 0.50 ceiling

    corrector.model = MockHighConfModel()
    feat = MLFeatures(
        tracked_features=150, inlier_ratio=0.95, mean_flow_mag=3.0, std_flow_mag=0.2,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(2, 0, 0), vio_v_norm=2.0, dt=0.05,
    )
    delta_v, conf = corrector.predict(feat)
    # attenuation = 1.0 → full correction passed through
    assert delta_v[0] == pytest.approx(float(predicted_dv[0]), abs=1e-5)
    assert delta_v[1] == pytest.approx(float(predicted_dv[1]), abs=1e-5)
    assert delta_v[2] == pytest.approx(float(predicted_dv[2]), abs=1e-5)


def test_15_first_frame_position_from_raw_vio():
    """Test 15 (A11): On first frame (prev_state=None), position is copied from raw VIO."""
    corrector = MLVelocityCorrector()
    corrector.is_loaded = True

    class MockModel:
        def forward(self, x):
            return np.array([0.5, 0.5, 0.5], dtype=np.float32), 0.9

    corrector.model = MockModel()
    raw = NavigationState(
        timestamp=0.05,
        position=(3.0, 1.5, -0.5),
        velocity=(1.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    feat = MLFeatures(
        tracked_features=100, inlier_ratio=0.9, mean_flow_mag=2.0, std_flow_mag=0.5,
        mean_accel=(0, 0, 9.8), var_accel=(0, 0, 0), mean_gyro=(0, 0, 0),
        vio_velocity=(1, 0, 0), vio_v_norm=1.0, dt=0.05,
    )
    corrected = corrector.correct_state(
        prev_corrected_state=None, raw_current_state=raw, features=feat, dt=0.05
    )
    # First frame → position copied from raw, only velocity is corrected
    assert corrected.position[0] == pytest.approx(3.0, abs=1e-5)
    assert corrected.position[1] == pytest.approx(1.5, abs=1e-5)
    assert corrected.position[2] == pytest.approx(-0.5, abs=1e-5)
    # Velocity should be corrected: [1.0+0.5, 0.0+0.5, 0.0+0.5] = [1.5, 0.5, 0.5]
    assert corrected.velocity[0] == pytest.approx(1.5, abs=1e-5)


def test_16_feature_vector_index_mapping():
    """Test 16 (A11): Verify exact feature index assignments in to_array()."""
    feat = MLFeatures(
        tracked_features=42.0,
        inlier_ratio=0.75,
        mean_flow_mag=3.14,
        std_flow_mag=1.1,
        mean_accel=(0.1, 0.2, 9.81),
        var_accel=(0.01, 0.02, 0.03),
        mean_gyro=(0.05, -0.1, 0.3),
        vio_velocity=(1.2, -0.4, 0.1),
        vio_v_norm=1.27,
        dt=0.05,
    )
    arr = feat.to_array()
    assert arr[0]  == pytest.approx(42.0,  abs=1e-5)   # tracked_features
    assert arr[1]  == pytest.approx(0.75,  abs=1e-5)   # inlier_ratio
    assert arr[2]  == pytest.approx(3.14,  abs=1e-5)   # mean_flow_mag
    assert arr[3]  == pytest.approx(1.1,   abs=1e-5)   # std_flow_mag
    assert arr[4]  == pytest.approx(0.1,   abs=1e-5)   # mean_accel[0]
    assert arr[5]  == pytest.approx(0.2,   abs=1e-5)   # mean_accel[1]
    assert arr[6]  == pytest.approx(9.81,  abs=1e-5)   # mean_accel[2]
    assert arr[7]  == pytest.approx(0.01,  abs=1e-5)   # var_accel[0]
    assert arr[8]  == pytest.approx(0.02,  abs=1e-5)   # var_accel[1]
    assert arr[9]  == pytest.approx(0.03,  abs=1e-5)   # var_accel[2]
    assert arr[10] == pytest.approx(0.05,  abs=1e-5)   # mean_gyro[0]
    assert arr[11] == pytest.approx(-0.1,  abs=1e-5)   # mean_gyro[1]
    assert arr[12] == pytest.approx(0.3,   abs=1e-5)   # mean_gyro[2]
    assert arr[13] == pytest.approx(1.2,   abs=1e-5)   # vio_velocity[0]
    assert arr[14] == pytest.approx(-0.4,  abs=1e-5)   # vio_velocity[1]
    assert arr[15] == pytest.approx(0.1,   abs=1e-5)   # vio_velocity[2]
    assert arr[16] == pytest.approx(1.27,  abs=1e-3)   # vio_v_norm (approx due to float cast)
    assert arr[17] == pytest.approx(0.05,  abs=1e-5)   # dt
