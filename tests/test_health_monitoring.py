"""Tests for GEONAV-AI health monitoring state machine (Phase 8 B14)."""

import pytest
from geonav.hardware.health import HealthMonitor, HealthState, SystemHealth


def _make_monitor(**kwargs) -> HealthMonitor:
    defaults = dict(
        min_features_for_tracking=10,
        min_inlier_ratio_for_tracking=0.3,
        degraded_timeout_frames=5,
    )
    defaults.update(kwargs)
    return HealthMonitor(**defaults)


def _tracking_update(monitor: HealthMonitor, timestamp: float = 1.0,
                     features: int = 100, inlier_ratio: float = 0.9) -> SystemHealth:
    """Helper: produce a healthy tracking update."""
    return monitor.update(
        timestamp=timestamp,
        vio_initialized=True,
        tracked_features=features,
        inlier_ratio=inlier_ratio,
        visual_success=True,
        imu_available=True,
        ml_confidence=0.8,
        position=(0.1, 0.2, 0.3),
        velocity=(0.5, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _degraded_update(monitor: HealthMonitor, timestamp: float = 1.0) -> SystemHealth:
    """Helper: produce a degraded (no visual) update."""
    return monitor.update(
        timestamp=timestamp,
        vio_initialized=True,
        tracked_features=2,
        inlier_ratio=0.0,
        visual_success=False,
        imu_available=True,
        ml_confidence=0.0,
        position=(0.1, 0.2, 0.3),
        velocity=(0.5, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )


# ── Initialization ───────────────────────────────────────────────────────────

def test_monitor_initial_state_is_initializing():
    """Test: Health monitor starts in INITIALIZING state."""
    monitor = _make_monitor()
    assert monitor.state == HealthState.INITIALIZING


def test_monitor_stays_initializing_without_vio():
    """Test: Monitor remains INITIALIZING when VIO not yet initialized."""
    monitor = _make_monitor()
    h = monitor.update(
        timestamp=0.1, vio_initialized=False, tracked_features=0,
        inlier_ratio=0.0, visual_success=False, imu_available=True, ml_confidence=0.0,
    )
    assert h.state == HealthState.INITIALIZING


def test_monitor_transitions_to_tracking_on_init():
    """Test: Sufficient features + VIO init triggers INITIALIZING → TRACKING."""
    monitor = _make_monitor()
    h = _tracking_update(monitor, timestamp=0.1)
    assert h.state == HealthState.TRACKING
    assert h.vio_initialized


# ── Tracking ─────────────────────────────────────────────────────────────────

def test_monitor_stays_tracking_on_good_frames():
    """Test: Monitor remains TRACKING across multiple good frames."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=0.1)  # → TRACKING
    for i in range(10):
        h = _tracking_update(monitor, timestamp=0.1 + (i + 1) * 0.05)
    assert h.state == HealthState.TRACKING


def test_monitor_tracking_to_degraded_on_visual_failure():
    """Test: TRACKING → DEGRADED when visual tracking fails."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=0.1)  # → TRACKING
    h = _degraded_update(monitor, timestamp=0.15)
    assert h.state == HealthState.DEGRADED


# ── Degraded ─────────────────────────────────────────────────────────────────

def test_monitor_degraded_recovers_to_tracking():
    """Test: DEGRADED → TRACKING when visual tracking recovers."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=0.1)  # → TRACKING
    _degraded_update(monitor, timestamp=0.15)  # → DEGRADED
    h = _tracking_update(monitor, timestamp=0.20)  # → TRACKING
    assert h.state == HealthState.TRACKING
    assert h.consecutive_degraded_frames == 0


def test_monitor_degraded_timeout_to_lost():
    """Test: DEGRADED → LOST after degraded_timeout_frames consecutive degraded frames."""
    monitor = _make_monitor(degraded_timeout_frames=3)
    _tracking_update(monitor, timestamp=0.1)  # → TRACKING
    t = 0.15
    last = None
    for _ in range(5):
        last = _degraded_update(monitor, timestamp=t)
        t += 0.05
    # After 3 degraded frames, should transition to LOST
    assert last.state == HealthState.LOST


def test_monitor_degraded_counter_increments():
    """Test: consecutive_degraded_frames increments each degraded frame."""
    monitor = _make_monitor(degraded_timeout_frames=20)
    _tracking_update(monitor, timestamp=0.1)
    for i in range(3):
        h = _degraded_update(monitor, timestamp=0.15 + i * 0.05)
    assert h.consecutive_degraded_frames >= 1


# ── Numerical failure ────────────────────────────────────────────────────────

def test_monitor_nan_position_transitions_to_lost():
    """Test: NaN in position immediately transitions any state to LOST."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=0.1)  # → TRACKING
    h = monitor.update(
        timestamp=0.15,
        vio_initialized=True, tracked_features=100, inlier_ratio=0.9,
        visual_success=True, imu_available=True, ml_confidence=0.8,
        position=(float("nan"), 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    assert h.state == HealthState.LOST
    assert not h.numerically_valid


def test_monitor_inf_velocity_transitions_to_lost():
    """Test: Inf in velocity immediately transitions to LOST."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=0.1)
    h = monitor.update(
        timestamp=0.15,
        vio_initialized=True, tracked_features=100, inlier_ratio=0.9,
        visual_success=True, imu_available=True, ml_confidence=0.8,
        position=(0.0, 0.0, 0.0),
        velocity=(float("inf"), 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    assert h.state == HealthState.LOST


def test_monitor_nan_in_initializing_state_goes_to_lost():
    """Test: Numerical failure in INITIALIZING state still → LOST."""
    monitor = _make_monitor()
    # Still in INITIALIZING
    h = monitor.update(
        timestamp=0.05,
        vio_initialized=False, tracked_features=0, inlier_ratio=0.0,
        visual_success=False, imu_available=True, ml_confidence=0.0,
        position=(float("nan"), 0.0, 0.0),
    )
    assert h.state == HealthState.LOST


# ── Reset ────────────────────────────────────────────────────────────────────

def test_monitor_reset_returns_to_initializing():
    """Test: reset() clears all state and returns to INITIALIZING."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=0.1)  # → TRACKING
    _degraded_update(monitor, timestamp=0.15)  # → DEGRADED
    monitor.reset()
    assert monitor.state == HealthState.INITIALIZING
    # First update after reset should start from INITIALIZING logic
    h = monitor.update(
        timestamp=0.0,
        vio_initialized=False, tracked_features=0, inlier_ratio=0.0,
        visual_success=False, imu_available=True, ml_confidence=0.0,
    )
    assert h.state == HealthState.INITIALIZING


def test_monitor_reset_after_lost_restarts_correctly():
    """Test: reset() after LOST allows full re-initialization."""
    monitor = _make_monitor(degraded_timeout_frames=2)
    _tracking_update(monitor, timestamp=0.1)
    for i in range(5):
        _degraded_update(monitor, timestamp=0.15 + i * 0.05)  # → LOST
    assert monitor.state == HealthState.LOST
    monitor.reset()
    assert monitor.state == HealthState.INITIALIZING
    h = _tracking_update(monitor, timestamp=1.0)
    assert h.state == HealthState.TRACKING


# ── Timestamp validation ─────────────────────────────────────────────────────

def test_monitor_detects_non_monotonic_timestamp():
    """Test: Non-monotonically increasing timestamps are flagged."""
    monitor = _make_monitor()
    _tracking_update(monitor, timestamp=1.0)
    h = _tracking_update(monitor, timestamp=0.5)  # Earlier timestamp
    assert not h.timestamp_valid


def test_monitor_valid_monotonic_timestamps():
    """Test: Monotonically increasing timestamps are reported as valid."""
    monitor = _make_monitor()
    h1 = _tracking_update(monitor, timestamp=1.0)
    h2 = _tracking_update(monitor, timestamp=1.05)
    assert h2.timestamp_valid
