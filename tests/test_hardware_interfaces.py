"""Tests for GEONAV-AI hardware abstraction layer interfaces (Phase 8 B14)."""

import pytest
import numpy as np

from geonav.hardware.interfaces import CameraSource, IMUSource, NavigationOutput, UAVIntegrationAdapter
from geonav.hardware.nav_output import NavigationMessage, ConsoleNavigationOutput, TrackingStatus
from geonav.hardware.euroc_adapter import EurocCameraSource, EurocIMUSource
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample


# ── Minimal concrete implementations for abstract interface testing ──────────

class _MockCameraSource(CameraSource):
    """Minimal CameraSource implementation for interface testing."""

    def __init__(self, frames=None):
        self._frames = frames or []
        self._idx = 0

    def frames(self):
        for f in self._frames:
            yield f

    def camera_matrix(self):
        return np.eye(3, dtype=np.float64)

    def reset(self):
        self._idx = 0


class _MockIMUSource(IMUSource):
    """Minimal IMUSource implementation for interface testing."""

    def __init__(self, samples=None):
        self._samples = samples or []

    def samples(self):
        for s in self._samples:
            yield s

    def reset(self):
        pass


class _MockNavigationOutput(NavigationOutput):
    """NavigationOutput that records all published messages."""

    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)

    def close(self):
        self.published.clear()


class _MockUAVAdapter(UAVIntegrationAdapter):
    """Minimal UAVIntegrationAdapter implementation."""

    def __init__(self):
        self._connected = False
        self.sent_messages = []

    def send_navigation_state(self, message):
        self.sent_messages.append(message)

    def is_connected(self):
        return self._connected

    def close(self):
        self._connected = False


# ── NavigationMessage Tests ──────────────────────────────────────────────────

def test_navigation_message_valid_state():
    """Test: NavigationMessage with valid finite values is marked numerically valid."""
    msg = NavigationMessage(
        timestamp=1.0,
        position=(1.0, 2.0, 3.0),
        velocity=(0.5, 0.0, -0.1),
        orientation=(1.0, 0.0, 0.0, 0.0),
        tracking_status=TrackingStatus.TRACKING,
        ml_confidence=0.85,
        tracked_features=120,
        inlier_ratio=0.9,
    )
    assert msg.is_numerically_valid
    assert msg.tracking_status == TrackingStatus.TRACKING
    assert msg.ml_confidence == pytest.approx(0.85, abs=1e-6)


def test_navigation_message_nan_position_invalid():
    """Test: NavigationMessage with NaN position is marked numerically invalid."""
    msg = NavigationMessage(
        timestamp=1.0,
        position=(float("nan"), 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    assert not msg.is_numerically_valid


def test_navigation_message_inf_velocity_invalid():
    """Test: NavigationMessage with Inf velocity is marked numerically invalid."""
    msg = NavigationMessage(
        timestamp=1.0,
        position=(0.0, 0.0, 0.0),
        velocity=(float("inf"), 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    assert not msg.is_numerically_valid


def test_navigation_message_to_dict_structure():
    """Test: NavigationMessage.to_dict() produces expected keys with correct units."""
    msg = NavigationMessage(
        timestamp=42.5,
        position=(1.0, 2.0, 3.0),
        velocity=(0.1, 0.2, 0.3),
        orientation=(1.0, 0.0, 0.0, 0.0),
        tracking_status=TrackingStatus.DEGRADED,
    )
    d = msg.to_dict()
    assert d["timestamp_s"] == pytest.approx(42.5, abs=1e-6)
    assert d["position_m"] == [1.0, 2.0, 3.0]
    assert d["velocity_mps"] == [0.1, 0.2, 0.3]
    assert d["tracking_status"] == "DEGRADED"
    assert "orientation_qwxyz" in d
    assert "ml_confidence" in d
    assert "is_numerically_valid" in d


def test_navigation_message_tracking_statuses():
    """Test: All TrackingStatus values are constructable in NavigationMessage."""
    for status in TrackingStatus:
        msg = NavigationMessage(
            timestamp=1.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            tracking_status=status,
        )
        assert msg.tracking_status == status


# ── CameraSource Interface Tests ─────────────────────────────────────────────

def test_camera_source_interface_contract():
    """Test: Mock CameraSource correctly implements the interface."""
    source = _MockCameraSource()
    assert hasattr(source, "frames")
    assert hasattr(source, "camera_matrix")
    assert hasattr(source, "reset")
    assert source.camera_matrix() is not None
    assert source.camera_matrix().shape == (3, 3)


def test_camera_source_empty_stream():
    """Test: Empty CameraSource produces no frames."""
    source = _MockCameraSource(frames=[])
    frames = list(source.frames())
    assert len(frames) == 0


def test_camera_source_reset():
    """Test: CameraSource.reset() can be called without error."""
    source = _MockCameraSource()
    source.reset()  # Should not raise


# ── IMUSource Interface Tests ────────────────────────────────────────────────

def test_imu_source_interface_contract():
    """Test: Mock IMUSource correctly implements the interface."""
    source = _MockIMUSource()
    assert hasattr(source, "samples")
    assert hasattr(source, "reset")


def test_imu_source_empty_stream():
    """Test: Empty IMUSource produces no samples."""
    source = _MockIMUSource(samples=[])
    samples = list(source.samples())
    assert len(samples) == 0


# ── NavigationOutput Interface Tests ────────────────────────────────────────

def test_navigation_output_publish_and_close():
    """Test: NavigationOutput correctly collects and clears messages."""
    output = _MockNavigationOutput()
    msg = NavigationMessage(
        timestamp=1.0, position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0),
    )
    output.publish(msg)
    output.publish(msg)
    assert len(output.published) == 2
    output.close()
    assert len(output.published) == 0


def test_console_navigation_output_no_crash():
    """Test: ConsoleNavigationOutput.publish() does not raise for valid messages."""
    output = ConsoleNavigationOutput(verbose=True)
    msg = NavigationMessage(
        timestamp=5.0, position=(1.0, 2.0, 3.0),
        velocity=(0.5, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0),
        tracking_status=TrackingStatus.TRACKING, ml_confidence=0.7,
        tracked_features=80,
    )
    output.publish(msg)  # Should not raise
    output.close()       # Should not raise


# ── UAVIntegrationAdapter Tests ──────────────────────────────────────────────

def test_uav_adapter_not_connected_initially():
    """Test: UAVIntegrationAdapter starts disconnected."""
    adapter = _MockUAVAdapter()
    assert not adapter.is_connected()


def test_uav_adapter_send_navigation_state():
    """Test: UAVIntegrationAdapter records sent navigation states."""
    adapter = _MockUAVAdapter()
    adapter._connected = True
    msg = NavigationMessage(
        timestamp=1.0, position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0),
    )
    adapter.send_navigation_state(msg)
    assert len(adapter.sent_messages) == 1


def test_uav_adapter_close():
    """Test: UAVIntegrationAdapter.close() transitions to disconnected."""
    adapter = _MockUAVAdapter()
    adapter._connected = True
    adapter.close()
    assert not adapter.is_connected()


# ── EuRoC Adapter Tests (requires dataset) ───────────────────────────────────

DATASET_PATH = "F:/GEONAV-AI/data/raw/euroc/MH_01_easy"


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATASET_PATH).is_dir(),
    reason="EuRoC dataset not available",
)
def test_euroc_camera_source_first_frame():
    """Test: EurocCameraSource yields at least one valid CameraFrame."""
    source = EurocCameraSource(DATASET_PATH)
    frames = []
    for i, f in enumerate(source.frames()):
        frames.append(f)
        if i >= 0:
            break
    assert len(frames) >= 1
    f0 = frames[0]
    assert isinstance(f0, CameraFrame)
    assert f0.timestamp > 0
    assert f0.frame_data is not None


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATASET_PATH).is_dir(),
    reason="EuRoC dataset not available",
)
def test_euroc_imu_source_first_sample():
    """Test: EurocIMUSource yields at least one valid IMUSample."""
    source = EurocIMUSource(DATASET_PATH)
    samples = []
    for i, s in enumerate(source.samples()):
        samples.append(s)
        if i >= 0:
            break
    assert len(samples) >= 1
    s0 = samples[0]
    assert isinstance(s0, IMUSample)
    assert s0.timestamp > 0
    assert len(s0.linear_acceleration) == 3
    assert len(s0.angular_velocity) == 3
