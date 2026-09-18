from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from geonav.sensors.camera import CameraFrame, CameraSensor


class DummyCameraSensor(CameraSensor):
    """Minimal concrete implementation for testing the CameraSensor interface."""

    def __init__(self, connected: bool = True) -> None:
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected

    def capture_frame(self) -> CameraFrame | None:
        if not self._connected:
            return None
        return CameraFrame(timestamp=1.0, frame_data=[[0, 1], [2, 3]])


def test_camera_frame_creation() -> None:
    """Test standard creation of CameraFrame with valid inputs."""
    frame = CameraFrame(timestamp=0.5, frame_data={"dummy": "pixel_buffer"})
    assert frame.timestamp == 0.5
    assert frame.frame_data == {"dummy": "pixel_buffer"}


def test_camera_frame_timestamp_validation() -> None:
    """Test that invalid timestamps raise appropriate errors."""
    with pytest.raises(ValueError, match="finite non-negative"):
        CameraFrame(timestamp=-1.0, frame_data=[0])

    with pytest.raises(ValueError, match="finite non-negative"):
        CameraFrame(timestamp=float("nan"), frame_data=[0])

    with pytest.raises(TypeError, match="float or integer"):
        CameraFrame(timestamp="invalid_ts", frame_data=[0])  # type: ignore

    with pytest.raises(TypeError, match="float or integer"):
        CameraFrame(timestamp=True, frame_data=[0])  # type: ignore


def test_camera_frame_data_validation() -> None:
    """Test that frame_data cannot be None."""
    with pytest.raises(ValueError, match="must not be None"):
        CameraFrame(timestamp=0.0, frame_data=None)


def test_camera_sensor_abstract() -> None:
    """Test that CameraSensor abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CameraSensor()  # type: ignore


def test_camera_sensor_behavior() -> None:
    """Test concrete implementation adhering to CameraSensor interface."""
    sensor = DummyCameraSensor(connected=True)
    assert sensor.is_connected() is True
    frame = sensor.capture_frame()
    assert frame is not None
    assert frame.timestamp == 1.0

    disconnected_sensor = DummyCameraSensor(connected=False)
    assert disconnected_sensor.is_connected() is False
    assert disconnected_sensor.capture_frame() is None
