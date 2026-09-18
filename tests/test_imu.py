from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from geonav.sensors.imu import IMUSample, IMUSensor


class DummyIMUSensor(IMUSensor):
    """Minimal concrete implementation for testing the IMUSensor interface."""

    def __init__(self, connected: bool = True) -> None:
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected

    def read_sample(self) -> IMUSample | None:
        if not self._connected:
            return None
        return IMUSample(
            timestamp=0.1,
            linear_acceleration=(0.0, 0.0, 9.81),
            angular_velocity=(0.01, -0.02, 0.005),
        )


def test_imu_sample_creation() -> None:
    """Test standard creation of IMUSample with valid inputs."""
    sample = IMUSample(
        timestamp=1.25,
        linear_acceleration=(0.1, -0.2, 9.8),
        angular_velocity=(0.01, 0.02, -0.03),
    )
    assert sample.timestamp == 1.25
    assert sample.linear_acceleration == (0.1, -0.2, 9.8)
    assert sample.angular_velocity == (0.01, 0.02, -0.03)


def test_imu_sample_timestamp_validation() -> None:
    """Test that invalid timestamps raise appropriate errors."""
    with pytest.raises(ValueError, match="finite non-negative"):
        IMUSample(
            timestamp=-0.5,
            linear_acceleration=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )

    with pytest.raises(TypeError, match="float or integer"):
        IMUSample(
            timestamp="invalid",  # type: ignore
            linear_acceleration=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )


def test_imu_sample_linear_acceleration_validation() -> None:
    """Test that linear acceleration must contain exactly 3 finite numerical values."""
    with pytest.raises(ValueError, match="exactly 3 values"):
        IMUSample(
            timestamp=1.0,
            linear_acceleration=(0.0, 0.0),  # type: ignore
            angular_velocity=(0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="exactly 3 values"):
        IMUSample(
            timestamp=1.0,
            linear_acceleration=(0.0, 0.0, 0.0, 0.0),  # type: ignore
            angular_velocity=(0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="finite numbers"):
        IMUSample(
            timestamp=1.0,
            linear_acceleration=(0.0, float("nan"), 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )


def test_imu_sample_angular_velocity_validation() -> None:
    """Test that angular velocity must contain exactly 3 finite numerical values."""
    with pytest.raises(ValueError, match="exactly 3 values"):
        IMUSample(
            timestamp=1.0,
            linear_acceleration=(0.0, 0.0, 0.0),
            angular_velocity=(0.0,),  # type: ignore
        )

    with pytest.raises(ValueError, match="finite numbers"):
        IMUSample(
            timestamp=1.0,
            linear_acceleration=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, "str", 0.0),  # type: ignore
        )


def test_imu_sensor_abstract() -> None:
    """Test that IMUSensor abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IMUSensor()  # type: ignore


def test_imu_sensor_behavior() -> None:
    """Test concrete implementation adhering to IMUSensor interface."""
    sensor = DummyIMUSensor(connected=True)
    assert sensor.is_connected() is True
    sample = sensor.read_sample()
    assert sample is not None
    assert sample.timestamp == 0.1

    disconnected = DummyIMUSensor(connected=False)
    assert disconnected.is_connected() is False
    assert disconnected.read_sample() is None
