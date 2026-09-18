from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.synchronization.synchronizer import SensorSynchronizer


def test_synchronizer_initialization() -> None:
    """Test initial state of SensorSynchronizer."""
    sync = SensorSynchronizer(max_camera_buffer=10, max_imu_buffer=50)
    assert sync.max_camera_buffer == 10
    assert sync.max_imu_buffer == 50
    assert len(sync.get_camera_buffer()) == 0
    assert len(sync.get_imu_buffer()) == 0


def test_synchronizer_invalid_initialization() -> None:
    """Test that non-positive buffer capacities raise ValueError."""
    with pytest.raises(ValueError, match="positive integers"):
        SensorSynchronizer(max_camera_buffer=0)

    with pytest.raises(ValueError, match="positive integers"):
        SensorSynchronizer(max_imu_buffer=-5)


def test_camera_buffering() -> None:
    """Test buffering of camera frames and FIFO eviction."""
    sync = SensorSynchronizer(max_camera_buffer=2, max_imu_buffer=10)
    f1 = CameraFrame(timestamp=1.0, frame_data=[1])
    f2 = CameraFrame(timestamp=2.0, frame_data=[2])
    f3 = CameraFrame(timestamp=3.0, frame_data=[3])

    sync.add_camera_frame(f1)
    sync.add_camera_frame(f2)
    assert len(sync.get_camera_buffer()) == 2
    assert sync.get_camera_buffer()[0].timestamp == 1.0

    # Exceeding buffer capacity evicts the oldest frame (FIFO)
    sync.add_camera_frame(f3)
    buf = sync.get_camera_buffer()
    assert len(buf) == 2
    assert buf[0].timestamp == 2.0
    assert buf[1].timestamp == 3.0


def test_imu_buffering() -> None:
    """Test buffering of IMU samples and FIFO eviction."""
    sync = SensorSynchronizer(max_camera_buffer=10, max_imu_buffer=2)
    s1 = IMUSample(timestamp=0.1, linear_acceleration=(0, 0, 0), angular_velocity=(0, 0, 0))
    s2 = IMUSample(timestamp=0.2, linear_acceleration=(0, 0, 0), angular_velocity=(0, 0, 0))
    s3 = IMUSample(timestamp=0.3, linear_acceleration=(0, 0, 0), angular_velocity=(0, 0, 0))

    sync.add_imu_sample(s1)
    sync.add_imu_sample(s2)
    assert len(sync.get_imu_buffer()) == 2

    # FIFO eviction
    sync.add_imu_sample(s3)
    buf = sync.get_imu_buffer()
    assert len(buf) == 2
    assert buf[0].timestamp == 0.2
    assert buf[1].timestamp == 0.3


def test_clear_buffers() -> None:
    """Test clearing sensor buffers."""
    sync = SensorSynchronizer()
    sync.add_camera_frame(CameraFrame(timestamp=1.0, frame_data=[1]))
    sync.add_imu_sample(
        IMUSample(timestamp=1.0, linear_acceleration=(0, 0, 0), angular_velocity=(0, 0, 0))
    )
    assert len(sync.get_camera_buffer()) == 1
    assert len(sync.get_imu_buffer()) == 1

    sync.clear_buffers()
    assert len(sync.get_camera_buffer()) == 0
    assert len(sync.get_imu_buffer()) == 0


def test_invalid_type_buffering() -> None:
    """Test that invalid types passed to buffering methods raise TypeError."""
    sync = SensorSynchronizer()
    with pytest.raises(TypeError, match="CameraFrame"):
        sync.add_camera_frame("not_a_frame")  # type: ignore

    with pytest.raises(TypeError, match="IMUSample"):
        sync.add_imu_sample("not_a_sample")  # type: ignore
