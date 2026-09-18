"""Unit tests for Phase 3 Camera-IMU synchronization and preprocessing."""

from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from geonav.config.settings import SyncConfig
from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.synchronization.synchronizer import (
    SensorSynchronizer,
    synchronize_streams,
    validate_camera_frames,
    validate_imu_samples,
)
from geonav.synchronization.types import SynchronizedMeasurement


def _make_frame(timestamp: float, frame_id: int = 0) -> CameraFrame:
    """Helper to create dummy CameraFrame."""
    return CameraFrame(timestamp=timestamp, frame_data={"id": frame_id})


def _make_imu(timestamp: float, ax: float = 0.0, ay: float = 0.0, az: float = 9.81) -> IMUSample:
    """Helper to create dummy IMUSample."""
    return IMUSample(
        timestamp=timestamp,
        linear_acceleration=(ax, ay, az),
        angular_velocity=(0.0, 0.0, 0.0),
    )


def test_1_camera_imu_synchronized_correctly() -> None:
    """Test 1: Camera frames and IMU samples are correctly synchronized."""
    c0 = _make_frame(1.0)
    c1 = _make_frame(2.0)
    i1 = _make_imu(1.5)

    results = list(synchronize_streams([c0, c1], [i1]))
    assert len(results) == 2

    # First frame
    assert results[0].camera_frame.timestamp == 1.0
    assert results[0].start_timestamp is None
    assert results[0].end_timestamp == 1.0
    assert results[0].is_first_frame is True
    assert len(results[0].imu_samples) == 0

    # Inter-frame interval (1.0, 2.0]
    assert results[1].camera_frame.timestamp == 2.0
    assert results[1].start_timestamp == 1.0
    assert results[1].end_timestamp == 2.0
    assert results[1].is_first_frame is False
    assert len(results[1].imu_samples) == 1
    assert results[1].imu_samples[0].timestamp == 1.5
    assert results[1].duration_s == pytest.approx(1.0)


def test_2_multiple_imu_samples_per_interval() -> None:
    """Test 2: Multiple IMU samples are associated with one camera interval in order."""
    c0 = _make_frame(1.0)
    c1 = _make_frame(2.0)
    imus = [_make_imu(1.1), _make_imu(1.4), _make_imu(1.7), _make_imu(1.9)]

    results = list(synchronize_streams([c0, c1], imus))
    assert len(results) == 2
    meas = results[1]
    assert meas.num_imu_samples == 4
    timestamps = [s.timestamp for s in meas.imu_samples]
    assert timestamps == [1.1, 1.4, 1.7, 1.9]


def test_3_no_imu_samples_in_interval_no_fabricated_data() -> None:
    """Test 3: No IMU samples exist in an interval. Verify no fake samples are created."""
    c0 = _make_frame(1.0)
    c1 = _make_frame(2.0)
    c2 = _make_frame(3.0)
    # IMU samples only exist between c1 and c2
    imus = [_make_imu(2.5)]

    results = list(synchronize_streams([c0, c1, c2], imus))
    assert len(results) == 3

    # Interval (1.0, 2.0] has no IMU samples
    assert len(results[1].imu_samples) == 0
    assert results[1].start_timestamp == 1.0
    assert results[1].end_timestamp == 2.0

    # Interval (2.0, 3.0] has the IMU sample
    assert len(results[2].imu_samples) == 1
    assert results[2].imu_samples[0].timestamp == 2.5


def test_4_first_camera_frame_handling() -> None:
    """Test 4: First camera frame is handled correctly with include_first_frame flag."""
    c0 = _make_frame(5.0)
    c1 = _make_frame(6.0)
    imus = [_make_imu(5.5)]

    # With include_first_frame=True (default)
    res_with_first = list(synchronize_streams([c0, c1], imus, SyncConfig(include_first_frame=True)))
    assert len(res_with_first) == 2
    assert res_with_first[0].is_first_frame is True
    assert res_with_first[0].start_timestamp is None
    assert res_with_first[0].end_timestamp == 5.0
    assert res_with_first[0].duration_s is None
    assert len(res_with_first[0].imu_samples) == 0

    # With include_first_frame=False
    res_without_first = list(synchronize_streams([c0, c1], imus, SyncConfig(include_first_frame=False)))
    assert len(res_without_first) == 1
    assert res_without_first[0].is_first_frame is False
    assert res_without_first[0].start_timestamp == 5.0
    assert res_without_first[0].end_timestamp == 6.0
    assert len(res_without_first[0].imu_samples) == 1


def test_5_imu_before_first_camera_excluded() -> None:
    """Test 5: IMU samples before the first camera frame are not incorrectly associated."""
    c0 = _make_frame(10.0)
    c1 = _make_frame(11.0)
    # Samples before and at first camera timestamp
    imus = [_make_imu(8.0), _make_imu(9.5), _make_imu(10.0), _make_imu(10.5)]

    results = list(synchronize_streams([c0, c1], imus))
    assert len(results) == 2

    # First frame has empty IMU
    assert len(results[0].imu_samples) == 0

    # Inter-frame interval (10.0, 11.0] must ONLY contain sample at 10.5
    assert len(results[1].imu_samples) == 1
    assert results[1].imu_samples[0].timestamp == 10.5


def test_6_imu_after_final_camera_excluded() -> None:
    """Test 6: IMU samples after the final camera frame are not incorrectly associated."""
    c0 = _make_frame(1.0)
    c1 = _make_frame(2.0)
    # Sample within interval and samples after final camera timestamp
    imus = [_make_imu(1.5), _make_imu(2.1), _make_imu(3.0)]

    results = list(synchronize_streams([c0, c1], imus))
    assert len(results) == 2

    # Inter-frame interval (1.0, 2.0] contains only 1.5
    assert len(results[1].imu_samples) == 1
    assert results[1].imu_samples[0].timestamp == 1.5


def test_7_exact_timestamp_boundary_policy() -> None:
    """Test 7: Exact timestamp boundary behavior follows T_previous < t <= T_current."""
    c0 = _make_frame(1.0)
    c1 = _make_frame(2.0)
    c2 = _make_frame(3.0)

    # Samples exactly on boundaries: 1.0, 2.0, 3.0
    imus = [_make_imu(1.0), _make_imu(2.0), _make_imu(3.0)]

    results = list(synchronize_streams([c0, c1, c2], imus))
    assert len(results) == 3

    # Sample at 1.0 is <= T_0 (first frame), so excluded from inter-frame interval
    assert len(results[0].imu_samples) == 0

    # Interval 1: (1.0, 2.0] -> includes 2.0 (right boundary inclusive)
    assert len(results[1].imu_samples) == 1
    assert results[1].imu_samples[0].timestamp == 2.0

    # Interval 2: (2.0, 3.0] -> 2.0 was left boundary (excluded), includes 3.0 (right boundary)
    assert len(results[2].imu_samples) == 1
    assert results[2].imu_samples[0].timestamp == 3.0


def test_8_out_of_order_timestamps_rejected() -> None:
    """Test 8: Out-of-order timestamps in camera or IMU streams raise ValueError."""
    # Out-of-order camera frames
    c_bad = [_make_frame(2.0), _make_frame(1.0)]
    with pytest.raises(ValueError, match="strictly increasing"):
        list(synchronize_streams(c_bad, [_make_imu(1.5)]))

    # Duplicate camera timestamp
    c_dup = [_make_frame(1.0), _make_frame(1.0)]
    with pytest.raises(ValueError, match="strictly increasing"):
        list(synchronize_streams(c_dup, [_make_imu(1.5)]))

    # Out-of-order IMU samples
    c_ok = [_make_frame(1.0), _make_frame(2.0)]
    i_bad = [_make_imu(1.8), _make_imu(1.2)]
    with pytest.raises(ValueError, match="chronologically non-decreasing"):
        list(synchronize_streams(c_ok, i_bad))


def test_9_nan_inf_sensor_values_rejected() -> None:
    """Test 9: NaN/Inf sensor values in camera or IMU raise ValueError."""
    # Instantiation rejection
    with pytest.raises(ValueError, match="finite non-negative"):
        _make_frame(float("nan"))

    with pytest.raises(ValueError, match="finite numbers"):
        _make_imu(1.0, ax=float("nan"))

    # Validator function rejection on corrupted frame
    f_nan = _make_frame(1.0)
    f_nan.timestamp = float("nan")
    with pytest.raises(ValueError, match="invalid timestamp"):
        validate_camera_frames([f_nan])

    f_inf = _make_frame(1.0)
    f_inf.timestamp = float("inf")
    with pytest.raises(ValueError, match="invalid timestamp"):
        validate_camera_frames([f_inf])

    # Validator function rejection on corrupted IMU sample
    s_nan = _make_imu(1.0)
    s_nan.linear_acceleration = (float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="non-finite linear acceleration"):
        validate_imu_samples([s_nan])

    s_inf = _make_imu(1.0)
    s_inf.angular_velocity = (0.0, float("inf"), 0.0)
    with pytest.raises(ValueError, match="non-finite angular velocity"):
        validate_imu_samples([s_inf])



def test_10_empty_inputs_handling() -> None:
    """Test 10: Empty camera/IMU inputs behave gracefully."""
    # Empty camera stream produces empty iterator
    assert list(synchronize_streams([], [_make_imu(1.0)])) == []

    # Empty IMU stream with require_imu_for_interval=False
    results = list(synchronize_streams([_make_frame(1.0), _make_frame(2.0)], []))
    assert len(results) == 2
    assert results[0].imu_samples == []
    assert results[1].imu_samples == []


def test_11_deterministic_reproducibility() -> None:
    """Test 11: Repeated synchronization with identical input produces identical output."""
    frames = [_make_frame(1.0), _make_frame(1.5), _make_frame(2.0)]
    imus = [_make_imu(1.1), _make_imu(1.3), _make_imu(1.7), _make_imu(1.9)]

    run1 = list(synchronize_streams(frames, imus))
    run2 = list(synchronize_streams(frames, imus))

    assert len(run1) == len(run2) == 3
    for m1, m2 in zip(run1, run2):
        assert m1.start_timestamp == m2.start_timestamp
        assert m1.end_timestamp == m2.end_timestamp
        assert len(m1.imu_samples) == len(m2.imu_samples)
        for s1, s2 in zip(m1.imu_samples, m2.imu_samples):
            assert s1.timestamp == s2.timestamp
            assert s1.linear_acceleration == s2.linear_acceleration
            assert s1.angular_velocity == s2.angular_velocity


def test_12_require_imu_for_interval_flag() -> None:
    """Test require_imu_for_interval flag raises ValueError on empty interval."""
    c0 = _make_frame(1.0)
    c1 = _make_frame(2.0)
    config = SyncConfig(require_imu_for_interval=True)

    # First frame has no IMU, but that is NOT an error
    # The inter-frame interval (1.0, 2.0] having no IMU DOES raise ValueError
    with pytest.raises(ValueError, match="No IMU samples found"):
        list(synchronize_streams([c0, c1], [], config=config))


def test_13_sensor_synchronizer_buffer_synchronization() -> None:
    """Test SensorSynchronizer.synchronize() over internal buffers."""
    sync = SensorSynchronizer()
    sync.add_camera_frame(_make_frame(1.0))
    sync.add_camera_frame(_make_frame(2.0))
    sync.add_imu_sample(_make_imu(0.5))  # Before first frame -> excluded from interval
    sync.add_imu_sample(_make_imu(1.2))  # Inside interval
    sync.add_imu_sample(_make_imu(1.8))  # Inside interval
    sync.add_imu_sample(_make_imu(2.5))  # After last frame -> excluded from interval

    measurements = sync.synchronize()
    assert len(measurements) == 2
    assert measurements[0].is_first_frame is True
    assert measurements[1].num_imu_samples == 2
    assert measurements[1].imu_samples[0].timestamp == 1.2
    assert measurements[1].imu_samples[1].timestamp == 1.8
