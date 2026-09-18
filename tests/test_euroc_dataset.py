"""Unit tests for EuRoC MAV visual-inertial dataset ingestion."""

from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pytest

from geonav.datasets.euroc.camera_loader import EurocCameraLoader
from geonav.datasets.euroc.imu_loader import EurocIMULoader
from geonav.datasets.euroc.loader import EurocDataset
from geonav.sensors.camera import DatasetCameraSensor
from geonav.sensors.imu import DatasetIMUSensor
from geonav.synchronization.synchronizer import SensorSynchronizer


@pytest.fixture
def mock_euroc_seq(tmp_path: Path) -> Path:
    """Fixture creating a minimal valid EuRoC dataset sequence directory structure."""
    seq_dir = tmp_path / "MH_01_mock"
    mav0_dir = seq_dir / "mav0"
    cam0_dir = mav0_dir / "cam0"
    imu0_dir = mav0_dir / "imu0"
    data_img_dir = cam0_dir / "data"

    data_img_dir.mkdir(parents=True)
    imu0_dir.mkdir(parents=True)

    # Create 3 small mock images (8-bit grayscale 4x4)
    img_array = np.full((4, 4), 128, dtype=np.uint8)
    img_files = ["1403636579758555000.png", "1403636579808555000.png", "1403636579858555000.png"]
    for img_name in img_files:
        cv2.imwrite(str(data_img_dir / img_name), img_array)

    # Write cam0 data.csv (un-ordered to test sorting)
    cam_csv = cam0_dir / "data.csv"
    cam_csv.write_text(
        "#timestamp [ns],filename\n"
        "1403636579858555000,1403636579858555000.png\n"
        "1403636579758555000,1403636579758555000.png\n"
        "1403636579808555000,1403636579808555000.png\n",
        encoding="utf-8",
    )

    # Write imu0 data.csv (un-ordered to test sorting)
    imu_csv = imu0_dir / "data.csv"
    imu_csv.write_text(
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        "1403636579800000000,0.01,-0.02,0.03,0.1,0.2,9.8\n"
        "1403636579700000000,0.02,-0.01,0.04,0.0,0.1,9.81\n",
        encoding="utf-8",
    )

    return seq_dir


def test_valid_imu_parsing_and_chronological_ordering(tmp_path: Path) -> None:
    """Test parsing of valid IMU CSV and automatic sorting by timestamp."""
    imu_dir = tmp_path / "imu0"
    imu_dir.mkdir()
    csv_file = imu_dir / "data.csv"
    csv_file.write_text(
        "# Header line\n"
        "2000000000, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0\n"
        "1000000000, -0.1, -0.2, -0.3, -1.0, -2.0, -3.0\n",
        encoding="utf-8",
    )

    loader = EurocIMULoader(imu_dir)
    assert len(loader) == 2

    # Verify chronological sorting (1000000000 ns first, then 2000000000 ns)
    first = loader[0]
    assert first.timestamp_ns == 1000000000
    assert first.timestamp == pytest.approx(1.0)
    assert first.angular_velocity == (-0.1, -0.2, -0.3)
    assert first.linear_acceleration == (-1.0, -2.0, -3.0)

    # Verify conversion to Phase 1 IMUSample
    sample = first.to_imu_sample()
    assert sample.timestamp == pytest.approx(1.0)
    assert sample.angular_velocity == (-0.1, -0.2, -0.3)
    assert sample.linear_acceleration == (-1.0, -2.0, -3.0)

    second = loader[1]
    assert second.timestamp_ns == 2000000000
    assert second.timestamp == pytest.approx(2.0)


def test_imu_malformed_rows_and_column_count(tmp_path: Path) -> None:
    """Test that malformed rows with incorrect column counts raise ValueError with line info."""
    imu_dir = tmp_path / "imu0"
    imu_dir.mkdir()
    csv_file = imu_dir / "data.csv"
    csv_file.write_text(
        "# Header\n"
        "1000000000, 0.1, 0.2, 0.3, 1.0, 2.0\n",  # Only 6 columns
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 7 values, found 6"):
        EurocIMULoader(imu_dir)


def test_imu_non_numeric_and_non_finite_values(tmp_path: Path) -> None:
    """Test that non-numeric or non-finite values in IMU rows raise ValueError."""
    imu_dir = tmp_path / "imu0"
    imu_dir.mkdir()
    csv_file = imu_dir / "data.csv"
    csv_file.write_text(
        "# Header\n"
        "1000000000, 0.1, invalid_val, 0.3, 1.0, 2.0, 3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Non-numeric value 'invalid_val'"):
        EurocIMULoader(imu_dir)

    csv_file.write_text(
        "# Header\n"
        "1000000000, 0.1, nan, 0.3, 1.0, 2.0, 3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Non-finite numerical value 'nan'"):
        EurocIMULoader(imu_dir)


def test_camera_csv_parsing_and_chronological_ordering(mock_euroc_seq: Path) -> None:
    """Test camera CSV index parsing, timestamp conversion, and sorting."""
    cam_loader = EurocCameraLoader(mock_euroc_seq / "mav0" / "cam0")
    assert len(cam_loader) == 3

    # Check chronological ordering
    ts_list = [img.timestamp_ns for img in cam_loader]
    assert ts_list == sorted(ts_list)
    assert ts_list[0] == 1403636579758555000
    assert cam_loader[0].timestamp == pytest.approx(1403636579.758555)
    assert cam_loader[0].filename == "1403636579758555000.png"
    assert cam_loader[0].file_path.is_file()


def test_camera_lazy_image_loading(mock_euroc_seq: Path) -> None:
    """Test that image pixels are only loaded when requested."""
    cam_loader = EurocCameraLoader(mock_euroc_seq / "mav0" / "cam0")
    img_entry = cam_loader[0]

    # Loading data loads the actual array
    arr = img_entry.load_data()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)

    # Conversion to CameraFrame
    frame = img_entry.to_camera_frame()
    assert frame.timestamp == pytest.approx(1403636579.758555)
    assert isinstance(frame.frame_data, np.ndarray)


def test_camera_missing_image_file_handling(tmp_path: Path) -> None:
    """Test that missing image files raise FileNotFoundError when loaded."""
    cam_dir = tmp_path / "cam0"
    (cam_dir / "data").mkdir(parents=True)
    csv_file = cam_dir / "data.csv"
    csv_file.write_text(
        "# Header\n"
        "1000000000, non_existent_image.png\n",
        encoding="utf-8",
    )

    loader = EurocCameraLoader(cam_dir)
    assert len(loader) == 1
    # Entry can be indexed
    img_entry = loader[0]
    # But loading pixels raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        img_entry.load_data()


def test_high_level_euroc_dataset_with_mav0(mock_euroc_seq: Path) -> None:
    """Test high-level EurocDataset with standard mav0 directory structure."""
    dataset = EurocDataset(mock_euroc_seq)
    assert dataset.num_images == 3
    assert dataset.num_imu_samples == 2

    # Iterators
    images = list(dataset.images())
    assert len(images) == 3
    assert images[0].timestamp_ns == 1403636579758555000

    imu_samples = list(dataset.imu())
    assert len(imu_samples) == 2
    assert imu_samples[0].timestamp_ns == 1403636579700000000

    # Metadata
    assert dataset.metadata.sequence_name == "MH_01_mock"
    assert dataset.metadata.num_images == 3
    assert dataset.metadata.num_imu_samples == 2


def test_high_level_euroc_dataset_direct_sequence_root(tmp_path: Path) -> None:
    """Test EurocDataset without mav0 wrapper (direct sequence root)."""
    seq_dir = tmp_path / "direct_seq"
    cam0_dir = seq_dir / "cam0"
    imu0_dir = seq_dir / "imu0"
    (cam0_dir / "data").mkdir(parents=True)
    imu0_dir.mkdir(parents=True)

    (cam0_dir / "data.csv").write_text(
        "# Header\n1000000000, 1000000000.png\n", encoding="utf-8"
    )
    cv2.imwrite(str(cam0_dir / "data" / "1000000000.png"), np.zeros((2, 2), dtype=np.uint8))

    (imu0_dir / "data.csv").write_text(
        "# Header\n1000000000, 0, 0, 0, 0, 0, 9.8\n", encoding="utf-8"
    )

    dataset = EurocDataset(seq_dir)
    assert dataset.num_images == 1
    assert dataset.num_imu_samples == 1


def test_dataset_missing_path_and_sensors(tmp_path: Path) -> None:
    """Test error handling when dataset path or sensor subdirectories are missing."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        EurocDataset(tmp_path / "non_existent_folder")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Camera folder 'cam0' not found"):
        EurocDataset(empty_dir)


def test_dataset_camera_and_imu_sensor_adapters(mock_euroc_seq: Path) -> None:
    """Test DatasetCameraSensor and DatasetIMUSensor adapters with Phase 1 SensorSynchronizer."""
    dataset = EurocDataset(mock_euroc_seq)

    cam_sensor = DatasetCameraSensor(dataset.images())
    imu_sensor = DatasetIMUSensor(dataset.imu())

    assert cam_sensor.is_connected() is True
    assert imu_sensor.is_connected() is True

    # Capture first camera frame
    frame1 = cam_sensor.capture_frame()
    assert frame1 is not None
    assert frame1.timestamp == pytest.approx(1403636579.758555)
    assert isinstance(frame1.frame_data, np.ndarray)

    # Read first IMU sample
    sample1 = imu_sensor.read_sample()
    assert sample1 is not None
    assert sample1.timestamp == pytest.approx(1403636579.7)

    # Feed into Phase 1 SensorSynchronizer
    sync = SensorSynchronizer()
    sync.add_camera_frame(frame1)
    sync.add_imu_sample(sample1)

    assert len(sync.get_camera_buffer()) == 1
    assert len(sync.get_imu_buffer()) == 1
    assert sync.get_camera_buffer()[0].timestamp == frame1.timestamp
    assert sync.get_imu_buffer()[0].timestamp == sample1.timestamp

    # Read until exhausted
    assert cam_sensor.capture_frame() is not None
    assert cam_sensor.capture_frame() is not None
    assert cam_sensor.capture_frame() is None  # Exhausted

    assert imu_sensor.read_sample() is not None
    assert imu_sensor.read_sample() is None  # Exhausted
