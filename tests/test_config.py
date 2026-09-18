from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geonav.config.settings import (
    CameraConfig,
    CoordinateConfig,
    DatasetConfig,
    IMUConfig,
    Settings,
    SyncConfig,
    VIOConfig,
)


def test_settings_initialization_defaults() -> None:
    """Test Settings initialization with default values."""
    settings = Settings()

    # Camera defaults
    assert isinstance(settings.camera, CameraConfig)
    assert settings.camera.width == 640
    assert settings.camera.height == 480
    assert settings.camera.fps == 30.0

    # IMU defaults
    assert isinstance(settings.imu, IMUConfig)
    assert settings.imu.rate_hz == 200.0

    # Sync defaults
    assert isinstance(settings.sync, SyncConfig)
    assert settings.sync.max_time_difference_s == 0.01
    assert settings.sync.camera_buffer_size == 100
    assert settings.sync.imu_buffer_size == 1000
    assert settings.sync.boundary_policy == "exclusive_left_inclusive_right"
    assert settings.sync.require_imu_for_interval is False
    assert settings.sync.include_first_frame is True

    # Coordinate defaults
    assert isinstance(settings.coordinate, CoordinateConfig)
    assert settings.coordinate.body_frame == "FRD"
    assert settings.coordinate.world_frame == "NED"
    assert settings.coordinate.camera_frame == "RDF"

    # Dataset defaults
    assert isinstance(settings.dataset, DatasetConfig)
    assert settings.dataset.dataset_path is None
    assert settings.dataset.camera_name == "cam0"
    assert settings.dataset.imu_name == "imu0"
    assert settings.dataset.timestamp_unit == "ns"
    assert settings.dataset.image_extension == "png"

    # VIO defaults
    assert isinstance(settings.vio, VIOConfig)
    assert settings.vio.max_features == 200
    assert settings.vio.min_features == 15
    assert settings.vio.ransac_threshold == 1.0
    assert settings.vio.ransac_prob == 0.99
    assert settings.vio.gravity_magnitude == 9.81
    assert settings.vio.use_imu_scale is True



def test_custom_settings_initialization() -> None:
    """Test Settings initialization with custom configuration values."""
    custom_camera = CameraConfig(width=1280, height=720, fps=60.0)
    custom_imu = IMUConfig(rate_hz=400.0)
    custom_sync = SyncConfig(max_time_difference_s=0.005, camera_buffer_size=50, imu_buffer_size=500)
    custom_coord = CoordinateConfig(body_frame="FLU", world_frame="ENU", camera_frame="RDF")

    settings = Settings(
        camera=custom_camera,
        imu=custom_imu,
        sync=custom_sync,
        coordinate=custom_coord,
    )

    assert settings.camera.width == 1280
    assert settings.camera.fps == 60.0
    assert settings.imu.rate_hz == 400.0
    assert settings.sync.max_time_difference_s == 0.005
    assert settings.coordinate.body_frame == "FLU"
    assert settings.coordinate.world_frame == "ENU"


def test_coordinate_config_conventions() -> None:
    """Test coordinate convention representations."""
    coord = CoordinateConfig()
    assert coord.body_frame == "FRD"
    assert coord.world_frame == "NED"
    assert coord.camera_frame == "RDF"
