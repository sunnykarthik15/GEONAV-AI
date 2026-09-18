"""Configuration settings and parameters for GEONAV-AI."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CameraConfig:
    """Camera sensor configuration parameters.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fps: Nominal capture frame rate in frames per second.
        fx: Focal length along x-axis in pixels (optional calibration parameter).
        fy: Focal length along y-axis in pixels (optional calibration parameter).
        cx: Principal point x-coordinate in pixels (optional calibration parameter).
        cy: Principal point y-coordinate in pixels (optional calibration parameter).
        distortion_coeffs: Lens distortion coefficients (k1, k2, p1, p2, [k3]) if calibrated.
    """

    width: int = 640
    height: int = 480
    fps: float = 30.0
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    distortion_coeffs: Optional[Tuple[float, ...]] = None


@dataclass
class IMUConfig:
    """IMU sensor configuration parameters.

    Attributes:
        rate_hz: Nominal IMU sampling frequency in Hertz.
    """

    rate_hz: float = 200.0


@dataclass
class SyncConfig:
    """Sensor synchronization parameters.

    Attributes:
        max_time_difference_s: Maximum allowable time difference in seconds for alignment (preserved from Phase 1).
        camera_buffer_size: Maximum number of camera frames to retain in buffer.
        imu_buffer_size: Maximum number of IMU samples to retain in buffer.
        boundary_policy: Timestamp boundary policy for associating IMU samples with camera intervals.
            Default: "exclusive_left_inclusive_right" (T_previous < t_IMU <= T_current).
        require_imu_for_interval: If True, inter-frame intervals with zero IMU samples raise ValueError.
            Default: False (outputs empty imu_samples list).
        include_first_frame: If True, emits initial camera frame measurement with start_timestamp=None.
            Default: True.
    """

    max_time_difference_s: float = 0.01
    camera_buffer_size: int = 100
    imu_buffer_size: int = 1000
    boundary_policy: str = "exclusive_left_inclusive_right"
    require_imu_for_interval: bool = False
    include_first_frame: bool = True


@dataclass
class CoordinateConfig:
    """Coordinate frame conventions used across GEONAV-AI.

    Conventions:
        body_frame: Vehicle body-fixed frame convention.
            Default: "FRD" (X-Forward, Y-Right, Z-Down), standard for aerial navigation.
        world_frame: Inertial/local navigation reference frame convention.
            Default: "NED" (X-North, Y-East, Z-Down), standard local tangent plane.
        camera_frame: Optical sensor frame convention.
            Default: "RDF" (X-Right, Y-Down, Z-Forward along optical axis).
    """

    body_frame: str = "FRD"
    world_frame: str = "NED"
    camera_frame: str = "RDF"


@dataclass
class DatasetConfig:
    """Configuration parameters for visual-inertial dataset ingestion.

    Attributes:
        dataset_path: Optional path to the local dataset sequence root directory.
        camera_name: Subdirectory name for camera stream (default: 'cam0').
        imu_name: Subdirectory name for IMU stream (default: 'imu0').
        timestamp_unit: Dataset timestamp unit (default: 'ns').
        image_extension: Expected image file extension (default: 'png').
    """

    dataset_path: Optional[str] = None
    camera_name: str = "cam0"
    imu_name: str = "imu0"
    timestamp_unit: str = "ns"
    image_extension: str = "png"


@dataclass
class ExtrinsicsConfig:
    """Camera-to-body/IMU extrinsic spatial calibration.

    Convention:
        T_BC = [R_BC | p_BC]
        Transforms vectors and points from Camera frame (C) to Body/IMU frame (B):
            v_B = R_BC * v_C
            p_B = R_BC * p_C + p_BC
        Inverted transformation:
            T_CB = T_BC^(-1) = [R_BC^T | -R_BC^T * p_BC]

    Attributes:
        rotation_matrix: 3x3 rotation matrix R_BC as tuple of tuples of floats.
                         Default: Identity (3x3).
        translation: 3D translation vector p_BC (x, y, z) in meters representing
                     camera optical center position in body coordinates. Default: (0.0, 0.0, 0.0).
        convention: Description of transformation convention (default: "T_BC: camera to body").
    """

    rotation_matrix: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    convention: str = "T_BC: camera to body"


@dataclass
class VIOConfig:
    """Configuration parameters for the Visual-Inertial Odometry pipeline.

    Attributes:
        max_features: Maximum number of visual features to track (default: 200).
        min_features: Minimum number of tracked geometric inliers required for valid visual motion (default: 15).
        ransac_threshold: Distance threshold for Essential matrix RANSAC in pixels (default: 1.0).
        ransac_prob: Success probability for Essential matrix RANSAC (default: 0.99).
        gravity_magnitude: Local gravitational acceleration magnitude in m/s^2 (default: 9.81).
        use_imu_scale: Whether inertial displacement norm is used to constrain visual translation scale (default: True).
        visual_rotation_weight: Weight in [0, 1] for visual rotation complementary fusion (default: 0.5).
        visual_direction_weight: Weight in [0, 1] for visual direction constraint on IMU displacement (default: 0.5).
        initial_gravity_alignment: Whether to align initial attitude using stationary accelerometer measurements (default: True).
        initial_alignment_min_samples: Minimum number of stationary IMU samples required for gravity alignment (default: 5).
        extrinsics: Spatial camera-to-body extrinsic calibration parameters (default: identity).
    """

    max_features: int = 200
    min_features: int = 15
    ransac_threshold: float = 1.0
    ransac_prob: float = 0.99
    gravity_magnitude: float = 9.81
    use_imu_scale: bool = True
    visual_rotation_weight: float = 0.5
    visual_direction_weight: float = 0.5
    initial_gravity_alignment: bool = True
    initial_alignment_min_samples: int = 5
    extrinsics: ExtrinsicsConfig = field(default_factory=ExtrinsicsConfig)



@dataclass
class MappingConfig:
    """Configuration parameters for the 3D point-cloud mapping subsystem.

    Attributes:
        enabled: Whether the mapping subsystem is active (default: True).
        min_triangulation_angle_deg: Minimum parallax angle in degrees between optical rays
            required to attempt triangulation (default: 1.0).
        max_reprojection_error_px: Maximum allowable reprojection error in pixels (default: 2.0).
        min_depth_m: Minimum allowable landmark depth in camera coordinates (default: 0.2 m).
        max_depth_m: Maximum allowable landmark depth in camera coordinates (default: 40.0 m).
        keyframe_translation_threshold_m: Minimum camera translation required to trigger a new keyframe (default: 0.15 m).
        keyframe_rotation_threshold_deg: Minimum camera rotation in degrees required to trigger a new keyframe (default: 5.0 deg).
        keyframe_interval_frames: Maximum frame interval before forcing a keyframe check (default: 15).
        min_tracked_features: Minimum number of tracked features required for keyframe consideration (default: 15).
        max_landmarks: Maximum number of landmarks to store in the map (default: 20000).
        extract_rgb: Whether to sample color values from the image for reconstructed landmarks (default: True).
    """

    enabled: bool = True
    min_triangulation_angle_deg: float = 1.0
    max_reprojection_error_px: float = 2.0
    min_depth_m: float = 0.2
    max_depth_m: float = 40.0
    keyframe_translation_threshold_m: float = 0.15
    keyframe_rotation_threshold_deg: float = 5.0
    keyframe_interval_frames: int = 15
    min_tracked_features: int = 15
    max_landmarks: int = 20000
    extract_rgb: bool = True


@dataclass
class Settings:
    """Root configuration aggregating all subsystem settings."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    coordinate: CoordinateConfig = field(default_factory=CoordinateConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vio: VIOConfig = field(default_factory=VIOConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
