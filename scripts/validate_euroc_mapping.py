"""Validation script for GEONAV-AI Phase 5 Local Sparse 3D Point-Cloud Mapping.

Executes real-data mapping on the EuRoC MH_01_easy sequence:
- Integrates VIOPipeline and LocalMapper
- Reconstructs 3D landmarks via two-view triangulation
- Exports 3D point cloud to PLY format
- Records quantitative mapping statistics to JSON
- Generates 3D visualization plot of point cloud and VIO trajectory
"""

import json
from pathlib import Path
import shutil
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure src directory is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from geonav.config.settings import CameraConfig, ExtrinsicsConfig, MappingConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.mapping.mapper import LocalMapper
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


def main() -> None:
    """Execute real-data Phase 5 mapping validation on EuRoC MH_01_easy."""
    print("=" * 70)
    print("GEONAV-AI — PHASE 5 LOCAL SPARSE MAPPING VALIDATION")
    print("=" * 70)

    seq_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    if not seq_path.is_dir():
        print(f"Error: EuRoC dataset sequence not found at: {seq_path}")
        return

    print(f"Loading real EuRoC dataset from: {seq_path}")
    dataset = EurocDataset(seq_path)
    total_images = dataset.num_images
    total_imu = dataset.num_imu_samples
    print(f"Indexed {total_images} camera images, {total_imu} IMU measurements.")

    # Load validated Camera Calibration and Extrinsics
    cam_calib = dataset.cam0_loader.calibration if hasattr(dataset, "cam0_loader") else None
    if cam_calib is None:
        # Fallback to cam0 loader directly
        from geonav.datasets.euroc.camera_loader import EurocCameraLoader
        cam_loader = EurocCameraLoader(seq_path / "mav0" / "cam0")
        cam_calib = cam_loader.calibration

    if cam_calib is not None:
        intrinsics = cam_calib.intrinsics or (458.654, 457.296, 367.215, 248.375)
        dist_coeffs = cam_calib.distortion_coefficients or (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)
        T_BS = cam_calib.T_BS
        R_BS = T_BS[:3, :3]
        t_BS = T_BS[:3, 3]
        print(f"Loaded camera calibration fx={intrinsics[0]:.2f}, fy={intrinsics[1]:.2f}, cx={intrinsics[2]:.2f}, cy={intrinsics[3]:.2f}")
    else:
        intrinsics = (458.654, 457.296, 367.215, 248.375)
        dist_coeffs = (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)
        R_BS = np.eye(3)
        t_BS = np.zeros(3)

    cam_config = CameraConfig(
        width=752,
        height=480,
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
        distortion_coeffs=tuple(float(x) for x in dist_coeffs),
    )

    ext_config = ExtrinsicsConfig(
        rotation_matrix=(
            (float(R_BS[0, 0]), float(R_BS[0, 1]), float(R_BS[0, 2])),
            (float(R_BS[1, 0]), float(R_BS[1, 1]), float(R_BS[1, 2])),
            (float(R_BS[2, 0]), float(R_BS[2, 1]), float(R_BS[2, 2])),
        ),
        translation=(float(t_BS[0]), float(t_BS[1]), float(t_BS[2])),
    )

    vio_config = VIOConfig(extrinsics=ext_config)
    map_config = MappingConfig(
        enabled=True,
        min_triangulation_angle_deg=1.0,
        max_reprojection_error_px=2.0,
        min_depth_m=0.2,
        max_depth_m=40.0,
        keyframe_translation_threshold_m=0.15,
        keyframe_rotation_threshold_deg=5.0,
        keyframe_interval_frames=15,
        min_tracked_features=15,
        max_landmarks=20000,
        extract_rgb=True,
    )

    pipeline = VIOPipeline(config=vio_config)
    mapper = LocalMapper(
        mapping_config=map_config,
        camera_config=cam_config,
        extrinsics_config=ext_config,
    )

    print("\nSynchronizing visual-inertial streams...")
    synced_stream = list(
        synchronize_streams(
            (img.to_camera_frame() for img in dataset.images()),
            (s.to_imu_sample() for s in dataset.imu()),
        )
    )
    print(f"Prepared {len(synced_stream)} synchronized measurement intervals.")

    # Artifacts output directory
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\nExecuting VIO tracking and 3D mapping...")
    t_start = time.time()
    frames_processed = 0
    nan_inf_detected = False

    trajectory_positions = []

    for idx, measurement in enumerate(synced_stream):
        # 1. Process VIO step
        nav_state = pipeline.process_measurement(measurement)
        frames_processed += 1

        if nav_state is not None:
            trajectory_positions.append(nav_state.position)
            if any(not np.isfinite(x) for x in nav_state.position) or any(not np.isfinite(x) for x in nav_state.velocity) or any(not np.isfinite(x) for x in nav_state.orientation):
                nan_inf_detected = True

        # 2. Process mapping step
        img_data = measurement.camera_frame.frame_data
        vis_res = pipeline._last_visual_result

        mapper.process_frame(
            timestamp=measurement.camera_frame.timestamp,
            image=img_data,
            nav_state=nav_state,
            visual_result=vis_res,
        )

        if frames_processed % 200 == 0 or frames_processed == len(synced_stream):
            stats = mapper.get_statistics()
            print(
                f"Frame {frames_processed:4d}/{len(synced_stream)} | "
                f"Keyframes: {stats.keyframes_count:3d} | "
                f"Landmarks: {stats.total_landmarks:4d} | "
                f"Mean Err: {stats.mean_reprojection_error:.2f} px | "
                f"FPS: {stats.fps:.1f}"
            )

    t_total = time.time() - t_start
    stats = mapper.get_statistics()
    stats.runtime_s = t_total
    stats.fps = frames_processed / t_total if t_total > 0 else 0.0

    traj_arr = np.array(trajectory_positions)
    traj_diffs = np.diff(traj_arr, axis=0) if len(traj_arr) > 1 else np.zeros((1, 3))
    trajectory_length = float(np.sum(np.linalg.norm(traj_diffs, axis=1)))

    print("\n" + "=" * 70)
    print("PHASE 5 MAPPING RESULTS SUMMARY:")
    print("=" * 70)
    print(f"  Total frames processed:          {frames_processed}")
    print(f"  Total keyframes formed:          {stats.keyframes_count}")
    print(f"  Triangulation attempts:          {stats.triangulation_attempts}")
    print(f"  Triangulation successes:         {stats.triangulation_successes}")
    print(f"  Total landmarks in map:          {stats.total_landmarks}")
    print(f"  Rejected - Parallax (<1.0 deg):  {stats.rejected_parallax}")
    print(f"  Rejected - Negative Depth:       {stats.rejected_negative_depth}")
    print(f"  Rejected - Depth Range:          {stats.rejected_depth_range}")
    print(f"  Rejected - Reprojection (>2px):  {stats.rejected_reprojection}")
    print(f"  Rejected - Numerical / SVD:      {stats.rejected_numerical}")
    print(f"  Mean Reprojection Error:         {stats.mean_reprojection_error:.3f} px")
    print(f"  Median Reprojection Error:       {stats.median_reprojection_error:.3f} px")
    print(f"  Max Reprojection Error:          {stats.max_reprojection_error:.3f} px")
    print(f"  Min Reconstructed Depth:         {stats.min_depth:.2f} m")
    print(f"  Max Reconstructed Depth:         {stats.max_depth:.2f} m")
    print(f"  VIO Trajectory Length:           {trajectory_length:.2f} m")
    print(f"  Map Spatial Extent (X, Y, Z):    {stats.spatial_extent}")
    print(f"  Total Execution Runtime:         {t_total:.2f} s")
    print(f"  Overall Processing Rate:         {stats.fps:.2f} FPS")
    print(f"  NaN / Inf detected:              {nan_inf_detected}")
    print("=" * 70)

    # Export PLY file
    ply_path = artifacts_dir / "phase5_MH_01_easy_map.ply"
    mapper.export_ply(ply_path)
    print(f"Exported 3D point cloud map: {ply_path} ({ply_path.stat().st_size / 1024:.1f} KB)")

    # Export Statistics JSON
    stats_dict = stats.to_dict()
    stats_dict["dataset"] = "EuRoC MH_01_easy"
    stats_dict["trajectory_length_m"] = trajectory_length
    stats_dict["nan_inf_detected"] = nan_inf_detected
    stats_json_path = artifacts_dir / "phase5_mapping_statistics.json"
    with open(stats_json_path, mode="w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Exported mapping statistics: {stats_json_path}")

    # Generate 3D visualization plot
    points = mapper.get_point_cloud().get_points()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Downsample points for scatter plot if large
    if len(points) > 0:
        step = max(1, len(points) // 3000)
        pts_sample = points[::step]
        ax.scatter(
            pts_sample[:, 0],
            pts_sample[:, 1],
            pts_sample[:, 2],
            c="tab:blue",
            s=2,
            alpha=0.6,
            label=f"3D Landmarks ({len(points)} total)",
        )

    if len(traj_arr) > 0:
        ax.plot(
            traj_arr[:, 0],
            traj_arr[:, 1],
            traj_arr[:, 2],
            c="tab:red",
            linewidth=2.0,
            label=f"VIO Camera Trajectory ({trajectory_length:.1f} m)",
        )

    ax.set_xlabel("X [m] (North / Forward)")
    ax.set_ylabel("Y [m] (East / Right)")
    ax.set_zlabel("Z [m] (Down)")
    ax.set_title("GEONAV-AI Phase 5: Reconstructed 3D Map & Camera Trajectory (MH_01_easy)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    map_img_path = artifacts_dir / "phase5_mapping_map.png"
    plt.savefig(map_img_path, dpi=150)
    plt.close()
    print(f"Generated 3D visualization plot: {map_img_path}")

    # Also copy artifacts to IDE brain directory if accessible
    ide_artifacts_dir = Path(r"C:\Users\karth\.gemini\antigravity-ide\brain\7acff3b5-f467-419a-a2b7-3f422b71c4fe")
    if ide_artifacts_dir.is_dir():
        for src_file in [ply_path, stats_json_path, map_img_path]:
            try:
                shutil.copy2(src_file, ide_artifacts_dir / src_file.name)
            except Exception:
                pass


if __name__ == "__main__":
    main()
