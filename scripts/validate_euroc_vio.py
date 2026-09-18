"""Validation script for GEONAV-AI Phase 4 VIO pipeline using real EuRoC MAV datasets."""

from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geonav.config.settings import ExtrinsicsConfig, VIOConfig
from geonav.datasets.euroc import EurocDataset
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline


def main() -> None:
    """Execute real-dataset validation path on local EuRoC sequences if available."""
    repo_root = Path(__file__).resolve().parent.parent
    euroc_raw_dir = repo_root / "data" / "raw" / "euroc"

    # Search for any EuRoC sequence folder under data/raw/euroc/
    candidates = []
    if euroc_raw_dir.is_dir():
        for p in euroc_raw_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                candidates.append(p)

    if not candidates:
        print("Real-dataset validation could not be executed because the EuRoC dataset is not available locally.")
        print(f"To run real validation, place a valid EuRoC sequence under: {euroc_raw_dir}")
        return

    seq_path = candidates[0]
    print(f"Found local EuRoC sequence at: {seq_path}")

    try:
        dataset = EurocDataset(seq_path)
        print(f"Loaded dataset: {dataset.metadata.sequence_name}")
        print(f"Total camera frames: {dataset.num_images}, Total IMU samples: {dataset.num_imu_samples}")

        vio_config = VIOConfig()
        if dataset.camera_extrinsics is not None:
            T_BS = dataset.camera_extrinsics
            R_BS = T_BS[:3, :3]
            t_BS = T_BS[:3, 3]
            vio_config.extrinsics = ExtrinsicsConfig(
                rotation_matrix=(
                    (float(R_BS[0, 0]), float(R_BS[0, 1]), float(R_BS[0, 2])),
                    (float(R_BS[1, 0]), float(R_BS[1, 1]), float(R_BS[1, 2])),
                    (float(R_BS[2, 0]), float(R_BS[2, 1]), float(R_BS[2, 2])),
                ),
                translation=(float(t_BS[0]), float(t_BS[1]), float(t_BS[2])),
            )
            print("Loaded camera-IMU extrinsic calibration T_BS from dataset:")
            print(T_BS)

        pipeline = VIOPipeline(config=vio_config)
        synced_stream = synchronize_streams(
            (img.to_camera_frame() for img in dataset.images()),
            (s.to_imu_sample() for s in dataset.imu()),
        )

        num_processed = 0
        for measurement in synced_stream:
            state = pipeline.process_measurement(measurement)
            num_processed += 1
            if num_processed % 50 == 0:
                print(f"Processed frame {num_processed}: pos={state.position if state else 'None'}")

        traj = pipeline.get_trajectory()
        print(f"VIO processing complete. Estimated trajectory length: {len(traj)} states.")
        if traj:
            final_p = traj[-1].position
            print(f"Final estimated position: x={final_p[0]:.3f}, y={final_p[1]:.3f}, z={final_p[2]:.3f} m")

    except Exception as err:
        print(f"Error during EuRoC validation: {err}")


if __name__ == "__main__":
    main()
