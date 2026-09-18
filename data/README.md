# GEONAV-AI Data Directory

This directory is designated for local datasets used by GEONAV-AI during development and validation.

## Purpose

GEONAV-AI is a software navigation module designed for UAV platforms. In GPS-denied navigation development, recorded visual-inertial datasets provide realistic sensor inputs prior to physical hardware integration.

> **CRITICAL RULE**: Real dataset files (images, CSVs, ROS bags) must **NOT** be committed to GitHub or Git history.
> The `.gitignore` is configured to ignore all raw and processed dataset contents while retaining folder placeholders.

## Expected Directory Layout for EuRoC MAV Datasets

When using the EuRoC MAV visual-inertial dataset, download a sequence (such as `MH_01_easy`) and place its uncompressed folder under `data/raw/euroc/`:

```
data/
├── raw/
│   └── euroc/
│       └── MH_01_easy/
│           └── mav0/
│               ├── cam0/
│               │   ├── data/
│               │   │   ├── 1403636579758555392.png
│               │   │   └── ...
│               │   ├── data.csv
│               │   └── sensor.yaml
│               └── imu0/
│                   ├── data.csv
│                   └── sensor.yaml
└── processed/
```

GEONAV-AI also supports direct sequence roots without the intermediate `mav0/` directory:
```
data/raw/euroc/MH_01_easy/
├── cam0/
│   ├── data.csv
│   └── data/
└── imu0/
    └── data.csv
```

## Data Ingestion Format

### Camera Stream (`cam0/`)
- **Index File**: `data.csv`
- **Format**:
  ```csv
  #timestamp [ns],filename
  1403636579758555392,1403636579758555392.png
  ```
- **Images**: 8-bit grayscale PNG files (typically WVGA 752×480) in the `data/` subdirectory.
- **Timestamps**: Monotonic nanosecond timestamps converted to seconds (`timestamp_ns * 1e-9`).
- **Loading**: Images are indexed lazily and loaded into memory as raw numerical arrays only when accessed.

### IMU Stream (`imu0/`)
- **Index File**: `data.csv`
- **Format**:
  ```csv
  #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
  1403636579758555392,0.0125,-0.0024,0.0081,0.125,0.045,9.812
  ```
- **Readings**:
  - Columns 1–3: 3-axis angular velocity / gyroscope measurements $(w_x, w_y, w_z)$ in rad/s.
  - Columns 4–6: 3-axis linear acceleration / accelerometer measurements $(a_x, a_y, a_z)$ in m/s².

## Current Phase Note

Phase 2 implements **ONLY visual-inertial data ingestion**. It parses and loads real sensor measurements and accurate timestamps from local files. Visual-Inertial Odometry (VIO), state estimation, and SLAM are not implemented in this phase.
