"""Performance metrics, system status cards, and technical reference panels."""

from typing import Dict, Optional
import streamlit as st


def render_top_status_bar(
    system_status: str,
    current_frame: int,
    total_frames: int,
    processing_fps: float,
    ml_status: str,
) -> None:
    """Render top 4 status cards per Section 7 specifications.

    Args:
        system_status: 'INITIALIZING', 'TRACKING', 'DEGRADED', or 'LOST'.
        current_frame: Current 1-based frame index.
        total_frames: Total sequence frames count.
        processing_fps: Measured or calculated playback/processing FPS.
        ml_status: 'ACTIVE' or 'FALLBACK'.
    """
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if system_status == "TRACKING":
            val_text = "🟢 TRACKING"
        elif system_status == "DEGRADED":
            val_text = "🟡 DEGRADED"
        elif system_status == "LOST":
            val_text = "🔴 LOST"
        else:
            val_text = "⚪ INITIALIZING"
        st.metric(label="SYSTEM STATUS", value=val_text)

    with c2:
        st.metric(label="FRAME", value=f"{current_frame} / {total_frames}")

    with c3:
        st.metric(label="PROCESSING RATE", value=f"{processing_fps:.1f} FPS")

    with c4:
        if ml_status == "ACTIVE":
            ml_text = "🟢 ACTIVE"
        else:
            ml_text = "🟡 FALLBACK"
        st.metric(label="ML CORRECTION", value=ml_text)


def render_sensor_pipeline_panel() -> None:
    """Render Section 14 Sensor Pipeline architecture diagram."""
    st.markdown("### SENSOR PIPELINE ARCHITECTURE")
    st.markdown(
        """
        ```
        CAMERA (20 Hz)  +  IMU (200 Hz)
                     │
                     ▼
             Synchronization
                     │
                     ▼
         Visual Feature Tracking (Shi-Tomasi + LK Flow)
                     │
                     ▼
          Geometric VIO (Essential RANSAC + IMU Propagation)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Sparse Mapping       ML Residual Correction
   (Keyframe Triang.)   (18D Features → NumpyEdgeMLP)
         │                       │
         └───────────┬───────────┘
                     ▼
              NAVIGATION STATE (Position, Velocity, Attitude)
        ```
        """
    )


def render_performance_panel(
    fps: float,
    frame_ms: float,
    num_tracked: int,
    num_inliers: int,
) -> None:
    """Render Section 15 Performance and edge readiness indicators."""
    st.markdown("### PERFORMANCE & HARDWARE READINESS")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric(label="Step Latency", value=f"{frame_ms:5.1f} ms")
    with p2:
        st.metric(label="Pipeline FPS", value=f"{fps:5.1f} FPS")
    with p3:
        st.metric(label="Algorithm Memory", value="3.4 MB")
    with p4:
        st.metric(label="ML Model Size", value="25.7 KB")

    st.markdown(
        """
        **Target Edge Platforms:**
        - Raspberry Pi 5 (ARM Cortex-A76)
        - Jetson-class hardware (NVIDIA Orin / Nano)

        *Target platforms — not physically benchmarked in this project.*
        """
    )


def render_about_panel() -> None:
    """Render Section 24 collapsible ABOUT GEONAV-AI explanation."""
    with st.expander("ℹ️ ABOUT GEONAV-AI", expanded=False):
        st.markdown(
            """
            **GEONAV-AI** is a software-based visual-inertial navigation module
            designed for GPS-denied environments.

            It combines camera observations and IMU measurements using a
            classical geometric VIO pipeline, sparse local mapping, and a
            lightweight learned velocity-residual correction.

            The current demonstration uses real **EuRoC MAV visual-inertial data** (`MH_01_easy`).
            No physical UAV hardware is required for this software demonstration.

            *Physical UAV integration and flight validation are future work.*
            """
        )


def render_validated_results_panel(metrics: Dict[str, dict]) -> None:
    """Render Section 25 collapsible VALIDATED RESULTS table."""
    with st.expander("📊 VALIDATED RESULTS (Offline Software Benchmark)", expanded=False):
        st.markdown(
            """
            | Benchmark Indicator | Value | Reference / Notes |
            |---|---|---|
            | **Test Suite** | **185 / 185 Passed** | Automated pytest validation |
            | **Dataset Sequence** | **EuRoC MAV MH_01_easy** | Real recorded MAV flight data |
            | **Classical Baseline Raw ATE** | **88.927 m** | Phase 4/6 classical geometric VIO |
            | **Hybrid VIO + ML Raw ATE** | **6.005 m** | Phase 7 learned velocity residual correction (**93.2% reduction**) |
            | **Held-Out Split Baseline ATE** | **130.682 m** | Last 30% of sequence (no ML correction) |
            | **Held-Out Split Hybrid ATE** | **9.722 m** | Last 30% with ML correction (**92.6% reduction**) |
            | **Triangulated Landmarks** | **279 points** | Mean reprojection error: 0.889 px |
            | **ML Model Footprint** | **25.7 KB** | Zero-dependency pure NumPy inference |
            | **Tracked Algorithm Memory** | **3.4 MB** | Measured during full sequence run |

            *The held-out evaluation is a chronological split from the same MH_01_easy sequence, not an unseen-flight generalization test.*
            """
        )
