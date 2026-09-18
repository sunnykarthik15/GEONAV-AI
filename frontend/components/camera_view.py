"""Camera view component displaying real EuRoC camera frames with tracked visual features."""

from typing import Optional
import numpy as np
import streamlit as st


def render_camera_view(
    image_rgb: Optional[np.ndarray],
    tracked_features: int,
    ransac_inliers: int,
    visual_status: str,
    frame_idx: int,
    total_frames: int,
) -> None:
    """Render the live camera display panel with overlaid feature metrics.

    Args:
        image_rgb: Processed camera frame with overlaid feature points.
        tracked_features: Count of actively tracked visual features.
        ransac_inliers: Count of features passing 5-point Essential RANSAC.
        visual_status: Operational status (e.g. 'TRACKING', 'DEGRADED').
        frame_idx: Current frame index.
        total_frames: Total sequence frame count.
    """
    st.markdown("### LIVE CAMERA (cam0)")

    if image_rgb is not None:
        st.image(
            image_rgb,
            caption=f"EuRoC MAV MH_01_easy — cam0 (Frame {frame_idx + 1} / {total_frames})",
            use_container_width=True,
        )
    else:
        st.info("No camera frame available.")

    # Status sub-bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Tracked Features", value=f"{tracked_features}")
    with col2:
        st.metric(label="RANSAC Inliers", value=f"{ransac_inliers}")
    with col3:
        status_color = "normal"
        if visual_status == "TRACKING":
            status_text = "🟢 TRACKING"
        elif visual_status == "DEGRADED":
            status_text = "🟡 DEGRADED"
        else:
            status_text = "⚪ INITIALIZING"
        st.metric(label="Visual Status", value=status_text)
