"""Sparse 3D landmark map visualization component using Plotly."""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_map_view(
    map_points: np.ndarray,
    map_colors: np.ndarray,
    landmark_count: int,
    mean_reproj_err: float,
    current_positions: Optional[np.ndarray] = None,
    current_idx: int = 0,
) -> None:
    """Render 3D scatter plot of sparse landmarks and current vehicle trajectory.

    Args:
        map_points: Array of shape (N, 3) for 3D landmark points [X, Y, Z].
        map_colors: Array of shape (N, 3) for RGB landmark colors.
        landmark_count: Total triangulated landmarks count.
        mean_reproj_err: Mean reprojection error in pixels.
        current_positions: Array of shape (M, 3) with vehicle trajectory positions.
        current_idx: Current active frame index.
    """
    st.markdown("### LOCAL SPARSE MAP (3D Triangulated Landmarks)")

    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Triangulated Landmarks", value=f"{landmark_count}")
    with m2:
        st.metric(label="Mean Reprojection Error", value=f"{mean_reproj_err:.3f} px")

    fig = go.Figure()

    # 1. 3D Landmarks
    if len(map_points) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=map_points[:, 1],  # East / Y
                y=map_points[:, 0],  # North / X
                z=-map_points[:, 2],  # Invert Z for Up display
                mode="markers",
                name="3D Landmarks",
                marker=dict(
                    size=2.5,
                    color="#005A9E",
                    opacity=0.75,
                ),
            )
        )

    # 2. 3D Camera Trajectory history
    if current_positions is not None and len(current_positions) > 0:
        hist_pos = current_positions[: current_idx + 1]
        fig.add_trace(
            go.Scatter3d(
                x=hist_pos[:, 1],
                y=hist_pos[:, 0],
                z=-hist_pos[:, 2],
                mode="lines",
                name="Camera Trajectory",
                line=dict(color="#D83B01", width=4),
            )
        )

        # 3. Current Camera Position
        curr_p = current_positions[current_idx]
        fig.add_trace(
            go.Scatter3d(
                x=[curr_p[1]],
                y=[curr_p[0]],
                z=[-curr_p[2]],
                mode="markers",
                name="Current Camera Pose",
                marker=dict(size=7, color="#E81123", symbol="diamond"),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Altitude (-Down) (m)",
            bgcolor="#FAFAFA",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("📌 **Note:** Precomputed sparse map from EuRoC demonstration sequence (Phase 5 keyframe triangulation).")
