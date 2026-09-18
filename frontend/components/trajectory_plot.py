"""2D trajectory comparison and position error plots using Plotly."""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_trajectory_plot(
    hybrid_positions: np.ndarray,
    baseline_positions: Optional[np.ndarray],
    gt_positions: Optional[np.ndarray],
    current_idx: int,
    show_ground_truth: bool = True,
    show_baseline: bool = False,
) -> None:
    """Render 2D top-down (North-East / X-Y) trajectory plot.

    Args:
        hybrid_positions: Array of shape (N, 3) for Hybrid VIO+ML trajectory.
        baseline_positions: Array of shape (N, 3) for Classical VIO trajectory.
        gt_positions: Array of shape (N, 3) for Ground Truth trajectory.
        current_idx: Current active frame index.
        show_ground_truth: Whether to render ground-truth path.
        show_baseline: Whether to render classical baseline path.
    """
    st.markdown("### ESTIMATED TRAJECTORY (North vs East)")

    fig = go.Figure()

    # 1. Ground Truth trajectory (full reference)
    if show_ground_truth and gt_positions is not None:
        valid_mask = ~np.isnan(gt_positions[:, 0])
        if np.any(valid_mask):
            fig.add_trace(
                go.Scatter(
                    x=gt_positions[valid_mask, 1],  # East (Y)
                    y=gt_positions[valid_mask, 0],  # North (X)
                    mode="lines",
                    name="Ground Truth (EuRoC)",
                    line=dict(color="#107C41", width=2.5, dash="dash"),
                )
            )

    # 2. Classical baseline trajectory (history up to current frame)
    if show_baseline and baseline_positions is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_positions[: current_idx + 1, 1],
                y=baseline_positions[: current_idx + 1, 0],
                mode="lines",
                name="Classical VIO Baseline",
                line=dict(color="#D83B01", width=1.8, dash="dot"),
            )
        )

    # 3. Hybrid VIO+ML trajectory (history up to current frame)
    fig.add_trace(
        go.Scatter(
            x=hybrid_positions[: current_idx + 1, 1],
            y=hybrid_positions[: current_idx + 1, 0],
            mode="lines",
            name="Hybrid VIO + ML (Estimated)",
            line=dict(color="#0078D4", width=2.5),
        )
    )

    # 4. Current Position Marker
    curr_x = hybrid_positions[current_idx, 0]
    curr_y = hybrid_positions[current_idx, 1]
    fig.add_trace(
        go.Scatter(
            x=[curr_y],
            y=[curr_x],
            mode="markers",
            name="Current Position",
            marker=dict(size=12, color="#D83B01", symbol="circle", line=dict(color="#FFFFFF", width=2)),
        )
    )

    fig.update_layout(
        xaxis_title="East / Y (m)",
        yaxis_title="North / X (m)",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=20, b=40),
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=True, gridcolor="#E1DFDD", zeroline=True, zerolinecolor="#605E5C"),
        yaxis=dict(showgrid=True, gridcolor="#E1DFDD", zeroline=True, zerolinecolor="#605E5C", scaleanchor="x", scaleratio=1),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("📌 **Note:** Ground truth trajectory is used only for offline software evaluation and demonstration comparison.")


def render_position_error_plot(
    timestamps: np.ndarray,
    position_errors: np.ndarray,
    current_idx: int,
) -> None:
    """Render position error over time with current statistics.

    Args:
        timestamps: Sequence timestamps in seconds.
        position_errors: Error array (norm of est - gt) in meters.
        current_idx: Current active frame index.
    """
    st.markdown("### POSITION ERROR OVER TIME")

    t_rel = timestamps[: current_idx + 1] - timestamps[0]
    errs = position_errors[: current_idx + 1]
    valid_errs = errs[~np.isnan(errs)]

    if len(valid_errs) > 0:
        curr_err = float(valid_errs[-1])
        rmse_err = float(np.sqrt(np.mean(valid_errs**2)))
        mean_err = float(np.mean(valid_errs))
    else:
        curr_err = 0.0
        rmse_err = 0.0
        mean_err = 0.0

    # Metrics row
    e1, e2, e3 = st.columns(3)
    with e1:
        st.metric(label="Current Position Error", value=f"{curr_err:5.2f} m")
    with e2:
        st.metric(label="ATE RMSE (So Far)", value=f"{rmse_err:5.2f} m")
    with e3:
        st.metric(label="Mean Error (So Far)", value=f"{mean_err:5.2f} m")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_rel,
            y=errs,
            mode="lines",
            name="Hybrid VIO Error",
            line=dict(color="#0078D4", width=2.0),
        )
    )

    fig.update_layout(
        xaxis_title="Elapsed Time (s)",
        yaxis_title="Position Error (m)",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=10, b=40),
        height=240,
        xaxis=dict(showgrid=True, gridcolor="#E1DFDD"),
        yaxis=dict(showgrid=True, gridcolor="#E1DFDD", rangemode="tozero"),
    )

    st.plotly_chart(fig, use_container_width=True)
