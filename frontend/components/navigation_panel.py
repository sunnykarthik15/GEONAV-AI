"""Navigation state and ML velocity residual correction panels."""

from typing import Optional, Sequence, Tuple
import streamlit as st


def render_navigation_panel(
    position: Tuple[float, float, float],
    velocity: Tuple[float, float, float],
    euler_deg: Tuple[float, float, float],
) -> None:
    """Render 3D navigation state estimate with SI units and coordinate frame annotations.

    Args:
        position: Estimated (X, Y, Z) position in meters [NED].
        velocity: Estimated (Vx, Vy, Vz) velocity in m/s [NED].
        euler_deg: Estimated (Roll, Pitch, Yaw) in degrees.
    """
    st.markdown("### NAVIGATION STATE")
    st.caption("Coordinates: World **NED** (+X North, +Y East, +Z Down) | Body **FRD** (+X Fwd, +Y Right, +Z Down)")

    # Position
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.metric(label="Pos X (North)", value=f"{position[0]:+7.3f} m")
    with p_col2:
        st.metric(label="Pos Y (East)", value=f"{position[1]:+7.3f} m")
    with p_col3:
        st.metric(label="Pos Z (Down)", value=f"{position[2]:+7.3f} m")

    # Velocity
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.metric(label="Vel Vx", value=f"{velocity[0]:+6.3f} m/s")
    with v_col2:
        st.metric(label="Vel Vy", value=f"{velocity[1]:+6.3f} m/s")
    with v_col3:
        st.metric(label="Vel Vz", value=f"{velocity[2]:+6.3f} m/s")

    # Orientation
    o_col1, o_col2, o_col3 = st.columns(3)
    with o_col1:
        st.metric(label="Roll (φ)", value=f"{euler_deg[0]:+6.1f}°")
    with o_col2:
        st.metric(label="Pitch (θ)", value=f"{euler_deg[1]:+6.1f}°")
    with o_col3:
        st.metric(label="Yaw (ψ)", value=f"{euler_deg[2]:+6.1f}°")


def render_ml_correction_panel(
    ml_status: str,
    confidence: float,
    delta_v: Tuple[float, float, float],
) -> None:
    """Render ML residual correction status, confidence gating, and correction vectors.

    Args:
        ml_status: 'ACTIVE' or 'FALLBACK'.
        confidence: Prediction confidence in [0.0, 1.0].
        delta_v: Applied velocity correction (ΔVx, ΔVy, ΔVz) in m/s.
    """
    st.markdown("### AI / ML RESIDUAL CORRECTION")

    st.markdown(
        """
        ```
        Classical VIO  ──>  Causal Features (18D)  ──>  NumpyEdgeMLP (25.7 KB)  ──>  ΔV Correction  ──>  Corrected State
        ```
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if ml_status == "ACTIVE":
            status_display = "🟢 ACTIVE"
        else:
            status_display = "🟡 FALLBACK"
        st.metric(label="ML Status", value=status_display)

    with c2:
        st.metric(label="Confidence", value=f"{confidence * 100:4.1f}%")

    with c3:
        corr_mag = (delta_v[0] ** 2 + delta_v[1] ** 2 + delta_v[2] ** 2) ** 0.5
        st.metric(label="||ΔV|| Magnitude", value=f"{corr_mag:5.3f} m/s")

    with c4:
        st.metric(label="Vertical ΔVz", value=f"{delta_v[2]:+5.3f} m/s")

    # Values breakdown
    st.caption(
        f"**Live Residuals:** ΔVx = `{delta_v[0]:+6.3f} m/s` | "
        f"ΔVy = `{delta_v[1]:+6.3f} m/s` | "
        f"ΔVz = `{delta_v[2]:+6.3f} m/s`"
    )

    st.info(
        "ℹ️ **Note:** The ML model corrects velocity residual drift in the classical VIO estimate. "
        "It does not replace the geometric VIO pipeline. No direct control commands are issued.",
        icon="ℹ️",
    )
