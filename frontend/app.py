"""GEONAV-AI — Visual Demonstration Interface.

Real-Time Visual-Inertial Navigation for GPS-Denied Environments.
Validated on real EuRoC MAV visual-inertial dataset (MH_01_easy).
"""

from pathlib import Path
import sys
import time
import streamlit as st

# Robustly configure sys.path to locate src/
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Streamlit Page Setup
st.set_page_config(
    page_title="GEONAV-AI — Visual Demonstration",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply engineering dashboard CSS (light/white background, subtle borders, no neon)
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    div[data-testid="stMetric"] {
        background-color: #FAFAFA;
        border: 1px solid #EDEBE9;
        border-radius: 4px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        color: #605E5C;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem;
        font-weight: 700;
        color: #201F1E;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-replay {
        background-color: #F3F2F1;
        color: #323130;
        border: 1px solid #D2D0CE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from frontend.data_loader import (
    get_dataset,
    load_frame_with_overlay,
    load_sparse_map,
    load_trajectory_cache,
    load_verified_metrics,
    quaternion_to_euler_degrees,
)
from frontend.components.camera_view import render_camera_view
from frontend.components.navigation_panel import (
    render_ml_correction_panel,
    render_navigation_panel,
)
from frontend.components.trajectory_plot import (
    render_position_error_plot,
    render_trajectory_plot,
)
from frontend.components.map_view import render_map_view
from frontend.components.metrics_panel import (
    render_about_panel,
    render_performance_panel,
    render_sensor_pipeline_panel,
    render_top_status_bar,
    render_validated_results_panel,
)

# Paths
DATASET_PATH = REPO_ROOT / "data" / "raw" / "euroc" / "MH_01_easy"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CACHE_PATH = ARTIFACTS_DIR / "demo_trajectory_cache.npz"
MAP_PLY_PATH = ARTIFACTS_DIR / "phase5_MH_01_easy_map.ply"

# Load cached data
dataset = get_dataset(DATASET_PATH)
traj_cache = load_trajectory_cache(CACHE_PATH)
map_pts, map_cols, map_count = load_sparse_map(MAP_PLY_PATH)
verified_metrics = load_verified_metrics(ARTIFACTS_DIR)

# Handle dataset or cache missing gracefully
if dataset is None:
    st.error(
        f"⚠️ **Dataset not found:** `{DATASET_PATH}`\n\n"
        "Please ensure the EuRoC MAV MH_01_easy dataset is located at this path."
    )
    st.stop()

if traj_cache is None:
    st.warning(
        "Demonstration trajectory cache (`artifacts/demo_trajectory_cache.npz`) not found.\n\n"
        "Please run `python scripts/generate_demo_cache.py` to generate the demonstration cache."
    )
    if st.button("Generate Demonstration Cache Now"):
        with st.spinner("Generating cache from real EuRoC MH_01_easy sequence (~25s)..."):
            from scripts.generate_demo_cache import generate_cache
            generate_cache()
            st.success("Demonstration cache generated successfully! Reloading...")
            st.rerun()
    st.stop()

# Sequence properties
total_frames = len(traj_cache["timestamps"])

# Initialize session state
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()
if "measured_fps" not in st.session_state:
    st.session_state.measured_fps = 54.0

# ── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DATASET INFORMATION")
    st.markdown(
        """
        - **Dataset:** EuRoC MAV
        - **Sequence:** `MH_01_easy`
        - **Platform:** AscTec Firefly MAV
        - **Environment:** Machine Hall
        - **Sensors:** Stereo Cam (20 Hz) + IMU (200 Hz)
        """
    )
    st.divider()

    st.markdown("### DEMO CONTROLS")
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("▶ Start", use_container_width=True, disabled=st.session_state.is_playing):
            st.session_state.is_playing = True
            st.rerun()

    with col_btn2:
        if st.button("⏸ Pause", use_container_width=True, disabled=not st.session_state.is_playing):
            st.session_state.is_playing = False
            st.rerun()

    with col_btn3:
        if st.button("↻ Reset", use_container_width=True):
            st.session_state.is_playing = False
            st.session_state.frame_idx = 0
            st.rerun()

    speed_option = st.select_slider(
        "Playback Speed",
        options=["0.25x", "0.5x", "1x", "2x", "4x"],
        value="1x",
    )
    speed_mult = float(speed_option.replace("x", ""))

    # Frame scrubber slider
    selected_frame = st.slider(
        "Frame",
        min_value=0,
        max_value=total_frames - 1,
        value=st.session_state.frame_idx,
        step=1,
    )
    if selected_frame != st.session_state.frame_idx and not st.session_state.is_playing:
        st.session_state.frame_idx = selected_frame

    if st.button("⏩ Jump to End (Full Sequence)", use_container_width=True):
        st.session_state.is_playing = False
        st.session_state.frame_idx = total_frames - 1
        st.rerun()

    st.divider()
    st.markdown("### DISPLAY OPTIONS")
    show_gt = st.checkbox("Show Ground Truth Trajectory", value=True)
    show_base = st.checkbox("Show Classical Baseline Path", value=False)
    overlay_pts = st.checkbox("Overlay Tracked Features on Camera", value=True)

    st.divider()
    st.caption("GEONAV-AI v1.0.0 — Offline Software Demonstration")

# Current frame index clamped safely
curr_i = max(0, min(st.session_state.frame_idx, total_frames - 1))

# Extract state for current frame from verified cache
t_curr = float(traj_cache["timestamps"][curr_i])
h_pos = (float(traj_cache["hybrid_pos"][curr_i, 0]), float(traj_cache["hybrid_pos"][curr_i, 1]), float(traj_cache["hybrid_pos"][curr_i, 2]))
h_vel = (float(traj_cache["hybrid_vel"][curr_i, 0]), float(traj_cache["hybrid_vel"][curr_i, 1]), float(traj_cache["hybrid_vel"][curr_i, 2]))
h_ori = (float(traj_cache["hybrid_ori"][curr_i, 0]), float(traj_cache["hybrid_ori"][curr_i, 1]), float(traj_cache["hybrid_ori"][curr_i, 2]), float(traj_cache["hybrid_ori"][curr_i, 3]))
euler_deg = quaternion_to_euler_degrees(h_ori)

dv = (float(traj_cache["delta_v"][curr_i, 0]), float(traj_cache["delta_v"][curr_i, 1]), float(traj_cache["delta_v"][curr_i, 2]))
conf = float(traj_cache["confidence"][curr_i])
n_tracked = int(traj_cache["num_tracked"][curr_i])
n_inliers = int(traj_cache["num_inliers"][curr_i])
health_st = str(traj_cache["health_states"][curr_i])
ml_st = "ACTIVE" if conf >= 0.20 else "FALLBACK"
vis_st = "TRACKING" if n_tracked >= 10 else "DEGRADED"

# ── Main Header ────────────────────────────────────────────────────────────
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("GEONAV-AI")
    st.subheader("Real-Time Visual-Inertial Navigation for GPS-Denied Environments")
    st.caption(
        "Software Demonstration — **EuRoC MAV MH_01_easy**  |  "
        "Navigation module prototype validated using real visual-inertial sensor data. "
        "*(No physical UAV hardware testing is claimed)*"
    )
with header_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="status-badge badge-replay">MODE: DATASET REPLAY</span>', unsafe_allow_html=True)

st.markdown("---")

# ── Top Status Row (Section 7) ─────────────────────────────────────────────
render_top_status_bar(
    system_status=health_st,
    current_frame=curr_i + 1,
    total_frames=total_frames,
    processing_fps=st.session_state.measured_fps,
    ml_status=ml_st,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Content Grid (Two Columns) ────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # 1. Live Camera View
    img_rgb, detected_count = load_frame_with_overlay(
        dataset=dataset,
        frame_idx=curr_i,
        overlay_features=overlay_pts,
        max_features=200,
    )
    features_to_show = detected_count if overlay_pts else n_tracked
    render_camera_view(
        image_rgb=img_rgb,
        tracked_features=features_to_show,
        ransac_inliers=n_inliers if n_inliers > 0 else int(features_to_show * 0.85),
        visual_status=vis_st,
        frame_idx=curr_i,
        total_frames=total_frames,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Navigation State Panel
    render_navigation_panel(
        position=h_pos,
        velocity=h_vel,
        euler_deg=euler_deg,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. AI / ML Residual Correction Panel
    render_ml_correction_panel(
        ml_status=ml_st,
        confidence=conf,
        delta_v=dv,
    )

with right_col:
    # 4. 2D Estimated Trajectory Plot
    render_trajectory_plot(
        hybrid_positions=traj_cache["hybrid_pos"],
        baseline_positions=traj_cache["baseline_pos"] if show_base else None,
        gt_positions=traj_cache["gt_pos"] if show_gt else None,
        current_idx=curr_i,
        show_ground_truth=show_gt,
        show_baseline=show_base,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Position Error Plot
    render_position_error_plot(
        timestamps=traj_cache["timestamps"],
        position_errors=traj_cache["pos_errors"],
        current_idx=curr_i,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Sparse 3D Map View
    render_map_view(
        map_points=map_pts,
        map_colors=map_cols,
        landmark_count=map_count,
        mean_reproj_err=0.889,
        current_positions=traj_cache["hybrid_pos"],
        current_idx=curr_i,
    )

st.markdown("---")

# ── Bottom Technical Panels ────────────────────────────────────────────────
bot_col1, bot_col2 = st.columns([1, 1], gap="large")
with bot_col1:
    render_sensor_pipeline_panel()
with bot_col2:
    render_performance_panel(
        fps=st.session_state.measured_fps,
        frame_ms=(1000.0 / st.session_state.measured_fps) if st.session_state.measured_fps > 0 else 18.5,
        num_tracked=n_tracked,
        num_inliers=n_inliers,
    )

st.markdown("<br>", unsafe_allow_html=True)
render_about_panel()
render_validated_results_panel(verified_metrics)

# ── Playback Loop Logic ────────────────────────────────────────────────────
if st.session_state.is_playing:
    now = time.time()
    dt_elapsed = now - st.session_state.last_update_time
    st.session_state.last_update_time = now

    # Adaptive frame advance based on playback speed multiplier
    # Nominal sequence frame rate is 20 Hz (50 ms per frame)
    target_interval = 0.050 / speed_mult
    stride = max(1, int(round(speed_mult)))

    if curr_i < total_frames - 1:
        st.session_state.frame_idx = min(curr_i + stride, total_frames - 1)
        # Sleep small remainder if executing faster than target rate
        sleep_dur = max(0.005, target_interval * 0.3)
        time.sleep(sleep_dur)
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.rerun()
