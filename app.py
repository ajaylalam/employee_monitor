"""
app.py - Main Streamlit application
This is the web interface for the Employee Activity Monitor.
It shows:
- Live video feed with detection boxes
- Real-time stats dashboard
- Activity graphs
- Absence alerts
- Downloadable session report

Run with: python3.11 -m streamlit run app.py
"""

# streamlit for the web UI
import streamlit as st

# OpenCV for video capture and processing
import cv2

# numpy for array operations
import numpy as np

# PIL for converting OpenCV frames to images Streamlit can display
from PIL import Image

# time for controlling frame rate and timestamps
import time

# datetime for readable timestamps
from datetime import datetime

# json for saving the report
import json

# os for file operations
import os

# plotly for interactive charts
import plotly.graph_objects as go
import plotly.express as px

# pandas for data manipulation
import pandas as pd

# import our custom tracker
from tracker import EmployeeMonitor, ABSENT_THRESHOLD_SECONDS, IDLE_THRESHOLD_SECONDS


# ── PAGE SETTINGS ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Activity Monitor",
    page_icon="👥",
    layout="wide"   # wide layout gives us more space for the dashboard
)

# ── CUSTOM STYLING ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .alert-card {
        background: #3d0000;
        border: 2px solid #ff0000;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .active-badge  { color: #00ff00; font-weight: bold; }
    .idle-badge    { color: #ffa500; font-weight: bold; }
    .absent-badge  { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────────
# Keep the monitor object alive between Streamlit reruns
if "monitor"      not in st.session_state: st.session_state.monitor      = None
if "running"      not in st.session_state: st.session_state.running      = False
if "activity_log" not in st.session_state: st.session_state.activity_log = []
if "frame_count"  not in st.session_state: st.session_state.frame_count  = 0


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.title("👥 Employee Activity Monitor")
st.markdown("Real-time workplace monitoring using YOLOv8 object detection and tracking")
st.divider()


# ── LAYOUT: Two columns - Video on left, Dashboard on right ───────────────────
col_video, col_dashboard = st.columns([3, 2])


with col_video:
    st.subheader("📹 Live Detection Feed")

    # Source selection - webcam or video file
    source_type = st.radio(
        "Video Source",
        ["Webcam", "Video File"],
        horizontal=True
    )

    video_file = None
    if source_type == "Video File":
        video_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov"],
            help="Upload a workplace video to analyse"
        )

    # Settings expander
    with st.expander("⚙️ Settings"):
        absent_threshold = st.slider(
            "Absent Alert Threshold (seconds)",
            min_value=10, max_value=300, value=60,
            help="How long someone must be absent before an alert is shown"
        )
        idle_threshold = st.slider(
            "Idle Threshold (seconds)",
            min_value=5, max_value=120, value=30,
            help="How long without movement before marking as idle"
        )
        show_confidence = st.checkbox("Show detection confidence", value=True)

    # Start / Stop buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("▶ Start Monitoring", use_container_width=True, type="primary")
    with btn_col2:
        stop_btn  = st.button("⏹ Stop Monitoring",  use_container_width=True)

    # Handle start
    if start_btn:
        st.session_state.monitor      = EmployeeMonitor()
        st.session_state.running      = True
        st.session_state.activity_log = []
        st.session_state.frame_count  = 0

    # Handle stop
    if stop_btn:
        st.session_state.running = False

    # Video display placeholder - this is where frames appear
    video_placeholder = st.empty()

    # Alert placeholder - shows red banners for absent people
    alert_placeholder = st.empty()


with col_dashboard:
    st.subheader("📊 Live Dashboard")

    # Metric cards placeholders
    metrics_placeholder = st.empty()

    st.divider()

    # Activity chart placeholder
    st.subheader("📈 Activity Timeline")
    chart_placeholder = st.empty()

    st.divider()

    # People list placeholder
    st.subheader("👤 People Status")
    people_placeholder = st.empty()

    st.divider()

    # Report download placeholder
    report_placeholder = st.empty()


# ── MAIN MONITORING LOOP ───────────────────────────────────────────────────────
if st.session_state.running and st.session_state.monitor is not None:

    monitor = st.session_state.monitor

    # Open video source
    if source_type == "Webcam":
        # 0 = default webcam on your laptop
        cap = cv2.VideoCapture(0)
    else:
        if video_file is None:
            st.warning("Please upload a video file first.")
            st.session_state.running = False
            cap = None
        else:
            # Save uploaded file temporarily
            temp_video_path = f"./temp_video.{video_file.name.split('.')[-1]}"
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture(temp_video_path)

    if cap is not None and cap.isOpened():

        # Run while monitoring is active
        while st.session_state.running:

            # Read one frame from the video
            ret, frame = cap.read()

            # If end of video file, loop back to start
            if not ret:
                if source_type == "Video File":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Run detection and tracking on this frame
            annotated_frame = monitor.update(frame)

            # Convert BGR (OpenCV format) to RGB (what Streamlit expects)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the video placeholder
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

            # ── UPDATE DASHBOARD ──────────────────────────────────────────────
            # Only update dashboard every 10 frames to save performance
            if st.session_state.frame_count % 10 == 0:

                trackers = monitor.trackers

                # Count people by status
                active_count = sum(1 for t in trackers.values() if t.status == "active")
                idle_count   = sum(1 for t in trackers.values() if t.status == "idle")
                absent_count = sum(1 for t in trackers.values() if t.status == "absent")
                total_count  = len(trackers)

                # ── METRIC CARDS ──────────────────────────────────────────────
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("👥 Total",   total_count)
                    m2.metric("🟢 Active",  active_count)
                    m3.metric("🟡 Idle",    idle_count)
                    m4.metric("🔴 Absent",  absent_count)

                # ── ABSENCE ALERTS ────────────────────────────────────────────
                with alert_placeholder.container():
                    alerts = []
                    for tid, tracker in trackers.items():
                        if tracker.status == "absent":
                            absent_secs = tracker.get_absent_duration_seconds()
                            if absent_secs > absent_threshold:
                                mins = int(absent_secs // 60)
                                secs = int(absent_secs % 60)
                                alerts.append(f"🚨 Person {tid} has been ABSENT for {mins:02d}:{secs:02d}")

                    if alerts:
                        for alert in alerts:
                            st.error(alert)

                # ── ACTIVITY TIMELINE CHART ───────────────────────────────────
                # Record current activity snapshot
                st.session_state.activity_log.append({
                    "time":   datetime.now().strftime("%H:%M:%S"),
                    "active": active_count,
                    "idle":   idle_count,
                    "absent": absent_count
                })

                # Only keep last 60 data points
                if len(st.session_state.activity_log) > 60:
                    st.session_state.activity_log = st.session_state.activity_log[-60:]

                # Create activity chart
                if len(st.session_state.activity_log) > 1:
                    df = pd.DataFrame(st.session_state.activity_log)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["time"], y=df["active"],
                                            name="Active", line=dict(color="#00ff00", width=2),
                                            fill="tozeroy", fillcolor="rgba(0,255,0,0.1)"))
                    fig.add_trace(go.Scatter(x=df["time"], y=df["idle"],
                                            name="Idle", line=dict(color="#ffa500", width=2),
                                            fill="tozeroy", fillcolor="rgba(255,165,0,0.1)"))
                    fig.add_trace(go.Scatter(x=df["time"], y=df["absent"],
                                            name="Absent", line=dict(color="#ff0000", width=2),
                                            fill="tozeroy", fillcolor="rgba(255,0,0,0.1)"))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        legend=dict(orientation="h"),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                # ── PEOPLE STATUS LIST ────────────────────────────────────────
                with people_placeholder.container():
                    for tid, tracker in trackers.items():
                        summary = tracker.get_summary()
                        status_emoji = {"active": "🟢", "idle": "🟡", "absent": "🔴"}.get(tracker.status, "⚪")

                        absent_str = ""
                        if tracker.status == "absent":
                            secs = tracker.get_absent_duration_seconds()
                            absent_str = f" | Gone: {int(secs//60):02d}:{int(secs%60):02d}"

                        st.markdown(
                            f"{status_emoji} **Person {tid}** — "
                            f"Active: {summary['active_minutes']}min | "
                            f"Idle: {summary['idle_minutes']}min | "
                            f"Activity: {summary['activity_percent']}%"
                            f"{absent_str}"
                        )

                # ── REPORT DOWNLOAD ───────────────────────────────────────────
                with report_placeholder.container():
                    report_data = monitor.get_full_report()
                    report_json = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="📥 Download Session Report",
                        data=report_json,
                        file_name=f"activity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            st.session_state.frame_count += 1

            # Small delay to control frame rate
            time.sleep(0.03)  # roughly 30fps

        # Release video capture when done
        cap.release()
        if source_type == "Video File" and os.path.exists("./temp_video.mp4"):
            os.remove("./temp_video.mp4")

    else:
        st.error("Could not open video source. Make sure your webcam is connected or upload a valid video file.")
        st.session_state.running = False

# ── SHOW INSTRUCTIONS WHEN NOT RUNNING ────────────────────────────────────────
else:
    with video_placeholder.container():
        st.info("""
        **How to use:**
        1. Select Webcam or upload a Video File
        2. Adjust the alert thresholds if needed
        3. Click **▶ Start Monitoring**
        4. The AI will detect and track all people in the frame
        5. Click **⏹ Stop Monitoring** when done
        6. Download the session report

        **What you will see:**
        - 🟢 Green box = Active person (moving)
        - 🟡 Orange box = Idle person (present but not moving)
        - 🔴 Red alert = Person has been absent too long
        """)
