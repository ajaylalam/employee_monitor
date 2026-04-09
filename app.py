"""
app.py - Employee Activity Monitor
Run with: python3.11 -m streamlit run app.py

This is the front-end of the AK VISION system.
It handles the user interface (built with Streamlit), the live video feed,
the real-time dashboard, and PDF report generation.

It works together with tracker.py — this file handles the UI,
tracker.py handles all the detection and tracking logic.
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF
from tracker import EmployeeMonitor, ABSENT_THRESHOLD_SECONDS, IDLE_THRESHOLD_SECONDS

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# This must be the FIRST Streamlit call in the file.
# Sets the browser tab title, icon, and layout mode.
st.set_page_config(
    page_title="AK VISION | Activity Monitor",
    page_icon="👁",
    layout="wide",           # Use the full browser width
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS STYLING ────────────────────────────────────────────────────────
# Injects custom CSS to give the app its futuristic dark/cyan look.
# This overrides Streamlit's default styles with a custom design system
# using the Orbitron, Share Tech Mono, and Rajdhani fonts from Google Fonts.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');

:root {
    --cyan:   #00f5ff;
    --green:  #00ff88;
    --red:    #ff2d55;
    --orange: #ff9500;
    --bg:     #020810;
    --bg2:    #050d1a;
    --bg3:    #0a1628;
    --border: rgba(0,245,255,0.15);
    --glow:   0 0 20px rgba(0,245,255,0.3);
}

* { box-sizing: border-box; }
html, body, .stApp { background: var(--bg) !important; color: #c8e6f0 !important; font-family: 'Rajdhani', sans-serif !important; }
.block-container { padding: 1rem 2rem !important; max-width: 100% !important; }
header, footer, #MainMenu { visibility: hidden; }
.stDeployButton { display: none; }

.nexus-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1.5rem;
    background: linear-gradient(90deg, rgba(0,245,255,0.05), transparent);
    border: 1px solid var(--border); border-radius: 4px; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.nexus-header::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--cyan); box-shadow: 0 0 15px var(--cyan);
}
.nexus-title {
    font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 900;
    color: var(--cyan); letter-spacing: 6px;
    text-shadow: 0 0 30px rgba(0,245,255,0.7), 0 0 60px rgba(0,245,255,0.3);
}
.nexus-subtitle { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: rgba(0,245,255,0.4); letter-spacing: 2px; margin-top: 2px; }
.nexus-time { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: var(--green); text-shadow: 0 0 10px var(--green); }

.panel-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: var(--cyan); letter-spacing: 3px; opacity: 0.6; margin-bottom: 0.4rem;
}

.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 0.8rem; }
.metric-card {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 3px; padding: 0.7rem 0.5rem; text-align: center; position: relative;
}
.metric-card::after { content: ''; position: absolute; top: 0; right: 0; width: 20px; height: 20px; border-top: 1px solid; border-right: 1px solid; border-color: inherit; opacity: 0.4; }
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; font-weight: 700; line-height: 1; }
.metric-label { font-family: 'Share Tech Mono', monospace; font-size: 0.5rem; letter-spacing: 2px; opacity: 0.5; margin-top: 3px; }
.m-total  .metric-value { color: var(--cyan);   text-shadow: 0 0 20px var(--cyan); }
.m-total  { border-color: var(--cyan) !important; }
.m-active .metric-value { color: var(--green);  text-shadow: 0 0 20px var(--green); }
.m-active { border-color: var(--green) !important; }
.m-idle   .metric-value { color: var(--orange); text-shadow: 0 0 20px var(--orange); }
.m-idle   { border-color: var(--orange) !important; }
.m-absent .metric-value { color: var(--red);    text-shadow: 0 0 20px var(--red); }
.m-absent { border-color: var(--red) !important; }

.person-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.45rem 0.7rem;
    background: var(--bg3); border-left: 3px solid var(--cyan);
    margin-bottom: 0.35rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.68rem;
}
.p-active { border-left-color: var(--green) !important; }
.p-idle   { border-left-color: var(--orange) !important; }
.p-absent { border-left-color: var(--red) !important; animation: pulse-row 1s infinite; }
@keyframes pulse-row { 0%,100%{opacity:1} 50%{opacity:0.6} }

.alert-bar {
    background: rgba(255,45,85,0.08); border: 1px solid var(--red);
    border-radius: 3px; padding: 0.4rem 0.8rem; margin-bottom: 0.3rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.68rem;
    color: var(--red); text-shadow: 0 0 8px var(--red);
    animation: pulse-alert 1s infinite;
}
@keyframes pulse-alert { 0%,100%{opacity:1} 50%{opacity:0.5} }

.stButton > button {
    font-family: 'Orbitron', sans-serif !important; font-size: 0.6rem !important;
    letter-spacing: 2px !important; border-radius: 2px !important;
    border: 1px solid var(--cyan) !important; background: transparent !important;
    color: var(--cyan) !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: rgba(0,245,255,0.08) !important; box-shadow: var(--glow) !important; }
.stButton > button[kind="primary"] { background: rgba(0,245,255,0.06) !important; }

.stRadio label, .stSlider label {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.68rem !important;
    color: rgba(0,245,255,0.6) !important; letter-spacing: 1px !important;
}
.streamlit-expanderHeader {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.68rem !important;
    color: var(--cyan) !important; background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
}
.stImage img { border: 1px solid var(--border) !important; box-shadow: 0 0 40px rgba(0,245,255,0.08) !important; }
.stDownloadButton > button {
    font-family: 'Orbitron', sans-serif !important; font-size: 0.58rem !important;
    letter-spacing: 2px !important; border-radius: 2px !important;
    border: 1px solid var(--green) !important; background: rgba(0,255,136,0.04) !important;
    color: var(--green) !important; width: 100% !important;
}
hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: var(--cyan); }
.stApp::before {
    content: ''; position: fixed; top:0; left:0; width:100%; height:100%;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.02) 2px,rgba(0,0,0,0.02) 4px);
    pointer-events: none; z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction.
# st.session_state persists variables across those reruns so we don't lose data.
# All key app variables are initialised here with defaults if they don't exist yet.
if "monitor" not in st.session_state:
    st.session_state.monitor = None          # Will hold the EmployeeMonitor instance
if "running" not in st.session_state:
    st.session_state.running = False         # True while monitoring is active
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []       # Log of active/idle/absent counts over time
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0         # Total frames processed this session
if "cap" not in st.session_state:
    st.session_state.cap = None              # OpenCV VideoCapture object (camera or video file)
if "source_type" not in st.session_state:
    st.session_state.source_type = "Webcam" # Default input source
if "camera_index" not in st.session_state:
    st.session_state.camera_index = 0       # Default to the first camera
if "absent_threshold" not in st.session_state:
    st.session_state.absent_threshold = 60  # Seconds before marking someone absent
if "idle_threshold" not in st.session_state:
    st.session_state.idle_threshold = 30    # Seconds of no movement before marking idle

# ── HEADER ────────────────────────────────────────────────────────────────────
# Renders the top banner with the app name and current system status.
# The status dot changes based on whether monitoring is running or not.
status_dot = "● MONITORING ACTIVE" if st.session_state.running else "○ SYSTEM STANDBY"
st.markdown(f"""
<div class="nexus-header">
    <div>
        <div class="nexus-title">AK VISION</div>
        <div class="nexus-subtitle">AK VISION · EMPLOYEE ACTIVITY MONITOR · POWERED BY YOLOV8</div>
    </div>
    <div class="nexus-time">{status_dot}</div>
</div>
""", unsafe_allow_html=True)

# ── PDF REPORT GENERATOR ──────────────────────────────────────────────────────
def generate_pdf_report(report: dict) -> bytes:
    """
    Builds a formatted PDF report from the session data and returns it as bytes.
    The PDF includes:
      - A styled title block
      - Session start/end times and total people detected
      - A colour-coded table with per-person stats
      - A plain-English observations section summarising what happened
      - A footer with the generation timestamp

    The 'report' dict comes from EmployeeMonitor.get_full_report() in tracker.py.
    Returns raw bytes so Streamlit can offer it as a file download.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title block
    # Dark navy background bar across the top of the page
    pdf.set_fill_color(10, 20, 40)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 18, "AK VISION - Employee Activity Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(160, 210, 230)
    pdf.cell(0, 8, "YOLOv8 Neural Detection Engine  |  Confidential", ln=True, align="C")
    pdf.ln(8)

    # ── Session info
    # Displays the session start/end time and how many people were tracked
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Session Summary", ln=True)
    pdf.set_draw_color(0, 180, 220)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Horizontal divider line
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(50, 6, "Session Start:", border=0)
    pdf.cell(0, 6, report["session_start"], ln=True)
    pdf.cell(50, 6, "Session End:", border=0)
    pdf.cell(0, 6, report["session_end"], ln=True)
    pdf.cell(50, 6, "Total People Detected:", border=0)
    pdf.cell(0, 6, str(report["total_people"]), ln=True)
    pdf.ln(6)

    # ── Per-person table
    # Shows a row for each tracked person with their activity breakdown
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Personnel Breakdown", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Column widths and header labels for the table
    col_w = [20, 28, 32, 32, 32, 36, 20]
    headers = ["Unit", "Status", "Active (min)", "Idle (min)", "Absent (min)", "Activity %", "Alert"]
    pdf.set_fill_color(10, 20, 40)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 8, h, border=1, fill=True, align="C")
    pdf.ln()

    # Background colours for each row depend on the person's current status
    status_colors = {
        "active": (220, 255, 220),  # Light green
        "idle":   (255, 240, 210),  # Light orange
        "absent": (255, 220, 220),  # Light red
    }
    pdf.set_font("Helvetica", "", 9)
    for p in report["people"]:
        status = p["status"]
        r, g, b = status_colors.get(status, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(col_w[0], 7, f"Unit {p['id']:02d}", border=1, fill=True, align="C")
        pdf.cell(col_w[1], 7, status.upper(), border=1, fill=True, align="C")
        pdf.cell(col_w[2], 7, str(p["active_minutes"]), border=1, fill=True, align="C")
        pdf.cell(col_w[3], 7, str(p["idle_minutes"]), border=1, fill=True, align="C")
        pdf.cell(col_w[4], 7, str(p["absent_minutes"]), border=1, fill=True, align="C")
        pdf.cell(col_w[5], 7, f"{p['activity_percent']}%", border=1, fill=True, align="C")
        # Show "YES" if an absence alert was triggered for this person, otherwise "-"
        pdf.cell(col_w[6], 7, "YES" if p["alert"] else "-", border=1, fill=True, align="C")
        pdf.ln()

    pdf.ln(8)

    # ── Plain-English narrative
    # Auto-generates a human-readable summary of the session findings
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Observations", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    people = report["people"]
    total  = len(people)

    if total == 0:
        pdf.multi_cell(0, 6, "No personnel were detected during this session.")
    else:
        # Split people into groups by their final status
        active_list  = [p for p in people if p["status"] == "active"]
        idle_list    = [p for p in people if p["status"] == "idle"]
        absent_list  = [p for p in people if p["status"] == "absent"]
        alerted      = [p for p in people if p["alert"]]  # Anyone who exceeded absence threshold

        lines = [
            f"During this session, {total} individual(s) were tracked by the AK VISION system.",
            "",
            f"At the time of export, {len(active_list)} person(s) were ACTIVE (moving), "
            f"{len(idle_list)} were IDLE (present but stationary), and "
            f"{len(absent_list)} had left the monitored area.",
            "",
        ]

        # Add a specific callout if any absence alerts were triggered
        if alerted:
            names = ", ".join(f"Unit {p['id']:02d}" for p in alerted)
            lines.append(
                f"ABSENCE ALERT was triggered for: {names}. "
                "These individuals exceeded the configured absence threshold and may require follow-up."
            )
            lines.append("")

        # Highlight the most and least active people
        top = max(people, key=lambda p: p["activity_percent"])
        low = min(people, key=lambda p: p["activity_percent"])
        lines.append(
            f"Highest activity: Unit {top['id']:02d} at {top['activity_percent']}% "
            f"({top['active_minutes']} min active)."
        )
        if total > 1:
            lines.append(
                f"Lowest activity: Unit {low['id']:02d} at {low['activity_percent']}% "
                f"({low['active_minutes']} min active, {low['absent_minutes']} min absent)."
            )

        # Calculate and show the team's average activity rate
        avg_activity = round(sum(p["activity_percent"] for p in people) / total, 1)
        lines.append(f"\nOverall team activity average: {avg_activity}%.")

        for line in lines:
            pdf.multi_cell(0, 6, line)

    # ── Footer
    # Shows who generated the report and when, at the bottom of the last page
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, f"Generated by AK VISION  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             align="C")

    return bytes(pdf.output())


# ── CAMERA ENUMERATION ────────────────────────────────────────────────────────
@st.cache_resource
def get_available_cameras():
    """
    Scans camera indices 0 through 9 and returns a list of indices
    that have a working camera attached.

    This is cached with @st.cache_resource so it only runs once per session —
    scanning cameras is slow and we don't want to repeat it on every page rerun.
    """
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    return cameras

# ── CONTROLS ──────────────────────────────────────────────────────────────────
# These controls are intentionally placed OUTSIDE the fragment functions below.
# Streamlit fragments re-run independently, and putting interactive widgets
# inside them would cause issues with file uploaders and other stateful widgets.
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 1, 1, 3])
with ctrl1:
    source_type = st.radio("INPUT SOURCE", ["Webcam", "Video File"], horizontal=True)
    st.session_state.source_type = source_type
with ctrl2:
    start_btn = st.button("▶ INITIATE", use_container_width=True, type="primary")
with ctrl3:
    stop_btn = st.button("■ TERMINATE", use_container_width=True)

# Show camera selector if using webcam, or file uploader if using video file
video_file = None
if source_type == "Webcam":
    available_cameras = get_available_cameras()
    if len(available_cameras) > 1:
        # Let the user pick which camera to use if multiple are available
        cam_labels = [f"Camera {i}" for i in available_cameras]
        selected_label = st.selectbox(
            "SELECT CAMERA", cam_labels,
            index=available_cameras.index(st.session_state.camera_index)
                  if st.session_state.camera_index in available_cameras else 0
        )
        st.session_state.camera_index = available_cameras[cam_labels.index(selected_label)]
    elif available_cameras:
        st.session_state.camera_index = available_cameras[0]  # Only one camera — use it automatically
    else:
        st.warning("No cameras detected.")
elif source_type == "Video File":
    video_file = st.file_uploader("UPLOAD FOOTAGE", type=["mp4", "avi", "mov"])

# Collapsible panel for tuning the detection thresholds
with st.expander("⚙ SYSTEM CONFIGURATION"):
    st.session_state.absent_threshold = st.slider(
        "ABSENT ALERT THRESHOLD (sec)", 10, 300, st.session_state.absent_threshold)
    st.session_state.idle_threshold = st.slider(
        "IDLE DETECTION THRESHOLD (sec)", 5, 120, st.session_state.idle_threshold)

# ── START / STOP LOGIC ────────────────────────────────────────────────────────

# When the user clicks INITIATE — set up a fresh monitoring session
if start_btn:
    # Release any previously open camera/video before opening a new one
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    st.session_state.monitor = EmployeeMonitor()  # Create a fresh tracker instance
    st.session_state.running = True
    st.session_state.activity_log = []
    st.session_state.frame_count = 0

    if source_type == "Webcam":
        st.session_state.cap = cv2.VideoCapture(st.session_state.camera_index)
        if st.session_state.cap.isOpened():
            # Set resolution to 640×480
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Flush the first few frames — webcams often return dark/blurry frames on startup
            for _ in range(5):
                st.session_state.cap.read()
    else:
        if video_file is not None:
            # Save the uploaded file to a temp location so OpenCV can read it
            temp_path = f"./temp_video.{video_file.name.split('.')[-1]}"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            st.session_state.cap = cv2.VideoCapture(temp_path)

# When the user clicks TERMINATE — stop monitoring.
# We do NOT release cap here. Releasing it from the main thread while the
# fragment's 33ms timer may be mid-read causes a SIGSEGV in OpenCV's macOS
# AVFoundation camera delegate (CaptureDelegate). The fragment releases cap
# safely from its own thread once it detects running=False.
if stop_btn:
    st.session_state.running = False

# ── LIVE DISPLAY FRAGMENT ─────────────────────────────────────────────────────
# Split into two fragments:
#   - monitoring_loop: has run_every timer, only active during monitoring
#   - standby_screen:  no timer, safe for file uploader / widget interactions
# This prevents the "Could not find fragment" error caused by the file uploader
# triggering a full rerun while a run_every timer is active.

@st.fragment
def standby_screen():
    """
    Displays the idle/waiting screen when monitoring is NOT running.
    Shows a centred placeholder with instructions on how to get started.
    This is a Streamlit fragment — it renders independently from the rest of the app.
    """
    col_video, _ = st.columns([3, 2])
    with col_video:
        st.markdown('<div class="panel-label">LIVE DETECTION FEED</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div style="background:#050d1a;border:1px solid rgba(0,245,255,0.12);border-radius:4px;padding:4rem 2rem;text-align:center;font-family:'Share Tech Mono',monospace;">
    <div style="font-size:2.5rem;margin-bottom:1.5rem;color:#00f5ff;text-shadow:0 0 30px rgba(0,245,255,0.5)">⬡</div>
    <div style="color:#00f5ff;letter-spacing:6px;font-size:0.85rem;margin-bottom:1rem;font-family:'Orbitron',sans-serif">AK VISION STANDBY</div>
    <div style="color:rgba(0,245,255,0.35);font-size:0.65rem;letter-spacing:2px;line-height:2.2">
        SELECT INPUT SOURCE &nbsp;·&nbsp; CONFIGURE THRESHOLDS &nbsp;·&nbsp; CLICK INITIATE<br><br>
        <span style="color:#00ff88">■</span> &nbsp;GREEN BOX = ACTIVE PERSONNEL &nbsp;&nbsp;
        <span style="color:#ff9500">■</span> &nbsp;ORANGE BOX = IDLE PERSONNEL &nbsp;&nbsp;
        <span style="color:#ff2d55">■</span> &nbsp;RED ALERT = ABSENT
    </div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=0.033)
def live_display():
    """
    The main live monitoring fragment — runs every ~33ms (≈30fps) while active.
    Left column: Shows the annotated video feed and any absence alert banners.
    Right column: Shows the metrics dashboard, activity timeline chart,
                  personnel status list, and the PDF report download button.

    Uses run_every so Streamlit automatically re-runs this fragment at ~30fps
    without re-running the entire app page (which would be much slower).
    """
    col_video, col_dash = st.columns([3, 2])

    with col_video:
        st.markdown('<div class="panel-label">LIVE DETECTION FEED</div>',
                    unsafe_allow_html=True)

        # If monitoring has stopped or hasn't started yet, show standby screen.
        # IMPORTANT: we also release cap here (not in the stop-button handler)
        # to avoid a race condition — releasing from the main thread while this
        # fragment thread is inside cap.read() causes a SIGSEGV on macOS.
        if not st.session_state.running or st.session_state.monitor is None:
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except Exception:
                    pass
                st.session_state.cap = None
            st.markdown("""
<div style="background:#050d1a;border:1px solid rgba(0,245,255,0.12);border-radius:4px;padding:4rem 2rem;text-align:center;font-family:'Share Tech Mono',monospace;">
    <div style="font-size:2.5rem;margin-bottom:1.5rem;color:#00f5ff;text-shadow:0 0 30px rgba(0,245,255,0.5)">⬡</div>
    <div style="color:#00f5ff;letter-spacing:6px;font-size:0.85rem;margin-bottom:1rem;font-family:'Orbitron',sans-serif">AK VISION STANDBY</div>
    <div style="color:rgba(0,245,255,0.35);font-size:0.65rem;letter-spacing:2px;line-height:2.2">
        SELECT INPUT SOURCE &nbsp;·&nbsp; CONFIGURE THRESHOLDS &nbsp;·&nbsp; CLICK INITIATE<br><br>
        <span style="color:#00ff88">■</span> &nbsp;GREEN BOX = ACTIVE PERSONNEL &nbsp;&nbsp;
        <span style="color:#ff9500">■</span> &nbsp;ORANGE BOX = IDLE PERSONNEL &nbsp;&nbsp;
        <span style="color:#ff2d55">■</span> &nbsp;RED ALERT = ABSENT
    </div>
</div>
""", unsafe_allow_html=True)
            return

        if st.session_state.cap is None:
            return

        cap = st.session_state.cap
        monitor = st.session_state.monitor

        # Check the camera/video is still accessible
        if not cap.isOpened():
            st.error("⚠ FEED UNAVAILABLE — CHECK INPUT SOURCE")
            st.session_state.running = False
            return

        try:
            ret, frame = cap.read()
        except Exception:
            st.error("⚠ FEED ERROR — CHECK INPUT SOURCE")
            st.session_state.running = False
            return

        # For video files, loop back to the start when the file ends
        if not ret:
            if st.session_state.source_type == "Video File":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to frame 0
                ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))

            # Run detection and tracking — returns the annotated frame
            annotated = monitor.update(frame)

            # Convert from BGR (OpenCV default) to RGB (what Streamlit/browsers expect)
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, channels="RGB", use_column_width=True)

            # Build and display flashing red alert banners for absent people
            absent_threshold = st.session_state.absent_threshold
            trackers = monitor.trackers
            alert_html = ""
            for tid, tracker in trackers.items():
                if tracker.status == "absent":
                    secs = tracker.get_absent_duration_seconds()
                    if secs > absent_threshold:
                        # Format absence time as MM:SS
                        m = int(secs // 60)
                        s = int(secs % 60)
                        alert_html += f'<div class="alert-bar">⚠ UNIT {tid:02d} — ABSENT {m:02d}:{s:02d}</div>'
            if alert_html:
                st.markdown(alert_html, unsafe_allow_html=True)

            st.session_state.frame_count += 1

            # Log activity counts every 3 frames (reduces log size while keeping smooth chart)
            if st.session_state.frame_count % 3 == 0:
                active_count = sum(1 for t in trackers.values() if t.status == "active")
                idle_count   = sum(1 for t in trackers.values() if t.status == "idle")
                absent_count = sum(1 for t in trackers.values() if t.status == "absent")
                st.session_state.activity_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "active": active_count, "idle": idle_count, "absent": absent_count
                })
                # Keep only the last 60 log entries to avoid unbounded memory growth
                if len(st.session_state.activity_log) > 60:
                    st.session_state.activity_log = st.session_state.activity_log[-60:]

    with col_dash:
        # Exit early if monitoring has stopped
        if not st.session_state.running or st.session_state.monitor is None:
            return

        monitor = st.session_state.monitor
        trackers = monitor.trackers

        # Count how many people are in each state right now
        active_count = sum(1 for t in trackers.values() if t.status == "active")
        idle_count   = sum(1 for t in trackers.values() if t.status == "idle")
        absent_count = sum(1 for t in trackers.values() if t.status == "absent")
        total_count  = len(trackers)

        # ── Metric cards: Total / Active / Idle / Absent
        st.markdown('<div class="panel-label">NEURAL METRICS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="metric-row">
    <div class="metric-card m-total"><div class="metric-value">{total_count}</div><div class="metric-label">TOTAL</div></div>
    <div class="metric-card m-active"><div class="metric-value">{active_count}</div><div class="metric-label">ACTIVE</div></div>
    <div class="metric-card m-idle"><div class="metric-value">{idle_count}</div><div class="metric-label">IDLE</div></div>
    <div class="metric-card m-absent"><div class="metric-value">{absent_count}</div><div class="metric-label">ABSENT</div></div>
</div>""", unsafe_allow_html=True)

        # ── Activity timeline: a line chart of active/idle/absent counts over time
        st.markdown('<div class="panel-label">ACTIVITY TIMELINE</div>', unsafe_allow_html=True)
        if len(st.session_state.activity_log) > 1:
            df = pd.DataFrame(st.session_state.activity_log)
            fig = go.Figure()
            # One line per status, each with its matching colour and a subtle fill beneath
            fig.add_trace(go.Scatter(x=df["time"], y=df["active"], name="ACTIVE", line=dict(
                color="#00ff88", width=2), fill="tozeroy", fillcolor="rgba(0,255,136,0.06)"))
            fig.add_trace(go.Scatter(x=df["time"], y=df["idle"],   name="IDLE",   line=dict(
                color="#ff9500", width=2), fill="tozeroy", fillcolor="rgba(255,149,0,0.06)"))
            fig.add_trace(go.Scatter(x=df["time"], y=df["absent"], name="ABSENT", line=dict(
                color="#ff2d55", width=2), fill="tozeroy", fillcolor="rgba(255,45,85,0.06)"))
            fig.update_layout(
                height=150, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#00f5ff", family="Share Tech Mono", size=8),
                legend=dict(orientation="h", y=1.15, font=dict(size=7)),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,245,255,0.05)", tickfont=dict(size=7))
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Personnel status list: one row per tracked person with their current stats
        st.markdown('<div class="panel-label">PERSONNEL STATUS</div>', unsafe_allow_html=True)
        people_html = ""
        for tid, tracker in trackers.items():
            summary = tracker.get_summary()
            status  = tracker.status
            css     = f"p-{status}"  # CSS class that sets the left-border colour
            col     = "#00ff88" if status == "active" else "#ff9500" if status == "idle" else "#ff2d55"
            gone    = ""
            if status == "absent":
                # Show how long they've been gone, formatted as MM:SS
                s    = tracker.get_absent_duration_seconds()
                gone = f" · GONE {int(s//60):02d}:{int(s % 60):02d}"
            people_html += f"""<div class="person-row {css}">
<span style="color:var(--cyan);min-width:60px;font-weight:bold">UNIT {tid:02d}</span>
<span style="color:{col}">{status.upper()}</span>
<span style="opacity:0.55;font-size:0.62rem">{summary['active_minutes']}m ACT · {summary['idle_minutes']}m IDLE · {summary['activity_percent']}%{gone}</span>
</div>"""
        st.markdown(people_html, unsafe_allow_html=True)

        # ── PDF report download button
        # Generates the full PDF on every render so it always reflects current stats
        report_data = monitor.get_full_report()
        pdf_bytes = generate_pdf_report(report_data)
        st.download_button(
            label="⬇ EXPORT SESSION REPORT (PDF)",
            key="dl_report",
            data=pdf_bytes,
            file_name=f"akvision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ── FRAGMENT ROUTER ───────────────────────────────────────────────────────────
# Always call live_display() — never switch conditionally between fragments.
# If we only rendered live_display() while running=True, its 33ms timer would
# still fire after stop, looking for a fragment that no longer exists and
# raising "Could not find fragment". By always rendering it, the fragment
# always exists. When running=False it shows the standby screen and returns.
live_display()
