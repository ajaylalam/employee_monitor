# Employee Activity Monitor

Real-time workplace monitoring system using YOLOv8 object detection and tracking.

The system watches a room via webcam or video file and automatically tracks whether employees are present, active, or idle. It raises alerts when someone has been absent too long and generates a full activity report at the end of the session.

---

## What It Does

- Detects every person in the video frame using YOLOv8
- Assigns each person a unique ID and tracks them across frames
- Shows a GREEN box when a person is active and moving
- Shows an ORANGE box when a person is present but not moving (idle)
- Shows a RED alert banner when a person has been absent too long
- Displays a live dashboard with active, idle and absent counts
- Shows a real-time activity timeline graph
- Tracks how long each person has been active, idle and absent
- Generates a downloadable JSON report at the end of the session

---

## Project Files

    employee_monitor/
    |
    |-- app.py            Streamlit web interface - the main app you run
    |-- tracker.py        Core detection and tracking engine
    |-- requirements.txt  All Python packages needed
    |-- README.md         This file

---

## How To Run

### Step 1 - Install packages

    python3.11 -m pip install -r requirements.txt

YOLOv8 model downloads automatically the first time you run the app. It is about 6MB.

### Step 2 - Run the app

    python3.11 -m streamlit run app.py

### Step 3 - Open browser

Go to http://localhost:8501

### Step 4 - Use the app

1. Select Webcam or upload a Video File
2. Adjust alert thresholds if needed (default: absent alert after 60 seconds)
3. Click Start Monitoring
4. Watch the live detection feed and dashboard
5. Click Stop Monitoring when done
6. Download the session report

---

## What The Boxes Mean

- GREEN box = person is active and moving
- ORANGE box = person is present but has not moved recently (idle)
- RED alert banner = person has left the frame and been absent too long

---

## Tech Stack

| Tool | What it does |
|---|---|
| YOLOv8 (Ultralytics) | Detects people in each video frame in real time |
| OpenCV | Reads video frames from webcam or file, draws boxes |
| Custom Tracker | Assigns IDs and tracks each person across frames |
| Scipy | Calculates distances to match detections to existing people |
| Streamlit | Web interface |
| Plotly | Live activity timeline charts |
| Pandas | Data processing for charts |

---

## Key Concepts

### YOLOv8
You Only Look Once version 8. The fastest and most accurate real-time object detection model available. It processes the whole frame in one pass and detects all people simultaneously. We use the nano version (yolov8n) which is the smallest and fastest variant.

### Object Tracking
Detection alone just finds people in each frame independently. Tracking connects detections across frames so the same person keeps the same ID. We use distance-based matching - if a new detection is close to a known person's last position, we assume it is the same person.

### Activity Detection
We track whether each person is moving by comparing their current position to their position from 10 frames ago. If they have moved more than 15 pixels they are ACTIVE. If they have not moved for 30 seconds they are IDLE.

### Absence Detection
If a person disappears from the frame entirely, we start an absence timer. If they are gone for more than 60 seconds (configurable), a red alert appears.

---

## Troubleshooting

If webcam does not open - make sure no other app is using it, then try again

If detection is slow - this is normal on CPU. On a GPU it runs at 30+ fps. On CPU expect 5-10 fps.

If you get a model download error - make sure you have internet connection for the first run
