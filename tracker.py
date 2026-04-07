"""
tracker.py - Core detection and tracking engine
This is the brain of the system.
It uses YOLOv8 to detect people in each video frame,
assigns each person a unique ID, and tracks their activity over time.

It records:
- Whether each person is present or absent
- How long they have been active vs idle
- When they left and came back
- Activity percentage over time
"""

# time for tracking how long people have been in certain states
import time

# datetime for readable timestamps
from datetime import datetime

# numpy for array operations on video frames
import numpy as np

# OpenCV for drawing boxes and text on video frames
import cv2

# collections for storing activity history efficiently
from collections import defaultdict, deque

# ultralytics provides YOLOv8 - the object detection model
# YOLOv8 is the latest version of the YOLO (You Only Look Once) model
# It can detect 80 different object types including people, in real time
from ultralytics import YOLO

# scipy for calculating distance between positions
from scipy.spatial.distance import cdist


# ── SETTINGS ──────────────────────────────────────────────────────────────────

# How many seconds of no movement before marking someone as IDLE
IDLE_THRESHOLD_SECONDS = 30

# How many seconds of absence before triggering an ABSENT alert
ABSENT_THRESHOLD_SECONDS = 60

# How much a person needs to move (in pixels) to be considered ACTIVE
# If they move less than this they are considered idle
MOVEMENT_THRESHOLD_PIXELS = 15

# How many past positions to store for each person (for movement detection)
POSITION_HISTORY_LENGTH = 10

# Colors for drawing boxes (BGR format - Blue Green Red)
COLOR_ACTIVE  = (0, 255, 0)    # green  = active and present
COLOR_IDLE    = (0, 165, 255)  # orange = present but not moving
COLOR_ABSENT  = (0, 0, 255)    # red    = not in frame / absent
COLOR_TEXT    = (255, 255, 255) # white  = text color
COLOR_ALERT   = (0, 0, 200)    # dark red for alert background


class PersonTracker:
    """
    Tracks a single detected person over time.
    Each person gets their own PersonTracker object.
    It stores their history and calculates their current status.
    """

    def __init__(self, person_id: int, initial_position: tuple):
        # Unique ID for this person (assigned by the tracking system)
        self.person_id = person_id

        # Current status: "active", "idle", or "absent"
        self.status = "active"

        # Store the last N positions to detect movement
        self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)
        self.position_history.append(initial_position)

        # Timestamps for time tracking
        self.first_seen_time   = time.time()   # when this person was first detected
        self.last_seen_time    = time.time()   # last time they appeared in frame
        self.last_active_time  = time.time()   # last time they were moving
        self.absent_since      = None          # when they disappeared from frame

        # Accumulated time counters (in seconds)
        self.total_active_seconds = 0
        self.total_idle_seconds   = 0
        self.total_absent_seconds = 0

        # History of status changes for the activity graph
        # Each entry is (timestamp, status)
        self.status_history = [(time.time(), "active")]

        # Alert flag - True if person has been absent too long
        self.alert_triggered = False

    def update_position(self, new_position: tuple):
        """
        Called every frame when this person is detected.
        Updates their position and recalculates their status.
        """
        current_time = time.time()

        # Add new position to history
        self.position_history.append(new_position)

        # Calculate how much they have moved recently
        # Compare current position to position from N frames ago
        if len(self.position_history) >= 2:
            old_position = self.position_history[0]
            movement = np.sqrt(
                (new_position[0] - old_position[0])**2 +
                (new_position[1] - old_position[1])**2
            )
        else:
            movement = 0

        # Update time accumulator based on previous status
        time_delta = current_time - self.last_seen_time
        if self.status == "active":
            self.total_active_seconds += time_delta
        elif self.status == "idle":
            self.total_idle_seconds += time_delta

        # Update last seen time
        self.last_seen_time = current_time

        # If they were absent before, record how long they were gone
        if self.absent_since is not None:
            absent_duration = current_time - self.absent_since
            self.total_absent_seconds += absent_duration
            self.absent_since = None
            self.alert_triggered = False

        # Determine new status based on movement
        if movement > MOVEMENT_THRESHOLD_PIXELS:
            # Person is moving - they are ACTIVE
            self.last_active_time = current_time
            new_status = "active"
        elif (current_time - self.last_active_time) > IDLE_THRESHOLD_SECONDS:
            # Person has not moved for a while - they are IDLE
            new_status = "idle"
        else:
            # Person is present but movement is small - still active
            new_status = "active"

        # Record status change if it changed
        if new_status != self.status:
            self.status_history.append((current_time, new_status))
        self.status = new_status

    def mark_absent(self):
        """
        Called when this person is NOT detected in the current frame.
        Marks them as absent and starts the absence timer.
        """
        current_time = time.time()

        # Only start the absent timer if they were not already absent
        if self.absent_since is None:
            self.absent_since = current_time
            self.status = "absent"
            self.status_history.append((current_time, "absent"))

        # Check if they have been absent long enough to trigger an alert
        absent_duration = current_time - self.absent_since
        if absent_duration > ABSENT_THRESHOLD_SECONDS:
            self.alert_triggered = True

    def get_absent_duration_seconds(self) -> float:
        """Returns how many seconds this person has currently been absent."""
        if self.absent_since is None:
            return 0
        return time.time() - self.absent_since

    def get_activity_percentage(self) -> float:
        """
        Returns what percentage of total tracked time the person was active.
        Used for the activity bar in the dashboard.
        """
        total = self.total_active_seconds + self.total_idle_seconds + self.total_absent_seconds
        if total == 0:
            return 0
        return round((self.total_active_seconds / total) * 100, 1)

    def get_summary(self) -> dict:
        """Returns a summary of this person's activity for the report."""
        return {
            "id":               self.person_id,
            "status":           self.status,
            "active_minutes":   round(self.total_active_seconds / 60, 1),
            "idle_minutes":     round(self.total_idle_seconds / 60, 1),
            "absent_minutes":   round(self.total_absent_seconds / 60, 1),
            "activity_percent": self.get_activity_percentage(),
            "alert":            self.alert_triggered
        }


class EmployeeMonitor:
    """
    The main monitoring system.
    Uses YOLOv8 to detect people in each frame,
    assigns IDs to each person, and tracks them using PersonTracker objects.
    """

    def __init__(self):
        # Load YOLOv8 model
        # yolov8n.pt = nano version - smallest and fastest (good for real time)
        # It downloads automatically the first time you run this
        print("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        print("Model loaded!")

        # Dictionary of active trackers: {person_id: PersonTracker}
        self.trackers = {}

        # Counter for assigning unique IDs to new people
        self.next_id = 1

        # Maximum distance (pixels) to match a detection to an existing tracker
        # If a new detection is within this distance of a known person,
        # we assume it is the same person
        self.max_match_distance = 80

        # Session start time for the daily report
        self.session_start = time.time()

        # Store recent activity data for graphs (last 60 seconds)
        self.activity_timeline = deque(maxlen=600)  # 10 data points per second for 60s

    def detect_people(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8 on a frame and return a list of detected person positions.
        Each position is the center point (x, y) of the person's bounding box.
        Also returns the bounding box coordinates for drawing.
        """
        # Run detection
        # classes=[0] means only detect class 0 which is "person" in YOLO
        # conf=0.5 means only return detections with 50% or higher confidence
        results = self.model(frame, classes=[0], conf=0.5, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate center point of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get confidence score
                confidence = float(box.conf[0])

                detections.append({
                    "center":     (center_x, center_y),
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": confidence
                })

        return detections

    def match_detections_to_trackers(self, detections: list) -> dict:
        """
        Match each new detection to an existing tracker (or create a new one).
        Uses distance-based matching - the nearest existing tracker gets the detection.
        This is a simple version of the Hungarian algorithm used in real tracking systems.
        """
        if not detections:
            return {}

        matched = {}  # {detection_index: tracker_id}

        if self.trackers:
            # Get current positions of all existing trackers
            tracker_ids       = list(self.trackers.keys())
            tracker_positions = [
                list(self.trackers[tid].position_history)[-1]
                for tid in tracker_ids
            ]

            # Get positions of all new detections
            detection_positions = [d["center"] for d in detections]

            # Calculate distance between every detection and every tracker
            distances = cdist(detection_positions, tracker_positions)

            # Match each detection to the nearest tracker
            for det_idx in range(len(detections)):
                nearest_tracker_idx = np.argmin(distances[det_idx])
                min_distance = distances[det_idx][nearest_tracker_idx]

                # Only match if they are close enough
                if min_distance < self.max_match_distance:
                    matched[det_idx] = tracker_ids[nearest_tracker_idx]

        return matched

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Main update function - called on every video frame.
        1. Detect people in the frame
        2. Match detections to existing trackers
        3. Update matched trackers
        4. Mark unmatched trackers as absent
        5. Create new trackers for unmatched detections
        6. Draw everything on the frame
        7. Return the annotated frame
        """
        # Step 1: Detect people
        detections = self.detect_people(frame)

        # Step 2: Match detections to existing trackers
        matched = self.match_detections_to_trackers(detections)

        # Step 3: Update matched trackers
        matched_tracker_ids = set(matched.values())
        for det_idx, tracker_id in matched.items():
            self.trackers[tracker_id].update_position(detections[det_idx]["center"])

        # Step 4: Mark unmatched trackers as absent
        for tracker_id in list(self.trackers.keys()):
            if tracker_id not in matched_tracker_ids:
                self.trackers[tracker_id].mark_absent()

        # Step 5: Create new trackers for unmatched detections
        matched_detection_indices = set(matched.keys())
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detection_indices:
                new_id = self.next_id
                self.next_id += 1
                self.trackers[new_id] = PersonTracker(new_id, detection["center"])

        # Step 6 & 7: Draw annotations and return frame
        annotated_frame = self.draw_annotations(frame, detections, matched)

        # Record activity snapshot for timeline
        active_count = sum(1 for t in self.trackers.values() if t.status == "active")
        self.activity_timeline.append((time.time(), active_count))

        return annotated_frame

    def draw_annotations(self, frame: np.ndarray, detections: list, matched: dict) -> np.ndarray:
        """
        Draw all visual elements on the video frame:
        - Colored bounding boxes around each person
        - Status label (ACTIVE / IDLE / ABSENT)
        - Person ID
        - Absent duration timer
        - Stats overlay in the corner
        """
        frame = frame.copy()

        # Draw boxes for currently detected people
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            center = detection["center"]

            # Get the tracker for this detection
            if det_idx in matched:
                tracker_id = matched[det_idx]
                tracker = self.trackers[tracker_id]
                status = tracker.status
                activity_pct = tracker.get_activity_percentage()
                label = f"Person {tracker_id} | {status.upper()} | {activity_pct}% active"
            else:
                status = "active"
                label = f"Person NEW"

            # Choose color based on status
            color = COLOR_ACTIVE if status == "active" else COLOR_IDLE

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

        # Draw absent alerts for people not currently in frame
        alert_y = 100
        for tracker_id, tracker in self.trackers.items():
            if tracker.status == "absent":
                absent_secs = tracker.get_absent_duration_seconds()
                absent_mins = int(absent_secs // 60)
                absent_secs_rem = int(absent_secs % 60)

                alert_text = f"⚠ Person {tracker_id} ABSENT: {absent_mins:02d}:{absent_secs_rem:02d}"

                # Draw red alert banner
                cv2.rectangle(frame, (10, alert_y - 20), (400, alert_y + 5), COLOR_ABSENT, -1)
                cv2.putText(frame, alert_text, (15, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
                alert_y += 35

        # Draw stats overlay in top left corner
        self.draw_stats_overlay(frame)

        return frame

    def draw_stats_overlay(self, frame: np.ndarray):
        """Draw a stats panel in the top left corner of the frame."""
        active_count  = sum(1 for t in self.trackers.values() if t.status == "active")
        idle_count    = sum(1 for t in self.trackers.values() if t.status == "idle")
        absent_count  = sum(1 for t in self.trackers.values() if t.status == "absent")
        total_tracked = len(self.trackers)

        # Calculate session duration
        session_secs = int(time.time() - self.session_start)
        session_mins = session_secs // 60
        session_secs_rem = session_secs % 60

        # Draw semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw stats text
        stats = [
            f"SESSION: {session_mins:02d}:{session_secs_rem:02d}",
            f"TOTAL TRACKED: {total_tracked}",
            f"ACTIVE:  {active_count}",
            f"IDLE:    {idle_count}",
            f"ABSENT:  {absent_count}",
            datetime.now().strftime("%H:%M:%S")
        ]

        colors_map = [COLOR_TEXT, COLOR_TEXT, COLOR_ACTIVE, COLOR_IDLE, COLOR_ABSENT, COLOR_TEXT]

        for i, (stat, color) in enumerate(zip(stats, colors_map)):
            cv2.putText(frame, stat, (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    def get_full_report(self) -> dict:
        """
        Generate a full activity report for all tracked people.
        This is saved to a JSON file at the end of the session.
        """
        report = {
            "session_start":    datetime.fromtimestamp(self.session_start).strftime("%Y-%m-%d %H:%M:%S"),
            "session_end":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_people":     len(self.trackers),
            "people":           [t.get_summary() for t in self.trackers.values()]
        }
        return report
