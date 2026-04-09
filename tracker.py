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

import time
from datetime import datetime
import numpy as np
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ── SETTINGS ──────────────────────────────────────────────────────────────────

# How many seconds of no movement before a person is considered "idle"
IDLE_THRESHOLD_SECONDS = 30

# How many seconds of being undetected before a person is considered "absent"
ABSENT_THRESHOLD_SECONDS = 60

# Minimum pixel distance a person must move to count as "active"
MOVEMENT_THRESHOLD_PIXELS = 15

# How many past positions to remember for each person (used to measure movement)
POSITION_HISTORY_LENGTH = 10

# If a person disappears from detection for this many frames, mark them absent
MAX_MISSED_FRAMES = 3

# Ignore detections smaller than this area (filters out tiny false positives)
MIN_DETECTION_AREA = 2500

# Bounding box colors for each status (BGR format used by OpenCV)
COLOR_ACTIVE = (0, 255, 0)    # Green  = person is moving
COLOR_IDLE = (0, 165, 255)    # Orange = person is present but still
COLOR_ABSENT = (0, 0, 255)    # Red    = person has left the frame
COLOR_TEXT = (255, 255, 255)  # White  = label text
COLOR_ALERT = (0, 0, 200)     # Dark red = absence alert banner


# ── PERSON TRACKER ────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Tracks a single person throughout the session.
    Each detected person gets their own PersonTracker instance
    which remembers their position, status, and time spent in each state.
    """

    def __init__(self, person_id: int, initial_position: tuple):
        """
        Set up a new tracker for a person seen for the first time.
        - person_id: unique number assigned to this person (1, 2, 3, ...)
        - initial_position: (x, y) pixel coordinate of their center in the frame
        """
        self.person_id = person_id
        self.status = "active"  # Everyone starts as active when first detected

        # A sliding window of recent positions — used to measure how much they moved
        self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)
        self.position_history.append(initial_position)

        # Timestamps to track how long they've been seen / active
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.last_active_time = time.time()  # Last time meaningful movement was detected

        # Set when the person goes absent; cleared when they return
        self.absent_since = None

        # Counts how many frames in a row the person was NOT detected
        self.missed_frames = 0

        # Accumulators — running totals of time spent in each state (in seconds)
        self.total_active_seconds = 0
        self.total_idle_seconds = 0
        self.total_absent_seconds = 0

        # A log of every status change with timestamps (for audit trail)
        self.status_history = [(time.time(), "active")]

        # Becomes True if the person has been absent longer than the threshold
        self.alert_triggered = False

    def update_position(self, new_position: tuple):
        """
        Called every frame when this person IS detected in the frame.
        Updates their position, calculates movement, and decides if they
        are still active (moving) or have become idle (standing still).
        """
        current_time = time.time()
        self.position_history.append(new_position)
        self.missed_frames = 0  # Reset because they were just seen

        # Measure how far they moved since the oldest stored position
        if len(self.position_history) >= 2:
            old_position = self.position_history[0]
            # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
            movement = np.sqrt(
                (new_position[0] - old_position[0])**2 +
                (new_position[1] - old_position[1])**2
            )
        else:
            movement = 0  # Not enough history yet to measure movement

        # How many seconds have passed since the last update
        time_delta = current_time - self.last_seen_time

        # Add elapsed time to whichever state bucket they were just in
        if self.status == "active":
            self.total_active_seconds += time_delta
        elif self.status == "idle":
            self.total_idle_seconds += time_delta

        self.last_seen_time = current_time

        # If they were absent and just came back, tally up the absence duration
        if self.absent_since is not None:
            absent_duration = current_time - self.absent_since
            self.total_absent_seconds += absent_duration
            self.absent_since = None      # Clear the absence clock
            self.alert_triggered = False  # Reset the alert since they returned

        # Decide new status based on movement
        if movement > MOVEMENT_THRESHOLD_PIXELS:
            # They moved enough — count as active and reset the idle clock
            self.last_active_time = current_time
            new_status = "active"
        elif (current_time - self.last_active_time) > IDLE_THRESHOLD_SECONDS:
            # Haven't moved significantly in a while — mark idle
            new_status = "idle"
        else:
            # Small movement but not yet idle — still considered active
            new_status = "active"

        # Only log a status change if the status actually changed
        if new_status != self.status:
            self.status_history.append((current_time, new_status))
        self.status = new_status

    def mark_absent(self):
        """
        Called when this person is NOT detected in the current frame
        (and has exceeded the missed-frames limit).
        Starts or continues timing how long they've been gone,
        and triggers an alert if they've been absent too long.
        """
        current_time = time.time()

        # Only start the absence clock once (the first time they go absent)
        if self.absent_since is None:
            self.absent_since = current_time
            self.status = "absent"
            self.status_history.append((current_time, "absent"))

        # Check if they've been gone long enough to trigger an alert
        absent_duration = current_time - self.absent_since
        if absent_duration > ABSENT_THRESHOLD_SECONDS:
            self.alert_triggered = True

    def increment_missed(self):
        """
        Called each frame this person is NOT detected (but not yet confirmed absent).
        After MAX_MISSED_FRAMES consecutive misses, officially marks them absent.
        This prevents flickering — a single missed frame won't immediately mark someone absent.
        """
        self.missed_frames += 1
        if self.missed_frames >= MAX_MISSED_FRAMES:
            self.mark_absent()

    def get_absent_duration_seconds(self) -> float:
        """
        Returns how many seconds this person has been continuously absent.
        Returns 0 if they are currently present.
        """
        if self.absent_since is None:
            return 0
        return time.time() - self.absent_since

    def get_activity_percentage(self) -> float:
        """
        Calculates what percentage of the session this person was actively moving.
        Formula: active_time / (active + idle + absent) * 100
        Returns 0 if the person has just appeared (no tracked time yet).
        """
        total = self.total_active_seconds + \
            self.total_idle_seconds + self.total_absent_seconds
        if total == 0:
            return 0
        return round((self.total_active_seconds / total) * 100, 1)

    def get_summary(self) -> dict:
        """
        Returns a snapshot of this person's stats as a dictionary.
        Used when generating reports and updating the dashboard.
        Times are converted from seconds to minutes for readability.
        """
        return {
            "id":               self.person_id,
            "status":           self.status,
            "active_minutes":   round(self.total_active_seconds / 60, 1),
            "idle_minutes":     round(self.total_idle_seconds / 60, 1),
            "absent_minutes":   round(self.total_absent_seconds / 60, 1),
            "activity_percent": self.get_activity_percentage(),
            "alert":            self.alert_triggered
        }


# ── EMPLOYEE MONITOR ──────────────────────────────────────────────────────────

class EmployeeMonitor:
    """
    The top-level manager that processes each video frame end-to-end.
    It owns the YOLO model, manages all PersonTracker instances,
    and produces annotated frames for the live feed.
    """

    def __init__(self):
        """
        Load the YOLOv8 model and prepare the internal state.
        This is called once when monitoring starts.
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")  # Load the nano (lightweight) YOLOv8 model
        self.model.to("mps")             # Use Apple Silicon GPU for faster inference
        print("Model loaded!")

        self.trackers = {}          # Dict of {person_id: PersonTracker} for everyone seen
        self.next_id = 1            # Auto-incrementing ID counter for new people
        self.max_match_distance = 80  # Max pixels a detection can move and still match the same tracker
        self.session_start = time.time()  # Record when this session began

        # Stores (timestamp, active_count) snapshots — last 600 samples (≈10 minutes at 1fps)
        self.activity_timeline = deque(maxlen=600)

    def detect_people(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8 on a single video frame and return a list of detected people.
        Filters out detections that are too small (likely false positives).
        Each detection is a dict with: center (x,y), bounding box, and confidence score.
        """
        # Run inference — classes=[0] means "person only", conf=0.5 means 50% confidence minimum
        results = self.model(frame, classes=[0], conf=0.5, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                # Get the corner coordinates of the bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Skip detections that are too small (noise / background objects)
                area = max(0, (x2 - x1) * (y2 - y1))
                if area < MIN_DETECTION_AREA:
                    continue

                # Calculate the center point of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                confidence = float(box.conf[0])
                detections.append({
                    "center":     (center_x, center_y),
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": confidence
                })
        return detections

    def match_detections_to_trackers(self, detections: list) -> dict:
        """
        Figures out which detected person in this frame corresponds to which
        known tracker from the previous frame.

        Uses the Hungarian algorithm — an optimisation method that finds the
        lowest-cost assignment between detections and existing trackers,
        based on how far apart their positions are.

        Returns a dict of {detection_index: tracker_id} for matched pairs.
        """
        # If there's nothing to match, return empty
        if not detections or not self.trackers:
            return {}

        tracked_ids = list(self.trackers.keys())

        # Get the most recent known position of each existing tracker
        tracker_positions = [
            list(self.trackers[tid].position_history)[-1]
            for tid in tracked_ids
        ]

        detection_positions = [d["center"] for d in detections]

        # Build a distance matrix: rows = detections, cols = trackers
        # Each cell = pixel distance between that detection and that tracker
        distances = cdist(detection_positions, tracker_positions)

        # Hungarian algorithm finds the globally optimal one-to-one matching
        matched = {}
        rows, cols = linear_sum_assignment(distances)
        for det_idx, tracker_idx in zip(rows, cols):
            # Only accept the match if the two positions are close enough
            if distances[det_idx, tracker_idx] < self.max_match_distance:
                matched[det_idx] = tracked_ids[tracker_idx]
        return matched

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        The main function called on every video frame.
        Steps:
          1. Detect all people in the frame using YOLO
          2. Match each detection to an existing tracker (or create a new one)
          3. Update matched trackers with new positions
          4. Mark unmatched trackers as absent (person left the frame)
          5. Create new trackers for unmatched detections (new person appeared)
          6. Draw all annotations on the frame and return it
        """
        # Step 1: Detect people in this frame
        detections = self.detect_people(frame)

        # Step 2: Match detections to known trackers
        matched = self.match_detections_to_trackers(detections)
        matched_tracker_ids = set(matched.values())

        # Step 3: Update trackers that were successfully matched
        for det_idx, tracker_id in matched.items():
            self.trackers[tracker_id].update_position(
                detections[det_idx]["center"])

        # Step 4: For trackers that had no matching detection, increment missed count
        for tracker_id in list(self.trackers.keys()):
            if tracker_id not in matched_tracker_ids:
                self.trackers[tracker_id].mark_absent()

        # Step 5: Create a brand new tracker for any detection with no match
        matched_detection_indices = set(matched.keys())
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detection_indices:
                new_id = self.next_id
                self.next_id += 1
                self.trackers[new_id] = PersonTracker(
                    new_id, detection["center"])

        # Step 6: Draw boxes, labels, and stats overlay onto the frame
        annotated_frame = self.draw_annotations(frame, detections, matched)

        # Log the current active count for the activity timeline chart
        active_count = sum(1 for t in self.trackers.values()
                           if t.status == "active")
        self.activity_timeline.append((time.time(), active_count))

        return annotated_frame

    def draw_annotations(self, frame: np.ndarray, detections: list, matched: dict) -> np.ndarray:
        """
        Draws bounding boxes and status labels on top of the video frame
        for every detected person. Also draws absence alert banners
        for anyone who has left the frame.
        Returns the annotated frame (does not modify the original).
        """
        frame = frame.copy()  # Work on a copy so the original isn't modified

        # Draw a box and label for each detected person
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]

            if det_idx in matched:
                # Known person — show their ID, status, and activity %
                tracker_id = matched[det_idx]
                tracker = self.trackers[tracker_id]
                status = tracker.status
                activity_pct = tracker.get_activity_percentage()
                label = f"Person {tracker_id} | {status.upper()} | {activity_pct}% active"
            else:
                # New person just appeared — no stats yet
                status = "active"
                label = "Person NEW"

            # Green for active, orange for idle
            color = COLOR_ACTIVE if status == "active" else COLOR_IDLE

            # Draw the bounding box rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw a filled background rectangle behind the label text for readability
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20),
                          (x1 + label_size[0], y1), color, -1)

            # Write the label text on top of the filled background
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

        # Draw red absence alert banners at the top of the frame for absent people
        alert_y = 100  # Starting Y position for the first alert banner
        for tracker_id, tracker in self.trackers.items():
            if tracker.status == "absent":
                absent_secs = tracker.get_absent_duration_seconds()
                absent_mins = int(absent_secs // 60)
                absent_secs_rem = int(absent_secs % 60)
                alert_text = f"Person {tracker_id} ABSENT: {absent_mins:02d}:{absent_secs_rem:02d}"

                # Draw a solid red background bar for the alert
                cv2.rectangle(frame, (10, alert_y - 20),
                              (400, alert_y + 5), COLOR_ABSENT, -1)
                cv2.putText(frame, alert_text, (15, alert_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
                alert_y += 35  # Move the next alert banner lower down

        # Draw the semi-transparent stats box in the top-left corner
        self.draw_stats_overlay(frame)
        return frame

    def draw_stats_overlay(self, frame: np.ndarray):
        """
        Draws a small semi-transparent stats panel in the top-left corner
        of the video frame showing live counts and session duration.
        Modifies the frame in place (no return value needed).
        """
        # Count how many people are in each state right now
        active_count = sum(1 for t in self.trackers.values()
                           if t.status == "active")
        idle_count = sum(1 for t in self.trackers.values()
                         if t.status == "idle")
        absent_count = sum(1 for t in self.trackers.values()
                           if t.status == "absent")
        total_tracked = len(self.trackers)

        # Format session elapsed time as MM:SS
        session_secs = int(time.time() - self.session_start)
        session_mins = session_secs // 60
        session_secs_rem = session_secs % 60

        # Draw a dark translucent rectangle as the background for the stats box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 140), (0, 0, 0), -1)
        # Blend overlay (60%) with the original frame (40%) for transparency effect
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # List of text lines and their corresponding display colors
        stats = [
            f"SESSION: {session_mins:02d}:{session_secs_rem:02d}",
            f"TOTAL TRACKED: {total_tracked}",
            f"ACTIVE:  {active_count}",
            f"IDLE:    {idle_count}",
            f"ABSENT:  {absent_count}",
            datetime.now().strftime("%H:%M:%S")  # Current wall-clock time
        ]
        colors_map = [COLOR_TEXT, COLOR_TEXT, COLOR_ACTIVE,
                      COLOR_IDLE, COLOR_ABSENT, COLOR_TEXT]

        # Draw each stat line spaced 20px apart
        for i, (stat, color) in enumerate(zip(stats, colors_map)):
            cv2.putText(frame, stat, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    def get_full_report(self) -> dict:
        """
        Builds and returns a complete session report as a dictionary.
        This is called by app.py when generating the PDF download.
        Includes session start/end times and a summary for every tracked person.
        """
        return {
            "session_start": datetime.fromtimestamp(self.session_start).strftime("%Y-%m-%d %H:%M:%S"),
            "session_end":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_people":  len(self.trackers),
            "people":        [t.get_summary() for t in self.trackers.values()]
        }
