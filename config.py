"""
=============================================================================
  config.py — Central Configuration for Drowsiness Detection System
=============================================================================
  All tunable parameters, file paths, and constants live here.
  No other module should contain hardcoded paths or magic numbers.
=============================================================================
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Root directory of the project (directory containing this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trained YOLO classification model weights
MODEL_PATH = os.path.join(BASE_DIR, "Model", "best.pt")

# Alarm sound file
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "sounds", "alarm.wav")

# Directory to save snapshot images when drowsiness is detected
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Directory for application log files
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Directory for static assets (icons, fonts, overlays)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ──────────────────────────────────────────────────────────────────────────────
# Camera Settings
# ──────────────────────────────────────────────────────────────────────────────

# Webcam device index (0 = default camera, 1 = external USB camera, etc.)
CAMERA_INDEX = 0

# Target resolution for webcam capture (width, height)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Target FPS for webcam capture
CAMERA_FPS = 30

# ──────────────────────────────────────────────────────────────────────────────
# Classification Settings
# ──────────────────────────────────────────────────────────────────────────────

# Confidence threshold — predictions below this are treated as uncertain
CONFIDENCE_THRESHOLD = 0.60

# Class name that indicates the driver is drowsy
# Must match exactly what the model was trained with
DROWSY_CLASS_NAME = "Drowsy"

# Input image size expected by the YOLO classifier (pixels)
# YOLO11m-cls default is 224x224
MODEL_INPUT_SIZE = 224

# ──────────────────────────────────────────────────────────────────────────────
# Alert / Alarm Settings
# ──────────────────────────────────────────────────────────────────────────────

# Number of consecutive "Drowsy" frames required before triggering the alarm.
# This prevents false positives from single-frame misclassifications.
# At 30 FPS, a value of 15 means ~0.5 seconds of sustained drowsiness.
DROWSY_FRAMES_THRESHOLD = 15

# Cooldown period (in seconds) after an alarm is triggered.
# Prevents the alarm from firing repeatedly in quick succession.
ALARM_COOLDOWN_SECONDS = 5.0

# Volume level for the alarm (0.0 = silent, 1.0 = max)
ALARM_VOLUME = 0.8

# ──────────────────────────────────────────────────────────────────────────────
# Logging Settings
# ──────────────────────────────────────────────────────────────────────────────

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Log file name (timestamped log files are created per session)
LOG_FILE_PREFIX = "drowsiness_session"

# Whether to also save snapshot images when drowsiness is detected
SAVE_SNAPSHOTS = True

# ──────────────────────────────────────────────────────────────────────────────
# Display / UI Settings
# ──────────────────────────────────────────────────────────────────────────────

# Window title for the OpenCV display
WINDOW_TITLE = "Drowsiness Detection System"

# Whether to show the confidence bar overlay on the video feed
SHOW_CONFIDENCE_OVERLAY = True

# Whether to show FPS counter on the video feed
SHOW_FPS = True

# Font scale for overlay text
OVERLAY_FONT_SCALE = 0.65

# Colors (BGR format)
COLOR_DROWSY = (0, 0, 255)       # Red — danger
COLOR_ALERT = (0, 200, 0)        # Green — safe / awake
COLOR_WARNING = (0, 165, 255)    # Orange — uncertain
COLOR_TEXT = (255, 255, 255)     # White — general text
COLOR_PANEL_BG = (30, 30, 30)   # Dark panel background
