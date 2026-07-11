"""
=============================================================================
  app.py — Main Application Entry Point
=============================================================================
  Orchestrates the Drowsiness Detection System:
    1. Initializes all components (camera, classifier, alert manager)
    2. Runs the real-time frame loop
    3. Classifies each frame and updates alert state
    4. Renders the HUD overlay (classification result, confidence bars, FPS)
    5. Handles keyboard controls and graceful shutdown

  Controls:
    q / ESC   — Quit the application
    s         — Save a snapshot of the current frame
    r         — Reset the alert state

  Usage:
    python app.py
=============================================================================
"""

import sys
import logging

import cv2
import numpy as np

import config
from utils import ensure_directories, setup_logger, get_timestamp, FPSCounter
from camera import Camera
from classifier import DrowsinessClassifier, ClassificationResult
from alert import AlertManager

logger: logging.Logger  # initialized in main()


# ──────────────────────────────────────────────────────────────────────────────
# HUD (Heads-Up Display) Rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    result: ClassificationResult,
    alert: AlertManager,
    fps: float,
) -> np.ndarray:
    """
    Render the heads-up display overlay on the video frame.

    Includes:
      - Status banner (DROWSY / AWAKE) with color coding
      - Confidence bars for all classes
      - Consecutive drowsy frame counter & progress bar
      - FPS counter
      - Timestamp
      - Alarm warning flash (when alarm is active)

    Args:
        frame:  The raw BGR video frame.
        result: Classification result from the current frame.
        alert:  The AlertManager for reading alarm state.
        fps:    Current frames-per-second.

    Returns:
        Annotated frame (copy — original is not modified).
    """
    display = frame.copy()
    h, w = display.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = config.OVERLAY_FONT_SCALE

    # ── Determine status color ──
    if alert.is_alarm_active:
        status_color = config.COLOR_DROWSY
        status_text = "⚠ DROWSY — WAKE UP!"
    elif result.is_drowsy:
        status_color = config.COLOR_WARNING
        status_text = f"DROWSY ({result.confidence:.0%})"
    else:
        status_color = config.COLOR_ALERT
        status_text = f"AWAKE ({result.confidence:.0%})"

    # ── Alarm flash border ──
    if alert.is_alarm_active:
        # Pulsing red border effect
        import time
        pulse = int(abs(math.sin(time.time() * 6)) * 12) + 4
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), config.COLOR_DROWSY, pulse)

    # ── Top status banner ──
    banner_h = 42
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    cv2.putText(
        display, status_text,
        (12, 30), font, 0.8, status_color, 2, cv2.LINE_AA,
    )

    # ── FPS counter (top right) ──
    if config.SHOW_FPS:
        fps_text = f"FPS: {fps:.0f}"
        (tw, _), _ = cv2.getTextSize(fps_text, font, fs, 1)
        cv2.putText(
            display, fps_text,
            (w - tw - 12, 28), font, fs, config.COLOR_TEXT, 1, cv2.LINE_AA,
        )

    # ── Confidence bars panel (bottom left) ──
    if config.SHOW_CONFIDENCE_OVERLAY:
        panel_x = 10
        panel_y = h - 20 - (len(result.all_probabilities) * 35)
        bar_width = min(250, w - 30)

        # Semi-transparent background
        panel_h = len(result.all_probabilities) * 35 + 15
        overlay2 = display.copy()
        cv2.rectangle(
            overlay2,
            (panel_x - 5, panel_y - 5),
            (panel_x + bar_width + 60, panel_y + panel_h),
            config.COLOR_PANEL_BG, -1,
        )
        cv2.addWeighted(overlay2, 0.65, display, 0.35, 0, display)

        for class_name, prob in result.all_probabilities.items():
            # Choose color based on class
            if class_name == config.DROWSY_CLASS_NAME:
                bar_color = config.COLOR_DROWSY
            else:
                bar_color = config.COLOR_ALERT

            # Label
            label = f"{class_name}: {prob:.0%}"
            cv2.putText(
                display, label,
                (panel_x, panel_y + 12),
                font, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA,
            )

            # Bar background
            bar_y = panel_y + 17
            cv2.rectangle(
                display,
                (panel_x, bar_y),
                (panel_x + bar_width, bar_y + 12),
                (60, 60, 60), -1,
            )

            # Bar fill
            fill_w = int(bar_width * prob)
            if fill_w > 0:
                cv2.rectangle(
                    display,
                    (panel_x, bar_y),
                    (panel_x + fill_w, bar_y + 12),
                    bar_color, -1,
                )

            panel_y += 35

    # ── Drowsy frame progress bar (bottom right) ──
    progress = alert.drowsy_progress
    prog_w = 150
    prog_h = 14
    prog_x = w - prog_w - 15
    prog_y = h - 30

    # Label
    drowsy_label = f"Alert: {alert.consecutive_drowsy_frames}/{config.DROWSY_FRAMES_THRESHOLD}"
    cv2.putText(
        display, drowsy_label,
        (prog_x, prog_y - 5),
        font, 0.4, config.COLOR_TEXT, 1, cv2.LINE_AA,
    )

    # Background track
    cv2.rectangle(
        display,
        (prog_x, prog_y),
        (prog_x + prog_w, prog_y + prog_h),
        (60, 60, 60), -1,
    )

    # Filled progress (color transitions from green → yellow → red)
    fill_w = int(prog_w * progress)
    if fill_w > 0:
        if progress < 0.5:
            prog_color = config.COLOR_ALERT
        elif progress < 0.85:
            prog_color = config.COLOR_WARNING
        else:
            prog_color = config.COLOR_DROWSY

        cv2.rectangle(
            display,
            (prog_x, prog_y),
            (prog_x + fill_w, prog_y + prog_h),
            prog_color, -1,
        )

    # ── Timestamp (bottom center) ──
    timestamp = get_timestamp()
    (tw, _), _ = cv2.getTextSize(timestamp, font, 0.4, 1)
    cv2.putText(
        display, timestamp,
        ((w - tw) // 2, h - 10),
        font, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
    )

    return display


# ──────────────────────────────────────────────────────────────────────────────
# Main Application Loop
# ──────────────────────────────────────────────────────────────────────────────

def run() -> None:
    """
    Launch the real-time Drowsiness Detection System.

    Opens the webcam, loads the classifier, and enters the frame loop.
    Press 'q' or ESC to exit.
    """
    global logger
    logger = setup_logger()
    ensure_directories()

    logger.info("=" * 60)
    logger.info("  Drowsiness Detection System — Starting")
    logger.info("=" * 60)

    # ── Initialize components ──
    classifier = DrowsinessClassifier()
    classifier.print_info()

    alert = AlertManager()
    fps_counter = FPSCounter()

    # ── Open camera ──
    camera = Camera()
    if not camera.open():
        logger.critical("Cannot start — camera is unavailable.")
        sys.exit(1)

    logger.info("System ready. Press 'q' or ESC to quit.\n")

    try:
        while True:
            # 1. Capture frame
            frame = camera.read()
            if frame is None:
                logger.warning("Skipping frame — capture returned None.")
                continue

            # 2. Classify
            result = classifier.classify(frame)

            # 3. Update alert state
            alert.update(result.is_drowsy)

            # 4. Save snapshot on alarm trigger (if enabled)
            if (
                alert.is_alarm_active
                and config.SAVE_SNAPSHOTS
                and alert.consecutive_drowsy_frames == config.DROWSY_FRAMES_THRESHOLD
            ):
                Camera.save_snapshot(frame)

            # 5. Render HUD
            fps_counter.tick()
            display = draw_hud(frame, result, alert, fps_counter.get())

            # 6. Show frame
            cv2.imshow(config.WINDOW_TITLE, display)

            # 7. Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):   # 'q' or ESC
                logger.info("Quit requested by user.")
                break
            elif key == ord("s"):       # Save snapshot
                Camera.save_snapshot(frame, prefix="manual_snapshot")
                logger.info("Manual snapshot saved.")
            elif key == ord("r"):       # Reset alert
                alert.reset()
                logger.info("Alert state reset by user.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        # ── Cleanup ──
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Application stopped. Goodbye.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

# Required for draw_hud's pulsing border effect
import math

if __name__ == "__main__":
    run()
