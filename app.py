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

import math
import config
from utils import ensure_directories, setup_logger, get_timestamp, FPSCounter
from camera import Camera
from classifier import DrowsinessClassifier, ClassificationResult
from alert import AlertManager
import ui
from session import SessionManager

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
    Render the professional heads-up dashboard layout on the video frame.
    All modular drawing details are delegated to the ui module.
    """
    display = frame.copy()
    w, h = display.shape[1], display.shape[0]

    # ── Alarm flashing border (sustained alarm warning) ──
    if alert.is_alarm_active:
        import time
        # Pulsing outline border width
        pulse = int(abs(math.sin(time.time() * 6)) * 12) + 4
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), pulse)

    # ── Render Dashboard UI panels ──
    # Auto-detect inference backend from path
    backend_type = "ONNX" if config.MODEL_PATH.endswith(".onnx") else "PyTorch"
    
    # 1. Header panel (Project details, model, backend name)
    ui.draw_header(
        frame=display,
        project_name="Drowsiness Detection",
        model_name="YOLO11m-cls",
        backend_name=backend_type,
    )
    
    # 2. Cyan FPS overlay inside the header panel
    ui.draw_fps(display, fps)

    # 3. Status panel (SAFE / DROWSY indicator light)
    ui.draw_status(display, alert.is_alarm_active)

    # 4. Central Prediction & Confidence progress bar overlay
    ui.draw_prediction(
        frame=display,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        is_drowsy=result.is_drowsy,
    )

    # 5. Footer panel (Alarm state, Exit controls, Clock)
    ui.draw_footer(display, alert.is_alarm_active)

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

    # Determine backend type for session summary
    backend_type = "ONNX" if config.MODEL_PATH.endswith(".onnx") else "PyTorch"
    session_manager = SessionManager(model_name="YOLO11m-cls", backend=backend_type)

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

            # 3. Update alert state (FSM transitions, alarm trigger, and snapshot logging)
            prev_alarm_state = alert.is_alarm_active
            alert.update(result.is_drowsy, frame)

            # Detect SAFE -> DROWSY state change to increment alarm/snapshot metrics
            if alert.is_alarm_active and not prev_alarm_state:
                session_manager.increment_alarm_activations()
                if config.SAVE_SNAPSHOTS:
                    session_manager.increment_snapshots_saved()

            # 5. Render HUD
            fps_counter.tick()
            current_fps = fps_counter.get()
            display = draw_hud(frame, result, alert, current_fps)

            # Update session frame statistics
            session_manager.register_frame(
                is_drowsy=result.is_drowsy,
                confidence=result.confidence,
                current_fps=current_fps,
            )

            # 6. Show frame
            cv2.imshow(config.WINDOW_TITLE, display)

            # 7. Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):   # 'q' or ESC
                logger.info("Quit requested by user.")
                break
            elif key == ord("s"):       # Save manual snapshot
                Camera.save_snapshot(frame, prefix="manual_snapshot")
                session_manager.increment_snapshots_saved()
                logger.info("Manual snapshot saved.")
            elif key == ord("r"):       # Reset alert
                alert.reset()
                logger.info("Alert state reset by user.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        # ── Cleanup & Session Summary ──
        camera.release()
        cv2.destroyAllWindows()
        session_manager.end_session()
        logger.info("Application stopped. Goodbye.\n")


if __name__ == "__main__":
    run()
