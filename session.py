"""
=============================================================================
  session.py — Session Management & Formatted Summary Reporting
=============================================================================
  Provides lightweight tracking of runtime statistics for each session.
  When the application exits, prints a dashboard-like report to the console
  and saves a matching text file summary to the logs/ directory.

  Abides by SOLID principles:
    - Single Responsibility: Manages only session metrics and report generation.
    - Open/Closed: Can be extended for custom output writers.
    - Dependency Inversion: Has no direct dependencies on classifiers,
      cameras, or audio alerts.
=============================================================================
"""

import os
import time
from datetime import datetime
import logging

import config

logger = logging.getLogger("drowsiness")


class SessionManager:
    """
    Tracks, updates, and reports drowsiness detection session statistics.
    Designed to run independently of webcam hardware and deep learning frameworks.
    """

    def __init__(self, model_name: str, backend: str):
        """
        Initialize the session manager and start recording time.

        Args:
            model_name: The name of the YOLO model (e.g., YOLO11m-cls).
            backend:    The inference backend engine (e.g., PyTorch, ONNX).
        """
        self._model_name = model_name
        self._backend = backend

        # Timing
        self._start_epoch = time.time()
        self._start_datetime = datetime.now()
        self._end_epoch: float | None = None
        self._end_datetime: datetime | None = None

        # Statistics
        self._total_frames = 0
        self._fps_values: list[float] = []
        self._drowsy_events = 0
        self._alarm_activations = 0
        self._snapshots_saved = 0
        self._confidence_sum = 0.0

    # ── Public Metrics Update Methods ─────────────────────────────────────

    def register_frame(self, is_drowsy: bool, confidence: float, current_fps: float) -> None:
        """
        Log metrics from a single processed camera frame.

        Args:
            is_drowsy:   Whether the current frame classified the driver as drowsy.
            confidence:  The prediction confidence score (float, 0.0 to 1.0).
            current_fps: The current instantaneous frame rate.
        """
        self._total_frames += 1
        self._confidence_sum += confidence

        # Ignore 0 or negative FPS startup noise for min/max tracking
        if current_fps > 0.1:
            self._fps_values.append(current_fps)

        if is_drowsy:
            self._drowsy_events += 1

    def increment_alarm_activations(self) -> None:
        """Increment the counter for FSM alarm triggers."""
        self._alarm_activations += 1

    def increment_snapshots_saved(self) -> None:
        """Increment the counter for auto-saved alert snapshots."""
        self._snapshots_saved += 1

    # ── Session Completion & Report Writing ───────────────────────────────

    def end_session(self) -> None:
        """
        Mark the session end time, output the report to the console,
        and write the session summary report file to logs/.
        """
        self._end_epoch = time.time()
        self._end_datetime = datetime.now()

        # Generate summary report string
        report = self._generate_report_string()

        # 1. Output to standard console
        print(report)

        # 2. Write to logs directory
        self._save_report_to_disk(report)

    # ── Private Helper Methods ────────────────────────────────────────────

    def _generate_report_string(self) -> str:
        """
        Formats the current stats into a clean, human-readable summary string.
        """
        # Duration calculation
        end_time = self._end_epoch if self._end_epoch is not None else time.time()
        duration_seconds = max(0.0, end_time - self._start_epoch)
        
        hours, remainder = divmod(int(duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # FPS Calculations (Min, Max, Avg)
        if self._fps_values:
            min_fps = min(self._fps_values)
            max_fps = max(self._fps_values)
            avg_fps = sum(self._fps_values) / len(self._fps_values)
        else:
            min_fps = max_fps = avg_fps = 0.0

        # Average Confidence calculation
        avg_confidence = (
            (self._confidence_sum / self._total_frames) if self._total_frames > 0 else 0.0
        )

        # Datetime display strings
        start_str = self._start_datetime.strftime("%d %b %Y %H:%M:%S")
        end_str = (
            self._end_datetime.strftime("%d %b %Y %H:%M:%S")
            if self._end_datetime
            else datetime.now().strftime("%d %b %Y %H:%M:%S")
        )

        # Build formatted report layout
        report = []
        report.append("======================================================")
        report.append("          DROWSINESS DETECTION SYSTEM")
        report.append("               SESSION SUMMARY")
        report.append("======================================================")
        report.append("")
        report.append(f"Start Time          : {start_str}")
        report.append(f"End Time            : {end_str}")
        report.append(f"Duration            : {duration_str}")
        report.append("")
        report.append(f"Model               : {self._model_name}")
        report.append(f"Backend             : {self._backend}")
        report.append("")
        report.append(f"Frames Processed    : {self._total_frames}")
        report.append("")
        report.append(f"Average FPS         : {avg_fps:.1f}")
        report.append(f"Maximum FPS         : {max_fps:.1f}")
        report.append(f"Minimum FPS         : {min_fps:.1f}")
        report.append("")
        report.append(f"Drowsy Events       : {self._drowsy_events}")
        report.append(f"Alarm Activations   : {self._alarm_activations}")
        report.append(f"Snapshots Saved     : {self._snapshots_saved}")
        report.append("")
        report.append(f"Average Confidence  : {avg_confidence:.1%} ")
        report.append("")
        report.append("======================================================")
        report.append("Thank you for using the system.")
        report.append("======================================================")
        
        return "\n".join(report)

    def _save_report_to_disk(self, report_content: str) -> None:
        """Write the summary text block to logs/session_summary_YYYYMMDD_HHMMSS.txt."""
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            timestamp = self._start_datetime.strftime("%Y%m%d_%H%M%S")
            filename = f"session_summary_{timestamp}.txt"
            filepath = os.path.join(config.LOG_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            logger.info(f"Session summary report saved to disk: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save session summary report to disk: {e}")
