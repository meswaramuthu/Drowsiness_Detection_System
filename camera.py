"""
=============================================================================
  camera.py — Webcam Capture & Frame Management
=============================================================================
  Handles all webcam interactions:
    - Opening / releasing the camera device
    - Reading frames
    - Applying pre-processing (resize, flip)
    - Saving snapshot images

  This module has NO knowledge of classification or alerts.
  It only deals with image acquisition.
=============================================================================
"""

import os
import logging

import cv2
import numpy as np

import config
from utils import get_file_timestamp

logger = logging.getLogger("drowsiness")


class Camera:
    """
    Manages the lifecycle of a webcam device.

    Usage:
        cam = Camera()
        cam.open()
        while cam.is_opened():
            frame = cam.read()
            if frame is not None:
                process(frame)
        cam.release()

    Or as a context manager:
        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        device_index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
    ):
        """
        Args:
            device_index: Camera device index (0 = default webcam).
            width:        Desired capture width in pixels.
            height:       Desired capture height in pixels.
            fps:          Desired frames per second.
        """
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    # ── Context Manager Support ───────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the webcam device and configure resolution/FPS.

        Returns:
            True if the camera was opened successfully, False otherwise.
        """
        logger.info(f"Opening camera (device index: {self._device_index})...")

        self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            logger.error(
                f"Failed to open camera at index {self._device_index}. "
                "Check that no other application is using it."
            )
            return False

        # Set resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Read back the actual values (camera may not support requested settings)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))

        logger.info(
            f"Camera opened — Resolution: {actual_w}x{actual_h}, FPS: {actual_fps}"
        )
        return True

    def release(self) -> None:
        """Release the webcam device and free resources."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released.")
        self._cap = None

    def is_opened(self) -> bool:
        """Check if the camera is currently open and accessible."""
        return self._cap is not None and self._cap.isOpened()

    # ── Frame Reading ─────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """
        Capture a single frame from the webcam.

        Returns:
            BGR numpy array of the captured frame, or None if capture fails.
        """
        if not self.is_opened():
            return None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera.")
            return None

        # Mirror the frame horizontally for a natural selfie-view
        frame = cv2.flip(frame, 1)

        return frame

    # ── Snapshot Saving ───────────────────────────────────────────────────

    @staticmethod
    def save_snapshot(
        frame: np.ndarray,
        output_dir: str = config.OUTPUT_DIR,
        prefix: str = "drowsy_snapshot",
    ) -> str | None:
        """
        Save a frame to disk as a timestamped JPEG snapshot.

        Args:
            frame:      The BGR image to save.
            output_dir: Directory to save the image in.
            prefix:     Filename prefix.

        Returns:
            The full path to the saved file, or None on failure.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{prefix}_{get_file_timestamp()}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            logger.info(f"Snapshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None
