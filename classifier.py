"""
=============================================================================
  classifier.py — YOLO Classification Inference Engine
=============================================================================
  Wraps the Ultralytics YOLO11m-cls model with a clean interface.

  Responsibilities:
    - Load the trained model from disk
    - Run classification inference on a single frame
    - Return structured results (class name, confidence, all probabilities)
    - Expose model metadata (class names, architecture info)

  This module has NO knowledge of webcams, alerts, or UI. It only
  knows about images and classification.
=============================================================================
"""

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger("drowsiness")


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Structured output from a single classification inference.

    Attributes:
        predicted_class: Name of the top-1 predicted class (e.g., "Drowsy").
        confidence:      Confidence score of the top-1 class (0.0 to 1.0).
        is_drowsy:       Convenience flag — True if predicted_class matches
                         the configured drowsy class name AND confidence
                         exceeds the threshold.
        all_probabilities: Dict mapping every class name to its probability.
    """
    predicted_class: str
    confidence: float
    is_drowsy: bool
    all_probabilities: dict[str, float]


# ──────────────────────────────────────────────────────────────────────────────
# Classifier Class
# ──────────────────────────────────────────────────────────────────────────────

class DrowsinessClassifier:
    """
    Encapsulates the YOLO11m-cls drowsiness classification model.

    Example:
        classifier = DrowsinessClassifier()
        result = classifier.classify(frame)
        if result.is_drowsy:
            trigger_alarm()
    """

    def __init__(self, model_path: str = config.MODEL_PATH):
        """
        Load the YOLO model from disk.

        Args:
            model_path: Path to the .pt weights file.
        """
        logger.info(f"Loading YOLO model from: {model_path}")
        self._model = YOLO(model_path)
        self._class_names: dict[int, str] = self._model.names
        self._task: str = getattr(self._model, "task", "classify")

        logger.info(
            f"Model loaded — Task: {self._task}, "
            f"Classes: {list(self._class_names.values())}"
        )

    # ── Public Properties ─────────────────────────────────────────────────

    @property
    def class_names(self) -> dict[int, str]:
        """Return the class index → name mapping."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Return the total number of classes."""
        return len(self._class_names)

    @property
    def task(self) -> str:
        """Return the model task type (should be 'classify')."""
        return self._task

    # ── Inference ─────────────────────────────────────────────────────────

    def classify(self, frame: np.ndarray) -> ClassificationResult:
        """
        Run classification on a single image frame.

        Args:
            frame: BGR image as a numpy array (any size — YOLO resizes internally).

        Returns:
            ClassificationResult with the prediction details.
        """
        # Run inference (suppress verbose logging from Ultralytics)
        results = self._model.predict(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # Extract classification probabilities from the first result
        result = results[0]
        probs = result.probs

        # Top-1 prediction
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf.cpu().numpy())
        top1_name = self._class_names.get(top1_idx, f"class_{top1_idx}")

        # Build the full probability map
        all_probs_array = probs.data.cpu().numpy()
        all_probabilities = {
            self._class_names[i]: float(all_probs_array[i])
            for i in range(len(self._class_names))
        }

        # Determine drowsiness
        is_drowsy = (
            top1_name == config.DROWSY_CLASS_NAME
            and top1_conf >= config.CONFIDENCE_THRESHOLD
        )

        return ClassificationResult(
            predicted_class=top1_name,
            confidence=top1_conf,
            is_drowsy=is_drowsy,
            all_probabilities=all_probabilities,
        )

    # ── Model Info ────────────────────────────────────────────────────────

    def print_info(self) -> None:
        """Print a detailed summary of the model architecture to the console."""
        separator = "=" * 60
        print(separator)
        print("  YOLO CLASSIFICATION MODEL INFO")
        print(separator)
        self._model.info()
        print(f"\n  Task      : {self._task}")
        print(f"  Classes   : {self.num_classes}")
        for idx, name in self._class_names.items():
            print(f"    [{idx}] {name}")
        print(separator)
