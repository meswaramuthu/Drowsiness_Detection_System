"""
=============================================================================
  Drowsiness Detection — YOLO Classification Model Analyzer
=============================================================================
  This script performs the following:
    1. Loads the trained YOLO classification model (best.pt) using Ultralytics.
    2. Prints model architecture, class names, and number of classes.
    3. Runs inference on a sample image (webcam capture or provided path).
    4. Overlays classification results with confidence bars on the image.
    5. Saves the annotated output image.

  NOTE: This model is a YOLO11m-cls (classification) model.
        It classifies the entire image as "Drowsy" or "Non_Drowsy".
        It does NOT produce bounding boxes (that would be a detection model).

  Usage:
    python analyze_model.py                        # Capture from webcam
    python analyze_model.py --image path/to/img    # Use a specific image
    python analyze_model.py --synthetic            # Use a generated test image
=============================================================================
"""

import argparse
import sys
import os
import textwrap

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Path to the trained model weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "best.pt")

# Directory to save annotated output images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Inference confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Colors for each class (BGR format)
CLASS_COLORS = {
    "Drowsy":     (0, 0, 255),     # Red — indicates danger / drowsiness
    "Non_Drowsy": (0, 200, 0),     # Green — indicates alert / awake
}

# Fallback color palette if class names differ from expected
FALLBACK_COLORS = [
    (0, 0, 255),     # Red
    (0, 200, 0),     # Green
    (255, 165, 0),   # Orange
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Loading & Information
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> YOLO:
    """Load a YOLO model from the given weights file."""
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)
    print("[INFO] Model loaded successfully.\n")
    return model


def print_model_info(model: YOLO) -> dict:
    """
    Print detailed information about the loaded YOLO model.
    Returns the class names dictionary.
    """
    separator = "=" * 70
    print(separator)
    print("  MODEL INFORMATION")
    print(separator)

    # --- Model Architecture ---
    print("\n📐 Architecture Summary:")
    print("-" * 40)
    model.info()

    # --- Class Names ---
    class_names = model.names  # dict: {index: class_name}
    num_classes = len(class_names)

    print(f"\n🏷️  Number of Classes: {num_classes}")
    print("-" * 40)
    for idx, name in class_names.items():
        print(f"  Class {idx}: {name}")

    # --- Task Type ---
    task = getattr(model, "task", "unknown")
    print(f"\n⚙️  Task Type: {task}")

    # --- Model File Size ---
    model_path = str(model.ckpt_path) if hasattr(model, "ckpt_path") else None
    if model_path and os.path.isfile(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📦 Model File Size: {size_mb:.1f} MB")

    print(separator + "\n")
    return class_names


# ──────────────────────────────────────────────────────────────────────────────
# 2. Image Acquisition
# ──────────────────────────────────────────────────────────────────────────────

def capture_from_webcam() -> np.ndarray:
    """Capture a single frame from the default webcam (index 0)."""
    print("[INFO] Opening webcam to capture a test frame...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Use --image or --synthetic instead.")
        sys.exit(1)

    # Allow the camera to warm up — discard initial dark frames
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to capture frame from webcam.")
        sys.exit(1)

    print(f"[INFO] Captured frame: {frame.shape[1]}x{frame.shape[0]} pixels.\n")
    return frame


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk."""
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Loaded image: {image_path}")
    print(f"       Dimensions: {frame.shape[1]}x{frame.shape[0]} pixels.\n")
    return frame


def generate_synthetic_image() -> np.ndarray:
    """
    Generate a synthetic test image with a face-like shape.
    Useful when no webcam or real image is available.
    """
    print("[INFO] Generating synthetic test image (640x480)...\n")

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background

    # Draw a simple face-like ellipse in the center
    center = (320, 220)
    cv2.ellipse(img, center, (100, 130), 0, 0, 360, (200, 180, 160), -1)

    # Eyes (semi-closed to simulate drowsiness)
    cv2.ellipse(img, (280, 200), (20, 6), 0, 0, 360, (60, 60, 60), -1)
    cv2.ellipse(img, (360, 200), (20, 6), 0, 0, 360, (60, 60, 60), -1)

    # Mouth (slightly open — yawning)
    cv2.ellipse(img, (320, 280), (30, 20), 0, 0, 360, (80, 50, 50), -1)

    # Add label
    cv2.putText(
        img, "Synthetic Test Image", (180, 440),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 3. Inference & Annotation (Classification)
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model: YOLO, image: np.ndarray, conf_threshold: float = CONFIDENCE_THRESHOLD) -> list:
    """
    Run YOLO inference on the given image.
    Returns the list of Results objects from Ultralytics.
    """
    print("[INFO] Running inference...")
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False,  # suppress per-image logging
    )
    print("[INFO] Inference complete.\n")
    return results


def annotate_classification(
    image: np.ndarray,
    results: list,
    class_names: dict,
) -> np.ndarray:
    """
    Annotate the image with classification results.

    Since this is a classification model (not detection), there are no
    bounding boxes. Instead, we overlay:
      - The predicted class label (top of image)
      - Confidence bars for ALL classes
      - A colored border indicating the prediction

    Args:
        image:       The input image (BGR, numpy array).
        results:     List of Ultralytics Results objects.
        class_names: Dict mapping class index -> class name.

    Returns:
        Annotated image with classification overlay.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for result in results:
        # ── Extract classification probabilities ──
        # result.probs contains the classification output
        probs = result.probs

        if probs is None:
            print("[WARN] No classification probabilities found in results.")
            continue

        # Get the top-1 prediction
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf.cpu().numpy())
        top1_name = class_names.get(top1_idx, f"class_{top1_idx}")

        # Get ALL class probabilities for the bar chart
        all_probs = probs.data.cpu().numpy()  # array of shape (num_classes,)

        # ── Determine the overlay color based on prediction ──
        if top1_name in CLASS_COLORS:
            pred_color = CLASS_COLORS[top1_name]
        else:
            pred_color = FALLBACK_COLORS[top1_idx % len(FALLBACK_COLORS)]

        # ── Draw a colored border around the image ──
        border_thickness = 8
        cv2.rectangle(
            annotated,
            (0, 0), (w - 1, h - 1),
            pred_color, border_thickness,
        )

        # ── Draw semi-transparent overlay panel at the top ──
        panel_height = 40 + (len(class_names) * 45)  # dynamic height
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # ── Draw the predicted class label ──
        label = f"Prediction: {top1_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2

        cv2.putText(
            annotated, label,
            (15, 30),
            font, font_scale, pred_color, thickness, cv2.LINE_AA,
        )

        # ── Draw confidence bars for each class ──
        bar_x_start = 15
        bar_max_width = w - 30  # leave margin on both sides
        bar_height = 22
        y_offset = 55  # starting Y for the first bar

        for idx in range(len(class_names)):
            name = class_names.get(idx, f"class_{idx}")
            prob = float(all_probs[idx])

            # Pick the color for this class
            if name in CLASS_COLORS:
                bar_color = CLASS_COLORS[name]
            else:
                bar_color = FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]

            # Class label text
            class_label = f"{name}: {prob:.1%}"
            cv2.putText(
                annotated, class_label,
                (bar_x_start, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Background bar (gray track)
            bar_y = y_offset + 5
            cv2.rectangle(
                annotated,
                (bar_x_start, bar_y),
                (bar_x_start + bar_max_width, bar_y + bar_height),
                (60, 60, 60), -1,
            )

            # Filled bar (proportional to confidence)
            filled_width = int(bar_max_width * prob)
            if filled_width > 0:
                cv2.rectangle(
                    annotated,
                    (bar_x_start, bar_y),
                    (bar_x_start + filled_width, bar_y + bar_height),
                    bar_color, -1,
                )

            # Confidence percentage on the bar
            cv2.putText(
                annotated, f"{prob:.1%}",
                (bar_x_start + filled_width + 5, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
            )

            y_offset += 45  # move to next bar

        # ── Print classification summary to console ──
        print(f"{'=' * 60}")
        print(f"  CLASSIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  🏆 Predicted Class : {top1_name}")
        print(f"  📊 Confidence      : {top1_conf:.1%}")
        print(f"{'─' * 60}")
        print(f"  All class probabilities:")
        for idx in range(len(class_names)):
            name = class_names.get(idx, f"class_{idx}")
            prob = float(all_probs[idx])
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            marker = " ◄── predicted" if idx == top1_idx else ""
            print(f"    {name:15s} [{bar}] {prob:.1%}{marker}")
        print(f"{'=' * 60}\n")

    return annotated


# ──────────────────────────────────────────────────────────────────────────────
# 4. Save Output
# ──────────────────────────────────────────────────────────────────────────────

def save_output(image: np.ndarray, output_dir: str) -> str:
    """Save the annotated image to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "classification_result.jpg")

    cv2.imwrite(output_path, image)
    print(f"[INFO] Annotated image saved to: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a trained YOLO Drowsiness Classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python analyze_model.py                        # Webcam capture
              python analyze_model.py --image photo.jpg      # Specific image
              python analyze_model.py --synthetic            # Synthetic test image
        """),
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a sample image for inference.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a generated synthetic test image (no webcam needed).",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help=f"Path to YOLO model weights (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for predictions (default: {CONFIDENCE_THRESHOLD}).",
    )

    args = parser.parse_args()

    # ── Step 1: Load the model ──
    model = load_model(args.model)

    # ── Step 2: Print model information ──
    class_names = print_model_info(model)

    # ── Step 3: Acquire a test image ──
    if args.image:
        image = load_image(args.image)
    elif args.synthetic:
        image = generate_synthetic_image()
    else:
        image = capture_from_webcam()

    # ── Step 4: Run inference ──
    results = run_inference(model, image, conf_threshold=args.conf)

    # ── Step 5: Annotate the image with classification results ──
    annotated = annotate_classification(image, results, class_names)

    # ── Step 6: Save the output ──
    save_output(annotated, OUTPUT_DIR)

    print("\n✅ Analysis complete. Check the 'output/' folder for results.\n")


if __name__ == "__main__":
    main()
