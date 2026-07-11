"""
=============================================================================
  ui.py — Professional OpenCV Dashboard User Interface
=============================================================================
  Handles all HUD rendering on the camera feed:
    - Renders rounded semi-transparent panels.
    - Displays project name, model name, backend type, and FPS.
    - Shows prediction class and a visual horizontal confidence progress bar.
    - Draws active status indicators (SAFE / DROWSY) and alarm state.
    - Overlay current timestamp.
=============================================================================
"""

import time
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Color Palette (BGR format)
# ──────────────────────────────────────────────────────────────────────────────
COLOR_SAFE = (0, 200, 0)        # Green
COLOR_DROWSY = (0, 0, 255)      # Red
COLOR_TEXT_WHITE = (255, 255, 255) # White for general text
COLOR_FPS_CYAN = (255, 255, 0)  # Cyan
COLOR_PANEL_BG = (20, 20, 20)   # Dark gray for background panels
COLOR_GRAY = (120, 120, 120)    # Light gray for borders/tracks


# ──────────────────────────────────────────────────────────────────────────────
# Graphic Utilities
# ──────────────────────────────────────────────────────────────────────────────

def draw_rounded_rectangle(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = -1,
    radius: int = 10,
) -> None:
    """
    Draw a rounded rectangle using OpenCV drawing functions.

    Args:
        img:       Image to draw on.
        pt1:       Top-left coordinate (x1, y1).
        pt2:       Bottom-right coordinate (x2, y2).
        color:     BGR color tuple.
        thickness: Thickness of the border. -1 for filled.
        radius:    Radius of the rounded corners.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Ensure radius is within valid bounds
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    radius = min(radius, w // 2, h // 2)

    if radius <= 0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    if thickness < 0:
        # Draw filled rectangles for center, left, and right portions
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
        cv2.rectangle(img, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)

        # Draw filled corner circles
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Draw border lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

        # Draw corner arcs (ellipses)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)


def draw_panel(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = COLOR_PANEL_BG,
    alpha: float = 0.55,
    radius: int = 12,
) -> None:
    """
    Draw a semi-transparent rounded background panel.
    """
    overlay = img.copy()
    draw_rounded_rectangle(overlay, pt1, pt2, color, thickness=-1, radius=radius)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


# ──────────────────────────────────────────────────────────────────────────────
# UI Drawing Modules
# ──────────────────────────────────────────────────────────────────────────────

def draw_header(
    frame: np.ndarray,
    project_name: str,
    model_name: str,
    backend_name: str,
) -> None:
    """
    Render the Top-Left metadata header panel.
    """
    # Header coordinates (Top-Left)
    pt1 = (15, 15)
    pt2 = (280, 100)

    # Semi-transparent background panel
    draw_panel(frame, pt1, pt2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Project Title
    cv2.putText(frame, project_name, (25, 38), font, 0.55, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
    
    # Model Weights info
    model_text = f"Model: {model_name}"
    cv2.putText(frame, model_text, (25, 60), font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
    
    # Inference engine backend
    backend_text = f"Backend: {backend_name}"
    cv2.putText(frame, backend_text, (25, 82), font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)


def draw_status(frame: np.ndarray, is_drowsy: bool) -> None:
    """
    Render the Top-Right status panel showing SAFE/DROWSY with indicator lights.
    """
    w = frame.shape[1]
    
    # Status panel coordinates
    pt1 = (w - 200, 15)
    pt2 = (w - 15, 75)

    draw_panel(frame, pt1, pt2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "STATUS:", (w - 185, 38), font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

    if is_drowsy:
        # Red LED circle indicator + text
        cv2.circle(frame, (w - 175, 56), 8, COLOR_DROWSY, -1)
        cv2.circle(frame, (w - 175, 56), 11, COLOR_DROWSY, 1, cv2.LINE_AA) # pulsing ring
        cv2.putText(frame, "DROWSY", (w - 155, 62), font, 0.65, COLOR_DROWSY, 2, cv2.LINE_AA)
    else:
        # Green LED circle indicator + text
        cv2.circle(frame, (w - 175, 56), 8, COLOR_SAFE, -1)
        cv2.circle(frame, (w - 175, 56), 11, COLOR_SAFE, 1, cv2.LINE_AA) # static outer ring
        cv2.putText(frame, "SAFE", (w - 155, 62), font, 0.65, COLOR_SAFE, 2, cv2.LINE_AA)


def draw_prediction(
    frame: np.ndarray,
    predicted_class: str,
    confidence: float,
    is_drowsy: bool,
) -> None:
    """
    Render the Center classification results overlay (Prediction & Confidence).
    """
    w, h = frame.shape[1], frame.shape[0]

    # Position: Bottom Center area of the viewport
    pt1 = (w // 2 - 160, h - 170)
    pt2 = (w // 2 + 160, h - 85)

    draw_panel(frame, pt1, pt2, alpha=0.6)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title label
    cv2.putText(frame, "PREDICTION", (w // 2 - 145, h - 148), font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

    # Class Name Value
    class_color = COLOR_DROWSY if is_drowsy else COLOR_SAFE
    cv2.putText(
        frame,
        predicted_class.upper(),
        (w // 2 - 35, h - 146),
        font,
        0.6,
        class_color,
        2,
        cv2.LINE_AA,
    )

    # Draw the progress bar inside this panel
    bar_pt1 = (w // 2 - 145, h - 122)
    bar_pt2 = (w // 2 + 75, h - 105)
    draw_progress_bar(frame, bar_pt1, bar_pt2, confidence, class_color)

    # Percentage string next to progress bar
    conf_text = f"{confidence:.0%}"
    cv2.putText(
        frame,
        conf_text,
        (w // 2 + 90, h - 108),
        font,
        0.5,
        COLOR_TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_progress_bar(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    progress: float,
    fill_color: tuple[int, int, int],
) -> None:
    """
    Draw a clean, horizontal confidence progress bar.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw background track
    draw_rounded_rectangle(frame, pt1, pt2, (45, 45, 45), thickness=-1, radius=6)

    # Draw filled value bar based on progress percentage
    width = x2 - x1
    filled_width = int(width * progress)
    
    if filled_width > 4:
        fill_pt2 = (x1 + filled_width, y2)
        draw_rounded_rectangle(frame, pt1, fill_pt2, fill_color, thickness=-1, radius=6)


def draw_footer(frame: np.ndarray, alarm_active: bool) -> None:
    """
    Render Bottom-Left alarm status, Bottom-Right exit instruction, and current time.
    """
    w, h = frame.shape[1], frame.shape[0]

    # 1. Bottom Left: Alarm State Panel
    al_pt1 = (15, h - 55)
    al_pt2 = (165, h - 15)
    draw_panel(frame, al_pt1, al_pt2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "ALARM:", (25, h - 30), font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
    
    if alarm_active:
        cv2.putText(frame, "ON", (90, h - 28), font, 0.6, COLOR_DROWSY, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "OFF", (90, h - 28), font, 0.6, COLOR_SAFE, 2, cv2.LINE_AA)

    # 2. Bottom Right: Exit Instruction Panel
    ex_pt1 = (w - 180, h - 55)
    ex_pt2 = (w - 15, h - 15)
    draw_panel(frame, ex_pt1, ex_pt2)
    cv2.putText(frame, "Press 'Q' to Exit", (w - 165, h - 30), font, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    # 3. Bottom Center: Current Time Panel
    time_str = time.strftime("%H:%M:%S")
    time_pt1 = (w // 2 - 60, h - 55)
    time_pt2 = (w // 2 + 60, h - 15)
    draw_panel(frame, time_pt1, time_pt2)
    
    # Center text placement
    (t_w, _), _ = cv2.getTextSize(time_str, font, 0.5, 1)
    cv2.putText(
        frame,
        time_str,
        (w // 2 - t_w // 2, h - 30),
        font,
        0.5,
        COLOR_TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Overlay the current FPS indicator inside the header panel boundary.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps_text = f"FPS: {fps:.1f}"
    
    # Overlay in cyan color near the bottom right of the top-left header panel
    cv2.putText(frame, fps_text, (180, 82), font, 0.45, COLOR_FPS_CYAN, 1, cv2.LINE_AA)
