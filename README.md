# 🚗 Drowsiness Detection System

A **real-time webcam-based drowsiness detection system** powered by a trained **YOLO11m-cls** classification model. The system monitors a driver's face through a webcam, classifies each frame as **Drowsy** or **Non_Drowsy**, and triggers an audio alarm when sustained drowsiness is detected.

---

## 📋 Features

- **Real-time classification** — processes webcam frames with YOLO11m-cls
- **Consecutive-frame logic** — avoids false positives by requiring sustained drowsiness before alarming
- **Audio alarm** — auto-generates a dual-tone siren WAV if no alarm file exists
- **HUD overlay** — live confidence bars, FPS counter, drowsy progress bar, and status banner
- **Snapshot saving** — automatically saves images when drowsiness is detected
- **Session logging** — timestamped log files for every session
- **Keyboard controls** — quit, snapshot, and reset via keyboard shortcuts

---

## 🏗️ Project Structure

```
Drowsiness_Detection_System/
│
├── app.py              # Main entry point — orchestrates the real-time loop
├── camera.py           # Webcam capture, frame reading, snapshot saving
├── classifier.py       # YOLO model loading & classification inference
├── alert.py            # Alarm system (sound generation, playback, cooldown)
├── config.py           # Central configuration (all paths, thresholds, constants)
├── utils.py            # Shared helpers (logging, timestamps, FPS counter)
├── analyze_model.py    # Standalone model analysis & inspection script
├── requirements.txt    # Python dependencies
├── README.md           # This file
│
├── Model/              # Trained model weights
│   ├── best.pt         #   PyTorch weights (YOLO11m-cls)
│   └── best.onnx       #   ONNX export (for edge/cross-platform deployment)
│
├── sounds/             # Audio files
│   └── alarm.wav       #   Alarm sound (auto-generated if missing)
│
├── outputs/            # Saved snapshot images (created at runtime)
├── logs/               # Session log files (created at runtime)
└── assets/             # Static assets (icons, fonts, overlays)
```

---

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | **Main application**. Initializes camera, classifier, and alert manager. Runs the real-time frame loop with HUD rendering and keyboard controls. |
| `camera.py` | **Webcam management**. Opens/releases the camera, reads frames, applies horizontal flip for selfie-view, and saves snapshot images. |
| `classifier.py` | **Classification engine**. Loads the YOLO11m-cls model, runs inference on a frame, and returns a structured `ClassificationResult` (predicted class, confidence, probabilities). |
| `alert.py` | **Alert system**. Tracks consecutive drowsy frames, triggers audio alarms after a threshold, manages cooldowns, generates alarm WAV files, and plays sound asynchronously. |
| `config.py` | **Configuration hub**. All tunable parameters — file paths, camera settings, classification thresholds, alert timings, display options, and colors. |
| `utils.py` | **Utilities**. Logging setup (file + console), timestamp formatting, directory initialization, and an FPS counter class. |
| `analyze_model.py` | **Model inspector**. Standalone script to print model architecture, class names, and run a test inference with annotated output. |
| `requirements.txt` | **Dependencies**. Lists all required Python packages. |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

```bash
python app.py
```

### 3. Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit the application |
| `s` | Save a manual snapshot |
| `r` | Reset the alert state |

---

## ⚙️ Configuration

All settings are in **`config.py`**. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.60` | Minimum confidence to classify as drowsy |
| `DROWSY_FRAMES_THRESHOLD` | `15` | Consecutive drowsy frames before alarm triggers |
| `ALARM_COOLDOWN_SECONDS` | `5.0` | Cooldown between alarm sounds |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `SAVE_SNAPSHOTS` | `True` | Auto-save images on drowsiness detection |

---

## 🧪 Model Analysis

To inspect the model without running the full system:

```bash
python analyze_model.py --synthetic            # Use synthetic test image
python analyze_model.py --image path/to/img    # Use a specific image
python analyze_model.py                        # Use webcam
```

---

## 🔧 Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLO11m-cls |
| Task | Image Classification |
| Parameters | 10,355,778 |
| GFLOPs | 39.6 |
| Classes | `Drowsy`, `Non_Drowsy` |
| Input Size | 224×224 |
| File Size | ~20 MB |

---

## 📄 License

This project is for educational and research purposes.