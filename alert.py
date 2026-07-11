"""
=============================================================================
  alert.py — Alert & Alarm System
=============================================================================
  Manages the drowsiness alert pipeline:
    - Tracks consecutive drowsy frame count
    - Triggers audio alarms with cooldown management
    - Generates an alarm WAV file if one doesn't exist
    - Plays sound asynchronously (non-blocking)

  This module has NO knowledge of cameras or classification models.
  It only knows "alert was triggered" or "alert was cleared".
=============================================================================
"""

import os
import time
import struct
import wave
import math
import logging
import threading

import config

logger = logging.getLogger("drowsiness")


# ──────────────────────────────────────────────────────────────────────────────
# Alarm Sound Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_alarm_wav(filepath: str) -> None:
    """
    Generate a loud, attention-grabbing alarm WAV file programmatically.

    The alarm uses a dual-tone pattern (alternating between two frequencies)
    to create an urgent, pulsing siren effect.

    Args:
        filepath: Destination path for the generated .wav file.
    """
    logger.info(f"Generating alarm sound: {filepath}")
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    sample_rate = 44100
    duration = 3.0              # Total duration in seconds
    freq_high = 880             # Hz — high tone (A5)
    freq_low = 660              # Hz — low tone (E5)
    pulse_rate = 4              # Pulses per second (siren oscillation speed)
    amplitude = 0.85            # Volume (0.0 to 1.0)

    num_samples = int(sample_rate * duration)
    samples = []

    for i in range(num_samples):
        t = i / sample_rate

        # Alternate between high and low frequency to create siren effect
        pulse = (math.sin(2 * math.pi * pulse_rate * t) + 1) / 2  # 0..1
        freq = freq_low + (freq_high - freq_low) * pulse

        # Generate the tone with slight amplitude modulation for urgency
        envelope = 0.7 + 0.3 * math.sin(2 * math.pi * 8 * t)  # tremolo
        sample = amplitude * envelope * math.sin(2 * math.pi * freq * t)

        # Clamp and convert to 16-bit integer
        sample_int = int(max(-1.0, min(1.0, sample)) * 32767)
        samples.append(sample_int)

    # Write WAV file
    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1)          # Mono
        wav_file.setsampwidth(2)          # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            struct.pack(f"<{len(samples)}h", *samples)
        )

    logger.info(f"Alarm sound generated ({duration}s, {sample_rate}Hz).")


# ──────────────────────────────────────────────────────────────────────────────
# Sound Playback (Cross-Platform)
# ──────────────────────────────────────────────────────────────────────────────

def _play_sound_blocking(filepath: str) -> None:
    """
    Play a WAV file. Uses platform-native methods where possible.
    This function blocks until playback completes.
    """
    try:
        import winsound
        winsound.PlaySound(filepath, winsound.SND_FILENAME)
        return
    except ImportError:
        pass  # Not on Windows

    try:
        # macOS / Linux fallback: use the 'playsound' package if available
        from playsound import playsound
        playsound(filepath)
        return
    except ImportError:
        pass

    # Last resort: use system command
    import platform
    system = platform.system()
    if system == "Darwin":
        os.system(f'afplay "{filepath}" &')
    elif system == "Linux":
        os.system(f'aplay "{filepath}" &')
    else:
        logger.warning("No audio playback method available on this platform.")


# ──────────────────────────────────────────────────────────────────────────────
# Alarm Player & FSM States
# ──────────────────────────────────────────────────────────────────────────────

from enum import Enum, auto
import numpy as np

class AlertState(Enum):
    SAFE = auto()
    DROWSY = auto()


class AlarmPlayer:
    """
    Manages background alarm playback loop.
    Controls thread lifecycle, allowing starting/stopping the alarm dynamically.
    """

    def __init__(self, sound_path: str):
        self.sound_path = sound_path
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start playing the alarm sound in a background thread if not already playing."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already playing
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sound playback immediately and terminate the background thread."""
        self._stop_event.set()
        # On Windows, stop any active winsound play immediately to stop the blocking thread
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            _play_sound_blocking(self.sound_path)
            # Short sleep between replays to prevent high CPU utilization
            time.sleep(0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Alert Manager (FSM)
# ──────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Manages the drowsiness alerting logic using a Finite State Machine.
    
    States:
      - SAFE: The driver is awake or under the consecutive frames threshold.
      - DROWSY: The driver is confirmed drowsy and alarm is active.

    Responsibilities:
      - Transitions SAFE -> DROWSY when consecutive frames threshold is met.
      - Transitions DROWSY -> SAFE when recovery (is_drowsy=False) occurs.
      - Ensures only one warning log and one snapshot are triggered per event.
      - Controls continuous alarm playback while in DROWSY state.
    """

    def __init__(
        self,
        frames_threshold: int = config.DROWSY_FRAMES_THRESHOLD,
        cooldown_seconds: float = config.ALARM_COOLDOWN_SECONDS,
        sound_path: str = config.ALARM_SOUND_PATH,
    ):
        self._frames_threshold = frames_threshold
        self._cooldown_seconds = cooldown_seconds
        self._sound_path = sound_path

        # State tracking
        self._state: AlertState = AlertState.SAFE
        self._consecutive_drowsy: int = 0
        
        # Asynchronous alarm player
        self._player = AlarmPlayer(self._sound_path)

        # Ensure the alarm sound file exists
        self._ensure_sound_file()

    def _ensure_sound_file(self) -> None:
        """Generate the alarm WAV file if it doesn't exist on disk."""
        if not os.path.isfile(self._sound_path):
            logger.warning(f"Alarm sound not found at: {self._sound_path}")
            generate_alarm_wav(self._sound_path)

    # ── Public Interface ──────────────────────────────────────────────────

    def update(self, is_drowsy: bool, frame: np.ndarray | None = None) -> None:
        """
        Call once per frame to update the alert state FSM.

        Args:
            is_drowsy: True if the current frame was classified as drowsy.
            frame:     The current raw BGR frame from camera (used for snapshot).
        """
        if self._state == AlertState.SAFE:
            if is_drowsy:
                self._consecutive_drowsy += 1
                if self._consecutive_drowsy >= self._frames_threshold:
                    # Transition: SAFE -> DROWSY
                    self._state = AlertState.DROWSY
                    logger.warning(
                        f"⚠️  [STATE CHANGE: SAFE -> DROWSY] Drowsiness detected! "
                        f"Sustained for {self._consecutive_drowsy} frames."
                    )
                    
                    # 1. Start the alarm
                    self._player.start()
                    
                    # 2. Save one snapshot
                    if frame is not None and config.SAVE_SNAPSHOTS:
                        try:
                            from camera import Camera
                            Camera.save_snapshot(frame)
                        except Exception as e:
                            logger.error(f"Failed to save drowsiness snapshot: {e}")
            else:
                self._consecutive_drowsy = 0

        elif self._state == AlertState.DROWSY:
            if is_drowsy:
                # While still DROWSY:
                # - Keep alarm player running (it loops internally)
                # - Do NOT trigger additional alarms
                # - Do NOT save repeated snapshots
                self._consecutive_drowsy += 1
                self._player.start()  # Safe check: no-op if already running
            else:
                # Transition: DROWSY -> SAFE
                self._state = AlertState.SAFE
                logger.info("✅ [STATE CHANGE: DROWSY -> SAFE] Driver recovered/awake.")
                
                # 1. Stop the alarm
                self._player.stop()
                
                # 2. Reset all counters
                self._consecutive_drowsy = 0

    def reset(self) -> None:
        """Reset the FSM to SAFE state and stop alarm player."""
        logger.info("Resetting AlertManager state to SAFE.")
        self._state = AlertState.SAFE
        self._consecutive_drowsy = 0
        self._player.stop()

    @property
    def state(self) -> AlertState:
        """Get the current AlertState of the FSM."""
        return self._state

    @property
    def is_alarm_active(self) -> bool:
        """True if the system is currently in the DROWSY state."""
        return self._state == AlertState.DROWSY

    @property
    def consecutive_drowsy_frames(self) -> int:
        """Number of consecutive frames classified as drowsy."""
        return self._consecutive_drowsy

    @property
    def drowsy_progress(self) -> float:
        """
        Progress toward alarm trigger, as a float from 0.0 to 1.0.
        """
        if self._state == AlertState.DROWSY:
            return 1.0
        if self._frames_threshold <= 0:
            return 1.0
        return min(1.0, self._consecutive_drowsy / self._frames_threshold)

