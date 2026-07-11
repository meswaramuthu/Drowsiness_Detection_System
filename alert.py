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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
# Alert Manager
# ──────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Manages the drowsiness alerting logic:
      - Counts consecutive drowsy frames.
      - Triggers the alarm only after the threshold is exceeded.
      - Respects a cooldown period between alarms.
      - Plays alarm sound asynchronously (non-blocking).

    Usage:
        alert = AlertManager()
        # In the frame loop:
        alert.update(is_drowsy=True)
        if alert.is_alarm_active:
            draw_alarm_overlay()
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
        self._consecutive_drowsy: int = 0
        self._is_alarm_active: bool = False
        self._last_alarm_time: float = 0.0
        self._alarm_thread: threading.Thread | None = None

        # Ensure the alarm sound file exists
        self._ensure_sound_file()

    def _ensure_sound_file(self) -> None:
        """Generate the alarm WAV file if it doesn't exist on disk."""
        if not os.path.isfile(self._sound_path):
            logger.warning(f"Alarm sound not found at: {self._sound_path}")
            generate_alarm_wav(self._sound_path)

    # ── Public Interface ──────────────────────────────────────────────────

    def update(self, is_drowsy: bool) -> None:
        """
        Call once per frame to update the alert state.

        Args:
            is_drowsy: True if the current frame was classified as drowsy.
        """
        if is_drowsy:
            self._consecutive_drowsy += 1
        else:
            self._consecutive_drowsy = 0
            self._is_alarm_active = False

        # Check if we should trigger the alarm
        if self._consecutive_drowsy >= self._frames_threshold:
            self._trigger_alarm()

    def reset(self) -> None:
        """Reset all alert state (e.g., when restarting monitoring)."""
        self._consecutive_drowsy = 0
        self._is_alarm_active = False

    @property
    def is_alarm_active(self) -> bool:
        """True if an alarm is currently triggered and active."""
        return self._is_alarm_active

    @property
    def consecutive_drowsy_frames(self) -> int:
        """Number of consecutive frames classified as drowsy."""
        return self._consecutive_drowsy

    @property
    def drowsy_progress(self) -> float:
        """
        Progress toward alarm trigger, as a float from 0.0 to 1.0.
        Useful for rendering a "building up" visual indicator.
        """
        if self._frames_threshold <= 0:
            return 1.0
        return min(1.0, self._consecutive_drowsy / self._frames_threshold)

    # ── Private Methods ───────────────────────────────────────────────────

    def _trigger_alarm(self) -> None:
        """Trigger the alarm if cooldown has elapsed."""
        now = time.time()
        elapsed = now - self._last_alarm_time

        self._is_alarm_active = True

        if elapsed < self._cooldown_seconds:
            return  # Still in cooldown — skip sound playback

        self._last_alarm_time = now
        logger.warning(
            f"⚠️  DROWSINESS ALARM — {self._consecutive_drowsy} consecutive frames!"
        )

        # Play the alarm sound in a separate thread to avoid blocking the frame loop
        if self._alarm_thread is None or not self._alarm_thread.is_alive():
            self._alarm_thread = threading.Thread(
                target=_play_sound_blocking,
                args=(self._sound_path,),
                daemon=True,
            )
            self._alarm_thread.start()
