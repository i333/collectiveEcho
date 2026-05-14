"""
Playback module — plays audio through the system default output.
Includes exciter conditioning: high-pass, resonance notch, soft limiter,
and high-shelf rolloff to clean up output through surface transducers.
"""

import logging
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from modules.effects import normalize_audio, to_int16

logger = logging.getLogger(__name__)


def _build_exciter_filters(sr: int, cfg: dict):
    """Build exciter conditioning filter chain (scipy SOS arrays).
    Returns list of (name, sos) tuples, or empty if scipy unavailable."""
    try:
        from scipy.signal import butter, iirnotch, tf2sos
    except ImportError:
        logger.warning("scipy not available — exciter conditioning disabled")
        return []

    filters = []

    # 1. High-pass: cut bass the exciter can't reproduce cleanly
    hp_freq = cfg.get("highpass_hz", 180)
    if hp_freq > 0:
        sos = butter(4, hp_freq, btype='high', fs=sr, output='sos')
        filters.append(("highpass_%dHz" % hp_freq, sos))

    # 2. Notch filter(s) at resonant frequencies of the mounting surface
    notches = cfg.get("notch_hz", [1200])
    notch_q = cfg.get("notch_q", 10)
    for freq in (notches if isinstance(notches, list) else [notches]):
        if 0 < freq < sr / 2:
            b, a = iirnotch(freq, notch_q, fs=sr)
            sos = tf2sos(b, a)
            filters.append(("notch_%dHz" % freq, sos))

    # 3. Low-pass to tame harsh highs from exciter
    lp_freq = cfg.get("lowpass_hz", 7000)
    if 0 < lp_freq < sr / 2:
        sos = butter(2, lp_freq, btype='low', fs=sr, output='sos')
        filters.append(("lowpass_%dHz" % lp_freq, sos))

    return filters


def _apply_filters(data: np.ndarray, filters: list) -> np.ndarray:
    """Apply a chain of SOS filters to audio."""
    if not filters:
        return data
    from scipy.signal import sosfilt
    out = data.copy()
    for name, sos in filters:
        if out.ndim == 2:
            for ch in range(out.shape[1]):
                out[:, ch] = sosfilt(sos, out[:, ch])
        else:
            out = sosfilt(sos, out)
    return out.astype(np.float32)


def _soft_limit(data: np.ndarray, threshold: float = 0.7,
                ratio: float = 4.0) -> np.ndarray:
    """Soft-knee limiter to prevent exciter distortion at peaks."""
    out = data.copy()
    mask = np.abs(out) > threshold
    out[mask] = np.sign(out[mask]) * (
        threshold + (np.abs(out[mask]) - threshold) / ratio
    )
    return out


class Playback:
    def __init__(self, cfg: dict):
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.device = cfg["audio"].get("output_device")
        self._lock = threading.Lock()

        # Build exciter conditioning filters
        exciter_cfg = cfg.get("exciter", {})
        self._exciter_enabled = exciter_cfg.get("enabled", False)
        self._exciter_filters = []
        self._limiter_threshold = exciter_cfg.get("limiter_threshold", 0.7)
        if self._exciter_enabled:
            self._exciter_filters = _build_exciter_filters(
                self.sample_rate, exciter_cfg)
            names = [n for n, _ in self._exciter_filters]
            logger.info("Exciter conditioning ON: %s, limiter=%.2f",
                        names, self._limiter_threshold)

    def _condition(self, data: np.ndarray) -> np.ndarray:
        """Apply exciter conditioning if enabled."""
        if not self._exciter_enabled:
            return data
        data = _apply_filters(data, self._exciter_filters)
        data = _soft_limit(data, self._limiter_threshold)
        return data

    def play_array(self, audio: np.ndarray, sample_rate: int | None = None,
                   blocking: bool = True):
        """Play a numpy array through speakers."""
        sr = sample_rate or self.sample_rate
        if len(audio) == 0:
            logger.warning("Playback: empty audio, nothing to play")
            return

        # Ensure float32 for normalization
        if audio.dtype == np.int16:
            data = audio.astype(np.float32) / 32768.0
        elif audio.dtype in (np.float32, np.float64):
            data = audio.astype(np.float32)
        else:
            data = normalize_audio(audio)

        # Normalize to audible level — the lav mic records very quietly
        peak = np.max(np.abs(data))
        if 0 < peak < 0.1:
            gain = 0.8 / peak
            data = data * gain
            logger.info("Playback auto-normalized: peak %.4f → gain %.1fx", peak, gain)

        # Exciter conditioning
        data = self._condition(data)

        with self._lock:
            try:
                # Use a large blocksize to avoid ALSA underruns on RPi
                sd.default.blocksize = 2048
                sd.default.latency = "high"
                sd.play(data, sr, device=self.device)
                if blocking:
                    sd.wait()
            except Exception:
                logger.exception("Playback failed")

    def play_file(self, path: str, blocking: bool = True):
        """Play a WAV/FLAC/OGG file."""
        try:
            data, sr = sf.read(path, dtype="float32")
            logger.info("Playing: %s (%.2fs)", path, len(data) / sr)
            self.play_array(data, sr, blocking=blocking)
        except Exception:
            logger.exception("Could not play file: %s", path)

    def stop(self):
        try:
            sd.stop()
        except Exception:
            pass
