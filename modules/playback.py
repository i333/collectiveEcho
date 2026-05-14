"""
Playback module — plays audio through the system default output.
Includes exciter conditioning: high-pass, resonance notch, soft limiter,
and high-shelf rolloff to clean up output through surface transducers.
"""

import logging
import threading

import numpy as np
import soundfile as sf

# sounddevice import is deferred to the play methods — PortAudio dep, see
# modules/audio_input.py for the same pattern.
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
    """Soft-knee limiter — early gentle reduction above threshold."""
    out = data.copy()
    mask = np.abs(out) > threshold
    out[mask] = np.sign(out[mask]) * (
        threshold + (np.abs(out[mask]) - threshold) / ratio
    )
    return out


def _brickwall_limit(data: np.ndarray, sample_rate: int,
                      ceiling: float = 0.95,
                      attack_ms: float = 1.0,
                      release_ms: float = 50.0) -> np.ndarray:
    """Smoothed brickwall limiter — hard ceiling on instantaneous peaks.

    Computes a frame-wise gain envelope: where the signal exceeds the ceiling,
    we attenuate by `ceiling / peak`. Envelope is smoothed (fast attack, slow
    release) so the gain reduction doesn't introduce clicks. Final output
    samples are also hard-clipped to ceiling as a safety floor.
    """
    if len(data) == 0:
        return data
    out = data.astype(np.float32, copy=True)
    if out.ndim == 2:
        abs_x = np.max(np.abs(out), axis=1)
    else:
        abs_x = np.abs(out)

    # Instantaneous gain reduction needed
    gr = np.where(abs_x > ceiling, ceiling / np.maximum(abs_x, 1e-9), 1.0)

    # Asymmetric envelope smoother — fast attack (gain reduction sets in
    # quickly), slow release (gain returns gradually to avoid pumping).
    attack_a = np.exp(-1.0 / max(1.0, attack_ms / 1000.0 * sample_rate))
    release_a = np.exp(-1.0 / max(1.0, release_ms / 1000.0 * sample_rate))
    env = np.empty_like(gr)
    env[0] = gr[0]
    for i in range(1, len(gr)):
        if gr[i] < env[i - 1]:    # gain reduction tightening — attack
            env[i] = attack_a * env[i - 1] + (1 - attack_a) * gr[i]
        else:                       # releasing
            env[i] = release_a * env[i - 1] + (1 - release_a) * gr[i]

    if out.ndim == 2:
        out *= env[:, None]
    else:
        out *= env

    # Safety hard clip — should never engage in normal use
    np.clip(out, -ceiling, ceiling, out=out)
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
        self._limiter_threshold = exciter_cfg.get("limiter_threshold", 0.5)
        self._limiter_ratio = exciter_cfg.get("limiter_ratio", 8.0)
        if self._exciter_enabled:
            self._exciter_filters = _build_exciter_filters(
                self.sample_rate, exciter_cfg)
            names = [n for n, _ in self._exciter_filters]
            logger.info("Exciter conditioning ON: %s, limiter=%.2f:%.1f",
                        names, self._limiter_threshold, self._limiter_ratio)

        # Master output stage — last safety net before the DAC.
        # `master_gain` cuts overall headroom so the exciter isn't driven
        # close to its mechanical limit. `master_ceiling` is the absolute
        # peak ceiling enforced by the brickwall limiter.
        playback_cfg = cfg.get("playback", {}) or {}
        self.master_gain = float(playback_cfg.get("master_gain", 0.55))
        self.master_ceiling = float(playback_cfg.get("master_ceiling", 0.92))
        self.master_limiter_enabled = bool(playback_cfg.get("master_limiter_enabled", True))
        logger.info("Output stage: master_gain=%.2f ceiling=%.2f brickwall=%s",
                    self.master_gain, self.master_ceiling, self.master_limiter_enabled)

    def _condition(self, data: np.ndarray) -> np.ndarray:
        """Apply exciter conditioning if enabled."""
        if not self._exciter_enabled:
            return data
        data = _apply_filters(data, self._exciter_filters)
        data = _soft_limit(data, self._limiter_threshold, self._limiter_ratio)
        return data

    def _master_stage(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Final output stage: master gain → brickwall limiter.

        This is the LAST thing that touches audio before the DAC. The
        brickwall guarantees the exciter never sees |sample| > ceiling.
        """
        data = data * self.master_gain
        if self.master_limiter_enabled:
            data = _brickwall_limit(data, sr,
                                     ceiling=self.master_ceiling,
                                     attack_ms=1.0, release_ms=50.0)
        # Log the actual peak hitting the DAC (helps diagnose distortion)
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
        logger.info("Output peak=%.3f rms=%.3f (after master×%.2f, ceil=%.2f)",
                    peak, rms, self.master_gain, self.master_ceiling)
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

        # Auto-boost only for very quiet raw V1 captures. V2 clips are
        # already loudness-normalized by the enhancer (RMS≈0.10), so this
        # mostly no-ops post-V2 — keeping it for backward compat with
        # any older fallback clips.
        peak = np.max(np.abs(data))
        if 0 < peak < 0.1:
            gain = 0.8 / peak
            data = data * gain
            logger.info("Playback auto-normalized: peak %.4f → gain %.1fx", peak, gain)

        # Exciter conditioning (filters + soft pre-limiter)
        data = self._condition(data)

        # Final output stage — master gain + brickwall ceiling.
        # This is the single source of truth for "loudness reaching the DAC".
        data = self._master_stage(data, sr)

        with self._lock:
            try:
                import sounddevice as sd
                # Probe the output device's preferred rate; resample our buffer
                # to match if needed. The AB13X USB DAC only supports 48000;
                # our pipeline operates at 16000.
                native_sr = sr
                if self.device is not None:
                    try:
                        info = sd.query_devices(self.device, kind="output")
                        native_sr = int(info["default_samplerate"])
                    except Exception:
                        pass
                if native_sr != sr:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(native_sr, sr)
                    up = native_sr // g
                    down = sr // g
                    if data.ndim == 2:
                        data = np.column_stack([
                            resample_poly(data[:, ch], up, down)
                            for ch in range(data.shape[1])
                        ]).astype(np.float32)
                    else:
                        data = resample_poly(data, up, down).astype(np.float32)
                    sr = native_sr
                    # Polyphase resampling can produce sample-level overshoot
                    # above the brickwall ceiling. Hard-clip as a safety floor.
                    np.clip(data, -self.master_ceiling, self.master_ceiling, out=data)

                # Use a large blocksize to avoid ALSA underruns
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
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
