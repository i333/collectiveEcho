"""
Clip enhancement — clean and balance a recording before it joins the chorus.

Steps (applied in order, all CPU-only):
  1. DC offset removal
  2. High-pass at 60Hz (removes rumble, AC hum below)
  3. Hum notch (50/60Hz) — narrow notch where electrical hum lives
  4. Spectral-subtraction denoise (estimates noise from the first 200ms of
     low-energy frames and subtracts it from the speech spectrum)
  5. Loudness normalization to target RMS (so layered clips balance)
  6. Soft-knee compression (tames peaks without flattening dynamics)
  7. Light de-essing (notch ~6kHz when sibilance energy spikes)

We don't apply a hard limiter here — the playback module's exciter chain has
its own limiter. The goal here is consistency across recordings, not maximum
loudness.

Returns the cleaned audio AND a small dict of measured-quality fields the
QualityAnalyzer can fold into its report (SNR estimate, spectral tilt, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancementMetrics:
    snr_db_before: float = 0.0
    snr_db_after: float = 0.0
    rms_before: float = 0.0
    rms_after: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_tilt_db_per_khz: float = 0.0
    noise_floor_db: float = -60.0
    applied: list[str] = None

    def __post_init__(self):
        if self.applied is None:
            self.applied = []


class ClipEnhancer:
    def __init__(self, cfg: dict):
        ecfg = cfg.get("enhancement", {}) or {}
        self.enabled = ecfg.get("enabled", True)
        # High-pass cutoff (Hz). 60Hz is good for voice — preserves vocal warmth.
        self.highpass_hz = ecfg.get("highpass_hz", 60)
        # Hum notch frequency (50 in EU, 60 in US). 0 disables.
        self.hum_hz = ecfg.get("hum_hz", 60)
        self.hum_q = ecfg.get("hum_q", 30)
        # Denoise strength multiplier (0 disables; 1.0 standard; up to 2.0 aggressive)
        self.denoise_strength = ecfg.get("denoise_strength", 1.0)
        # Target RMS for normalization. 0.10 = ~−20dBFS, conservative.
        self.target_rms = ecfg.get("target_rms", 0.10)
        self.max_gain_db = ecfg.get("max_gain_db", 18.0)
        # Compressor
        self.comp_threshold = ecfg.get("comp_threshold", 0.5)
        self.comp_ratio = ecfg.get("comp_ratio", 2.5)
        # De-esser
        self.deess_freq_hz = ecfg.get("deess_freq_hz", 6500)
        self.deess_threshold = ecfg.get("deess_threshold", 0.4)
        # Sample rate (inferred at call time but cached after first call)
        self._sample_rate = None
        self._sos_cache: dict[str, np.ndarray] = {}

    # --- public API ------------------------------------------------------

    def enhance(self, audio: np.ndarray, sample_rate: int
                ) -> tuple[np.ndarray, EnhancementMetrics]:
        """Run the enhancement chain. Returns (audio_float32, metrics).

        Input may be int16 or float; output is float32 in [-1, 1].
        """
        metrics = EnhancementMetrics()
        if len(audio) == 0:
            return audio.astype(np.float32) if audio.dtype != np.float32 else audio, metrics

        # Float, mono
        x = _to_float_mono(audio)
        metrics.rms_before = float(np.sqrt(np.mean(x ** 2)))

        # --- pre-measure ---
        noise_floor_lin, snr_before = _estimate_snr(x, sample_rate)
        metrics.snr_db_before = snr_before
        metrics.noise_floor_db = (
            20 * np.log10(noise_floor_lin) if noise_floor_lin > 0 else -120.0
        )

        if not self.enabled:
            metrics.rms_after = metrics.rms_before
            metrics.snr_db_after = metrics.snr_db_before
            return x, metrics

        # 1. DC removal
        x = x - np.mean(x)
        metrics.applied.append("dc_removal")

        # 2. High-pass
        if self.highpass_hz > 0:
            x = self._biquad_filter(x, sample_rate, "highpass", self.highpass_hz)
            metrics.applied.append(f"hp_{self.highpass_hz}Hz")

        # 3. Hum notch
        if self.hum_hz > 0:
            x = self._iir_notch(x, sample_rate, self.hum_hz, self.hum_q)
            metrics.applied.append(f"notch_{self.hum_hz}Hz")

        # 4. Spectral-subtraction denoise
        if self.denoise_strength > 0 and noise_floor_lin > 1e-6:
            x = _spectral_subtraction_denoise(
                x, sample_rate,
                noise_floor=noise_floor_lin,
                strength=self.denoise_strength,
            )
            metrics.applied.append(f"denoise×{self.denoise_strength:.1f}")

        # 5. Loudness normalize
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms > 1e-6:
            gain = self.target_rms / rms
            gain_db = 20 * np.log10(gain)
            # Cap the gain so we don't amplify near-silent recordings into noise
            if gain_db > self.max_gain_db:
                gain = 10 ** (self.max_gain_db / 20)
                gain_db = self.max_gain_db
            x = x * gain
            metrics.applied.append(f"loudness {gain_db:+.1f}dB")

        # 6. Soft-knee compression
        x = _soft_compress(x, self.comp_threshold, self.comp_ratio)
        metrics.applied.append(f"comp {self.comp_threshold:.2f}:{self.comp_ratio:.1f}")

        # 7. Light de-essing (single-band)
        if self.deess_freq_hz < sample_rate / 2:
            x = _light_de_ess(x, sample_rate,
                              freq_hz=self.deess_freq_hz,
                              threshold=self.deess_threshold)
            metrics.applied.append("de_ess")

        # Safety: hard ceiling at 0.99 to avoid integer overflow on save
        peak = float(np.max(np.abs(x)))
        if peak > 0.99:
            x = x / peak * 0.99
            metrics.applied.append("peak_safety")

        x = x.astype(np.float32)

        # --- post-measure ---
        metrics.rms_after = float(np.sqrt(np.mean(x ** 2)))
        _, snr_after = _estimate_snr(x, sample_rate)
        metrics.snr_db_after = snr_after
        metrics.spectral_centroid_hz, metrics.spectral_tilt_db_per_khz = \
            _spectral_features(x, sample_rate)

        logger.info(
            "Enhance: SNR %.1f→%.1fdB | RMS %.4f→%.4f | centroid=%.0fHz tilt=%+.1fdB/kHz | %s",
            metrics.snr_db_before, metrics.snr_db_after,
            metrics.rms_before, metrics.rms_after,
            metrics.spectral_centroid_hz, metrics.spectral_tilt_db_per_khz,
            ", ".join(metrics.applied),
        )
        return x, metrics

    # --- helpers (filters built once per sample-rate per type) -----------

    def _sos_for(self, sample_rate: int, kind: str, freq: float,
                 order: int = 2) -> np.ndarray:
        key = f"{kind}_{sample_rate}_{freq:.0f}_{order}"
        if key in self._sos_cache:
            return self._sos_cache[key]
        from scipy.signal import butter
        nyq = sample_rate / 2
        if freq >= nyq:
            freq = nyq * 0.95
        sos = butter(order, freq, btype=kind, fs=sample_rate, output="sos")
        self._sos_cache[key] = sos
        return sos

    def _biquad_filter(self, x: np.ndarray, sample_rate: int,
                        kind: str, freq: float) -> np.ndarray:
        from scipy.signal import sosfilt
        sos = self._sos_for(sample_rate, kind, freq)
        return sosfilt(sos, x).astype(np.float32)

    def _iir_notch(self, x: np.ndarray, sample_rate: int,
                    freq_hz: float, q: float) -> np.ndarray:
        from scipy.signal import iirnotch, lfilter
        b, a = iirnotch(freq_hz, q, fs=sample_rate)
        return lfilter(b, a, x).astype(np.float32)


# --- module-level helpers ------------------------------------------------

def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        x = audio.astype(np.float32) / 32768.0
    else:
        x = audio.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def _estimate_snr(x: np.ndarray, sample_rate: int,
                  frame_ms: float = 20.0) -> tuple[float, float]:
    """Estimate noise floor (linear RMS) and SNR (dB) from frame energies.

    Treats the quietest 10% of frames as noise and the loudest 30% as speech.
    Returns (noise_floor_linear, snr_db).
    """
    if len(x) < int(0.1 * sample_rate):
        return 1e-6, 0.0
    frame_len = max(1, int(frame_ms / 1000.0 * sample_rate))
    n_frames = len(x) // frame_len
    if n_frames < 5:
        return 1e-6, 0.0
    frames = x[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    sorted_rms = np.sort(frame_rms)
    quiet = sorted_rms[: max(1, n_frames // 10)]
    loud = sorted_rms[-max(1, int(n_frames * 0.3)):]
    noise = float(np.mean(quiet))
    speech = float(np.mean(loud))
    if noise <= 1e-9:
        snr_db = 60.0
    else:
        snr_db = 20 * float(np.log10(speech / noise))
    return noise, snr_db


def _spectral_features(x: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Spectral centroid (Hz) and tilt (dB/kHz, negative = darker)."""
    if len(x) < 1024:
        return 0.0, 0.0
    n = 2 ** int(np.ceil(np.log2(min(8192, len(x)))))
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag = spec / (np.sum(spec) + 1e-12)
    centroid = float(np.sum(freqs * mag))
    # Tilt: linear regression of log-magnitude vs frequency
    eps = 1e-9
    log_mag = 20 * np.log10(spec + eps)
    # Skip DC bin
    f = freqs[1:]
    m = log_mag[1:]
    if len(f) < 8:
        return centroid, 0.0
    a, b = np.polyfit(f, m, 1)
    tilt_db_per_khz = float(a * 1000.0)
    return centroid, tilt_db_per_khz


def _spectral_subtraction_denoise(x: np.ndarray, sample_rate: int,
                                    noise_floor: float,
                                    strength: float = 1.0,
                                    frame_ms: float = 32.0,
                                    overlap: float = 0.5) -> np.ndarray:
    """STFT-based spectral subtraction. Cheap, effective for stationary noise.

    Subtracts a magnitude estimate scaled by `strength`; leaves phase intact.
    Floor at 0.1× input magnitude to avoid musical noise (the gurgling
    artifact of overly aggressive subtraction).
    """
    frame = max(64, int(frame_ms / 1000.0 * sample_rate))
    hop = max(1, int(frame * (1.0 - overlap)))
    window = np.hanning(frame).astype(np.float32)

    # Pad so we can fully reconstruct
    pad = frame
    xp = np.concatenate([np.zeros(pad, dtype=np.float32), x,
                         np.zeros(pad + frame, dtype=np.float32)])
    n_frames = 1 + (len(xp) - frame) // hop
    out = np.zeros_like(xp)
    wsum = np.zeros_like(xp)

    # Estimate noise spectrum from the lowest-energy 10% of frames in the input.
    # noise_floor already gives us the RMS level; we need a spectral shape.
    # Run a fast pass to find quiet frames.
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = xp[i * hop: i * hop + frame] * window
        energy[i] = float(np.mean(seg * seg))
    quiet_idx = np.argsort(energy)[: max(1, n_frames // 10)]
    noise_mag = np.zeros(frame // 2 + 1, dtype=np.float32)
    for qi in quiet_idx:
        seg = xp[qi * hop: qi * hop + frame] * window
        noise_mag += np.abs(np.fft.rfft(seg))
    noise_mag /= max(1, len(quiet_idx))
    noise_mag *= strength

    # Apply subtraction
    for i in range(n_frames):
        seg = xp[i * hop: i * hop + frame] * window
        sp = np.fft.rfft(seg)
        mag = np.abs(sp)
        phase = np.angle(sp)
        cleaned = np.maximum(mag - noise_mag, mag * 0.1)
        seg_clean = np.fft.irfft(cleaned * np.exp(1j * phase), n=frame).astype(np.float32)
        out[i * hop: i * hop + frame] += seg_clean * window
        wsum[i * hop: i * hop + frame] += window * window

    wsum = np.where(wsum > 1e-6, wsum, 1.0)
    out = out / wsum
    # Strip padding
    return out[pad: pad + len(x)].astype(np.float32)


def _soft_compress(x: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    """Soft-knee compressor. Operates sample-wise on the envelope (RMS-ish).
    For an art piece we don't need pristine timing — this is fine.
    """
    out = x.copy()
    abs_x = np.abs(out)
    over = abs_x > threshold
    out[over] = np.sign(out[over]) * (threshold + (abs_x[over] - threshold) / ratio)
    return out


def _light_de_ess(x: np.ndarray, sample_rate: int, freq_hz: float,
                  threshold: float) -> np.ndarray:
    """Detect sibilance band energy; attenuate sample-by-sample when it spikes.

    Cheap dual-stream approach: split into low (<freq_hz) and high (>=freq_hz),
    measure high-band envelope, attenuate the high stream when its envelope
    exceeds threshold, then recombine.
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError:
        return x
    sos_low = butter(2, freq_hz, btype="low", fs=sample_rate, output="sos")
    sos_high = butter(2, freq_hz, btype="high", fs=sample_rate, output="sos")
    low = sosfilt(sos_low, x).astype(np.float32)
    high = sosfilt(sos_high, x).astype(np.float32)
    # High-band envelope via smoothed absolute value (5ms attack)
    win = max(1, int(0.005 * sample_rate))
    abs_high = np.abs(high)
    # Fast moving average via cumulative sum
    csum = np.cumsum(abs_high, dtype=np.float64)
    csum = np.concatenate([[0], csum])
    env = ((csum[win:] - csum[:-win]) / win).astype(np.float32)
    env = np.pad(env, (len(high) - len(env), 0), mode="edge")
    # Gain reduction where env exceeds threshold
    over = env > threshold
    gr = np.ones_like(env)
    gr[over] = threshold / np.maximum(env[over], 1e-6)
    return low + high * gr
