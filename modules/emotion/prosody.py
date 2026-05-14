"""
Prosody-based emotion analyzer.

CPU-only, no ML deps. Computes pitch, pitch variance, energy, energy variance,
"speaking rate" (zero-crossing-rate proxy), and jitter from a clip and maps them
to dimensional valence/arousal/dominance using well-known prosody → affect rules:

  - High pitch + high pitch variance + high energy → high arousal
  - Slow rate + low pitch + low energy            → low arousal (calm/sad)
  - Wide pitch range + brighter spectrum          → positive valence
  - Low energy + low pitch + low variance         → negative valence (flat/sad)
  - High energy + low jitter                      → high dominance (assertive)

This is a *proxy* — accuracy is mediocre on individual clips, but the
artistic point is to give the piece a meaningful, reactive emotion signal
right now. When the Wav2Vec2 SER model lands tomorrow, this becomes the
fallback path / live (during-capture) reading.
"""

import logging

import numpy as np

from .base import BaseEmotionAnalyzer
from .types import EmotionReport

logger = logging.getLogger(__name__)


class ProsodyEmotion(BaseEmotionAnalyzer):
    def __init__(self, cfg: dict):
        self.sample_rate = cfg["audio"]["sample_rate"]
        ec = cfg.get("emotion", {})
        # Calibration scalars — tune from observed values
        self.pitch_low_hz = ec.get("pitch_low_hz", 110.0)    # quiet/sad floor
        self.pitch_high_hz = ec.get("pitch_high_hz", 280.0)  # excited ceiling
        self.energy_low = ec.get("energy_low", 0.005)
        self.energy_high = ec.get("energy_high", 0.10)
        # Smoothing applied to the dimensional output (clamps wild swings)
        self.smoothing = ec.get("smoothing", 0.0)
        self._last: EmotionReport | None = None

    def analyze(self, audio: np.ndarray, sample_rate: int) -> EmotionReport:
        rep = EmotionReport()
        if len(audio) == 0:
            rep.notes.append("empty_audio")
            return rep

        # Normalize to float32 [-1, 1]
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        else:
            x = audio.astype(np.float32)
        # Mono
        if x.ndim == 2:
            x = x.mean(axis=1)

        sr = sample_rate

        # --- 1. Framewise pitch + energy --------------------------------
        frame_len = int(0.04 * sr)   # 40ms frames
        hop = int(0.02 * sr)         # 20ms hop
        if len(x) < frame_len * 4:
            rep.notes.append("clip_too_short_for_prosody")
            return rep

        n_frames = 1 + (len(x) - frame_len) // hop
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, frame_len),
            strides=(x.strides[0] * hop, x.strides[0]),
            writeable=False,
        )

        # Energy per frame
        energy = np.sqrt(np.mean(frames * frames, axis=1))

        # Voiced mask — frames with non-trivial energy
        voiced = energy > max(self.energy_low * 0.5, 1e-4)
        voiced_count = int(np.sum(voiced))

        if voiced_count < 3:
            rep.energy_mean = float(np.mean(energy))
            rep.notes.append("no_voiced_frames")
            return rep

        # --- 2. Pitch via autocorrelation on voiced frames --------------
        pitches = _pitch_track_autocorr(frames[voiced], sr,
                                         f_min=70.0, f_max=500.0)
        pitches = pitches[pitches > 0]

        if len(pitches) < 3:
            mean_pitch = 0.0
            pitch_std = 0.0
            jitter = 0.0
        else:
            mean_pitch = float(np.mean(pitches))
            pitch_std = float(np.std(pitches))
            # Jitter — frame-to-frame fractional change in period
            periods = sr / pitches
            d = np.abs(np.diff(periods)) / periods[:-1]
            jitter = float(np.mean(d))

        # --- 3. Speaking-rate proxy via zero-crossing rate --------------
        zcr_frames = np.sum(np.diff(np.sign(frames), axis=1) != 0, axis=1) / frame_len
        speaking_rate = float(np.mean(zcr_frames[voiced]) * sr * 0.5)  # roughly Hz of voicing transitions

        # --- 4. Dimensional mapping -------------------------------------
        # Arousal: combine energy and pitch elevation
        energy_norm = _norm(float(np.mean(energy[voiced])),
                            self.energy_low, self.energy_high)
        pitch_norm = (
            _norm(mean_pitch, self.pitch_low_hz, self.pitch_high_hz)
            if mean_pitch > 0 else 0.5
        )
        arousal = 0.55 * energy_norm + 0.45 * pitch_norm
        arousal = float(np.clip(arousal, 0.0, 1.0))

        # Valence: wider pitch range + moderate energy + lower jitter → positive
        pitch_range_norm = _norm(pitch_std, 5.0, 60.0)
        jitter_pen = _norm(jitter, 0.01, 0.08)  # higher jitter → more negative
        valence = 0.5 * pitch_range_norm + 0.3 * energy_norm - 0.5 * jitter_pen
        # Center on 0 with range roughly [-1, 1]
        valence = float(np.clip((valence - 0.3) * 2.0, -1.0, 1.0))

        # Dominance: high energy + low jitter + non-trivial pitch range
        dominance = 0.6 * energy_norm + 0.2 * pitch_range_norm - 0.4 * jitter_pen
        dominance = float(np.clip(dominance + 0.3, 0.0, 1.0))

        # --- 5. Categorical label from dimensions -----------------------
        label = _label_from_dims(valence, arousal)

        rep.valence = valence
        rep.arousal = arousal
        rep.dominance = dominance
        rep.label = label
        rep.confidence = 0.5  # heuristic — backends with real models should set higher
        rep.mean_pitch_hz = mean_pitch
        rep.pitch_std_hz = pitch_std
        rep.speaking_rate = speaking_rate
        rep.energy_mean = float(np.mean(energy[voiced]))
        rep.jitter = jitter

        # Smoothing across consecutive calls (useful for live tracking)
        if self.smoothing > 0 and self._last is not None:
            a = self.smoothing
            rep.valence = a * self._last.valence + (1 - a) * rep.valence
            rep.arousal = a * self._last.arousal + (1 - a) * rep.arousal
            rep.dominance = a * self._last.dominance + (1 - a) * rep.dominance
        self._last = rep

        logger.info(
            "Emotion[prosody]: %s (V=%+.2f A=%.2f D=%.2f) | pitch=%.0fHz±%.1f rate=%.1f jitter=%.3f",
            rep.label, rep.valence, rep.arousal, rep.dominance,
            rep.mean_pitch_hz, rep.pitch_std_hz, rep.speaking_rate, rep.jitter,
        )
        return rep


# --- helpers -----------------------------------------------------------------

def _norm(x: float, lo: float, hi: float) -> float:
    """Linear normalize x∈[lo, hi] → [0, 1], clipped."""
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _pitch_track_autocorr(frames: np.ndarray, sr: int,
                          f_min: float = 70.0, f_max: float = 500.0) -> np.ndarray:
    """Per-frame autocorrelation pitch. Returns Hz array, 0 where unvoiced.

    frames: (N, L) float32.
    """
    n_frames, frame_len = frames.shape
    min_lag = max(1, int(sr / f_max))
    max_lag = min(frame_len - 1, int(sr / f_min))
    if max_lag <= min_lag + 2:
        return np.zeros(n_frames, dtype=np.float32)

    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = frames[i]
        # Remove DC
        seg = seg - seg.mean()
        # Autocorrelation via FFT-free direct method (frames are short)
        ac = np.correlate(seg, seg, mode="full")
        ac = ac[len(ac) // 2:]
        if ac[0] <= 1e-8:
            continue
        search = ac[min_lag:max_lag]
        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]
        # Confidence: peak must be a real peak relative to ac[0]
        if peak_val / (ac[0] + 1e-8) > 0.30:
            lag = peak_idx + min_lag
            out[i] = sr / lag
    return out


def _label_from_dims(valence: float, arousal: float) -> str:
    """Discrete affect label from a valence/arousal point.

    Standard Russell circumplex mapping with a 'neutral' middle bucket.
    """
    if abs(valence) < 0.20 and 0.35 < arousal < 0.65:
        return "neutral"
    if valence >= 0.20 and arousal >= 0.55:
        return "excited" if arousal > 0.75 else "happy"
    if valence >= 0.20 and arousal < 0.45:
        return "tender" if arousal < 0.30 else "warm"
    if valence < -0.20 and arousal >= 0.55:
        return "angry" if valence < -0.4 else "anxious"
    if valence < -0.20 and arousal < 0.45:
        return "sad" if valence < -0.4 else "subdued"
    return "neutral"
