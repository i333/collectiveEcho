"""
Audio effects — layering, volume scaling, delay, fade, simple reverb,
pitch detuning, stereo panning, low-pass rolloff.

All operate on float32 numpy arrays normalized to [-1, 1].
Mono arrays are 1-D; stereo arrays are 2-D with shape (N, 2).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

PHI = 1.618033988749895  # golden ratio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convert int16 to float32 [-1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def fade_in(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply linear fade-in."""
    if n_samples <= 0 or len(audio) == 0:
        return audio
    n = min(n_samples, len(audio))
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if out.ndim == 2:
        ramp = ramp[:, np.newaxis]
    out[:n] *= ramp
    return out


def fade_out(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply linear fade-out."""
    if n_samples <= 0 or len(audio) == 0:
        return audio
    n = min(n_samples, len(audio))
    out = audio.copy()
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    if out.ndim == 2:
        ramp = ramp[:, np.newaxis]
    out[-n:] *= ramp
    return out


def apply_volume(audio: np.ndarray, volume: float) -> np.ndarray:
    return audio * volume


def apply_delay(audio: np.ndarray, delay_samples: int) -> np.ndarray:
    """Prepend silence to shift audio forward in time."""
    if delay_samples <= 0:
        return audio
    if audio.ndim == 2:
        silence = np.zeros((delay_samples, audio.shape[1]), dtype=audio.dtype)
    else:
        silence = np.zeros(delay_samples, dtype=audio.dtype)
    return np.concatenate([silence, audio])


def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono (N,) to stereo (N, 2)."""
    if audio.ndim == 2:
        return audio
    return np.column_stack([audio, audio])


def stereo_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    """Apply stereo panning. pan: -1.0=full left, 0.0=center, 1.0=full right.
    Uses constant-power panning (sin/cos law)."""
    stereo = mono_to_stereo(audio)
    # Map pan [-1, 1] to angle [0, pi/2]
    angle = (pan + 1.0) * 0.25 * np.pi
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    stereo[:, 0] *= left_gain
    stereo[:, 1] *= right_gain
    return stereo


def pitch_shift(audio: np.ndarray, cents: float) -> np.ndarray:
    """Pitch-shift by resampling. Cheap but changes duration slightly.
    cents: +/- hundredths of a semitone (e.g. +7 = slightly sharper)."""
    if abs(cents) < 0.5:
        return audio
    # Ratio: to shift up, we resample to fewer samples then stretch back
    ratio = 2.0 ** (cents / 1200.0)
    n_out = int(len(audio) / ratio)
    if n_out < 2:
        return audio
    x_old = np.linspace(0, 1, len(audio), dtype=np.float32)
    x_new = np.linspace(0, 1, n_out, dtype=np.float32)
    if audio.ndim == 2:
        shifted = np.column_stack([
            np.interp(x_new, x_old, audio[:, ch])
            for ch in range(audio.shape[1])
        ])
    else:
        shifted = np.interp(x_new, x_old, audio)
    return shifted.astype(np.float32)


def lowpass_ema(audio: np.ndarray, sample_rate: int,
               cutoff_hz: float) -> np.ndarray:
    """Single-pole IIR low-pass (EMA). Vectorized via scipy.signal.lfilter.

    Equivalent to y[n] = y[n-1] + alpha * (x[n] - y[n-1]), but ~50x faster
    than the Python loop it replaced.
    """
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    try:
        from scipy.signal import lfilter
        b = [alpha]
        a = [1.0, alpha - 1.0]
        if audio.ndim == 2:
            return np.column_stack([
                lfilter(b, a, audio[:, ch]).astype(np.float32)
                for ch in range(audio.shape[1])
            ])
        return lfilter(b, a, audio).astype(np.float32)
    except ImportError:
        # Pure-Python fallback (slow but correct)
        out = audio.copy()
        if out.ndim == 2:
            for ch in range(out.shape[1]):
                for i in range(1, len(out)):
                    out[i, ch] = out[i - 1, ch] + alpha * (out[i, ch] - out[i - 1, ch])
        else:
            for i in range(1, len(out)):
                out[i] = out[i - 1] + alpha * (out[i] - out[i - 1])
        return out


def lowpass_ema_fast(audio: np.ndarray, sample_rate: int,
                     cutoff_hz: float) -> np.ndarray:
    """Vectorized low-pass via scipy if available, falls back to EMA loop."""
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return audio
    try:
        from scipy.signal import butter, sosfilt
        # 2nd order Butterworth — smooth rolloff, very cheap
        sos = butter(2, cutoff_hz, btype='low', fs=sample_rate, output='sos')
        if audio.ndim == 2:
            return np.column_stack([
                sosfilt(sos, audio[:, ch]).astype(np.float32)
                for ch in range(audio.shape[1])
            ])
        return sosfilt(sos, audio).astype(np.float32)
    except ImportError:
        return lowpass_ema(audio, sample_rate, cutoff_hz)


def noise_gate(audio: np.ndarray, sample_rate: int,
               threshold_rms: float = 0.005,
               frame_ms: float = 20,
               attack_ms: float = 5,
               release_ms: float = 50) -> np.ndarray:
    """
    Simple noise gate — attenuate frames below threshold.
    Smooths transitions with attack/release envelopes.
    """
    if len(audio) == 0:
        return audio

    samples = normalize_audio(audio) if audio.dtype == np.int16 else audio.copy()
    # Work on mono signal for gate detection
    if samples.ndim == 2:
        detect = samples[:, 0]
    else:
        detect = samples

    frame_len = int(frame_ms / 1000.0 * sample_rate)
    if frame_len < 1:
        return samples

    n_frames = len(detect) // frame_len
    if n_frames == 0:
        return samples

    trimmed = detect[:n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

    gate = np.where(frame_rms >= threshold_rms, 1.0, 0.02).astype(np.float32)

    attack_frames = max(1, int(attack_ms / frame_ms))
    release_frames = max(1, int(release_ms / frame_ms))
    # Asymmetric attack/release smoother. Vectorized via a one-pass loop in
    # numpy — gate length is small (frames at 20ms), so this is fast enough,
    # but we avoid the per-sample math by working at frame resolution.
    smoothed = np.copy(gate)
    attack_step = 1.0 / attack_frames
    release_step = 1.0 / release_frames
    for i in range(1, len(smoothed)):
        diff = smoothed[i] - smoothed[i - 1]
        if diff > 0:
            smoothed[i] = smoothed[i - 1] + min(diff, attack_step)
        else:
            smoothed[i] = smoothed[i - 1] + max(diff, -release_step)

    gate_samples = np.repeat(smoothed, frame_len)
    if samples.ndim == 2:
        gate_samples = gate_samples[:, np.newaxis]
    out = samples[:len(gate_samples)] * gate_samples

    if len(samples) > len(gate_samples):
        tail = samples[len(gate_samples):]
        if samples.ndim == 2:
            out = np.concatenate([out, tail * smoothed[-1]])
        else:
            out = np.concatenate([out, tail * smoothed[-1]])

    return out


def simple_reverb(audio: np.ndarray, sample_rate: int,
                  decay: float = 0.3, delay_ms: float = 60) -> np.ndarray:
    """Simple multi-tap comb-filter reverb. Works on mono or stereo."""
    delay_samples = int(delay_ms / 1000.0 * sample_rate)
    if delay_samples <= 0 or decay <= 0:
        return audio

    out = audio.copy()
    for tap in range(1, 5):
        offset = delay_samples * tap
        gain = decay ** tap
        if offset < len(out):
            end = min(len(audio), len(out) - offset)
            out[offset: offset + end] += audio[:end] * gain

    return out


def layer_clips(clips: list[np.ndarray], sample_rate: int,
                cfg_playback: dict) -> np.ndarray:
    """
    Layer multiple clips with spatial separation techniques:
    - Golden-ratio delay offsets (natural, non-repeating spacing)
    - Stereo panning (alternating L/R with increasing width)
    - Pitch detuning (±7 cents, alternating sharp/flat)
    - Low-pass rolloff on older layers (simulates distance)
    - Softer volume decay (0.85) — EQ/panning create perceived distance
    - Transient softening on older layers (reduces consonant smearing)
    - Shared reverb on final mix (places all voices in one "room")

    clips[0] = oldest, clips[-1] = newest.
    Newest is loudest, centered, full-frequency.
    """
    if not clips:
        return np.array([], dtype=np.float32)

    n = len(clips)
    fade_in_sec = cfg_playback.get("fade_in_sec", 0.1)
    fade_out_sec = cfg_playback.get("fade_out_sec", 0.3)
    reverb_on = cfg_playback.get("reverb_enabled", False)
    reverb_decay = cfg_playback.get("reverb_decay", 0.3)
    reverb_delay = cfg_playback.get("reverb_delay_ms", 60)

    # Tuning parameters
    base_delay_sec = cfg_playback.get("layer_delay_sec", 0.10)
    volume_decay = cfg_playback.get("layer_volume_decay", 0.85)
    detune_cents = cfg_playback.get("detune_cents", 7.0)
    lp_oldest_hz = cfg_playback.get("lp_oldest_hz", 3500)
    lp_newest_hz = cfg_playback.get("lp_newest_hz", 8000)
    transient_soften_ms = cfg_playback.get("transient_soften_ms", 20)

    # Panning positions: center for newest, alternating L/R for older
    # Layer 0 (newest) = 0.0 (center)
    # Layer 1 = -0.30 (left), layer 2 = +0.30 (right)
    # Layer 3 = -0.60, layer 4 = +0.60, etc.
    pan_step = 0.30
    max_pan = 0.90

    fade_in_n = int(fade_in_sec * sample_rate)
    fade_out_n = int(fade_out_sec * sample_rate)
    soften_n = int(transient_soften_ms / 1000.0 * sample_rate)

    processed = []
    for i, clip in enumerate(clips):
        audio = normalize_audio(clip)
        age = n - 1 - i  # 0 for newest, higher for older

        # --- Pitch detuning: alternate sharp/flat, newest stays natural ---
        if age > 0:
            sign = 1 if age % 2 == 1 else -1
            audio = pitch_shift(audio, sign * detune_cents)

        # --- Convert to stereo for panning ---
        audio = mono_to_stereo(audio)

        # --- Stereo panning ---
        if age > 0:
            pan_amount = min(pan_step * age, max_pan)
            pan_sign = -1 if age % 2 == 1 else 1  # odd=left, even=right
            audio = stereo_pan(audio, pan_sign * pan_amount)
        else:
            # Newest stays centered (stereo_pan at 0.0)
            audio = stereo_pan(audio, 0.0)

        # --- Low-pass rolloff: older = darker (simulates distance) ---
        if age > 0 and n > 1:
            # Interpolate cutoff: newest=lp_newest_hz, oldest=lp_oldest_hz
            t = age / max(1, n - 1)
            cutoff = lp_newest_hz + t * (lp_oldest_hz - lp_newest_hz)
            audio = lowpass_ema_fast(audio, sample_rate, cutoff)

        # --- Volume: newest=1.0, older decays ---
        vol = volume_decay ** age
        audio = apply_volume(audio, vol)

        # --- Golden-ratio delay offsets ---
        if age > 0:
            # Delay sequence: base * phi^0, base * phi^1, base * phi^2, ...
            delay_sec = base_delay_sec * (PHI ** (age - 1))
            delay_samples = int(delay_sec * sample_rate)
            audio = apply_delay(audio, delay_samples)

        # --- Transient softening on older layers ---
        if age >= 2 and soften_n > 0:
            audio = fade_in(audio, soften_n)

        # --- Fades ---
        audio = fade_in(audio, fade_in_n)
        audio = fade_out(audio, fade_out_n)

        processed.append(audio)

    # --- Mix all layers into one stereo buffer ---
    max_len = max(len(a) for a in processed)
    mixed = np.zeros((max_len, 2), dtype=np.float32)
    for audio in processed:
        mixed[:len(audio)] += audio

    # --- Shared reverb on the final mix ---
    if reverb_on:
        mixed = simple_reverb(mixed, sample_rate,
                              decay=reverb_decay, delay_ms=reverb_delay)

    # --- Normalize to audible level ---
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.8
        logger.debug("Normalized layered mix (peak was %.4f)", peak)

    return mixed
