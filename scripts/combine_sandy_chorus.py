"""One-off: combine every curated i_love_you_sandy clip into a single chorus
mix and write the result as a stereo WAV next to the source clips.

This is a variant of the runtime chorus tuned so all 19 voices are equally
important and individually audible:
  - All clips play at full volume (no decay — each voice matters).
  - Linear delays between clip starts (~1.5s) so each new voice has a beat
    to be heard before the next joins.
  - Alternating L/R panning with growing width for spatial separation.
  - Light alternating pitch detune (±7 cents) for chorus shimmer.
  - Shared light reverb on the final mix.

Order is chronological (oldest first → newest last) by manifest timestamp.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from modules.effects import (  # noqa: E402
    apply_delay,
    fade_in,
    fade_out,
    mono_to_stereo,
    normalize_audio,
    pitch_shift,
    simple_reverb,
    stereo_pan,
)

CURATED_DIR = REPO / "audio" / "curated" / "i_love_you_sandy"
MANIFEST = CURATED_DIR / "manifest.json"
CONFIG = REPO / "config.yaml"
OUT_WAV = CURATED_DIR / "chorus_all_sandy.wav"

# Tuning — all voices equal, clear separation at the start.
INTER_CLIP_DELAY_SEC = 1.5    # gap between successive clip starts
DETUNE_CENTS = 7.0
PAN_STEP = 0.18               # smaller step than runtime — keeps 19 voices in field
PAN_MAX = 0.85
FADE_IN_SEC = 0.05
FADE_OUT_SEC = 0.25
REVERB_DECAY = 0.20           # lighter than runtime to keep words intelligible
REVERB_DELAY_MS = 50
TRAIL_PAD_SEC = 0.3           # trailing silence kept after the last audible sample


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    sr = cfg["audio"]["sample_rate"]

    manifest = json.loads(MANIFEST.read_text())
    clusters = sorted(manifest["clusters"], key=lambda c: c["timestamp"])

    clips: list[np.ndarray] = []
    used: list[str] = []
    for c in clusters:
        path = REPO / c["curated_path"]
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        data, file_sr = sf.read(str(path), dtype="int16")
        if file_sr != sr:
            print(f"skip (sr={file_sr} != {sr}): {path}")
            continue
        if data.ndim == 2:
            data = data[:, 0]
        clips.append(data)
        used.append(path.name)

    if not clips:
        print("no clips loaded")
        return 1

    n = len(clips)
    print(f"layering {n} clips (oldest → newest, equal volume, {INTER_CLIP_DELAY_SEC}s spacing):")
    for nm in used:
        print(f"  {nm}")

    fade_in_n = int(FADE_IN_SEC * sr)
    fade_out_n = int(FADE_OUT_SEC * sr)

    processed: list[np.ndarray] = []
    for i, clip in enumerate(clips):
        audio = normalize_audio(clip)

        # Light pitch detune for chorus shimmer; first voice stays natural.
        if i > 0:
            sign = 1 if i % 2 == 1 else -1
            audio = pitch_shift(audio, sign * DETUNE_CENTS)

        audio = mono_to_stereo(audio)

        # Alternating L/R pan with widening field. First voice centered.
        if i == 0:
            audio = stereo_pan(audio, 0.0)
        else:
            pan_sign = -1 if i % 2 == 1 else 1
            pan_amt = min(PAN_STEP * ((i + 1) // 2), PAN_MAX)
            audio = stereo_pan(audio, pan_sign * pan_amt)

        # Linear time offset: clip i starts at i * INTER_CLIP_DELAY_SEC.
        if i > 0:
            audio = apply_delay(audio, int(i * INTER_CLIP_DELAY_SEC * sr))

        audio = fade_in(audio, fade_in_n)
        audio = fade_out(audio, fade_out_n)
        processed.append(audio)

    # Mix
    max_len = max(len(a) for a in processed)
    mixed = np.zeros((max_len, 2), dtype=np.float32)
    for audio in processed:
        mixed[: len(audio)] += audio

    # Shared light reverb places everyone in one room without smearing words.
    mixed = simple_reverb(mixed, sr, decay=REVERB_DECAY, delay_ms=REVERB_DELAY_MS)

    # Trim trailing silence to land cleanly.
    abs_mono = np.max(np.abs(mixed), axis=1)
    nonzero = np.where(abs_mono > 0.001)[0]
    if len(nonzero):
        end = min(len(mixed), nonzero[-1] + int(TRAIL_PAD_SEC * sr))
        mixed = mixed[:end]

    # Peak normalize to -1.9 dBFS so the layered mix doesn't clip.
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0:
        mixed = mixed / peak * 0.8

    dur = len(mixed) / sr
    print(f"mix: {dur:.2f}s, peak_pre_norm={peak:.3f}, shape={mixed.shape}")

    sf.write(str(OUT_WAV), mixed, sr, subtype="PCM_16")
    print(f"wrote: {OUT_WAV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
