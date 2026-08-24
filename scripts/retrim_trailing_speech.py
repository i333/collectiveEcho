#!/usr/bin/env python3
"""Re-trim ily clips that leak words before or after the "I love you"
phrase — e.g. a trailing "Sa…" of "Sandy" at the end, or a leading
"Sandy," at the start. Both directions handled in one pass.

Strategy: re-run Vosk on each clip with word-level timings. Locate the
"i love you" span (or its prefix when Vosk misses "i"). Cut to
``[first_phrase_word.start - pad,  you.end + pad]``. Pure-audio energy
heuristics can't tell mid-phrase pauses from extra words — Vosk word
timing can.

Read-only by default. Pass ``--apply`` to overwrite (originals backed up
alongside the file as ``<name>.bak.<timestamp>``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf


POST_YOU_PAD_MS = 50   # keep this much audio after "you" ends for natural decay
PRE_PHRASE_PAD_MS = 0  # cut at phrase start exactly — no pre-pad. The 50ms
                       # we used to keep here kept leaking "the"/"and"/etc.
MIN_OVERRUN_MS = 80    # flag if clip extends ≥ this much past you.end
MIN_LEADIN_MS = 50     # flag if clip starts ≥ this much before first phrase word
I_VARIANTS = {"i", "im", "ive", "ill"}
LOVE_VARIANTS = {"love", "loved", "luv", "lovin", "loving"}


@dataclass
class TrimResult:
    path: str
    duration_sec: float
    cut_start_sec: float
    cut_end_sec: float | None        # None = keep original end
    new_duration_sec: float | None   # None = no trim needed
    phrase_start_sec: float | None
    you_end_sec: float | None
    transcript: str
    leading_tokens: list[str]
    extra_tokens: list[str]
    reason: str


def _vosk_words(model, sample_rate, audio_path):
    """Return Vosk's word-level timing for one wav file."""
    import json
    import vosk
    import wave
    rec = vosk.KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)
    with wave.open(audio_path, "rb") as wf:
        if (wf.getnchannels() != 1 or wf.getsampwidth() != 2
                or wf.getframerate() != sample_rate):
            from scipy.signal import resample_poly
            from math import gcd
            data, sr = sf.read(audio_path, dtype="int16")
            if data.ndim == 2:
                data = data.mean(axis=1).astype("int16")
            if sr != sample_rate:
                g = gcd(sr, sample_rate)
                data = resample_poly(data, sample_rate // g, sr // g).astype("int16")
            rec.AcceptWaveform(data.tobytes())
        else:
            rec.AcceptWaveform(wf.readframes(wf.getnframes()))
    final = json.loads(rec.FinalResult())
    return final.get("result", []) or [], final.get("text", "")


YOU_VARIANTS = {"you", "u", "yo", "ya", "yew"}


def _has_voiced_past(audio: np.ndarray, sample_rate: int,
                      start_sec: float,
                      min_voiced_ms: int = 80,
                      silence_rms: float = 0.012) -> int:
    """Return total voiced milliseconds past start_sec (≥ silence_rms in
    25-ms frames). Lets us detect trailing speech too short for Vosk to
    recognize as a word."""
    start = int(start_sec * sample_rate)
    frame = max(1, int(0.025 * sample_rate))
    voiced_ms = 0
    for i in range(start, len(audio), frame):
        chunk = audio[i:i + frame]
        if len(chunk) == 0:
            break
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if rms >= silence_rms:
            voiced_ms += 25
    return voiced_ms


def _has_voiced_in(audio: np.ndarray, sample_rate: int,
                    start_sec: float, end_sec: float,
                    silence_rms: float = 0.012) -> int:
    """Voiced ms within [start_sec, end_sec). Returns 0 if the range is empty."""
    s = max(0, int(start_sec * sample_rate))
    e = min(len(audio), int(end_sec * sample_rate))
    if e <= s:
        return 0
    frame = max(1, int(0.025 * sample_rate))
    voiced = 0
    for i in range(s, e, frame):
        chunk = audio[i:i + frame]
        if len(chunk) == 0:
            break
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if rms >= silence_rms:
            voiced += 25
    return voiced


def _analyze(path: str, model, sample_rate: int) -> TrimResult:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr

    words, text = _vosk_words(model, sample_rate, path)
    empty = TrimResult(path, dur, 0.0, None, None, None, None, text, [], [], "")
    if not words:
        return TrimResult(**{**empty.__dict__, "reason": "vosk returned no word timing"})

    # Locate the phrase start: prefer "i", fall back to "love" if Vosk
    # missed the leading vowel (common on short or quiet onsets).
    phrase_start_idx = None
    for i, w in enumerate(words):
        if w["word"] in I_VARIANTS:
            phrase_start_idx = i
            break
    if phrase_start_idx is None:
        for i, w in enumerate(words):
            if w["word"] in LOVE_VARIANTS:
                phrase_start_idx = i
                break
    if phrase_start_idx is None:
        # Vosk found neither "i" nor "love" — the clip is not a clean
        # "I love you" (might be background noise, partial word, or a
        # totally different phrase that the live detector misclassified).
        # When --reject-trailing is on, callers route this to the
        # rejected directory via cut_end_sec=0.
        return TrimResult(**{**empty.__dict__,
                              "cut_end_sec": 0.0,
                              "new_duration_sec": 0.0,
                              "reason": "no 'i'/'love' anchor in transcript — not a clean ily"})

    phrase_start_sec = words[phrase_start_idx]["start"]
    leading_tokens = [w["word"] for w in words[:phrase_start_idx]]

    # Locate the LAST 'you' (handles "i love you you" etc.)
    you_idx = None
    you_end = None
    for i in range(phrase_start_idx, len(words)):
        if words[i]["word"] in YOU_VARIANTS:
            you_idx = i
            you_end = words[i]["end"]
    if you_end is None:
        # No "you" but we have "i love" — keep up to last love-variant + small pad
        for i in range(len(words) - 1, phrase_start_idx - 1, -1):
            if words[i]["word"] in LOVE_VARIANTS or words[i]["word"] in I_VARIANTS:
                you_end = words[i]["end"]
                you_idx = i
                break
    if you_end is None:
        return TrimResult(**{**empty.__dict__,
                              "phrase_start_sec": phrase_start_sec,
                              "reason": "no phrase end anchor"})

    extra_tokens = [w["word"] for w in words[you_idx + 1:]]

    leadin_ms = phrase_start_sec * 1000.0
    overrun_ms = (dur - you_end) * 1000.0

    # Voiced-in-leading-pad detection — catches "ndy" / "Sa,i love" cases
    # where Vosk didn't label the partial leading word.
    voiced_leadin_ms = _has_voiced_in(
        audio, sr, 0.0,
        max(0.0, phrase_start_sec - PRE_PHRASE_PAD_MS / 1000.0))
    voiced_trail_ms = _has_voiced_in(
        audio, sr, you_end + POST_YOU_PAD_MS / 1000.0, dur)

    has_leading = (leadin_ms >= MIN_LEADIN_MS and
                   (bool(leading_tokens) or voiced_leadin_ms >= 30))
    has_trailing = (overrun_ms >= MIN_OVERRUN_MS and
                    (bool(extra_tokens) or voiced_trail_ms >= 30))

    if not has_leading and not has_trailing:
        return TrimResult(**{**empty.__dict__,
                              "phrase_start_sec": phrase_start_sec,
                              "you_end_sec": you_end,
                              "leading_tokens": leading_tokens,
                              "extra_tokens": extra_tokens,
                              "reason": (
                                  f"clean (leadin={leadin_ms:.0f}ms, overrun={overrun_ms:.0f}ms)")})

    cut_start = max(0.0, phrase_start_sec - PRE_PHRASE_PAD_MS / 1000.0) if has_leading else 0.0
    cut_end = min(dur, you_end + POST_YOU_PAD_MS / 1000.0) if has_trailing else dur
    new_dur = cut_end - cut_start
    reason_parts = []
    if has_leading:
        reason_parts.append(
            f"leading {'words=' + ' '.join(leading_tokens) if leading_tokens else f'sub-word voice {voiced_leadin_ms}ms'}")
    if has_trailing:
        reason_parts.append(
            f"trailing {'words=' + ' '.join(extra_tokens) if extra_tokens else f'sub-word voice {voiced_trail_ms}ms'}")
    return TrimResult(
        path=path,
        duration_sec=dur,
        cut_start_sec=cut_start,
        cut_end_sec=cut_end,
        new_duration_sec=new_dur,
        phrase_start_sec=phrase_start_sec,
        you_end_sec=you_end,
        transcript=text,
        leading_tokens=leading_tokens,
        extra_tokens=extra_tokens,
        reason=" | ".join(reason_parts),
    )


def main():
    import yaml, vosk
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="state.json")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--bucket", default="ily")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--reject-trailing", action="store_true",
                   help="Move clips with trailing speech out of the bucket "
                        "instead of trimming them. Continued-speech clips "
                        "indicate the visitor wasn't doing a complete 'I love "
                        "you', so they shouldn't be in the chorus at all.")
    p.add_argument("--min-current-dur", type=float, default=0.50,
                   help="Skip clips shorter than this (default 0.50s — most "
                        "'i love you' clips are 0.7–1.2s so we want to scan them)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("retrim")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.state) as f:
        state = json.load(f)

    model_path = cfg.get("vosk", {}).get("model_path",
                                            "models/vosk-model-small-en-us-0.15")
    sample_rate = cfg["audio"]["sample_rate"]
    vosk.SetLogLevel(-1)
    model = vosk.Model(model_path)

    paths = [c["path"] for c in state["clips"]
             if c.get("phrase_type") == args.bucket
             and os.path.exists(c["path"])]
    log.info("Scanning %d %s clips with vosk word timing…", len(paths), args.bucket)

    results = []
    skipped_short = 0
    for path in paths:
        try:
            r = _analyze(path, model, sample_rate)
        except Exception as e:
            log.warning("Analyze failed for %s: %s", path, e)
            continue
        if r.duration_sec < args.min_current_dur:
            skipped_short += 1
            continue
        results.append(r)

    flagged = [r for r in results if r.new_duration_sec is not None]
    flagged.sort(key=lambda r: -(r.duration_sec - r.new_duration_sec))
    log.info("Analysed %d clips (skipped %d shorter than %.2fs)",
              len(results), skipped_short, args.min_current_dur)
    log.info("Flagged %d with leading or trailing content:\n", len(flagged))
    for r in flagged:
        cut_ms = (r.duration_sec - r.new_duration_sec) * 1000
        log.info("  -%4dms  %.2fs→%.2fs  cut=[%.2f-%.2f]  txt=%r  -- %s  -- %s",
                  int(cut_ms), r.duration_sec, r.new_duration_sec,
                  r.cut_start_sec, r.cut_end_sec or r.duration_sec,
                  r.transcript, r.reason, os.path.basename(r.path))

    if not args.apply:
        log.info("\nRead-only. Re-run with --apply to overwrite (backups kept).")
        return

    if not flagged:
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    rejected_dir = os.path.join("audio", "live", "i_love_you", "rejected", stamp)
    retrimmed_paths: set[str] = set()
    rejected_paths: set[str] = set()
    new_durations: dict[str, float] = {}

    for r in flagged:
        # Decide trim vs reject. Trailing voice means continued speech →
        # not a clean "I love you", so it shouldn't be in the chorus. We
        # only trim if the issue is purely leading (Vosk hallucination
        # like "the" / quiet "I" onset).
        has_trailing = (r.cut_end_sec is not None
                        and r.cut_end_sec < r.duration_sec - 1e-6)
        if args.reject_trailing and has_trailing:
            try:
                os.makedirs(rejected_dir, exist_ok=True)
                dst = os.path.join(rejected_dir, os.path.basename(r.path))
                shutil.move(r.path, dst)
                rejected_paths.add(r.path)
                log.info("  rejected (trailing speech) %s → %s",
                          os.path.basename(r.path), os.path.basename(rejected_dir))
            except Exception:
                log.exception("Failed to reject %s", r.path)
            continue

        # Safeguard — if the trim target is too short to be a meaningful
        # clip (e.g. came from "no anchor in transcript"), don't write a
        # near-empty file. Leave it alone for human review.
        if r.new_duration_sec is not None and r.new_duration_sec < 0.30:
            log.warning("Skipping %s: trim target only %.2fs — likely needs rejection. "
                         "Re-run with --reject-trailing.",
                         os.path.basename(r.path), r.new_duration_sec)
            continue

        # Otherwise trim in place.
        bak = f"{r.path}.bak.{stamp}"
        try:
            shutil.copy2(r.path, bak)
            audio, sr_ = sf.read(r.path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            s = int(r.cut_start_sec * sr_)
            e = int((r.cut_end_sec if r.cut_end_sec is not None
                     else r.duration_sec) * sr_)
            new_audio = audio[s:e]
            sf.write(r.path, new_audio, sr_)
            log.info("  trimmed %s : %.2fs → %.2fs",
                      os.path.basename(r.path), r.duration_sec,
                      r.new_duration_sec)
            retrimmed_paths.add(r.path)
            new_durations[r.path] = r.new_duration_sec
        except Exception:
            log.exception("Failed to retrim %s", r.path)

    # Update state.json: invalidate embeddings on trimmed clips, remove
    # entries entirely for rejected clips. Backups for both audio + state.
    if retrimmed_paths or rejected_paths:
        bak = f"{args.state}.bak.{stamp}"
        shutil.copy2(args.state, bak)
        new_clips = []
        invalidated = 0
        removed = 0
        for c in state["clips"]:
            if c["path"] in rejected_paths:
                removed += 1
                continue
            if c["path"] in retrimmed_paths:
                c["speaker_embedding"] = []
                c["speaker_id"] = ""
                c["duration_sec"] = new_durations[c["path"]]
                invalidated += 1
            new_clips.append(c)
        state["clips"] = new_clips
        with open(args.state, "w") as f:
            json.dump(state, f, indent=2)
        log.info("State.json: invalidated %d trimmed entries, removed %d "
                  "rejected entries (backup → %s).",
                  invalidated, removed, os.path.basename(bak))

    log.info("Done. Trimmed %d, rejected %d.",
              len(retrimmed_paths), len(rejected_paths))


if __name__ == "__main__":
    main()
