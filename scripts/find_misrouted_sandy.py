#!/usr/bin/env python3
"""Find ily clips that are actually Sandy's voice.

Two detection modes:

  --mode=transcript : re-run Vosk on each ily clip looking for "sandy"
      in the transcript. Useful for visitors who said "I love you Sandy"
      but were misrouted by the live phrase matcher. Limitation: clips
      are trimmed to the phrase boundary at save time, so any "Sandy"
      after "I love you" was probably cut off before save.

  --mode=speaker   : embed each ily clip and rank by cosine similarity
      to the mean of Sandy's known fallback recordings
      (audio/fallback/i_love_you_sandy/*.wav). Catches BOTH cases —
      Sandy herself speaking and visitor utterances that share her
      voice characteristics (won't happen, just means it's robust to
      trim).

Read-only by default. Pass ``--move`` to actually relocate matched
WAVs into the ily_sandy bucket and update state.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


SANDY_PATTERN = re.compile(r"\bsand[iy]e?\b", re.IGNORECASE)


def load_vosk(model_path: str, sample_rate: int):
    import vosk
    vosk.SetLogLevel(-1)
    model = vosk.Model(model_path)
    return model, sample_rate


def transcribe(model, sample_rate: int, audio_path: str) -> str:
    """Return concatenated transcript text for one audio file."""
    import wave
    import vosk
    rec = vosk.KaldiRecognizer(model, sample_rate)
    rec.SetWords(False)
    with wave.open(audio_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
            # Resample / convert on the fly via soundfile + numpy
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype="int16")
            if data.ndim == 2:
                data = data.mean(axis=1).astype("int16")
            if sr != sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, sample_rate)
                data = resample_poly(data, sample_rate // g, sr // g).astype("int16")
            rec.AcceptWaveform(data.tobytes())
        else:
            rec.AcceptWaveform(wf.readframes(wf.getnframes()))
    final = json.loads(rec.FinalResult()).get("text", "")
    return final.strip()


def detect_by_speaker(state, cfg, threshold: float, log) -> list[dict]:
    """Rank ily clips by cosine sim to the mean of Sandy's fallback files."""
    import numpy as np
    import soundfile as sf
    from modules.speaker import SpeakerIdentifier

    sandy_files = sorted([
        os.path.join("audio/fallback/i_love_you_sandy", f)
        for f in os.listdir("audio/fallback/i_love_you_sandy")
        if f.lower().endswith((".wav", ".flac", ".ogg"))
    ])
    if not sandy_files:
        log.error("No fallback files in audio/fallback/i_love_you_sandy")
        return []
    log.info("Sandy reference files: %s", [os.path.basename(f) for f in sandy_files])

    sid = SpeakerIdentifier(cfg)
    sid.warmup()

    def embed(path):
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return sid.embed(audio, sr)

    ref_embs = np.stack([embed(f) for f in sandy_files])
    ref_mean = ref_embs.mean(axis=0)
    n = float(np.linalg.norm(ref_mean))
    if n > 1e-9:
        ref_mean = ref_mean / n
    log.info("Sandy reference centroid built from %d files", len(sandy_files))

    candidates: list[dict] = []
    for clip in state["clips"]:
        if clip.get("phrase_type") != "ily":
            continue
        emb = clip.get("speaker_embedding")
        if not emb:
            continue
        sim = float(np.dot(np.asarray(emb, dtype=np.float32), ref_mean))
        if sim >= threshold:
            candidates.append({
                "path": clip["path"],
                "timestamp": clip["timestamp"],
                "score": clip.get("score"),
                "speaker_id": clip.get("speaker_id"),
                "sandy_sim": sim,
            })

    candidates.sort(key=lambda c: -c["sandy_sim"])
    return candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--state", default="state.json")
    p.add_argument("--bucket", default="ily")
    p.add_argument("--target-bucket", default="ily_sandy")
    p.add_argument("--mode", choices=["transcript", "speaker"], default="speaker",
                   help="transcript=re-run Vosk; speaker=embedding similarity to Sandy")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Speaker mode: cosine sim threshold to flag as Sandy")
    p.add_argument("--move", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("find-sandy")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.state) as f:
        state = json.load(f)

    target_dir = (cfg.get("memory", {}).get("live_dir_too")
                  if args.target_bucket == "ily_too"
                  else f"audio/live/i_love_you_sandy"
                  if args.target_bucket == "ily_sandy"
                  else cfg.get("memory", {}).get("live_dir", "audio/live"))

    log.info("Scanning bucket=%s mode=%s → %s",
              args.bucket, args.mode, args.target_bucket)

    if args.mode == "speaker":
        candidates = detect_by_speaker(state, cfg, args.threshold, log)
        log.info("Found %d clips with cos≥%.2f to Sandy reference:",
                  len(candidates), args.threshold)
        for c in candidates:
            log.info("  sim=%.3f score=%.2f  %s  sid=%s",
                      c["sandy_sim"], c["score"] or 0.0,
                      os.path.basename(c["path"]), c["speaker_id"][:14] if c["speaker_id"] else "?")
    else:
        model_path = cfg.get("vosk", {}).get("model_path", "models/vosk-model-small-en-us-0.15")
        sample_rate = cfg["audio"]["sample_rate"]
        log.info("Loading vosk model: %s", model_path)
        model, sr = load_vosk(model_path, sample_rate)
        candidates = []
        scanned = 0
        for clip in state["clips"]:
            if clip.get("phrase_type") != args.bucket:
                continue
            if not os.path.exists(clip["path"]):
                continue
            scanned += 1
            try:
                text = transcribe(model, sr, clip["path"])
            except Exception as e:
                log.warning("Transcribe failed for %s: %s", clip["path"], e)
                continue
            if SANDY_PATTERN.search(text):
                candidates.append({
                    "path": clip["path"],
                    "timestamp": clip["timestamp"],
                    "transcript": text,
                    "score": clip.get("score"),
                    "speaker_id": clip.get("speaker_id"),
                })
                log.info("  CANDIDATE %s | %r", os.path.basename(clip["path"]), text)
        log.info("Scanned %d clips, found %d sandy-mentions", scanned, len(candidates))

    if not args.move:
        log.info("Read-only run. Re-run with --move to relocate.")
        return

    if not candidates:
        return

    os.makedirs(target_dir, exist_ok=True)
    path_to_new: dict[str, str] = {}
    for cand in candidates:
        old = cand["path"]
        new = os.path.join(target_dir, os.path.basename(old))
        # On collision, suffix with __movedN
        n = 1
        while os.path.exists(new):
            base, ext = os.path.splitext(os.path.basename(old))
            new = os.path.join(target_dir, f"{base}__moved{n}{ext}")
            n += 1
        shutil.move(old, new)
        path_to_new[old] = new
        log.info("  moved %s → %s", os.path.basename(old), new)

    # Update state.json: change phrase_type + path on matched entries
    moved = 0
    for entry in state["clips"]:
        if entry["path"] in path_to_new:
            entry["path"] = path_to_new[entry["path"]]
            entry["phrase_type"] = args.target_bucket
            moved += 1
    with open(args.state, "w") as f:
        json.dump(state, f, indent=2)
    log.info("Updated state.json: %d clips relocated → %s",
              moved, args.target_bucket)


if __name__ == "__main__":
    main()
