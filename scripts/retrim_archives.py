#!/usr/bin/env python3
"""One-time pass: retrim every clip sitting in an archive subdir using the
fixed phrase locator (which now handles every bucket, not just ily). For
each archived clip:
  1. Re-transcribe with the configured ASR backend.
  2. Run the PhraseMatcher → quality.QualityAnalyzer to locate phrase span.
  3. If phrase is found AND the clip is longer than `max_clean_dur_sec`,
     trim to the phrase span, re-save the file, update the manifest entry.
  4. If phrase is NOT found at all, delete the file and drop the entry
     from the manifest (cleanup of noise that survived the original save).

Never touches audio/fallback/* (Sandy memorial responses etc.).
Default mode is DRY-RUN. Pass --apply to actually mutate disk + manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

ROOT = "/home/utku/Desktop/projects/collectiveEcho"
sys.path.insert(0, ROOT)

import yaml  # noqa: E402
import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402

from modules.phrase_match import PhraseMatcher, buckets_from_config  # noqa: E402
from modules.asr import create_transcriber  # noqa: E402
from modules.emotion import create_emotion_analyzer  # noqa: E402
from modules.quality import QualityAnalyzer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually trim/delete files and rewrite manifests.")
    ap.add_argument("--max-clean-dur-sec", type=float, default=2.0,
                    help="Clips longer than this are candidates for retrim "
                         "(default 2.0s — typical good capture is 0.7–1.5s).")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    matcher = PhraseMatcher(buckets_from_config(cfg))
    transcriber = create_transcriber(cfg)
    transcriber.warmup()
    emotion = create_emotion_analyzer(cfg)
    try:
        emotion.warmup()
    except Exception:
        pass
    quality = QualityAnalyzer(cfg, transcriber=transcriber,
                               emotion_analyzer=emotion, matcher=matcher)
    sample_rate = cfg["audio"]["sample_rate"]

    # Find every archive subdir under every live bucket dir
    live_root = os.path.join(ROOT, "audio", "live")
    manifests: list[str] = []
    for bucket_name in os.listdir(live_root):
        archive_root = os.path.join(live_root, bucket_name, "archive")
        if not os.path.isdir(archive_root):
            continue
        for ts in sorted(os.listdir(archive_root)):
            mp = os.path.join(archive_root, ts, "manifest.json")
            if os.path.isfile(mp):
                manifests.append(mp)

    print(f"Found {len(manifests)} archive manifests\n")

    totals = {"trimmed": 0, "kept_short": 0, "deleted_no_phrase": 0,
              "deleted_missing": 0, "errors": 0}

    for manifest_path in manifests:
        archive_dir = os.path.dirname(manifest_path)
        rel = os.path.relpath(manifest_path, ROOT)
        print(f"=== {rel} ===")
        try:
            with open(manifest_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR reading manifest: {e}")
            totals["errors"] += 1
            continue

        new_clips = []
        for entry in data.get("clips", []):
            basename = os.path.basename(entry.get("path", ""))
            full = os.path.join(archive_dir, basename)
            if not os.path.exists(full):
                print(f"  DROP (missing file): {basename}")
                totals["deleted_missing"] += 1
                continue
            try:
                audio_i16, sr = sf.read(full, dtype="int16")
                if sr != sample_rate:
                    # Should match — pipeline always saves at 16k
                    print(f"  WARN sr mismatch: {sr} vs {sample_rate} → {basename}")
                dur = len(audio_i16) / float(sr)

                # Quick exit for already-short clips
                if dur <= args.max_clean_dur_sec:
                    new_clips.append(entry)
                    totals["kept_short"] += 1
                    continue

                # Re-analyze with full pipeline (transcribe + emotion + locate)
                report = quality.analyze(audio_i16, sr)
                if not report.has_phrase or report.phrase_start_sec < 0:
                    print(f"  DELETE (no phrase, dur={dur:.1f}s): {basename}  transcript={report.emotion.label if report.emotion else '?'}  txt={getattr(report, 'transcript', '')!r}")
                    if args.apply:
                        try:
                            os.remove(full)
                        except Exception as e:
                            print(f"    rm failed: {e}")
                    totals["deleted_no_phrase"] += 1
                    continue

                trimmed = quality.trim_to_phrase(audio_i16, sr, report)
                trim_dur = len(trimmed) / float(sr)
                if len(trimmed) == 0 or trim_dur >= dur * 0.95:
                    # Trim didn't actually shorten meaningfully — keep original
                    new_clips.append(entry)
                    totals["kept_short"] += 1
                    continue

                print(f"  TRIM: {basename}  {dur:.1f}s → {trim_dur:.2f}s")
                if args.apply:
                    sf.write(full, trimmed, sr, subtype="PCM_16")
                # Update entry with new duration; preserve everything else
                entry["duration_sec"] = trim_dur
                new_clips.append(entry)
                totals["trimmed"] += 1
            except Exception as e:
                print(f"  ERROR processing {basename}: {e}")
                totals["errors"] += 1
                new_clips.append(entry)

        data["clips"] = new_clips
        if args.apply:
            try:
                # Backup the manifest once before rewriting
                bak = manifest_path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(manifest_path, bak)
                with open(manifest_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"  manifest write failed: {e}")
        print()

    print("=" * 60)
    print(f"Summary: {totals}")
    if not args.apply:
        print("\n(dry-run — pass --apply to actually mutate)")


if __name__ == "__main__":
    main()
