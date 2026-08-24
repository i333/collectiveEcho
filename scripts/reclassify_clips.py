#!/usr/bin/env python3
"""One-time cleanup pass over saved live clips.

For every entry in state.json:
  1. Re-transcribe the WAV using the configured ASR backend.
  2. Run the PhraseMatcher on the transcript.
  3. If the match disagrees with the stored phrase_type → MOVE the file into
     the correct bucket's live_dir (renamed prefix), update phrase_type.
  4. If no match (likely speaker-bleed noise or silence) → DELETE the file
     and drop the entry from state.json.

Never touches audio/fallback/* — those are user-curated and protected.
state.json is backed up to state.json.bak.<timestamp> before any mutation.

Default mode is DRY-RUN. Pass --apply to actually mutate disk and state.
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

from modules.phrase_match import PhraseMatcher, buckets_from_config  # noqa: E402
from modules.asr import create_transcriber  # noqa: E402
from modules.memory_store import MemoryStore  # noqa: E402


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _store_path(p: str) -> str:
    """Format a path the way state.json stores it (relative to ROOT)."""
    if p.startswith(ROOT + os.sep):
        return p[len(ROOT) + 1:]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually move/delete and rewrite state.json. Without "
                         "this flag, only prints what would happen.")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--state", default=os.path.join(ROOT, "state.json"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    matcher = PhraseMatcher(buckets_from_config(cfg))
    memory = MemoryStore(cfg)
    memory.configure_buckets(matcher.buckets)

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Loading transcriber ({cfg.get('asr', {}).get('backend', 'vosk')})...")
    transcriber = create_transcriber(cfg)
    transcriber.warmup()

    with open(args.state) as f:
        data = json.load(f)
    clips = data.get("clips", [])
    print(f"Total clips in state: {len(clips)}\n")

    moved = 0
    deleted_missing = 0
    deleted_no_phrase = 0
    deleted_no_save = 0
    kept = 0
    errors = 0
    new_clips: list[dict] = []

    for entry in clips:
        path_raw = entry["path"]
        path_full = _resolve(path_raw)
        current_pt = entry.get("phrase_type", "ily")

        # SAFETY: never touch fallback files (none should be in state, but guard).
        if "/fallback/" in path_full:
            print(f"  SKIP (fallback): {path_raw}")
            new_clips.append(entry)
            kept += 1
            continue

        if not os.path.exists(path_full):
            print(f"  DROP (missing file): [{current_pt}] {path_raw}")
            deleted_missing += 1
            continue

        try:
            audio, sr = sf.read(path_full, dtype="int16")
            result = transcriber.transcribe(audio, sr)
            transcript = (result.text or "").lower().strip()
        except Exception as e:
            print(f"  ERROR transcribing {path_raw}: {e}")
            errors += 1
            new_clips.append(entry)
            continue

        hit = matcher.find(transcript)

        if hit is None:
            print(f"  DELETE (no_phrase, was [{current_pt}]): "
                   f"transcript={transcript!r} → {path_raw}")
            if args.apply:
                try:
                    os.remove(path_full)
                except Exception as e:
                    print(f"    rm failed: {e}")
            deleted_no_phrase += 1
            continue

        correct_pt = hit.phrase_type

        if correct_pt == current_pt:
            # Already correctly tagged. Verify the file is in the right dir
            # though — if not, move it (handles the legacy ily_sandy clip
            # still living in the ily live_dir).
            correct_dir = _resolve(memory.bucket_live_dir(correct_pt))
            if os.path.dirname(path_full) == correct_dir:
                kept += 1
                new_clips.append(entry)
                continue
            # Re-tagged correctly but wrong directory → move
            print(f"  RELOCATE (correct tag, wrong dir): [{current_pt}] "
                   f"{path_raw}")
        else:
            print(f"  RECLASSIFY [{current_pt} → {correct_pt}]: "
                   f"transcript={transcript!r}")

        # Both reclassify and relocate land here — compute new path.
        if not memory.bucket_allows_save(correct_pt):
            print(f"    correct bucket {correct_pt!r} disallows save → DELETE")
            if args.apply:
                try:
                    os.remove(path_full)
                except Exception as e:
                    print(f"    rm failed: {e}")
            deleted_no_save += 1
            continue

        correct_dir = _resolve(memory.bucket_live_dir(correct_pt))
        old_filename = os.path.basename(path_full)
        if old_filename.startswith(f"{current_pt}_"):
            new_filename = f"{correct_pt}_" + old_filename[len(current_pt) + 1:]
        elif not old_filename.startswith(f"{correct_pt}_"):
            new_filename = f"{correct_pt}_" + old_filename
        else:
            new_filename = old_filename
        new_path_full = os.path.join(correct_dir, new_filename)

        if os.path.abspath(new_path_full) == os.path.abspath(path_full):
            # No actual move needed
            kept += 1
            updated = dict(entry)
            updated["phrase_type"] = correct_pt
            new_clips.append(updated)
            continue

        print(f"    {path_raw}  →  {_store_path(new_path_full)}")
        if args.apply:
            try:
                os.makedirs(correct_dir, exist_ok=True)
                # If destination already exists (collision), append a suffix
                final_path = new_path_full
                n = 1
                while os.path.exists(final_path):
                    base, ext = os.path.splitext(new_path_full)
                    final_path = f"{base}__dup{n}{ext}"
                    n += 1
                os.rename(path_full, final_path)
                updated = dict(entry)
                updated["phrase_type"] = correct_pt
                updated["path"] = _store_path(final_path)
                new_clips.append(updated)
                moved += 1
            except Exception as e:
                print(f"    move failed: {e}")
                errors += 1
                new_clips.append(entry)
        else:
            # Dry-run: keep the entry but pretend it'd be updated
            updated = dict(entry)
            updated["phrase_type"] = correct_pt
            updated["path"] = _store_path(new_path_full)
            new_clips.append(updated)
            moved += 1

    print()
    print(f"Summary:  moved={moved}  deleted_missing={deleted_missing}  "
           f"deleted_no_phrase={deleted_no_phrase}  "
           f"deleted_no_save={deleted_no_save}  kept={kept}  errors={errors}")
    print(f"Final clip count: {len(new_clips)} (was {len(clips)})")

    if args.apply:
        backup = args.state + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(args.state, backup)
        print(f"\nBackup written: {backup}")
        data["clips"] = new_clips
        with open(args.state, "w") as f:
            json.dump(data, f, indent=2)
        print(f"state.json updated.")
    else:
        print("\n(dry-run — pass --apply to actually mutate)")


if __name__ == "__main__":
    main()
