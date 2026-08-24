#!/usr/bin/env python3
"""Curate one best-quality clip per speaker cluster for a given bucket.

Loads state.json, filters clips by `--bucket`, runs the same offline
average-link + centroid-refinement clustering the chorus pipeline uses,
then copies each cluster's highest-score clip into a fresh destination
directory. Writes a manifest.json alongside so you can trace every
curated file back to its source clip + cluster.

Example:
    python3 scripts/curate_best_per_speaker.py --bucket ily_sandy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from modules.memory_store import ClipEntry, _cluster_speakers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", required=True,
                   help="phrase_type bucket id, e.g. ily_sandy")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--state", default="state.json")
    p.add_argument("--dest", default=None,
                   help="Output dir (default: audio/curated/i_love_you_<bucket-id>)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Average-link cosine sim threshold (default: config speaker.similarity_threshold)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("curate")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.state) as f:
        state = json.load(f)

    raw = [c for c in state["clips"]
           if c.get("phrase_type") == args.bucket and c.get("speaker_embedding")]
    if not raw:
        log.error("No embedded clips found for bucket %r", args.bucket)
        sys.exit(1)

    log.info("Bucket %s: %d embedded clips", args.bucket, len(raw))

    # Rehydrate to ClipEntry so we can call _cluster_speakers directly.
    clips = [ClipEntry(
        path=c["path"], timestamp=c["timestamp"], rms=c["rms"], score=c["score"],
        duration_sec=c["duration_sec"], is_intimate=c["is_intimate"],
        phrase_type=c["phrase_type"], speaker_id="",
        speaker_embedding=c["speaker_embedding"],
    ) for c in raw]

    threshold = (args.threshold
                 if args.threshold is not None
                 else cfg.get("speaker", {}).get("similarity_threshold", 0.30))
    sids = _cluster_speakers(clips, threshold=float(threshold),
                              logger=log, bucket_label=args.bucket)
    for c, sid in zip(clips, sids):
        c.speaker_id = sid

    # Group clips per speaker cluster
    by_speaker: dict[str, list[ClipEntry]] = defaultdict(list)
    for c in clips:
        by_speaker[c.speaker_id].append(c)
    log.info("Clusters: %d total", len(by_speaker))

    dest = (args.dest if args.dest
            else f"audio/curated/i_love_you_{args.bucket.removeprefix('ily_') if args.bucket.startswith('ily_') else args.bucket}")
    os.makedirs(dest, exist_ok=True)
    log.info("Destination: %s", dest)

    manifest = []
    skipped = 0
    for sid, members in sorted(by_speaker.items(), key=lambda kv: -len(kv[1])):
        # Pick best: highest score, tie-broken by most-recent timestamp.
        best = max(members, key=lambda c: ((c.score or 0.0), c.timestamp))
        src = best.path
        if not os.path.exists(src):
            log.warning("Source missing for cluster %s best clip: %s", sid, src)
            skipped += 1
            continue
        # Destination filename includes cluster size so it's easy to spot
        # which voices have the most recordings.
        base, ext = os.path.splitext(os.path.basename(src))
        out_name = f"{sid}_n{len(members)}_{base}{ext}"
        out_path = os.path.join(dest, out_name)
        shutil.copy2(src, out_path)
        manifest.append({
            "speaker_id": sid,
            "cluster_size": len(members),
            "source_path": src,
            "curated_path": out_path,
            "score": best.score,
            "duration_sec": best.duration_sec,
            "timestamp": best.timestamp,
            "valence": best.valence,
            "arousal": best.arousal,
            "dominance": best.dominance,
            "emotion_label": best.emotion_label,
            "cluster_member_paths": [c.path for c in members],
        })

    manifest_path = os.path.join(dest, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "bucket": args.bucket,
            "threshold": threshold,
            "n_clips": len(clips),
            "n_clusters": len(by_speaker),
            "n_curated": len(manifest),
            "skipped_missing_source": skipped,
            "clusters": manifest,
        }, f, indent=2)

    log.info("Curated %d files (%d skipped); manifest → %s",
              len(manifest), skipped, manifest_path)


if __name__ == "__main__":
    main()
