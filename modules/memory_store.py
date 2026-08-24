"""
Memory store — manages the ring of accepted recordings.

Stores per-clip emotion vectors (valence/arousal/dominance) so:
  - Response selection can match or contrast the speaker's emotion
  - Eviction keeps the collection *diverse* instead of just FIFO

Persists state to state.json so clips survive restarts.
"""

import json
import logging
import math
import os
import glob as globmod
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClipEntry:
    path: str
    timestamp: str
    rms: float
    score: float
    duration_sec: float
    is_intimate: bool
    phrase_type: str = "ily"             # "ily" or "ily_too"
    # Emotion fields (V2). Defaults preserve back-compat with old state.json.
    valence: float = 0.0                 # [-1, 1]
    arousal: float = 0.5                 # [0, 1]
    dominance: float = 0.5               # [0, 1]
    emotion_label: str = "neutral"
    last_played_ts: float = 0.0          # epoch seconds; 0 if never
    # Speaker identification (V2.1). Empty string means unidentified —
    # treated as a distinct speaker in dedup so old clips behave sensibly
    # until backfill assigns ids.
    speaker_id: str = ""
    speaker_embedding: list = field(default_factory=list)


class MemoryStore:
    def __init__(self, cfg: dict):
        mem = cfg["memory"]
        self.max_clips = mem["max_clips"]
        self.state_file = mem["state_file"]

        # Backward-compatible default dirs (ily / ily_too) — used if the
        # `phrases.buckets` config is absent or doesn't override them.
        self.live_dir = mem["live_dir"]
        self.live_dir_too = mem.get("live_dir_too", "audio/live/i_love_you_too")
        self.fallback_dir = mem["fallback_dir"]
        self.fallback_dir_too = mem.get("fallback_dir_too", "audio/fallback/i_love_you_too")

        # Per-bucket directory map. Filled by Mirror via set_bucket_dirs()
        # from the phrase-matcher config. Defaults wire up ily + ily_too
        # to the legacy paths above so V1 deployments keep working.
        self._bucket_dirs: dict[str, dict[str, str]] = {
            "ily": {"live": self.live_dir, "fallback": self.fallback_dir},
            "ily_too": {"live": self.live_dir_too, "fallback": self.fallback_dir_too},
        }
        # Buckets that should never save user recordings (e.g. Sandy)
        self._no_save_buckets: set[str] = set()

        # Eviction policy: "fifo" (legacy) | "diversity"
        self.eviction_policy = mem.get("eviction_policy", "diversity")
        self.evict_weight_score = mem.get("evict_weight_score", 0.45)
        self.evict_weight_similarity = mem.get("evict_weight_similarity", 0.35)
        self.evict_weight_age = mem.get("evict_weight_age", 0.20)

        self.clips: list[ClipEntry] = []
        # Threshold the on-disk speaker_ids were clustered with. Used by
        # backfill_speakers() to auto-recluster after a config change.
        self.clustering_threshold: float | None = None
        os.makedirs(self.live_dir_too, exist_ok=True)
        os.makedirs(self.fallback_dir_too, exist_ok=True)
        self._load_state()

    # --- bucket configuration --------------------------------------------

    def configure_buckets(self, buckets):
        """Register PhraseBucket list. Each bucket may declare
        metadata.live_dir / fallback_dir / save_user_recordings overrides."""
        for b in buckets:
            md = getattr(b, "metadata", {}) or {}
            live = md.get("live_dir")
            fallback = md.get("fallback_dir")
            # If not specified, derive from bucket id with a sane default.
            if not live:
                live = f"audio/live/i_love_you_{b.id}" if b.id not in ("ily", "ily_too") else \
                       (self.live_dir_too if b.id == "ily_too" else self.live_dir)
            if not fallback:
                fallback = f"audio/fallback/i_love_you_{b.id}" if b.id not in ("ily", "ily_too") else \
                           (self.fallback_dir_too if b.id == "ily_too" else self.fallback_dir)
            self._bucket_dirs[b.id] = {"live": live, "fallback": fallback}
            os.makedirs(live, exist_ok=True)
            os.makedirs(fallback, exist_ok=True)
            if md.get("save_user_recordings", True) is False:
                self._no_save_buckets.add(b.id)
        logger.info("MemoryStore buckets configured: %s | no-save: %s",
                     list(self._bucket_dirs.keys()), sorted(self._no_save_buckets))

    def bucket_live_dir(self, phrase_type: str) -> str:
        return self._bucket_dirs.get(
            phrase_type, self._bucket_dirs.get("ily", {"live": self.live_dir})
        )["live"]

    def bucket_fallback_dir(self, phrase_type: str) -> str:
        return self._bucket_dirs.get(
            phrase_type, self._bucket_dirs.get("ily", {"fallback": self.fallback_dir})
        )["fallback"]

    def bucket_allows_save(self, phrase_type: str) -> bool:
        return phrase_type not in self._no_save_buckets

    # --- persistence ------------------------------------------------------

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                thr = data.get("clustering_threshold")
                if isinstance(thr, (int, float)):
                    self.clustering_threshold = float(thr)
                for entry in data.get("clips", []):
                    if not os.path.exists(entry["path"]):
                        continue
                    # Filter to fields ClipEntry knows about, so adding new
                    # fields later doesn't crash on old state.json
                    allowed = ClipEntry.__dataclass_fields__.keys()
                    safe = {k: v for k, v in entry.items() if k in allowed}
                    self.clips.append(ClipEntry(**safe))
                logger.info("Loaded %d clips from state (clustering_threshold=%s)",
                             len(self.clips), self.clustering_threshold)
            except Exception:
                logger.exception("Failed to load state file")
                self.clips = []

    def _save_state(self):
        data = {"clips": [asdict(c) for c in self.clips]}
        if self.clustering_threshold is not None:
            data["clustering_threshold"] = self.clustering_threshold
        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.exception("Failed to save state file")

    # --- clip management --------------------------------------------------

    def add_clip(self, path: str, timestamp: str, rms: float,
                 score: float, duration_sec: float, is_intimate: bool,
                 phrase_type: str = "ily",
                 valence: float = 0.0, arousal: float = 0.5,
                 dominance: float = 0.5,
                 emotion_label: str = "neutral",
                 speaker_id: str = "",
                 speaker_embedding: list | None = None):
        entry = ClipEntry(
            path=path,
            timestamp=timestamp,
            rms=rms,
            score=score,
            duration_sec=duration_sec,
            is_intimate=is_intimate,
            phrase_type=phrase_type,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            emotion_label=emotion_label,
            speaker_id=speaker_id,
            speaker_embedding=list(speaker_embedding) if speaker_embedding is not None else [],
        )
        self.clips.append(entry)

        while len(self.clips) > self.max_clips:
            self._evict_one()

        self._save_state()
        logger.info("Clip added (%d/%d): %s [emo=%s V=%+.2f A=%.2f spk=%s]",
                     len(self.clips), self.max_clips, path,
                     emotion_label, valence, arousal, speaker_id or "?")

    # --- eviction ---------------------------------------------------------

    def _evict_one(self):
        """Remove one clip according to the configured policy."""
        if not self.clips:
            return
        if self.eviction_policy == "fifo" or len(self.clips) < 5:
            target_idx = 0
        else:
            target_idx = self._pick_for_eviction()

        evicted = self.clips.pop(target_idx)
        try:
            if os.path.exists(evicted.path):
                os.remove(evicted.path)
                logger.info("Evicted clip [%s, idx=%d]: %s",
                             self.eviction_policy, target_idx, evicted.path)
        except OSError:
            logger.warning("Could not remove evicted clip: %s", evicted.path)

    def _pick_for_eviction(self) -> int:
        """Diversity-based eviction.

        For each clip, compute a "redundancy" score:
            - higher if its emotion vector is close to many others (commonly held)
            - higher if its quality score is low
            - higher if it's old (FIFO bias as tiebreaker)
        Evict the clip with the highest redundancy.
        """
        n = len(self.clips)

        # Quality penalty (low score → higher redundancy)
        scores = [c.score for c in self.clips]
        score_term = [1.0 - max(0.0, min(1.0, s)) for s in scores]

        # Similarity penalty — for each clip, mean inverse-distance to others
        # in (valence, arousal, dominance) space.
        sim_term = []
        for i, ci in enumerate(self.clips):
            dists = []
            for j, cj in enumerate(self.clips):
                if i == j:
                    continue
                dv = ci.valence - cj.valence
                da = ci.arousal - cj.arousal
                dd = ci.dominance - cj.dominance
                d = math.sqrt(dv * dv + da * da + dd * dd)
                dists.append(d)
            if dists:
                # Crowded (low avg distance) → high sim_term
                avg_d = sum(dists) / len(dists)
                sim_term.append(1.0 / (1.0 + 4.0 * avg_d))
            else:
                sim_term.append(0.5)

        # Age penalty — older index → higher
        age_term = [i / max(1, n - 1) for i in range(n)]

        # Combine
        ws = self.evict_weight_score
        wsim = self.evict_weight_similarity
        wage = self.evict_weight_age
        # Slight age preference: when scores are roughly equal, prefer evicting older.
        # But weight similarity high so we preserve rare clips.
        redundancy = [
            ws * score_term[i] + wsim * sim_term[i] + wage * age_term[i]
            for i in range(n)
        ]
        # Pick highest, breaking ties by oldest
        worst_i = max(range(n), key=lambda i: (redundancy[i], age_term[i]))
        logger.debug("Eviction candidate idx=%d redundancy=%.3f", worst_i, redundancy[worst_i])
        return worst_i

    # --- selection --------------------------------------------------------

    def get_recent_clips(self, n: Optional[int] = None) -> list[ClipEntry]:
        """Return most recent n clips, newest last."""
        if n is None:
            n = self.max_clips
        return self.clips[-n:]

    def get_clips_by_type(self, phrase_type: str, n: Optional[int] = None) -> list[ClipEntry]:
        typed = [c for c in self.clips if c.phrase_type == phrase_type]
        if n is not None:
            typed = typed[-n:]
        return typed

    def get_fallback_clips(self, phrase_type: str = "ily") -> list[str]:
        d = self.bucket_fallback_dir(phrase_type)
        patterns = [
            os.path.join(d, "*.wav"),
            os.path.join(d, "*.flac"),
            os.path.join(d, "*.ogg"),
        ]
        files = []
        for pat in patterns:
            files.extend(globmod.glob(pat))
        files.sort()
        return files

    def archive_live_clips(self, exclude_buckets: set[str] | None = None) -> dict:
        """Move every saved live clip into a timestamped archive subdir under
        its current directory, write a manifest.json alongside the moved
        files preserving full ClipEntry metadata (score, emotion, V/A/D,
        timestamp, etc.), and remove the entries from state.json so they
        no longer appear in any pool/layered selection.

        Files are PRESERVED on disk (moved, not deleted) so nothing is lost.
        The companion restore_archives() reads the manifest to put clips
        back with all metadata intact.

        Returns a per-bucket count of archived clips.
        """
        exclude_buckets = exclude_buckets or set()
        archive_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        counts: dict = {}
        kept_clips: list[ClipEntry] = []
        # manifest_groups[archive_dir] = list of (entry_dict, original_relative_path)
        manifest_groups: dict[str, list[dict]] = {}
        for c in self.clips:
            if c.phrase_type in exclude_buckets:
                kept_clips.append(c)
                continue
            path_full = c.path if os.path.isabs(c.path) else c.path
            if not os.path.exists(path_full):
                # Already missing — drop the entry quietly
                counts[c.phrase_type] = counts.get(c.phrase_type, 0) + 1
                continue
            bucket_dir = os.path.dirname(path_full)
            archive_dir = os.path.join(bucket_dir, "archive", archive_ts)
            try:
                os.makedirs(archive_dir, exist_ok=True)
                new_path = os.path.join(archive_dir, os.path.basename(path_full))
                os.rename(path_full, new_path)
                # Stash the entry dict + ORIGINAL path so restore can put
                # the file back to its proper bucket dir later.
                entry_dict = asdict(c)
                entry_dict["_original_path"] = c.path
                manifest_groups.setdefault(archive_dir, []).append(entry_dict)
                counts[c.phrase_type] = counts.get(c.phrase_type, 0) + 1
            except Exception:
                logger.exception("Archive failed for %s — keeping in state", c.path)
                kept_clips.append(c)

        # Write one manifest.json per archive subdir so restore_archives()
        # can find and roll them back.
        for archive_dir, entries in manifest_groups.items():
            try:
                with open(os.path.join(archive_dir, "manifest.json"), "w") as f:
                    json.dump({
                        "archived_at": archive_ts,
                        "clips": entries,
                    }, f, indent=2)
            except Exception:
                logger.exception("Failed to write manifest in %s", archive_dir)

        self.clips = kept_clips
        self._save_state()
        logger.info("Archived live clips → archive/%s : %s",
                     archive_ts, counts)
        return counts

    def restore_archives(self) -> dict:
        """Walk every bucket's archive subdirs, read each manifest.json, move
        the clips back into the bucket's live dir, and re-add the entries
        (with original metadata: score, emotion, V/A/D, timestamp) to state.

        Idempotent: a manifest with no remaining files is skipped. After
        successful restore, the manifest.json is removed so the same archive
        isn't re-restored next time.

        On filename collisions (a freshly-captured clip already occupies the
        original path), the restored clip is renamed with a __restoredN
        suffix and its entry path is updated.

        Returns per-bucket count of restored clips.
        """
        counts: dict = {}
        restored_total = 0
        skipped_missing = 0
        # Find every archive subdir under every known bucket's live dir.
        # Live dirs come from configure_buckets — fall back to the legacy
        # ily/ily_too if no buckets registered.
        live_dirs: set[str] = set()
        for d in self._bucket_dirs.values():
            live_dirs.add(d.get("live", ""))
        live_dirs.discard("")

        manifest_paths: list[str] = []
        for live_dir in live_dirs:
            archive_root = os.path.join(live_dir, "archive")
            if not os.path.isdir(archive_root):
                continue
            for ts_dir in sorted(os.listdir(archive_root)):
                full_ts = os.path.join(archive_root, ts_dir)
                manifest = os.path.join(full_ts, "manifest.json")
                if os.path.isfile(manifest):
                    manifest_paths.append(manifest)

        for manifest_path in manifest_paths:
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
            except Exception:
                logger.exception("Could not read manifest %s — skipping", manifest_path)
                continue

            archive_dir = os.path.dirname(manifest_path)
            for entry_dict in data.get("clips", []):
                original_path = entry_dict.pop("_original_path", None)
                if not original_path:
                    # Old archives without _original_path — best-effort: put
                    # back into the bucket's live_dir using the basename.
                    original_path = os.path.join(
                        self.bucket_live_dir(entry_dict.get("phrase_type", "ily")),
                        os.path.basename(entry_dict.get("path", "")),
                    )
                src = os.path.join(archive_dir, os.path.basename(original_path))
                if not os.path.exists(src):
                    skipped_missing += 1
                    continue
                # Handle collision at destination
                dest = original_path
                n = 1
                while os.path.exists(dest):
                    base, ext = os.path.splitext(original_path)
                    dest = f"{base}__restored{n}{ext}"
                    n += 1
                try:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.rename(src, dest)
                    entry_dict["path"] = dest
                    # Re-create ClipEntry with only known fields (forward-compat)
                    allowed = ClipEntry.__dataclass_fields__.keys()
                    safe = {k: v for k, v in entry_dict.items() if k in allowed}
                    self.clips.append(ClipEntry(**safe))
                    pt = entry_dict.get("phrase_type", "ily")
                    counts[pt] = counts.get(pt, 0) + 1
                    restored_total += 1
                except Exception:
                    logger.exception("Restore failed for %s", src)

            # Manifest done — remove it AND the (now-empty) archive subdir if empty.
            try:
                os.remove(manifest_path)
                # Try to remove the timestamped archive dir if empty
                if not os.listdir(archive_dir):
                    os.rmdir(archive_dir)
            except Exception:
                pass

        self._save_state()
        logger.info("Restored %d clips from %d manifests : %s (missing: %d)",
                     restored_total, len(manifest_paths), counts, skipped_missing)
        return counts

    def has_live_clips(self, phrase_type: str | None = None) -> bool:
        if phrase_type is None:
            return len(self.clips) > 0
        return any(c.phrase_type == phrase_type for c in self.clips)

    def clip_count(self) -> int:
        return len(self.clips)

    # --- emotion-aware selection (new) -----------------------------------

    def pick_by_emotion(self, target_v: float, target_a: float,
                         n: int = 1, mode: str = "match",
                         phrase_type: str | None = None) -> list[ClipEntry]:
        """Pick clips by emotion proximity.

        mode:
          "match"      — closest in (V, A) space (echo the speaker's feeling)
          "contrast"   — farthest (offer a counter-emotion)
          "diverse"    — spread across the space (one match, one mid, one far)
        """
        pool = (
            [c for c in self.clips if c.phrase_type == phrase_type]
            if phrase_type else list(self.clips)
        )
        if not pool:
            return []

        def d(c: ClipEntry) -> float:
            return math.sqrt(
                (c.valence - target_v) ** 2 + (c.arousal - target_a) ** 2
            )

        scored = sorted(pool, key=d)
        if mode == "match":
            chosen = scored[:n]
        elif mode == "contrast":
            chosen = scored[-n:][::-1]
        elif mode == "diverse":
            # Evenly sample across the sorted list
            if n <= 0:
                return []
            if n >= len(scored):
                chosen = scored
            else:
                step = max(1, len(scored) // n)
                chosen = scored[::step][:n]
        else:
            chosen = random.sample(scored, k=min(n, len(scored)))
        return chosen

    # --- speaker identification ------------------------------------------

    def speaker_centroids(self) -> list[tuple[str, "np.ndarray"]]:
        """Return [(speaker_id, mean_embedding)] for every speaker that has
        at least one embedded clip. Centroids are L2-normalised so cosine
        comparisons against a fresh embedding behave correctly."""
        import numpy as np
        groups: dict[str, list[list[float]]] = {}
        for c in self.clips:
            if not c.speaker_id or not c.speaker_embedding:
                continue
            groups.setdefault(c.speaker_id, []).append(c.speaker_embedding)
        out = []
        for sid, embs in groups.items():
            arr = np.asarray(embs, dtype=np.float32)
            mean = arr.mean(axis=0)
            n = float(np.linalg.norm(mean))
            if n > 1e-9:
                mean = mean / n
            out.append((sid, mean))
        return out

    def recluster_speakers(self, identifier) -> dict:
        """Re-run online clustering on already-embedded clips, ignoring any
        existing speaker_id assignments. Use this after retuning the
        similarity threshold so old assignments get rebuilt without paying
        the cost of re-embedding every clip.
        """
        for c in self.clips:
            c.speaker_id = ""
        return self.backfill_speakers(identifier)

    def backfill_speakers(self, identifier) -> dict:
        """Embed any clip lacking a speaker_embedding, cluster every clip
        without a speaker_id, persist once at the end.

        Returns a summary {clips_embedded, clips_identified, speakers_total}.
        """
        import numpy as np
        import soundfile as sf

        if not identifier or not getattr(identifier, "enabled", False):
            return {"clips_embedded": 0, "clips_identified": 0, "speakers_total": 0}

        # If the configured similarity threshold differs from the one the
        # on-disk speaker_ids were clustered with, drop those assignments
        # so the loop below re-clusters all clips at the current threshold.
        cur_thr = float(identifier.similarity_threshold)
        if self.clustering_threshold is not None and \
                abs(self.clustering_threshold - cur_thr) > 1e-6:
            logger.info("Threshold changed (%.3f → %.3f) — clearing speaker_ids for recluster",
                         self.clustering_threshold, cur_thr)
            for c in self.clips:
                c.speaker_id = ""
        self.clustering_threshold = cur_thr

        # 1. Embed clips that don't have an embedding yet.
        embedded = 0
        for c in self.clips:
            if c.speaker_embedding:
                continue
            if not os.path.exists(c.path):
                continue
            try:
                audio, sr = sf.read(c.path, dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                emb = identifier.embed(audio, sr)
                c.speaker_embedding = emb.tolist()
                embedded += 1
                if embedded % 25 == 0:
                    logger.info("Backfill: embedded %d clips so far…", embedded)
            except Exception:
                logger.exception("Backfill embed failed for %s", c.path)

        # 2. Cluster all embedded clips per bucket using OFFLINE average-link
        #    agglomerative clustering. Average-link looks at the mean of all
        #    cross-cluster pairwise sims, which is more stable than online
        #    centroid drift and avoids the chain-collapse failure mode of
        #    single-link / online-sequential. After the initial cut we run
        #    centroid-refinement iterations so noisy outlier clips can
        #    re-attach to whichever cluster their MEAN embedding lives in.
        identified = 0
        bucket_to_clips: dict[str, list] = {}
        for c in self.clips:
            if not c.speaker_embedding:
                continue
            bucket_to_clips.setdefault(c.phrase_type, []).append(c)

        for bucket, clips_in_bucket in bucket_to_clips.items():
            if not clips_in_bucket:
                continue
            assignments = _cluster_speakers(
                clips_in_bucket,
                threshold=cur_thr,
                logger=logger,
                bucket_label=bucket,
            )
            for c, sid in zip(clips_in_bucket, assignments):
                c.speaker_id = sid
                identified += 1

        speakers_total = len({c.speaker_id for c in self.clips if c.speaker_id})
        if embedded or identified:
            self._save_state()
        logger.info(
            "Speaker backfill: embedded=%d identified=%d → %d speakers across %d clips",
            embedded, identified, speakers_total, len(self.clips),
        )
        return {
            "clips_embedded": embedded,
            "clips_identified": identified,
            "speakers_total": speakers_total,
        }

    # --- selection helpers (continued) -----------------------------------

    def mark_played(self, paths: list[str]):
        import time as _t
        now = _t.time()
        path_set = set(paths)
        changed = False
        for c in self.clips:
            if c.path in path_set:
                c.last_played_ts = now
                changed = True
        if changed:
            self._save_state()


# ----------------------------------------------------------------------
# Speaker clustering — offline batch with average-link + centroid refine
# ----------------------------------------------------------------------

def _cluster_speakers(clips: list, threshold: float, logger,
                       bucket_label: str = "",
                       refinement_min_sim: float | None = None,
                       max_iterations: int = 10) -> list[str]:
    """Cluster a list of ClipEntry by speaker_embedding.

    Returns a parallel list of speaker_id strings (one per input clip).

    Algorithm:
      1. Stack L2-normalised embeddings into an NxD matrix.
      2. Compute the NxN cosine similarity matrix.
      3. Run scipy hierarchical clustering with the 'average' linkage on
         distances 1 - sim. Cut the dendrogram at (1 - threshold).
      4. Refinement loop: recompute each cluster's mean embedding (a
         smoother prototype than any single clip), then reassign every
         clip to whichever centroid it's closest to as long as the sim
         clears refinement_min_sim. Iterate until assignments stabilise.

    Average-link is more robust than online sequential clustering: the
    cluster-similarity is the MEAN of all cross-cluster pair sims, so a
    chain of weak links can't drag unrelated clips together. The refine
    step then absorbs noisy outliers via centroid attraction.

    Speaker ids are derived from a hash of each cluster's earliest-by-
    timestamp clip path, so the same cluster gets the same id across
    reclusters / restarts. (Eviction of the anchor clip is the only
    case where the id changes — rare.)
    """
    import hashlib
    import uuid
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(clips)
    if n == 0:
        return []
    if n == 1:
        return [f"spk_{uuid.uuid4().hex[:8]}"]

    embs = np.stack([
        np.asarray(c.speaker_embedding, dtype=np.float32) for c in clips
    ])
    # Re-normalize (defensive — the encoder already does this)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-9, None)

    sim = embs @ embs.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # scipy expects a 1-D condensed distance (upper triangle)
    cond_dist = squareform(np.clip(dist, 0.0, 2.0), checks=False)

    Z = linkage(cond_dist, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    initial_n = len(set(labels))

    # Refinement: assign each clip to closest centroid.
    if refinement_min_sim is None:
        # Slightly looser than the cut threshold — centroids are stable
        # enough to absorb noisy clips that didn't make the initial cut.
        refinement_min_sim = max(0.15, threshold - 0.10)

    for iteration in range(max_iterations):
        # Compute L2-normalised centroids per cluster
        unique = np.unique(labels)
        centroids = np.zeros((len(unique), embs.shape[1]), dtype=np.float32)
        label_index = {int(lbl): i for i, lbl in enumerate(unique)}
        for lbl in unique:
            mask = labels == lbl
            mean = embs[mask].mean(axis=0)
            n_ = np.linalg.norm(mean)
            if n_ > 1e-9:
                mean = mean / n_
            centroids[label_index[int(lbl)]] = mean

        # For each clip, find the closest centroid AMONG CLUSTERS IT
        # ISN'T ALREADY IN. Without this, a singleton's "best centroid"
        # is itself (sim=1.0) and the refinement never triggers a move.
        clip_to_centroid = embs @ centroids.T  # N x K
        for i in range(n):
            clip_to_centroid[i, label_index[int(labels[i])]] = -1.0
        best_idx = np.argmax(clip_to_centroid, axis=1)
        best_sim = clip_to_centroid[np.arange(n), best_idx]

        new_labels = labels.copy()
        for i in range(n):
            if best_sim[i] >= refinement_min_sim:
                new_labels[i] = unique[best_idx[i]]
        changed = int((new_labels != labels).sum())
        labels = new_labels
        if changed == 0:
            break

    refined_n = len(set(labels))
    logger.info(
        "Speaker clustering [%s]: %d clips → %d clusters (avg-link cut %.2f → "
        "centroid-refine min_sim %.2f, %d iters reduced %d→%d clusters)",
        bucket_label, n, refined_n, threshold, refinement_min_sim,
        iteration + 1, initial_n, refined_n,
    )

    # Map scipy label ints to STABLE speaker_ids — each cluster's id is
    # a hash of its earliest-by-timestamp member's path, so the same
    # cluster lands on the same id across reclusters / restarts.
    label_to_sid: dict[int, str] = {}
    for lbl in set(int(x) for x in labels):
        members = [i for i, l in enumerate(labels) if int(l) == lbl]
        anchor = min(members, key=lambda i: clips[i].timestamp)
        h = hashlib.sha1(clips[anchor].path.encode("utf-8")).hexdigest()[:8]
        label_to_sid[lbl] = f"spk_{h}"
    return [label_to_sid[int(lbl)] for lbl in labels]
