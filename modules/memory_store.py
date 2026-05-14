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
                for entry in data.get("clips", []):
                    if not os.path.exists(entry["path"]):
                        continue
                    # Filter to fields ClipEntry knows about, so adding new
                    # fields later doesn't crash on old state.json
                    allowed = ClipEntry.__dataclass_fields__.keys()
                    safe = {k: v for k, v in entry.items() if k in allowed}
                    self.clips.append(ClipEntry(**safe))
                logger.info("Loaded %d clips from state", len(self.clips))
            except Exception:
                logger.exception("Failed to load state file")
                self.clips = []

    def _save_state(self):
        data = {"clips": [asdict(c) for c in self.clips]}
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
                 emotion_label: str = "neutral"):
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
        )
        self.clips.append(entry)

        while len(self.clips) > self.max_clips:
            self._evict_one()

        self._save_state()
        logger.info("Clip added (%d/%d): %s [emo=%s V=%+.2f A=%.2f]",
                     len(self.clips), self.max_clips, path,
                     emotion_label, valence, arousal)

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
