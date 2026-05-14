"""
Memory store — manages the ring of most-recent accepted recordings.
Persists state to state.json so clips survive restarts.
"""

import json
import logging
import os
import glob as globmod
from dataclasses import dataclass, asdict
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
    phrase_type: str = "ily"  # "ily" or "ily_too"


class MemoryStore:
    def __init__(self, cfg: dict):
        mem = cfg["memory"]
        self.max_clips = mem["max_clips"]
        self.live_dir = mem["live_dir"]
        self.live_dir_too = mem.get("live_dir_too", "audio/live/i_love_you_too")
        self.fallback_dir = mem["fallback_dir"]
        self.fallback_dir_too = mem.get("fallback_dir_too", "audio/fallback/i_love_you_too")
        self.state_file = mem["state_file"]
        self.clips: list[ClipEntry] = []
        os.makedirs(self.live_dir_too, exist_ok=True)
        os.makedirs(self.fallback_dir_too, exist_ok=True)
        self._load_state()

    # --- persistence ------------------------------------------------------

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                for entry in data.get("clips", []):
                    # Only keep entries whose files still exist
                    if os.path.exists(entry["path"]):
                        self.clips.append(ClipEntry(**entry))
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
                 phrase_type: str = "ily"):
        entry = ClipEntry(
            path=path,
            timestamp=timestamp,
            rms=rms,
            score=score,
            duration_sec=duration_sec,
            is_intimate=is_intimate,
            phrase_type=phrase_type,
        )
        self.clips.append(entry)

        # Evict oldest if over capacity
        while len(self.clips) > self.max_clips:
            evicted = self.clips.pop(0)
            try:
                if os.path.exists(evicted.path):
                    os.remove(evicted.path)
                    logger.info("Evicted old clip: %s", evicted.path)
            except OSError:
                logger.warning("Could not remove evicted clip: %s", evicted.path)

        self._save_state()
        logger.info("Clip added to memory (%d/%d): %s",
                     len(self.clips), self.max_clips, path)

    def get_recent_clips(self, n: Optional[int] = None) -> list[ClipEntry]:
        """Return most recent n clips, newest last."""
        if n is None:
            n = self.max_clips
        return self.clips[-n:]

    def get_clips_by_type(self, phrase_type: str, n: Optional[int] = None) -> list[ClipEntry]:
        """Return clips of a specific phrase type, newest last."""
        typed = [c for c in self.clips if c.phrase_type == phrase_type]
        if n is not None:
            typed = typed[-n:]
        return typed

    def get_fallback_clips(self, phrase_type: str = "ily") -> list[str]:
        """Return paths to fallback clips for a phrase type."""
        d = self.fallback_dir_too if phrase_type == "ily_too" else self.fallback_dir
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
