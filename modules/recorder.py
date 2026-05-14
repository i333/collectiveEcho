"""
Recorder — captures the complete clip around a wake-word detection.
Combines the pre-detection rolling buffer with post-detection recording.
"""

import logging
import os
import time
from datetime import datetime

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, cfg: dict, memory_store=None):
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.live_dir = cfg["memory"]["live_dir"]
        self.live_dir_too = cfg["memory"].get("live_dir_too", "audio/live/i_love_you_too")
        os.makedirs(self.live_dir, exist_ok=True)
        os.makedirs(self.live_dir_too, exist_ok=True)
        # Optional: MemoryStore knows per-bucket directories. If supplied, we
        # route saves through it; otherwise we fall back to the legacy paths.
        self._memory = memory_store

    def capture_clip(self, audio_input) -> tuple[np.ndarray, str]:
        """
        Grab pre-buffer + post-detection audio and save to a timestamped WAV.
        Returns (audio_array, file_path).
        """
        pre = audio_input.get_pre_buffer()
        logger.info("Pre-buffer captured: %d samples (%.2f s)",
                     len(pre), len(pre) / self.sample_rate)

        post = audio_input.capture_post_detection()
        logger.info("Post-detection captured: %d samples (%.2f s)",
                     len(post), len(post) / self.sample_rate)

        return self._combine_and_save(pre, post)

    def capture_clip_with_pre(self, pre: np.ndarray, audio_input,
                              phrase_type: str = "ily") -> tuple[np.ndarray, str]:
        """
        Use an already-snapshotted pre-buffer plus fresh post-detection audio.
        This avoids losing the phrase when there's a delay between detection
        and capture (e.g. WLED timeout).
        """
        logger.info("Pre-buffer (pre-snapshotted): %d samples (%.2f s)",
                     len(pre), len(pre) / self.sample_rate)

        post = audio_input.capture_post_detection()
        logger.info("Post-detection captured: %d samples (%.2f s)",
                     len(post), len(post) / self.sample_rate)

        return self._combine_and_save(pre, post, phrase_type=phrase_type)

    def _combine_and_save(self, pre: np.ndarray, post: np.ndarray,
                          phrase_type: str = "ily") -> tuple[np.ndarray, str]:

        if len(pre) > 0 and len(post) > 0:
            clip = np.concatenate([pre, post])
        elif len(pre) > 0:
            clip = pre
        elif len(post) > 0:
            clip = post
        else:
            logger.warning("No audio captured")
            clip = np.array([], dtype=np.int16)

        # Resolve save directory — prefer memory store's per-bucket map so
        # arbitrary phrase buckets (e.g. ily_sandy) write to their own folder.
        if self._memory is not None and hasattr(self._memory, "bucket_live_dir"):
            save_dir = self._memory.bucket_live_dir(phrase_type)
        elif phrase_type == "ily_too":
            save_dir = self.live_dir_too
        else:
            save_dir = self.live_dir
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = phrase_type or "ily"
        filename = f"{prefix}_{ts}.wav"
        filepath = os.path.join(save_dir, filename)
        if len(clip) > 0:
            sf.write(filepath, clip, self.sample_rate, subtype="PCM_16")
            logger.info("Clip saved: %s (%.2f s)", filepath, len(clip) / self.sample_rate)
        else:
            filepath = ""

        return clip, filepath

    def save_clip(self, audio: np.ndarray, path: str | None = None) -> str:
        """Save an arbitrary audio array to a WAV file."""
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"ily_{ts}.wav"
            path = os.path.join(self.live_dir, filename)
        sf.write(path, audio, self.sample_rate, subtype="PCM_16")
        return path
