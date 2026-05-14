"""
Rolling audio buffer that continuously captures microphone input.
Keeps a configurable window of recent audio so the wake phrase
is never lost.
"""

import collections
import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioInput:
    def __init__(self, cfg: dict):
        audio_cfg = cfg["audio"]
        self.sample_rate = audio_cfg["sample_rate"]
        self.channels = audio_cfg["channels"]
        self.device = audio_cfg.get("input_device")
        self.pre_buf_sec = audio_cfg["pre_detection_buffer_sec"]
        self.post_buf_sec = audio_cfg["post_detection_sec"]

        # Rolling buffer sized for pre-detection window
        buf_samples = int(self.pre_buf_sec * self.sample_rate)
        self._ring = collections.deque(maxlen=buf_samples)
        self._lock = threading.Lock()
        self._stream = None
        self._recording = False
        self._post_buf: list[np.ndarray] = []
        self._post_samples_needed = 0
        self._post_done = threading.Event()
        self._first_audio_logged = False

    # --- streaming --------------------------------------------------------

    def start(self):
        """Start continuous microphone capture."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio input started (sr=%d, dev=%s)", self.sample_rate, self.device)

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio input stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio input status: %s", status)
        samples = indata[:, 0].copy()  # mono
        with self._lock:
            self._ring.extend(samples)
            if not self._first_audio_logged:
                self._first_audio_logged = True
                rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
                logger.info("Mic capture confirmed — first audio received (rms=%.6f)", rms)
            if self._recording:
                self._post_buf.append(samples.copy())
                captured = sum(len(b) for b in self._post_buf)
                if captured >= self._post_samples_needed:
                    self._recording = False
                    self._post_done.set()

    # --- capture around detection -----------------------------------------

    def get_pre_buffer(self) -> np.ndarray:
        """Return a copy of the current rolling buffer (pre-detection audio)."""
        with self._lock:
            return np.array(list(self._ring), dtype=np.int16)

    def capture_post_detection(self) -> np.ndarray:
        """Block until post-detection audio is captured, then return it."""
        self._post_samples_needed = int(self.post_buf_sec * self.sample_rate)
        self._post_buf = []
        self._post_done.clear()
        self._recording = True
        self._post_done.wait(timeout=self.post_buf_sec + 2.0)
        self._recording = False
        if self._post_buf:
            return np.concatenate(self._post_buf)
        return np.array([], dtype=np.int16)

    def get_current_chunk(self, duration_sec: float = 0.1) -> np.ndarray:
        """Return the most recent N seconds from the rolling buffer."""
        n = int(duration_sec * self.sample_rate)
        with self._lock:
            buf = list(self._ring)
        if len(buf) >= n:
            return np.array(buf[-n:], dtype=np.int16)
        return np.array(buf, dtype=np.int16)
