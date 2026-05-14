"""
Rolling audio buffer that continuously captures microphone input.
Keeps a configurable window of recent audio so the wake phrase
is never lost.

Adds adaptive post-detection capture: keep recording until silence
(N ms below a noise threshold) is detected, with a hard upper limit
so we never run forever.
"""

import collections
import logging
import threading
import time

import numpy as np

# sounddevice imports PortAudio at module load time, which crashes on
# systems where libportaudio2 isn't installed. We defer the import to start()
# so the rest of the application can still load (analyze clips, run --test
# with the simulate detector and a mocked input, etc.).
logger = logging.getLogger(__name__)


class AudioInput:
    def __init__(self, cfg: dict):
        audio_cfg = cfg["audio"]
        self.sample_rate = audio_cfg["sample_rate"]
        self.channels = audio_cfg["channels"]
        self.device = audio_cfg.get("input_device")
        self.pre_buf_sec = audio_cfg["pre_detection_buffer_sec"]
        # Fixed-mode post buffer (legacy path)
        self.post_buf_sec = audio_cfg["post_detection_sec"]

        # Adaptive capture parameters
        adaptive_cfg = audio_cfg.get("adaptive_capture", {}) or {}
        self.adaptive_enabled = adaptive_cfg.get("enabled", True)
        self.min_post_sec = adaptive_cfg.get("min_post_sec", 1.0)
        self.max_post_sec = adaptive_cfg.get("max_post_sec",
                                              self.post_buf_sec + 2.0)
        # End-of-utterance silence threshold, RMS in float32 [-1, 1]
        self.eos_silence_rms = adaptive_cfg.get("eos_silence_rms", 0.005)
        # Trailing silence duration before we stop capturing
        self.eos_silence_sec = adaptive_cfg.get("eos_silence_sec", 0.7)

        # Rolling buffer sized for pre-detection window
        buf_samples = int(self.pre_buf_sec * self.sample_rate)
        self._ring = collections.deque(maxlen=buf_samples)
        self._lock = threading.Lock()
        self._stream = None

        # Post-detection capture state
        self._recording = False
        self._post_buf: list[np.ndarray] = []
        self._post_samples_needed = 0      # legacy fixed mode
        self._post_done = threading.Event()

        # Adaptive-mode state
        self._adaptive = False
        self._eos_silence_frames_needed = 0
        self._eos_silence_frames_seen = 0
        self._eos_max_samples = 0
        self._eos_min_samples = 0
        self._eos_captured = 0

        # Native-rate / resampling state — set in start()
        self._native_rate = self.sample_rate
        self._resample_in = False
        self._resample_up = 1
        self._resample_down = 1
        self._resample_poly = None

        self._first_audio_logged = False

    # --- streaming --------------------------------------------------------

    def start(self):
        """Start continuous microphone capture.

        Many USB mics (incl. the DCMT lavalier) don't natively support 16kHz.
        We open the stream at the device's preferred rate and resample to
        `self.sample_rate` inside the callback before the ring buffer sees it.
        Downstream code (Vosk, quality, emotion) keeps operating at 16k.
        """
        import sounddevice as sd   # lazy — needs libportaudio2 on the system

        # Probe the device's preferred rate so the OpenStream call doesn't
        # reject 16k. If the device happens to support 16k natively, no
        # resampling is needed.
        native_rate = int(self.sample_rate)
        if self.device is not None:
            try:
                info = sd.query_devices(self.device, kind="input")
                native_rate = int(info["default_samplerate"])
            except Exception:
                logger.warning("Could not query device %s, defaulting to %d",
                                self.device, self.sample_rate)

        self._native_rate = native_rate
        self._resample_in = (native_rate != self.sample_rate)

        if self._resample_in:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(self.sample_rate, native_rate)
            self._resample_up = self.sample_rate // g
            self._resample_down = native_rate // g
            self._resample_poly = resample_poly
            logger.info("Audio input: mic native=%dHz, downsampling to %dHz (poly %d/%d)",
                         native_rate, self.sample_rate,
                         self._resample_up, self._resample_down)
        else:
            self._resample_poly = None

        self._stream = sd.InputStream(
            samplerate=native_rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            blocksize=int(native_rate * 0.064),  # ~64ms blocks
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio input started (native_sr=%d, target_sr=%d, dev=%s, adaptive=%s)",
                     native_rate, self.sample_rate, self.device, self.adaptive_enabled)

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio input stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio input status: %s", status)
        samples = indata[:, 0].copy()  # mono int16

        # Resample to target rate if the device's native rate differs.
        if self._resample_poly is not None:
            f32 = samples.astype(np.float32) / 32768.0
            resampled = self._resample_poly(f32, self._resample_up, self._resample_down)
            samples = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

        with self._lock:
            self._ring.extend(samples)
            if not self._first_audio_logged:
                self._first_audio_logged = True
                rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
                logger.info("Mic capture confirmed — first audio received (rms=%.6f)", rms)

            if not self._recording:
                return

            self._post_buf.append(samples.copy())
            self._eos_captured += len(samples)

            if self._adaptive:
                # Compute RMS of this block in float [-1, 1]
                rms = float(np.sqrt(np.mean(
                    (samples.astype(np.float32) / 32768.0) ** 2)))
                if rms < self.eos_silence_rms:
                    self._eos_silence_frames_seen += len(samples)
                else:
                    self._eos_silence_frames_seen = 0

                # Stop if: minimum reached AND (enough trailing silence OR hit max)
                hit_max = self._eos_captured >= self._eos_max_samples
                got_min = self._eos_captured >= self._eos_min_samples
                got_eos = self._eos_silence_frames_seen >= self._eos_silence_frames_needed
                if hit_max or (got_min and got_eos):
                    self._recording = False
                    self._post_done.set()
            else:
                if self._eos_captured >= self._post_samples_needed:
                    self._recording = False
                    self._post_done.set()

    # --- capture around detection -----------------------------------------

    def get_pre_buffer(self) -> np.ndarray:
        """Return a copy of the current rolling buffer (pre-detection audio)."""
        with self._lock:
            return np.array(list(self._ring), dtype=np.int16)

    def capture_post_detection(self, adaptive: bool | None = None) -> np.ndarray:
        """Capture post-detection audio.

        If `adaptive` is None, falls back to the class-level setting.
        Adaptive mode keeps recording until the user stops talking (or hits
        max_post_sec). Fixed mode records exactly post_buf_sec.
        """
        use_adaptive = self.adaptive_enabled if adaptive is None else adaptive

        with self._lock:
            self._adaptive = use_adaptive
            self._post_buf = []
            self._eos_captured = 0
            self._eos_silence_frames_seen = 0
            self._post_done.clear()

            if use_adaptive:
                self._eos_min_samples = int(self.min_post_sec * self.sample_rate)
                self._eos_max_samples = int(self.max_post_sec * self.sample_rate)
                self._eos_silence_frames_needed = int(
                    self.eos_silence_sec * self.sample_rate)
                self._post_samples_needed = 0
            else:
                self._post_samples_needed = int(self.post_buf_sec * self.sample_rate)
                self._eos_max_samples = self._post_samples_needed

            self._recording = True

        # Wait up to the maximum possible duration + slack
        timeout = (self.max_post_sec if use_adaptive else self.post_buf_sec) + 2.0
        self._post_done.wait(timeout=timeout)
        with self._lock:
            self._recording = False
            buf = list(self._post_buf)

        if buf:
            captured = np.concatenate(buf)
            mode = "adaptive" if use_adaptive else "fixed"
            logger.info("Post-detection captured (%s): %d samples (%.2fs)",
                         mode, len(captured), len(captured) / self.sample_rate)
            return captured
        return np.array([], dtype=np.int16)

    def get_current_chunk(self, duration_sec: float = 0.1) -> np.ndarray:
        """Return the most recent N seconds from the rolling buffer."""
        n = int(duration_sec * self.sample_rate)
        with self._lock:
            buf = list(self._ring)
        if len(buf) >= n:
            return np.array(buf[-n:], dtype=np.int16)
        return np.array(buf, dtype=np.int16)
