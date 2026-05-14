"""
Wake-word detection — swappable backends.

Backends:
  - SimulatedDetector: stdin-based trigger for testing
  - VoskDetector: offline speech recognition, matches "i love you" in transcript
  - PorcupineDetector: Picovoice Porcupine for offline wake-word
"""

import abc
import json
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)


class BaseDetector(abc.ABC):
    """Interface every detector must implement."""

    @abc.abstractmethod
    def start(self, on_detected: callable, audio_input=None):
        """Begin listening. Call on_detected(phrase) when wake phrase is heard.
        phrase is 'ily' for 'I love you' or 'ily_too' for 'I love you too'."""

    @abc.abstractmethod
    def stop(self):
        """Stop listening and release resources."""


# --------------------------------------------------------------------------
# Simulated detector — press Enter to trigger
# --------------------------------------------------------------------------

class SimulatedDetector(BaseDetector):
    def __init__(self, cfg: dict):
        self._running = False
        self._thread = None
        self._callback = None

    def start(self, on_detected: callable, audio_input=None):
        self._callback = on_detected
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Simulated detector started — press Enter to trigger")

    def _loop(self):
        while self._running:
            try:
                line = sys.stdin.readline()
                if not self._running:
                    break
                if line is not None:
                    text = line.strip().lower()
                    phrase = "ily_too" if "too" in text else "ily"
                    logger.info("Simulated detection triggered (%s)", phrase)
                    if self._callback:
                        self._callback(phrase)
            except EOFError:
                break

    def stop(self):
        self._running = False
        logger.info("Simulated detector stopped")


# --------------------------------------------------------------------------
# Vosk detector — reads from the shared AudioInput rolling buffer
# --------------------------------------------------------------------------

class VoskDetector(BaseDetector):
    """
    Listens continuously using Vosk offline speech recognition.
    Reads audio from the shared AudioInput buffer (same mic stream
    used for recording) so detection and capture are always in sync.
    """

    # "too" variants must be checked FIRST (longer match wins)
    TOO_PHRASES = [
        "i love you too",
        "i love u too",
        "i love you to",
        "i love u to",
    ]
    TRIGGER_PHRASES = [
        "i love you",
        "i love u",
        "i love yo",
    ]

    def __init__(self, cfg: dict):
        vosk_cfg = cfg.get("vosk", {})
        self.model_path = vosk_cfg.get("model_path", "models/vosk-model-small-en-us-0.15")
        self.sample_rate = cfg["audio"]["sample_rate"]
        self._cooldown_sec = vosk_cfg.get("cooldown_sec", 3.0)
        self._running = False
        self._thread = None
        self._callback = None
        self._audio_input = None
        self._last_trigger = 0.0

    def start(self, on_detected: callable, audio_input=None):
        try:
            from vosk import Model, KaldiRecognizer
        except ImportError:
            raise RuntimeError("Vosk backend requires vosk. Install: pip install vosk")

        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"Vosk model not found at {self.model_path}. "
                "Download from https://alphacephei.com/vosk/models"
            )

        if audio_input is None:
            raise RuntimeError("VoskDetector requires a shared AudioInput instance")

        self._audio_input = audio_input
        self._callback = on_detected
        self._model = Model(self.model_path)
        self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
        self._recognizer.SetWords(False)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Vosk detector started (model=%s, sr=%d, shared audio)",
                     self.model_path, self.sample_rate)

    def _loop(self):
        """Poll the shared AudioInput for new audio chunks and feed to Vosk."""
        poll_sec = 0.15  # how often to grab audio
        chunk_duration = 0.2  # seconds of audio to grab each time

        # Wait for audio input to start producing data
        time.sleep(0.5)

        while self._running:
            try:
                chunk = self._audio_input.get_current_chunk(chunk_duration)
                if len(chunk) == 0:
                    time.sleep(poll_sec)
                    continue

                audio_bytes = chunk.tobytes()

                if self._recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self._check_trigger(text, "final")
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if text:
                        self._check_trigger(text, "partial")

                time.sleep(poll_sec)

            except Exception:
                if self._running:
                    logger.exception("Vosk processing error")
                    time.sleep(0.5)

    def _check_trigger(self, text: str, result_type: str):
        text_lower = text.lower().strip()

        # Check "too" variants first (longer match wins)
        for phrase in self.TOO_PHRASES:
            if phrase in text_lower:
                self._fire_trigger(text_lower, phrase, "ily_too", result_type)
                return

        for phrase in self.TRIGGER_PHRASES:
            if phrase in text_lower:
                self._fire_trigger(text_lower, phrase, "ily", result_type)
                return

    def _fire_trigger(self, text: str, phrase: str, phrase_type: str, result_type: str):
        now = time.time()
        if now - self._last_trigger < self._cooldown_sec:
            logger.debug("Vosk trigger suppressed (cooldown): '%s'", text)
            return
        self._last_trigger = now
        logger.info("Vosk detected '%s' [%s] in %s result: '%s'",
                    phrase, phrase_type, result_type, text)
        if self._callback:
            self._callback(phrase_type)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("Vosk detector stopped")


# --------------------------------------------------------------------------
# Porcupine detector
# --------------------------------------------------------------------------

class PorcupineDetector(BaseDetector):
    """
    Offline wake-word detection using Picovoice Porcupine.
    Requires pvporcupine and pyaudio.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg.get("porcupine", {})
        self._running = False
        self._thread = None
        self._callback = None
        self._porcupine = None
        self._pa = None
        self._audio_stream = None

    def start(self, on_detected: callable, audio_input=None):
        try:
            import pvporcupine
            import pyaudio
        except ImportError:
            raise RuntimeError(
                "Porcupine backend requires pvporcupine and pyaudio. "
                "Install them: pip install pvporcupine pyaudio"
            )

        self._callback = on_detected
        access_key = self._cfg.get("access_key", "")
        keyword_path = self._cfg.get("keyword_path", "")
        model_path = self._cfg.get("model_path") or None
        sensitivity = self._cfg.get("sensitivity", 0.5)

        if keyword_path:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[keyword_path],
                model_path=model_path,
                sensitivities=[sensitivity],
            )
        else:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=["ok google"],  # placeholder
                sensitivities=[sensitivity],
            )
            logger.warning(
                "No custom keyword_path set — using placeholder keyword. "
                "Train a custom 'I love you' keyword at console.picovoice.ai"
            )

        self._pa = pyaudio.PyAudio()
        self._audio_stream = self._pa.open(
            rate=self._porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._porcupine.frame_length,
        )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Porcupine detector started")

    def _loop(self):
        import struct

        while self._running:
            try:
                pcm = self._audio_stream.read(
                    self._porcupine.frame_length, exception_on_overflow=False
                )
                pcm_unpacked = struct.unpack_from(
                    "h" * self._porcupine.frame_length, pcm
                )
                result = self._porcupine.process(pcm_unpacked)
                if result >= 0:
                    logger.info("Porcupine detected wake word (index=%d)", result)
                    if self._callback:
                        self._callback()
            except Exception:
                if self._running:
                    logger.exception("Porcupine processing error")

    def stop(self):
        self._running = False
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        if self._pa is not None:
            self._pa.terminate()
        if self._porcupine is not None:
            self._porcupine.delete()
        logger.info("Porcupine detector stopped")


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def create_detector(cfg: dict) -> BaseDetector:
    backend = cfg.get("wake_detector", "simulate")
    if backend == "simulate":
        return SimulatedDetector(cfg)
    elif backend == "vosk":
        return VoskDetector(cfg)
    elif backend == "porcupine":
        return PorcupineDetector(cfg)
    else:
        raise ValueError(f"Unknown wake_detector backend: {backend}")
