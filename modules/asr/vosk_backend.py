"""Vosk ASR backend — wraps the existing transcript verification path."""

import json
import logging
import os

import numpy as np

from .base import BaseTranscriber
from .types import TranscriptResult, Word

logger = logging.getLogger(__name__)


class VoskTranscriber(BaseTranscriber):
    def __init__(self, cfg: dict):
        vcfg = cfg.get("vosk", {})
        self.model_path = vcfg.get("model_path", "models/vosk-model-small-en-us-0.15")
        self.sample_rate = cfg["audio"]["sample_rate"]
        self._model = None
        if not os.path.exists(self.model_path):
            logger.warning("Vosk model not found at %s — transcribe() will be a no-op",
                           self.model_path)

    def warmup(self):
        if self._model is not None or not os.path.exists(self.model_path):
            return
        try:
            from vosk import Model
            self._model = Model(self.model_path)
            logger.info("Vosk transcriber loaded: %s", self.model_path)
        except Exception:
            logger.exception("Could not load Vosk model")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptResult:
        if self._model is None:
            self.warmup()
        if self._model is None:
            # Model missing — fail open with empty result so caller can decide
            return TranscriptResult(text="", confidence=0.0)

        try:
            from vosk import KaldiRecognizer
        except ImportError:
            logger.error("vosk not installed")
            return TranscriptResult(text="", confidence=0.0)

        # Vosk needs int16 PCM
        if audio.dtype != np.int16:
            audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_i16 = audio

        rec = KaldiRecognizer(self._model, sample_rate)
        rec.SetWords(True)
        chunk_size = 4000
        for i in range(0, len(audio_i16), chunk_size):
            rec.AcceptWaveform(audio_i16[i:i + chunk_size].tobytes())

        result = json.loads(rec.FinalResult())
        text = (result.get("text") or "").lower().strip()
        words = [
            Word(text=w.get("word", "").lower(),
                 start=float(w.get("start", 0.0)),
                 end=float(w.get("end", 0.0)),
                 confidence=float(w.get("conf", 1.0)))
            for w in result.get("result", [])
        ]
        # Average word confidence as a coarse overall confidence
        if words:
            avg_conf = float(np.mean([w.confidence for w in words]))
        else:
            avg_conf = 0.0 if not text else 0.5
        return TranscriptResult(
            text=text,
            language="en",
            confidence=avg_conf,
            words=words,
        )
