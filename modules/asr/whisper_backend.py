"""
faster-whisper ASR backend — Jetson GPU streaming-class transcription.

Stub for now. Activates once `faster-whisper` is installed (waiting on the
bigger SD card). Default model: distil-small.en (~250MB, very fast on Orin).

To enable tomorrow:
  pip install faster-whisper
  Set config.asr.backend: whisper
  Optionally: config.whisper.model_size: distil-small.en | small | medium
"""

import logging

import numpy as np

from .base import BaseTranscriber
from .types import TranscriptResult, Word

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber):
    def __init__(self, cfg: dict):
        wcfg = cfg.get("whisper", {})
        self.model_size = wcfg.get("model_size", "distil-small.en")
        self.device = wcfg.get("device", "cuda")
        self.compute_type = wcfg.get("compute_type", "int8_float16")
        self.language = wcfg.get("language", "en")
        self.beam_size = wcfg.get("beam_size", 1)
        self.vad_filter = wcfg.get("vad_filter", True)

        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError as e:
            raise ImportError(f"WhisperTranscriber requires faster-whisper: {e}")

        self._model = None

    def warmup(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        logger.info("Loading Whisper model: %s on %s (%s)",
                    self.model_size, self.device, self.compute_type)
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        logger.info("Whisper model ready")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptResult:
        if self._model is None:
            self.warmup()

        # faster-whisper wants float32 mono at 16kHz
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        else:
            x = audio.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sample_rate != 16000:
            # Quick poly-resample via numpy interp — fine for ASR
            ratio = 16000 / sample_rate
            n_out = int(len(x) * ratio)
            x = np.interp(np.linspace(0, len(x), n_out, endpoint=False),
                          np.arange(len(x)), x).astype(np.float32)

        segments, info = self._model.transcribe(
            x,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=True,
        )
        text_parts = []
        words: list[Word] = []
        seg_list = []
        for seg in segments:
            text_parts.append(seg.text)
            seg_list.append((seg.start, seg.end, seg.text))
            if seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=(w.word or "").strip().lower(),
                        start=float(w.start or 0.0),
                        end=float(w.end or 0.0),
                        confidence=float(getattr(w, "probability", 1.0) or 1.0),
                    ))
        text = " ".join(p.strip() for p in text_parts).strip().lower()
        confidence = (
            float(np.mean([w.confidence for w in words])) if words else 0.5
        )
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", self.language) or self.language,
            confidence=confidence,
            words=words,
            segments=seg_list,
        )
