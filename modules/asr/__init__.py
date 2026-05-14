"""
ASR (speech recognition) — pluggable backends.

Backends:
  - vosk: offline Kaldi-based, small footprint, Pi-friendly (current default)
  - whisper: faster-whisper (CTranslate2) on Jetson GPU — drops in tomorrow

Common interface (BaseTranscriber):
  .transcribe(audio: np.ndarray, sr: int) -> TranscriptResult

TranscriptResult carries text, word-level timings, language, and a confidence.
"""

from .types import TranscriptResult, Word
from .base import BaseTranscriber
from .vosk_backend import VoskTranscriber


def create_transcriber(cfg: dict) -> BaseTranscriber:
    """Factory: pick an ASR backend based on config.asr.backend."""
    backend = cfg.get("asr", {}).get("backend", "vosk")
    if backend == "vosk":
        return VoskTranscriber(cfg)
    if backend in ("whisper", "faster_whisper", "faster-whisper"):
        try:
            from .whisper_backend import WhisperTranscriber
        except ImportError as e:
            raise RuntimeError(
                f"whisper backend needs faster-whisper (not yet installed): {e}. "
                "Install on the Jetson with: pip install faster-whisper. "
                "Then set config.asr.backend: whisper."
            )
        return WhisperTranscriber(cfg)
    raise ValueError(f"Unknown asr backend: {backend}")


__all__ = [
    "TranscriptResult",
    "Word",
    "BaseTranscriber",
    "VoskTranscriber",
    "create_transcriber",
]
