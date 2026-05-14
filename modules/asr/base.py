"""Base class for ASR backends."""

import abc

import numpy as np

from .types import TranscriptResult


class BaseTranscriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptResult:
        """Return a transcript with word-level timings if available."""

    def warmup(self):
        """Optional: pre-load models. Called once at startup."""
        return
