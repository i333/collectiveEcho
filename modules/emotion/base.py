"""Base class for emotion analyzers."""

import abc

import numpy as np

from .types import EmotionReport


class BaseEmotionAnalyzer(abc.ABC):
    """Every backend implements analyze(audio, sr) → EmotionReport."""

    @abc.abstractmethod
    def analyze(self, audio: np.ndarray, sample_rate: int) -> EmotionReport:
        """Return an EmotionReport for the clip."""

    def warmup(self):
        """Optional: pre-load models / compile kernels. Called once at startup."""
        return
