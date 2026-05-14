"""
Emotion analysis — pluggable backends.

Backends:
  - prosody: pitch/energy/rate heuristics → valence/arousal proxies (CPU, no ML deps)
  - wav2vec_ser: HuggingFace audeering Wav2Vec2 SER model (Jetson GPU, ~1.3GB model)

Common interface (EmotionAnalyzer):
  .analyze(audio: np.ndarray, sr: int) -> EmotionReport

EmotionReport carries continuous valence/arousal/dominance in [-1, 1] / [0, 1]
plus a categorical label and a confidence. The categorical label is derived
from the dimensional values when not available natively from the backend.
"""

from .types import EmotionReport
from .base import BaseEmotionAnalyzer
from .prosody import ProsodyEmotion


def create_emotion_analyzer(cfg: dict) -> BaseEmotionAnalyzer:
    """Factory: pick an emotion backend based on config.emotion.backend."""
    backend = cfg.get("emotion", {}).get("backend", "prosody")
    if backend == "prosody":
        return ProsodyEmotion(cfg)
    if backend in ("wav2vec_ser", "ser", "wav2vec2"):
        try:
            from .wav2vec_ser import Wav2Vec2Emotion
        except ImportError as e:
            raise RuntimeError(
                f"wav2vec_ser backend needs torch + transformers (not yet installed): {e}. "
                "Install on the Jetson once the bigger SD card is in, then flip "
                "config.emotion.backend back to 'wav2vec_ser'. Falling back to 'prosody'."
            )
        return Wav2Vec2Emotion(cfg)
    raise ValueError(f"Unknown emotion backend: {backend}")


__all__ = [
    "EmotionReport",
    "BaseEmotionAnalyzer",
    "ProsodyEmotion",
    "create_emotion_analyzer",
]
