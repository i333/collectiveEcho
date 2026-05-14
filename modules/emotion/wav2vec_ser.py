"""
Wav2Vec2 SER backend — Jetson GPU emotion classification.

Stub for now. Activates when torch + transformers are installed (waiting for
the bigger SD card). Default model:
  audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim

Outputs continuous valence / arousal / dominance in [-1, 1] / [0, 1], so it
plugs into EmotionReport without any mapping changes.

To enable tomorrow:
  pip install torch torchaudio transformers
  Set config.emotion.backend: wav2vec_ser
"""

import logging

import numpy as np

from .base import BaseEmotionAnalyzer
from .types import EmotionReport

logger = logging.getLogger(__name__)


class Wav2Vec2Emotion(BaseEmotionAnalyzer):
    DEFAULT_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def __init__(self, cfg: dict):
        ec = cfg.get("emotion", {})
        self.model_name = ec.get("model", self.DEFAULT_MODEL)
        self.device = ec.get("device", "cuda")
        self.sample_rate_model = 16000

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Wav2Vec2Emotion requires torch + transformers: {e}"
            )

        self._model = None
        self._processor = None

    def warmup(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification

        logger.info("Loading SER model: %s on %s", self.model_name, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda").eval().half()
        else:
            self._model = self._model.eval()
        logger.info("SER model ready")

    def analyze(self, audio: np.ndarray, sample_rate: int) -> EmotionReport:
        import torch

        if self._model is None:
            self.warmup()

        # Convert to float32 mono [-1, 1]
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        else:
            x = audio.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)

        # Resample if needed (audeering model is 16k)
        if sample_rate != self.sample_rate_model:
            import torchaudio
            x_t = torch.from_numpy(x).unsqueeze(0)
            x_t = torchaudio.functional.resample(x_t, sample_rate, self.sample_rate_model)
            x = x_t.squeeze(0).numpy()

        inputs = self._processor(x, sampling_rate=self.sample_rate_model,
                                  return_tensors="pt", padding=True)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda").half() if v.dtype == torch.float else v.to("cuda")
                      for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model(**inputs)
        # audeering model returns 3-vector: arousal, dominance, valence in [0, 1]
        # We remap valence to [-1, 1] for our convention.
        vec = out.logits.squeeze().detach().float().cpu().numpy()
        arousal = float(np.clip(vec[0], 0.0, 1.0))
        dominance = float(np.clip(vec[1], 0.0, 1.0))
        valence_01 = float(np.clip(vec[2], 0.0, 1.0))
        valence = valence_01 * 2.0 - 1.0

        from .prosody import _label_from_dims
        label = _label_from_dims(valence, arousal)

        rep = EmotionReport(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            label=label,
            confidence=0.9,
        )
        logger.info(
            "Emotion[SER]: %s (V=%+.2f A=%.2f D=%.2f)",
            rep.label, rep.valence, rep.arousal, rep.dominance,
        )
        return rep
