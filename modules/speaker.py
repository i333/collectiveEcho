"""
Speaker identification — diarize the chorus pool so a single voice can't
dominate a response.

Uses SpeechBrain's ECAPA-TDNN encoder (192-d embeddings, VoxCeleb-trained).
Online clustering: each clip is matched against the running per-speaker
centroid; if cosine similarity clears `similarity_threshold`, the clip joins
that speaker, otherwise a new speaker id is minted.

The model and threshold are configured under config.yaml `speaker:`.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerIdentifier:
    DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    SAMPLE_RATE_MODEL = 16000

    def __init__(self, cfg: dict):
        sc = cfg.get("speaker", {}) or {}
        self.enabled: bool = sc.get("enabled", True)
        self.model_name: str = sc.get("model", self.DEFAULT_MODEL)
        self.device: str = sc.get("device", "cuda")
        # Cosine similarity above which two clips are deemed the same person.
        # ECAPA on VoxCeleb has EER around cosine ≈ 0.25 (sim ≈ 0.75), so 0.65
        # is a slightly loose default — fine for in-room re-recordings, still
        # strict enough that visiting voices don't collapse into one cluster.
        self.similarity_threshold: float = float(sc.get("similarity_threshold", 0.65))
        # Where SpeechBrain caches the downloaded checkpoint.
        self.savedir: str = sc.get("savedir", "models/speaker_ecapa_voxceleb")
        self._encoder = None

    def warmup(self):
        """Load the ECAPA encoder once. Safe to call repeatedly."""
        if not self.enabled or self._encoder is not None:
            return
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                f"SpeakerIdentifier requires speechbrain + torch: {e}"
            )
        # SpeechBrain's parser wants "cuda:N" or "cpu" — bare "cuda" warns.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device = self.device if ":" in self.device else "cuda:0"
        else:
            device = "cpu"
        logger.info("Loading speaker encoder: %s on %s", self.model_name, device)
        self._encoder = EncoderClassifier.from_hparams(
            source=self.model_name,
            run_opts={"device": device},
            savedir=self.savedir,
        )
        logger.info("Speaker encoder ready (threshold cos≥%.2f)",
                     self.similarity_threshold)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a 192-d L2-normalized speaker embedding as float32."""
        if self._encoder is None:
            self.warmup()
        import torch

        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        else:
            x = audio.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)

        if sample_rate != self.SAMPLE_RATE_MODEL:
            import torchaudio
            x = torchaudio.functional.resample(
                torch.from_numpy(x).unsqueeze(0),
                sample_rate, self.SAMPLE_RATE_MODEL,
            ).squeeze(0).numpy()

        wav = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            emb = self._encoder.encode_batch(wav).squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float32, copy=False)
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
        return emb

    def assign(self, embedding: np.ndarray,
                centroids: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
        """Return (speaker_id, best_similarity).

        Picks the existing speaker whose centroid is closest in cosine space
        if that distance clears `similarity_threshold`; otherwise mints a new
        UUID-based id.
        """
        if not centroids:
            return self._new_id(), 0.0
        best_id, best_sim = max(
            ((sid, _cosine(embedding, c)) for sid, c in centroids),
            key=lambda x: x[1],
        )
        if best_sim >= self.similarity_threshold:
            return best_id, best_sim
        return self._new_id(), best_sim

    @staticmethod
    def _new_id() -> str:
        return f"spk_{uuid.uuid4().hex[:8]}"
