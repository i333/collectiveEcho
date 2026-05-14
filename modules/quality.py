"""
Audio quality analysis — RMS, clipping, noise floor, duration.

V2 changes:
  - Transcript verification is delegated to a pluggable ASR backend
    (modules.asr) via dependency injection. No more in-method Vosk imports.
  - Emotion features (V/A/D) flow through the QualityReport via the
    EmotionReport — quality_score is independent of emotion, but downstream
    response selection can read both off the report.
  - Phrase matching uses PhraseMatcher (phonetic + Levenshtein).
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from modules.phrase_match import PhraseMatcher
from modules.asr import BaseTranscriber, TranscriptResult
from modules.emotion import EmotionReport, BaseEmotionAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    rms: float = 0.0
    peak: float = 0.0
    clipping_ratio: float = 0.0
    noise_floor_rms: float = 0.0
    duration_sec: float = 0.0
    intensity_std: float = 0.0
    score: float = 0.0
    is_intimate: bool = False
    is_strong: bool = False
    is_usable: bool = True
    has_phrase: bool = False
    has_too: bool = False                 # legacy flag, kept for back-compat
    matched_bucket: str = ""              # bucket id from PhraseMatcher (e.g. "ily_sandy")
    transcript: str = ""
    transcript_confidence: float = 0.0
    phrase_match_confidence: float = 0.0
    phrase_start_sec: float = -1.0
    phrase_end_sec: float = -1.0
    reject_reasons: list[str] = field(default_factory=list)
    # New: emotion vector (continuous) — populated when an analyzer is wired
    emotion: EmotionReport = field(default_factory=EmotionReport)
    # Acoustic-quality additions
    snr_db: float = 0.0                   # signal-to-noise ratio (dB)
    spectral_centroid_hz: float = 0.0     # brightness of the clip
    spectral_tilt_db_per_khz: float = 0.0 # spectral balance: -ve = darker
    phrase_only_rms: float = 0.0          # RMS measured on the phrase region only


class QualityAnalyzer:
    def __init__(self, cfg: dict,
                 transcriber: BaseTranscriber | None = None,
                 emotion_analyzer: BaseEmotionAnalyzer | None = None,
                 matcher: PhraseMatcher | None = None):
        t = cfg["thresholds"]
        self.intimate_rms = t["intimate_rms"]
        self.strong_rms = t["strong_rms"]
        self.noise_floor = t["noise_floor_rms"]
        self.max_clip_ratio = t["max_clipping_ratio"]
        self.min_dur = t["min_duration_sec"]
        self.max_dur = t["max_duration_sec"]

        q = cfg["quality"]
        self.save_threshold = q["save_threshold"]
        self.unusable_threshold = q["unusable_threshold"]

        self.sample_rate = cfg["audio"]["sample_rate"]

        # Trim padding around the located phrase
        self.trim_pad_before = 0.15
        self.trim_pad_after = 0.25

        # Pluggable backends — injected so factory wiring is in one place
        self.transcriber = transcriber
        self.emotion_analyzer = emotion_analyzer

        # Use the orchestrator-provided matcher so quality-stage matches the
        # same configured buckets as the wake detector. Falls back to defaults.
        self.matcher = matcher or PhraseMatcher()

    # --- main entry -------------------------------------------------------

    def analyze(self, audio: np.ndarray, sample_rate: int) -> QualityReport:
        report = QualityReport()
        if len(audio) == 0:
            report.is_usable = False
            report.reject_reasons.append("empty")
            return report

        # Float [-1, 1]
        if audio.dtype == np.int16:
            samples = audio.astype(np.float32) / 32768.0
        else:
            samples = audio.astype(np.float32)

        report.duration_sec = len(samples) / sample_rate
        report.rms = float(np.sqrt(np.mean(samples ** 2)))
        report.peak = float(np.max(np.abs(samples)))

        clip_threshold = 0.99
        clipped = int(np.sum(np.abs(samples) >= clip_threshold))
        report.clipping_ratio = clipped / len(samples) if len(samples) > 0 else 0.0

        # Noise floor + intensity variation
        frame_len = int(0.02 * sample_rate)
        frame_rms = None
        if len(samples) >= frame_len:
            n_frames = len(samples) // frame_len
            frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            bottom = np.sort(frame_rms)[: max(1, n_frames // 10)]
            report.noise_floor_rms = float(np.mean(bottom))
            report.intensity_std = float(np.std(frame_rms))

            # SNR (dB) — speech RMS / noise RMS
            top = np.sort(frame_rms)[-max(1, int(n_frames * 0.3)):]
            speech_rms = float(np.mean(top))
            if report.noise_floor_rms > 1e-7:
                report.snr_db = 20 * float(np.log10(speech_rms / report.noise_floor_rms))
            else:
                report.snr_db = 60.0

        # Spectral features (centroid, tilt)
        if len(samples) >= 1024:
            from modules.clip_enhancer import _spectral_features
            report.spectral_centroid_hz, report.spectral_tilt_db_per_khz = \
                _spectral_features(samples, sample_rate)

        # --- Transcribe via backend ---
        self._verify_transcript(audio, sample_rate, report)

        # --- Emotion analysis ---
        if self.emotion_analyzer is not None:
            try:
                report.emotion = self.emotion_analyzer.analyze(audio, sample_rate)
            except Exception:
                logger.exception("Emotion analysis failed — leaving defaults")

        # If we found the phrase boundaries, compute phrase-only RMS
        if report.has_phrase and report.phrase_start_sec >= 0:
            ps = int(report.phrase_start_sec * sample_rate)
            pe = int(report.phrase_end_sec * sample_rate)
            phrase_samples = samples[ps:pe]
            if len(phrase_samples) > 0:
                phrase_rms = float(np.sqrt(np.mean(phrase_samples ** 2)))
                report.phrase_only_rms = phrase_rms
                report.rms = phrase_rms

        report.is_intimate = report.rms < self.intimate_rms
        report.is_strong = report.rms >= self.strong_rms

        # --- Quality score ---
        score = 1.0
        reasons = []

        if not report.has_phrase:
            score -= 0.6
            reasons.append("no_phrase_detected")
        if report.rms < self.noise_floor:
            score -= 0.5
            reasons.append("below_noise_floor")
        if report.clipping_ratio > self.max_clip_ratio:
            score -= 0.3
            reasons.append("clipping")
        if report.duration_sec < self.min_dur:
            score -= 0.3
            reasons.append("too_short")
        if report.duration_sec > self.max_dur:
            score -= 0.2
            reasons.append("too_long")
        if 0.03 < report.rms < 0.9:
            score += 0.1

        # SNR-based scoring — the single most important quality signal we
        # weren't measuring before. <6dB: definitely unusable. >18dB: clean.
        if report.snr_db < 6.0:
            score -= 0.35
            reasons.append(f"low_snr_{report.snr_db:.1f}dB")
        elif report.snr_db < 12.0:
            score -= 0.1
            reasons.append(f"mediocre_snr_{report.snr_db:.1f}dB")
        elif report.snr_db > 18.0:
            score += 0.1   # reward clean recordings

        # Spectral tilt — very dark recordings (-15 dB/kHz or worse) usually
        # mean the mic was muffled or the speaker was off-axis. Penalize.
        if report.spectral_tilt_db_per_khz < -15.0:
            score -= 0.1
            reasons.append("muffled")

        report.score = max(0.0, min(1.0, score))
        report.reject_reasons = reasons
        report.is_usable = report.score >= self.unusable_threshold

        logger.info(
            "Quality: rms=%.4f snr=%.1fdB tilt=%+.1fdB/kHz clip=%.4f dur=%.2fs "
            "score=%.2f intimate=%s strong=%s phrase=%s bucket=%s emo=%s V=%+.2f A=%.2f "
            "transcript='%s' reasons=%s",
            report.rms, report.snr_db, report.spectral_tilt_db_per_khz,
            report.clipping_ratio, report.duration_sec, report.score,
            report.is_intimate, report.is_strong,
            report.has_phrase, report.matched_bucket or "-",
            report.emotion.label, report.emotion.valence, report.emotion.arousal,
            report.transcript, reasons,
        )
        return report

    # --- transcript path --------------------------------------------------

    def _verify_transcript(self, audio: np.ndarray, sample_rate: int,
                            report: QualityReport):
        if self.transcriber is None:
            report.has_phrase = True  # fail open
            return
        try:
            tr: TranscriptResult = self.transcriber.transcribe(audio, sample_rate)
        except Exception:
            logger.exception("Transcribe failed — failing open")
            report.has_phrase = True
            return

        report.transcript = tr.text
        report.transcript_confidence = tr.confidence
        if not tr.text:
            return

        hit = self.matcher.find(tr.text)
        if hit is not None:
            report.has_phrase = True
            report.matched_bucket = hit.phrase_type
            report.has_too = hit.phrase_type == "ily_too"
            report.phrase_match_confidence = hit.confidence
            if tr.words:
                self._locate_phrase_timing(tr.words, report)
        else:
            logger.info("Phrase not found in transcript: '%s'", tr.text)

    def _locate_phrase_timing(self, words, report: QualityReport):
        """Walk word list to find the phrase span; cache start/end timings.

        Extends the span past 'you' to include either 'too/to/two' or any
        per-bucket extra anchor word (e.g. 'sandy') so trimming preserves
        the trailing name in the recording.
        """
        texts = [w.text for w in words]
        # Build a "name extension" set from the matched bucket's anchor_extra.
        bucket = self.matcher.get_bucket(report.matched_bucket) if report.matched_bucket else None
        extra_anchors = set((bucket.anchor_extra if bucket else []) or [])

        for i in range(len(texts) - 2):
            if (texts[i] == "i"
                    and texts[i + 1] == "love"
                    and texts[i + 2] in ("you", "u", "yo", "ya", "yew")):
                report.phrase_start_sec = max(0, words[i].start - self.trim_pad_before)
                end_idx = i + 2
                # Look one word past 'you' for "too/to" OR a bucket anchor name.
                if i + 3 < len(texts):
                    nxt = texts[i + 3]
                    if nxt in ("too", "to", "two"):
                        end_idx = i + 3
                        report.has_too = True
                    elif nxt in extra_anchors:
                        end_idx = i + 3
                report.phrase_end_sec = words[end_idx].end + self.trim_pad_after
                return
        # Fallback: locate around 'love'
        for i, w in enumerate(texts):
            if w == "love":
                start_idx = max(0, i - 1)
                end_idx = min(len(words) - 1, i + 1)
                report.phrase_start_sec = max(0, words[start_idx].start - self.trim_pad_before)
                report.phrase_end_sec = words[end_idx].end + self.trim_pad_after
                return

    # --- trim and gates ---------------------------------------------------

    def trim_to_phrase(self, audio: np.ndarray, sample_rate: int,
                       report: QualityReport) -> np.ndarray:
        if report.phrase_start_sec < 0 or report.phrase_end_sec < 0:
            return audio
        start = int(report.phrase_start_sec * sample_rate)
        end = int(report.phrase_end_sec * sample_rate)
        end = min(end, len(audio))
        if start >= end:
            return audio
        trimmed = audio[start:end]
        logger.info("Trimmed clip: %.2fs → %.2fs (%.2fs)",
                     report.phrase_start_sec, report.phrase_end_sec,
                     len(trimmed) / sample_rate)
        return trimmed

    def should_save(self, report: QualityReport) -> bool:
        return report.has_phrase and report.score >= self.save_threshold

    def should_use_fallback(self, report: QualityReport) -> bool:
        return not report.is_usable
