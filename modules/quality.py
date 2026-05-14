"""
Audio quality analysis — RMS, clipping, noise floor, duration, pitch intensity.
Includes transcript verification via Vosk to confirm "I love you" is present,
and phrase trimming to isolate just the spoken phrase.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    rms: float = 0.0
    peak: float = 0.0
    clipping_ratio: float = 0.0
    noise_floor_rms: float = 0.0
    duration_sec: float = 0.0
    mean_pitch_hz: float = 0.0        # 0 = not computed
    intensity_std: float = 0.0
    score: float = 0.0                 # overall 0-1
    is_intimate: bool = False
    is_strong: bool = False
    is_usable: bool = True
    has_phrase: bool = False
    has_too: bool = False              # "I love you too" detected
    transcript: str = ""
    phrase_start_sec: float = -1.0
    phrase_end_sec: float = -1.0
    reject_reasons: list[str] = field(default_factory=list)


class QualityAnalyzer:
    TRIGGER_WORDS = ["i", "love", "you"]
    PHRASE_VARIANTS = ["i love you", "i love u", "i love yo"]
    TOO_VARIANTS = ["i love you too", "i love u too", "i love you to", "i love u to"]

    def __init__(self, cfg: dict):
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

        # Padding around the phrase when trimming (seconds)
        self.trim_pad_before = 0.15
        self.trim_pad_after = 0.25

        # Load Vosk model for transcript verification
        self._recognizer_model = None
        vosk_cfg = cfg.get("vosk", {})
        model_path = vosk_cfg.get("model_path", "models/vosk-model-small-en-us-0.15")
        if os.path.exists(model_path):
            try:
                from vosk import Model
                self._recognizer_model = Model(model_path)
                logger.info("Quality analyzer: Vosk model loaded for transcript verification")
            except Exception:
                logger.warning("Could not load Vosk model for quality check — skipping transcript verification")

    def analyze(self, audio: np.ndarray, sample_rate: int) -> QualityReport:
        """Analyze a clip and return a QualityReport."""
        report = QualityReport()

        if len(audio) == 0:
            report.is_usable = False
            report.reject_reasons.append("empty")
            return report

        # Normalize to float [-1, 1]
        if audio.dtype == np.int16:
            samples = audio.astype(np.float32) / 32768.0
        else:
            samples = audio.astype(np.float32)

        report.duration_sec = len(samples) / sample_rate

        # RMS
        report.rms = float(np.sqrt(np.mean(samples ** 2)))

        # Peak
        report.peak = float(np.max(np.abs(samples)))

        # Clipping — samples at or very near max
        clip_threshold = 0.99
        clipped = np.sum(np.abs(samples) >= clip_threshold)
        report.clipping_ratio = float(clipped / len(samples)) if len(samples) > 0 else 0.0

        # Noise floor — RMS of quietest 10% of short frames
        frame_len = int(0.02 * sample_rate)  # 20ms frames
        frame_rms = None
        if len(samples) >= frame_len:
            n_frames = len(samples) // frame_len
            frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            bottom = np.sort(frame_rms)[: max(1, n_frames // 10)]
            report.noise_floor_rms = float(np.mean(bottom))

        # Intensity variation (std of frame RMS)
        if frame_rms is not None:
            report.intensity_std = float(np.std(frame_rms))

        # Basic pitch estimation via autocorrelation (lightweight)
        try:
            report.mean_pitch_hz = self._estimate_pitch(samples, sample_rate)
        except Exception:
            report.mean_pitch_hz = 0.0

        # Transcript verification — check "I love you" is actually spoken
        self._verify_transcript(audio, sample_rate, report)

        # Classify volume (use phrase-only RMS if we found the phrase)
        if report.has_phrase and report.phrase_start_sec >= 0:
            phrase_start = int(report.phrase_start_sec * sample_rate)
            phrase_end = int(report.phrase_end_sec * sample_rate)
            phrase_samples = samples[phrase_start:phrase_end]
            if len(phrase_samples) > 0:
                phrase_rms = float(np.sqrt(np.mean(phrase_samples ** 2)))
                report.rms = phrase_rms
                logger.info("Using phrase-only RMS: %.6f", phrase_rms)

        report.is_intimate = report.rms < self.intimate_rms
        report.is_strong = report.rms >= self.strong_rms

        # Quality score
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

        # Reward moderate RMS
        if 0.03 < report.rms < 0.9:
            score += 0.1

        report.score = max(0.0, min(1.0, score))
        report.reject_reasons = reasons
        report.is_usable = report.score >= self.unusable_threshold

        logger.info(
            "Quality: rms=%.4f peak=%.4f clip=%.4f dur=%.2fs score=%.2f "
            "intimate=%s strong=%s phrase=%s transcript='%s' reasons=%s",
            report.rms, report.peak, report.clipping_ratio,
            report.duration_sec, report.score,
            report.is_intimate, report.is_strong,
            report.has_phrase, report.transcript, reasons,
        )
        return report

    def _verify_transcript(self, audio: np.ndarray, sample_rate: int,
                           report: QualityReport):
        """Run Vosk on the clip to verify it contains 'I love you' and find word timings."""
        if self._recognizer_model is None:
            # No model — skip verification, assume phrase is present
            report.has_phrase = True
            return

        try:
            from vosk import KaldiRecognizer

            rec = KaldiRecognizer(self._recognizer_model, sample_rate)
            rec.SetWords(True)

            # Feed audio in chunks
            if audio.dtype != np.int16:
                audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            chunk_size = 4000
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                rec.AcceptWaveform(chunk.tobytes())

            # Get final result
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").lower().strip()
            report.transcript = text
            logger.info("Transcript: '%s'", text)

            # Check for "too" variant first (longer match)
            for variant in self.TOO_VARIANTS:
                if variant in text:
                    report.has_phrase = True
                    report.has_too = True
                    break

            # Fall back to base phrase
            if not report.has_phrase:
                for variant in self.PHRASE_VARIANTS:
                    if variant in text:
                        report.has_phrase = True
                        break

            if not report.has_phrase:
                logger.info("Phrase 'I love you' NOT found in transcript")
                return

            # Find word timings to locate the phrase
            words = result.get("result", [])
            if words:
                self._find_phrase_timing(words, report)
            else:
                logger.debug("No word timings available from Vosk")

        except Exception:
            logger.exception("Transcript verification failed")
            report.has_phrase = True  # fail open — don't reject on Vosk error

    def _find_phrase_timing(self, words: list[dict], report: QualityReport):
        """Find the start/end time of 'I love you' in the word list."""
        # words is a list of {"word": "...", "start": float, "end": float, "conf": float}
        word_texts = [w.get("word", "").lower() for w in words]

        # Scan for the sequence "i", "love", "you"/"u"/"yo" [+ "too"/"to"]
        for i in range(len(word_texts) - 2):
            if (word_texts[i] == "i" and
                word_texts[i + 1] == "love" and
                word_texts[i + 2] in ("you", "u", "yo")):
                report.phrase_start_sec = max(0, words[i]["start"] - self.trim_pad_before)
                # Check for trailing "too"/"to"
                end_idx = i + 2
                if (i + 3 < len(word_texts) and
                        word_texts[i + 3] in ("too", "to", "two")):
                    end_idx = i + 3
                    report.has_too = True
                report.phrase_end_sec = words[end_idx]["end"] + self.trim_pad_after
                logger.info("Phrase located: %.2fs — %.2fs (too=%s)",
                            report.phrase_start_sec, report.phrase_end_sec,
                            report.has_too)
                return

        # Fallback: look for "love" and add padding
        for i, w in enumerate(word_texts):
            if w == "love":
                start_idx = max(0, i - 1)
                end_idx = min(len(words) - 1, i + 1)
                report.phrase_start_sec = max(0, words[start_idx]["start"] - self.trim_pad_before)
                report.phrase_end_sec = words[end_idx]["end"] + self.trim_pad_after
                logger.info("Phrase (approx via 'love'): %.2fs — %.2fs",
                            report.phrase_start_sec, report.phrase_end_sec)
                return

    def trim_to_phrase(self, audio: np.ndarray, sample_rate: int,
                       report: QualityReport) -> np.ndarray:
        """Trim audio to just the phrase region identified in the report."""
        if report.phrase_start_sec < 0 or report.phrase_end_sec < 0:
            logger.debug("No phrase timing — returning original audio")
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

    @staticmethod
    def _estimate_pitch(samples: np.ndarray, sr: int) -> float:
        """Simple autocorrelation pitch estimate on a short segment."""
        win = int(0.05 * sr)
        if len(samples) < win * 2:
            return 0.0

        energy = np.convolve(samples ** 2, np.ones(win), mode="valid")
        start = int(np.argmax(energy))
        segment = samples[start: start + win]

        corr = np.correlate(segment, segment, mode="full")
        corr = corr[len(corr) // 2:]

        min_lag = int(sr / 500)
        max_lag = int(sr / 70)
        if max_lag >= len(corr):
            return 0.0

        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return 0.0

        peak = int(np.argmax(search)) + min_lag
        if peak == 0:
            return 0.0

        return float(sr / peak)

    def should_save(self, report: QualityReport) -> bool:
        return report.has_phrase and report.score >= self.save_threshold

    def should_use_fallback(self, report: QualityReport) -> bool:
        return not report.is_usable
