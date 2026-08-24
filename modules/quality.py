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
    matched_span: str = ""                # the actual phrase pattern that matched
                                          # (e.g. "i am a monster") — used by
                                          # _locate_phrase_timing to find the
                                          # span in the word list, not just for
                                          # ily-family buckets.
    transcript: str = ""
    transcript_confidence: float = 0.0
    phrase_match_confidence: float = 0.0
    phrase_start_sec: float = -1.0
    phrase_end_sec: float = -1.0
    # True when there's voiced audio meaningfully past phrase_end_sec —
    # i.e. the visitor kept talking after "you". For the plain `ily`
    # bucket we use this to reject the clip from save (a "clean" I love
    # you is followed by silence; continuation suggests "I love you
    # Sandy" / "I love you so much" — different tone, not chorus-ready).
    has_continuation: bool = False
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

        # Trim padding around the located phrase. Both pads were 0.15/0.25
        # originally; on review, those routinely leaked partial words on
        # either end ("Sandy, I love you" → "ndy, I love you", and
        # "I love you Sandy" → "I love you Sa…"). 0.08 each side keeps
        # the natural decay of the vowels while staying short of any
        # adjacent syllable. Lift back up if a particular voice's
        # consonant decay needs more breathing room.
        # Both pads were progressively tightened as we kept catching leaks:
        # 0.15/0.25 → 0.08/0.10 → 0.05/0.05 → 0.0/0.05. Pure-zero pre-pad
        # shaved the "i" vowel onset (Vosk's word.start is at the
        # consonant boundary, not the actual sibilant attack). 0.02 keeps
        # the attack while staying short of any hallucinated leading word
        # — the i_set guard in Strategy 3 of _set_phrase_timing already
        # prevents "the"/"and" from being included as the prior word.
        self.trim_pad_before = 0.02
        self.trim_pad_after = 0.05

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

            # Continuation detection: measure voiced audio in a window
            # starting 50 ms past phrase_end_sec (skip natural decay) up to
            # 350 ms past. ≥ 30 ms triggers — even a partial "Sa" can be
            # heard in the chorus mix, so we err strict.
            window_start = report.phrase_end_sec + 0.05
            window_end = min(report.duration_sec, report.phrase_end_sec + 0.35)
            if window_end > window_start:
                ws = int(window_start * sample_rate)
                we = int(window_end * sample_rate)
                frame_n = max(1, int(0.025 * sample_rate))
                voiced_ms = 0
                for i in range(ws, we, frame_n):
                    chunk = samples[i:i + frame_n]
                    if len(chunk) == 0:
                        break
                    if float(np.sqrt(np.mean(chunk ** 2))) >= 0.012:
                        voiced_ms += 25
                if voiced_ms >= 30:
                    report.has_continuation = True

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
            report.matched_span = hit.matched_span
            report.has_too = hit.phrase_type == "ily_too"
            report.phrase_match_confidence = hit.confidence
            if tr.words:
                self._locate_phrase_timing(tr.words, report)
        else:
            logger.info("Phrase not found in transcript: '%s'", tr.text)

    def _locate_phrase_timing(self, words, report: QualityReport):
        """Walk word list to find the phrase span; cache start/end timings.

        Strategy:
          1. PREFERRED: locate the matched_span (e.g. 'i am a monster',
             'i love you sandy') token-by-token in the word list. Allows
             apostrophe / pronoun-variant normalization so 'i'm a monster'
             matches Vosk word boundaries like ['im', 'a', 'monster'].
          2. FALLBACK (ily-family only): hardcoded 'i love you' pattern
             matching for cases where the matcher matched via name_anchor
             or phonetic and matched_span doesn't actually appear in the
             transcript.

        Extends the span past the end to include either 'too/to/two' or any
        per-bucket extra anchor word (e.g. 'sandy') so trimming preserves
        trailing context.

        WITHOUT this generalization (pre-fix), only 'i'/'love'/'you' word
        sequences got phrase_start_sec set — every monster (and future
        non-ily) clip skipped trimming and saved the full 5s capture.
        """
        if not words:
            return
        texts = [w.text for w in words]
        bucket = self.matcher.get_bucket(report.matched_bucket) if report.matched_bucket else None
        extra_anchors = set((bucket.anchor_extra if bucket else []) or [])

        # --- Strategy 1: matched_span direct lookup -----------------------
        if report.matched_span:
            span_words = [t for t in report.matched_span.lower().split() if t]
            timing = self._find_span_timing(texts, span_words)
            if timing is not None:
                start_idx, end_idx = timing
                # Extend past end for 'too' or anchor_extra word
                if end_idx + 1 < len(texts):
                    nxt = texts[end_idx + 1]
                    if nxt in ("too", "to", "two"):
                        end_idx += 1
                        report.has_too = True
                    elif nxt in extra_anchors:
                        end_idx += 1
                report.phrase_start_sec = max(0, words[start_idx].start - self.trim_pad_before)
                report.phrase_end_sec = words[end_idx].end + self.trim_pad_after
                return

        # --- Strategy 2: hardcoded 'i love you' fallback ------------------
        # Catches name_anchor / phonetic matches where matched_span doesn't
        # appear verbatim in the ASR output.
        for i in range(len(texts) - 2):
            if (texts[i] == "i"
                    and texts[i + 1] == "love"
                    and texts[i + 2] in ("you", "u", "yo", "ya", "yew")):
                report.phrase_start_sec = max(0, words[i].start - self.trim_pad_before)
                end_idx = i + 2
                if i + 3 < len(texts):
                    nxt = texts[i + 3]
                    if nxt in ("too", "to", "two"):
                        end_idx = i + 3
                        report.has_too = True
                    elif nxt in extra_anchors:
                        end_idx = i + 3
                report.phrase_end_sec = words[end_idx].end + self.trim_pad_after
                return
        # Last-ditch: locate around 'love' alone. Only include the word
        # BEFORE 'love' if it's actually an 'i' variant — Vosk often
        # hallucinates "the"/"and"/"that" when the leading "I" vowel is
        # quiet, and including that filler word leaks into playback.
        i_set = {"i", "im", "ive", "ill"}
        for i, w in enumerate(texts):
            if w == "love":
                start_idx = i - 1 if (i > 0 and texts[i - 1] in i_set) else i
                end_idx = min(len(words) - 1, i + 1)
                report.phrase_start_sec = max(0, words[start_idx].start - self.trim_pad_before)
                report.phrase_end_sec = words[end_idx].end + self.trim_pad_after
                return

    @staticmethod
    def _find_span_timing(texts: list[str], span: list[str]) -> tuple[int, int] | None:
        """Sliding-window search for `span` inside `texts`. Allows minor
        tokenization differences:
          - apostrophes are stripped before compare ("i'm" == "im")
          - "you/u/yo/ya/yew" are interchangeable
          - "i / im / ive" are interchangeable
        Returns (start_idx, end_idx) inclusive, or None if no match.
        """
        n = len(span)
        if n == 0 or n > len(texts):
            return None

        you_set = {"you", "u", "yo", "ya", "yew"}
        i_set = {"i", "im", "ive", "ill"}

        def _norm(t: str) -> str:
            return t.replace("'", "").replace("’", "").lower()

        def _eq(transcript_tok: str, span_tok: str) -> bool:
            t = _norm(transcript_tok)
            s = _norm(span_tok)
            if t == s:
                return True
            if s in you_set and t in you_set:
                return True
            if s in i_set and t in i_set:
                return True
            return False

        for i in range(len(texts) - n + 1):
            if all(_eq(texts[i + j], span[j]) for j in range(n)):
                return (i, i + n - 1)
        return None

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
        # Plain "ily" bucket is the chorus default — only accept clean
        # "I love you" takes (silence after). Continued speech is a
        # different tone (calling someone's name, follow-on words) and
        # tends to feel out of place when layered with others. Other
        # buckets (ily_too, ily_sandy, monster) expect a trailing word
        # by design, so don't apply the gate there.
        if report.has_continuation and report.matched_bucket == "ily":
            if "continued_speech" not in report.reject_reasons:
                report.reject_reasons.append("continued_speech")
            return False
        return report.has_phrase and report.score >= self.save_threshold

    def should_use_fallback(self, report: QualityReport) -> bool:
        return not report.is_usable
