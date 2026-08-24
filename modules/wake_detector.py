"""
Wake-word detection — swappable backends.

Backends:
  - SimulatedDetector: stdin-based trigger for testing
  - VoskDetector: offline speech recognition, matches "i love you" in transcript
  - PorcupineDetector: Picovoice Porcupine for offline wake-word
"""

import abc
import json
import logging
import os
import sys
import threading
import time

from modules.phrase_match import PhraseMatcher

logger = logging.getLogger(__name__)


class BaseDetector(abc.ABC):
    """Interface every detector must implement."""

    @abc.abstractmethod
    def start(self, on_detected: callable, audio_input=None):
        """Begin listening. Call on_detected(phrase) when wake phrase is heard.
        phrase is 'ily' for 'I love you' or 'ily_too' for 'I love you too'."""

    @abc.abstractmethod
    def stop(self):
        """Stop listening and release resources."""


# --------------------------------------------------------------------------
# Simulated detector — press Enter to trigger
# --------------------------------------------------------------------------

class SimulatedDetector(BaseDetector):
    def __init__(self, cfg: dict):
        self._running = False
        self._thread = None
        self._callback = None

    def start(self, on_detected: callable, audio_input=None):
        self._callback = on_detected
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Simulated detector started — press Enter to trigger")

    def _loop(self):
        while self._running:
            try:
                line = sys.stdin.readline()
                if not self._running:
                    break
                if line is not None:
                    text = line.strip().lower()
                    phrase = "ily_too" if "too" in text else "ily"
                    logger.info("Simulated detection triggered (%s)", phrase)
                    if self._callback:
                        self._callback(phrase)
            except EOFError:
                break

    def stop(self):
        self._running = False
        logger.info("Simulated detector stopped")


# --------------------------------------------------------------------------
# Vosk detector — reads from the shared AudioInput rolling buffer
# --------------------------------------------------------------------------

class VoskDetector(BaseDetector):
    """
    Listens continuously using Vosk offline speech recognition.
    Reads audio from the shared AudioInput buffer (same mic stream
    used for recording) so detection and capture are always in sync.
    """

    def __init__(self, cfg: dict, matcher: PhraseMatcher | None = None,
                 playback=None):
        vosk_cfg = cfg.get("vosk", {})
        self.model_path = vosk_cfg.get("model_path", "models/vosk-model-small-en-us-0.15")
        self.sample_rate = cfg["audio"]["sample_rate"]
        self._cooldown_sec = vosk_cfg.get("cooldown_sec", 3.0)
        # Confidence floor for accepting a partial-result trigger
        self._partial_conf_min = vosk_cfg.get("partial_confidence_min", 0.75)
        # Confidence floor for accepting a final-result trigger.
        # Final results bypass partial gating; this catches phonetic/fuzzy
        # matches that fire on noise transcripts. Set 0.0 to disable.
        self._final_conf_min = vosk_cfg.get("final_confidence_min", 0.0)
        # Mute the detector while the speaker is playing AND for a short tail
        # afterward — stops the chorus from re-triggering itself through the
        # mic. Tail covers room reverb + ALSA buffer drain. Tune via
        # vosk.mute_cooldown_sec in config.
        self._mute_cooldown_sec = vosk_cfg.get("mute_cooldown_sec", 0.6)
        # Defer partial-result triggers by this many seconds so a more
        # specific (higher-priority) bucket can supersede before we fire.
        # Without this, "i love you" (ily) triggers the layered response
        # before "sandy" arrives in the transcript and ily_sandy never gets
        # a chance. Final results fire immediately and supersede pending.
        self._defer_partial_sec = vosk_cfg.get("defer_partial_sec", 0.6)
        # Cross-validation against a parallel FULL-VOCAB recognizer.
        # Grammar mode forces Kaldi to pick from our wake-phrase list, so
        # random speech ("have you", "rectangle") gets transcribed as the
        # closest wake phrase at conf=1.0. The full-vocab recognizer reads
        # the same audio without that bias — if it doesn't also hear a
        # wake-anchor word, we suppress the grammar trigger as overfit.
        cv_cfg = vosk_cfg.get("cross_validate", {}) or {}
        self._cv_enabled = cv_cfg.get("enabled", True)
        # Strict pair validation: require the honest transcript to contain a
        # love-word ADJACENT to an addressee token ("love you"/"love ya"/
        # "love sandy"), not merely any love-prefixed word anywhere. Kills
        # the "I'd love to…" / "lovely" conversational false-validations that
        # anchor-only checking lets through. Set false to revert to the old
        # any-anchor behaviour.
        self._cv_require_pair = cv_cfg.get("require_pair", True)
        # Drop the last full-vocab final if it's older than this — staleness
        # protects against firing on a fresh grammar hit while validating
        # against text from minutes ago.
        self._cv_staleness_sec = cv_cfg.get("staleness_sec", 4.0)
        self._playback = playback
        self._running = False
        self._thread = None
        self._callback = None
        self._audio_input = None
        self._last_trigger = 0.0
        # Pending partial trigger awaiting a possible higher-priority
        # replacement OR its fire deadline. Tuple:
        # (hit, fire_at_monotonic, text, result_type, priority)
        self._pending: tuple | None = None
        # Use the orchestrator-provided matcher (config-driven buckets) when
        # available; otherwise fall back to a default-bucket matcher.
        self._matcher = matcher or PhraseMatcher()
        # Cross-validation state — most recent full-vocab final result and
        # when it landed (monotonic). Partial text is read on demand from
        # _recognizer_full.PartialResult() at validation time.
        self._recognizer_full = None
        self._full_last_final: str = ""
        self._full_last_final_at: float = 0.0

    def set_playback(self, playback):
        """Late-bind the Playback instance (avoids circular construction)."""
        self._playback = playback

    def start(self, on_detected: callable, audio_input=None):
        try:
            from vosk import Model, KaldiRecognizer
        except ImportError:
            raise RuntimeError("Vosk backend requires vosk. Install: pip install vosk")

        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"Vosk model not found at {self.model_path}. "
                "Download from https://alphacephei.com/vosk/models"
            )

        if audio_input is None:
            raise RuntimeError("VoskDetector requires a shared AudioInput instance")

        self._audio_input = audio_input
        self._callback = on_detected
        self._model = Model(self.model_path)
        # Build a phrase grammar from the matcher's bucket patterns. Vosk's
        # small model has weak language modeling and consistently mishears
        # named/uncommon words (e.g. "sandy" → "for", "i'm", or dropped) in
        # streaming mode. Grammar mode biases the streaming recognizer
        # strongly toward our known wake phrases without affecting the
        # separate full-vocab ASR pass that runs after capture.
        self._grammar_json = self._build_grammar()
        self._recognizer = self._make_recognizer()
        if self._cv_enabled:
            self._recognizer_full = self._make_full_vocab_recognizer()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Vosk detector started (model=%s, sr=%d, grammar=%s, cross_validate=%s, shared audio)",
                     self.model_path, self.sample_rate,
                     "on" if self._grammar_json else "off",
                     "on" if self._cv_enabled else "off")

    def _build_grammar(self) -> str | None:
        """Construct a JSON grammar list from every bucket's patterns + slang.
        Returns None if grammar mode is disabled or there are no phrases —
        in that case the recognizer falls back to full-vocab mode."""
        if not self._matcher:
            return None
        phrases: list[str] = []
        seen: set[str] = set()
        for b in self._matcher.buckets:
            for p in b.patterns:
                if p and p not in seen:
                    phrases.append(p)
                    seen.add(p)
            for s in b.slang:
                if s and s not in seen:
                    phrases.append(s)
                    seen.add(s)
        if not phrases:
            return None
        # "[unk]" is required so the recognizer has a fallback for non-matching
        # speech and noise — without it the recognizer can stall.
        phrases.append("[unk]")
        return json.dumps(phrases)

    def _make_recognizer(self):
        """Build a fresh KaldiRecognizer with the cached grammar (if any).
        Used at start and after each playback-mute → un-mute cycle."""
        from vosk import KaldiRecognizer
        if self._grammar_json:
            r = KaldiRecognizer(self._model, self.sample_rate, self._grammar_json)
        else:
            r = KaldiRecognizer(self._model, self.sample_rate)
        r.SetWords(False)
        return r

    def _make_full_vocab_recognizer(self):
        """Build a fresh full-vocab KaldiRecognizer for cross-validation.
        No grammar bias — transcribes whatever is actually spoken so we can
        sanity-check the grammar recognizer's forced matches."""
        from vosk import KaldiRecognizer
        r = KaldiRecognizer(self._model, self.sample_rate)
        r.SetWords(False)
        return r

    def _loop(self):
        """Poll the shared AudioInput for new audio chunks and feed to Vosk.

        While playback is active (and during a brief cooldown after), we drain
        audio chunks WITHOUT feeding them to Vosk so the speaker output can't
        re-trigger the wake. On un-mute we rebuild the recognizer so any
        partial-state from before the playback doesn't immediately fire.
        """
        poll_sec = 0.15  # how often to grab audio
        chunk_duration = 0.2  # seconds of audio to grab each time

        # Wait for audio input to start producing data
        time.sleep(0.5)

        was_muted = False

        while self._running:
            try:
                if self._is_muted_by_playback():
                    # Drain so the rolling buffer doesn't pile up chorus audio
                    self._audio_input.get_current_chunk(chunk_duration)
                    if not was_muted:
                        logger.debug("Vosk muted — playback active")
                        # Drop any partial-debounce pending — it was matched
                        # against pre-playback audio and shouldn't fire after.
                        self._pending = None
                    was_muted = True
                    time.sleep(poll_sec)
                    continue

                if was_muted:
                    # Reset Vosk state so half-decoded pre-playback audio
                    # can't fire a stale partial right after un-muting.
                    self._recognizer = self._make_recognizer()
                    if self._cv_enabled:
                        self._recognizer_full = self._make_full_vocab_recognizer()
                        self._full_last_final = ""
                        self._full_last_final_at = 0.0
                    was_muted = False
                    logger.debug("Vosk un-muted — recognizer reset")

                chunk = self._audio_input.get_current_chunk(chunk_duration)
                if len(chunk) == 0:
                    time.sleep(poll_sec)
                    self._maybe_fire_pending()
                    continue

                audio_bytes = chunk.tobytes()

                # Feed full-vocab recognizer FIRST so its state reflects the
                # same audio as the grammar recognizer when we validate below.
                if self._cv_enabled:
                    self._feed_full_vocab(audio_bytes)

                if self._recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self._check_trigger(text, "final")
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if text:
                        self._check_trigger(text, "partial")

                # Fire any pending partial whose deadline has passed and
                # which wasn't superseded by a higher-priority match.
                self._maybe_fire_pending()

                time.sleep(poll_sec)

            except Exception:
                if self._running:
                    logger.exception("Vosk processing error")
                    time.sleep(0.5)

    def _is_muted_by_playback(self) -> bool:
        if self._playback is None:
            return False
        if self._playback.is_playing():
            return True
        return self._playback.time_since_playback_end() < self._mute_cooldown_sec

    def _feed_full_vocab(self, audio_bytes: bytes):
        """Drive the cross-validation recognizer without firing any triggers
        from it — its only job is to provide an unbiased recent transcript
        that _validate_full_vocab can compare against grammar hits."""
        try:
            if self._recognizer_full.AcceptWaveform(audio_bytes):
                r = json.loads(self._recognizer_full.Result())
                t = (r.get("text") or "").strip()
                if t:
                    self._full_last_final = t
                    self._full_last_final_at = time.monotonic()
        except Exception:
            logger.exception("Full-vocab recognizer error")

    def _full_vocab_snapshot(self) -> str:
        """Concatenate the last full-vocab final (if fresh) with its current
        partial. This is the text we validate against."""
        parts = []
        if self._full_last_final and \
                (time.monotonic() - self._full_last_final_at) <= self._cv_staleness_sec:
            parts.append(self._full_last_final)
        try:
            p = json.loads(self._recognizer_full.PartialResult())
            partial = (p.get("partial") or "").strip()
            if partial:
                parts.append(partial)
        except Exception:
            pass
        return " ".join(parts).strip().lower()

    def _validate_full_vocab(self, candidate_text: str, phrase_type: str) -> bool:
        """Return True if the full-vocab recognizer's recent transcript also
        contains the wake phrase's honest signature. With require_pair (the
        default) that means a love-word adjacent to an addressee token — or
        the bucket's own anchor word for anchor_words buckets — rather than
        any love-prefixed word anywhere (which conversational speech trips
        constantly). Grammar mode is still allowed to disambiguate WHICH
        bucket (e.g. it hears "sandy" where full-vocab may not); only the
        core phrase must show up in honest transcription for us to fire."""
        if not self._cv_enabled:
            return True
        full_text = self._full_vocab_snapshot()
        if not full_text:
            logger.info(
                "Vosk cross-validate rejected [%s]: full-vocab empty | grammar='%s'",
                phrase_type, candidate_text)
            return False
        if self._cv_require_pair:
            valid = self._matcher.validate_bucket_in_transcript(full_text, phrase_type)
        else:
            valid = self._matcher.quick_contains_any_anchor(full_text)
        if not valid:
            logger.info(
                "Vosk cross-validate rejected [%s]: no %s in full-vocab | grammar='%s' full='%s'",
                phrase_type, "phrase pair" if self._cv_require_pair else "anchor",
                candidate_text, full_text)
            return False
        logger.debug(
            "Vosk cross-validate passed [%s]: grammar='%s' full='%s'",
            phrase_type, candidate_text, full_text)
        return True

    def _check_trigger(self, text: str, result_type: str):
        text_lower = text.lower().strip()
        if not self._matcher.quick_contains_any_anchor(text_lower):
            return
        hit = self._matcher.find(text_lower)
        if hit is None:
            return
        # Be stricter on partial-result triggers to avoid premature fires
        if result_type == "partial" and hit.confidence < self._partial_conf_min:
            return
        # Same gate for final results — blocks phonetic/fuzzy hits on noise
        if result_type == "final" and hit.confidence < self._final_conf_min:
            logger.debug("Vosk final suppressed (conf=%.2f<%.2f via=%s): %s",
                          hit.confidence, self._final_conf_min, hit.matched_via, text_lower)
            return

        # Final result: fire immediately, supersede any pending partial.
        if result_type == "final":
            self._pending = None
            self._fire_trigger(text_lower, hit.matched_span, hit.phrase_type,
                                result_type, hit.confidence, hit.matched_via)
            return

        # Partial: defer briefly so a more specific bucket can supersede.
        # Higher priority replaces the pending hit but keeps the original
        # deadline (so the total wait is bounded by defer_partial_sec).
        bucket = self._matcher.get_bucket(hit.phrase_type)
        new_priority = bucket.priority if bucket else 0

        if self._pending is None:
            fire_at = time.monotonic() + self._defer_partial_sec
            self._pending = (hit, fire_at, text_lower, result_type, new_priority)
            logger.debug("Vosk partial pending [%s prio=%d, fire in %.2fs]: '%s'",
                          hit.phrase_type, new_priority, self._defer_partial_sec, text_lower)
        elif new_priority > self._pending[4]:
            logger.info("Vosk partial superseded: %s(prio=%d) → %s(prio=%d): '%s'",
                         self._pending[0].phrase_type, self._pending[4],
                         hit.phrase_type, new_priority, text_lower)
            self._pending = (hit, self._pending[1], text_lower, result_type, new_priority)

    def _maybe_fire_pending(self):
        """Fire a pending partial trigger whose defer deadline has passed."""
        if self._pending is None:
            return
        if time.monotonic() < self._pending[1]:
            return
        hit, _, text, result_type, _ = self._pending
        self._pending = None
        self._fire_trigger(text, hit.matched_span, hit.phrase_type,
                            result_type, hit.confidence, hit.matched_via)

    def _fire_trigger(self, text: str, phrase: str, phrase_type: str,
                       result_type: str, confidence: float, matched_via: str):
        now = time.time()
        if now - self._last_trigger < self._cooldown_sec:
            logger.debug("Vosk trigger suppressed (cooldown): '%s'", text)
            return
        # Cross-validate against the unbiased full-vocab recognizer before
        # committing — kills grammar-forced overfits like "have you" → "love
        # you" without rejecting honest sandy/monster hits.
        if not self._validate_full_vocab(text, phrase_type):
            return
        self._last_trigger = now
        logger.info("Vosk detected '%s' [%s] in %s result (via=%s, conf=%.2f): '%s'",
                    phrase, phrase_type, result_type, matched_via, confidence, text)
        if self._callback:
            self._callback(phrase_type)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("Vosk detector stopped")


# --------------------------------------------------------------------------
# Porcupine detector
# --------------------------------------------------------------------------

class PorcupineDetector(BaseDetector):
    """
    Offline wake-word detection using Picovoice Porcupine.
    Requires pvporcupine and pyaudio.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg.get("porcupine", {})
        self._running = False
        self._thread = None
        self._callback = None
        self._porcupine = None
        self._pa = None
        self._audio_stream = None

    def start(self, on_detected: callable, audio_input=None):
        try:
            import pvporcupine
            import pyaudio
        except ImportError:
            raise RuntimeError(
                "Porcupine backend requires pvporcupine and pyaudio. "
                "Install them: pip install pvporcupine pyaudio"
            )

        self._callback = on_detected
        access_key = self._cfg.get("access_key", "")
        keyword_path = self._cfg.get("keyword_path", "")
        model_path = self._cfg.get("model_path") or None
        sensitivity = self._cfg.get("sensitivity", 0.5)

        if keyword_path:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[keyword_path],
                model_path=model_path,
                sensitivities=[sensitivity],
            )
        else:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=["ok google"],  # placeholder
                sensitivities=[sensitivity],
            )
            logger.warning(
                "No custom keyword_path set — using placeholder keyword. "
                "Train a custom 'I love you' keyword at console.picovoice.ai"
            )

        self._pa = pyaudio.PyAudio()
        self._audio_stream = self._pa.open(
            rate=self._porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._porcupine.frame_length,
        )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Porcupine detector started")

    def _loop(self):
        import struct

        while self._running:
            try:
                pcm = self._audio_stream.read(
                    self._porcupine.frame_length, exception_on_overflow=False
                )
                pcm_unpacked = struct.unpack_from(
                    "h" * self._porcupine.frame_length, pcm
                )
                result = self._porcupine.process(pcm_unpacked)
                if result >= 0:
                    logger.info("Porcupine detected wake word (index=%d)", result)
                    if self._callback:
                        self._callback()
            except Exception:
                if self._running:
                    logger.exception("Porcupine processing error")

    def stop(self):
        self._running = False
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        if self._pa is not None:
            self._pa.terminate()
        if self._porcupine is not None:
            self._porcupine.delete()
        logger.info("Porcupine detector stopped")


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def create_detector(cfg: dict, matcher: PhraseMatcher | None = None,
                    playback=None) -> BaseDetector:
    backend = cfg.get("wake_detector", "simulate")
    if backend == "simulate":
        return SimulatedDetector(cfg)
    elif backend == "vosk":
        return VoskDetector(cfg, matcher=matcher, playback=playback)
    elif backend == "porcupine":
        return PorcupineDetector(cfg)
    else:
        raise ValueError(f"Unknown wake_detector backend: {backend}")
