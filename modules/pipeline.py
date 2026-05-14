"""
TriggerPipeline — the state machine that runs on each wake-phrase detection.

Was a 100-line god method in main.py. Now a clean sequence of stages:

    1. Snapshot     — copy pre-buffer immediately (no blocking calls before this)
    2. Wake         — fire WLED ack/flash
    3. EarlyRespond — if memory has clips, start playback now (in parallel with capture)
    4. Capture      — record post-detection (adaptive end-of-utterance)
    5. Transcribe   — run ASR backend, locate phrase timing
    6. Emotion      — run emotion backend, attach to QualityReport
    7. Score        — derive overall quality score, decide save/reject
    8. Trim         — cut clip to phrase boundaries if found
    9. Respond      — if no early response, respond now informed by analysis
   10. Save         — persist clip + emotion into memory
   11. FadeOut      — settle LEDs back to idle

Each stage is a method on TriggerPipeline. The orchestrator (Mirror) wires
this up with concrete collaborators; this module owns ordering and side
effects between them.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import soundfile as sf

from modules.clip_enhancer import ClipEnhancer
from modules.effects import layer_clips, normalize_audio
from modules.emotion import EmotionReport
from modules.memory_store import ClipEntry, MemoryStore
from modules.mood import MoodState
from modules.quality import QualityAnalyzer, QualityReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Mutable state carried through the stages of one trigger."""
    phrase_type: str = "ily"
    trigger_time: datetime = field(default_factory=datetime.now)
    mood: MoodState | None = None

    pre_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    clip: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    clip_path: str = ""

    report: QualityReport | None = None
    responded_early: bool = False
    response_paths: list[str] = field(default_factory=list)


class TriggerPipeline:
    def __init__(self, cfg, audio_input, recorder, quality: QualityAnalyzer,
                 memory: MemoryStore, playback, wled, wled_rt=None,
                 mood_keeper=None, matcher=None):
        self.cfg = cfg
        self.audio_input = audio_input
        self.recorder = recorder
        self.quality = quality
        self.memory = memory
        self.playback = playback
        self.wled = wled
        self.wled_rt = wled_rt          # optional WLEDRealtime (UDP DRGB)
        self.mood_keeper = mood_keeper
        self.matcher = matcher          # for reading per-bucket metadata
        # Clip enhancer — applied on save so playback is consistent.
        self.enhancer = ClipEnhancer(cfg)

        self.mode = cfg.get("mode", "hybrid")
        self.response_mode = cfg.get("response_mode", "layered")
        self.too_only = cfg.get("too_only", False)
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.max_layers = cfg["playback"]["max_layers"]

        # Emotion-aware selection config
        ec = cfg.get("emotion", {})
        # "match" | "contrast" | "diverse"
        self.selection_mode = ec.get("selection_mode", "match")
        # How much of the candidate pool to keep after emotion-filtering. The
        # final layer set is RANDOM-SAMPLED from this top-K so each trigger
        # feels different even at the same emotion target. Higher = more
        # variety; lower = tighter emotional match.
        self.emotion_pool_size = ec.get("pool_size", 16)
        # Layer count cap modulated by mood.intimacy
        self.intimacy_layer_cap_min = 1

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, phrase_type: str = "ily"):
        ctx = PipelineContext(phrase_type=phrase_type)
        if self.mood_keeper is not None:
            ctx.mood = self.mood_keeper.current()
            self.mood_keeper.record_trigger()

        try:
            self._stage_snapshot(ctx)
            self._stage_wake(ctx)
            self._stage_early_respond(ctx)
            self._stage_capture(ctx)
            self._stage_analyze(ctx)
            self._stage_reconcile_phrase_type(ctx)
            self._stage_trim_and_resave(ctx)
            self._stage_log(ctx)
            self._stage_respond_late(ctx)
            self._stage_save(ctx)
        finally:
            self._stage_fadeout(ctx)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_snapshot(self, ctx: PipelineContext):
        """Snapshot pre-buffer immediately. Critical: no blocking call before this."""
        ctx.pre_audio = self.audio_input.get_pre_buffer()

    def _stage_wake(self, ctx: PipelineContext):
        """Fire WLED acknowledge. Best-effort, non-blocking."""
        try:
            if self.wled_rt is not None:
                self.wled_rt.flash((255, 255, 255), hold_ms=80, fade_ms=180)
            else:
                self.wled.flash()
            self.wled.wake_neutral()
        except Exception:
            logger.exception("WLED wake stage failed (continuing)")

    def _stage_early_respond(self, ctx: PipelineContext):
        """If memory has clips, start playback now in parallel with capture.

        Selection is RANDOM (random.sample) from the candidate pool — same
        emotion target won't produce the same chorus twice.
        """
        # Check if this bucket has a special policy (e.g. Sandy: response only,
        # no layering, only specific files)
        bucket_meta = self._bucket_metadata(ctx.phrase_type)
        response_files_only = bucket_meta.get("response_files_only", False)

        has_existing = self.memory.has_live_clips() or (
            self.mode in ("fallback", "hybrid") and self.memory.get_fallback_clips(ctx.phrase_type)
        )
        if response_files_only:
            # Only use this bucket's fallback files. No layering, single random pick.
            fb = self.memory.get_fallback_clips(ctx.phrase_type)
            if not fb:
                logger.warning("Bucket %s is response_files_only but has no fallbacks", ctx.phrase_type)
                return
            pick = random.choice(fb)
            ctx.responded_early = True
            ctx.response_paths = [pick]
            logger.info("Response (early, %s files-only): SINGLE → %s", ctx.phrase_type, pick)
            self.wled.intimate()
            self._react_and_play_file(pick, valence=0.0, arousal=0.5)
            return

        if not has_existing:
            return

        paths = self._gather_response_paths(ctx.phrase_type)
        if not paths:
            return

        # Modulate layer count by mood intimacy (intimate → fewer layers)
        layers_cap = max(
            self.intimacy_layer_cap_min,
            int(round(self.max_layers * (1.2 - (ctx.mood.intimacy if ctx.mood else 0.5)))),
        )
        layers_cap = min(layers_cap, len(paths))

        ctx.responded_early = True

        if self.response_mode == "single" or layers_cap == 1:
            pick = random.choice(paths)
            logger.info("Response (early): SINGLE → %s", pick)
            self.wled.intimate()
            self._react_and_play_file(pick, valence=0.0, arousal=0.5)
            ctx.response_paths = [pick]
        else:
            # *** Randomized layered selection ***
            picks = random.sample(paths, k=layers_cap)
            clips = []
            kept_paths = []
            for p in picks:
                try:
                    data, _ = sf.read(p, dtype="int16")
                    clips.append(data)
                    kept_paths.append(p)
                except Exception:
                    logger.warning("Could not read clip: %s", p)
            if not clips:
                ctx.responded_early = False
                return
            if len(clips) == 1:
                logger.info("Response (early): SINGLE (random) → %s", kept_paths[0])
                self.wled.intimate()
                self._react_and_play_file(kept_paths[0], 0.0, 0.5)
            else:
                logger.info("Response (early): LAYERED %d random clips: %s",
                             len(clips), [p.split("/")[-1] for p in kept_paths])
                self.wled.layered()
                mixed = layer_clips(clips, self.sample_rate,
                                     self._mood_modulated_playback_cfg(ctx))
                self._react_and_play_array(mixed, 0.0, 0.5)
            ctx.response_paths = kept_paths

        self.memory.mark_played(ctx.response_paths)

    def _stage_capture(self, ctx: PipelineContext):
        clip, clip_path = self.recorder.capture_clip_with_pre(
            ctx.pre_audio, self.audio_input, phrase_type=ctx.phrase_type)
        ctx.clip = clip
        ctx.clip_path = clip_path
        logger.info("Clip captured: %d samples (%.2fs) → %s",
                     len(clip), len(clip) / self.sample_rate if len(clip) else 0,
                     clip_path)

    def _stage_analyze(self, ctx: PipelineContext):
        if len(ctx.clip) > 0:
            ctx.report = self.quality.analyze(ctx.clip, self.sample_rate)
        else:
            r = QualityReport()
            r.is_usable = False
            r.reject_reasons = ["no_audio"]
            ctx.report = r
            logger.warning("No audio captured — using dummy report")

    def _stage_reconcile_phrase_type(self, ctx: PipelineContext):
        """Reconcile detector-hint phrase_type with what the transcript actually
        showed. The transcript is more authoritative: it sees the whole clip and
        knows whether the user said 'too', or named Sandy."""
        if not ctx.report:
            return
        if ctx.report.matched_bucket and ctx.report.matched_bucket != ctx.phrase_type:
            logger.info("Reconcile phrase_type: detector=%s → transcript=%s",
                         ctx.phrase_type, ctx.report.matched_bucket)
            ctx.phrase_type = ctx.report.matched_bucket
        elif ctx.report.has_too and ctx.phrase_type != "ily_too":
            ctx.phrase_type = "ily_too"

    def _stage_trim_and_resave(self, ctx: PipelineContext):
        if len(ctx.clip) == 0 or ctx.report is None:
            return
        if not ctx.report.has_phrase:
            logger.warning("Phrase not found in recording — will not save")
            return
        # 1. Trim to phrase span
        trimmed = self.quality.trim_to_phrase(ctx.clip, self.sample_rate, ctx.report)
        if len(trimmed) == 0:
            return
        ctx.clip = trimmed

        # 2. Enhance (denoise + loudness normalize + compress + de-ess)
        #    so this clip blends with the chorus instead of standing out.
        try:
            enhanced, metrics = self.enhancer.enhance(trimmed, self.sample_rate)
            # Use the enhanced version for both playback and on-disk save
            ctx.clip = enhanced
            # Annotate the report with measured-after metrics (overwrite the
            # pre-enhancement SNR / spectral tilt readings).
            if ctx.report is not None:
                ctx.report.snr_db = metrics.snr_db_after
                ctx.report.spectral_centroid_hz = metrics.spectral_centroid_hz
                ctx.report.spectral_tilt_db_per_khz = metrics.spectral_tilt_db_per_khz
        except Exception:
            logger.exception("Clip enhancement failed — falling back to raw trimmed audio")

        # 3. Save the (trimmed + enhanced) clip back to its path
        if ctx.clip_path:
            self.recorder.save_clip(ctx.clip, ctx.clip_path)
            logger.info("Trimmed+Enhanced [%s] → %s (%.2fs)",
                         ctx.phrase_type, ctx.clip_path,
                         len(ctx.clip) / self.sample_rate)

    def _stage_log(self, ctx: PipelineContext):
        if ctx.report is None:
            return
        r = ctx.report
        e: EmotionReport = r.emotion
        logger.info(
            "TRIGGER | type=%s | time=%s | clip=%s | rms=%.4f peak=%.4f clip_ratio=%.4f "
            "dur=%.2fs | score=%.2f intimate=%s strong=%s usable=%s too=%s | "
            "emo=%s V=%+.2f A=%.2f D=%.2f | reasons=%s",
            ctx.phrase_type, ctx.trigger_time.isoformat(), ctx.clip_path,
            r.rms, r.peak, r.clipping_ratio, r.duration_sec, r.score,
            r.is_intimate, r.is_strong, r.is_usable, r.has_too,
            e.label, e.valence, e.arousal, e.dominance,
            r.reject_reasons,
        )

    def _stage_respond_late(self, ctx: PipelineContext):
        """Respond now if we didn't earlier (no prior clips in memory)."""
        if ctx.responded_early or ctx.report is None:
            return
        r = ctx.report
        if not r.has_phrase:
            logger.info("No phrase detected — rejecting")
            self.wled.reject()
            return
        if self.quality.should_use_fallback(r) and self.mode != "live":
            self._respond_fallback(r)
            return
        if self.response_mode == "single" or r.is_intimate:
            self._respond_intimate(ctx)
        else:
            self._respond_layered(ctx)

    def _stage_save(self, ctx: PipelineContext):
        if not ctx.clip_path or ctx.report is None:
            return
        # Respect per-bucket save policy (e.g. Sandy bucket: never save user recordings)
        if not self.memory.bucket_allows_save(ctx.phrase_type):
            logger.info("Clip NOT SAVED [%s] — bucket policy save_user_recordings=false",
                         ctx.phrase_type)
            return
        if not self.quality.should_save(ctx.report) or self.mode == "fallback":
            logger.info("Clip REJECTED [%s] (score=%.2f, reasons=%s)",
                         ctx.phrase_type, ctx.report.score, ctx.report.reject_reasons)
            return
        r = ctx.report
        e = r.emotion
        self.memory.add_clip(
            path=ctx.clip_path,
            timestamp=ctx.trigger_time.isoformat(),
            rms=r.rms,
            score=r.score,
            duration_sec=r.duration_sec,
            is_intimate=r.is_intimate,
            phrase_type=ctx.phrase_type,
            valence=e.valence,
            arousal=e.arousal,
            dominance=e.dominance,
            emotion_label=e.label,
        )
        logger.info("Clip SAVED [%s] → %s (score=%.2f, emo=%s)",
                     ctx.phrase_type, ctx.clip_path, r.score, e.label)

    def _stage_fadeout(self, ctx: PipelineContext):
        try:
            time.sleep(1.0)
            self.wled.fade_out()
            time.sleep(2.0)
            self.wled.idle()
            if self.wled_rt is not None:
                self.wled_rt.stop_reacting()
        except Exception:
            logger.exception("Fadeout stage failed")

    # ------------------------------------------------------------------
    # Late-response strategies (informed by analysis + emotion)
    # ------------------------------------------------------------------

    def _respond_intimate(self, ctx: PipelineContext):
        r = ctx.report
        e = r.emotion
        logger.info("Response: INTIMATE [%s] V=%+.2f A=%.2f", e.label, e.valence, e.arousal)
        self.wled.intimate()
        path = self._pick_intimate_path(ctx)
        if path:
            self._react_and_play_file(path, e.valence, e.arousal)
            self.memory.mark_played([path])
        elif len(ctx.clip) > 0:
            from modules.effects import apply_volume, fade_in, fade_out
            soft = normalize_audio(ctx.clip)
            soft = apply_volume(soft, 0.5)
            soft = fade_in(soft, int(0.2 * self.sample_rate))
            soft = fade_out(soft, int(0.3 * self.sample_rate))
            self._react_and_play_array(soft, e.valence, e.arousal)

    def _respond_layered(self, ctx: PipelineContext):
        r = ctx.report
        e = r.emotion
        logger.info("Response: LAYERED [%s] V=%+.2f A=%.2f", e.label, e.valence, e.arousal)
        self.wled.layered()

        # Emotion-aware: pick clips that match the speaker's emotion
        n = self.max_layers
        if ctx.mood is not None:
            n = max(2, int(round(self.max_layers * (1.0 - 0.4 * ctx.mood.intimacy + 0.3 * e.arousal))))
            n = min(n, self.max_layers)

        # 1. Build an emotion-FILTERED pool of size emotion_pool_size...
        #    (the top-K nearest in V/A space, NOT the final picks)
        # 2. ...then RANDOM-SAMPLE n-1 clips from it. Same speaker emotion
        #    won't pull the same chorus twice.
        # When the bucket is ily_too, only use ily_too clips. For default ily,
        # any clip in the matched bucket counts.
        pool_size = max(n, self.emotion_pool_size)
        pool = self.memory.pick_by_emotion(
            e.valence, e.arousal,
            n=pool_size,
            mode=self.selection_mode,
            phrase_type=ctx.phrase_type if ctx.phrase_type in ("ily_too",) else None,
        )
        # Filter out clips played very recently (last 30s) to avoid repeats
        import time as _t
        now = _t.time()
        fresh = [c for c in pool if (now - c.last_played_ts) > 30.0]
        # If filtering wiped the pool, fall back to the full pool
        pool = fresh if fresh else pool

        # Randomized sample from the emotion-matched pool
        k = min(n - 1, len(pool))
        chosen = random.sample(pool, k=k) if k > 0 else []

        clips_to_layer = []
        chosen_paths = []
        for entry in chosen:
            try:
                data, _ = sf.read(entry.path, dtype="int16")
                clips_to_layer.append(data)
                chosen_paths.append(entry.path)
            except Exception:
                logger.warning("Could not read clip: %s", entry.path)

        # Top up from fallbacks if needed
        if self.mode in ("fallback", "hybrid"):
            needed = n - len(clips_to_layer) - (1 if len(ctx.clip) > 0 else 0)
            fbs = self.memory.get_fallback_clips(ctx.phrase_type) or []
            random.shuffle(fbs)
            for fb_path in fbs[:max(0, needed)]:
                try:
                    data, _ = sf.read(fb_path, dtype="int16")
                    clips_to_layer.append(data)
                    chosen_paths.append(fb_path)
                except Exception:
                    pass

        if len(ctx.clip) > 0:
            clips_to_layer.append(ctx.clip)

        if not clips_to_layer:
            logger.warning("No clips available for layered response")
            self.wled.reject()
            return

        logger.info("Layered picks (random sample of %d from emotion pool of %d): %s",
                     len(chosen_paths), len(pool),
                     [p.split("/")[-1] for p in chosen_paths])

        mixed = layer_clips(clips_to_layer, self.sample_rate,
                             self._mood_modulated_playback_cfg(ctx))
        self._react_and_play_array(mixed, e.valence, e.arousal)
        if chosen_paths:
            self.memory.mark_played(chosen_paths)

    def _respond_fallback(self, r: QualityReport):
        logger.info("Response: FALLBACK (score=%.2f)", r.score)
        self.wled.reject()
        fb = self.memory.get_fallback_clips()
        if fb:
            self.playback.play_file(fb[0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_response_paths(self, phrase_type: str = "ily") -> list[str]:
        """Gather candidate paths from live + fallback for the matched bucket.

        For specific buckets (ily_too, ily_sandy, ...) we ONLY draw from that
        bucket's own pool — Sandy's response shouldn't pull random clips from
        the general ily pool, and "i love you too" shouldn't either.
        """
        paths: list[str] = []
        if self.response_mode == "single" and self.too_only:
            if self.mode in ("live", "hybrid"):
                paths.extend(e.path for e in self.memory.get_clips_by_type("ily_too"))
            if self.mode in ("fallback", "hybrid"):
                paths.extend(self.memory.get_fallback_clips("ily_too"))
            return paths

        if phrase_type not in ("ily",):
            # Bucket-specific (ily_too, ily_sandy, etc.) — pure pool
            if self.mode in ("live", "hybrid"):
                paths.extend(e.path for e in self.memory.get_clips_by_type(phrase_type))
            if self.mode in ("fallback", "hybrid"):
                paths.extend(self.memory.get_fallback_clips(phrase_type))
            return paths

        # Default ily bucket — broader pool
        if self.response_mode == "single":
            if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
                paths.extend(e.path for e in self.memory.get_recent_clips())
            if self.mode in ("fallback", "hybrid"):
                paths.extend(self.memory.get_fallback_clips("ily"))
        else:
            if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
                paths.extend(e.path for e in self.memory.get_recent_clips())
            if self.mode in ("fallback", "hybrid"):
                paths.extend(self.memory.get_fallback_clips("ily"))
        return paths

    def _bucket_metadata(self, phrase_type: str) -> dict:
        """Return per-bucket metadata dict (response_files_only, etc.).
        Empty if matcher not provided or bucket not known."""
        if self.matcher is None:
            return {}
        b = self.matcher.get_bucket(phrase_type)
        return (getattr(b, "metadata", {}) or {}) if b else {}

    def _pick_intimate_path(self, ctx: PipelineContext) -> str | None:
        """For intimate response, build an emotion-matched pool and pick
        randomly within it so the same speaker gets variety across triggers."""
        e = ctx.report.emotion if ctx.report else None
        bucket = ctx.phrase_type
        bucket_filter = bucket if bucket not in ("ily",) else None
        if e is not None and self.memory.has_live_clips():
            pool = self.memory.pick_by_emotion(
                e.valence, e.arousal,
                n=min(self.emotion_pool_size, max(4, self.max_layers)),
                mode="match",
                phrase_type=bucket_filter,
            )
            if pool:
                return random.choice(pool).path
        if self.memory.has_live_clips():
            # Random recent clip rather than always the newest
            recent = self.memory.get_recent_clips(min(self.emotion_pool_size, 16))
            if recent:
                return random.choice(recent).path
        fb = self.memory.get_fallback_clips(bucket)
        if fb:
            return random.choice(fb)
        return None

    def _react_and_play_file(self, path: str, valence: float, arousal: float):
        try:
            data, file_sr = sf.read(path, dtype="float32")
            self._start_lighting_react(data, file_sr, valence, arousal)
            self.playback.play_file(path)
        finally:
            self._stop_lighting_react()

    def _react_and_play_array(self, audio: np.ndarray, valence: float, arousal: float):
        try:
            self._start_lighting_react(audio, self.sample_rate, valence, arousal)
            self.playback.play_array(audio, self.sample_rate)
        finally:
            self._stop_lighting_react()

    def _start_lighting_react(self, audio: np.ndarray, sr: int,
                              valence: float, arousal: float):
        if self.wled_rt is not None:
            # 30Hz matches WLED-Gledopto's hardware paint ceiling on 210 LEDs
            # (probed 2026-05-14: strip runs at ~44 fps max even when fed faster).
            # Lower-end controllers cap even lower — set in config if needed.
            update_hz = self.cfg.get("wled", {}).get("realtime", {}).get("update_hz", 30.0)
            self.wled_rt.react_to_audio(audio, sr,
                                         valence=valence, arousal=arousal,
                                         update_hz=update_hz)
        else:
            self.wled.react_to_audio(audio, sr)

    def _stop_lighting_react(self):
        if self.wled_rt is not None:
            self.wled_rt.stop_reacting()
        else:
            self.wled.stop_reacting()

    def _mood_modulated_playback_cfg(self, ctx: PipelineContext) -> dict:
        """Return a playback cfg dict with mood-driven tweaks layered on top."""
        cfg = dict(self.cfg["playback"])
        if ctx.mood is None:
            return cfg
        m = ctx.mood
        # Slower pace at night (deeper, more deliberate layering)
        cfg["layer_delay_sec"] = cfg.get("layer_delay_sec", 0.35) * m.pace
        # More intimacy → less reverb mix (closer voices)
        if "reverb_decay" in cfg:
            cfg["reverb_decay"] = max(0.05, cfg["reverb_decay"] * (1.2 - 0.5 * m.intimacy))
        return cfg
