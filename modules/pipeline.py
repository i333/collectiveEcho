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
import os
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
    # Set True once the lights-to-idle "mic ready" signal has been spawned.
    # Prevents _stage_fadeout from re-firing the visual transition.
    signal_ready_fired: bool = False
    # Speaker identification — populated by _stage_speaker if the
    # identifier is wired and the clip survives the quality gate.
    speaker_id: str = ""
    speaker_embedding: list = field(default_factory=list)
    # Arrival order among chorus triggers; 0 = not registered (commands).
    seq: int = 0


class TriggerPipeline:
    def __init__(self, cfg, audio_input, recorder, quality: QualityAnalyzer,
                 memory: MemoryStore, playback, wled, wled_rt=None,
                 mood_keeper=None, matcher=None, speaker=None):
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
        self.speaker = speaker          # optional SpeakerIdentifier
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

        self._init_freshness(cfg)

        # Sleep state — "jetson sleep" pauses all trigger handling except
        # the wake command; "jetson wake up" resumes. In-memory ON PURPOSE:
        # a reboot always comes back awake (the required boot default).
        # While asleep the detector and idle lights keep running and the
        # LED strip breathes in the sleep color as a visible indicator.
        self.asleep = False
        rt_cfg = cfg.get("wled", {}).get("realtime", {}) or {}
        self.sleep_color = tuple(rt_cfg.get("sleep_color", [20, 40, 140]))
        self.sleep_breath_s = float(rt_cfg.get("sleep_breath_s", 6.0))

        # Rolling save/reject counters — emitted every `_log_every` triggers
        # so a sudden spike in rejections (mic moved, threshold drift) is
        # visible in the log without grepping individual TRIGGER lines.
        self._trigger_count = 0
        self._save_count = 0
        self._reject_reasons: dict[str, int] = {}
        self._log_every = 10

    def _init_freshness(self, cfg):
        pb = cfg.get("playback", {})
        # Anything (speaker, take, fallback file) played within this window
        # goes to the back of the line for the random picks. Soft: when the
        # pool is too small, the least-recently-played fill the gap.
        self.replay_cooldown_sec = pb.get("replay_cooldown_sec", 120.0)
        # A random-slot speaker may play any take within this score margin
        # of their best, rotating instead of always the single best take.
        self.take_quality_margin = pb.get("take_quality_margin", 0.1)
        # How long a quick re-trigger waits for the previous trigger's clip
        # to finish saving, so the visitor can hear their own take.
        self.wait_for_previous_save_sec = pb.get("wait_for_previous_save_sec", 1.5)
        self._save_cv = threading.Condition()
        self._run_seq = 0
        self._unsaved: set[int] = set()

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, phrase_type: str = "ily"):
        ctx = PipelineContext(phrase_type=phrase_type)

        # Command buckets short-circuit — they're system actions, not chorus
        # triggers. No capture, no response audio, no save.
        bucket_meta = self._bucket_metadata(phrase_type)
        command = bucket_meta.get("command")

        # Sleep gate — while asleep, ONLY "jetson wake up" gets through.
        # Everything else (love phrases, monster, other commands) is dropped
        # before any capture, LED flash, or mood bookkeeping happens.
        if self.asleep and command != "wake_listening":
            logger.info("Trigger ignored — asleep [%s]", phrase_type)
            return

        if command:
            self._run_command(ctx, command)
            return

        if self.mood_keeper is not None:
            ctx.mood = self.mood_keeper.current()
            self.mood_keeper.record_trigger()

        with self._save_cv:
            self._run_seq += 1
            ctx.seq = self._run_seq
            self._unsaved.add(ctx.seq)

        try:
            self._stage_snapshot(ctx)
            self._stage_wake(ctx)
            self._stage_early_respond(ctx)
            self._stage_capture(ctx)
            self._stage_analyze(ctx)
            self._stage_reconcile_phrase_type(ctx)
            # IMPORTANT: speaker embedding runs BEFORE trim so ECAPA sees
            # the full ~5s captured audio. Trimmed phrase-only clips are
            # too short (~1s) and produce unstable embeddings — see DEVLOG.
            self._stage_speaker(ctx)
            self._stage_trim_and_resave(ctx)
            self._stage_log(ctx)
            self._stage_respond_late(ctx)
            self._stage_save(ctx)
        finally:
            # Before fadeout (which can sleep ~3s): a waiting re-trigger
            # only needs the save, not the lights.
            self._release_save_slot(ctx)
            self._stage_fadeout(ctx)

    def _release_save_slot(self, ctx: PipelineContext):
        """This trigger's clip is saved (or never will be) — unblock any
        later trigger waiting in _wait_for_earlier_saves."""
        with self._save_cv:
            self._unsaved.discard(ctx.seq)
            self._save_cv.notify_all()

    def _wait_for_earlier_saves(self, ctx: PipelineContext):
        """A visitor who repeats the phrase right after the chorus re-triggers
        while their previous take is still in capture/analysis, so it isn't
        in memory yet and the promised "you hear yourself next time" misses.
        Hold this response briefly until earlier triggers have saved; only
        ever waits in that race, never on an idle mirror."""
        timeout = self.wait_for_previous_save_sec
        if timeout <= 0:
            return

        def earlier_pending():
            return any(s < ctx.seq for s in self._unsaved)

        with self._save_cv:
            if not earlier_pending():
                return
            t0 = time.monotonic()
            done = self._save_cv.wait_for(lambda: not earlier_pending(), timeout)
        waited = time.monotonic() - t0
        if done:
            logger.info("Held response %.2fs for the previous trigger's clip to save",
                         waited)
        else:
            logger.info("Previous trigger still processing after %.1fs — responding "
                         "without its clip", waited)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _run_command(self, ctx: PipelineContext, command: str):
        """Run a non-chorus system command. The detector wake gesture (LED
        flash) still fires so the user gets visible acknowledgement; then
        the command-specific handler runs."""
        logger.info("Command trigger: %s [%s]", command, ctx.phrase_type)
        try:
            self._stage_wake(ctx)
        except Exception:
            logger.exception("Command wake stage failed (continuing)")

        if command == "archive_clips":
            self._command_archive_clips(ctx)
        elif command == "restore_archives":
            self._command_restore_archives(ctx)
        elif command == "sleep_listening":
            self._command_sleep(ctx)
        elif command == "wake_listening":
            self._command_wake(ctx)
        else:
            logger.warning("Unknown command %r on bucket %s", command, ctx.phrase_type)

        # Always close out lights via the same signal-ready path so the
        # strip returns to idle.
        self._spawn_signal_ready(ctx)

    def _command_archive_clips(self, ctx: PipelineContext):
        """Archive every live clip into a timestamped subdir, clear state.
        Fallback dirs (Sandy memorial responses, etc.) are untouched."""
        try:
            counts = self.memory.archive_live_clips()
            total = sum(counts.values())
            logger.info("ARCHIVE COMMAND complete — %d clips archived: %s",
                         total, counts)
            self._command_confirmation_fx()
        except Exception:
            logger.exception("Archive command failed")

    def _command_restore_archives(self, ctx: PipelineContext):
        """Restore every archived clip back into the active pool with full
        metadata, then run the same confirmation FX so the user knows it
        landed."""
        try:
            counts = self.memory.restore_archives()
            total = sum(counts.values())
            logger.info("RESTORE COMMAND complete — %d clips restored: %s",
                         total, counts)
            self._command_confirmation_fx()
        except Exception:
            logger.exception("Restore command failed")

    def _command_sleep(self, ctx: PipelineContext):
        """Pause active listening until wake_listening. The detector keeps
        running (so 'jetson wake up' is still heard) and the strip switches
        to a slow breathing pulse in the sleep color — a visible 'asleep'
        indicator that overrides the idle preset until wake."""
        if self.asleep:
            logger.info("SLEEP COMMAND — already asleep")
            return
        self.asleep = True
        logger.info("SLEEP COMMAND — active listening paused "
                     "(say 'jetson wake up' to resume)")
        try:
            if self.wled_rt is not None:
                self.wled_rt.start_sleep_indicator(
                    color=self.sleep_color, period_s=self.sleep_breath_s)
        except Exception:
            logger.exception("Sleep indicator failed (still asleep)")

    def _command_wake(self, ctx: PipelineContext):
        """Resume active listening and return the lights to normal idle."""
        was_asleep = self.asleep
        self.asleep = False
        logger.info("WAKE COMMAND — active listening resumed%s",
                     "" if was_asleep else " (was already awake)")
        try:
            if self.wled_rt is not None:
                self.wled_rt.stop_sleep_indicator()
        except Exception:
            logger.exception("Stopping sleep indicator failed")
        if was_asleep:
            self._command_confirmation_fx()

    def _command_confirmation_fx(self):
        """Brief full-strip rainbow swirl as visual ack for command commands.
        Falls back to the layered HTTP preset if UDP isn't available."""
        try:
            if self.wled_rt is not None:
                self.wled.set_max_brightness()
                # 1.2s of full-strip rainbow, no audio dependency
                fake_audio = np.ones(int(self.sample_rate * 1.2), dtype=np.float32)
                self.wled_rt.react_to_audio(
                    fake_audio, self.sample_rate,
                    palette="rainbow",
                    update_hz=self.cfg.get("wled", {}).get("realtime", {}).get("update_hz", 30.0),
                )
                # Let it run its course (1.2s + ~0.4s fade)
                time.sleep(1.6)
            else:
                self.wled.layered()
                time.sleep(0.8)
        except Exception:
            logger.exception("Command confirmation FX failed")

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

        if response_files_only:
            # Only use this bucket's fallback files. No layering, single pick,
            # never the file that just played.
            fb = self.memory.get_fallback_clips(ctx.phrase_type)
            if not fb:
                logger.warning("Bucket %s is response_files_only but has no fallbacks", ctx.phrase_type)
                return
            pick = self._fresh_first(fb, self.memory.last_played)[0]
            ctx.responded_early = True
            ctx.response_paths = [pick]
            logger.info("Response (early, %s files-only): SINGLE → %s", ctx.phrase_type, pick)
            self.wled.intimate()
            self._react_and_play_file(pick, valence=0.0, arousal=0.5)
            self.memory.mark_played(ctx.response_paths)
            self._spawn_signal_ready(ctx)
            return

        self._wait_for_earlier_saves(ctx)
        has_existing = self.memory.has_live_clips() or (
            self.mode in ("fallback", "hybrid") and self.memory.get_fallback_clips(ctx.phrase_type)
        )
        if not has_existing:
            return

        live_entries, fallback_paths = self._gather_split(ctx.phrase_type)
        all_paths_count = len(live_entries) + len(fallback_paths)
        if all_paths_count == 0:
            return

        # Chorus size is fixed by config (recent + random); mood no longer
        # shrinks the layer count. max_layers is still the absolute ceiling.
        layers_cap = min(self.max_layers, all_paths_count)

        ctx.responded_early = True

        if self.response_mode == "single" or layers_cap == 1:
            # Single-pick path: prefer the highest-quality clip from the
            # most-recent distinct speaker. Fall back to fallbacks if no
            # live clip clears the quality floor.
            min_q = self.cfg["playback"].get("layered_min_quality", 0.0)
            pinned_paths, _, _ = self._build_layered_picks(
                live_entries, [], recent_n=1, random_n=0,
                max_layers=1, min_q=min_q,
            )
            if pinned_paths:
                pick = pinned_paths[0]
            elif fallback_paths:
                pick = self._fresh_first(fallback_paths, self.memory.last_played)[0]
            else:
                pick = live_entries[-1].path  # last resort: newest live
            logger.info("Response (early): SINGLE → %s", pick)
            self.wled.intimate()
            self._react_and_play_file(pick, valence=0.0, arousal=0.5)
            ctx.response_paths = [pick]
        else:
            # Speaker-dedup chorus:
            #   - First `recent_n` slots: most-recent DISTINCT speakers,
            #     deterministic. Per speaker we play their newest clip.
            #   - Next `random_n` slots: random sample of OTHER distinct
            #     speakers, recently-heard ones last; one of their top takes.
            #   - Fallbacks top-up only if live speakers can't fill the budget.
            # See _build_layered_picks for the full rule.
            recent_n = self.cfg["playback"].get("layered_recent_count", 3)
            random_n = self.cfg["playback"].get("layered_random_count", 5)
            min_q = self.cfg["playback"].get("layered_min_quality", 0.0)
            dedup_thr = self.cfg["playback"].get("chorus_dedup_threshold", 0.0)
            pinned_paths, sampled_paths, info = self._build_layered_picks(
                live_entries, fallback_paths,
                recent_n=recent_n, random_n=random_n,
                max_layers=layers_cap, min_q=min_q,
                chorus_dedup_threshold=dedup_thr,
            )
            picks = sampled_paths + pinned_paths   # pinned at end → loudest

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
                logger.info("Response (early): SINGLE (sparse pool) → %s", kept_paths[0])
                self.wled.intimate()
                self._react_and_play_file(kept_paths[0], 0.0, 0.5)
            else:
                logger.info(
                    "Response (early): LAYERED %d clips [%d pinned-recent + %d random "
                    "from %d distinct speakers, %d random replayed within %.0fs, "
                    "dropped %d below min_q=%.2f, centroid-dedup rejected %d (cos≥%.2f)]: %s",
                    len(clips), info["pinned_count"], info["sampled_count"],
                    info["distinct_speakers"], info["replayed"], self.replay_cooldown_sec,
                    info["dropped_low_q"], min_q,
                    info["embedding_rejects"], dedup_thr,
                    [p.split("/")[-1] for p in kept_paths])
                self.wled.layered()
                mixed = layer_clips(clips, self.sample_rate,
                                     self._mood_modulated_playback_cfg(ctx))
                # Layered chorus uses the rainbow palette — full-spectrum
                # band drifting along the strip, brightness following audio.
                # Matches the celebratory feel of the WLED preset 4.
                self._react_and_play_array(mixed, 0.0, 0.5, palette="rainbow")
            ctx.response_paths = kept_paths

        self.memory.mark_played(ctx.response_paths)
        self._spawn_signal_ready(ctx)

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
        knows whether the user said 'too', or named Sandy. When the bucket
        changes, also relocate the captured clip into the new bucket's live
        dir so trim_and_resave (and final memory entry) point at the right
        folder — otherwise an ily_sandy clip ends up in audio/live/i_love_you/."""
        if not ctx.report:
            return
        new_type = None
        if ctx.report.matched_bucket and ctx.report.matched_bucket != ctx.phrase_type:
            new_type = ctx.report.matched_bucket
        elif ctx.report.has_too and ctx.phrase_type != "ily_too":
            new_type = "ily_too"
        if new_type is None:
            return

        old_type = ctx.phrase_type
        logger.info("Reconcile phrase_type: detector=%s → transcript=%s",
                     old_type, new_type)
        ctx.phrase_type = new_type

        if not ctx.clip_path:
            return
        try:
            new_dir = self.memory.bucket_live_dir(new_type)
        except Exception:
            return
        if not new_dir:
            return
        old_filename = os.path.basename(ctx.clip_path)
        # Recorder names files "<phrase_type>_<timestamp>.wav" (see
        # modules/recorder.py:save_clip). Swap the prefix so the file
        # name stays consistent with the bucket it now belongs to.
        if old_filename.startswith(f"{old_type}_"):
            new_filename = f"{new_type}_" + old_filename[len(old_type) + 1:]
        else:
            new_filename = old_filename
        new_path = os.path.join(new_dir, new_filename)
        if os.path.abspath(new_path) == os.path.abspath(ctx.clip_path):
            return
        try:
            os.makedirs(new_dir, exist_ok=True)
            os.rename(ctx.clip_path, new_path)
            logger.info("Reconcile clip path: %s → %s",
                         ctx.clip_path, new_path)
            ctx.clip_path = new_path
        except Exception:
            logger.exception("Could not relocate clip to reconciled bucket dir")

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
            self._respond_fallback(ctx)
            return
        if self.response_mode == "single" or r.is_intimate:
            self._respond_intimate(ctx)
        else:
            self._respond_layered(ctx)

    def _stage_speaker(self, ctx: PipelineContext):
        """Compute the speaker embedding for the captured clip and assign a
        speaker_id by matching against existing speaker centroids. Runs on
        the FULL captured audio (pre-trim, pre-enhance) so ECAPA gets the
        ~5s of context it needs for stable embeddings. Skipped if the
        identifier isn't wired, the clip will be rejected anyway, or the
        clip's bucket doesn't persist user recordings (no benefit)."""
        if self.speaker is None or not self.speaker.enabled:
            return
        if ctx.report is None or not self.quality.should_save(ctx.report):
            return
        if not self.memory.bucket_allows_save(ctx.phrase_type):
            return
        if len(ctx.clip) == 0:
            return
        try:
            emb = self.speaker.embed(ctx.clip, self.sample_rate)
            sid, sim = self.speaker.assign(emb, self.memory.speaker_centroids())
            ctx.speaker_id = sid
            ctx.speaker_embedding = emb.tolist()
            logger.info("Speaker: %s (cos=%.2f) for %s",
                         sid, sim, ctx.clip_path)
        except Exception:
            logger.exception("Speaker embedding failed — saving clip without id")

    def _stage_save(self, ctx: PipelineContext):
        if not ctx.clip_path or ctx.report is None:
            return
        # Respect per-bucket save policy (e.g. Sandy bucket: never save user recordings)
        if not self.memory.bucket_allows_save(ctx.phrase_type):
            logger.info("Clip NOT SAVED [%s] — bucket policy save_user_recordings=false",
                         ctx.phrase_type)
            self._record_reject(ctx.phrase_type, "bucket_no_save")
            return
        if not self.quality.should_save(ctx.report) or self.mode == "fallback":
            logger.info("Clip REJECTED [%s] (score=%.2f, reasons=%s)",
                         ctx.phrase_type, ctx.report.score, ctx.report.reject_reasons)
            for reason in (ctx.report.reject_reasons or ["unknown"]):
                self._record_reject(ctx.phrase_type, reason)
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
            speaker_id=ctx.speaker_id,
            speaker_embedding=ctx.speaker_embedding,
        )
        logger.info("Clip SAVED [%s] → %s (score=%.2f, emo=%s, spk=%s, %.1fs after trigger)",
                     ctx.phrase_type, ctx.clip_path, r.score, e.label,
                     ctx.speaker_id or "?",
                     (datetime.now() - ctx.trigger_time).total_seconds())
        self._record_save(ctx.phrase_type)

    def _record_save(self, phrase_type: str):
        self._trigger_count += 1
        self._save_count += 1
        self._maybe_emit_heartbeat()

    def _record_reject(self, phrase_type: str, reason: str):
        self._trigger_count += 1
        self._reject_reasons[reason] = self._reject_reasons.get(reason, 0) + 1
        self._maybe_emit_heartbeat()

    def _maybe_emit_heartbeat(self):
        if self._log_every <= 0 or self._trigger_count % self._log_every != 0:
            return
        n = self._trigger_count
        saved = self._save_count
        rejected = n - saved
        # Sort reasons by frequency for the headline
        reasons_sorted = sorted(self._reject_reasons.items(),
                                 key=lambda kv: -kv[1])
        reasons_str = ", ".join(f"{r}={c}" for r, c in reasons_sorted) or "-"
        save_pct = (saved / n * 100.0) if n else 0.0
        logger.info(
            "Trigger heartbeat (last %d total): %d saved (%.0f%%), %d rejected. "
            "Reasons: %s",
            n, saved, save_pct, rejected, reasons_str,
        )

    def _stage_fadeout(self, ctx: PipelineContext):
        # If signal_ready already kicked off the lights-to-idle transition
        # right after playback, don't re-fire the animation here at end of
        # pipeline (would visibly flash). Just be a safety net for paths
        # that never produced playback (no clips, rejected, etc.).
        if ctx.signal_ready_fired:
            return
        try:
            time.sleep(1.0)
            self.wled.fade_out()
            time.sleep(2.0)
            self.wled.idle()
            if self.wled_rt is not None:
                self.wled_rt.stop_reacting()
        except Exception:
            logger.exception("Fadeout stage failed")

    def _spawn_signal_ready(self, ctx: PipelineContext):
        """Schedule the visual 'mic is hot again' cue. Sleeps mute_cooldown
        (the same window the wake detector uses to ignore speaker bleed),
        then fades the lights down to the idle preset. Runs in a daemon
        thread so the rest of the pipeline (capture, save) keeps moving."""
        if ctx.signal_ready_fired:
            return
        ctx.signal_ready_fired = True
        cooldown = self.cfg.get("vosk", {}).get("mute_cooldown_sec", 0.6)

        def _go():
            try:
                time.sleep(cooldown)
                if self.wled_rt is not None:
                    try:
                        self.wled_rt.stop_reacting()
                    except Exception:
                        pass
                try:
                    self.wled.fade_out()
                except Exception:
                    pass
                # fade_out is a short transition preset; give it a beat
                # before snapping to the idle preset.
                time.sleep(0.4)
                try:
                    self.wled.idle()
                except Exception:
                    pass
            except Exception:
                logger.exception("Signal-ready transition failed")

        threading.Thread(target=_go, daemon=True).start()

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
            self._spawn_signal_ready(ctx)
        elif len(ctx.clip) > 0:
            from modules.effects import apply_volume, fade_in, fade_out
            soft = normalize_audio(ctx.clip)
            soft = apply_volume(soft, 0.5)
            soft = fade_in(soft, int(0.2 * self.sample_rate))
            soft = fade_out(soft, int(0.3 * self.sample_rate))
            self._react_and_play_array(soft, e.valence, e.arousal)
            self._spawn_signal_ready(ctx)

    def _respond_layered(self, ctx: PipelineContext):
        r = ctx.report
        e = r.emotion
        logger.info("Response: LAYERED [%s] V=%+.2f A=%.2f", e.label, e.valence, e.arousal)
        self.wled.layered()

        # Chorus size is the absolute ceiling; mood drives pace + reverb
        # downstream but not layer count anymore.
        n = self.max_layers

        # Build the same speaker-dedup chorus the early-response path uses.
        # When the bucket is non-default, only that bucket's clips count;
        # plain "ily" stays a catch-all minus isolated buckets — see
        # _gather_split for the full rule.
        live_entries, fallback_paths = self._gather_split(ctx.phrase_type)
        recent_n = self.cfg["playback"].get("layered_recent_count", 3)
        random_n = self.cfg["playback"].get("layered_random_count", 5)
        min_q = self.cfg["playback"].get("layered_min_quality", 0.0)
        dedup_thr = self.cfg["playback"].get("chorus_dedup_threshold", 0.0)
        pinned_paths, sampled_paths, info = self._build_layered_picks(
            live_entries, fallback_paths,
            recent_n=recent_n, random_n=random_n,
            max_layers=n, min_q=min_q,
            chorus_dedup_threshold=dedup_thr,
        )
        picks = sampled_paths + pinned_paths  # pinned at end → loudest

        clips_to_layer = []
        chosen_paths = []
        for p in picks:
            try:
                data, _ = sf.read(p, dtype="int16")
                clips_to_layer.append(data)
                chosen_paths.append(p)
            except Exception:
                logger.warning("Could not read clip: %s", p)

        # The user's just-spoken phrase (ctx.clip) is intentionally NOT
        # appended here — it gets saved at end-of-pipeline and shows up in
        # the NEXT trigger's response (via layered_recent_count). Including
        # it now would echo the speaker back to themselves, which feels
        # like a delay-line rather than a chorus.

        if not clips_to_layer:
            logger.warning("No clips available for layered response")
            self.wled.reject()
            return

        logger.info(
            "Layered picks: %d clips [%d pinned-recent + %d random from %d distinct "
            "speakers, %d random replayed within %.0fs, dropped %d below min_q=%.2f, "
            "centroid-dedup rejected %d (cos≥%.2f)]: %s",
            len(chosen_paths), info["pinned_count"], info["sampled_count"],
            info["distinct_speakers"], info["replayed"], self.replay_cooldown_sec,
            info["dropped_low_q"], min_q,
            info["embedding_rejects"], dedup_thr,
            [p.split("/")[-1] for p in chosen_paths])

        mixed = layer_clips(clips_to_layer, self.sample_rate,
                             self._mood_modulated_playback_cfg(ctx))
        # Late-respond layered also uses rainbow — same chorus semantics.
        self._react_and_play_array(mixed, e.valence, e.arousal, palette="rainbow")
        if chosen_paths:
            self.memory.mark_played(chosen_paths)
        self._spawn_signal_ready(ctx)

    def _respond_fallback(self, ctx: PipelineContext):
        logger.info("Response: FALLBACK [%s] (score=%.2f)",
                     ctx.phrase_type, ctx.report.score if ctx.report else 0.0)
        self.wled.reject()
        # Stay inside the matched bucket — never pull ily fallbacks for a
        # monster trigger and vice versa.
        fb = self.memory.get_fallback_clips(ctx.phrase_type)
        if fb:
            pick = self._fresh_first(fb, self.memory.last_played)[0]
            self.playback.play_file(pick)
            self.memory.mark_played([pick])
            self._spawn_signal_ready(ctx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_response_paths(self, phrase_type: str = "ily") -> list[str]:
        """Convenience: live + fallback flattened (used by callers that don't
        care about the split). Layered playback uses _gather_split() instead
        so it can guarantee the most-recent live clips in every chorus and
        weight by quality."""
        live_entries, fallback = self._gather_split(phrase_type)
        return [e.path for e in live_entries] + fallback

    def _gather_split(self, phrase_type: str = "ily") -> tuple[list[ClipEntry], list[str]]:
        """Return (live_entries, fallback_paths). live_entries are full
        ClipEntry objects in chronological order — newest at the end — so
        callers can pin the most-recent voices AND access per-clip quality
        for weighted selection.

        For specific buckets (ily_too, monster, ily_sandy, ...) we ONLY draw
        from that bucket's own pool. The default ily bucket is broader but
        excludes isolated buckets (monster, etc.) so cross-fire is impossible.
        """
        live: list[ClipEntry] = []
        fallback: list[str] = []

        if self.response_mode == "single" and self.too_only:
            if self.mode in ("live", "hybrid"):
                live.extend(self.memory.get_clips_by_type("ily_too"))
            if self.mode in ("fallback", "hybrid"):
                fallback.extend(self.memory.get_fallback_clips("ily_too"))
            return live, fallback

        if phrase_type not in ("ily",):
            if self.mode in ("live", "hybrid"):
                live.extend(self.memory.get_clips_by_type(phrase_type))
            if self.mode in ("fallback", "hybrid"):
                fallback.extend(self.memory.get_fallback_clips(phrase_type))
            return live, fallback

        # Default ily bucket — pool is STRICTLY clips saved with
        # phrase_type=="ily". We used to fold in non-isolated buckets
        # (e.g. ily_too) here, but those carry their own trailing word
        # ("too") past "you" — which played in an ily chorus sounds like
        # "I love you sa…". Strict-only keeps the chorus clean.
        if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
            live.extend(self.memory.get_clips_by_type("ily"))
        if self.mode in ("fallback", "hybrid"):
            fallback.extend(self.memory.get_fallback_clips("ily"))
        return live, fallback

    def _build_layered_picks(self, live_entries: list[ClipEntry],
                              fallback_paths: list[str],
                              recent_n: int, random_n: int,
                              max_layers: int, min_q: float,
                              chorus_dedup_threshold: float = 0.0,
                              ) -> tuple[list[str], list[str], dict]:
        """Build the layered chorus pick list under the speaker-dedup rule.

        Output is two parallel lists ``(pinned_paths, sampled_paths)``:
          - ``pinned_paths`` are the first-N most-recent DISTINCT speakers,
            ordered with the most recent last. Per pinned speaker we play
            their NEWEST clip (it already cleared the save-threshold gate,
            so freshness wins). The pinned set ignores ``min_q``.
          - ``sampled_paths`` are up to ``random_n`` additional speakers,
            sampled WITHOUT replacement from every other speaker in the pool
            (anyone not already pinned). Speakers heard within
            ``replay_cooldown_sec`` go to the back of the line (see
            _fresh_first), so back-to-back triggers bring different voices.
            Per sampled speaker we play one of their top takes (within
            ``take_quality_margin`` of their best), rotating so a speaker
            with several good takes doesn't always replay the same one.

        Total layer count is capped at ``max_layers`` and at the number of
        distinct speakers + fallbacks available. Fallback paths are only
        appended if the live pool can't fill ``recent_n + random_n``.

        Legacy clips without a ``speaker_id`` are treated as distinct
        speakers (one slot each) so they don't all collapse into a single
        chorus position before backfill completes.
        """
        # 1. Group live entries by speaker. For each speaker, remember their
        #    newest clip (pinned slots), best-quality clip (min_q gate) and
        #    all takes (random slots rotate among the top ones).
        by_sid: dict[str, list[ClipEntry]] = {}
        nospk: list[ClipEntry] = []
        for c in live_entries:
            if c.speaker_id:
                by_sid.setdefault(c.speaker_id, []).append(c)
            else:
                nospk.append(c)

        # speakers: (newest_ts, newest_clip, best_clip, sid)
        speakers: list[tuple[str, ClipEntry, ClipEntry, str]] = []
        takes_by_sid: dict[str, list[ClipEntry]] = {}
        for sid, clips in by_sid.items():
            newest = max(clips, key=lambda c: c.timestamp)
            best = max(clips, key=lambda c: (c.score or 0.0, c.timestamp))
            speakers.append((newest.timestamp, newest, best, sid))
            takes_by_sid[sid] = clips
        # Un-clustered clips each stand alone — newest == best == the clip.
        for i, c in enumerate(nospk):
            sid = f"__nospk_{i}_{c.path}"
            speakers.append((c.timestamp, c, c, sid))
            takes_by_sid[sid] = [c]

        # When each speaker was last heard (any of their takes).
        now = time.time()
        spk_played = {sid: max(c.last_played_ts for c in takes)
                      for sid, takes in takes_by_sid.items()}
        replayed = 0

        def _recently_heard(sid: str) -> bool:
            return now - spk_played[sid] < self.replay_cooldown_sec

        # 2. Sort speakers newest-first and apply selection with embedding
        #    dedup as a runtime safety net. Even when speaker_id clustering
        #    fragments a person across multiple ids (common for short noisy
        #    clips), an embedding-cosine check at chorus time rejects any
        #    candidate too close to anything already selected.
        speakers.sort(key=lambda x: x[0], reverse=True)
        recent_n = max(0, recent_n)
        random_n = max(0, random_n)

        # Build per-speaker centroids (mean of all clips in each cluster,
        # L2-normalised). The chorus dedup compares each candidate against
        # the CENTROIDS of already-selected speakers — that's smoother
        # than pairwise-to-individual-clip because the mean of N noisy
        # embeddings is closer to the speaker's true voice than any
        # single one. Critical for legacy short-trimmed clips where
        # individual embeddings are unstable.
        import numpy as np
        centroid_by_sid: dict[str, np.ndarray] = {}
        if chorus_dedup_threshold > 0:
            sid_to_embs: dict[str, list[np.ndarray]] = {}
            for c in live_entries:
                if c.speaker_id and c.speaker_embedding:
                    sid_to_embs.setdefault(c.speaker_id, []).append(
                        np.asarray(c.speaker_embedding, dtype=np.float32))
            for sid, embs_list in sid_to_embs.items():
                arr = np.stack(embs_list)
                mean = arr.mean(axis=0)
                n_ = float(np.linalg.norm(mean))
                if n_ > 1e-9:
                    centroid_by_sid[sid] = (mean / n_).astype(np.float32)

        selected_centroids: list[tuple[str, np.ndarray]] = []  # (sid, centroid)
        embedding_rejects = 0

        def _too_similar(clip: ClipEntry, sid: str) -> bool:
            nonlocal embedding_rejects
            if chorus_dedup_threshold <= 0 or not selected_centroids:
                return False
            cand = centroid_by_sid.get(sid)
            if cand is None:
                if not clip.speaker_embedding:
                    return False
                cand = np.asarray(clip.speaker_embedding, dtype=np.float32)
            for _, c in selected_centroids:
                if float(np.dot(cand, c)) >= chorus_dedup_threshold:
                    embedding_rejects += 1
                    return True
            return False

        def _accept(clip: ClipEntry, sid: str):
            cand = centroid_by_sid.get(sid)
            if cand is None and clip.speaker_embedding:
                cand = np.asarray(clip.speaker_embedding, dtype=np.float32)
            if cand is not None:
                selected_centroids.append((sid, cand))

        # 2a. Pinned: walk speakers newest-first, take up to recent_n that
        #     survive the centroid dedup. Speakers skipped here become
        #     ineligible for the random pool too (their voice is already
        #     represented by whichever pinned speaker they collided with).
        pinned_clips: list[ClipEntry] = []
        pinned_sids: list[str] = []
        excluded_sids: set[str] = set()
        for entry in speakers:
            if len(pinned_clips) >= recent_n:
                break
            ts, newest_clip, best_clip, sid = entry
            if _too_similar(newest_clip, sid):
                excluded_sids.add(sid)
                continue
            _accept(newest_clip, sid)
            pinned_clips.append(newest_clip)
            pinned_sids.append(sid)

        # 2b. Random pool: speakers not chosen for pinned AND not embedding-
        #     excluded by a pinned voice. Gate by min_q on the speaker's
        #     BEST clip (the one we'd actually play).
        pinned_sid_set = set(pinned_sids)
        rest = [e for e in speakers if e[3] not in pinned_sid_set and e[3] not in excluded_sids]
        qualified_rest = [e for e in rest if (e[2].score or 0.0) >= min_q]
        dropped_low_q = len(rest) - len(qualified_rest)

        layers_budget = max(0, max_layers - len(pinned_clips))
        random_budget = min(random_n, layers_budget)
        sampled_clips: list[ClipEntry] = []
        sampled_sids: set[str] = set()
        if random_budget > 0 and qualified_rest:
            # Fresh-first shuffle of the qualified pool, then greedily accept
            # up to the budget while honoring the embedding dedup. The
            # shuffle keeps selection random; the filter keeps the chorus
            # diverse.
            candidates = self._fresh_first(
                qualified_rest, lambda e: spk_played[e[3]], now)
            for entry in candidates:
                if len(sampled_clips) >= random_budget:
                    break
                sid = entry[3]
                take = self._pick_take(takes_by_sid[sid], min_q, now)
                if _too_similar(take, sid):
                    continue
                _accept(take, sid)
                sampled_clips.append(take)
                sampled_sids.add(sid)
                replayed += _recently_heard(sid)

        # 2c. Underfill relaxation — if strict dedup left the chorus
        #     unexpectedly thin (e.g. the bucket only has a few
        #     centroid-similar speakers), relax the dedup threshold by
        #     1.5× and run one more pass over speakers previously
        #     excluded. Cap at the original target. Better a slightly
        #     less diverse chorus than three voices when we asked for
        #     eight.
        relax_filled = 0
        target = recent_n + random_n
        current = len(pinned_clips) + len(sampled_clips)
        if (current < target and chorus_dedup_threshold > 0
                and (excluded_sids or qualified_rest)):
            relaxed = min(chorus_dedup_threshold * 1.5, 0.7)
            # Reconsider everything we passed over — pinned-excluded plus
            # any qualified random candidates that lost the dedup race.
            second_pass = [e for e in speakers
                           if e[3] not in pinned_sid_set
                           and e[3] not in sampled_sids]
            second_pass = self._fresh_first(
                second_pass, lambda e: spk_played[e[3]], now)
            for entry in second_pass:
                if len(pinned_clips) + len(sampled_clips) >= target:
                    break
                ts, newest_clip, best_clip, sid = entry
                if (newest_clip.score or 0.0) < min_q and (best_clip.score or 0.0) < min_q:
                    continue
                cand_clip = self._pick_take(takes_by_sid[sid], min_q, now)
                # Reuse _too_similar but with the temporarily-relaxed threshold.
                orig_thr = chorus_dedup_threshold
                # Inline check at relaxed threshold (don't mutate closure state).
                emb = (np.asarray(cand_clip.speaker_embedding, dtype=np.float32)
                       if cand_clip.speaker_embedding else None)
                too_close = False
                if emb is not None:
                    for _, c in selected_centroids:
                        if float(np.dot(emb, c)) >= relaxed:
                            too_close = True
                            break
                if too_close:
                    continue
                _accept(cand_clip, sid)
                sampled_clips.append(cand_clip)
                sampled_sids.add(sid)
                replayed += _recently_heard(sid)
                relax_filled += 1

        # ``layer_clips`` treats the LAST element as loudest/centered, so we
        # output pinned with newest-LAST.
        pinned_paths = list(reversed([c.path for c in pinned_clips]))
        sampled_paths: list[str] = [c.path for c in sampled_clips]

        # 4. Top up from fallbacks if the live pool is too small to hit
        #    recent_n + random_n. Each fallback path is treated as distinct.
        remaining = (recent_n + random_n) - len(pinned_paths) - len(sampled_paths)
        remaining = min(remaining, max_layers - len(pinned_paths) - len(sampled_paths))
        if remaining > 0 and fallback_paths:
            fb = self._fresh_first(fallback_paths, self.memory.last_played, now)
            sampled_paths.extend(fb[:remaining])

        info = {
            "distinct_speakers": len(speakers),
            "pinned_count": len(pinned_paths),
            "sampled_count": len(sampled_paths),
            "replayed": replayed,
            "dropped_low_q": dropped_low_q,
            "embedding_rejects": embedding_rejects,
            "relax_filled": relax_filled,
            "pinned_sids": pinned_sids,
        }
        return pinned_paths, sampled_paths, info

    def _fresh_first(self, items: list, last_played, now: float | None = None) -> list:
        """Random order, except anything played within replay_cooldown_sec
        goes to the back — least recently played first among those. With a
        big pool nothing from the last few triggers repeats; with a small one
        it rotates through everyone instead of repeating at random.
        ``last_played(item)`` returns epoch seconds, 0 if never played."""
        if now is None:
            now = time.time()
        fresh, stale = [], []
        for it in items:
            ts = last_played(it)
            if now - ts < self.replay_cooldown_sec:
                stale.append((ts, it))
            else:
                fresh.append(it)
        random.shuffle(fresh)
        random.shuffle(stale)
        stale.sort(key=lambda x: x[0])  # stable sort keeps ties shuffled
        return fresh + [it for _, it in stale]

    def _pick_take(self, takes: list[ClipEntry], min_q: float, now: float) -> ClipEntry:
        """One take for a random-slot speaker: any take within
        take_quality_margin of their best (and at least min_q), preferring
        one that hasn't played recently."""
        best = max((c.score or 0.0) for c in takes)
        floor = max(min_q, best - self.take_quality_margin)
        eligible = [c for c in takes if (c.score or 0.0) >= floor] or takes
        return self._fresh_first(eligible, lambda c: c.last_played_ts, now)[0]

    def _weighted_sample(self, items: list, weights: list[float], k: int) -> list:
        """Weighted sample without replacement (Efraimidis-Spirakis). Each
        item gets key = u^(1/weight); top-k by key are returned. O(n)."""
        if not items or k <= 0:
            return []
        if k >= len(items):
            shuffled = list(zip(items, weights))
            random.shuffle(shuffled)
            return [it for it, _ in shuffled]
        keyed = []
        for it, w in zip(items, weights):
            w = max(float(w), 1e-9)
            u = random.random() or 1e-9
            key = u ** (1.0 / w)
            keyed.append((key, it))
        keyed.sort(reverse=True, key=lambda x: x[0])
        return [it for _, it in keyed[:k]]

    def _isolated_phrase_types(self) -> set[str]:
        """Bucket ids that should never be mixed into the default ily pool.
        Empty set if the matcher isn't wired or no isolated buckets exist."""
        if self.matcher is None:
            return set()
        return self.matcher.isolated_bucket_ids()

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
                return self._fresh_first(pool, lambda c: c.last_played_ts)[0].path
        if self.memory.has_live_clips():
            # Random recent clip — must respect bucket isolation. For an
            # isolated bucket (monster, ...) only its own clips count; for
            # the ily catch-all, exclude any isolated bucket's clips.
            recent = self.memory.get_recent_clips(min(self.emotion_pool_size, 16))
            isolated = self._isolated_phrase_types()
            if bucket_filter is not None:
                recent = [c for c in recent if c.phrase_type == bucket_filter]
            else:
                recent = [c for c in recent if c.phrase_type not in isolated]
            if recent:
                return self._fresh_first(recent, lambda c: c.last_played_ts)[0].path
        fb = self.memory.get_fallback_clips(bucket)
        if fb:
            return self._fresh_first(fb, self.memory.last_played)[0]
        return None

    def _react_and_play_file(self, path: str, valence: float, arousal: float,
                              palette: str = "emotion"):
        try:
            data, file_sr = sf.read(path, dtype="float32")
            self._start_lighting_react(data, file_sr, valence, arousal, palette)
            self.playback.play_file(path)
        finally:
            self._stop_lighting_react()

    def _react_and_play_array(self, audio: np.ndarray, valence: float, arousal: float,
                               palette: str = "emotion"):
        try:
            self._start_lighting_react(audio, self.sample_rate, valence, arousal, palette)
            self.playback.play_array(audio, self.sample_rate)
        finally:
            self._stop_lighting_react()

    def _start_lighting_react(self, audio: np.ndarray, sr: int,
                              valence: float, arousal: float,
                              palette: str = "emotion"):
        if self.wled_rt is not None:
            # CRITICAL: WLED's master brightness applies on top of UDP DRGB
            # frames. The fade_out preset drops bri to ~5/255, so without
            # this reset the next chorus paints at ~2% perceived brightness.
            try:
                self.wled.set_max_brightness()
            except Exception:
                pass
            # 30Hz matches WLED-Gledopto's hardware paint ceiling on 210 LEDs
            # (probed 2026-05-14: strip runs at ~44 fps max even when fed faster).
            update_hz = self.cfg.get("wled", {}).get("realtime", {}).get("update_hz", 30.0)
            self.wled_rt.react_to_audio(audio, sr,
                                         valence=valence, arousal=arousal,
                                         palette=palette,
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
        cfg["layer_gap_sec"] = cfg.get("layer_gap_sec", 0.16) * m.pace
        # More intimacy → less reverb mix (closer voices)
        if "reverb_decay" in cfg:
            cfg["reverb_decay"] = max(0.05, cfg["reverb_decay"] * (1.2 - 0.5 * m.intimacy))
        return cfg
