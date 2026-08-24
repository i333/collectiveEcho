#!/usr/bin/env python3
"""
I Love You Mirror — main orchestrator.

Wires collaborators, owns the run loop, and delegates trigger handling to
TriggerPipeline. The actual pipeline logic lives in modules/pipeline.py.
"""

import argparse
import logging
import os
import sys
import threading
import time

import yaml

from modules.audio_input import AudioInput
from modules.wake_detector import create_detector
from modules.recorder import Recorder
from modules.quality import QualityAnalyzer
from modules.memory_store import MemoryStore
from modules.playback import Playback
from modules.wled import WLEDController
from modules.asr import create_transcriber
from modules.emotion import create_emotion_analyzer
from modules.mood import MoodKeeper
from modules.phrase_match import PhraseMatcher, buckets_from_config
from modules.pipeline import TriggerPipeline
from modules.speaker import SpeakerIdentifier


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("file", "logs/mirror.log")
    log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


class Mirror:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.test_mode = cfg.get("test_mode", False)
        self.logger = logging.getLogger("mirror")

        # Build the phrase matcher ONCE from config and share it with every
        # consumer (detector, quality analyzer, pipeline). This guarantees
        # they all agree on bucket priorities and anti-overfitting rules.
        self.buckets = buckets_from_config(cfg)
        self.matcher = PhraseMatcher(buckets=self.buckets)
        bucket_summary = [f"{b.id}(prio={b.priority}, patterns={len(b.patterns)})"
                          for b in self.matcher.buckets]
        self.logger.info("Phrase buckets: %s", ", ".join(bucket_summary))

        # Core IO
        self.audio_input = AudioInput(cfg)
        self.memory = MemoryStore(cfg)
        self.memory.configure_buckets(self.buckets)
        self.recorder = Recorder(cfg, memory_store=self.memory)
        self.playback = Playback(cfg)
        # Detector needs the playback ref so it can mute the mic during
        # chorus output (otherwise the speaker re-triggers Vosk through
        # the mic and feeds back).
        self.detector = create_detector(cfg, matcher=self.matcher,
                                         playback=self.playback)

        # WLED — keep HTTP path; optionally enable UDP realtime
        self.wled = WLEDController(cfg)
        self.wled_rt = self._maybe_build_wled_rt(cfg)

        # Pluggable ML backends
        self.transcriber = create_transcriber(cfg)
        self.transcriber.warmup()
        self.emotion = create_emotion_analyzer(cfg)
        try:
            self.emotion.warmup()
        except Exception:
            self.logger.exception("Emotion warmup failed (continuing)")

        # Quality analyzer with injected backends + shared matcher
        self.quality = QualityAnalyzer(
            cfg,
            transcriber=self.transcriber,
            emotion_analyzer=self.emotion,
            matcher=self.matcher,
        )

        # Slow behavioral state
        self.mood_keeper = MoodKeeper(state_file=cfg["memory"].get("state_file", "state.json"))

        # Speaker identification — warms up the ECAPA encoder (cuda if avail.)
        # and backfills speaker_ids for any previously-saved clip that hasn't
        # been clustered yet. Soft-fails if speechbrain isn't installed.
        self.speaker = self._maybe_build_speaker(cfg)
        if self.speaker is not None:
            try:
                self.memory.backfill_speakers(self.speaker)
            except Exception:
                self.logger.exception("Speaker backfill failed (continuing)")

        # Trigger pipeline
        self.pipeline = TriggerPipeline(
            cfg=cfg,
            audio_input=self.audio_input,
            recorder=self.recorder,
            quality=self.quality,
            memory=self.memory,
            playback=self.playback,
            wled=self.wled,
            wled_rt=self.wled_rt,
            mood_keeper=self.mood_keeper,
            matcher=self.matcher,
            speaker=self.speaker,
        )

        self._running = False

    def _maybe_build_speaker(self, cfg: dict):
        """Build + warm up the speaker identifier if enabled in config.
        Returns None if disabled or speechbrain isn't installed (so the
        chorus pipeline still runs, just without speaker dedup)."""
        sc = cfg.get("speaker", {}) or {}
        if not sc.get("enabled", True):
            return None
        try:
            ident = SpeakerIdentifier(cfg)
            ident.warmup()
            return ident
        except Exception:
            self.logger.exception(
                "SpeakerIdentifier unavailable — chorus will skip speaker dedup")
            return None

    @staticmethod
    def _maybe_build_wled_rt(cfg: dict):
        wled_cfg = cfg.get("wled", {})
        rt_cfg = wled_cfg.get("realtime", {})
        if not rt_cfg.get("enabled", False):
            return None
        try:
            from modules.wled_udp import WLEDRealtime
            return WLEDRealtime(
                host=rt_cfg.get("host", wled_cfg.get("host", "127.0.0.1")),
                num_leds=rt_cfg.get("num_leds", 60),
                port=rt_cfg.get("port", 21324),
                hold_timeout_s=rt_cfg.get("hold_timeout_s", 2),
            )
        except Exception:
            logging.getLogger("mirror").exception("Failed to init WLED UDP realtime")
            return None

    def start(self):
        self._running = True
        self.logger.info("=== I Love You Mirror starting ===")
        self.logger.info(
            "Mode: %s | Response: %s | Detector: %s | ASR: %s | Emotion: %s | UDP-WLED: %s | Test: %s",
            self.cfg.get("mode"),
            self.cfg.get("response_mode", "layered"),
            self.cfg.get("wake_detector"),
            self.cfg.get("asr", {}).get("backend", "vosk"),
            self.cfg.get("emotion", {}).get("backend", "prosody"),
            self.wled_rt is not None,
            self.test_mode,
        )

        mic_ok = self._start_audio_input()

        if self.wled_rt is not None:
            self.wled_rt.off()

        if mic_ok:
            # Always release a stale realtime override: a previous run may have
            # left lor=2 from the mic alarm, which would silently block the
            # reactive UDP layer for this whole session.
            self.wled.clear_mic_alarm()
        else:
            # Deaf: nothing can ever trigger, and the service still reports
            # "active". Make the failure visible instead of silent.
            self.logger.error(
                "Microphone unavailable after retries — WLED set to ORANGE; "
                "polling in the background for recovery"
            )
            self.wled.mic_alarm()
            threading.Thread(target=self._mic_recovery_loop, daemon=True).start()

        self.detector.start(on_detected=self._on_detection, audio_input=self.audio_input)

        self.logger.info("Mirror is running. Waiting for 'I love you'...")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
        finally:
            self.stop()

    def _start_audio_input(self, attempts: int = 5, delay_sec: float = 2.0) -> bool:
        """Open the mic, retrying a few times before giving up.

        `systemctl restart` can race the previous process still holding the USB
        device (PortAudio reports "Invalid number of channels"). That used to
        leave the mirror running permanently deaf while looking healthy.
        """
        for i in range(1, attempts + 1):
            try:
                self.audio_input.start()
                if i > 1:
                    self.logger.info("Audio input opened on attempt %d", i)
                return True
            except Exception as e:
                self.logger.warning(
                    "Could not start audio input (attempt %d/%d): %s",
                    i, attempts, e,
                )
                if i < attempts:
                    time.sleep(delay_sec)
        return False

    def _mic_recovery_loop(self, interval_sec: float = 30.0):
        """Poll for the mic; exit once it is back so systemd restarts cleanly.

        Restarting is safer than hot-attaching the stream here — the detector
        has already been handed this AudioInput instance.
        """
        while self._running:
            time.sleep(interval_sec)
            try:
                self.audio_input.start()
            except Exception:
                continue
            self.logger.error(
                "Microphone is available again — exiting so systemd restarts "
                "the service with a live audio path"
            )
            os._exit(1)

    def stop(self):
        self._running = False
        self.detector.stop()
        self.audio_input.stop()
        try:
            self.wled.fade_out()
        except Exception:
            pass
        if self.wled_rt is not None:
            try:
                self.wled_rt.off()
                self.wled_rt.close()
            except Exception:
                pass
        try:
            self.mood_keeper.close()
        except Exception:
            pass
        self.logger.info("=== Mirror stopped ===")

    def _on_detection(self, phrase_type: str = "ily"):
        # Run pipeline in a thread so the detector loop is never blocked
        threading.Thread(
            target=self.pipeline.run,
            args=(phrase_type,),
            daemon=True,
        ).start()


def main():
    parser = argparse.ArgumentParser(description="I Love You Mirror")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Enable test mode (simulate detector)")
    parser.add_argument("--single", action="store_true",
                        help="Single mode — play one random existing clip per trigger")
    parser.add_argument("--too", action="store_true",
                        help="Only respond from 'I love you too' clips (implies --single)")
    parser.add_argument("--udp-wled", action="store_true",
                        help="Force-enable WLED realtime UDP (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.test:
        cfg["test_mode"] = True
        cfg["wake_detector"] = "simulate"
    if args.single or args.too:
        cfg["response_mode"] = "single"
    if args.too:
        cfg["too_only"] = True
    if args.udp_wled:
        cfg.setdefault("wled", {}).setdefault("realtime", {})["enabled"] = True

    setup_logging(cfg)
    mirror = Mirror(cfg)
    mirror.start()


if __name__ == "__main__":
    main()
