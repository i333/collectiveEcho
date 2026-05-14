#!/usr/bin/env python3
"""
I Love You Mirror — main orchestrator.

Listens for "I love you", captures the phrase, analyzes quality,
plays back voice memories, and drives WLED lighting.
"""

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime

import random

import numpy as np
import soundfile as sf
import yaml

from modules.audio_input import AudioInput
from modules.wake_detector import create_detector
from modules.recorder import Recorder
from modules.quality import QualityAnalyzer
from modules.memory_store import MemoryStore
from modules.playback import Playback
from modules.effects import layer_clips, normalize_audio, to_int16, noise_gate
from modules.wled import WLEDController


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
        self.mode = cfg.get("mode", "hybrid")
        self.response_mode = cfg.get("response_mode", "layered")  # "single" or "layered"
        self.too_only = cfg.get("too_only", False)  # --too: only respond from ily_too folder
        self.test_mode = cfg.get("test_mode", False)

        self.audio_input = AudioInput(cfg)
        self.detector = create_detector(cfg)
        self.recorder = Recorder(cfg)
        self.quality = QualityAnalyzer(cfg)
        self.memory = MemoryStore(cfg)
        self.playback = Playback(cfg)
        self.wled = WLEDController(cfg)

        self.logger = logging.getLogger("mirror")
        self._running = False

    def start(self):
        self._running = True
        self.logger.info("=== I Love You Mirror starting ===")
        too_label = " (too-only)" if self.too_only else ""
        self.logger.info("Mode: %s | Response: %s%s | Detector: %s | Test: %s",
                         self.mode, self.response_mode, too_label,
                         self.cfg.get("wake_detector"), self.test_mode)

        # Start audio capture (skip in pure test mode without mic)
        try:
            self.audio_input.start()
        except Exception as e:
            self.logger.warning("Could not start audio input: %s", e)
            self.logger.info("Continuing without live mic — playback/WLED still work")

        # Set idle LEDs
        self.wled.idle()

        # Start wake-word detector (pass shared audio_input for Vosk)
        self.detector.start(on_detected=self._on_detection, audio_input=self.audio_input)

        self.logger.info("Mirror is running. Waiting for 'I love you'...")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.detector.stop()
        self.audio_input.stop()
        self.wled.fade_out()
        self.logger.info("=== Mirror stopped ===")

    def _on_detection(self, phrase_type: str = "ily"):
        """Called when wake phrase is detected. Runs the full response pipeline."""
        # Run in a thread so we don't block the detector
        t = threading.Thread(target=self._handle_trigger, args=(phrase_type,), daemon=True)
        t.start()

    def _handle_trigger(self, phrase_type: str = "ily"):
        trigger_time = datetime.now()
        phrase_label = "I love you too" if phrase_type == "ily_too" else "I love you"
        self.logger.info(">>> TRIGGER [%s] at %s", phrase_label, trigger_time.isoformat())

        # 1. Snapshot pre-buffer IMMEDIATELY — before anything that might block.
        #    The ring buffer keeps filling, so any delay here loses the phrase.
        pre_audio = self.audio_input.get_pre_buffer()

        # 2. Flash LEDs to acknowledge, then settle to wake preset
        self.wled.flash()
        time.sleep(0.3)
        self.wled.wake_neutral()

        # 3. If we already have recordings, respond NOW with existing clips
        #    while capture + analysis happen in the background.
        has_existing = (self.memory.has_live_clips() or
                        (self.mode in ("fallback", "hybrid") and
                         self.memory.get_fallback_clips()))
        responded_early = False

        if has_existing:
            self.logger.info("Responding immediately with existing clips")
            self._respond_with_existing()
            responded_early = True

        # 4. Capture post-detection audio, combine with pre-buffer snapshot
        clip, clip_path = self.recorder.capture_clip_with_pre(
            pre_audio, self.audio_input, phrase_type=phrase_type)
        sr = self.cfg["audio"]["sample_rate"]
        self.logger.info("Clip captured: %d samples (%.2fs) → %s",
                         len(clip), len(clip) / sr if len(clip) > 0 else 0, clip_path)

        # 5. Analyze quality + verify transcript
        if len(clip) > 0:
            report = self.quality.analyze(clip, sr)
        else:
            from modules.quality import QualityReport
            report = QualityReport()
            report.is_usable = False
            report.reject_reasons = ["no_audio"]
            self.logger.warning("No audio captured — using dummy report")

        # Reconcile phrase type: trust transcript if it found "too", else use detector hint
        if report.has_too:
            phrase_type = "ily_too"
        # If detector said "ily_too" but transcript didn't find "too", keep detector's call
        # (transcript may have missed the trailing word)

        phrase_label = "I love you too" if phrase_type == "ily_too" else "I love you"

        # 6. Trim to just the phrase if timing was found
        if len(clip) > 0 and report.has_phrase:
            clip = self.quality.trim_to_phrase(clip, sr, report)
            if clip_path:
                self.recorder.save_clip(clip, clip_path)
                self.logger.info("Trimmed [%s] → %s (%.2fs)",
                                 phrase_label, clip_path, len(clip) / sr)
        elif len(clip) > 0 and not report.has_phrase:
            self.logger.warning("Phrase not found in recording — will not save")

        self._log_trigger(trigger_time, clip_path, report, phrase_type)

        # 7. If we didn't respond early, respond now based on analysis
        if not responded_early:
            if not report.has_phrase:
                self.logger.info("No phrase detected — rejecting")
                self.wled.reject()
            elif self.quality.should_use_fallback(report) and self.mode != "live":
                self._respond_fallback(report)
            elif self.response_mode == "single" or report.is_intimate:
                self._respond_intimate(clip, clip_path, report)
            else:
                self._respond_layered(clip, clip_path, report)

        # 8. Save if quality passes (requires phrase)
        if clip_path and self.quality.should_save(report) and self.mode != "fallback":
            self.memory.add_clip(
                path=clip_path,
                timestamp=trigger_time.isoformat(),
                rms=report.rms,
                score=report.score,
                duration_sec=report.duration_sec,
                is_intimate=report.is_intimate,
                phrase_type=phrase_type,
            )
            self.logger.info("Clip SAVED [%s] → %s (score=%.2f)",
                             phrase_label, clip_path, report.score)
        else:
            self.logger.info("Clip REJECTED [%s] (score=%.2f, reasons=%s)",
                             phrase_label, report.score, report.reject_reasons)

        # 9. Fade back to idle
        time.sleep(1.0)
        self.wled.fade_out()
        time.sleep(2.0)
        self.wled.idle()

    # --- response strategies ----------------------------------------------

    def _gather_response_paths(self) -> list[str]:
        """Gather clip paths for response, respecting too_only and mode."""
        all_paths = []

        if self.response_mode == "single" and self.too_only:
            # --too: only use ily_too clips
            if self.mode in ("live", "hybrid"):
                all_paths.extend(e.path for e in self.memory.get_clips_by_type("ily_too"))
            if self.mode in ("fallback", "hybrid"):
                all_paths.extend(self.memory.get_fallback_clips("ily_too"))
        elif self.response_mode == "single":
            # Single mode: both ily and ily_too clips
            if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
                all_paths.extend(e.path for e in self.memory.get_recent_clips())
            if self.mode in ("fallback", "hybrid"):
                all_paths.extend(self.memory.get_fallback_clips("ily"))
                all_paths.extend(self.memory.get_fallback_clips("ily_too"))
        else:
            # Layered mode: only ily clips (too clips don't layer well)
            if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
                all_paths.extend(e.path for e in self.memory.get_recent_clips())
            if self.mode in ("fallback", "hybrid"):
                all_paths.extend(self.memory.get_fallback_clips("ily"))

        return all_paths

    def _respond_with_existing(self):
        """Respond immediately using only existing clips from memory.
        Called before capture/analysis so the user hears something fast.
        In single mode, picks one random clip. In layered mode, layers all."""
        sr = self.cfg["audio"]["sample_rate"]
        all_paths = self._gather_response_paths()

        if not all_paths:
            self.logger.warning("No existing clips for early response")
            return

        # --- Single mode: pick one random clip ---
        if self.response_mode == "single":
            pick = random.choice(all_paths)
            self.logger.info("Response (early): SINGLE random → %s", pick)
            self.wled.intimate()
            try:
                data, file_sr = sf.read(pick, dtype="float32")
                self.wled.react_to_audio(data, file_sr)
                self.playback.play_file(pick)
            finally:
                self.wled.stop_reacting()
            return

        # --- Layered mode ---
        clips_to_layer = []
        max_layers = self.cfg["playback"]["max_layers"]
        for path in all_paths[:max_layers]:
            try:
                data, file_sr = sf.read(path, dtype="int16")
                clips_to_layer.append(data)
            except Exception:
                self.logger.warning("Could not read clip: %s", path)

        if len(clips_to_layer) == 1:
            self.logger.info("Response (early): SINGLE → %s", all_paths[0])
            self.wled.intimate()
            try:
                data, file_sr = sf.read(all_paths[0], dtype="float32")
                self.wled.react_to_audio(data, file_sr)
                self.playback.play_file(all_paths[0])
            finally:
                self.wled.stop_reacting()
        elif clips_to_layer:
            self.logger.info("Response (early): LAYERED %d clips", len(clips_to_layer))
            self.wled.layered()
            mixed = layer_clips(clips_to_layer, sr, self.cfg["playback"])
            self.wled.react_to_audio(mixed, sr)
            try:
                self.playback.play_array(mixed, sr)
            finally:
                self.wled.stop_reacting()

    def _respond_intimate(self, clip: np.ndarray, clip_path: str, report):
        """Soft, intimate response — play one previous recording gently."""
        self.logger.info("Response: INTIMATE (rms=%.4f)", report.rms)
        self.wled.intimate()

        try:
            if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
                recent = self.memory.get_recent_clips(1)
                data, file_sr = sf.read(recent[0].path, dtype="float32")
                self.wled.react_to_audio(data, file_sr)
                self.playback.play_file(recent[0].path)
            elif self.mode in ("fallback", "hybrid"):
                fallbacks = self.memory.get_fallback_clips()
                if fallbacks:
                    data, file_sr = sf.read(fallbacks[0], dtype="float32")
                    self.wled.react_to_audio(data, file_sr)
                    self.playback.play_file(fallbacks[0])
                elif len(clip) > 0:
                    from modules.effects import apply_volume, fade_in, fade_out
                    sr = self.cfg["audio"]["sample_rate"]
                    soft = normalize_audio(clip)
                    soft = apply_volume(soft, 0.5)
                    soft = fade_in(soft, int(0.2 * sr))
                    soft = fade_out(soft, int(0.3 * sr))
                    self.wled.react_to_audio(soft, sr)
                    self.playback.play_array(soft, sr)
            elif len(clip) > 0:
                from modules.effects import apply_volume
                sr = self.cfg["audio"]["sample_rate"]
                soft = normalize_audio(clip)
                soft = apply_volume(soft, 0.5)
                self.wled.react_to_audio(soft, sr)
                self.playback.play_array(soft, sr)
        finally:
            self.wled.stop_reacting()

    def _respond_layered(self, clip: np.ndarray, clip_path: str, report):
        """Layered response — multiple voices echoing."""
        self.logger.info("Response: LAYERED (rms=%.4f)", report.rms)
        self.wled.layered()

        sr = self.cfg["audio"]["sample_rate"]
        clips_to_layer = []

        if self.mode in ("live", "hybrid") and self.memory.has_live_clips():
            for entry in self.memory.get_recent_clips():
                try:
                    data, file_sr = sf.read(entry.path, dtype="int16")
                    clips_to_layer.append(data)
                except Exception:
                    self.logger.warning("Could not read clip: %s", entry.path)

        # In hybrid/fallback, supplement with fallbacks if not enough
        if self.mode in ("fallback", "hybrid"):
            needed = self.cfg["playback"]["max_layers"] - len(clips_to_layer)
            if needed > 0:
                for fb_path in self.memory.get_fallback_clips()[:needed]:
                    try:
                        data, file_sr = sf.read(fb_path, dtype="int16")
                        clips_to_layer.append(data)
                    except Exception:
                        self.logger.warning("Could not read fallback: %s", fb_path)

        # Add the current clip as the newest layer
        if len(clip) > 0:
            clips_to_layer.append(clip)

        if not clips_to_layer:
            self.logger.warning("No clips available for layered response")
            self.wled.reject()
            return

        mixed = layer_clips(clips_to_layer, sr, self.cfg["playback"])
        self.wled.react_to_audio(mixed, sr)
        try:
            self.playback.play_array(mixed, sr)
        finally:
            self.wled.stop_reacting()

    def _respond_fallback(self, report):
        """Use prerecorded fallback clips."""
        self.logger.info("Response: FALLBACK (score=%.2f)", report.score)
        self.wled.reject()

        fallbacks = self.memory.get_fallback_clips()
        if fallbacks:
            # Play a single fallback
            self.playback.play_file(fallbacks[0])
        else:
            self.logger.warning("No fallback clips available")

    # --- logging ----------------------------------------------------------

    def _log_trigger(self, trigger_time, clip_path, report, phrase_type="ily"):
        self.logger.info(
            "TRIGGER LOG | type=%s | time=%s | clip=%s | rms=%.4f | peak=%.4f | "
            "clip_ratio=%.4f | dur=%.2f | score=%.2f | intimate=%s | "
            "strong=%s | usable=%s | too=%s | reasons=%s",
            phrase_type, trigger_time.isoformat(), clip_path,
            report.rms, report.peak, report.clipping_ratio,
            report.duration_sec, report.score,
            report.is_intimate, report.is_strong,
            report.is_usable, report.has_too, report.reject_reasons,
        )


def main():
    parser = argparse.ArgumentParser(description="I Love You Mirror")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Enable test mode (simulate detector)")
    parser.add_argument("--single", action="store_true", help="Single mode — play one random existing clip per trigger")
    parser.add_argument("--too", action="store_true", help="Only respond from 'I love you too' clips (implies --single)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.test:
        cfg["test_mode"] = True
        cfg["wake_detector"] = "simulate"
    if args.single or args.too:
        cfg["response_mode"] = "single"
    if args.too:
        cfg["too_only"] = True

    setup_logging(cfg)
    mirror = Mirror(cfg)
    mirror.start()


if __name__ == "__main__":
    main()
