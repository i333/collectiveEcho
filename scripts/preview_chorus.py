#!/usr/bin/env python3
"""Hear a whole visit's worth of chorus responses without the mirror.

Simulates one visitor triggering the mirror several times in a row. Every
response goes through the production pick + layering code
(TriggerPipeline._stage_early_respond) AND the playback output stage
(exciter EQ, master gain, brickwall limiter), so the WAV is what the amp
would receive. Prints which voices each response used, which ones repeat
from the previous response, and how hard the limiter worked.

Voice pool, one of:
  (default)     state.json + audio/ under --root. Point --root at a copy of
                the Jetson's install to hear real visitors with their real
                speaker ids — speechbrain is NOT needed, ids and embeddings
                are stored in state.json.
  --virtual N   N stand-in visitors made by pitch-shifting your own trimmed
                takes, 1–4 takes each. For testing alone on a laptop.

Read-only: state.json is never written; play times are simulated in memory
and the clock is simulated too (--spacing), so nothing actually waits.

Examples:
    python3 scripts/preview_chorus.py --virtual 12
    python3 scripts/preview_chorus.py --root ~/jetson-snapshot --ab
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import soundfile as sf
import yaml

from modules.effects import layer_clips, normalize_audio, pitch_shift, to_int16
from modules.memory_store import ClipEntry, MemoryStore
from modules.mood import MoodKeeper
from modules.phrase_match import PhraseMatcher, buckets_from_config
from modules.pipeline import PipelineContext, TriggerPipeline
from modules.playback import Playback

GAP_BETWEEN_TRIGGERS_SEC = 1.5


class _Quiet:
    """Stands in for WLED / mic / recorder / quality — every call is a no-op."""

    def __getattr__(self, name):
        return lambda *a, **k: None


class _CapturePlayback(Playback):
    """The real output stage, captured instead of sent to the DAC."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.last = None

    def play_array(self, audio, sample_rate=None, blocking=True):
        sr = sample_rate or self.sample_rate
        # Skips play_array's auto-boost: it only fires for peak < 0.1, which
        # an RMS-normalized chorus or an enhanced clip never is.
        conditioned = self._condition(normalize_audio(audio))
        out = self._master_stage(conditioned, sr)
        self.last = (conditioned * self.master_gain, out, sr)

    def play_file(self, path, blocking=True):
        data, sr = sf.read(path, dtype="float32")
        self.play_array(data, sr)


def _limiter_stats(pre: np.ndarray, out: np.ndarray, sr: int):
    """(max gain reduction dB, % of time reducing > 1 dB, out rms dBFS, out peak)."""
    frame = max(1, sr // 100)
    n = min(len(pre), len(out)) // frame
    if n == 0:
        return 0.0, 0.0, -120.0, 0.0
    a = np.abs(pre[:n * frame]).reshape(n, frame, -1).max(axis=(1, 2))
    b = np.abs(out[:n * frame]).reshape(n, frame, -1).max(axis=(1, 2))
    loud = a > 1e-4
    gr = np.zeros(n)
    gr[loud] = 20 * np.log10(np.maximum(b[loud], 1e-9) / a[loud])
    rms = float(np.sqrt(np.mean(out ** 2)))
    return (float(gr.min()), float(np.mean(gr < -1.0) * 100),
            20 * np.log10(max(rms, 1e-9)), float(np.abs(out).max()))


def _build_virtual_pool(mem, n, source_dir, vdir, bucket, rng, sr):
    src = sorted(p for p in glob.glob(os.path.join(source_dir, "*.wav"))
                 if sf.info(p).duration < 2.5)
    if not src:
        sys.exit(f"No trimmed takes (<2.5s .wav) in {source_dir} — record a few "
                 "with main.py first, or pass --source.")
    os.makedirs(vdir, exist_ok=True)
    # One distinct pitch per visitor, spread over ±5 semitones but never
    # within ±1 of the original so nobody sounds exactly like you.
    semis = [s for s in np.linspace(-5, 5, n + 2) if abs(s) >= 1.0][:n]
    while len(semis) < n:
        semis.append(rng.uniform(-5, 5))
    rng.shuffle(semis)
    mem.clips = []
    for v, semi in enumerate(semis):
        for j in range(rng.choice([1, 1, 1, 2, 2, 3, 4])):
            audio, file_sr = sf.read(rng.choice(src), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if file_sr != sr:
                sys.exit(f"Source takes must be {sr} Hz (got {file_sr}).")
            audio = pitch_shift(audio, semi * 100 + rng.uniform(-15, 15))
            path = os.path.join(vdir, f"virtual{v:02d}_take{j}.wav")
            sf.write(path, to_int16(audio), sr, subtype="PCM_16")
            mem.clips.append(ClipEntry(
                path=path, timestamp=f"2026-01-01T00:{v:02d}:{j:02d}",
                rms=0.1, score=rng.choice([0.95, 1.0, 1.05, 1.1]),
                duration_sec=len(audio) / sr, is_intimate=False,
                phrase_type=bucket, speaker_id=f"virtual{v:02d}"))
    print(f"Pool: {n} stand-in visitors, {len(mem.clips)} takes "
          f"(pitch-shifted from {len(src)} takes in {source_dir})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=os.path.join(REPO, "config.yaml"))
    ap.add_argument("--root", default=REPO,
                    help="install dir holding state.json + audio/ (default: this repo)")
    ap.add_argument("--virtual", type=int, default=0, metavar="N",
                    help="use N pitch-shifted stand-in visitors instead of state.json")
    ap.add_argument("--source", default=os.path.join(REPO, "audio/live/i_love_you"),
                    help="takes to pitch-shift for --virtual")
    ap.add_argument("--bucket", default="ily")
    ap.add_argument("--triggers", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=15.0,
                    help="simulated seconds between triggers")
    ap.add_argument("--hour", type=int, default=None,
                    help="time-of-day mood to apply (default: now)")
    ap.add_argument("--ab", action="store_true",
                    help="also render the same picks with the old layer timing")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="chorus_preview")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the pipeline's own log lines")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="  %(name)s: %(message)s")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rng = random.Random(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(os.path.abspath(os.path.expanduser(args.root)))  # clip paths are relative

    # Simulated clock: every module calls time.time() at call time, so this
    # drives last_played_ts and the replay cooldown without real waiting.
    sim_now = [time.time()]
    time.time = lambda: sim_now[0]

    matcher = PhraseMatcher(buckets=buckets_from_config(cfg))
    mem = MemoryStore(cfg)
    mem._save_state = lambda: None  # read-only: never touch state.json
    mem.configure_buckets(matcher.buckets)
    sr = cfg["audio"]["sample_rate"]

    if args.virtual:
        _build_virtual_pool(mem, args.virtual, args.source,
                            os.path.join(out_dir, "virtual_takes"),
                            args.bucket, rng, sr)
    else:
        n_spk = len({c.speaker_id or c.path for c in mem.clips
                     if c.phrase_type == args.bucket})
        n_clips = sum(c.phrase_type == args.bucket for c in mem.clips)
        if not n_clips:
            sys.exit(f"No '{args.bucket}' clips in {os.getcwd()}/state.json. Copy the "
                     "Jetson's state.json + audio/ there, or use --virtual N.")
        print(f"Pool: {n_clips} takes from {n_spk} speakers in {os.getcwd()}/state.json")

    playback = _CapturePlayback(cfg)
    pipe = TriggerPipeline(cfg, _Quiet(), _Quiet(), _Quiet(), mem, playback,
                           _Quiet(), wled_rt=None, mood_keeper=None,
                           matcher=matcher, speaker=None)
    when = datetime.now() if args.hour is None else datetime.now().replace(hour=args.hour, minute=0)
    mood = MoodKeeper(state_file=os.path.join(out_dir, "state.json")).current(when)
    pb = cfg["playback"]
    recent_n = pb.get("layered_recent_count", 3)
    sid_of = {c.path: (c.speaker_id or os.path.basename(c.path)) for c in mem.clips}

    print(f"Mood: {mood.period} (pace ×{mood.pace}) | {args.triggers} triggers "
          f"{args.spacing:.0f}s apart | replay_cooldown_sec={pipe.replay_cooldown_sec:g}\n")

    visit, visit_old, prev_random = [], [], set()
    heard, repeats, max_gr = set(), [], 0.0
    t_in_visit = 0.0
    for t in range(1, args.triggers + 1):
        ctx = PipelineContext(phrase_type=args.bucket)
        ctx.mood = mood
        playback.last = None
        pipe._stage_early_respond(ctx)
        if playback.last is None:
            print(f"#{t}: no response — pool is empty for '{args.bucket}'")
            sim_now[0] += args.spacing
            continue
        pre, out, out_sr = playback.last
        paths = ctx.response_paths  # play order: loudest/front voice last
        voices = [sid_of.get(p, "fallback:" + os.path.basename(p)) for p in paths]
        front = set(voices[-recent_n:]) if len(voices) > 1 else set(voices)
        rand = [v for v in voices if v not in front]
        rep = [v for v in rand if v in prev_random]
        repeats.append(len(rep))
        heard.update(voices)
        gr, gr_pct, rms_db, peak = _limiter_stats(pre, out, out_sr)
        max_gr = min(max_gr, gr)

        print(f"#{t} @ {t_in_visit:5.1f}s in visit.wav — {len(voices)} voices, "
              f"{len(out) / out_sr:.1f}s")
        print(f"    front:  {', '.join(reversed(voices[-recent_n:])) if len(voices) > 1 else voices[0]}")
        if rand:
            print(f"    random: {', '.join(rand)}"
                  + (f"   ← repeated from #{t - 1}: {', '.join(rep)}" if rep else ""))
        print(f"    output: rms {rms_db:.1f} dBFS, peak {peak:.3f} "
              f"(ceiling {playback.master_ceiling:g}) | limiter: max {gr:.1f} dB, "
              f"working {gr_pct:.0f}% of the time")
        prev_random = set(rand)

        silence = np.zeros((int(GAP_BETWEEN_TRIGGERS_SEC * out_sr), 2), np.float32)
        visit += [_stereo(out), silence]
        if args.ab:
            old_cfg = pipe._mood_modulated_playback_cfg(ctx)
            old_cfg.update(layer_spacing="phi_growth", pan_width=0.9)
            clips = [sf.read(p, dtype="int16")[0] for p in paths]
            mixed = layer_clips(clips, sr, old_cfg) if len(clips) > 1 else normalize_audio(clips[0])
            playback.play_array(mixed, sr)
            o_gr, o_pct, o_rms, _ = _limiter_stats(*playback.last)
            print(f"    old timing: {len(playback.last[1]) / out_sr:.1f}s, rms {o_rms:.1f} dBFS"
                  f" | limiter: max {o_gr:.1f} dB, working {o_pct:.0f}% of the time")
            visit_old += [_stereo(playback.last[1]), silence]
        t_in_visit += len(out) / out_sr + GAP_BETWEEN_TRIGGERS_SEC
        sim_now[0] += args.spacing

    if not visit:
        return
    sf.write(os.path.join(out_dir, "visit.wav"), np.concatenate(visit), sr, subtype="PCM_16")
    if visit_old:
        sf.write(os.path.join(out_dir, "visit_old_timing.wav"),
                 np.concatenate(visit_old), sr, subtype="PCM_16")
    print(f"\nSummary: {len(heard)} different voices heard over the visit; random voices "
          f"repeated from the previous response: {np.mean(repeats):.1f} on average; "
          f"strongest limiting {max_gr:.1f} dB")
    print(f"Wrote {os.path.join(out_dir, 'visit.wav')}"
          + (" and visit_old_timing.wav (old start times, same voices)" if visit_old else ""))
    print("Audio is post-exciter-EQ at the mirror's level: it will sound thin and "
          "quiet on laptop speakers — that's the 180 Hz high-pass and 0.25 ceiling.")


def _stereo(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 2 else np.column_stack([x, x])


if __name__ == "__main__":
    main()
