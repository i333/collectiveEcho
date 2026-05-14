#!/usr/bin/env python3
"""
Calibration script for I Love You Mirror.

Records samples and suggests threshold values:
  1. Room noise (5 seconds)
  2. Soft "I love you" (5 seconds)
  3. Loud "I love you" (5 seconds)
"""

import sys
import time

import numpy as np
import sounddevice as sd
import yaml


SAMPLE_RATE = 16000
DURATION = 5


def record(prompt: str) -> np.ndarray:
    print(f"\n>>> {prompt}")
    print(f"    Recording for {DURATION} seconds...")
    input("    Press Enter when ready...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    for i in range(DURATION, 0, -1):
        print(f"    {i}...", end=" ", flush=True)
        time.sleep(1)
    sd.wait()
    print("Done!")
    return audio[:, 0]


def rms(audio: np.ndarray) -> float:
    samples = audio.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


def peak(audio: np.ndarray) -> float:
    samples = audio.astype(np.float32) / 32768.0
    return float(np.max(np.abs(samples)))


def main():
    print("=" * 50)
    print("  I Love You Mirror — Calibration")
    print("=" * 50)

    # 1. Room noise
    noise = record("ROOM NOISE: Stay quiet, let the room breathe.")
    noise_rms = rms(noise)
    noise_peak = peak(noise)
    print(f"    Noise RMS: {noise_rms:.6f}  Peak: {noise_peak:.6f}")

    # 2. Soft "I love you"
    soft = record("SOFT 'I LOVE YOU': Whisper or say it softly.")
    soft_rms = rms(soft)
    soft_peak = peak(soft)
    print(f"    Soft RMS: {soft_rms:.6f}  Peak: {soft_peak:.6f}")

    # 3. Loud "I love you"
    loud = record("LOUD 'I LOVE YOU': Say it with feeling!")
    loud_rms = rms(loud)
    loud_peak = peak(loud)
    print(f"    Loud RMS: {loud_rms:.6f}  Peak: {loud_peak:.6f}")

    # Suggest thresholds
    print("\n" + "=" * 50)
    print("  Suggested thresholds")
    print("=" * 50)

    noise_floor = noise_rms * 1.5
    intimate_threshold = (noise_rms + soft_rms) / 2
    strong_threshold = (soft_rms + loud_rms) / 2

    print(f"  noise_floor_rms:  {noise_floor:.6f}  (noise RMS x 1.5)")
    print(f"  intimate_rms:     {intimate_threshold:.6f}  (midpoint noise-soft)")
    print(f"  strong_rms:       {strong_threshold:.6f}  (midpoint soft-loud)")
    print()

    # Offer to update config
    print("Copy these into config.yaml under 'thresholds:'")
    print()
    print(f"  noise_floor_rms: {noise_floor:.6f}")
    print(f"  intimate_rms: {intimate_threshold:.6f}")
    print(f"  strong_rms: {strong_threshold:.6f}")
    print()

    answer = input("Update config.yaml automatically? [y/N] ").strip().lower()
    if answer == "y":
        try:
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            cfg["thresholds"]["noise_floor_rms"] = round(noise_floor, 6)
            cfg["thresholds"]["intimate_rms"] = round(intimate_threshold, 6)
            cfg["thresholds"]["strong_rms"] = round(strong_threshold, 6)
            with open("config.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print("config.yaml updated!")
        except Exception as e:
            print(f"Could not update config: {e}")
    else:
        print("Skipped. Update manually.")


if __name__ == "__main__":
    main()
