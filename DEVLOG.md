# I Love You Mirror — Development Log

## Session: 2026-05-11

### Hardware

- Raspberry Pi 3
- USB microphone (system default device)
- WLED LEDs on local network (`http://wled.local`) — not connected during this session
- Target migration: NVIDIA Jetson Orin Nano

### Issues Found & Fixed

---

#### 1. Pi Dropping SSH Connections (Undervoltage)

**Symptom:** Pi closed the SSH connection mid-session with no application error.

**Root cause:** `dmesg` showed `hwmon hwmon1: Undervoltage detected!` — the power supply is not delivering enough current. This causes system instability and dropped connections. Not a software bug.

**Evidence:**
```
[   15.998202] hwmon hwmon1: Undervoltage detected!
[   18.014213] hwmon hwmon1: Voltage normalised
```

**Fix:** Use a proper 5V/2.5A (or higher) power supply for the RPi 3. A USB cable with thin gauge wire can also cause voltage drop — use a short, thick cable.

---

#### 2. Pre-Buffer Captured Too Late — Phrase Lost from Recording

**Symptom:** Vosk wake detector successfully detects "i love you" and triggers the pipeline. But the quality analyzer's transcript verification always returns an empty string `''`, rejecting every clip with `no_phrase_detected`.

**Root cause:** Timing race in `_handle_trigger()` (main.py). The original order was:

```
1. wled.wake_neutral()     ← blocks 2-5 seconds (WLED unreachable, HTTP timeout)
2. recorder.capture_clip() ← calls get_pre_buffer() HERE — too late
```

The `AudioInput` ring buffer keeps filling continuously via the mic callback. By the time `get_pre_buffer()` was called (after the WLED timeout), the 4-second rolling window had shifted forward, and the actual "i love you" audio was no longer in the buffer. The captured clip contained only post-phrase silence/ambient.

The quality analyzer's Vosk transcript verification then correctly transcribed this silence as `''` and rejected it.

**Timeline from logs showing the delay:**
```
04:16:41 — Vosk detects "i love you"
04:16:46 — WLED timeout fires (5 seconds later)
04:16:46 — get_pre_buffer() called — phrase is GONE from ring buffer
04:16:48 — post-detection capture done
04:16:54 — Vosk verification: transcript='' → rejected
```

**Fix:** Snapshot the ring buffer **immediately** when detection fires, before any blocking call:

```python
# In _handle_trigger():
pre_audio = self.audio_input.get_pre_buffer()  # FIRST — before WLED
self.wled.wake_neutral()                        # THEN — may block
clip, clip_path = self.recorder.capture_clip_with_pre(pre_audio, self.audio_input)
```

Added `capture_clip_with_pre()` to `Recorder` which accepts an already-snapshotted pre-buffer instead of re-reading the (now-stale) ring buffer.

**Files changed:**
- `main.py` — reordered `_handle_trigger()`, snapshot pre-buffer first
- `modules/recorder.py` — added `capture_clip_with_pre()`, extracted `_combine_and_save()`

---

#### 3. Response Delayed by Analysis — Playback Too Slow

**Symptom:** Even when valid recordings exist in memory, the user has to wait through the full capture + Vosk analysis + trimming + save pipeline (~7-10 seconds) before hearing any audio response.

**Root cause:** The original `_handle_trigger()` flow was strictly sequential:

```
capture → analyze → trim → save → respond → fade
```

All steps had to complete before the response played. On the RPi 3, Vosk transcript verification alone takes ~5-6 seconds.

**Fix:** If existing clips are already saved in memory, play the response **immediately** (right after WLED wake, before capture even starts), then do capture/analysis/saving afterward:

```
New flow (with existing clips):
  1. snapshot pre-buffer
  2. wake LEDs
  3. RESPOND NOW with existing clips  ← user hears response immediately
  4. capture post-detection audio
  5. analyze + trim + save            ← happens while user already heard response

New flow (first trigger, no existing clips):
  1. snapshot pre-buffer
  2. wake LEDs
  3. capture
  4. analyze
  5. respond based on analysis        ← falls back to original behavior
  6. save
```

Added `_respond_with_existing()` method which plays a single or layered response from memory clips without needing the current clip or quality report.

**Files changed:**
- `main.py` — added `_respond_with_existing()`, restructured `_handle_trigger()` with `responded_early` flag

---

### Current Pipeline Flow (After Fixes)

```
Vosk detects "i love you"
  │
  ├─ 1. Snapshot pre-buffer (instant)
  ├─ 2. Wake LEDs
  │
  ├─ 3. Existing clips in memory?
  │     YES → Play response immediately (single or layered)
  │     NO  → Continue to analysis first
  │
  ├─ 4. Capture post-detection (2s recording)
  ├─ 5. Vosk transcript verification (~5-6s on RPi 3)
  ├─ 6. Trim to phrase boundaries
  │
  ├─ 7. No early response? → Respond now based on analysis
  │
  ├─ 8. Save to memory if quality passes (score >= 0.4, phrase found)
  └─ 9. Fade LEDs back to idle
```

---

#### 4. Layered Audio Too Cramped — No Spatial Separation

**Symptom:** When multiple "i love you" clips play together, the layered response sounds muddy, cramped, and unnatural. All voices occupy the same sonic space.

**Root cause:** The original `layer_clips()` only used three tools — fixed time delay (0.15s), volume decay (0.75), and per-layer reverb. Every voice sat at center panning, same pitch, same frequency range. Phase cancellation between similar waveforms made it worse.

**Fix:** Rewrote `layer_clips()` with professional vocal layering techniques, all chosen for low CPU cost on RPi 3:

| Technique | What it does | CPU cost |
|-----------|-------------|----------|
| **Golden-ratio delay offsets** | Delays scale by phi (1.618) per layer: 100ms, 162ms, 262ms, 424ms... Non-repeating, natural spacing. | Free |
| **Stereo panning** | Newest = center, older alternate L/R at increasing width (30%, 60%, 90%). Constant-power sin/cos law. | Near-free |
| **Pitch detuning ±7 cents** | Alternating sharp/flat on older layers. Prevents phase cancellation, adds chorus thickness. | Cheap (resampling) |
| **Low-pass rolloff** | Older layers get darker (8kHz → 3.5kHz cutoff). Simulates distance — far voices are naturally muffled. | Cheap (2nd-order Butterworth via scipy) |
| **Softer volume decay (0.85)** | Was 0.75. The panning/EQ now create perceived distance, so less volume drop keeps the chorus fuller. | Free |
| **Transient softening** | 20ms fade-in on older layers' attacks. Reduces consonant smearing from multiple "I"s hitting together. | Free |
| **Shared reverb** | One reverb on the final mix instead of per-layer. Places all voices in the same "room", saves CPU. | Saves CPU |

**New helper functions added to `modules/effects.py`:**
- `mono_to_stereo()` — converts (N,) to (N, 2)
- `stereo_pan(audio, pan)` — constant-power L/R panning
- `pitch_shift(audio, cents)` — cheap resampling-based pitch shift
- `lowpass_ema_fast(audio, sr, cutoff)` — Butterworth via scipy, EMA fallback

**Playback updated** (`modules/playback.py`) to pass stereo channel count to `sounddevice`.

**New config values in `playback:` section of `config.yaml`:**
- `detune_cents: 7.0`
- `lp_oldest_hz: 3500`
- `lp_newest_hz: 8000`
- `transient_soften_ms: 20`
- `layer_delay_sec: 0.10` (was 0.15, now base for golden-ratio scaling)
- `layer_volume_decay: 0.85` (was 0.75)
- `reverb_delay_ms: 60` (was 40)

---

### Key Config Values (config.yaml)

| Setting | Value | Notes |
|---------|-------|-------|
| `mode` | `hybrid` | Prefer live clips, supplement with fallbacks |
| `wake_detector` | `vosk` | Offline speech recognition |
| `pre_detection_buffer_sec` | `4.0` | Rolling buffer before detection |
| `post_detection_sec` | `2.0` | Recording after detection |
| `wled.timeout_sec` | `2` | HTTP timeout for WLED calls |
| `quality.save_threshold` | `0.4` | Minimum score to persist clip |
| `thresholds.intimate_rms` | `0.005` | Below = whisper |
| `thresholds.strong_rms` | `0.02` | Above = projected voice |
| `thresholds.noise_floor_rms` | `0.0005` | Below = silence/unusable |
| `thresholds.max_duration_sec` | `5.0` | Clips longer than this get penalized |
| `playback.layer_delay_sec` | `0.10` | Base delay, scaled by golden ratio per layer |
| `playback.layer_volume_decay` | `0.85` | Softer decay — panning/EQ add distance |
| `playback.detune_cents` | `7.0` | Pitch shift per layer (alternating +/-) |
| `playback.lp_oldest_hz` | `3500` | Low-pass cutoff on oldest layer |
| `playback.lp_newest_hz` | `8000` | Low-pass cutoff on newest layer |
| `playback.transient_soften_ms` | `20` | Attack fade on older layers |
| `playback.reverb_delay_ms` | `60` | Shared reverb pre-delay |

### Current State

- 3 clips saved in `state.json` (all 6.05s, scores 0.9) — these were captured after the pre-buffer fix
- WLED not connected during testing (`WLED unreachable at http://wled.local`)
- PulseAudio ALSA driver warnings in syslog (`snd_usb_audio` wakeup bug) — cosmetic, not affecting capture

### Known Issues / Future Work

- **Undervoltage:** Need better power supply before deployment
- **Clip duration always 6.05s:** Pre-buffer (4s) + post (2s) = 6s every time. The `max_duration_sec: 5.0` threshold penalizes every clip with `too_long`. Either increase `max_duration_sec` to 6.5, or reduce `pre_detection_buffer_sec` to 3.0
- **PulseAudio ALSA warnings:** `snd_usb_audio` driver issue logged by PulseAudio — not causing failures but worth monitoring
- **WLED timeout still blocks:** Even with the pre-buffer fix, `wake_neutral()` blocks for up to `timeout_sec` when WLED is unreachable. Could move WLED calls to a fire-and-forget thread
- **Vosk verification speed:** ~5-6 seconds on RPi 3. Will be much faster on Jetson. Could also skip verification when confidence from the wake detector is high
- **First trigger has no early response:** On first-ever trigger (no clips in memory), the user still waits for the full pipeline. Could add a "first time" fallback sound
