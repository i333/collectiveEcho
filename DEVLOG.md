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

---

## Session: 2026-05-14 (V2 prep on Jetson Orin Nano Super)

### Hardware
- NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- JetPack R36.4, CUDA 12.6, Python 3.10.12
- 27GB microSD, **5.5GB free** at start of session → 4.9GB free at end (~600MB used)
- 128GB A2 V30 endurance card on order (arrives next day) — will migrate before installing heavy ML stack

### Goal
V1 had a 5–6s Vosk transcript stall and only RMS-driven "intimacy" classification. For V2 the user wants:
1. Better phrase detection (low-latency, noise/accent-robust)
2. Realtime emotion detection driving lighting + response selection
3. WLED UDP DRGB so colour can be per-LED and emotion-driven, not preset-only

Approach: lay all the architecture / non-ML work now on the existing card, leave clean pluggable abstractions so the GPU-bound ASR + SER models drop in tomorrow with a one-line config change once the larger card is in.

### What changed this session (jetson-v2-prep branch)

#### New modules

- `modules/wled_udp.py` — `WLEDRealtime`. Direct UDP DRGB/DNRGB driver to a WLED controller. Per-LED RGB at 60+ Hz. Includes `emotion_field(valence, arousal, brightness)` which maps continuous emotion to a colored, slightly-noisy LED field (hue from valence, saturation/jitter from arousal, brightness from audio envelope).
- `modules/emotion/` — pluggable emotion backends.
  - `prosody.py` (default, CPU-only, no ML deps): autocorrelation pitch + framewise energy + ZCR + jitter → `(valence, arousal, dominance)` in continuous space + a Russell-circumplex categorical label.
  - `wav2vec_ser.py`: stub for `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`. Activates when `torch + transformers` are installed.
  - `EmotionReport`: continuous V/A/D + categorical label + raw prosody features.
- `modules/asr/` — pluggable ASR backends.
  - `vosk_backend.py`: wraps the existing transcript verification with `TranscriptResult` (text, word timings, confidence).
  - `whisper_backend.py`: stub for `faster-whisper`. Activates when `faster-whisper` is installed.
- `modules/phrase_match.py` — `PhraseMatcher`. Replaces hard-coded substring variants with a small built-in Metaphone + sliding-window Levenshtein over phonetic codes. Catches "i luv u", "ay love yew", "ily" as a word. 11/11 test cases pass.
- `modules/mood.py` — `MoodKeeper`. Persists lifetime trigger count and returns a `MoodState` modulated by time-of-day (morning/afternoon/evening/night → warmth, pace, intimacy, palette bias). Subtly changes the artwork across the day.
- `modules/pipeline.py` — `TriggerPipeline`. Replaces the 100-line god method `_handle_trigger` with a clean state machine: Snapshot → Wake → EarlyRespond → Capture → Analyze → ReconcilePhraseType → Trim/Resave → Log → RespondLate → Save → FadeOut.

#### Modified modules

- `modules/quality.py`:
  - Transcript verification now delegates to an injected `BaseTranscriber` (no more inline Vosk imports).
  - Emotion analysis runs via injected `BaseEmotionAnalyzer`; result lives on `QualityReport.emotion`.
  - Phrase match uses `PhraseMatcher`, supports "ya"/"yew" variants and TOO detection via word sequence.
- `modules/wake_detector.py`:
  - Vosk detector uses `PhraseMatcher` instead of hard-coded substrings.
  - Partial-result triggers now require a configurable confidence floor (`vosk.partial_confidence_min: 0.75`).
- `modules/audio_input.py`:
  - **Adaptive post-capture**: keep recording until N ms of silence (`audio.adaptive_capture.eos_silence_*`) or hit `max_post_sec`. Fixes the "every clip is 6.05s" problem.
  - Lazy `sounddevice` import — boot doesn't crash on systems without PortAudio.
- `modules/memory_store.py`:
  - `ClipEntry` carries `valence/arousal/dominance/emotion_label/last_played_ts`. Backward compatible with old `state.json` (missing fields fill defaults on load).
  - **Diversity-based eviction** when over capacity: combines low score, similarity in V/A/D space, and age. Preserves rare emotional outliers, discards redundant clips.
  - `pick_by_emotion(target_v, target_a, n, mode)` — selection modes: `match` / `contrast` / `diverse`.
- `modules/effects.py`:
  - `lowpass_ema` now uses `scipy.signal.lfilter` — ~50× faster than the per-sample loop.
  - `noise_gate` smoother tightened, no behavior change.
- `modules/playback.py`: lazy `sounddevice` import.
- `main.py`: trimmed to a wiring shell. Constructs collaborators, builds `TriggerPipeline`, owns the loop. Logic lives in `pipeline.py`. Added `--udp-wled` flag.

#### Config additions (`config.yaml`)

```yaml
asr: {backend: vosk}                 # flip to "whisper" once installed
whisper:
  model_size: distil-small.en
  device: cuda
  compute_type: int8_float16

emotion:
  backend: prosody                   # flip to "wav2vec_ser" once installed
  selection_mode: match              # match | contrast | diverse
  pitch_low_hz: 110.0
  pitch_high_hz: 280.0

audio:
  adaptive_capture:
    enabled: true
    min_post_sec: 1.0
    max_post_sec: 5.0
    eos_silence_rms: 0.005
    eos_silence_sec: 0.7

memory:
  eviction_policy: diversity         # or "fifo" for V1 behavior
  evict_weight_score: 0.45
  evict_weight_similarity: 0.35
  evict_weight_age: 0.20

wled:
  realtime:
    enabled: false                   # flip true (or pass --udp-wled)
    host: "192.168.0.11"
    port: 21324
    num_leds: 60
```

#### Tests run

- All 8 existing clips analyzed end-to-end: Vosk transcribes correctly, prosody analyzer reads sensible V/A vectors for each (mostly low-positive / mid-arousal — the recordings sound flat/serious in test conditions).
- `pick_by_emotion(0.3, 0.5)` returns nearest 3 in match mode, farthest 3 in contrast mode. Vectors confirmed.
- `PhraseMatcher` 11/11 test cases pass including phonetic ("i luv u", "ay love yew") and TOO variants.
- Full `TriggerPipeline.run()` exercised with mocked `audio_input`, real ASR+emotion+memory+recorder — clip saved with emotion fields, lighting/wled calls made, fadeout runs.
- `python3 main.py --test` boots cleanly: loads 8 clips with new schema, Vosk warms up, pipeline initialized, simulated trigger fires LAYERED early response.

### What's queued for after the bigger card lands

These are 1-line config flips once the deps are installed:

```bash
pip install --user faster-whisper           # ~200MB + 250MB model
pip install --user torch torchaudio transformers   # ~3GB + 1.3GB SER model
```

then in `config.yaml`:
```yaml
asr:    {backend: whisper}
emotion: {backend: wav2vec_ser, device: cuda}
```

That collapses the 5–6s Vosk verification to ~200–300ms Whisper inference on the Orin GPU and gives true SER-class emotion readings instead of prosody proxies.

### Round 2 (same session) — feedback fixes

The user reviewed the V2 build and flagged five concerns. Each addressed:

#### 1. Audit: are single + layered responses *actually* random?

Single mode: yes, was already `random.choice`.
**Layered mode: NO, it was deterministic.** Both the early-respond path (took first N from chronological recent list) and the emotion-aware late-respond path (took the N nearest emotion neighbours, no shuffle) produced the same chorus every time for the same conditions. **Fixed** in both paths:

  - **Early respond:** `random.sample(paths, k=layers_cap)` from the full candidate pool.
  - **Late respond (emotion-aware):** build an emotion-FILTERED pool of size `emotion.pool_size` (default 16), then `random.sample(pool, k=n-1)` from it. Same emotion target still pulls thematically similar clips, but the specific N varies every trigger.
  - Also added a "recently played" filter — clips played in the last 30s are skipped on the next trigger so back-to-back triggers don't immediately repeat.
  - Verified empirically: 5 triggers with the same target emotion picked 5 different sets, with all 8 clips appearing across the runs.

#### 2. Config-driven keyword buckets ("I love you Sandy")

Was hardcoded ily / ily_too. **Rebuilt as config-driven buckets** in `modules/phrase_match.py`:

```yaml
phrases:
  buckets:
    - id: ily_sandy
      priority: 30
      patterns: ["i love you sandy", "i love u sandy", "i love ya sandy"]
      anchor_extra: [sandy]
      fallback_dir: audio/fallback/i_love_you_sandy
      save_user_recordings: false        # don't pollute Sandy's pool
      response_files_only: true          # only Sandy's voice answers Sandy

    - id: ily_too
      priority: 20
      patterns: ["i love you too", ...]

    - id: ily
      priority: 10
      patterns: ["i love you", "i loved you", ...]
      slang: [ily, iloveyou, iluvu]
```

Buckets are checked in priority order — longer/more-specific phrases beat shorter ones, so "I love you Sandy" routes to Sandy even when "I love you too" is also in the transcript.

New routing rules:
  - **Word-boundary exact match** (most reliable signal).
  - **Name-anchor match**: if a bucket has `anchor_extra` (e.g. "sandy") AND both the universal love-anchor AND the bucket's anchor word are present, route to that bucket regardless of word order. Catches "sandy i love you", "i love sandy", "i love you sandy boy".
  - **Phonetic substring** (Metaphone) for accent / partial captures.
  - **Levenshtein DISABLED by default** (was firing too loosely — see §4).

The matcher is built ONCE in `main.py` and shared with the wake detector, the quality verifier, and the pipeline so they all agree on bucket routing.

Per-bucket policies wired through `MemoryStore.configure_buckets(...)`:
  - `live_dir` / `fallback_dir` → `bucket_live_dir(id)`, `bucket_fallback_dir(id)`
  - `save_user_recordings: false` → `bucket_allows_save(id)` returns False, pipeline skips save
  - `response_files_only: true` → pipeline plays a single random fallback from that bucket, never layers

To add a new bucket: drop entries into `config.yaml → phrases.buckets`. No code changes.

Verified empirically with 24-case test (Sandy variants + base ily + ily_too + false-positive guards): 24/24 pass.

#### 3. Anti-overfitting on wake/verify phrases

Concrete problem: Metaphone collapses `love / leave / live / lava / loaf / leaf` all to `LF`. Without a guard, "I leave you" would phonetically match "I love you". Also substring "lov" appears inside `glove / clover / olive` — substring-only checks let those slip through.

**Fixes layered into PhraseMatcher:**

  - **Universal love-anchor regex** at top of `find()`:
    `\b(lov|luv)[a-z]*\b | \bily\b | \biluvu\b | \biloveyou\b`
    Must match at least once before any phonetic/Levenshtein pass runs. The anchor requires the love-word to START a word, not just appear as a substring — so `glove`, `clover`, `olive` are blocked.
  - **Word-boundary exact match** (`\b...\b` regex) for literal patterns — `love you` matches in `i love you` but not in `glove yours`.
  - **Levenshtein backend disabled by default** (`max_phonetic_edits=0`, `min_confidence_for_partial=0.85`). The default ily bucket caught every realistic mangled phrase via exact + phonetic-substring; Levenshtein was producing false positives like `i love sandy → ily` (1-edit window match on `I LF Y` vs `I LF S`).

15/15 false-positive cases now blocked: `i leave you`, `i live you`, `i lava you`, `glove you`, `gloves you`, `clover you`, `olive you`, `the loaf you bought`, `i leaf you`, `happily ever after`, `ilya is my friend`, `i loved someone else`, `i moved you`, `i lift you`, `i love wallaby`.

#### 4. Better quality assessment

Added to `QualityReport`:
  - **`snr_db`** — signal-to-noise ratio, computed from frame-energy distribution (loudest 30% / quietest 10%). Single most important missing metric.
  - **`spectral_centroid_hz`** — brightness of the clip (FFT-weighted mean frequency).
  - **`spectral_tilt_db_per_khz`** — overall spectral balance via linear regression on log-magnitude. Negative tilt = muffled / off-axis mic.
  - **`phrase_only_rms`** — RMS measured on the phrase region only, so trailing dead air can't drag down a good phrase's score.

Score function updated:
  - SNR < 6dB → score −0.35 + `low_snr_*` reason
  - SNR 6–12dB → score −0.10 + `mediocre_snr_*` reason
  - SNR > 18dB → score +0.10 (reward clean recordings)
  - Tilt < −15 dB/kHz → score −0.10 + `muffled` reason

Verified on the 8 existing V1 clips: all read SNR ~35dB (clean lav recordings) with tilt ~−6 dB/kHz (warm but not muffled). Scores stay at 1.00 — quality additions are additive, don't break existing-clip handling.

#### 5. Clip enhancement before serve-back

New `modules/clip_enhancer.py` — `ClipEnhancer.enhance(audio, sr) → (cleaned_audio, EnhancementMetrics)`. CPU-only chain applied during `_stage_trim_and_resave`:

  1. **DC offset removal**
  2. **High-pass at 60Hz** (Butterworth, removes rumble + AC-floor)
  3. **Hum notch at 60Hz** (configurable; set to 50 for EU)
  4. **Spectral-subtraction denoise** — estimates noise from the quietest 10% of frames, subtracts the magnitude spectrum, floors at 0.1× to avoid musical-noise artifacts
  5. **Loudness normalization to target RMS** (default 0.10, ≈ −20 dBFS) with `max_gain_db` cap — clips that were captured quietly come up to the same perceived level as louder ones, so the layered chorus blends evenly
  6. **Soft-knee compressor** (threshold 0.5, ratio 2.5)
  7. **Light de-essing** at 6.5kHz when sibilance energy spikes
  8. **Peak safety** at 0.99 to prevent overflow

Saved enhanced versions replace the raw recordings in place. The original raw audio is the trimmed-to-phrase region (so we keep the artistic content, dropping silence) — enhancement runs on that.

Measured on one V1 clip: SNR 35.9dB → 43.4dB, RMS 0.080 → 0.100, applied: `dc_removal, hp_60Hz, notch_60Hz, denoise×1.0, loudness +2.1dB, comp 0.50:2.5, de_ess`.

Config block:

```yaml
enhancement:
  enabled: true
  highpass_hz: 60
  hum_hz: 60                # 50 in EU
  hum_q: 30
  denoise_strength: 1.0
  target_rms: 0.10
  max_gain_db: 18.0
  comp_threshold: 0.5
  comp_ratio: 2.5
  deess_freq_hz: 6500
  deess_threshold: 0.4
```

DeepFilterNet 2 as a deeper upgrade is queued for after the GPU stack lands.

### Open items / next-day work

- **Install `libportaudio2` + `libsndfile1`** on the Jetson (`sudo apt install libportaudio2 libsndfile1`). Needed before live mic works — boot succeeds without it but `audio_input.start()` raises.
- **Calibrate prosody thresholds for the actual mic + room.** `emotion.pitch_low_hz / pitch_high_hz / energy_*` may need tuning; run `calibrate.py` and watch the prosody log line during real triggers.
- **Test WLED UDP path** against the real WLED controller. Set `wled.realtime.enabled: true` + correct `host` + `num_leds`.
- **Tune `eos_silence_rms` to the room noise floor.** Default 0.005 is for a quiet space; gallery may need 0.01 or higher.
- **Migrate to 128GB card**, restore from card image, then install the two `pip` blocks above.
- **Speaker embeddings (ECAPA-TDNN)** + **diarization** — deferred from earlier scoping. Can layer on after the V2 base is validated.
- **WLED palette presets per emotion period** — design 4–6 palettes (intimate/warm/cool/excited/melancholic) and have `WLEDRealtime` cross-fade between them as `(valence, arousal)` drifts during playback.

