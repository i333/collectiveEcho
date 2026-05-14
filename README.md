# I Love You Mirror

An offline interactive mirror installation that listens for "I love you," captures the voice, and plays back layered voice memories through WLED-lit ambiance.

## Requirements

- Raspberry Pi 3 (or NVIDIA Jetson Orin Nano)
- USB microphone
- Speaker/exciter connected via amp to audio output
- WLED-controlled LEDs on the local network
- Python 3.9+

## Install

```bash
# Install system audio libraries (one-time)
sudo apt-get install -y libportaudio2 libsndfile1

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Set Microphone

List available audio devices:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Set `audio.input_device` in `config.yaml` to the device index of your USB mic, or leave as `null` for system default.

## Test Recording

Quick recording test:

```bash
python3 -c "
import sounddevice as sd
import soundfile as sf
import numpy as np
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='int16')
print('Recording 3 seconds...')
sd.wait()
sf.write('test_recording.wav', audio, 16000)
print('Saved test_recording.wav')
"
```

Play it back:

```bash
python3 -c "
import sounddevice as sd
import soundfile as sf
data, sr = sf.read('test_recording.wav')
sd.play(data, sr)
sd.wait()
"
```

## Test WLED

```bash
# Check WLED is reachable
curl http://wled.local/json/state

# Trigger a preset
curl -X POST http://wled.local/json/state -d '{"ps": 2}'
```

Set `wled.host` in `config.yaml` if your WLED is at a different address.

## Run the App

### Test / dry-run mode (no microphone needed for wake detection)

```bash
python3 main.py --test
```

Press **Enter** to simulate an "I love you" detection. Each press triggers the full pipeline: WLED wake, recording, analysis, playback, fade.

### Live mode

```bash
python3 main.py
```

Set `wake_detector: porcupine` in config.yaml for real wake-word detection (requires Porcupine setup — see below).

## Calibrate Thresholds

```bash
python3 calibrate.py
```

This records room noise, a soft phrase, and a loud phrase, then suggests values for `noise_floor_rms`, `intimate_rms`, and `strong_rms`.

## Add Fallback Recordings

Place `.wav` files in `audio/fallback/i_love_you/`:

```
audio/fallback/i_love_you/
  ily_gentle_01.wav
  ily_gentle_02.wav
  ily_strong_01.wav
```

These are used when live recordings are missing or poor quality (depending on `mode` setting).

## Change Thresholds

Edit `config.yaml` under `thresholds:`:

```yaml
thresholds:
  intimate_rms: 0.08      # Below this = whisper/intimate
  strong_rms: 0.25         # Above this = projected voice
  noise_floor_rms: 0.01    # Below this = silence/unusable
  max_clipping_ratio: 0.05 # Above this = distorted
```

Or run `calibrate.py` to auto-detect values for your space.

## Operating Modes

Set `mode` in `config.yaml`:

| Mode       | Behavior |
|------------|----------|
| `live`     | Only use live recordings |
| `fallback` | Only use prerecorded fallback clips |
| `hybrid`   | Prefer live, fall back to prerecorded if needed |

## Porcupine Wake-Word Setup

1. Sign up at [console.picovoice.ai](https://console.picovoice.ai)
2. Train a custom "I love you" keyword and download the `.ppn` file
3. Get your access key
4. Update `config.yaml`:

```yaml
wake_detector: porcupine
porcupine:
  access_key: "YOUR_KEY"
  keyword_path: "path/to/i-love-you.ppn"
  sensitivity: 0.5
```

5. Install extra deps: `pip install pvporcupine pyaudio`

## Project Structure

```
main.py                      # Orchestrator
config.yaml                  # All settings
calibrate.py                 # Threshold calibration
state.json                   # Persistent clip memory
modules/
  audio_input.py             # Rolling mic buffer
  wake_detector.py           # Swappable wake-word backends
  recorder.py                # Clip capture & save
  quality.py                 # Audio analysis & scoring
  memory_store.py            # Ring buffer of accepted clips
  playback.py                # Audio output
  effects.py                 # Layering, reverb, fades
  wled.py                    # WLED HTTP control
audio/
  live/i_love_you/           # Accepted live recordings
  fallback/i_love_you/       # Curated fallback clips
logs/
  mirror.log                 # Event log
```

## Jetson Orin Nano Migration

The code is structured to run on Jetson with minimal changes:
- Replace `sounddevice` with ALSA/PulseAudio bindings if needed
- GPU-accelerated effects can be added in `modules/effects.py`
- All config is externalized — no code changes needed for hardware differences

## V2 (jetson-v2-prep branch) — Quick Reference

The V2 build adds emotion-aware response selection, time-of-day mood drift,
phonetic phrase matching, adaptive end-of-utterance capture, diversity-based
memory eviction, WLED UDP DRGB direct color, and pluggable ASR + emotion
backends. See `DEVLOG.md` "Session: 2026-05-14" for the full change list.

### Boot / first run on Jetson

```bash
sudo apt install -y libportaudio2 libsndfile1     # PortAudio for sounddevice
python3 -m pip install --user -r requirements.txt # if you don't have a venv yet
python3 main.py --test                            # press Enter to fake a detection
```

### Switching to GPU ML backends (after the bigger SD card is in)

```bash
pip install --user faster-whisper                            # ~200MB + 250MB model
pip install --user torch torchaudio transformers             # ~3GB + 1.3GB SER model
```

then in `config.yaml`:

```yaml
asr:
  backend: whisper        # collapses Vosk's 5-6s verification to ~200-300ms

emotion:
  backend: wav2vec_ser    # real SER instead of prosody proxies
  device: cuda
```

No other code changes. Restart `main.py`.

### Turning on WLED UDP DRGB

In `config.yaml`:

```yaml
wled:
  realtime:
    enabled: true
    host: "192.168.0.11"    # WLED controller IP (no scheme)
    port: 21324             # WLED listens on 21324 by default
    num_leds: 60            # set to your actual strip length
```

Or pass `--udp-wled` to `main.py` to force-enable. The HTTP preset path
(`wled.idle()`, `wled.layered()`, etc.) keeps working — UDP is additive,
used for per-LED color driven by `(valence, arousal)` and the audio envelope.

### Emotion-aware response selection

`config.yaml → emotion.selection_mode`:

| Mode       | Behavior |
|------------|----------|
| `match`    | Layered echo picks memories with the *closest* emotion to what the speaker just said. |
| `contrast` | Picks the *farthest* emotion — counter-emotional response. |
| `diverse`  | Spreads picks across the emotional space — most varied chorus. |

### Time-of-day mood

`modules/mood.py` modulates layer pacing, intimacy (which scales layer count
and reverb), and WLED palette bias across the day. Persisted across restarts
in `mood_state.json` (lifetime trigger count). No config knobs required —
just runs.
