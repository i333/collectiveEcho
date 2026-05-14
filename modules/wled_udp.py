"""
WLED realtime UDP driver — direct per-LED color control.

Bypasses HTTP entirely. WLED firmware natively listens on:
  - DRGB (port 21324): 3 bytes/LED, up to ~490 LEDs per packet
  - DNRGB (port 21324, protocol byte 4): multi-packet, unlimited length
  - WARLS (port 21324, protocol byte 1): index-addressed sparse updates
  - DDP (port 4048): chunked, unlimited

We use DNRGB by default (capacity-agnostic) with a DRGB fast path for small
strips. All packets carry a "timeout in seconds before WLED returns to its
normal effect" — we set 2s, longer than our update interval, so the strip
holds the last frame if the sender stalls.

This driver is *additive* to the existing wled.py HTTP controller. Use HTTP
for presets and config; use UDP for live, expressive, per-frame color.
"""

import logging
import math
import socket
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# WLED realtime protocol IDs (first byte of the UDP payload)
WLED_DRGB = 0x02
WLED_DNRGB = 0x04
WLED_WARLS = 0x01

DEFAULT_PORT = 21324
DEFAULT_TIMEOUT_S = 2  # seconds the strip holds the last frame if we stall


class WLEDRealtime:
    """
    Send per-LED RGB frames to a WLED controller over UDP.

    Build a frame as a (num_leds, 3) uint8 ndarray and call .send(frame).
    Or use the higher-level helpers: solid(), gradient(), envelope_pulse().

    Thread-safe: socket sends are serialized via an internal lock.
    """

    def __init__(self, host: str, num_leds: int, port: int = DEFAULT_PORT,
                 hold_timeout_s: int = DEFAULT_TIMEOUT_S):
        # Strip any scheme/path the user might have copied from the HTTP config
        host = host.replace("http://", "").replace("https://", "")
        host = host.split("/")[0].split(":")[0]
        self.host = host
        self.port = port
        self.num_leds = int(num_leds)
        self.hold_timeout_s = max(1, min(255, int(hold_timeout_s)))

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self._lock = threading.Lock()

        # Last frame sent (for diffing / smoothing)
        self._last_frame = np.zeros((self.num_leds, 3), dtype=np.uint8)

        # Reactive loop state
        self._react_stop = threading.Event()
        self._react_thread: threading.Thread | None = None

        logger.info("WLEDRealtime → %s:%d, %d LEDs", self.host, self.port, self.num_leds)

    # --- low-level frame send --------------------------------------------

    def send(self, frame: np.ndarray):
        """Send a (num_leds, 3) uint8 frame.

        Chooses DRGB for small strips (<= 490 LEDs) and DNRGB for larger.
        Smaller payloads are slightly faster; DNRGB supports any length.
        """
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 2 or frame.shape[1] != 3:
            raise ValueError(f"frame must be (N, 3) uint8, got {frame.shape}")

        n = min(frame.shape[0], self.num_leds)
        if self.num_leds <= 490:
            # DRGB: [protocol, timeout, R0, G0, B0, R1, G1, B1, ...]
            payload = bytes([WLED_DRGB, self.hold_timeout_s]) + frame[:n].tobytes()
            self._send_bytes(payload)
        else:
            # DNRGB: [protocol, timeout, start_high, start_low, RGB...]
            # Each packet covers up to 489 LEDs; multiple packets cover all.
            chunk = 489
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                header = bytes([
                    WLED_DNRGB,
                    self.hold_timeout_s,
                    (start >> 8) & 0xFF,
                    start & 0xFF,
                ])
                payload = header + frame[start:end].tobytes()
                self._send_bytes(payload)

        self._last_frame = frame[:n].copy() if n == frame.shape[0] else frame[:n].copy()

    def _send_bytes(self, payload: bytes):
        with self._lock:
            try:
                self._sock.sendto(payload, (self.host, self.port))
            except OSError as e:
                # WLED might be unreachable. Don't spam logs — just drop the frame.
                logger.debug("UDP send failed: %s", e)

    # --- high-level frame builders ---------------------------------------

    def solid(self, r: int, g: int, b: int):
        """Fill the whole strip with one color."""
        frame = np.tile(
            np.array([r, g, b], dtype=np.uint8), (self.num_leds, 1)
        )
        self.send(frame)

    def off(self):
        self.solid(0, 0, 0)

    def gradient(self, c1: tuple[int, int, int], c2: tuple[int, int, int]):
        """Linear gradient from c1 (LED 0) to c2 (LED N-1)."""
        t = np.linspace(0.0, 1.0, self.num_leds, dtype=np.float32)[:, None]
        a = np.array(c1, dtype=np.float32)[None, :]
        b = np.array(c2, dtype=np.float32)[None, :]
        frame = (a * (1.0 - t) + b * t).astype(np.uint8)
        self.send(frame)

    def emotion_field(self, valence: float, arousal: float,
                       brightness: float = 1.0):
        """
        Map a (valence, arousal) emotion point to a colored, modulated field.

        valence: -1.0 (negative/sad) → +1.0 (positive/happy)
                 maps hue: blue/violet → amber/warm
        arousal: 0.0 (calm) → 1.0 (excited)
                 maps saturation + spatial variation
        brightness: 0.0 → 1.0  global brightness multiplier

        The field is *not* uniform — even at the same emotion point, LEDs
        vary slightly along the strip so it doesn't look like a flashlight.
        """
        valence = max(-1.0, min(1.0, float(valence)))
        arousal = max(0.0, min(1.0, float(arousal)))
        brightness = max(0.0, min(1.0, float(brightness)))

        # Hue: blue (240°) at valence=-1, amber (35°) at valence=+1
        hue_neg = 240.0 / 360.0
        hue_pos = 35.0 / 360.0
        base_hue = hue_neg + (hue_pos - hue_neg) * ((valence + 1.0) * 0.5)

        # Per-LED hue jitter scales with arousal (calm = unified, excited = scattered)
        n = self.num_leds
        hue_jitter = (np.sin(np.linspace(0, 2 * math.pi, n)) * 0.04 * arousal)
        hue = (base_hue + hue_jitter) % 1.0

        # Saturation rises with both valence extremity and arousal
        sat = 0.55 + 0.45 * (abs(valence) * 0.5 + arousal * 0.5)
        sat = np.full(n, min(1.0, sat), dtype=np.float32)

        # Per-LED brightness with a slow ripple driven by arousal
        ripple = 0.85 + 0.15 * np.sin(np.linspace(0, 3 * math.pi, n)) * arousal
        val = brightness * ripple

        frame = _hsv_to_rgb_array(hue, sat, val)
        self.send(frame)

    # --- audio-reactive realtime loop ------------------------------------

    def react_to_audio(self, audio: np.ndarray, sr: int,
                       valence: float = 0.0, arousal: float = 0.5,
                       update_hz: float = 60.0,
                       min_brightness: float = 0.08,
                       max_brightness: float = 1.0):
        """Start a thread that emits frames at update_hz, brightness following
        the audio envelope, color set by (valence, arousal).

        Call stop_reacting() when playback ends.
        """
        self.stop_reacting()
        self._react_stop.clear()

        env = _compute_envelope(audio, sr, update_hz)
        self._react_thread = threading.Thread(
            target=self._react_loop,
            args=(env, update_hz, valence, arousal, min_brightness, max_brightness),
            daemon=True,
        )
        self._react_thread.start()

    def _react_loop(self, envelope: np.ndarray, update_hz: float,
                    valence: float, arousal: float,
                    min_b: float, max_b: float):
        interval = 1.0 / update_hz
        next_t = time.monotonic()
        for level in envelope:
            if self._react_stop.is_set():
                break
            brightness = min_b + (max_b - min_b) * float(level)
            self.emotion_field(valence, arousal, brightness)
            next_t += interval
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Falling behind — re-sync rather than spiral
                next_t = time.monotonic()

    def stop_reacting(self):
        self._react_stop.set()
        if self._react_thread is not None and self._react_thread.is_alive():
            self._react_thread.join(timeout=0.5)
        self._react_thread = None

    # --- expressive one-shots --------------------------------------------

    def flash(self, color: tuple[int, int, int] = (255, 255, 255),
              hold_ms: int = 80, fade_ms: int = 200):
        """Bright flash → fade. Used to acknowledge detection."""
        def _do():
            self.solid(*color)
            time.sleep(hold_ms / 1000.0)
            steps = 8
            for i in range(steps, -1, -1):
                t = i / steps
                self.solid(int(color[0] * t), int(color[1] * t), int(color[2] * t))
                time.sleep((fade_ms / 1000.0) / steps)
            self.off()
        threading.Thread(target=_do, daemon=True).start()

    def close(self):
        self.stop_reacting()
        try:
            self._sock.close()
        except OSError:
            pass


# --- helpers -------------------------------------------------------------

def _hsv_to_rgb_array(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV → RGB. All inputs in [0, 1]. Returns uint8 (N, 3)."""
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i = i % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _compute_envelope(audio: np.ndarray, sr: int, update_hz: float) -> np.ndarray:
    """RMS envelope at update_hz rate, normalized to [0, 1]."""
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    if mono.dtype == np.int16:
        mono = mono.astype(np.float32) / 32768.0
    else:
        mono = mono.astype(np.float32)

    window = max(1, int(sr / update_hz))
    n_frames = len(mono) // window
    if n_frames == 0:
        return np.array([0.5], dtype=np.float32)

    frames = mono[:n_frames * window].reshape(n_frames, window)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(np.max(rms))
    if peak > 0:
        return (rms / peak).astype(np.float32)
    return rms.astype(np.float32)
