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

        # Sleep-indicator loop state — deliberately separate from the
        # reactive loop so stop_reacting() (fired by the signal-ready
        # transition after every command) can't kill the asleep breathing.
        self._sleep_stop = threading.Event()
        self._sleep_thread: threading.Thread | None = None

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
                 hue path travels the WARM side of the wheel
                 (blue 240° → purple → red 360°/0° → amber 35°),
                 NOT through green — neutral V=0 lands at magenta-ish,
                 not green.
        arousal: 0.0 (calm) → 1.0 (excited)
                 governs saturation (calm=desaturated/near-white, excited=vivid),
                 spatial jitter amplitude, temporal flow speed, and sparkle
                 density. The visible difference between calm and excited at
                 the same hue is dramatic.
        brightness: 0.0 → 1.0  global brightness multiplier

        Time-driven: each call advances a phase so the spatial wave drifts
        along the strip — slow drift when calm, faster when excited. Holding
        the same (V, A) for several seconds doesn't show a static pattern.
        """
        valence = max(-1.0, min(1.0, float(valence)))
        arousal = max(0.0, min(1.0, float(arousal)))
        brightness = max(0.0, min(1.0, float(brightness)))

        n = self.num_leds

        # --- Hue: warm-side path -----------------------------------------
        # Blue (240°) → purple (300°) → red (360°/0°) → amber (35°). Going
        # through 1.0 (modulo) keeps interpolation linear without crossing
        # green. At valence=0 we land near 317° (magenta/pink), which feels
        # neutral-but-warm rather than the old "neutral=green" surprise.
        hue_neg = 240.0 / 360.0          # blue
        hue_pos = (360.0 + 35.0) / 360.0  # amber, expressed past 1.0 so
                                          # interp goes through 1.0/0.0
        t_v = (valence + 1.0) * 0.5      # 0..1
        base_hue = (hue_neg + (hue_pos - hue_neg) * t_v) % 1.0

        # --- Temporal flow ------------------------------------------------
        # Phase advances at flow_speed cycles/sec. SMOOTH range: ~50s for one
        # strip-length when calm, ~5.5s when excited. Earlier 0.05..0.45 felt
        # jittery — this is much more meditative.
        now = time.monotonic()
        flow_speed = 0.02 + 0.16 * arousal
        phase = (now * flow_speed) * 2.0 * math.pi

        # --- Per-LED hue spread — wide enough at high arousal to read as
        # "rainbow" along the strip, tight at low arousal so it's a single
        # mood color. 0.04 (calm, ~14° spread) → 0.45 (excited, ~160° = nearly
        # half the wheel = strong rainbow band that flows along the strip).
        jitter_amp = 0.04 + 0.45 * arousal
        hue_wave = np.sin(np.linspace(0.0, 2.0 * math.pi, n) + phase) * jitter_amp
        hue = (base_hue + hue_wave) % 1.0

        # --- Saturation: low at calm/neutral, high at excited or extreme V
        # (0,0) → 0.20 (near-white, soft);  (1,0) → 0.50 (warm but soft);
        # (0,1) → 0.70 (vivid);              (1,1) → 1.00 (maxed).
        sat_value = min(1.0, 0.20 + 0.30 * abs(valence) + 0.50 * arousal)
        sat = np.full(n, sat_value, dtype=np.float32)

        # --- Brightness ripple — much gentler than before, ALWAYS floored
        # so the strip never goes fully dark. ripple_amp now 0.05..0.15 (was
        # 0.10..0.35), and val is clamped to a min visible level after the
        # brightness multiplier so dips in the audio envelope can't blank
        # individual LEDs to 0.
        ripple_amp = 0.05 + 0.10 * arousal
        ripple = (1.0 - ripple_amp) + ripple_amp * np.sin(
            np.linspace(0.0, 3.0 * math.pi, n) + phase * 1.7)
        val = brightness * ripple
        # Visible-floor: brightness=0.08 minimum from the reactive loop, so a
        # 0.10 floor here keeps every LED at least faintly lit even at the
        # quietest audio moment. Set on the array so sparkle additions still
        # land on top.
        np.maximum(val, 0.10, out=val)

        # --- Sparkle: scattered bright pulses, but MUCH gentler than before.
        # Threshold raised 0.4→0.65 so only genuinely high-arousal triggers
        # spark. Density halved (0.05→0.025), reseed slowed (5Hz→2.5Hz) so
        # sparkles linger ~400ms instead of 200ms — reads as breathing rather
        # than churning. Brighter delta also reduced (0.45→0.25).
        if arousal > 0.65:
            sparkle_density = 0.025 * (arousal - 0.65) / 0.35   # 0..0.025
            n_sparkles = max(1, int(n * sparkle_density))
            seed = int(now * 2.5)                                # 2.5 changes/sec
            rng = np.random.RandomState(seed)
            sparkle_idx = rng.choice(n, size=min(n_sparkles, n), replace=False)
            val[sparkle_idx] = np.minimum(1.0, val[sparkle_idx] + 0.25)
            sat[sparkle_idx] = sat[sparkle_idx] * 0.6

        frame = _hsv_to_rgb_array(hue, sat, val)
        self.send(frame)

    # --- audio-reactive realtime loop ------------------------------------

    def react_to_audio(self, audio: np.ndarray, sr: int,
                       valence: float = 0.0, arousal: float = 0.5,
                       palette: str = "emotion",
                       update_hz: float = 60.0,
                       min_brightness: float = 0.08,
                       max_brightness: float = 1.0):
        """Start a thread that emits frames at update_hz, brightness following
        the audio envelope.

        palette:
          "emotion"  — emotion_field driven by (valence, arousal). Used for
                       intimate / single-clip playback so the strip "feels
                       like" the speaker.
          "rainbow"  — full-spectrum hue band that drifts along the strip,
                       brightness follows audio. Used for layered chorus
                       playback (matches the WLED preset 4 vibe).

        Call stop_reacting() when playback ends.
        """
        self.stop_reacting()
        self._react_stop.clear()

        env = _compute_envelope(audio, sr, update_hz)
        target = self._react_loop_rainbow if palette == "rainbow" else self._react_loop
        self._react_thread = threading.Thread(
            target=target,
            args=(env, update_hz, valence, arousal, min_brightness, max_brightness),
            daemon=True,
        )
        self._react_thread.start()

    def _react_loop(self, envelope: np.ndarray, update_hz: float,
                    valence: float, arousal: float,
                    min_b: float, max_b: float):
        interval = 1.0 / update_hz
        next_t = time.monotonic()
        last_brightness = max_b
        for level in envelope:
            if self._react_stop.is_set():
                break
            brightness = min_b + (max_b - min_b) * float(level)
            last_brightness = brightness
            self.emotion_field(valence, arousal, brightness)
            next_t += interval
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Falling behind — re-sync rather than spiral
                next_t = time.monotonic()
        # Smooth fade-down so WLED's hold_timeout doesn't freeze on a bright
        # frame (which then snaps to whatever the next preset commands).
        self._fade_out_emotion(valence, arousal, from_b=last_brightness)

    def _react_loop_rainbow(self, envelope: np.ndarray, update_hz: float,
                            valence: float, arousal: float,
                            min_b: float, max_b: float):
        """Full-spectrum rainbow band drifting along the strip — brightness
        follows audio envelope, hue cycles continuously across all LEDs."""
        interval = 1.0 / update_hz
        next_t = time.monotonic()
        n = self.num_leds
        # Position of each LED on the hue wheel (0..1 across the strip)
        positions = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        sat = np.full(n, 0.95, dtype=np.float32)
        last_brightness = max_b
        # Drift speed: 0.10 cycles/sec → one full rainbow rotation every 10s.
        # Slow enough to feel meditative, fast enough to be visibly "alive".
        drift_cps = 0.10
        for level in envelope:
            if self._react_stop.is_set():
                break
            brightness = min_b + (max_b - min_b) * float(level)
            last_brightness = brightness
            now = time.monotonic()
            phase = now * drift_cps
            hue = (positions + phase) % 1.0
            val = np.full(n, max(brightness, 0.18), dtype=np.float32)
            frame = _hsv_to_rgb_array(hue, sat, val)
            self.send(frame)
            next_t += interval
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.monotonic()
        # Smooth rainbow fade-down
        self._fade_out_rainbow(positions, sat, from_b=last_brightness)

    def _fade_out_emotion(self, valence: float, arousal: float, from_b: float):
        """Linear fade from `from_b` down to 0.10 over ~0.4s, ~12 frames."""
        steps = 12
        target = 0.10
        for i in range(steps):
            if self._react_stop.is_set():
                break
            t = (i + 1) / steps
            b = from_b + (target - from_b) * t
            self.emotion_field(valence, arousal, max(b, target))
            time.sleep(0.035)

    def _fade_out_rainbow(self, positions: np.ndarray, sat: np.ndarray,
                           from_b: float):
        """Same fade for the rainbow palette — keeps drift active during fade."""
        steps = 12
        target = 0.12
        n = positions.shape[0]
        drift_cps = 0.10
        for i in range(steps):
            if self._react_stop.is_set():
                break
            t = (i + 1) / steps
            b = from_b + (target - from_b) * t
            now = time.monotonic()
            phase = now * drift_cps
            hue = (positions + phase) % 1.0
            val = np.full(n, max(b, target), dtype=np.float32)
            frame = _hsv_to_rgb_array(hue, sat, val)
            self.send(frame)
            time.sleep(0.035)

    def stop_reacting(self):
        self._react_stop.set()
        if self._react_thread is not None and self._react_thread.is_alive():
            self._react_thread.join(timeout=0.5)
        self._react_thread = None

    # --- sleep indicator --------------------------------------------------

    def start_sleep_indicator(self, color: tuple[int, int, int] = (20, 40, 140),
                              period_s: float = 6.0, update_hz: float = 8.0,
                              min_b: float = 0.05, max_b: float = 0.30):
        """Slow breathing pulse in a distinct color — the visible "asleep"
        state. Streams frames continuously (low rate) so it overrides the
        idle preset until stop_sleep_indicator() lets WLED fall back."""
        self.stop_sleep_indicator()
        self._sleep_stop.clear()
        self._sleep_thread = threading.Thread(
            target=self._sleep_loop,
            args=(color, period_s, update_hz, min_b, max_b),
            daemon=True,
        )
        self._sleep_thread.start()
        logger.info("Sleep indicator started (color=%s, period=%.1fs)", color, period_s)

    def _sleep_loop(self, color: tuple[int, int, int], period_s: float,
                    update_hz: float, min_b: float, max_b: float):
        interval = 1.0 / update_hz
        base = np.array(color, dtype=np.float32)
        start = time.monotonic()
        while not self._sleep_stop.wait(interval):
            t = time.monotonic() - start
            # Cosine breath between min_b and max_b
            b = min_b + (max_b - min_b) * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period_s))
            frame = np.tile((base * b).astype(np.uint8), (self.num_leds, 1))
            self.send(frame)
        # Quick fade to black so the hold_timeout fallback to the idle
        # preset doesn't snap from a lit frame.
        for i in range(6, -1, -1):
            frame = np.tile((base * min_b * (i / 6.0)).astype(np.uint8),
                            (self.num_leds, 1))
            self.send(frame)
            time.sleep(0.04)

    def stop_sleep_indicator(self):
        self._sleep_stop.set()
        if self._sleep_thread is not None and self._sleep_thread.is_alive():
            self._sleep_thread.join(timeout=1.0)
        self._sleep_thread = None

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
        self.stop_sleep_indicator()
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
