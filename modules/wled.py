"""
WLED integration — sends HTTP preset requests to a WLED controller.
All requests are fire-and-forget (non-blocking) so they never delay
the audio response pipeline.

Includes real-time audio-reactive brightness: during playback, the
LED brightness tracks the audio envelope so the light breathes with
the voice.
"""

import logging
import threading
import time

import numpy as np
import requests

logger = logging.getLogger(__name__)


class WLEDController:
    def __init__(self, cfg: dict):
        wled_cfg = cfg["wled"]
        self.host = wled_cfg["host"].rstrip("/")
        self.timeout = wled_cfg.get("timeout_sec", 2)
        self.presets = wled_cfg.get("presets", {})

        # Reactive brightness range (0-255)
        reactive_cfg = wled_cfg.get("reactive", {})
        self.bri_min = reactive_cfg.get("bri_min", 20)
        self.bri_max = reactive_cfg.get("bri_max", 255)
        self.update_hz = reactive_cfg.get("update_hz", 12)

        self._react_stop = threading.Event()

    # --- low-level ---------------------------------------------------------

    def _send(self, preset_id: int):
        """Fire-and-forget: send preset in a background thread."""
        t = threading.Thread(target=self._do_send, args=(preset_id,), daemon=True)
        t.start()

    def _do_send(self, preset_id: int):
        url = f"{self.host}/json/state"
        payload = {"ps": preset_id}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.ok:
                logger.info("WLED preset %d activated", preset_id)
            else:
                logger.warning("WLED responded %d: %s", resp.status_code, resp.text[:200])
        except requests.ConnectionError:
            logger.warning("WLED unreachable at %s", self.host)
        except requests.Timeout:
            logger.warning("WLED timed out at %s", self.host)
        except Exception:
            logger.exception("WLED request failed")

    def _set_state(self, payload: dict):
        """Fire-and-forget state update (non-preset)."""
        t = threading.Thread(target=self._do_set_state, args=(payload,), daemon=True)
        t.start()

    def _do_set_state(self, payload: dict):
        url = f"{self.host}/json/state"
        try:
            requests.post(url, json=payload, timeout=0.8)
        except Exception:
            pass  # best-effort, don't log every missed frame

    # --- preset shortcuts --------------------------------------------------

    def idle(self):
        self._send(self.presets.get("idle", 1))

    def wake_neutral(self):
        self._send(self.presets.get("wake_neutral", 2))

    def intimate(self):
        self._send(self.presets.get("intimate_low_volume", 3))

    def layered(self):
        self._send(self.presets.get("layered_high_volume", 4))

    def reject(self):
        self._send(self.presets.get("reject_or_unclear", 5))

    def fade_out(self):
        self._send(self.presets.get("fade_out", 6))

    # --- detection flash ---------------------------------------------------

    def flash(self):
        """Quick bright flash to acknowledge detection, then dim back."""
        def _do_flash():
            url = f"{self.host}/json/state"
            try:
                # Flash bright white
                requests.post(url, json={
                    "on": True, "bri": 255, "transition": 0,
                    "seg": [{"col": [[255, 255, 255]], "fx": 0}],
                }, timeout=self.timeout)
                time.sleep(0.15)
                # Dim back down quickly
                requests.post(url, json={
                    "bri": 60, "transition": 3,
                }, timeout=self.timeout)
            except Exception:
                pass
        t = threading.Thread(target=_do_flash, daemon=True)
        t.start()

    # --- audio-reactive brightness -----------------------------------------

    def _compute_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute RMS envelope at update_hz rate. Returns array of 0.0-1.0."""
        if audio.ndim == 2:
            mono = audio[:, 0]
        else:
            mono = audio
        mono = mono.astype(np.float32)
        if mono.dtype == np.int16:
            mono = mono / 32768.0

        window = int(sr / self.update_hz)
        n_frames = len(mono) // window
        if n_frames == 0:
            return np.array([0.5])

        frames = mono[:n_frames * window].reshape(n_frames, window)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Normalize to 0-1 range
        peak = np.max(rms)
        if peak > 0:
            env = rms / peak
        else:
            env = rms
        return env

    def react_to_audio(self, audio: np.ndarray, sr: int):
        """Start a background thread that modulates WLED brightness to follow
        the audio envelope in real-time during playback.
        Call stop_reacting() when playback ends."""
        self._react_stop.clear()
        envelope = self._compute_envelope(audio, sr)
        t = threading.Thread(
            target=self._react_loop, args=(envelope,), daemon=True)
        t.start()

    def _react_loop(self, envelope: np.ndarray):
        """Send brightness updates at update_hz, timed to match playback."""
        interval = 1.0 / self.update_hz
        url = f"{self.host}/json/state"
        bri_range = self.bri_max - self.bri_min

        for i, level in enumerate(envelope):
            if self._react_stop.is_set():
                break

            bri = int(self.bri_min + level * bri_range)
            bri = max(self.bri_min, min(self.bri_max, bri))

            try:
                requests.post(url, json={"bri": bri, "transition": 1},
                              timeout=0.5)
            except Exception:
                pass  # skip missed frames

            # Sleep for the remainder of the interval
            if not self._react_stop.is_set():
                time.sleep(interval)

        logger.info("WLED reactive loop ended (%d frames)", len(envelope))

    def stop_reacting(self):
        """Stop the audio-reactive brightness loop."""
        self._react_stop.set()
