"""
Slow behavioral drift — the piece responds differently across the day.

Persisted state:
  - lifetime_triggers: total number of "I love you" detections ever
  - session_start: epoch when this process started

Time-of-day mapping (local time):
   05:00–11:00  "morning"   bright, slightly quicker layers, warmer
   11:00–17:00  "afternoon" balanced
   17:00–22:00  "evening"   warmer, slower layers, deeper reverb
   22:00–05:00  "night"     intimate, slowest, near-monochrome

Returned MoodState modulates a small set of run-time parameters that downstream
modules consult. Nothing here is destructive — every modulation is bounded and
config-overridable.
"""

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MoodState:
    period: str = "afternoon"             # morning/afternoon/evening/night
    warmth: float = 0.0                   # -1..+1, cool → warm
    pace: float = 1.0                     # multiplier on layer delays (>1 = slower)
    intimacy: float = 0.5                 # 0..1, modulates layer count + reverb
    palette_bias_v: float = 0.0           # additive offset for WLED valence
    palette_bias_a: float = 0.0           # additive offset for WLED arousal
    lifetime_triggers: int = 0
    notes: list[str] = field(default_factory=list)


class MoodKeeper:
    """Persists lifetime trigger count and computes current MoodState."""

    def __init__(self, state_file: str = "state.json"):
        self.state_file = state_file
        self._mood_state_path = self._mood_path_for(state_file)
        self.lifetime_triggers = 0
        self._load()

    @staticmethod
    def _mood_path_for(state_file: str) -> str:
        # Keep mood state alongside the clip state, but in its own file so the
        # main state.json schema stays clean.
        d = os.path.dirname(state_file) or "."
        return os.path.join(d, "mood_state.json")

    def _load(self):
        if not os.path.exists(self._mood_state_path):
            return
        try:
            with open(self._mood_state_path, "r") as f:
                data = json.load(f)
            self.lifetime_triggers = int(data.get("lifetime_triggers", 0))
            logger.info("MoodKeeper: lifetime_triggers=%d", self.lifetime_triggers)
        except Exception:
            logger.exception("MoodKeeper: could not load mood state")

    def _save(self):
        try:
            with open(self._mood_state_path, "w") as f:
                json.dump({"lifetime_triggers": self.lifetime_triggers}, f)
        except Exception:
            logger.exception("MoodKeeper: could not save mood state")

    def record_trigger(self):
        self.lifetime_triggers += 1
        # Save sparingly — every 10 triggers — to avoid disk wear
        if self.lifetime_triggers % 10 == 0:
            self._save()

    def current(self, now: dt.datetime | None = None) -> MoodState:
        if now is None:
            now = dt.datetime.now()
        h = now.hour + now.minute / 60.0

        if 5.0 <= h < 11.0:
            period = "morning"
            warmth = 0.2
            pace = 0.9
            intimacy = 0.45
            bias_v = +0.15
            bias_a = +0.05
        elif 11.0 <= h < 17.0:
            period = "afternoon"
            warmth = 0.0
            pace = 1.0
            intimacy = 0.5
            bias_v = 0.0
            bias_a = 0.0
        elif 17.0 <= h < 22.0:
            period = "evening"
            warmth = 0.3
            pace = 1.1
            intimacy = 0.65
            bias_v = +0.10
            bias_a = -0.05
        else:
            period = "night"
            warmth = 0.15
            pace = 1.25
            intimacy = 0.85
            bias_v = -0.05
            bias_a = -0.15

        return MoodState(
            period=period,
            warmth=warmth,
            pace=pace,
            intimacy=intimacy,
            palette_bias_v=bias_v,
            palette_bias_a=bias_a,
            lifetime_triggers=self.lifetime_triggers,
        )

    def close(self):
        self._save()
