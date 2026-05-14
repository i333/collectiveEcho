"""Shared types for the emotion package."""

from dataclasses import dataclass, field, asdict


@dataclass
class EmotionReport:
    """Continuous + categorical emotion description of a single clip.

    Dimensional axes (industry-standard SER outputs):
        valence    [-1, 1]  negative → positive
        arousal    [ 0, 1]  calm → excited
        dominance  [ 0, 1]  submissive → assertive

    Categorical label is derived from the dimensional point if the backend
    doesn't provide one natively. Confidence is the backend's own self-rated
    certainty (1.0 if not provided).
    """

    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5

    label: str = "neutral"
    confidence: float = 1.0

    # Optional prosody features (populated by prosody backend, often empty for SER)
    mean_pitch_hz: float = 0.0
    pitch_std_hz: float = 0.0
    speaking_rate: float = 0.0    # syllables/sec proxy
    energy_mean: float = 0.0
    jitter: float = 0.0

    # Free-form notes from the backend
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    # --- mapping helpers used by lighting / response selection -----------

    def color_temperature(self) -> str:
        """Coarse color bucket — warm/cool/neutral."""
        if self.valence > 0.25:
            return "warm"
        if self.valence < -0.25:
            return "cool"
        return "neutral"

    def intensity(self) -> str:
        """Coarse intensity — calm/medium/intense."""
        if self.arousal < 0.33:
            return "calm"
        if self.arousal > 0.66:
            return "intense"
        return "medium"
