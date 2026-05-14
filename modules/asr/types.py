"""Shared types for the ASR package."""

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptResult:
    text: str = ""
    language: str = ""
    confidence: float = 1.0
    words: list[Word] = field(default_factory=list)
    # Optional per-segment timings — populated by Whisper, not Vosk
    segments: list[tuple[float, float, str]] = field(default_factory=list)
