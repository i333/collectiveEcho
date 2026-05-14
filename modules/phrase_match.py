"""
Phonetic + edit-distance phrase matching, with **configurable keyword buckets**.

A "bucket" is a labelled set of phrase variants that map to a particular
response strategy. Each bucket has:
    id:                       short stable identifier (e.g. "ily", "ily_sandy")
    patterns:                 list of literal phrases (lower-case)
    slang:                    list of single-token slang forms (e.g. "ily")
    priority:                 int — higher = checked first (longer/more
                              specific phrases must beat shorter ones)
    anchor_extra:             list of extra word-start anchors that, in
                              addition to the universal love/luv/ily anchors,
                              count as valid (e.g. "sandy" for the Sandy bucket).
                              The bucket's match still requires the BASE
                              love-word anchor — we never trigger purely on
                              a name without the love phrase.

The matcher loads buckets from config (or uses sensible defaults that
preserve the V1 ily/ily_too behaviour).

Anti-overfitting rules (apply to every bucket):
  1. The transcript must contain a *word* starting with "lov" / "luv", or one
     of the slang tokens ("ily", "iluvu", "iloveyou"). Substring "lov" alone
     inside "glove" / "clover" does NOT count — Metaphone collapses too many
     unrelated words to "LF" without this gate.
  2. Exact-match passes use word boundaries: "love you" matches "i love you"
     but not "glove yours".
  3. Phonetic / Levenshtein passes are only considered after the anchor gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Minimal Metaphone (subset of double-metaphone — single primary code is enough
# for our purposes, and avoids a 600-line dependency).
# --------------------------------------------------------------------------

_VOWELS = set("AEIOU")


def metaphone(word: str) -> str:
    """Compute a Metaphone-like phonetic code for an English word."""
    if not word:
        return ""
    w = word.upper().strip()
    w = re.sub(r"[^A-Z]", "", w)
    if not w:
        return ""

    if w.startswith(("KN", "GN", "PN", "AE", "WR")):
        w = w[1:]
    if w.startswith("X"):
        w = "S" + w[1:]
    if w.startswith("WH"):
        w = "W" + w[2:]

    out = []
    i = 0
    n = len(w)
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else ""

        if c == w[i - 1] and c != "C" and i > 0:
            i += 1
            continue

        if c in _VOWELS:
            if i == 0:
                out.append(c)
            i += 1
            continue

        if c == "B":
            if not (i == n - 1 and w[i - 1] == "M"):
                out.append("B")
        elif c == "C":
            if nxt == "H":
                out.append("X")
                i += 1
            elif nxt in ("I", "E", "Y"):
                out.append("S")
            else:
                out.append("K")
        elif c == "D":
            if nxt == "G" and i + 2 < n and w[i + 2] in "EIY":
                out.append("J")
                i += 2
            else:
                out.append("T")
        elif c == "F":
            out.append("F")
        elif c == "G":
            if nxt == "H":
                if i > 0 and w[i - 1] not in _VOWELS:
                    pass
                else:
                    out.append("F")
                i += 1
            elif nxt == "N":
                pass
            elif nxt in ("E", "I", "Y"):
                out.append("J")
            else:
                out.append("K")
        elif c == "H":
            if i > 0 and w[i - 1] in _VOWELS and (nxt == "" or nxt not in _VOWELS):
                pass
            else:
                out.append("H")
        elif c == "J":
            out.append("J")
        elif c == "K":
            if i > 0 and w[i - 1] == "C":
                pass
            else:
                out.append("K")
        elif c == "L":
            out.append("L")
        elif c == "M":
            out.append("M")
        elif c == "N":
            out.append("N")
        elif c == "P":
            if nxt == "H":
                out.append("F")
                i += 1
            else:
                out.append("P")
        elif c == "Q":
            out.append("K")
        elif c == "R":
            out.append("R")
        elif c == "S":
            if nxt == "H":
                out.append("X")
                i += 1
            else:
                out.append("S")
        elif c == "T":
            if nxt == "H":
                out.append("0")
                i += 1
            else:
                out.append("T")
        elif c == "V":
            out.append("F")
        elif c == "W":
            if i + 1 < n and w[i + 1] in _VOWELS:
                out.append("W")
        elif c == "X":
            out.append("KS")
        elif c == "Y":
            if i + 1 < n and w[i + 1] in _VOWELS:
                out.append("Y")
        elif c == "Z":
            out.append("S")

        i += 1

    return "".join(out)


def phonetic_phrase(phrase: str) -> str:
    return " ".join(metaphone(w) for w in phrase.split() if w)


def _whole_phrase_in(phrase: str, text: str) -> bool:
    """Word-boundary substring check. 'love you' matches inside 'i love you'
    but NOT inside 'glove yours'."""
    pat = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pat, text) is not None


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# PhraseBucket + PhraseMatcher
# --------------------------------------------------------------------------

@dataclass
class PhraseBucket:
    id: str
    patterns: list[str] = field(default_factory=list)
    slang: list[str] = field(default_factory=list)
    priority: int = 0
    # Extra word-start anchors required to "be present" in addition to the
    # universal love-anchor. Empty = bucket has no extra-word requirement.
    anchor_extra: list[str] = field(default_factory=list)
    # Free-form metadata passed through to the response layer
    metadata: dict = field(default_factory=dict)

    # Precomputed lookups (filled by PhraseMatcher)
    _patterns_phonetic: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _slang_phonetic: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _anchor_extra_re: re.Pattern | None = field(default=None, repr=False)


@dataclass
class PhraseHit:
    phrase_type: str          # bucket id
    matched_span: str
    confidence: float
    matched_via: str          # "exact" | "phonetic" | "levenshtein"


# Universal "love-word" anchor — at least one of these must be present for
# any phonetic/Levenshtein match to be considered. Exact-match passes don't
# need it because they already require word-boundary "lov"/"luv"/"ily".
_LOVE_ANCHOR_RE = re.compile(
    r"(?:\b(?:lov|luv)[a-z]*\b)|(?:\bily\b)|(?:\biluvu\b)|(?:\biloveyou\b)"
)


def _default_buckets() -> list[PhraseBucket]:
    """Backward-compatible defaults (ily + ily_too) used when config has no
    `phrases` block. Mirrors V1 behaviour."""
    return [
        PhraseBucket(
            id="ily_too",
            patterns=[
                "i love you too", "i love u too",
                "i love you to", "i love u to",
                "i love ya too", "i love yew too",
            ],
            priority=20,
        ),
        PhraseBucket(
            id="ily",
            patterns=[
                "i love you", "i love u", "i love ya", "i love yew",
                "i loved you", "i loved u",
                "love you", "love u",
            ],
            slang=["ily", "iloveyou", "iluvu"],
            priority=10,
        ),
    ]


def buckets_from_config(cfg: dict) -> list[PhraseBucket]:
    """Build PhraseBucket list from config. Falls back to defaults if missing."""
    phrases_cfg = cfg.get("phrases") or {}
    bucket_defs = phrases_cfg.get("buckets")
    if not bucket_defs:
        return _default_buckets()

    out: list[PhraseBucket] = []
    for raw in bucket_defs:
        out.append(PhraseBucket(
            id=str(raw["id"]),
            patterns=[p.lower() for p in raw.get("patterns", []) if p],
            slang=[s.lower() for s in raw.get("slang", []) if s],
            priority=int(raw.get("priority", 0)),
            anchor_extra=[a.lower() for a in raw.get("anchor_extra", []) if a],
            metadata={k: v for k, v in raw.items()
                       if k not in ("id", "patterns", "slang", "priority", "anchor_extra")},
        ))
    return out


class PhraseMatcher:
    """Match keyword phrases in noisy transcripts, with configurable buckets."""

    def __init__(self, buckets: list[PhraseBucket] | None = None,
                 max_phonetic_edits: int = 0,
                 min_confidence_for_partial: float = 0.85):
        self.max_phonetic_edits = max_phonetic_edits
        self.min_confidence_for_partial = min_confidence_for_partial
        self.buckets = sorted(
            buckets or _default_buckets(),
            key=lambda b: -b.priority,                # higher first
        )
        self._compile_buckets()

    def _compile_buckets(self):
        for b in self.buckets:
            b._patterns_phonetic = [(p, phonetic_phrase(p)) for p in b.patterns]
            b._slang_phonetic = [(s, metaphone(s)) for s in b.slang]
            if b.anchor_extra:
                # Require at least one of these to appear as a whole word
                pat = r"|".join(rf"\b{re.escape(a)}\b" for a in b.anchor_extra)
                b._anchor_extra_re = re.compile(pat)
            else:
                b._anchor_extra_re = None

    # --- public ----------------------------------------------------------

    def find(self, text: str) -> PhraseHit | None:
        if not text:
            return None
        text = text.lower().strip()

        # Universal love-anchor — required for phonetic/Levenshtein passes.
        has_love_anchor = bool(_LOVE_ANCHOR_RE.search(text))

        # Buckets are sorted by priority. Walk in order — first hit wins.
        for b in self.buckets:
            if b._anchor_extra_re is not None and not b._anchor_extra_re.search(text):
                continue   # bucket requires an extra word that isn't present

            # 1. Exact substring with word boundaries (most reliable signal)
            for v in b.patterns:
                if _whole_phrase_in(v, text):
                    return PhraseHit(b.id, v, 1.0, "exact")
            for v in b.slang:
                if f" {v} " in f" {text} " or text == v:
                    return PhraseHit(b.id, v, 0.85, "exact")

            # 2. Name-based routing: a bucket with anchor_extra (e.g. Sandy)
            #    is essentially "the BUCKET for this named recipient." If
            #    BOTH the love-anchor AND the bucket-anchor word are present
            #    in the transcript, route to this bucket regardless of word
            #    order. Catches:
            #       "sandy i love you"        (anchor first)
            #       "i love sandy"            (no "you")
            #       "i love you, sandy boy"   (trailing words)
            #    Without this, we'd fall through to ily and lose Sandy.
            if b.anchor_extra and has_love_anchor:
                # We already verified b._anchor_extra_re.search(text) above.
                return PhraseHit(b.id, b.patterns[0] if b.patterns else b.id,
                                  0.90, "name_anchor")

            # 3. Phonetic substring — only if love-anchor is present.
            if not has_love_anchor:
                continue

            text_phonetic = phonetic_phrase(text)
            for orig, phon in b._patterns_phonetic:
                if phon and phon in text_phonetic:
                    return PhraseHit(b.id, orig, 0.80, "phonetic")

            # 4. Levenshtein — DISABLED by default (max_phonetic_edits=0).
            #    Metaphone collapses too many words (love/leave/live/lava all
            #    → "LF") for edit-distance to be safe at the phonetic level.
            #    Exact + phonetic-substring catches all realistic variants.
            if self.max_phonetic_edits > 0:
                best: tuple[float, PhraseHit] | None = None
                for orig, phon in b._patterns_phonetic:
                    if not phon:
                        continue
                    win = len(phon)
                    if win < 3 or win > len(text_phonetic):
                        continue
                    for i in range(0, len(text_phonetic) - win + 1):
                        window = text_phonetic[i:i + win]
                        d = levenshtein(phon, window)
                        if d <= self.max_phonetic_edits:
                            conf = 1.0 - (d / max(win, 1))
                            if conf >= self.min_confidence_for_partial:
                                hit = PhraseHit(b.id, orig, conf, "levenshtein")
                                if best is None or conf > best[0]:
                                    best = (conf, hit)
                if best is not None:
                    return best[1]

        return None

    def quick_contains_love(self, text: str) -> bool:
        """Cheap pre-filter for streaming partials."""
        if not text:
            return False
        return _LOVE_ANCHOR_RE.search(text.lower()) is not None

    def get_bucket(self, bucket_id: str) -> PhraseBucket | None:
        for b in self.buckets:
            if b.id == bucket_id:
                return b
        return None
