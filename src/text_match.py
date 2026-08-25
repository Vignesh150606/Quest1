"""
Shared text normalization + fuzzy matching (Phase 1).

Not in PHASES.md's file list for Phase 1 (which names only ingest.py, types.py,
test_ingest.py). Added anyway because the arbiter (Phase 4) compares Candidate.similarity
scores across modalities -- subtitle, ASR, and OCR -- and that comparison is only
meaningful if all three normalize and score text the same way. Phase 1's subtitle
fast-path is simply the first consumer of this module; Phases 2 and 3 are expected to
import it rather than reimplement it.
"""

import re
import unicodedata

from rapidfuzz import fuzz

DEFAULT_MATCH_THRESHOLD = 0.85  # 0-1 scale (matches Candidate.confidence's scale)

_UNICODE_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
}
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, unify unicode quotes/dashes to ASCII, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _UNICODE_QUOTES.items():
        text = text.replace(src, dst)
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def similarity(target: str, candidate: str) -> float:
    """
    Fuzzy match score (0-1) of target against candidate, after normalizing both.

    Uses partial_ratio rather than ratio: target is typically a short phrase that is a
    substring of a longer subtitle cue / transcript segment, and ratio would unfairly
    penalize the surrounding words that aren't part of the target phrase. Use this for
    "does this longer text contain the target" (subtitle cues, OCR'd lines) -- NOT for
    scoring a word-window extent against the target, where partial_ratio's substring
    tolerance makes short windows look like false-positive perfect matches. For that,
    use window_similarity below.
    """
    return fuzz.partial_ratio(normalize(target), normalize(candidate)) / 100.0


def window_similarity(target: str, window_text: str) -> float:
    """
    Fuzzy match score (0-1) of target against a word window whose extent is meant to
    match target exactly (e.g. an ASR word-window candidate for the target phrase).

    Uses ratio, not partial_ratio: ratio penalizes both missing and extra content, so a
    window that's too short or too long scores lower than one whose boundaries actually
    align with the target. partial_ratio would score `window_similarity("my mind rebels
    at stagnation", "my mind")` as a perfect match (1.0), since "my mind" is a substring
    match -- exactly the false positive this function exists to avoid.
    """
    return fuzz.ratio(normalize(target), normalize(window_text)) / 100.0
