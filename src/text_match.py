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

DEFAULT_MATCH_THRESHOLD = 85.0  # rapidfuzz 0-100 scale

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
    Fuzzy match score (0-100) of target against candidate, after normalizing both.

    Uses partial_ratio rather than ratio: target is typically a short phrase that is a
    substring of a longer subtitle cue / transcript segment, and ratio would unfairly
    penalize the surrounding words that aren't part of the target phrase.
    """
    return fuzz.partial_ratio(normalize(target), normalize(candidate))
