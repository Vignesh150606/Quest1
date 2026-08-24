"""
Phase 3 - synthetic captioned clip fixture.

The example video is a spoken-dialogue case with no on-screen caption (see CLAUDE.md's
"Facts Established by Manual Investigation") -- so the OCR track needs its own fixture
that actually exercises it.

Builds a short video with a burned-in caption appearing at a known timestamp.
"""


def make_synthetic_clip(output_path: str, text: str, onset_s: float, duration_s: float = 5.0) -> dict:
    """
    Generate a short video at output_path with `text` burned in starting at onset_s.

    Returns: {"path": output_path, "text": text, "onset_s": onset_s, "fps": <fps used>}
    """
    raise NotImplementedError
