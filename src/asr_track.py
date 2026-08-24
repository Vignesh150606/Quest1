"""
Phase 2 - ASR track.

Consumes: VideoAsset.audio_path (Phase 1), target phrase, Candidate schema (Phase 1)
Produces: list[Candidate], modality="asr", event_type="speech_onset"

Use faster-whisper with word_timestamps=True (native word-level timestamps) -- not
WhisperX. See CLAUDE.md "Known Gotchas": segment-level timestamps are too coarse for a
~1-2 second line.

Verification (see PHASES.md): pytest tests/test_asr_track.py::test_example_video
"""

from src.types import VideoAsset, Candidate


def find_candidates(asset: VideoAsset, target_phrase: str) -> list[Candidate]:
    """Transcribe audio with word-level timestamps, fuzzy-match target_phrase, return candidates."""
    raise NotImplementedError
