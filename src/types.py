"""
Shared data contracts for the pipeline (Phase 1).

Defines the types passed between Ingest -> ASR/OCR tracks -> Arbiter -> Refine -> Report.
Introduced in Phase 1 (not Phase 4) because the subtitle fast-path already needs to
construct a Candidate, and both the ASR and OCR tracks (Phase 2/3) depend on this schema
existing before they can run.

See PHASES.md Phase 1 and CLAUDE.md's "Candidate schema" section.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


Modality = Literal["subtitle", "asr", "ocr"]
EventType = Literal["speech_onset", "visual_text_onset"]


@dataclass
class VideoMetadata:
    fps: float
    duration_s: float
    width: int
    height: int


@dataclass
class VideoAsset:
    video_path: str
    audio_path: str
    metadata: VideoMetadata
    subtitle_paths: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    modality: Modality
    event_type: EventType
    timestamp: float
    end_timestamp: Optional[float]
    matched_text: str
    normalized_text: str
    similarity: float
    confidence: Optional[float] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class FrameMatch:
    frame_idx: int
    timestamp_s: float
    text: str
    image_path: str
    modality: Modality
    match_score: float


@dataclass
class AmbiguousResult:
    candidates: list  # list[Candidate] that disagree beyond the arbiter's tolerance
    reason: str
