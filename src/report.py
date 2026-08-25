"""
Phase 6 - Report.

Consumes: FrameMatch(es) or AmbiguousResult/None (Phase 4/5), VideoAsset (Phase 1)
Produces: report.json, formatted stdout summary (Timestamp / Frame / Text)

report.json is one shape regardless of status: top-level frame/image_path/timestamp
fields carry the primary answer (null for "not_found"), and "candidates" lists every
alternative FrameMatch when status is "ambiguous". src/main.py's run_pipeline() is
responsible for refining every winning Candidate -- the single winner, or every
AmbiguousResult alternative -- into a FrameMatch via src/refine.py BEFORE calling
write_report(); this module only formats and serializes, it never seeks/decodes video.

Verification (see PHASES.md): pytest tests/test_end_to_end.py::test_full_run
"""

import json
import os
from typing import Optional, Union

from src.types import AmbiguousResult, FrameMatch


def write_report(
    result: Union[FrameMatch, AmbiguousResult, None],
    output_dir: str,
    *,
    video_url: Optional[str] = None,
    target_phrase: Optional[str] = None,
) -> str:
    """
    Write report.json and print a formatted stdout summary. Returns the path to
    report.json.

    result=None (arbiter found nothing above threshold) produces status="not_found"
    with null frame/image_path/timestamp fields -- not an exception. This is a
    legitimate outcome (the phrase genuinely isn't in this video, or nothing cleared
    the confidence/similarity thresholds), not a pipeline failure.

    result=AmbiguousResult (its .candidates must already be FrameMatch objects, refined
    by the caller) produces status="ambiguous": the top-level fields report the
    highest-match_score alternative as the primary answer, and every alternative
    (including that one) is listed in "candidates" -- satisfies CLAUDE.md's "never
    silently pick one and hide the disagreement" while still giving the evaluator a
    single usable answer at the top level.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.json")

    if result is None:
        report = _build_report("not_found", None, [], video_url, target_phrase)
    elif isinstance(result, AmbiguousResult):
        primary = max(result.candidates, key=lambda fm: fm.match_score)
        report = _build_report("ambiguous", primary, result.candidates, video_url, target_phrase)
    else:
        report = _build_report("match", result, [], video_url, target_phrase)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    return report_path


def _build_report(
    status: str,
    primary: Optional[FrameMatch],
    alternatives: list,
    video_url: Optional[str],
    target_phrase: Optional[str],
) -> dict:
    return {
        "status": status,
        "video_url": video_url,
        "dialogue_text": target_phrase,
        "timestamp": _format_timestamp(primary.timestamp_s) if primary else None,
        "timestamp_s": primary.timestamp_s if primary else None,
        "frame": primary.frame_idx if primary else None,
        "extracted_text": primary.text if primary else None,
        "image_path": primary.image_path if primary else None,
        "modality": primary.modality if primary else None,
        "match_score": primary.match_score if primary else None,
        "candidates": [_frame_match_to_dict(fm) for fm in alternatives],
    }


def _frame_match_to_dict(fm: FrameMatch) -> dict:
    return {
        "modality": fm.modality,
        "timestamp": _format_timestamp(fm.timestamp_s),
        "timestamp_s": fm.timestamp_s,
        "frame": fm.frame_idx,
        "extracted_text": fm.text,
        "image_path": fm.image_path,
        "match_score": fm.match_score,
    }


def _format_timestamp(seconds: float) -> str:
    """HH:MM:SS.sss -- explicitly required by the problem statement's output format."""
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _print_summary(report: dict) -> None:
    print(f"Status:    {report['status']}")
    if report["status"] == "not_found":
        print("No match found above the confidence/similarity thresholds.")
        return
    print(f"Timestamp: {report['timestamp']}")
    print(f"Frame:     {report['frame']}")
    print(f"Text:      {report['extracted_text']}")
    if report["status"] == "ambiguous":
        print(
            f"Note: {len(report['candidates'])} disagreeing candidates found -- "
            f"see report.json's 'candidates' list for the alternatives."
        )
