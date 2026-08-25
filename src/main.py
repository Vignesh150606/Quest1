"""
Phase 6 - CLI entry point.

Wires: ingest -> (subtitle fast-path | ASR + OCR tracks) -> arbiter -> refine -> report

Usage:
    python -m src.main --url <video_url> --dialogue-text "<target phrase>" --output <dir>

Verification (see PHASES.md): pytest tests/test_end_to_end.py::test_full_run
"""

import argparse
import sys
from typing import Optional

from src.arbiter import reconcile
from src.asr_track import find_candidates as find_asr_candidates
from src.ingest import prepare_asset, try_subtitle_fast_path
from src.ocr_track import find_candidates as find_ocr_candidates
from src.refine import to_frame_match
from src.report import write_report
from src.types import AmbiguousResult, Candidate


def run_pipeline(
    video_url: str,
    dialogue_text: str,
    output_dir: str,
    *,
    model_size: str = "small",
    work_dir: Optional[str] = None,
    skip_ocr: bool = False,
) -> str:
    """
    Callable entry point (used by tests, wrapped by main() for CLI use).

    Runs ingest -> subtitle fast-path -> ASR/OCR tracks -> arbiter -> refine -> report.
    Returns the path to the written report.json.

    Both ASR and OCR tracks run unconditionally by default -- CLAUDE.md's "Agreed
    Architecture": the shared dual-track pipeline is the default, and adding a
    confidence-threshold short-circuit needs its own measured latency/reliability
    justification before it's added, not just an assumption that OCR is "too slow".
    Phase 3's planned full-video OCR timing measurement was deliberately deferred (see
    prompts.txt) -- without that evidence, no automatic short-circuit is added here.
    skip_ocr is a manual escape hatch for the caller's own judgment (e.g. a known
    spoken-dialogue-only video), not an automatic policy this function decides on its
    own.
    """
    asset = prepare_asset(video_url, work_dir=work_dir)

    candidates: list[Candidate] = []

    subtitle_hit = try_subtitle_fast_path(asset, dialogue_text)
    if subtitle_hit is not None:
        candidates.append(subtitle_hit)

    candidates.extend(find_asr_candidates(asset, dialogue_text, model_size=model_size))

    if not skip_ocr:
        candidates.extend(find_ocr_candidates(asset, dialogue_text))

    arbiter_result = reconcile(candidates)

    if arbiter_result is None:
        result_for_report = None
    elif isinstance(arbiter_result, AmbiguousResult):
        frame_matches = [
            to_frame_match(c, asset, output_dir) for c in arbiter_result.candidates
        ]
        result_for_report = AmbiguousResult(
            candidates=frame_matches, reason=arbiter_result.reason
        )
    else:
        result_for_report = to_frame_match(arbiter_result, asset, output_dir)

    return write_report(
        result_for_report, output_dir, video_url=video_url, target_phrase=dialogue_text
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find the exact frame where a dialogue appears in a video URL."
    )
    parser.add_argument("--url", required=True, help="Video URL to search")
    parser.add_argument(
        "--dialogue-text", required=True, help="Target dialogue text to locate"
    )
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument(
        "--model",
        default="small",
        help="faster-whisper model size for the ASR track (default: small)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Download/cache directory (default: $QUEST1_WORK_DIR or ./.cache)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help=(
            "Skip the OCR track. Manual escape hatch only -- no automatic OCR "
            "short-circuit is applied by default; see run_pipeline()'s docstring."
        ),
    )
    args = parser.parse_args()

    try:
        report_path = run_pipeline(
            args.url,
            args.dialogue_text,
            args.output,
            model_size=args.model,
            work_dir=args.work_dir,
            skip_ocr=args.skip_ocr,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
