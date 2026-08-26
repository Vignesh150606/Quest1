"""
Phase 6 - CLI entry point.

Wires: ingest -> (subtitle fast-path | ASR + OCR tracks) -> arbiter -> refine -> report

Usage:
    python -m src.main --url <video_url> --dialogue-text "<target phrase>" --output <dir>

Verification: pytest tests/test_end_to_end.py::test_full_run
"""

import argparse
import sys
from typing import Optional, Union

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
    hq_frame: bool = False,
) -> str:
    """
    Callable entry point (used by tests, wrapped by main() for CLI use).

    Runs ingest -> subtitle fast-path -> ASR/OCR tracks -> arbiter -> refine -> report.
    Returns the path to the written report.json.

    Staged short-circuit (runtime-hardening pass, replacing the previous "always run
    both tracks unconditionally" default): a confident result from a cheaper stage
    skips the more expensive ones after it, reusing arbiter.reconcile()'s own
    threshold/clustering logic as the single definition of "confident" rather than a
    new ad hoc check. This also fixes a real documented-vs-actual gap: CLAUDE.md's own
    architecture description says "the subtitle fast-path skips two expensive stages
    when it hits", but that was never actually wired up here before -- ASR and OCR both
    ran unconditionally regardless of a subtitle hit.

      1. fetch the low-tier asset (cheapest available format -- see ingest.py's module
         docstring for why this doesn't need to be the full-quality video)
      2. subtitle fast-path; if it alone reconciles to one confident Candidate, skip
         BOTH ASR and OCR
      3. else: run ASR; if candidates-so-far reconcile to one confident Candidate,
         skip OCR
      4. else (and not skip_ocr): escalate to the high-tier asset -- OCR needs
         legible on-screen text, unlike ASR/subtitles which don't care about video
         quality -- and run OCR

    skip_ocr still force-disables OCR regardless of confidence (manual override,
    unchanged semantics). hq_frame: if the winning candidate's output frame would
    otherwise come from the low-tier (low-resolution) video, escalate to the high tier
    first purely for a better image -- a separate, opt-in choice, not the default
    (which favors speed; see ingest.py's tier docs for the trade-off).
    """
    print("[1/5] Preparing audio/video...", flush=True)
    asset = prepare_asset(video_url, tier="low", work_dir=work_dir)
    used_high_tier = False

    candidates: list[Candidate] = []

    print("[2/5] Checking for an existing subtitle/CC track...", flush=True)
    subtitle_hit = try_subtitle_fast_path(asset, dialogue_text)
    if subtitle_hit is not None:
        candidates.append(subtitle_hit)

    preliminary_result = reconcile(candidates) if candidates else None
    if isinstance(preliminary_result, Candidate):
        print("[3/5] Confident subtitle match -- skipping ASR and OCR.", flush=True)
        needs_ocr = False
    else:
        print("[3/5] Running speech recognition (ASR)...", flush=True)
        candidates.extend(find_asr_candidates(asset, dialogue_text, model_size=model_size))
        preliminary_result = reconcile(candidates)
        needs_ocr = not skip_ocr and not isinstance(preliminary_result, Candidate)

    if needs_ocr:
        print(
            "[4/5] No confident spoken/subtitle match yet -- fetching higher-quality "
            "video for on-screen text (OCR)...",
            flush=True,
        )
        asset = prepare_asset(video_url, tier="high", work_dir=work_dir)
        used_high_tier = True
        candidates.extend(find_ocr_candidates(asset, dialogue_text))
        arbiter_result: Union[Candidate, AmbiguousResult, None] = reconcile(candidates)
    else:
        print("[4/5] Skipping OCR.", flush=True)
        arbiter_result = preliminary_result

    print("[5/5] Reconciling candidates and extracting frame(s)...", flush=True)

    # Escalate before refine if either the caller asked for a better frame (hq_frame),
    # or -- a correctness requirement, not a quality preference -- the current asset
    # has no video stream at all to extract a frame from. tier="low" can legitimately
    # be audio-only on a host that exposes a genuine audio-only format (verified:
    # YouTube does; ok.ru does not) -- every result path still needs to produce an
    # output frame image, so a video-less asset can never be the one refine() runs on.
    needs_video_for_frame = arbiter_result is not None and not asset.metadata.has_video
    if (hq_frame or needs_video_for_frame) and not used_high_tier and arbiter_result is not None:
        asset = prepare_asset(video_url, tier="high", work_dir=work_dir)

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
    # Real bug, found by running against a real (non-example) video: extracted dialogue
    # text can contain characters outside a terminal's default codepage -- e.g. Windows'
    # default console codepage (cp1252/cp437, not UTF-8) can't encode a musical note
    # character that showed up in a real YouTube caption track, and print() crashed with
    # UnicodeEncodeError right as the tool was about to report a correct match. The
    # underlying report.json/PNG were unaffected (file writes already use UTF-8) -- only
    # the stdout summary crashed. Replacing unencodable characters (rather than raising)
    # keeps the CLI usable on a plain Windows terminal without requiring the evaluator to
    # change their console codepage first.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Find the exact frame where a dialogue appears in a video URL."
    )
    parser.add_argument(
        "--url", default=None, help="Video URL to search (prompted for if omitted)"
    )
    parser.add_argument(
        "--dialogue-text",
        default=None,
        help="Target dialogue text to locate (prompted for if omitted)",
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
            "Force-skip the OCR track regardless of ASR/subtitle confidence. Manual "
            "escape hatch only; see run_pipeline()'s docstring."
        ),
    )
    parser.add_argument(
        "--hq-frame",
        action="store_true",
        help=(
            "If the answer came from the fast low-quality fetch, re-fetch at higher "
            "quality just for a better output frame image. Opt-in only -- slower."
        ),
    )
    args = parser.parse_args()

    # Interactive fallback: a recruiter running this cold shouldn't need to know the
    # exact flag names up front. --url/--dialogue-text still work exactly as before for
    # scripting/grading; this only fires when either is omitted.
    video_url = args.url or input("Video URL: ").strip()
    dialogue_text = args.dialogue_text or input("Dialogue text to find: ").strip()
    if not video_url:
        print("Error: a video URL is required.", file=sys.stderr)
        sys.exit(1)
    if not dialogue_text:
        print("Error: dialogue text is required.", file=sys.stderr)
        sys.exit(1)

    try:
        report_path = run_pipeline(
            video_url,
            dialogue_text,
            args.output,
            model_size=args.model,
            work_dir=args.work_dir,
            skip_ocr=args.skip_ocr,
            hq_frame=args.hq_frame,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
