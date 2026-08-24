"""
Phase 6 - CLI entry point.

Wires: ingest -> (subtitle fast-path | ASR + OCR tracks) -> arbiter -> refine -> report

Usage:
    python -m src.main --url <video_url> --dialogue-text "<target phrase>" --output <dir>

Verification (see PHASES.md): pytest tests/test_end_to_end.py::test_full_run
"""

import argparse


def run_pipeline(video_url: str, dialogue_text: str, output_dir: str) -> str:
    """
    Callable entry point (used by tests, wrapped by main() for CLI use).

    Runs ingest -> subtitle fast-path -> ASR/OCR tracks -> arbiter -> refine -> report.
    Returns the path to the written report.json.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(
        description="Find the exact frame where a dialogue appears in a video URL."
    )
    parser.add_argument("--url", required=True, help="Video URL to search")
    parser.add_argument(
        "--dialogue-text", required=True, help="Target dialogue text to locate"
    )
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()

    report_path = run_pipeline(args.url, args.dialogue_text, args.output)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
