"""
Phase 6 - End-to-end tests.

Verification target: pytest tests/test_end_to_end.py::test_full_run
Remove the @pytest.mark.skip decorators below once Phases 1-5 and src/main.py's
run_pipeline() are implemented.
"""

import json
import os

import pytest

EXAMPLE_URL = "https://ok.ru/video/248244667877"
TARGET_PHRASE = "My mind rebels at stagnation"


@pytest.mark.skip(reason="Unskip once Phases 1-5 and run_pipeline() are implemented")
def test_full_run(tmp_path):
    from src.main import run_pipeline

    output_dir = str(tmp_path)
    result_path = run_pipeline(EXAMPLE_URL, TARGET_PHRASE, output_dir)

    with open(result_path) as f:
        report = json.load(f)

    assert report["frame"] is not None
    assert os.path.exists(report["image_path"])


@pytest.mark.skip(reason="Unskip once Phases 1-5 and run_pipeline() are implemented")
def test_no_candidates_found(tmp_path):
    from src.main import run_pipeline

    output_dir = str(tmp_path)
    result_path = run_pipeline(
        EXAMPLE_URL, "this phrase will never appear in this video xyz123", output_dir
    )

    with open(result_path) as f:
        report = json.load(f)

    assert report["status"] == "not_found"
