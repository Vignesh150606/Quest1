"""
Phase 6 - Report.

Consumes: FrameMatch(es) or AmbiguousResult/None (Phase 4/5), VideoAsset (Phase 1)
Produces: report.json, formatted stdout summary (Timestamp / Frame / Text)

Verification (see PHASES.md): pytest tests/test_end_to_end.py::test_full_run
"""

from typing import Union

from src.types import FrameMatch, AmbiguousResult


def write_report(
    result: Union[FrameMatch, AmbiguousResult, None], output_dir: str
) -> str:
    """
    Write report.json and print a formatted stdout summary. Returns the path to
    report.json. `result=None` must produce a status of "not_found", not a crash.
    """
    raise NotImplementedError
