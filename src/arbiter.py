"""
Phase 4 - Arbiter.

Consumes: list[Candidate] from the subtitle fast-path (Phase 1), ASR track (Phase 2),
          and OCR track (Phase 3)
Produces: Candidate | AmbiguousResult | None

Deterministic policy (see CLAUDE.md -- no ML calibration model):
  1. Reject candidates below a per-modality confidence threshold.
  2. Cluster remaining candidates that fall within a temporal tolerance window.
  3. Within a cluster, prefer the higher-confidence candidate.
  4. If modalities disagree beyond the tolerance window, return AmbiguousResult --
     never silently pick one and hide the disagreement.

Verification (see PHASES.md): pytest tests/test_arbiter.py::test_reconciliation_policy
"""

from typing import Optional, Union

from src.types import Candidate, AmbiguousResult


def reconcile(candidates: list[Candidate]) -> Optional[Union[Candidate, AmbiguousResult]]:
    """Apply the deterministic reconciliation policy described above."""
    raise NotImplementedError
