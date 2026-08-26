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

from src.types import AmbiguousResult, Candidate

# Per-modality confidence floor below which a candidate is discarded outright, before
# clustering. Deliberately loose (no ML calibration model, per CLAUDE.md) -- these
# reject only candidates with almost no signal, not a precision instrument. ASR's floor
# is lower than subtitle/OCR's because faster-whisper's mean word probability over a
# short window runs lower than a clean OCR line or an authored subtitle cue even on a
# genuinely correct match.
CONFIDENCE_THRESHOLDS = {"subtitle": 0.5, "asr": 0.4, "ocr": 0.5}
SIMILARITY_THRESHOLD = 0.85
CLUSTER_TOLERANCE_S = 2.0


def reconcile(
    candidates: list[Candidate], *, tolerance_s: float = CLUSTER_TOLERANCE_S
) -> Optional[Union[Candidate, AmbiguousResult]]:
    """
    Apply the deterministic reconciliation policy described in the module docstring.

    Clustering is chained: a candidate need only be within tolerance_s of its nearest
    neighbor in time to join a cluster, not within tolerance_s of every member -- same
    convention as the ASR/OCR tracks' own dedup/clustering.

    Within a cluster, the winner ranks by (confidence, similarity) descending --
    confidence first, similarity as the tiebreaker (see PHASES_1_7_PLAN.md: this is
    what the scaffold's "higher_confidence" test case actually exercises, despite both
    of its candidates sharing the same confidence). This can shift the reported onset
    by up to tolerance_s versus the earliest candidate in the cluster; CLAUDE.md's
    "prefer the higher-confidence candidate" is followed as specified -- see
    APPROACH.md for that trade-off against the literal "first appears" wording.

    Zero surviving candidates -> None. One surviving cluster -> its winner, returned
    directly as a Candidate. More than one surviving cluster -> AmbiguousResult
    carrying each cluster's winner -- never a silent pick between genuinely
    disagreeing modalities.
    """
    survivors = [c for c in candidates if _passes_thresholds(c)]
    if not survivors:
        return None

    clusters = _cluster_by_timestamp(survivors, tolerance_s)
    winners = [_best_in_cluster(cluster) for cluster in clusters]

    if len(winners) == 1:
        return winners[0]

    return AmbiguousResult(
        candidates=winners,
        reason=(
            f"{len(winners)} candidate clusters disagree by more than {tolerance_s}s "
            f"-- modalities point to different moments in the video, not noisy "
            f"variants of the same one."
        ),
    )


def _passes_thresholds(candidate: Candidate) -> bool:
    if candidate.similarity < SIMILARITY_THRESHOLD:
        return False
    if candidate.confidence is None:
        # No confidence signal to reject on -- treat as passing rather than penalizing
        # a modality for not providing one. In practice all three tracks always set it.
        return True
    threshold = CONFIDENCE_THRESHOLDS.get(candidate.modality, 0.5)
    return candidate.confidence >= threshold


def _cluster_by_timestamp(candidates: list[Candidate], tolerance_s: float) -> list[list[Candidate]]:
    ordered = sorted(candidates, key=lambda c: c.timestamp)
    clusters = [[ordered[0]]]
    for c in ordered[1:]:
        if c.timestamp - clusters[-1][-1].timestamp <= tolerance_s:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return clusters


def _best_in_cluster(cluster: list[Candidate]) -> Candidate:
    return max(
        cluster,
        key=lambda c: (c.confidence if c.confidence is not None else 0.0, c.similarity),
    )
