"""
Phase 1 - Ingest tests.

Verification target: pytest tests/test_ingest.py::test_prepare_asset
Remove the @pytest.mark.skip below once src/ingest.py is implemented.
"""

import os

import pytest

from src.ingest import prepare_asset

OK_RU_URL = "https://ok.ru/video/248244667877"
# TODO: add a second URL from a different domain here (per PHASES.md Phase 1's
# "parametrized over the ok.ru URL + one other domain" requirement) once you've picked one.


@pytest.mark.skip(reason="Unskip once src/ingest.py (Phase 1) is implemented")
@pytest.mark.parametrize("url", [OK_RU_URL])
def test_prepare_asset(url):
    asset = prepare_asset(url)
    assert asset.metadata.fps > 0
    assert asset.metadata.duration_s > 0  # TODO: tighten to expected range (~3261s for the example)
    assert os.path.exists(asset.audio_path)
    assert os.path.getsize(asset.audio_path) > 0
