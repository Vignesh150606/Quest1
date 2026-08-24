"""
Phase 0 - Environment sanity.

This is the one test file that's real from the start, not a stub -- its whole job is
to confirm the toolchain installs and imports cleanly before any pipeline code exists.

Verification target: pytest tests/test_environment.py::test_toolchain
"""

import shutil
import subprocess


def test_toolchain():
    import faster_whisper  # noqa: F401
    import paddleocr  # noqa: F401
    import videocr  # noqa: F401
    import av  # noqa: F401
    import rapidfuzz  # noqa: F401

    assert shutil.which("ffmpeg") is not None, "ffmpeg not found on PATH"
    assert shutil.which("ffprobe") is not None, "ffprobe not found on PATH"

    for tool in ("ffmpeg", "ffprobe"):
        result = subprocess.run([tool, "-version"], capture_output=True)
        assert result.returncode == 0, f"{tool} -version failed"
