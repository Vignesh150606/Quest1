"""
Phase 0 - Environment sanity. Extended in Phase 3 with an import-order regression test.

This is the one test file that's real from the start, not a stub -- its whole job is
to confirm the toolchain installs and imports cleanly before any pipeline code exists.

Verification target: pytest tests/test_environment.py::test_toolchain
"""

import shutil
import subprocess
import sys


def test_toolchain():
    import faster_whisper  # noqa: F401
    import paddleocr  # noqa: F401
    import av  # noqa: F401
    import rapidfuzz  # noqa: F401

    assert shutil.which("ffmpeg") is not None, "ffmpeg not found on PATH"
    assert shutil.which("ffprobe") is not None, "ffprobe not found on PATH"

    for tool in ("ffmpeg", "ffprobe"):
        result = subprocess.run([tool, "-version"], capture_output=True)
        assert result.returncode == 0, f"{tool} -version failed"


def test_paddle_ctranslate2_import_order_is_safe():
    """
    paddle and ctranslate2 (faster-whisper's backend) each ship their own
    libiomp5md.dll on Windows; importing paddle first makes ctranslate2 fail with
    OSError: [WinError 127] (verified in Phase 0 -- see prompts.txt). src/__init__.py
    guards against this by importing ctranslate2 before anything else in the package.

    Runs in a fresh subprocess (not just `import src` in-process) so this actually
    proves the guard works for a cold interpreter -- the failure mode is order-of-first-
    DLL-load, which a subprocess reproduces faithfully and an already-warm test process
    might not.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import src; import paddleocr; import faster_whisper"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import order guard failed (exit {result.returncode}):\n{result.stderr}"
    )
