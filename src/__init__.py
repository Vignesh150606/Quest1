"""
Import-order guard (Phase 3).

paddle and ctranslate2 (faster-whisper's backend) each ship their own copy of
libiomp5md.dll (the Intel OpenMP runtime) on Windows. Whichever loads first wins the
process; importing paddle first makes ctranslate2 fail with
`OSError: [WinError 127] The specified procedure could not be found` (verified in
Phase 0 -- see prompts.txt). The reverse order works. Phase 6 imports both
faster_whisper and paddleocr in the same process, so this must be resolved before any
src module can safely import paddle/paddleocr.

Importing ctranslate2 here, before anything else in this package, guarantees the safe
order for every `from src.X import Y` regardless of which submodule triggers it first.
"""

import ctranslate2  # noqa: F401  -- MUST precede any paddle/paddleocr import
