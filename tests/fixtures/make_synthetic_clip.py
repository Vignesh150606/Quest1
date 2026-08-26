"""
Phase 3 - synthetic captioned clip fixture.

The example video is a spoken-dialogue case with no on-screen caption (see CLAUDE.md's
"Facts Established by Manual Investigation") -- so the OCR track needs its own fixture
that actually exercises it.

Builds a short video with a burned-in caption appearing at a known timestamp.
"""

import os
import subprocess

_FPS = 25
_SIZE = "640x480"
_FONTSIZE = 48
_FALLBACK_FONTFILE = "C\\:/Windows/Fonts/arial.ttf"  # drawtext escaping: ':' -> '\:'


def make_synthetic_clip(output_path: str, text: str, onset_s: float, duration_s: float = 5.0) -> dict:
    """
    Generate a short video at output_path with `text` burned in starting at onset_s.

    The clip runs from t=0 (blank, no caption) to t=onset_s+duration_s (caption visible
    for the final duration_s seconds) -- fixed at 25 fps so a chosen onset_s that's a
    multiple of the frame period lands on an exact frame boundary (e.g. 2.0s -> frame 50).

    Returns: {"path": output_path, "text": text, "onset_s": onset_s, "fps": <fps used>}
    """
    total_duration = onset_s + duration_s
    escaped_text = _escape_drawtext_text(text)

    # font=Arial (relying on the ffmpeg build's bundled fontconfig) is tried first per
    # PHASES_1_7_PLAN.md, but verified to fail on this machine with "Fontconfig error:
    # Cannot load default config file" -- the fallback to an explicit fontfile= is not
    # a hypothetical, it's what actually runs here.
    result = subprocess.run(
        _build_cmd(output_path, escaped_text, total_duration, onset_s, "font=Arial"),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not os.path.exists(output_path):
        fallback = subprocess.run(
            _build_cmd(
                output_path, escaped_text, total_duration, onset_s,
                f"fontfile='{_FALLBACK_FONTFILE}'",
            ),
            capture_output=True,
            text=True,
        )
        if fallback.returncode != 0:
            raise RuntimeError(
                "ffmpeg drawtext failed with both font=Arial and the explicit fontfile "
                f"fallback ({_FALLBACK_FONTFILE}).\n"
                f"font=Arial stderr (tail):\n{result.stderr[-1500:]}\n\n"
                f"fontfile fallback stderr (tail):\n{fallback.stderr[-1500:]}"
            )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg produced an empty/missing clip at {output_path!r}")

    return {"path": output_path, "text": text, "onset_s": onset_s, "fps": float(_FPS)}


def _build_cmd(
    output_path: str, escaped_text: str, total_duration: float, onset_s: float, font_arg: str
) -> list[str]:
    vf = (
        f"drawtext=text='{escaped_text}':{font_arg}:fontcolor=white:fontsize={_FONTSIZE}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:enable='gte(t\\,{onset_s})'"
    )
    return [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={_SIZE}:d={total_duration}:r={_FPS}",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-shortest",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        output_path,
    ]


def _escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )
