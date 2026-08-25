#!/usr/bin/env python3
"""
Per-phase verification runner.

Usage:
    python verify.py <phase_number>
    python verify.py all

Each phase's test starts marked @pytest.mark.skip until you implement that phase's
src module. Remove the skip decorator for the phase you just built, then run
`python verify.py N` for that phase specifically. Running a still-skipped phase will
report "skipped", not pass/fail -- that's expected, not a bug in this script.

Phase 7 (Packaging) isn't a pytest target -- it's verified manually via the docker
build/run command in PHASE_CHECKLIST.md.
"""

import subprocess
import sys

PHASE_TESTS = {
    0: "tests/test_environment.py",
    1: "tests/test_ingest.py",
    2: "tests/test_asr_track.py",
    3: "tests/test_ocr_track.py",
    4: "tests/test_arbiter.py",
    5: "tests/test_refine.py::test_frame_accuracy",
    6: "tests/test_end_to_end.py",
}


def run_phase(n: int) -> bool:
    if n == 7:
        print(
            "Phase 7 is verified manually -- see PHASE_CHECKLIST.md for the "
            "docker build/run command."
        )
        return True

    target = PHASE_TESTS.get(n)
    if target is None:
        print(f"No phase {n}")
        return False

    print(f"\n=== Verifying Phase {n}: {target} ===")
    result = subprocess.run(["pytest", target, "-v"])
    ok = result.returncode == 0
    print(f"Phase {n}: exit code {result.returncode} "
          f"(0 = every collected test passed or was skipped as expected)")
    return ok


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify.py <phase_number>|all")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        results = {n: run_phase(n) for n in list(PHASE_TESTS) + [7]}
        print("\n=== Summary ===")
        for n in sorted(results):
            print(f"Phase {n}: {'OK' if results[n] else 'FAIL'}")
        sys.exit(0 if all(results.values()) else 1)
    else:
        sys.exit(0 if run_phase(int(arg)) else 1)


if __name__ == "__main__":
    main()
