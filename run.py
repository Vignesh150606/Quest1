#!/usr/bin/env python3
"""
One-command local setup + launch for the web app: creates/updates the Python venv,
installs backend deps, installs frontend deps, then starts the FastAPI backend and
the Next.js dev server together. Ctrl+C stops both.

This exists purely as a convenience wrapper around the steps documented in README.md's
"Local development" section -- it doesn't change what runs, just automates typing it.
The CLI (`python -m src.main`) and the manual two-terminal flow both still work exactly
as before; this is an additional way to start the same two processes, not a
replacement for either.

Usage:
    python run.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
WEB_DIR = ROOT / "web"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd, cwd=None) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_venv() -> None:
    if venv_python().exists():
        return
    print("Creating virtual environment (.venv)...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_backend_deps() -> None:
    # pip is a fast no-op when everything's already satisfied, so it's safe to always
    # run this rather than trying to detect "already installed" ourselves.
    print("Checking backend dependencies...")
    run([str(venv_python()), "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)


def ensure_frontend_env_file() -> None:
    env_local = WEB_DIR / ".env.local"
    env_example = WEB_DIR / ".env.example"
    if not env_local.exists():
        shutil.copy(env_example, env_local)
        print(f"Created {env_local.relative_to(ROOT)} from .env.example")


def find_npm() -> str:
    npm = shutil.which("npm")
    if npm is None:
        print(
            "ERROR: npm not found on PATH. Install Node.js (https://nodejs.org/) first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return npm


def install_frontend_deps(npm: str) -> None:
    # npm skips reinstalling anything already satisfied by node_modules/package-lock.json,
    # so -- same reasoning as pip above -- it's safe to always run this.
    print("Checking frontend dependencies...")
    run([npm, "install"], cwd=WEB_DIR)


def main() -> None:
    ensure_venv()
    install_backend_deps()
    ensure_frontend_env_file()
    npm = find_npm()
    install_frontend_deps(npm)

    print()
    print("Starting backend  -> http://127.0.0.1:8000")
    print("Starting frontend -> http://localhost:3000")
    print("(Ctrl+C stops both)")
    print()

    backend = subprocess.Popen(
        [
            str(venv_python()),
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=ROOT,
    )
    # shell=True on Windows: npm resolves to npm.cmd there, which subprocess.Popen
    # can only launch reliably through the shell -- verified directly (without it,
    # Windows raises FileNotFoundError even though shutil.which finds npm.cmd fine).
    frontend = subprocess.Popen([npm, "run", "dev"], cwd=WEB_DIR, shell=(os.name == "nt"))

    try:
        while True:
            time.sleep(1)
            if backend.poll() is not None or frontend.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for proc in (backend, frontend):
            if proc.poll() is None:
                proc.terminate()
        for proc in (backend, frontend):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
