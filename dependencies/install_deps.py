"""
Install all required Python dependencies for the project.

Usage (from repo root):
    python dependencies/install_deps.py

This installs the pinned versions from info/requirements.txt using the
current interpreter (sys.executable).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    requirements = repo_root / "info" / "requirements.txt"
    if not requirements.exists():
        raise SystemExit(f"requirements.txt not found at {requirements}")

    print(f"Installing dependencies from {requirements} ...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Dependency installation failed with exit code {exc.returncode}") from exc

    print("All dependencies installed successfully.")


if __name__ == "__main__":
    main()
