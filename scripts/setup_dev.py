#!/usr/bin/env python3
"""
Petal Project Bootstrap Script
-----------------------------
Sets up a world-class Python development environment for new contributors.
- Creates a .venv (Python 3.11)
- Installs uv, pre-commit, and dev dependencies
- Installs pre-commit hooks
- Runs pip-audit and pytest
- Prints next steps
"""
import os
import shutil
import subprocess
import sys
from typing import Any

VENV_DIR = ".venv"
PYTHON_VERSION = "3.11"


def run(cmd: str, check: bool = True, shell: bool = True, **kwargs: Any) -> None:
    print(f"\033[94m$ {cmd}\033[0m")
    try:
        subprocess.run(cmd, check=check, shell=shell, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"\033[91mError: Command failed: {cmd}\033[0m")
        sys.exit(e.returncode)


def find_python311() -> str:
    # Try to find python3.11 on PATH
    candidates = [
        shutil.which("python3.11"),
        shutil.which("py"),
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for exe in candidates:
        if not exe:
            continue
        try:
            out = subprocess.check_output([exe, "--version"], text=True)
            if PYTHON_VERSION in out:
                return exe
        except Exception:
            continue
    print(
        f"\033[91mPython {PYTHON_VERSION} not found. Please install it and try again.\033[0m"
    )
    sys.exit(1)


def main() -> None:
    # 1. Create venv if not exists
    if not os.path.isdir(VENV_DIR):
        python = find_python311()
        print(f"Creating virtual environment with {python}...")
        run(f'"{python}" -m venv {VENV_DIR}')
    else:
        print(f"Virtual environment '{VENV_DIR}' already exists.")

    # 2. Determine venv bin path
    if os.name == "nt":
        venv_bin = os.path.join(VENV_DIR, "Scripts")
    else:
        venv_bin = os.path.join(VENV_DIR, "bin")

    pip = os.path.join(venv_bin, "pip")
    python = os.path.join(venv_bin, "python")

    # 3. Upgrade pip, install uv and pre-commit
    run(f'"{pip}" install --upgrade pip')
    run(f'"{pip}" install uv pre-commit')

    # 4. Install project dependencies (editable + dev)
    run(f'"{venv_bin}/uv" pip install -e .[dev]')

    # 5. Install pre-commit hooks
    run(f'"{venv_bin}/pre-commit" install')

    # 6. Run pip-audit and pytest as a check
    print("\nRunning pip-audit...")
    run(f'"{venv_bin}/pip-audit"', check=False)
    print("\nRunning pytest...")
    run(f'"{venv_bin}/pytest"', check=False)

    # 7. Print next steps
    print(
        """
\033[92m✅ Development environment is ready!\033[0m

Activate your virtual environment:
  Windows PowerShell: .venv\Scripts\Activate.ps1
  Windows CMD:        .venv\Scripts\activate.bat
  Mac/Linux:          source .venv/bin/activate

- Pre-commit hooks are installed and will run on every commit.
- Run 'pytest' to test, 'pip-audit' to check security, and 'uv pip install -e .[dev]' to update deps.
- See README.md for more info.
"""
    )


if __name__ == "__main__":
    main()
