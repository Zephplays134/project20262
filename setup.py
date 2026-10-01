"""Developer setup for Brahma Evo.

Creates/uses a project-local `.venv`, installs requirements into it, and pulls
the Playwright browsers. Safe to re-run: steps that are already satisfied are
skipped, so it is cheap to invoke after pulling changes.

Usage:
    python setup.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"

# Windows resolves a bare "python" to the Microsoft Store alias on some machines,
# which exits 9009 without running anything. Detect that so the error message
# points at the real problem instead of a confusing "not recognized".
def _is_store_alias(exe: str) -> bool:
    return Path(exe).resolve() == Path(
        os.environ.get("LOCALAPPDATA", ""),
    ) / "Microsoft" / "WindowsApps" / "python.exe"


def venv_python() -> Path | None:
    """Return the interpreter inside `.venv`, or None if there isn't one yet."""
    candidate = (
        VENV_DIR / "Scripts" / "python.exe"
        if os.name == "nt"
        else VENV_DIR / "bin" / "python"
    )
    return candidate if candidate.exists() else None


def check_disk_space(minimum_gb: float = 2.0) -> None:
    """Fail early and clearly if there isn't room to build the venv.

    A full disk makes `python -m venv` fail deep inside `ensurepip` with
    `OSError: [WinError 112] There is not enough space on the disk`, which says
    nothing about what to do next. The venv plus PyQt6, opencv, mediapipe and
    Playwright needs a couple of GB, so check up front and say so plainly.
    """
    try:
        usage = shutil.disk_usage(ROOT)
    except OSError:
        return  # Can't measure; let the real operation report the problem.

    free_gb = usage.free / (1024**3)
    if free_gb < minimum_gb:
        raise SystemExit(
            f"ERROR: not enough free disk space ({free_gb:.2f} GB free, "
            f"{minimum_gb:.0f} GB required).\n"
            "Free up space and re-run. On Windows you can reclaim space with:\n"
            "    cleanmgr\n"
            "or by removing old files under %TEMP%."
        )
    print(f"  -> Disk space OK ({free_gb:.1f} GB free)")


def run(args: list[str], what: str) -> None:
    print(f"  -> {what}")
    # check=True so a failed step stops the script instead of cascading into a
    # half-configured environment. The previous version used check=True only for
    # pip and let the Playwright step fail silently.
    result = subprocess.run(args, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"ERROR: {what} failed (exit code {result.returncode}).")


def main() -> None:
    if _is_store_alias(sys.executable):
        print(
            "ERROR: 'python' resolves to the Microsoft Store alias.\n"
            "Install Python from python.org and tick 'Add python.exe to PATH', "
            "or re-run using a full path to a real interpreter."
        )
        raise SystemExit(1)

    check_disk_space()

    interpreter = venv_python()
    if interpreter is None:
        print(f"Creating virtual environment in {VENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        interpreter = venv_python()
        if interpreter is None:
            raise SystemExit("ERROR: virtual environment creation failed.")
        print("Virtual environment created.")
    else:
        print(f"Reusing existing virtual environment: {interpreter}")

    py = str(interpreter)
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading pip")
    run([py, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing requirements.txt")
    run([py, "-m", "playwright", "install"],
        "Installing Playwright browsers")

    # Fail loudly here rather than at first launch with a bare ImportError.
    print("Verifying the environment ...")
    result = subprocess.run(
        [py, "-c", "import main"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout or "")
        print(result.stderr or "")
        raise SystemExit(
            "ERROR: the app failed to import after install. See the traceback above."
        )

    print("\nSetup complete. Start the app with:")
    print(f'    "{py}" main.py')
    print("or just run start_brahma.bat")


if __name__ == "__main__":
    main()