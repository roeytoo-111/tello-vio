"""Static analysis over every Python file that ships.

Exists because of a real escape: `np.array(...)` was written in the driver,
which imports `numpy` (not `numpy as np`). `python -m py_compile` passed --
it only checks syntax -- so the package built, installed and launched, and
died with NameError at construction time, on the drone, with the user
watching.

pyflakes catches undefined names without importing the module, which matters
here: the ROS nodes cannot be imported in a bare pytest run (no rclpy context),
so import-time checking is not available to us. This is the cheapest guard that
would have caught it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
ROOTS = [REPO / "workspace" / "src", REPO / "scripts", REPO / "docs"]


def _python_files():
    files = []
    for root in ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            parts = set(f.parts)
            if parts & {"build", "install", "log", "__pycache__", ".pydeps"}:
                continue
            files.append(f)
    return sorted(files)


def test_there_are_files_to_check():
    files = _python_files()
    assert len(files) > 20, f"only found {len(files)} python files; check ROOTS"


def test_no_undefined_names_or_unused_imports():
    files = _python_files()
    if not files:
        pytest.skip("no python files found")
    try:
        proc = subprocess.run([sys.executable, "-m", "pyflakes", *map(str, files)],
                              capture_output=True, text=True, timeout=120)
    except FileNotFoundError:                       # pragma: no cover
        pytest.skip("pyflakes not installed")
    if proc.returncode == 1 and "No module named" in proc.stderr:
        pytest.skip("pyflakes not installed")
    assert proc.returncode == 0, (
        "pyflakes found problems:\n" + proc.stdout + proc.stderr)


def test_driver_does_not_use_an_undefined_numpy_alias():
    """Pin the specific escape: the driver imports `numpy`, not `numpy as np`."""
    src = (REPO / "workspace" / "src" / "tello" / "tello" / "node.py")
    if not src.exists():
        pytest.skip("driver not found")
    text = src.read_text()
    assert "import numpy\n" in text
    assert "import numpy as np" not in text, \
        "if you add the alias, drop this test -- but do it deliberately"
    # Ignore the one comment that quotes djitellopy's own source verbatim.
    code = [l for l in text.splitlines()
            if "np." in l and not l.strip().startswith("#")]
    assert not code, f"undefined numpy alias used in code: {code}"
