"""Write ``constraints.txt``: the exact package versions of a tested environment.

Run it from the venv that just passed the full test suite and evaluation:

    python scripts/freeze_constraints.py

pip *constraints* (``pip install -c constraints.txt ...``) only pin packages
that actually get installed, so one file serves the Docker image, CI and dev
setups. Local version labels (``torch==2.11.0+cu128``) are stripped and URL
installs (the spaCy model) are skipped, so the file also resolves on Linux.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import re
import subprocess
import sys
from pathlib import Path

HEADER = """\
# Exact versions verified together on {date} (Python {python}, {system}).
# Used as pip constraints, e.g. pip install -c constraints.txt ".[dev]"; only packages that
# get installed are pinned. Some pins (numpy 2.5) need Python >= 3.12.
# Regenerate after upgrading and re-testing: python scripts/freeze_constraints.py
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("constraints.txt"))
    args = parser.parse_args()

    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pins = sorted(
        (re.sub(r"\+[\w.]+$", "", line) for line in frozen if "==" in line and " @ " not in line),
        key=str.lower,
    )
    header = HEADER.format(
        date=dt.date.today().isoformat(),
        python=platform.python_version(),
        system=platform.system(),
    )
    args.out.write_text(header + "\n".join(pins) + "\n", encoding="utf-8")
    print(f"{len(pins)} pins -> {args.out}")


if __name__ == "__main__":
    main()
