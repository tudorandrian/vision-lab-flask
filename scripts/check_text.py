"""Fail if any tracked text file contains an em dash (U+2014). House style: plain hyphens."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EM_DASH = "\u2014"
BINARY = {".jpg", ".jpeg", ".png", ".ico", ".pt", ".pth", ".lock"}


def main() -> int:
    listing = subprocess.run(
        ["git", "ls-files"],  # noqa: S607 - git is resolved from PATH on purpose
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = listing.stdout.splitlines()
    offences = []
    for name in tracked:
        path = Path(name)
        if path.suffix.lower() in BINARY or "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line:
                offences.append(f"{name}:{number}")
    if offences:
        print("em dash found, use a hyphen instead:", *offences, sep="\n  ")
        return 1
    print(f"checked {len(tracked)} tracked files, no em dash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
