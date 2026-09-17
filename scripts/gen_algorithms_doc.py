"""Write docs/algorithms.md from vision_lab.catalog. Run with --check in CI to detect drift."""

from __future__ import annotations

import sys
from pathlib import Path

from vision_lab.catalog import CATALOG

TARGET = Path(__file__).resolve().parent.parent / "docs" / "algorithms.md"


def render() -> str:
    lines = [
        "# Algorithms and models",
        "",
        "Generated from `src/vision_lab/catalog.py` by `scripts/gen_algorithms_doc.py`.",
        "Edit the catalogue, not this file. The same text appears in the app under Algorithms.",
        "",
    ]
    for group in CATALOG:
        lines += [
            f"## {group.title}",
            "",
            "| Name | What it does | Reference |",
            "| --- | --- | --- |",
        ]
        lines += [f"| {e.title} | {e.summary} | {e.reference} |" for e in group.entries]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != text:
            print("docs/algorithms.md is out of date: run python scripts/gen_algorithms_doc.py")
            return 1
        return 0
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
