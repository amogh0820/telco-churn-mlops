"""Build notebooks/*.ipynb from their `# %%` percent-format .py sources.

Keeping the notebook's source as plain Python means cells are syntax-checked by
the linter and diffs are reviewable. This script regenerates the .ipynb.

Run:  python -m scripts.build_notebook
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def split_cells(source: str) -> list[tuple[str, str]]:
    """Return [(kind, body), ...] where kind is 'code' or 'markdown'."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buffer: list[str] = []

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buffer:
                cells.append((kind, "\n".join(buffer).strip("\n")))
                buffer = []
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        buffer.append(line)

    if buffer:
        cells.append((kind, "\n".join(buffer).strip("\n")))

    return [(k, b) for k, b in cells if b.strip()]


def strip_comment_prefix(body: str) -> str:
    """Turn a block of `# text` lines into plain markdown."""
    out = []
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.append(stripped[1:].removeprefix(" "))
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def to_notebook(source: str) -> dict:
    cells = []
    for kind, body in split_cells(source):
        text = strip_comment_prefix(body) if kind == "markdown" else body
        lines = [f"{line}\n" for line in text.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": lines,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    built = 0
    for py_path in sorted(NOTEBOOK_DIR.glob("*.py")):
        source = py_path.read_text()
        # Drop the module docstring, which is a note to maintainers, not a cell.
        if source.lstrip().startswith('"""'):
            source = source.split('"""', 2)[2].lstrip("\n")
        notebook = to_notebook(source)
        out_path = py_path.with_suffix(".ipynb")
        out_path.write_text(json.dumps(notebook, indent=1) + "\n")
        print(f"{py_path.name} -> {out_path.name} ({len(notebook['cells'])} cells)")
        built += 1
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
