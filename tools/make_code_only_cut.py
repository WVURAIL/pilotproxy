"""Write the code-only cut of this repository: the tree without the CHIME-derived files.

    python tools/make_code_only_cut.py OUTPUT_DIR

The withheld paths are the fenced block under "Code-only cut" in CHIME_DERIVED.md (a path
ending in "/" withholds a directory). Files come from `git ls-files` in a git checkout and
from a directory walk otherwise. OUTPUT_DIR must not exist. Every listed path must exist in
the source tree, so a stale list fails instead of silently withholding nothing.

The cut keeps CHIME_DERIVED.md itself, so it still says what was withheld; the tests that
read a withheld file skip.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST_FILE = ROOT / "CHIME_DERIVED.md"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
WITHHELD = "withheld from the code-only cut (CHIME_DERIVED.md)"
# The project profile needs an era list, so the cut writes empty but valid ones.
STAND_INS = {
    "projects/chime_atsc/eras/author_eras_2026-09-23.json": {
        "version": WITHHELD, "criterion": WITHHELD, "channels": {}},
    "projects/chime_atsc/eras/transmitter_off.json": {
        "schema": "pilot_proxy_transmitter_off_v1", "source": WITHHELD, "bands": {}},
}


def withheld_paths(text: str) -> list[str]:
    """The paths of the fenced block that follows the "## Code-only cut" heading."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Code-only cut")
    except StopIteration:
        raise SystemExit(f"{LIST_FILE.name}: no '## Code-only cut' section") from None
    fence = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("```"))
    end = next(i for i in range(fence + 1, len(lines)) if lines[i].startswith("```"))
    return [line.strip() for line in lines[fence + 1:end] if line.strip() and not line.startswith("#")]


def tree_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
        files = [f for f in out.decode().split("\0") if f]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
                  if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts))


def is_withheld(path: str, withheld: list[str]) -> bool:
    return any(path == w or (w.endswith("/") and path.startswith(w)) for w in withheld)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f"{out} exists")
    withheld = withheld_paths(LIST_FILE.read_text(encoding="utf-8"))
    files = tree_files(ROOT)
    missing = [w for w in withheld if not any(f == w or (w.endswith("/") and f.startswith(w)) for f in files)]
    if missing:
        print("listed in CHIME_DERIVED.md but not in the tree:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    kept = dropped = 0
    for rel in files:
        if is_withheld(rel, withheld):
            dropped += 1
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
        kept += 1
    for rel, data in STAND_INS.items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"code-only cut at {out}: {kept} files kept, {dropped} withheld ({len(withheld)} listed paths), {len(STAND_INS)} era stand-ins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
