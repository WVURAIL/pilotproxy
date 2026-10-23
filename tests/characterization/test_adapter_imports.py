"""No general layer (characterization, products) imports the narrowband-marker detector directly."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "pilot_proxy"
ADAPTER = "pilot_proxy.detectors.narrowband_marker"


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def test_no_general_layer_imports_the_detector_adapter():
    found = {}
    for package in ("characterization", "products"):
        for path in sorted((ROOT / package).rglob("*.py")):
            hits = {m for m in _imports(path) if m == ADAPTER or m.startswith(ADAPTER + ".")}
            if hits:
                found[str(path.relative_to(ROOT))] = sorted(hits)
    assert found == {}
