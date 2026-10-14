from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_specs_use_current_schema_and_canonical_geometry() -> None:
    generator = _load_script(
        "generate_doc_specs", ROOT / "tools" / "generate_doc_specs.py"
    )
    rendered = generator.render_specs()

    assert r"pilotproxy\_per\_pilot\_product\_v5" in rendered
    assert r"pilotproxy\_detector\_datatrawl" not in rendered
    assert r"\newcommand{\ppDetectorWindowSamples}{128}" in rendered
    assert r"\newcommand{\ppFineNumBins}{256}" in rendered
    assert r"\newcommand{\ppChimeStreams}{2048}" in rendered


def test_doc_specs_check_is_non_mutating() -> None:
    output_path = ROOT / "docs" / "generated" / "specs.tex"
    before = output_path.read_bytes() if output_path.exists() else None
    subprocess.run(
        [sys.executable, "tools/generate_doc_specs.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = output_path.read_bytes() if output_path.exists() else None
    assert after == before


def test_vocabulary_check_normalizes_tex_escaped_identifiers() -> None:
    checker = _load_script(
        "check_current_product_vocabulary",
        ROOT / "scripts" / "check_current_product_vocabulary.py",
    )
    normalized = checker.searchable_text(
        r"pilotproxy\_detector\_datatrawl\_v3"
    )
    assert "pilotproxy_detector_datatrawl_v3" in normalized
