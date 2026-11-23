"""CHIME_DERIVED.md accounts for every docs/evidence entry, so the code-only cut cannot ship one by omission."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cut():
    spec = spec_from_file_location("make_code_only_cut", ROOT / "tools" / "make_code_only_cut.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_evidence_entry_is_withheld_or_marked_not_chime_data():
    cut = _cut()
    text = cut.LIST_FILE.read_text(encoding="utf-8")
    withheld, not_chime = cut.withheld_paths(text), cut.withheld_paths(text, "## Not CHIME data")
    assert not set(withheld) & set(not_chime)
    entries = [f"docs/evidence/{p.name}" + ("/" if p.is_dir() else "")
               for p in sorted((ROOT / "docs" / "evidence").iterdir())
               if p.name not in cut.SKIP_DIRS and not p.name.startswith(".")]
    unlisted = [e for e in entries if e not in withheld + not_chime]
    assert not unlisted, f"list these in CHIME_DERIVED.md under Code-only cut or Not CHIME data: {unlisted}"
    # in this tree every marked folder exists, so a stale entry fails as well
    assert all((ROOT / p).exists() for p in not_chime)
