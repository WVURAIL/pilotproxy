"""Capture CLI exit status distinguishes valid replay, mismatch, and refusal."""

import hashlib
import json

import pytest

from pilot_proxy import cli
from pilot_proxy.capture_reference import CHORD_METADATA_SIZE
from test_capture_replay import make_plan, update_raw


def _read(path):
    def refuse(value):
        raise AssertionError(f"nonstandard JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=refuse)


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_public_replay_and_validation_preserve_invalid_support_and_source(tmp_path, capsys):
    plan = make_plan(tmp_path)
    before = _hashes(plan.parent)
    output = tmp_path / "replay"
    assert cli.main(["replay-fine-capture", "--plan", str(plan),
                     "--output-dir", str(output)]) == 0
    assert "invalid_support; 1 invalid rows retained" in capsys.readouterr().out
    report = _read(output / "replay.json")
    assert report["product_agreement"] is True
    assert report["telescope_qualification"] is False
    assert (report["num_blocks"], report["num_frequencies"],
            report["score_rows"], report["invalid_rows"]) == (2, 2, 2, 1)
    rows = _read(output / "scores_CH14.json")["scores"]
    assert [r["valid"] for r in rows] == [True, False]
    assert [r["required_multiplier_q16"] for r in rows] == ["82009", None]
    assert [r["n_bulk"] for r in rows] == [123, 0]
    assert len(_read(output / "population_CH14.json")["scores"]) == 2
    manifest = _read(output / "manifest.json")
    assert manifest["plan_path"] == str(plan.resolve())
    assert manifest["output_files_sha256"] == {
        name: digest for name, digest in _hashes(output).items() if name != "manifest.json"
    }
    published = _hashes(output)
    assert cli.main(["validate-fine-capture-replay", "--replay-dir", str(output)]) == 0
    assert "verified: invalid_support" in capsys.readouterr().out
    assert _hashes(output) == published
    assert _hashes(plan.parent) == before


@pytest.mark.parametrize("kind", ["masks", "powers"])
def test_public_replay_mismatch_returns_failure_and_retains_diagnostics(tmp_path, capsys, kind):
    plan = make_plan(tmp_path)
    update_raw(plan, kind, lambda data: data.__setitem__(
        4 + CHORD_METADATA_SIZE, data[4 + CHORD_METADATA_SIZE] ^ 1,
    ))
    output = tmp_path / "replay"
    with pytest.raises(SystemExit) as status:
        cli.main(["replay-fine-capture", "--plan", str(plan), "--output-dir", str(output)])
    assert status.value.code == 1
    assert "product_mismatch" in capsys.readouterr().out
    report = _read(output / "replay.json")
    assert report["product_agreement"] is False
    assert report["invalid_rows"] == 1
    assert len(_read(output / "scores_CH14.json")["scores"]) == 2
    assert not (output / "population_CH14.json").exists()
    published = _hashes(output)
    with pytest.raises(SystemExit) as status:
        cli.main(["validate-fine-capture-replay", "--replay-dir", str(output)])
    assert status.value.code == 1
    assert "product_mismatch" in capsys.readouterr().out
    assert _hashes(output) == published


def test_public_validation_recomputes_even_after_output_digest_is_refreshed(tmp_path, capsys):
    plan = make_plan(tmp_path)
    output = tmp_path / "replay"
    cli.main(["replay-fine-capture", "--plan", str(plan), "--output-dir", str(output)])
    capsys.readouterr()
    path = output / "scores_CH14.json"
    scores = _read(path)
    scores["scores"][0]["required_multiplier_q16"] = "82010"
    path.write_text(json.dumps(scores))
    manifest_path = output / "manifest.json"
    manifest = _read(manifest_path)
    manifest["output_files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    before = _hashes(output)
    with pytest.raises(SystemExit, match="Replay result mismatch"):
        cli.main(["validate-fine-capture-replay", "--replay-dir", str(output)])
    assert "verified" not in capsys.readouterr().out
    assert _hashes(output) == before


def test_public_replay_refuses_changed_inputs_and_existing_output(tmp_path, capsys):
    plan = make_plan(tmp_path)
    output = tmp_path / "replay"
    raw = plan.parent / "voltage.raw"
    raw.write_bytes(raw.read_bytes() + b"changed")
    with pytest.raises(SystemExit, match="File identity mismatch"):
        cli.main(["replay-fine-capture", "--plan", str(plan), "--output-dir", str(output)])
    assert not output.exists()
    output.mkdir()
    (output / "keep.txt").write_text("prior result")
    with pytest.raises(SystemExit, match="must not exist"):
        cli.main(["replay-fine-capture", "--plan", str(plan), "--output-dir", str(output)])
    assert (output / "keep.txt").read_text() == "prior result"
    assert list(output.iterdir()) == [output / "keep.txt"]
    assert "Replay written" not in capsys.readouterr().out
