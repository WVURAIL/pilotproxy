"""The characterize, products, and bench command groups dispatch to their modules."""
from __future__ import annotations

import json

import pytest

from characterization_fixtures import products_dir
from pilot_proxy import cli
from pilot_proxy.characterization import run


@pytest.mark.parametrize("group,steps", [
    ("characterize", {"archive", "histogram-frames", "fine-histogram-frames", "spectrogram", "score-cache"}),
    ("products", {"check-archive"}),
    ("bench", {"estimator-transfer", "transfer-points"}),
])
def test_the_groups_name_their_steps(group, steps):
    table = {"characterize": cli.CHARACTERIZE_STEPS, "products": cli.PRODUCTS_STEPS, "bench": cli.BENCH_STEPS}[group]
    assert set(table) == steps
    for module in table.values():
        assert callable(getattr(__import__(module, fromlist=["main"]), "main"))


@pytest.mark.parametrize("argv", [["characterize", "spectrogram", "--help"], ["characterize", "archive", "--help"],
                                  ["products", "check-archive", "--help"]])
def test_step_help_is_the_steps_own(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 0
    assert "usage: pilot-proxy " + " ".join(argv[:2]) in capsys.readouterr().out


def test_characterize_archive_from_the_command_line(tmp_path, capsys):
    products = products_dir(tmp_path)
    out = tmp_path / "out"
    assert cli.main(["characterize", "archive", "--products", str(products), "--out", str(out),
                     "--campaign-last-month", "2024-12"]) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{"):])
    assert summary["channels"] == [29, 34]
    assert (out / "characterization" / "products-preflight.json").is_file()
    ledger = json.loads((out / "characterization" / "ledger" / "run.json").read_text())
    assert ledger["campaign_last_month"] == "2024-12"


@pytest.mark.parametrize("label", [
    "2026-00", "2026-13", "2026-1", "2026-01-01", "0000-01",
    " 2026-01", "2026-01 ", "",
])
def test_invalid_campaign_month_is_refused_before_writing_output(tmp_path, label):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="campaign.last.month.*YYYY-MM"):
        cli.main(["characterize", "archive", "--products", str(tmp_path / "products"),
                  "--out", str(out), "--campaign-last-month", label])
    assert not out.exists()


@pytest.mark.parametrize("label,index", [
    ("0001-01", 12), ("2026-01", 24312), ("2026-12", 24323),
    ("9999-12", 119999),
])
def test_campaign_month_keeps_the_requested_calendar_month(label, index):
    assert run._month_index(label) == index


def test_an_unknown_step_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["characterize", "verdict"])
