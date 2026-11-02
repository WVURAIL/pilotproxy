"""The frame export of a control band. The role of the band is set by the profile, and no threshold is drawn."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.characterization import histogram_frames as export

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = _load("v5_fixture_control", ROOT / "tests/products/test_pilotproxy_v5.py")
CONTROL_FREQ_ID = 491
CONTROL_PILOT_HZ = 608.309441e6


def _profile():
    project = pytest.importorskip("pilot_proxy.config.project")
    return project.load_project(project.default_project_dir())


def _plan(roles):
    plan = pytest.importorskip("pilot_proxy.config.frequency_plan")
    bands = [{"label": str(ch), "low_mhz": 470 + 6 * (ch - 14), "high_mhz": 476 + 6 * (ch - 14), "role": role,
              **({"target_freq_id": CONTROL_FREQ_ID} if role == "control" else {})} for ch, role in roles]
    frequency_plan = plan.frequency_plan_from_mapping({"name": "test", "bands": bands}, where="test")
    return SimpleNamespace(frequency_plan=frequency_plan, target_freq_id=lambda band: band.target_freq_id)


FORBIDDEN_KEY = re.compile(r"(^|_)eta(_|$)|threshold|toleran|verdict|excis|ruling|undetermined|keep")


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _control_product(tmp_path, monkeypatch, channel=37, freq_id=CONTROL_FREQ_ID):
    real = fixture.products.freq_id
    monkeypatch.setattr(fixture.products, "freq_id", lambda ch: freq_id if ch == channel else real(ch))
    path = fixture._write_product(tmp_path / f"{freq_id}.npz", channel, frames=120, units=12)
    fixture._replace(path, baseband_power_linear=np.full((120, 1), 4.0))
    return path


def test_the_profile_declares_channel_37_the_control_read_at_491():
    project = _profile()
    assert project.frequency_plan.labels("control") == ("37",)
    band = export.control_band(project, 37, CONTROL_FREQ_ID, CONTROL_PILOT_HZ)
    assert (band.label, band.role, band.target_freq_id) == ("37", "control", CONTROL_FREQ_ID)
    with pytest.raises(ValueError, match="not control"):
        export.control_band(project, 33, 552, 584.309441e6)
    with pytest.raises(ValueError, match="freq_id 491"):
        export.control_band(project, 37, 490, CONTROL_PILOT_HZ)
    with pytest.raises(ValueError, match="outside band"):
        export.control_band(project, 37, CONTROL_FREQ_ID, 614.5e6)
    with pytest.raises(ValueError, match="not a band"):
        export.control_band(project, 38, 500, 614.309441e6)


def test_the_role_comes_from_the_plan_not_the_channel_number():
    screened = _plan([(36, "screened"), (37, "screened")])
    with pytest.raises(ValueError, match="role 'screened'"):
        export.control_band(screened, 37, CONTROL_FREQ_ID, CONTROL_PILOT_HZ)
    moved = _plan([(36, "control"), (37, "screened")])
    assert export.control_band(moved, 36, CONTROL_FREQ_ID, 602.309441e6).label == "36"


def test_the_control_period_spans_the_selected_frames_months_only():
    month = np.array([-1, 24240, 24240, 24243, 24250, 24251])
    selected = np.array([True, True, False, True, True, False])
    finite = np.array([False, True, True, False, True, True])
    units = np.array([0, 1, 1, 2, 3, 4])
    period = export.control_period(month, selected, finite, units)
    assert (period["first_month"], period["last_month"]) == ("2020-01", "2020-11")
    assert period["months"] == ["2020-01", "2020-04", "2020-11"]
    assert (period["frames"], period["frames_without_time"], period["units"]) == (3, 1, 3)
    assert (period["era"], period["state"], period["evidence"]) == (1, "control", "archive start")
    with pytest.raises(ValueError, match="no health-selected"):
        export.control_period(month, np.zeros(6, dtype=bool), finite, units)


def test_a_screened_product_is_refused_as_a_control(tmp_path):
    _profile()
    product = fixture._write_product(tmp_path / "552.npz", 33, frames=12, units=4)
    with pytest.raises(ValueError, match="not control"):
        export.run_control(product, tmp_path / "frames")


def test_the_control_export_draws_no_threshold(tmp_path, monkeypatch):
    _profile()
    product = _control_product(tmp_path, monkeypatch)
    frames = tmp_path / "frames"
    manifest = export.run_control(product, frames)
    assert manifest["role"] == "control" and [c["channel"] for c in manifest["channels"]] == [37]
    metadata = json.loads((frames / "ch37.json").read_text())
    assert (metadata["role"], metadata["freq_id"], metadata["band"]["target_freq_id"]) == ("control", 491, 491)
    assert len(metadata["era_definitions"]) == 1 and metadata["current_era"] == 1
    assert not [key for key in _keys(metadata) if FORBIDDEN_KEY.search(key.lower())]
    text = json.dumps(metadata).lower()
    for word in ("toleran", "verdict", "excis", "ruling", "undetermined", "ledger"):
        assert word not in text, word
    with np.load(frames / "ch37.npz", allow_pickle=False) as z:
        eligible, era = z["histogram_eligible"], z["era_id"]
        np.testing.assert_array_equal(z["current_histogram_eligible"], eligible & (era == 1))
        assert np.all(era[z["membership_month"] >= 0] == 1)
    assert metadata["counts"]["current_histogram_eligible"] == int(eligible.sum())
