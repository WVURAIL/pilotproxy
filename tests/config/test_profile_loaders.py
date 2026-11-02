# coding=utf-8
"""Schema and refusal tests for the profile loaders (pilot_proxy.config)."""
from __future__ import annotations

import json
import shutil

import pytest

yaml = pytest.importorskip("yaml")

from pilot_proxy import paths  # noqa: E402
from pilot_proxy.config._files import ProfileError  # noqa: E402
from pilot_proxy.config.detector_config import detector_config_from_mapping  # noqa: E402
from pilot_proxy.config.eras import load_era_list  # noqa: E402
from pilot_proxy.config.frequency_plan import frequency_plan_from_mapping  # noqa: E402
from pilot_proxy.config.integration_model import integration_model_from_mapping  # noqa: E402
from pilot_proxy.config.interference import interference_template_from_mapping  # noqa: E402
from pilot_proxy.config.project import (  # noqa: E402
    default_project,
    default_project_dir,
    load_project,
)
from pilot_proxy.config.register import (  # noqa: E402
    RegisterSideError,
    load_register,
    register_from_mapping,
)

PROJECT_DIR = default_project_dir()


def _plan(*bands):
    return {"name": "test", "bands": list(bands)}


def _band(label, low, high, role="screened", **extra):
    return {"label": label, "low_mhz": low, "high_mhz": high, "role": role, **extra}


def test_default_project_is_the_repository_profile():
    assert PROJECT_DIR == paths.DATA_ROOT / "projects" / "chime_atsc"
    project = default_project().load_all()
    assert project.name == "chime_atsc"
    assert project.instrument.name == "chime"
    assert project.detector_adapter == project.interference.kind == "narrowband_marker"
    assert set(project.file_sha256()) == {
        "project", "frequency_plan", "interference_template", "detector_config",
        "integration_model", "register", "author_eras", "transmitter_off",
    }


def test_plan_roles_and_band_lookup():
    plan = default_project().frequency_plan
    assert plan.labels("control") == ("37",)
    assert plan.labels() == plan.labels("screened") + plan.labels("control")
    with pytest.raises(KeyError):
        plan.band("99")
    with pytest.raises(ValueError):
        plan.bands("guard")


def test_plan_refuses_ambiguous_or_inconsistent_bands():
    ok = frequency_plan_from_mapping(_plan(_band("1", 10, 16)), where="t")
    assert ok.band("1").width_hz == 6e6
    with pytest.raises(ProfileError, match="quoted string"):
        frequency_plan_from_mapping(_plan(_band(14, 470, 476)), where="t")
    with pytest.raises(ProfileError, match="unique"):
        frequency_plan_from_mapping(_plan(_band("1", 10, 16), _band("1", 16, 22)), where="t")
    with pytest.raises(ProfileError, match="overlap"):
        frequency_plan_from_mapping(_plan(_band("1", 10, 16), _band("2", 15, 22)), where="t")
    with pytest.raises(ProfileError, match="exceed"):
        frequency_plan_from_mapping(_plan(_band("1", 16, 10)), where="t")
    with pytest.raises(ProfileError, match="target_freq_id"):
        frequency_plan_from_mapping(_plan(_band("1", 10, 16, role="control")), where="t")
    with pytest.raises(ProfileError, match="role"):
        frequency_plan_from_mapping(_plan(_band("1", 10, 16, role="guard")), where="t")
    with pytest.raises(ProfileError, match="unknown key"):
        frequency_plan_from_mapping(_plan(_band("1", 10, 16, centre_mhz=13)), where="t")


def test_yaml_exponent_without_sign_is_refused_not_misread():
    # PyYAML (YAML 1.1) reads 6.0e6 as the string "6.0e6", and the loader refuses it
    data = yaml.safe_load(
        "kind: narrowband_marker\nband_width_hz: 6.0e6\nresidual_shape: band_thermal\n"
        "reference: r\nmarker: {offset_hz: 1.0, marker_to_band_db: 1.0, "
        "capture_efficiency: 1.0}\n")
    assert data["band_width_hz"] == "6.0e6"
    with pytest.raises(ProfileError, match="must be a number"):
        interference_template_from_mapping(data, where="t")


def test_interference_template_checks_the_marker_block():
    base = {"kind": "narrowband_marker", "band_width_hz": 6000000.0,
            "residual_shape": "band_thermal", "reference": "r",
            "marker": {"offset_hz": 309441.0, "marker_to_band_db": 11.3,
                       "capture_efficiency": 1.0}}
    assert interference_template_from_mapping(base, where="t").marker.offset_hz == 309441.0
    for key, value, message in (
            ("offset_hz", 7.0e6, "inside the band"),
            ("capture_efficiency", 1.5, r"\(0, 1\]"),
            ("capture_efficiency", 0.0, "> 0")):
        bad = json.loads(json.dumps(base))
        bad["marker"][key] = value
        with pytest.raises(ProfileError, match=message):
            interference_template_from_mapping(bad, where="t")
    missing = {k: v for k, v in base.items() if k != "marker"}
    with pytest.raises(ProfileError, match="marker"):
        interference_template_from_mapping(missing, where="t")
    with pytest.raises(ProfileError, match="kind"):
        interference_template_from_mapping({**base, "kind": "wideband"}, where="t")


def test_detector_config_and_integration_model_checks():
    good = {"detector_window": 64, "fine_bins": 512, "nfft": 16384, "feed_sum": "all_inputs"}
    assert detector_config_from_mapping(good, where="t").fine_bins == 512
    with pytest.raises(ProfileError, match="2 nfft"):
        detector_config_from_mapping({**good, "fine_bins": 256}, where="t")
    with pytest.raises(ProfileError, match="integer"):
        detector_config_from_mapping({**good, "nfft": 16384.0}, where="t")
    model = {"frame_seconds": 0.04194304}
    assert integration_model_from_mapping(model, where="t").frame_seconds == 0.04194304
    with pytest.raises(ProfileError, match="variance_split"):
        integration_model_from_mapping({**model, "variance_split": "off"}, where="t")


def _era_files(tmp_path, channels, bands):
    author = tmp_path / "eras.json"
    author.write_text(json.dumps({"version": "v", "criterion": "c", "channels": channels}))
    off = tmp_path / "off.json"
    off.write_text(json.dumps({"schema": "pilot_proxy_transmitter_off_v1", "bands": bands}))
    return author, off


def test_era_list_refuses_bad_spans_and_intervals(tmp_path):
    interval = {"from": "2020-01", "through": None, "evidence": "e",
                "independently_verified": False}
    author, off = _era_files(tmp_path, {"5": [["2019-01", "2019-06", "a"],
                                              ["2019-08", "2020-01", "b"]]},
                             {"5": [interval]})
    eras = load_era_list(author, off, ("high", "low"))
    assert eras.off_from() == {5: "2020-01"} and eras.off_through() == {}
    for channels, message in (
            ({"5": [["2019-01", "2019-06", "a"], ["2019-06", "2020-01", "b"]]}, "overlap"),
            ({"5": [["2019-13", "2019-06", "a"]]}, "YYYY-MM"),
            ({"5": [["2019-06", "2019-01", "a"]]}, "precedes")):
        author, off = _era_files(tmp_path, channels, {"5": [interval]})
        with pytest.raises(ProfileError, match=message):
            load_era_list(author, off, ("high",))
    author, off = _era_files(tmp_path, {"5": [["2019-01", "2019-06", "a"]]},
                             {"5": [{**interval, "from": None}]})
    with pytest.raises(ProfileError, match="both be null"):
        load_era_list(author, off, ("high",))
    author, off = _era_files(tmp_path, {"5": [["2019-01", "2019-06", "a"]]},
                             {"5": [{**interval, "independently_verified": "no"}]})
    with pytest.raises(ProfileError, match="true or false"):
        load_era_list(author, off, ("high",))


def _entry(**overrides):
    entry = {"id": "floor.test", "side": "detector", "value": [1, [2.0, 3]], "units": "u",
             "basis": "policy", "status": "provisional", "rationale": "r",
             "evidence": ["x"], "sensitivity_values": []}
    entry.update(overrides)
    return entry


def _register(*entries, side="detector"):
    return {"schema": "pilot_proxy_detector_register_v1", "side": side,
            "decisions": list(entries)}


def test_register_is_side_checked():
    register = register_from_mapping(_register(_entry()), where="t")
    assert register.value("floor.test") == (1, (2.0, 3))
    assert register.decision("floor.test").record()["value"] == [1, [2.0, 3]]
    with pytest.raises(RegisterSideError, match="refuses"):
        register_from_mapping(_register(_entry(side="science")), where="t")
    with pytest.raises(RegisterSideError, match="cannot be loaded"):
        register_from_mapping(_register(_entry(), side="science"), where="t")
    with pytest.raises(RegisterSideError):
        load_register(PROJECT_DIR / "detector_register.json", side="science")


def test_register_refuses_malformed_entries():
    for entry, message in (
            (_entry(status="final"), "status"),
            (_entry(basis="guess"), "basis"),
            (_entry(evidence=[]), "needs evidence"),
            (_entry(id="floor..x"), "dotted"),
            ({k: v for k, v in _entry().items() if k != "units"}, "missing")):
        with pytest.raises(ProfileError, match=message):
            register_from_mapping(_register(entry), where="t")
    assert register_from_mapping(_register(_entry(status="open", evidence=[])),
                                 where="t").value("floor.test") == (1, (2.0, 3))
    with pytest.raises(ProfileError, match="duplicate"):
        register_from_mapping(_register(_entry(), _entry()), where="t")


def test_project_files_must_stay_inside_the_profile(tmp_path):
    target = tmp_path / "profile"
    shutil.copytree(PROJECT_DIR, target)
    assert load_project(target).load_all().frequency_plan.labels()[-1] == "37"
    spec = yaml.safe_load((target / "project.yaml").read_text())
    spec["frequency_plan"] = "../elsewhere.yaml"
    (target / "project.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(ProfileError, match="leaves the project directory"):
        load_project(target)


def test_project_refuses_a_template_for_another_adapter(tmp_path):
    target = tmp_path / "profile"
    shutil.copytree(PROJECT_DIR, target)
    spec = yaml.safe_load((target / "project.yaml").read_text())
    spec["detector_adapter"] = "wideband_energy"
    (target / "project.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(ProfileError, match="detector_adapter"):
        load_project(target).interference


def test_setup_ships_the_profile_as_a_wheel_resource(monkeypatch):
    setuptools = pytest.importorskip("setuptools")
    root = paths.SOURCE_CHECKOUT_ROOT
    if root is None:
        pytest.skip("needs a source checkout")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = {"__file__": str(root / "setup.py"), "__name__": "setup_under_test"}
    exec(compile((root / "setup.py").read_text(), str(root / "setup.py"), "exec"), namespace)
    shipped = set(namespace["_runtime_resource_files"]())
    profile = {path.relative_to(root).as_posix()
               for path in (root / "projects").rglob("*") if path.is_file()}
    assert profile and profile <= shipped
