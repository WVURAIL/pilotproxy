"""Monthly spectral reductions and their authenticated input records."""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pytest

from pilot_proxy import cli
from pilot_proxy.characterization import spectrogram as sp


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "CHANNELS", (14, 29))
    monkeypatch.setattr(sp, "NFFT", 8)
    monkeypatch.setattr(sp, "FINE_HZ", 2.0)
    monkeypatch.setattr(sp, "HALF", 1)
    products, frames, source = (tmp_path / name for name in ("products", "frames", "release"))
    products.mkdir()
    frames.mkdir()
    files, exports = {}, {}
    for channel in sp.CHANNELS:
        product = products / f"ch{channel}.npz"
        codes = np.zeros((4, 8), dtype=np.int16)
        # The nominal bin is zero; the window wraps across bins 7, 0, 1.
        codes[:, [7, 0, 1]] = [[100, 200, 300], [-32768, 600, 700],
                               [9000, 9000, 9000], [-100, -200, -300]]
        np.savez(product, physical_channel=channel, freq_id=channel, nfft=8,
                 frame_unit_index=np.arange(4), psd_db_step_per_code=0.01,
                 psd_db_invalid_code=-32768, chime_frequency_hz=1000.0,
                 pilot_frequency_hz=999.5, psd_frame_db_i16=codes)
        np.savez(frames / f"ch{channel}.npz",
                 histogram_eligible=np.array([True, True, False, True]),
                 frame_month=np.array([100, 100, 101, 102]),
                 frame_unit_index=np.arange(4))
        ledger = source / f"archive/ledger/channels/ch{channel}_fid{channel}.json"
        era = source / f"archive/channels/ch{channel}/eras.json"
        for path in (ledger, era):
            path.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps(dict(channel=channel, freq_id=channel,
                                        product=product.name,
                                        product_sha256=_digest(product.read_bytes()))))
        era.write_bytes(b'{"eras": [{"evidence": "archive start"}]}\n')
        for path in (ledger, era):
            raw = path.read_bytes()
            files[path.relative_to(source).as_posix()] = dict(sha256=_digest(raw), bytes=len(raw))
        raw = (frames / f"ch{channel}.npz").read_bytes()
        exports[f"frames/ch{channel}.npz"] = dict(sha256=_digest(raw), bytes=len(raw))
    eras_manifest = tmp_path / "eras-release.json.gz"
    eras_manifest.write_bytes(gzip.compress(json.dumps({"files": files}).encode(), mtime=0))
    frames_manifest = tmp_path / "frames-release.json"
    frames_manifest.write_text(json.dumps({"files": exports}))
    return dict(products=products, frames=frames, release=source,
                frames_release_manifest=frames_manifest,
                eras_release_manifest=eras_manifest)


def _replace_eligibility(release, channel, eligible):
    path = release["frames"] / f"ch{channel}.npz"
    with np.load(path) as saved:
        fields = {name: saved[name] for name in saved.files}
    fields["histogram_eligible"] = np.asarray(eligible, dtype=bool)
    np.savez(path, **fields)
    manifest = release["frames_release_manifest"]
    record = json.loads(manifest.read_text())
    raw = path.read_bytes()
    record["files"][f"frames/ch{channel}.npz"] = dict(sha256=_digest(raw), bytes=len(raw))
    manifest.write_text(json.dumps(record))


def test_monthly_spectrum_uses_eligible_frames_and_ascending_sky_frequency(release, tmp_path):
    output = tmp_path / "output"
    argv = ["characterize", "spectrogram"]
    for key, value in release.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    assert cli.main(argv + ["--out", str(output)]) == 0
    with np.load(output / "spectrogram.npz") as saved:
        np.testing.assert_array_equal(saved["month_index"], [100, 101, 102])
        np.testing.assert_array_equal(saved["frames"], [[2, 0, 1], [2, 0, 1]])
        np.testing.assert_allclose(saved["psd_db"][:, 0], [[5, 4, 1], [5, 4, 1]])
        assert np.isnan(saved["psd_db"][:, 1]).all()
        np.testing.assert_allclose(saved["psd_db"][:, 2], [[-3, -2, -1], [-3, -2, -1]])
        np.testing.assert_array_equal(saved["sky_offset_hz"], [[-1.5, 0.5, 2.5]] * 2)
        np.testing.assert_array_equal(saved["nominal_fine_bin"], [0, 0])
        np.testing.assert_array_equal(saved["nominal_bin_fraction"], [0.25, 0.25])
    for channel in sp.CHANNELS:
        original = release["release"] / f"archive/channels/ch{channel}/eras.json"
        assert gzip.decompress((output / f"eras/ch{channel}_eras.json.gz").read_bytes()) == original.read_bytes()
    repeated = tmp_path / "repeated"
    sp.reduce_products(**release, out=repeated)
    for path in output.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (repeated / path.relative_to(output)).read_bytes()


def test_manifest_records_both_authenticated_release_manifests(release, tmp_path):
    out = tmp_path / "output"
    sp.reduce_products(**release, out=out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["eras_source"]["release_manifest_sha256"] == _digest(
        gzip.decompress(release["eras_release_manifest"].read_bytes()))
    assert manifest["frames_source"]["release_manifest_sha256"] == _digest(
        release["frames_release_manifest"].read_bytes())


def test_channel_without_eligible_frames_keeps_empty_rows(release, tmp_path):
    _replace_eligibility(release, 29, [False] * 4)
    out = tmp_path / "output"
    sp.reduce_products(**release, out=out)
    with np.load(out / "spectrogram.npz") as saved:
        np.testing.assert_array_equal(saved["month_index"], [100, 101, 102])
        assert not saved["frames"][1].any()
        assert np.isnan(saved["psd_db"][1]).all()
        np.testing.assert_array_equal(saved["sky_offset_hz"][1], [-1.5, 0.5, 2.5])


def test_empty_cohort_is_refused_before_creating_output(release, tmp_path):
    for channel in sp.CHANNELS:
        _replace_eligibility(release, channel, [False] * 4)
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="no eligible frames"):
        sp.reduce_products(**release, out=out)
    assert not out.exists()


@pytest.mark.parametrize("kind, message", [
    ("ledger", "ledger differs"), ("product", "product differs"),
    ("frames", "frame export differs"), ("eras", "eras differ"),
])
def test_changed_inputs_are_refused_before_publishing_output(release, tmp_path, kind, message):
    paths = dict(ledger=release["release"] / "archive/ledger/channels/ch14_fid14.json",
                 product=release["products"] / "ch14.npz",
                 frames=release["frames"] / "ch14.npz",
                 eras=release["release"] / "archive/channels/ch14/eras.json")
    path = paths[kind]
    path.write_bytes(path.read_bytes() + b" ")
    out = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        sp.reduce_products(**release, out=out)
    assert not out.exists()
