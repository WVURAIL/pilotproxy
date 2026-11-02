"""Failed injections preserve source data and do not publish stale manifests."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime import injection


def _source(path, data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = np.full((16, 2), 0x88, dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        handle.attrs["freq"] = 470.3125
        handle.create_dataset("baseband", data=data)
        handle.create_dataset("sibling", data=np.arange(3))
    return path


def _inject(source, output, **settings):
    return injection.inject_tone_into_baseband_file(
        source, output,
        **({"baseband_frequency_hz": 0, "amplitude_lsb": 2, "phase_seed": 0} | settings),
    )


def _directory(sources, output, **settings):
    return injection.inject_directory(
        sources, output,
        **({"baseband_frequency_hz": 0, "amplitude_lsb": 2, "phase_seed": 0} | settings),
    )


@pytest.mark.parametrize("data", [
    np.full((16, 2), 128.5), np.full((16, 2), 128, dtype=np.uint16),
    np.zeros(16, dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8),
    np.zeros((16, 0), dtype=np.uint8),
])
@pytest.mark.parametrize("amplitude", [0, 2])
def test_unsupported_native_data_preserves_existing_output(tmp_path, data, amplitude):
    source = _source(tmp_path / "input.h5", data)
    output = tmp_path / "output.h5"
    output.write_bytes(b"previous output")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="baseband"):
        _inject(source, output, amplitude_lsb=amplitude)
    assert source.read_bytes() == original
    assert output.read_bytes() == b"previous output"


@pytest.mark.parametrize("storage", ["external_link", "external_storage", "virtual"])
def test_injection_refuses_baseband_that_is_not_stored_in_the_copied_file(tmp_path, storage):
    backing = _source(tmp_path / "backing.h5")
    source = tmp_path / "input.h5"
    raw_path = tmp_path / "samples.raw"
    with h5py.File(source, "w") as handle:
        handle.attrs["freq"] = 470.3125
        if storage == "external_link":
            handle["baseband"] = h5py.ExternalLink(str(backing), "baseband")
        elif storage == "external_storage":
            data = handle.create_dataset("baseband", shape=(16, 2), dtype=np.uint8,
                                         external=[(str(raw_path), 0, 32)])
            data[...] = 0x88
        else:
            layout = h5py.VirtualLayout(shape=(16, 2), dtype=np.uint8)
            layout[:] = h5py.VirtualSource(str(backing), "baseband", shape=(16, 2))
            handle.create_virtual_dataset("baseband", layout)
    protected = {path: path.read_bytes() for path in (source, backing, raw_path) if path.exists()}
    output = tmp_path / "out.h5"
    with pytest.raises(ValueError, match="baseband"):
        _inject(source, output)
    assert not output.exists()
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["copy", "verification"])
def test_failed_file_injection_does_not_replace_output(tmp_path, monkeypatch, existing, stage):
    source = _source(tmp_path / "source.h5")
    output = tmp_path / "output.h5"
    original = source.read_bytes()
    if existing:
        output.write_bytes(b"previous output")
    if stage == "copy":
        def fail_copy(source_path, destination):
            Path(destination).write_bytes(b"partial copy")
            raise OSError("injected copy failure")
        monkeypatch.setattr(injection.shutil, "copy2", fail_copy)
    else:
        def fail_verification(*args):
            raise RuntimeError("injected verification failure")
        monkeypatch.setattr(injection, "verify_injection_preserves_siblings", fail_verification)
    with pytest.raises((OSError, RuntimeError), match="injected .* failure"):
        _inject(source, output)
    assert source.read_bytes() == original
    assert output.read_bytes() == b"previous output" if existing else not output.exists()
    assert set(tmp_path.iterdir()) == ({source, output} if existing else {source})


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_source_output_alias_is_refused_without_changing_source(tmp_path, alias):
    source = _source(tmp_path / "source.h5")
    output = source if alias == "same_path" else tmp_path / "alias.h5"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="source|same"):
        _inject(source, output)
    assert source.read_bytes() == original


def test_direct_frequency_limit_is_checked_before_output_changes(tmp_path):
    source = _source(tmp_path / "source.h5")
    output = tmp_path / "output.h5"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="outside"):
        _inject(source, output, baseband_frequency_hz=1e300)
    assert output.read_bytes() == b"previous output"


@pytest.mark.parametrize("seed", [-1, 1.5, True])
@pytest.mark.parametrize("directory", [False, True])
def test_invalid_seed_is_refused_without_publishing(tmp_path, seed, directory):
    source = _source(tmp_path / "source.h5")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="phase_seed"):
        if directory:
            _directory([source], output, phase_seed=seed)
        else:
            _inject(source, output, phase_seed=seed)
    assert not output.exists()


def test_successful_replacement_does_not_follow_an_existing_output_symlink(tmp_path):
    source = _source(tmp_path / "source.h5")
    protected = tmp_path / "other.h5"
    protected.write_bytes(b"unrelated output target")
    output = tmp_path / "output.h5"
    output.symlink_to(protected)
    entry = _inject(source, output, phase_seed=np.int64(0))
    assert not output.is_symlink()
    assert protected.read_bytes() == b"unrelated output target"
    assert entry["output"] == str(output)
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle["baseband"], np.tile([0x76, 0x8A], (16, 1)))


def test_nan_siblings_and_attributes_survive_injection(tmp_path):
    source = _source(tmp_path / "source.h5")
    with h5py.File(source, "a") as handle:
        handle.attrs["missing"] = np.array([1.0, np.nan])
        handle.create_dataset("float_metadata", data=[np.nan, 3.0])
        handle.create_dataset("complex_metadata", data=[complex(np.nan, 2), complex(3, np.nan)])
        handle.create_dataset("records", data=np.array([(1, np.nan), (2, 3.0)],
                                                       dtype=[("id", "i4"), ("value", "f8")]))
        handle.create_dataset("labels", data=np.array(["one", "two"], dtype=h5py.string_dtype()))
    original = source.read_bytes()
    output = tmp_path / "out.h5"
    _inject(source, output)
    assert source.read_bytes() == original
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle["float_metadata"], [np.nan, 3.0])
        np.testing.assert_array_equal(handle.attrs["missing"], [1.0, np.nan])
        assert handle["labels"].asstr()[...].tolist() == ["one", "two"]


def test_nan_metadata_does_not_hide_changes_to_other_complex_component(tmp_path):
    source = _source(tmp_path / "source.h5")
    with h5py.File(source, "a") as handle:
        handle.create_dataset("complex_metadata", data=[complex(np.nan, 2)])
    output = tmp_path / "output.h5"
    shutil.copy2(source, output)
    with h5py.File(output, "a") as handle:
        handle["complex_metadata"][...] = [complex(np.nan, 3)]
    with pytest.raises(RuntimeError, match="sibling dataset"):
        injection.verify_injection_preserves_siblings(source, output)


def test_duplicate_basenames_are_refused_before_creating_output(tmp_path):
    sources = [_source(tmp_path / folder / "capture.h5") for folder in ("one", "two")]
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="duplicate|same.*name"):
        _directory(sources, output)
    assert not output.exists()


@pytest.mark.parametrize("case", ["empty", "manifest_name", "output_directory", "manifest_directory"])
def test_directory_layout_conflicts_are_refused_without_partial_files(tmp_path, case):
    source = _source(tmp_path / "source.h5")
    sources = [source]
    output = tmp_path / "output"
    if case == "empty":
        sources = []
    elif case == "manifest_name":
        sources = [_source(tmp_path / injection.INJECTION_MANIFEST_FILENAME)]
    elif case == "output_directory":
        (output / source.name).mkdir(parents=True)
    else:
        (output / injection.INJECTION_MANIFEST_FILENAME).mkdir(parents=True)
    previous = set(output.iterdir()) if output.exists() else set()
    with pytest.raises(ValueError, match="source|manifest|directory"):
        _directory(sources, output)
    assert (set(output.iterdir()) if output.exists() else set()) == previous


def test_directory_source_alias_is_refused_before_first_output(tmp_path):
    output = tmp_path / "output"
    sources = [_source(tmp_path / "input/a.h5"), _source(output / "b.h5")]
    original = sources[1].read_bytes()
    with pytest.raises(ValueError, match="source|same"):
        _directory(sources, output)
    assert sources[1].read_bytes() == original
    assert not (output / "a.h5").exists()
    assert not (output / injection.INJECTION_MANIFEST_FILENAME).exists()


def test_directory_manifest_must_not_replace_a_source_through_its_alias(tmp_path):
    output = tmp_path / "output"
    target = _source(output / injection.INJECTION_MANIFEST_FILENAME)
    source = tmp_path / "capture.h5"
    source.symlink_to(target)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="manifest.*source|source.*manifest"):
        _directory([source], output)
    assert source.read_bytes() == original
    assert target.read_bytes() == original
    assert set(output.iterdir()) == {target}


def _previous_run(tmp_path):
    sources = [_source(tmp_path / "input" / name) for name in ("a.h5", "b.h5")]
    output = tmp_path / "output"
    _directory(sources, output, amplitude_lsb=0)
    (output / "notes.txt").write_text("keep")
    previous = {path: path.read_bytes() for path in output.iterdir()}
    return sources, output, previous


def test_failed_directory_staging_preserves_complete_previous_run(tmp_path, monkeypatch):
    sources, output, previous = _previous_run(tmp_path)
    verify = injection.verify_injection_preserves_siblings
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected verification failure")
        return verify(*args)

    monkeypatch.setattr(injection, "verify_injection_preserves_siblings", fail_second)
    with pytest.raises(RuntimeError, match="injected verification failure"):
        _directory(sources, output)
    assert all(path.read_bytes() == content for path, content in previous.items())
    assert set(output.iterdir()) == set(previous)


def test_failed_directory_publication_cannot_retain_old_manifest(tmp_path, monkeypatch):
    sources, output, previous = _previous_run(tmp_path)
    replace = os.replace

    def fail_second(source, destination, *args, **kwargs):
        if Path(destination) == output / "b.h5":
            raise OSError("injected publication failure")
        return replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="injected publication failure"):
        _directory(sources, output)
    assert not (output / injection.INJECTION_MANIFEST_FILENAME).exists()
    assert (output / "a.h5").read_bytes() != previous[output / "a.h5"]
    assert (output / "b.h5").read_bytes() == previous[output / "b.h5"]
    assert (output / "notes.txt").read_text() == "keep"
    assert set(output.iterdir()) == {output / "a.h5", output / "b.h5", output / "notes.txt"}


def test_successful_directory_rerun_replaces_files_and_publishes_final_paths(tmp_path):
    sources, output, _ = _previous_run(tmp_path)
    protected = {path: path.read_bytes() for path in sources}
    entries = _directory(sources, output)
    report = json.loads((output / injection.INJECTION_MANIFEST_FILENAME).read_text())
    assert report["files"] == entries
    assert [entry["phase_seed"] for entry in entries] == [0, 1]
    assert [entry["output"] for entry in entries] == [str(output / path.name) for path in sources]
    assert [entry["source"] for entry in entries] == [str(path) for path in sources]
    for source in sources:
        with h5py.File(output / source.name) as handle:
            assert np.any(handle["baseband"][...] != 0x88)
            np.testing.assert_array_equal(handle["sibling"], np.arange(3))
    assert all(path.read_bytes() == content for path, content in protected.items())
    assert (output / "notes.txt").read_text() == "keep"
    assert set(output.iterdir()) == {output / "a.h5", output / "b.h5", output / "notes.txt",
                                    output / injection.INJECTION_MANIFEST_FILENAME}
