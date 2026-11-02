# coding=utf-8
from __future__ import annotations

import hashlib

from pilot_proxy.provenance import (
    detector_version_build_id,
    detector_version_geometry,
    file_sha256,
    sidecar_manifest_path,
)
from pilot_proxy.product_contract import PER_PILOT_PRODUCT_SCHEMA_TOKEN


def _ver(version="1.0.0", source="aaa111", kernel="2.1.0", kernel_sha="c85f50dd",
         schema=PER_PILOT_PRODUCT_SCHEMA_TOKEN, k=128):
    return (f"pilot-proxy/{version} source={source} kernel={kernel} "
            f"kernel_sha256={kernel_sha} {schema} K={k}")


def test_file_sha256_returns_digest_for_existing_file(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"kernel provenance\n")

    assert file_sha256(path) == hashlib.sha256(b"kernel provenance\n").hexdigest()


def test_file_sha256_returns_none_for_missing_file(tmp_path) -> None:
    assert file_sha256(tmp_path / "missing.bin") is None
    assert file_sha256(None) is None


def test_sidecar_manifest_path_uses_weight_manifest_convention(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    assert sidecar_manifest_path(path) == tmp_path / "weights.bin.manifest.json"


def test_package_source_sha256_is_memoized_per_process(tmp_path) -> None:
    from pilot_proxy.provenance import package_source_sha256

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_bytes(b"x = 1\n")
    package_source_sha256.cache_clear()
    first = package_source_sha256(pkg)

    # A mid-run change to the tree must not change the stamp of this process.
    (pkg / "a.py").write_bytes(b"x = 2\n")
    assert package_source_sha256(pkg) == first

    # A fresh process (cache cleared) observes the new tree.
    package_source_sha256.cache_clear()
    assert package_source_sha256(pkg) != first


# The detector_version token rule is enforced in two places, the detector's
# resume guard and combine's _check_invariants. Both import
# detector_version_geometry, so the tests below specify both.

def test_geometry_drops_build_tokens_and_keeps_the_rest() -> None:
    assert detector_version_geometry(_ver()) == (
        "kernel=2.1.0",
        "kernel_sha256=c85f50dd",
        PER_PILOT_PRODUCT_SCHEMA_TOKEN,
        "K=128",
    )


def test_release_version_bump_is_geometry_identical() -> None:
    """The release bump from 0.3.0.dev0 to 1.0.0 leaves the geometry unchanged.

    The bump changes the version label and, through the edit to __init__.py,
    the source tree hash. Neither is part of the geometry, so pilot products
    of one survey that were completed in different runs may stack across the bump.
    Resume still requires the exact full build identity.
    """
    before = _ver(version="0.3.0.dev0", source="02066f1d6337")
    after = _ver(version="1.0.0", source="0c66af82f98b")
    assert before != after
    assert detector_version_geometry(before) == detector_version_geometry(after)


def test_source_only_change_is_geometry_identical() -> None:
    assert (detector_version_geometry(_ver(source="aaa111"))
            == detector_version_geometry(_ver(source="bbb222")))


def test_geometry_token_changes_are_not_forgiven() -> None:
    base = detector_version_geometry(_ver())
    for changed in (
        _ver(kernel="3.0.0"),
        _ver(kernel_sha="deadbeef"),
        _ver(schema="pilotproxy_detector_datatrawl_v2"),
        _ver(k=256),
    ):
        assert detector_version_geometry(changed) != base


def test_build_id_pairs_version_with_source() -> None:
    assert detector_version_build_id(
        _ver(version="0.3.0.dev0", source="02066f1d6337a3e1f05b")
    ) == "0.3.0.dev0@02066f1d6337"


def test_build_id_tolerates_missing_tokens() -> None:
    assert detector_version_build_id("kernel=2.1.0 K=128") == "?@?"
