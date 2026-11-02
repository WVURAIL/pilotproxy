"""Reject incompatible stream contracts before reading or changing a product."""
from types import SimpleNamespace

import pytest

from pilot_proxy.archive import pipeline
from pilot_proxy.archive.interfaces import PluginInfo, RunContext, stream_compatibility


@pytest.mark.parametrize(
    "reader_kind,accepted,detail",
    [
        ("complex", ("visibility",),
         "reader 'reader' emits 'complex', but analyzer 'analyzer' accepts only ['visibility']"),
        ("", ("complex",),
         "stream contract is incomplete (missing reader 'reader' stream_kind)"),
        ("complex", (),
         "stream contract is incomplete (missing analyzer 'analyzer' accepts_stream_kinds)"),
        ("", (),
         "stream contract is incomplete (missing reader 'reader' stream_kind and analyzer 'analyzer' accepts_stream_kinds)"),
    ],
)
def test_engine_contract_refusal_preserves_existing_product(
    tmp_path, reader_kind, accepted, detail,
):
    reader = SimpleNamespace(info=PluginInfo(
        name="reader", kind="reader", summary="reader", stream_kind=reader_kind,
    ))
    analyzer = SimpleNamespace(info=PluginInfo(
        name="analyzer", kind="analyzer", summary="analyzer",
        accepts_stream_kinds=accepted,
    ))
    output = tmp_path / "existing-product.npz"
    output.write_bytes(b"existing scientific product")
    scratch = tmp_path / "scratch"
    context = RunContext(instrument=None, options={"selection": "unchanged"})

    def units():
        raise AssertionError("incompatible stream consumed the unit iterator")
        yield

    with pytest.raises(SystemExit) as failure:
        pipeline.run(
            source=SimpleNamespace(), reader=reader, analyzer=analyzer,
            units=units(), out_path=str(output), tmp_dir=str(scratch),
            ctx=context, verbose=False,
        )
    assert str(failure.value) == (
        f"incompatible reader/analyzer stream: {detail}. Choose a matching pair."
    )
    assert output.read_bytes() == b"existing scientific product"
    assert not scratch.exists()
    assert context.options == {"selection": "unchanged"}
    # Refusal releases the product lock, allowing a corrected run to retry.
    with pipeline._OutputLock(str(output)):
        pass


@pytest.mark.parametrize(
    "accepted,detail",
    [
        (("complex",), "reader 'reader' emits 'complex'; analyzer 'analyzer' accepts it"),
        (("*",), "analyzer 'analyzer' explicitly accepts every stream kind"),
    ],
)
def test_matching_contract_diagnostic_distinguishes_explicit_wildcard(accepted, detail):
    reader = PluginInfo(name="reader", kind="reader", summary="reader", stream_kind="complex")
    analyzer = PluginInfo(name="analyzer", kind="analyzer", summary="analyzer", accepts_stream_kinds=accepted)
    contract = stream_compatibility(reader, analyzer)
    assert contract.compatible is True
    assert contract.detail == detail
