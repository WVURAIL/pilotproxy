"""Headless configuration and cleanup on errors."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfFile
import pytest

from pilot_proxy import figure_style as style


@pytest.fixture
def axes(monkeypatch):
    monkeypatch.setattr(style.shutil, "which", lambda name: None)
    with matplotlib.rc_context():
        style.configure(require_tex=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        yield ax
        plt.close(fig)


def test_missing_tex_is_explicit_and_preview_remains_available(axes):
    assert not matplotlib.rcParams["text.usetex"]
    with pytest.raises(RuntimeError, match="requires latex"):
        style.configure(require_tex=True)


def test_subset_hook_restores_even_after_renderer_failure(monkeypatch):
    original = lambda charset: "ORIGINAL+"
    monkeypatch.setattr(PdfFile, "_get_subset_prefix", staticmethod(original), raising=False)
    with pytest.raises(RuntimeError, match="renderer failed"):
        with style.stable_pdf_subset_tags():
            assert PdfFile._get_subset_prefix(["b", "a"]) == style.stable_subset_prefix(["a", "b"])
            raise RuntimeError("renderer failed")
    assert PdfFile._get_subset_prefix([]) == "ORIGINAL+"
    monkeypatch.delattr(PdfFile, "_get_subset_prefix")
    with style.stable_pdf_subset_tags():
        assert not hasattr(PdfFile, "_get_subset_prefix")
