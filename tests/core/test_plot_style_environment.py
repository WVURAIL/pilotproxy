"""TeX selection requires executable commands in the configured PATH."""

import matplotlib
import pytest

from pilot_proxy import plot_style


def _commands(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "1")
    for name in ("latex", "dvipng", "kpsewhich"):
        path = tmp_path / name
        if kind == "directory":
            path.mkdir()
        else:
            path.write_text("#!/bin/sh\nprintf '/fixture/package.sty\\n'\n")
            path.chmod(0o755 if kind == "executable" else 0o644)


@pytest.mark.parametrize("kind", ["directory", "nonexecutable"])
def test_optional_tex_uses_mathtext_when_helper_names_cannot_execute(tmp_path, monkeypatch, kind):
    _commands(tmp_path, monkeypatch, kind)
    with matplotlib.rc_context():
        plt = plot_style.setup_matplotlib()
        assert matplotlib.rcParams["text.usetex"] is False
        fig, ax = plt.subplots()
        ax.set_title(r"$Q = 2$")
        output = tmp_path / "mathtext.png"
        fig.savefig(output)
        plt.close(fig)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("kind", ["directory", "nonexecutable"])
def test_dissertation_tex_refuses_unexecutable_helpers_before_package_lookup(tmp_path, monkeypatch, kind):
    _commands(tmp_path, monkeypatch, kind)

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("missing executables reached package lookup")

    monkeypatch.setattr(plot_style.subprocess, "run", unexpected_lookup)
    with matplotlib.rc_context():
        with pytest.raises(RuntimeError, match="requires latex, dvipng, kpsewhich"):
            plot_style.setup_matplotlib(dissertation_style=True)


def test_executable_path_commands_enable_the_requested_tex_style(tmp_path, monkeypatch):
    _commands(tmp_path, monkeypatch, "executable")
    with matplotlib.rc_context():
        plot_style.setup_matplotlib(dissertation_style=True)
        assert matplotlib.rcParams["text.usetex"] is True
        assert matplotlib.rcParams["font.serif"] == ["Latin Modern Roman"]
        assert r"\usepackage{lmodern}" in matplotlib.rcParams["text.latex.preamble"]
    assert plot_style._command_available("latex")
    # This helper intentionally accepts command names from PATH, not arbitrary paths.
    assert not plot_style._command_available(str(tmp_path / "latex"))


def test_missing_latex_package_is_a_strict_style_error(tmp_path, monkeypatch):
    _commands(tmp_path, monkeypatch, "executable")
    (tmp_path / "kpsewhich").write_text("#!/bin/sh\nexit 1\n")
    with matplotlib.rc_context():
        with pytest.raises(RuntimeError, match="requires the LaTeX lmodern package"):
            plot_style.setup_matplotlib(dissertation_style=True)
