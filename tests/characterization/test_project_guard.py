"""A project whose bound layers differ is refused."""
from __future__ import annotations

import pytest

from characterization_fixtures import characterize, products_dir, project_copy
from pilot_proxy.config._files import ProfileError
from pilot_proxy.config.project import load_project


def test_a_project_with_another_register_is_refused(tmp_path):
    project = project_copy(tmp_path)
    register = load_project(project).files["register"]
    register.write_text(register.read_text(encoding="utf-8").rstrip() + "\n\n", encoding="utf-8")
    with pytest.raises(ProfileError, match=r"its register differ from the default project's"):
        characterize(tmp_path, products=products_dir(tmp_path), project_dir=project, out="refused")
    assert not (tmp_path / "refused" / "characterization").exists()


def test_a_project_with_another_detector_configuration_is_refused(tmp_path):
    project = project_copy(tmp_path)
    config = load_project(project).files["detector_config"]
    config.write_text(config.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(ProfileError, match=r"detector_config"):
        characterize(tmp_path, products=products_dir(tmp_path), project_dir=project, out="refused")
