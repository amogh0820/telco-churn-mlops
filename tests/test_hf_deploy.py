"""Tests for the Hugging Face Spaces deploy helper.

huggingface_hub itself is never imported at module scope in
deploy_to_spaces.py -- only inside main(), after argument parsing -- so
these tests exercise the file-assembly logic without that dependency
installed, exactly as `test_hf_deploy_import_does_not_require_huggingface_hub`
checks directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.config import MODEL_PATH

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.huggingface.deploy_to_spaces import build_upload_dir


def test_deploy_module_does_not_import_huggingface_hub_at_load_time():
    """huggingface_hub is a dev-only dependency, imported lazily inside main().

    If a future edit moved that import to module scope, this test file
    would fail to collect in any environment without huggingface_hub
    installed -- which is exactly the environment this guards against.
    """
    assert "huggingface_hub" not in sys.modules


def test_build_upload_dir_exits_without_a_trained_model(tmp_path):
    if MODEL_PATH.exists():
        pytest.skip("a real model is present; the missing-model path isn't exercised here")
    with pytest.raises(SystemExit):
        build_upload_dir(tmp_path / "space")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make train` first")
def test_build_upload_dir_produces_exactly_what_the_dockerfile_copies(tmp_path):
    dest = tmp_path / "space"
    build_upload_dir(dest)

    assert (dest / "Dockerfile").is_file()
    assert (dest / "README.md").is_file()
    assert (dest / "requirements.txt").is_file()
    assert (dest / "src" / "api" / "main.py").is_file()
    assert (dest / "artifacts" / "model.joblib").is_file()
    assert (dest / "artifacts" / "model_metadata.json").is_file()

    dockerfile = (dest / "Dockerfile").read_text()
    readme = (dest / "README.md").read_text()
    assert "app_port: 7860" in readme
    assert "EXPOSE 7860" in dockerfile
    assert "sdk: docker" in readme


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make train` first")
def test_build_upload_dir_excludes_dev_only_files(tmp_path):
    """Tests, the notebook, and AWS scripts have no reason to be in the Space."""
    dest = tmp_path / "space"
    build_upload_dir(dest)
    pushed = {p.name for p in dest.rglob("*") if p.is_file()}
    assert "test_api.py" not in pushed
    assert not any(name.endswith(".ipynb") for name in pushed)
    assert "push_to_ecr.sh" not in pushed
