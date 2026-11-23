"""Test outputs stay out of the repository unless a versioned sample is updated on purpose."""

from config.paths import PROJECT_ROOT, SAMPLES_DIR
from tests.sample_outputs import output_path


def test_outputs_go_to_the_temporary_directory_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("UPDATE_SAMPLES", raising=False)

    path = output_path("depth_output_sample.png", tmp_path)

    assert path == tmp_path / "depth_output_sample.png"
    assert not path.resolve().is_relative_to(PROJECT_ROOT.resolve())


def test_update_samples_writes_a_versioned_sample_into_data_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("UPDATE_SAMPLES", "1")

    assert output_path("comparison_output_sample.png", tmp_path) == SAMPLES_DIR / "comparison_output_sample.png"
    assert output_path("depth_output_sample.png", tmp_path) == SAMPLES_DIR / "depth_output_sample.png"


def test_output_of_a_personal_pair_never_lands_in_the_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("UPDATE_SAMPLES", "1")

    path = output_path("real_comparison_output_ejemplo.png", tmp_path)

    assert path.parent == tmp_path
    assert not path.resolve().is_relative_to(PROJECT_ROOT.resolve())
