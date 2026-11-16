"""Timing of the complete analysis (scripts.benchmark_pipeline), with model doubles."""

import pytest

from config import AppSettings, DevicePreference, SegmentationSettings
from controllers import PipelineController
from scripts import benchmark_pipeline as bench
from services.inference import DeviceManager
from services.segmentation import SegFormerSegmenter
from tests.doubles import FakeSegmentationModelLoader, SceneDepthEstimator
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder


@pytest.fixture(scope="module")
def pair_files(tmp_path_factory):
    from PIL import Image

    folder = tmp_path_factory.mktemp("pair")
    builder = SyntheticSceneBuilder(640, 480, seed=7)
    before, after = folder / "before.png", folder / "after.png"
    Image.fromarray(builder.render([SceneObject(center_x=0.3)])).save(before)
    Image.fromarray(builder.render([SceneObject(center_x=0.65)])).save(after)
    return before, after


def _controller() -> PipelineController:
    segmenter = SegFormerSegmenter(
        SegmentationSettings(), DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    return PipelineController(AppSettings(), depth_estimator=SceneDepthEstimator(), segmenter=segmenter)


def test_settings_force_both_models_onto_the_device():
    settings = bench.settings_for(DevicePreference.CPU)
    assert settings.depth.device is DevicePreference.CPU
    assert settings.segmentation.device is DevicePreference.CPU


def test_first_pair_pays_the_model_loading_and_the_rest_do_not(pair_files):
    before, after = pair_files
    timing = bench.measure_device(_controller(), "cpu", [("a", before, after), ("b", before, after)], repetitions=2)

    first, second = timing.pairs
    assert first.first_seconds is not None and "loading" in first.first_stages
    assert second.first_seconds is None
    assert len(first.warm_seconds) == 2
    assert "loading" not in first.warm_stage_means
    assert first.warm_stage_means["depth"] >= 0.0
    assert first.image_size == (640, 480)
    assert any("caja" in change and "DESPLAZAMIENTO" in change for change in first.changes)
    assert timing.device_names["segmentation"] == "CPU"


def test_report_is_printed_in_spanish(pair_files, capsys):
    before, after = pair_files
    timing = bench.measure_device(_controller(), "cpu", [("sintético", before, after)], repetitions=1)
    bench.print_device(timing)
    out = capsys.readouterr().out
    assert "Primer análisis" in out
    assert "Con modelos en memoria" in out
    assert "Por etapa" in out
