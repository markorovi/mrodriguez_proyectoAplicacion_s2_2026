"""End-to-end orchestration with real alignment and comparison, and model doubles."""

import numpy as np
import pytest

from config import AlignmentSettings, AppSettings, DevicePreference, SegmentationSettings
from controllers import STAGES, PipelineController
from models import (
    UNIDENTIFIED_LABEL,
    AlignmentError,
    ChangeType,
    ComparisonReport,
    ImageData,
    InvalidImageError,
)
from services.inference import DeviceManager
from services.segmentation import SegFormerSegmenter
from tests.doubles import FakeSegmentationModelLoader, SceneDepthEstimator
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder


@pytest.fixture(scope="module")
def builder() -> SyntheticSceneBuilder:
    return SyntheticSceneBuilder(640, 480, seed=7)


@pytest.fixture
def before(builder) -> ImageData:
    return ImageData(builder.render([SceneObject(center_x=0.3)]), "before.png")


@pytest.fixture
def moved(builder) -> ImageData:
    return ImageData(builder.render([SceneObject(center_x=0.65)]), "after.png")


@pytest.fixture
def removed(builder) -> ImageData:
    return ImageData(builder.render([]), "after.png")


@pytest.fixture
def estimator() -> SceneDepthEstimator:
    return SceneDepthEstimator()


def _controller(estimator, settings: AppSettings | None = None, segmenter=None) -> PipelineController:
    settings = settings or AppSettings()
    if segmenter is None and settings.segmentation.enabled:
        segmenter = SegFormerSegmenter(
            settings.segmentation, DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
        )
    return PipelineController(settings, depth_estimator=estimator, segmenter=segmenter)


def test_displaced_object_is_reported_end_to_end(estimator, before, moved):
    report = _controller(estimator).analyze(before, moved)

    assert isinstance(report, ComparisonReport)
    assert report.has_changes
    assert len(report.objects) == 1
    obj = report.objects[0]
    assert obj.change_type is ChangeType.DISPLACED
    assert obj.label.name == "caja"
    assert obj.distance_cm > 30.0
    assert obj.position_before is not None and obj.position_after is not None
    assert report.objects_of(ChangeType.DISPLACED) == [obj]


def test_removed_object_is_reported_end_to_end(estimator, before, removed):
    report = _controller(estimator).analyze(before, removed)

    assert [(o.change_type, o.label.name) for o in report.objects] == [(ChangeType.REMOVED, "caja")]


def test_unchanged_scene_has_no_changes(estimator, before):
    report = _controller(estimator).analyze(before, before)
    assert not report.has_changes
    assert report.warnings == ()


def test_report_includes_alignment_timings_and_annotated_image(estimator, before, moved):
    report = _controller(estimator).analyze(before, moved)

    assert report.alignment.succeeded
    assert report.alignment.inliers >= 10
    assert set(report.stage_seconds) == set(STAGES)
    assert report.total_seconds == pytest.approx(sum(report.stage_seconds.values()))
    assert report.annotated_image.shape == before.pixels.shape
    assert report.annotated_image.dtype == np.uint8
    # Boxes and captions were drawn over the "after" view.
    assert np.count_nonzero(np.any(report.annotated_image != moved.pixels, axis=2)) > 500
    assert report.comparison.difference_m.shape == (before.height, before.width)


def test_progress_announces_every_stage_in_order(estimator, before, moved):
    announced = []
    _controller(estimator).analyze(before, moved, progress=announced.append)
    assert tuple(announced) == STAGES


def test_progress_is_announced_when_analyzing_files(estimator, before, moved, write_image):
    announced = []
    _controller(estimator).analyze_files(
        write_image(before.pixels, "before.png"), write_image(moved.pixels, "after.png"), announced.append
    )
    assert tuple(announced) == STAGES


def test_models_are_loaded_in_their_own_stage_only_the_first_time(estimator, before, moved):
    controller = _controller(estimator)
    assert not controller.models_loaded

    first = controller.analyze(before, moved)
    assert controller.models_loaded
    assert "loading" in first.stage_seconds

    announced = []
    second = controller.analyze(before, moved, progress=announced.append)
    assert "loading" not in second.stage_seconds
    assert tuple(announced) == tuple(stage for stage in STAGES if stage != "loading")


def test_controller_exposes_its_acquisition_service(estimator):
    controller = _controller(estimator)
    assert controller.acquisition.validator.settings == controller.settings.image


def test_both_views_go_through_depth_estimation(estimator, before, moved):
    _controller(estimator).analyze(before, moved)
    assert estimator.calls == 2


def test_image_files_are_loaded_and_validated(estimator, before, moved, write_image):
    before_path = write_image(before.pixels, "before.png")
    after_path = write_image(moved.pixels, "after.png")

    report = _controller(estimator).analyze_files(before_path, after_path)
    assert [o.change_type for o in report.objects] == [ChangeType.DISPLACED]


def test_invalid_image_stops_the_analysis(estimator, before):
    tiny = ImageData(np.zeros((100, 100, 3), np.uint8), "tiny.png")
    with pytest.raises(InvalidImageError):
        _controller(estimator).analyze(before, tiny)
    assert estimator.calls == 0


def test_validation_warnings_are_carried_into_the_report(estimator, builder):
    dark = builder.render([SceneObject(center_x=0.3)], brightness=0.12)
    report = _controller(estimator).analyze(ImageData(dark, "a"), ImageData(dark, "b"))

    assert any(text.startswith("Imagen 'antes':") for text in report.warnings)
    assert any(text.startswith("Imagen 'después':") for text in report.warnings)


def test_failed_alignment_falls_back_to_unaligned_comparison(estimator, before, removed):
    settings = AppSettings(alignment=AlignmentSettings(min_inliers=100_000))
    report = _controller(estimator, settings).analyze(before, removed)

    assert report.alignment.used_fallback and not report.alignment.succeeded
    assert any("sin alinear" in text for text in report.warnings)
    assert [o.change_type for o in report.objects] == [ChangeType.REMOVED]


def test_failed_alignment_raises_when_fallback_is_disabled(estimator, before, removed):
    settings = AppSettings(alignment=AlignmentSettings(min_inliers=100_000, fallback_to_identity=False))
    with pytest.raises(AlignmentError):
        _controller(estimator, settings).analyze(before, removed)


def test_segmentation_can_be_disabled(estimator, before, removed):
    settings = AppSettings(segmentation=SegmentationSettings(enabled=False))
    report = _controller(estimator, settings).analyze(before, removed)
    assert report.objects[0].label.name == UNIDENTIFIED_LABEL


def test_close_releases_every_model(estimator, before, moved):
    segmenter = SegFormerSegmenter(
        SegmentationSettings(), DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    with _controller(estimator, segmenter=segmenter) as controller:
        controller.analyze(before, moved)
        assert estimator.is_loaded and segmenter.is_loaded
    assert not estimator.is_loaded and not segmenter.is_loaded


# --- Replaceable segmentation model (RA-003) ------------------------------------------------------

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

import controllers.pipeline_controller as pipeline_module  # noqa: E402
from tests.doubles import SceneSegmenter  # noqa: E402


def test_another_segmenter_can_replace_segformer(estimator, before, moved):
    segmenter = SceneSegmenter()
    assert not isinstance(segmenter, SegFormerSegmenter)

    report = _controller(estimator, segmenter=segmenter).analyze(before, moved)

    assert [obj.change_type for obj in report.objects] == [ChangeType.DISPLACED]
    # "before", the aligned "after" and the original "after" for the indoor check.
    assert segmenter.segment_calls == 3
    assert segmenter.load_calls == 1


def test_default_segmenter_comes_from_the_factory(monkeypatch, estimator, before, moved):
    requested = []
    replacement = SceneSegmenter()

    def fake_create_segmenter(settings):
        requested.append(settings)
        return replacement

    monkeypatch.setattr(pipeline_module, "create_segmenter", fake_create_segmenter)
    settings = AppSettings()

    controller = PipelineController(settings, depth_estimator=estimator)
    controller.analyze(before, moved)

    assert requested == [settings.segmentation]
    assert replacement.segment_calls == 3


def test_controller_does_not_name_the_concrete_segmenter():
    source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
    assert "SegFormerSegmenter" not in source

    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "services.segmentation.segformer_segmenter" not in imported
    assert "services.segmentation.base_segmenter" in imported
