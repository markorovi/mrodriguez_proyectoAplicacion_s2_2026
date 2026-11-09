"""End-to-end orchestration with real alignment and comparison, and model doubles."""

import numpy as np
import pytest

from config import AlignmentSettings, AppSettings, DevicePreference, SegmentationSettings
from controllers import PipelineController
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

STAGES = {"acquisition", "depth", "alignment", "segmentation", "comparison", "rendering"}


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
    assert set(report.stage_seconds) == STAGES
    assert report.total_seconds == pytest.approx(sum(report.stage_seconds.values()))
    assert report.annotated_image.shape == before.pixels.shape
    assert report.annotated_image.dtype == np.uint8
    # Boxes and captions were drawn over the "after" view.
    assert np.count_nonzero(np.any(report.annotated_image != moved.pixels, axis=2)) > 500
    assert report.comparison.difference_m.shape == (before.height, before.width)


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
