"""Segmentation confidence and the rule that hides doubtful object names (RF-006, RF-007).

The segmenter keeps how sure it is of every pixel. A changed object whose
class the model is not sure about is shown as an unidentified object instead
of with a wrong name. Only the displayed name changes: the persistence filter
and the pairing by label keep seeing the same labels as before, so the change
types, distances and number of objects stay the same.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from config import AppSettings, CameraIntrinsics, ComparisonSettings, DevicePreference, SegmentationSettings
from controllers import PipelineController
from models import UNIDENTIFIED_LABEL, ChangeType, DepthMap, ImageData, SegmentationMask, SemanticLabel
from services.comparison import SpatialComparisonService
from services.inference import DeviceManager
from services.segmentation import RegionLabeler, SegFormerSegmenter
from tests.doubles import (
    BOX_ID,
    FAKE_ID2LABEL,
    WALL_ID,
    FakeSegmentationModelLoader,
    SceneDepthEstimator,
    SceneSegmenter,
    box_mask,
    scene_depth,
)
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

LOW, HIGH = 0.2, 0.95
THRESHOLD = 0.5
INTRINSICS = CameraIntrinsics()
RED, BLUE = (180, 60, 50), (40, 70, 200)


@pytest.fixture(scope="module")
def builder() -> SyntheticSceneBuilder:
    return SyntheticSceneBuilder(640, 480, seed=7)


def _mask(class_ids: np.ndarray, confidence: float | None = None) -> SegmentationMask:
    values = None if confidence is None else np.full(class_ids.shape, confidence, np.float32)
    return SegmentationMask(
        class_ids.astype(np.int32), FAKE_ID2LABEL, "fake", "CPU",
        ignored_class_ids=frozenset({WALL_ID}), confidence=values,
    )


def _box_region() -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((8, 8), WALL_ID, np.int32)
    ids[2:6, 2:6] = BOX_ID
    region = np.zeros((8, 8), bool)
    region[1:7, 1:7] = True
    return ids, region


# --- (a) to (c): the mask carries an optional confidence --------------------------------------------


def test_mask_confidence_is_optional():
    ids, _ = _box_region()
    assert SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU").confidence is None


def test_mask_rejects_a_confidence_of_another_shape():
    ids, _ = _box_region()
    with pytest.raises(ValueError):
        SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU", confidence=np.ones((4, 4), np.float32))


def test_segformer_reports_a_confidence_per_pixel(scene_image):
    segmenter = SegFormerSegmenter(
        SegmentationSettings(), DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    mask = segmenter.segment(scene_image)

    assert mask.confidence is not None
    assert mask.confidence.shape == mask.class_ids.shape
    assert mask.confidence.dtype == np.float32
    assert float(mask.confidence.min()) >= 0.0 and float(mask.confidence.max()) <= 1.0


def test_segmenter_without_the_confidence_hook_still_works(scene_image):
    mask = SceneSegmenter().segment(scene_image)

    assert mask.confidence is None
    assert np.all(mask.class_ids[box_mask(scene_image.pixels)] == BOX_ID)


# --- (d) to (f): RegionLabeler.describe -------------------------------------------------------------


def test_describe_without_confidence_is_unchanged():
    ids, region = _box_region()
    label = RegionLabeler(min_label_confidence=THRESHOLD).describe(_mask(ids), region, apply_confidence=True)

    assert label == SemanticLabel("caja", class_name="box", coverage=16 / 36)
    assert label.confidence is None


def test_describe_keeps_the_name_unless_asked_to_apply_the_confidence():
    ids, region = _box_region()
    label = RegionLabeler(min_label_confidence=THRESHOLD).describe(_mask(ids, LOW), region)

    assert label.name == "caja" and label.class_name == "box"
    assert label.confidence == pytest.approx(LOW)


def test_describe_hides_a_doubtful_name_only_when_applying_the_confidence():
    ids, region = _box_region()
    labeler = RegionLabeler(min_label_confidence=THRESHOLD)

    doubtful = labeler.describe(_mask(ids, LOW), region, apply_confidence=True)
    sure = labeler.describe(_mask(ids, HIGH), region, apply_confidence=True)

    assert doubtful.name == UNIDENTIFIED_LABEL and not doubtful.is_identified
    assert sure.name == "caja" and sure.confidence == pytest.approx(HIGH)


# --- (g): the setting -------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_minimum_label_confidence_must_be_between_zero_and_one(value):
    with pytest.raises(ValidationError):
        SegmentationSettings(min_label_confidence=value)


def test_minimum_label_confidence_is_immutable():
    settings = SegmentationSettings(min_label_confidence=0.3)
    assert settings.min_label_confidence == 0.3
    assert 0.0 <= SegmentationSettings().min_label_confidence <= 1.0
    with pytest.raises(ValidationError):
        settings.min_label_confidence = 0.9


# --- (h): end to end, only the displayed name changes -----------------------------------------------


class _ConfidenceSceneSegmenter(SceneSegmenter):
    """The scene segmenter, equally sure of every pixel."""

    def __init__(self, confidence: float) -> None:
        super().__init__()
        self._confidence = confidence

    def _predict_with_confidence(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        class_ids = self._predict(rgb)
        return class_ids, np.full(class_ids.shape, self._confidence, np.float32)


def test_doubtful_object_is_unidentified_with_the_same_type_and_distance(builder):
    before = ImageData(builder.render([SceneObject(center_x=0.3)]), "before.png")
    after = ImageData(builder.render([SceneObject(center_x=0.65)]), "after.png")
    settings = AppSettings(segmentation=SegmentationSettings(min_label_confidence=THRESHOLD))

    def run(confidence):
        controller = PipelineController(
            settings, depth_estimator=SceneDepthEstimator(), segmenter=_ConfidenceSceneSegmenter(confidence)
        )
        return controller.analyze(before, after).objects

    doubtful, sure = run(LOW), run(HIGH)

    assert [obj.label.name for obj in sure] == ["caja"]
    assert [obj.label.name for obj in doubtful] == [UNIDENTIFIED_LABEL]
    assert [obj.change_type for obj in doubtful] == [obj.change_type for obj in sure] == [ChangeType.DISPLACED]
    assert doubtful[0].distance_cm == sure[0].distance_cm


# --- (i) to (k): the comparison keeps pairing and filtering as before -------------------------------


def _depth(values: np.ndarray) -> DepthMap:
    return DepthMap(values=values.astype(np.float32), is_metric=True, model_name="test", device="CPU")


def _compare(before_rgb, after_rgb, depth_before, depth_after, mask_before, mask_after):
    service = SpatialComparisonService(
        ComparisonSettings(),
        labeler=RegionLabeler(min_label_confidence=THRESHOLD),
        device_manager=DeviceManager(DevicePreference.AUTO),
    )
    return service.compare(
        ImageData(before_rgb, "before"), ImageData(after_rgb, "after"),
        _depth(depth_before), _depth(depth_after), INTRINSICS,
        segmentation_before=mask_before, segmentation_after=mask_after,
    )


def _colored_boxes(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = (rgb[..., c].astype(np.int16) for c in range(3))
    return ((red - green) > 70) | ((blue - red) > 90)


def test_doubtful_objects_are_still_paired_by_their_class(builder):
    before = builder.render([SceneObject(center_x=0.3, color=RED)])
    after = builder.render([SceneObject(center_x=0.65, color=BLUE)])  # turned: other colors
    depth_before, depth_after = (scene_depth(rgb, box_offset_m=0.0) for rgb in (before, after))
    depth_before[_colored_boxes(before)] -= 0.6
    depth_after[_colored_boxes(after)] -= 0.6

    def boxes(rgb):
        return _mask(np.where(_colored_boxes(rgb), BOX_ID, WALL_ID), LOW)

    result = _compare(before, after, depth_before, depth_after, boxes(before), boxes(after))

    assert [obj.change_type for obj in result.objects] == [ChangeType.DISPLACED]
    assert result.objects[0].label.name == UNIDENTIFIED_LABEL


def test_doubtful_persistent_area_is_still_discarded(builder):
    scene = builder.render([SceneObject(center_x=0.3)])
    bumped = scene_depth(scene)
    bumped[100:200, 400:500] += 0.5
    marked = scene.copy()
    marked[100:200, 400:500] = (90, 90, 200)
    same_everywhere = _mask(np.full((480, 640), BOX_ID), LOW)

    result = _compare(scene, marked, scene_depth(scene), bumped, same_everywhere, same_everywhere)

    assert result.objects == ()


def test_name_comes_from_the_view_the_model_is_sure_about(builder):
    before = builder.render([SceneObject(center_x=0.3)])
    after = builder.render([SceneObject(center_x=0.65)])

    def boxes(rgb, confidence):
        return _mask(np.where(box_mask(rgb), BOX_ID, WALL_ID), confidence)

    one_sure = _compare(
        before, after, scene_depth(before), scene_depth(after), boxes(before, LOW), boxes(after, HIGH)
    )
    none_sure = _compare(
        before, after, scene_depth(before), scene_depth(after), boxes(before, LOW), boxes(after, LOW)
    )

    assert [obj.label.name for obj in one_sure.objects] == ["caja"]
    assert [obj.label.name for obj in none_sure.objects] == [UNIDENTIFIED_LABEL]
    assert one_sure.objects[0].distance_cm == none_sure.objects[0].distance_cm
