"""Pairing of a lone vacated region with a lone occupied one, confirmed by point tracking.

An object whose colors change between views (other faces, other light) fails
the color pairing and, when the segmentation does not know it, the pairing by
label. When exactly one vacated and one occupied region are left, of
comparable size, they become one displaced object only if keypoints that move
with the object confirm it, and only if the occupied region is not an
estimation artifact (the same thing seen in both views).
"""

import numpy as np
import pytest
from pydantic import ValidationError

from config import CameraIntrinsics, ComparisonSettings, DevicePreference
from models import ChangeType, DepthMap, ImageData, SegmentationMask
from services.comparison import SpatialComparisonService
from services.inference import DeviceManager
from tests.doubles import BOX_ID, FAKE_ID2LABEL, WALL_ID, scene_depth
from utils.synthetic_scene import SyntheticSceneBuilder

INTRINSICS = CameraIntrinsics()
SIDE, Y, X_OLD, X_NEW = 90, 300, 150, 400
RED_TINT, BLUE_TINT, GREEN_TINT = (1.0, 0.35, 0.25), (0.25, 0.4, 1.0), (0.3, 1.0, 0.35)


@pytest.fixture(scope="module")
def builder() -> SyntheticSceneBuilder:
    return SyntheticSceneBuilder(640, 480, seed=7)


def _patch(tint, textured: bool, side: int = SIDE) -> np.ndarray:
    """A square object: a random gray texture (or a flat gray) tinted with ``tint``."""
    if textured:
        gray = np.random.default_rng(3).integers(60, 256, size=(side, side)).astype(np.float32)
    else:
        gray = np.full((side, side), 200.0, np.float32)
    return np.clip(gray[..., None] * np.array(tint, np.float32), 0, 255).astype(np.uint8)


def _scene(builder, placements):
    """Room image and depth with each ``(patch, x, y)`` pasted 0.6 m in front of the room."""
    rgb = builder.render([]).copy()
    depth = scene_depth(rgb, box_offset_m=0.0)
    for patch, x, y in placements:
        h, w = patch.shape[:2]
        rgb[y : y + h, x : x + w] = patch
        depth[y : y + h, x : x + w] -= 0.6
    return rgb, depth


def _depth(values):
    return DepthMap(values=values.astype(np.float32), is_metric=True, model_name="test", device="CPU")


def _compare(before, after, segmentation=(None, None), **overrides):
    (rgb_b, depth_b), (rgb_a, depth_a) = before, after
    service = SpatialComparisonService(
        ComparisonSettings(**overrides), device_manager=DeviceManager(DevicePreference.AUTO)
    )
    return service.compare(
        ImageData(rgb_b, "before"), ImageData(rgb_a, "after"), _depth(depth_b), _depth(depth_a), INTRINSICS,
        segmentation_before=segmentation[0], segmentation_after=segmentation[1],
    ).objects


def _mask(boxes) -> SegmentationMask:
    ids = np.full((480, 640), WALL_ID, np.int32)
    for x, y, side in boxes:
        ids[y : y + side, x : x + side] = BOX_ID
    return SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID}))


def _types(objects):
    return sorted(obj.change_type.name for obj in objects)


def _summary(objects):
    return [(o.change_type, o.area_px, o.bbox_before, o.bbox_after, o.distance_cm) for o in objects]


def _recolored_move(builder, textured=True):
    before = _scene(builder, [(_patch(RED_TINT, textured), X_OLD, Y)])
    after = _scene(builder, [(_patch(BLUE_TINT, textured), X_NEW, Y)])
    return before, after


def test_object_whose_colors_change_is_paired_when_tracking_confirms_it(builder):
    before, after = _recolored_move(builder)

    paired = _compare(before, after)
    unpaired = _compare(before, after, pair_leftover_by_tracking=False)

    assert _types(unpaired) == ["ADDED", "REMOVED"]
    assert [obj.change_type for obj in paired] == [ChangeType.DISPLACED]
    expected_cm = 100 * (X_NEW - X_OLD) * paired[0].position_before.z / INTRINSICS.fx
    assert paired[0].distance_cm == pytest.approx(expected_cm, rel=0.25)


def test_persistent_vacated_region_is_still_paired(builder):
    before, after = _recolored_move(builder)
    # The segmentation sees the same object over the old place in both views: the vacated
    # region looks persistent, the occupied one does not.
    segmentation = (_mask([(X_OLD, Y, SIDE)]), _mask([(X_OLD, Y, SIDE), (X_NEW, Y, SIDE)]))

    assert [obj.change_type for obj in _compare(before, after, segmentation)] == [ChangeType.DISPLACED]


def test_flat_object_without_points_to_follow_stays_removed_and_added(builder):
    before, after = _recolored_move(builder, textured=False)
    assert _types(_compare(before, after)) == ["ADDED", "REMOVED"]


def test_persistent_occupied_region_is_not_paired(builder):
    before, after = _recolored_move(builder)
    # Same class over the new place in both views: the occupied region is an artifact.
    segmentation = (_mask([(X_OLD, Y, SIDE), (X_NEW, Y, SIDE)]), _mask([(X_NEW, Y, SIDE)]))

    objects = _compare(before, after, segmentation)
    assert ChangeType.DISPLACED not in [obj.change_type for obj in objects]
    assert _summary(objects) == _summary(_compare(before, after, segmentation, pair_leftover_by_tracking=False))


def test_two_vacated_regions_and_one_occupied_are_not_paired(builder):
    before = _scene(builder, [(_patch(RED_TINT, True), X_OLD, Y), (_patch(GREEN_TINT, False), 300, 120)])
    after = _scene(builder, [(_patch(BLUE_TINT, True), X_NEW, Y)])

    objects = _compare(before, after)
    assert _types(objects) == ["ADDED", "REMOVED", "REMOVED"]
    assert _summary(objects) == _summary(_compare(before, after, pair_leftover_by_tracking=False))


def test_regions_of_very_different_size_are_not_paired(builder):
    before = _scene(builder, [(_patch(RED_TINT, True), X_OLD, Y)])
    after = _scene(builder, [(_patch(BLUE_TINT, True, side=2 * SIDE), X_NEW, Y - SIDE)])  # area ratio 4 > 3

    objects = _compare(before, after)
    assert ChangeType.DISPLACED not in [obj.change_type for obj in objects]
    assert _summary(objects) == _summary(_compare(before, after, pair_leftover_by_tracking=False))


def test_color_and_label_pairing_come_first(builder):
    same_colors = (
        _scene(builder, [(_patch(RED_TINT, True), X_OLD, Y)]),
        _scene(builder, [(_patch(RED_TINT, True), X_NEW, Y)]),
    )
    by_label = _recolored_move(builder, textured=False)
    labels = (_mask([(X_OLD, Y, SIDE)]), _mask([(X_NEW, Y, SIDE)]))

    for (before, after), segmentation in ((same_colors, (None, None)), (by_label, labels)):
        objects = _compare(before, after, segmentation)
        assert [obj.change_type for obj in objects] == [ChangeType.DISPLACED]
        assert _summary(objects) == _summary(_compare(before, after, segmentation, pair_leftover_by_tracking=False))


def test_setting_is_validated_and_immutable():
    assert ComparisonSettings().pair_leftover_by_tracking is True
    with pytest.raises(ValidationError):
        ComparisonSettings(pair_leftover_by_tracking="quizás")
    settings = ComparisonSettings(pair_leftover_by_tracking=False)
    with pytest.raises(ValidationError):
        settings.pair_leftover_by_tracking = True
