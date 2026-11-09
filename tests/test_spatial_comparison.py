"""Pinhole back-projection, 3-D distances and classification of changed objects."""

import numpy as np
import pytest
import torch

from config import CameraIntrinsics, ComparisonSettings, DevicePreference
from models import (
    UNIDENTIFIED_LABEL,
    ChangedObject,
    ChangeType,
    DepthMap,
    ImageData,
    Point3D,
    SegmentationMask,
    SemanticLabel,
)
from services.comparison import SpatialComparisonService
from services.inference import DeviceManager
from services.segmentation import RegionLabeler
from tests.doubles import BOX_ID, FAKE_ID2LABEL, WALL_ID, box_mask, scene_depth
from utils.geometry import backproject_depth, backproject_pixels
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

INTRINSICS = CameraIntrinsics()
BEFORE_X, AFTER_X = 0.3, 0.65


# --- Pinhole back-projection -------------------------------------------------


def test_principal_point_projects_onto_the_optical_axis():
    point = backproject_pixels(torch.tensor(INTRINSICS.cx), torch.tensor(INTRINSICS.cy), torch.tensor(2.5), INTRINSICS)
    assert point.tolist() == pytest.approx([0.0, 0.0, 2.5])


def test_one_focal_length_off_axis_is_one_depth_away_laterally():
    u = torch.tensor(INTRINSICS.cx + INTRINSICS.fx)
    v = torch.tensor(INTRINSICS.cy - INTRINSICS.fy)
    point = backproject_pixels(u, v, torch.tensor(3.0), INTRINSICS)
    assert point.tolist() == pytest.approx([3.0, -3.0, 3.0], rel=1e-5)


def test_depth_map_back_projection_matches_the_pinhole_formulas():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 5.0, size=(48, 64)).astype(np.float32)
    points = backproject_depth(torch.from_numpy(depth), INTRINSICS).numpy()

    v, u = np.mgrid[0:48, 0:64]
    assert points.shape == (48, 64, 3)
    np.testing.assert_allclose(points[..., 0], (u - INTRINSICS.cx) * depth / INTRINSICS.fx, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(points[..., 1], (v - INTRINSICS.cy) * depth / INTRINSICS.fy, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(points[..., 2], depth)


def test_back_projection_rejects_non_2d_depth():
    with pytest.raises(ValueError):
        backproject_depth(torch.ones(2, 3, 4), INTRINSICS)


# --- 3-D displacement ----------------------------------------------------------


def test_distance_is_euclidean_norm_of_displacement_in_centimeters():
    obj = ChangedObject(
        object_id=1,
        change_type=ChangeType.DISPLACED,
        label=SemanticLabel("caja", "box", 1.0),
        depth_change_cm=0.0,
        area_px=100,
        position_before=Point3D(0.1, 0.2, 2.0),
        position_after=Point3D(0.4, 0.2, 2.4),
    )
    displacement = obj.displacement
    assert (displacement.x, displacement.y, displacement.z) == pytest.approx((0.3, 0.0, 0.4))
    assert obj.distance_cm == pytest.approx(50.0)
    assert obj.summary == "caja: DESPLAZAMIENTO (50.0 cm)"


def test_removed_object_has_no_displacement():
    obj = ChangedObject(1, ChangeType.REMOVED, SemanticLabel("caja"), 60.0, 100, position_before=Point3D(0, 0, 2))
    assert obj.displacement is None and obj.distance_cm is None


# --- Comparison of synthetic scenes ----------------------------------------------


@pytest.fixture(scope="module")
def builder() -> SyntheticSceneBuilder:
    return SyntheticSceneBuilder(640, 480, seed=7)


def _render(builder, *centers) -> np.ndarray:
    return builder.render([SceneObject(center_x=x) for x in centers])


def _depth(values: np.ndarray) -> DepthMap:
    return DepthMap(values=values.astype(np.float32), is_metric=True, model_name="test", device="CPU")


def _service(**overrides) -> SpatialComparisonService:
    return SpatialComparisonService(
        ComparisonSettings(**overrides), device_manager=DeviceManager(DevicePreference.AUTO)
    )


def _compare(before_rgb, after_rgb, service=None, depth_before=None, depth_after=None, **kwargs):
    service = service or _service()
    return service.compare(
        ImageData(before_rgb, "before"),
        ImageData(after_rgb, "after"),
        _depth(scene_depth(before_rgb) if depth_before is None else depth_before),
        _depth(scene_depth(after_rgb) if depth_after is None else depth_after),
        INTRINSICS,
        **kwargs,
    )


def _segmentation(rgb: np.ndarray) -> SegmentationMask:
    ids = np.where(box_mask(rgb), BOX_ID, WALL_ID).astype(np.int32)
    return SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID}))


def test_displaced_object_is_measured_in_3d(builder):
    before, after = _render(builder, BEFORE_X), _render(builder, AFTER_X)
    result = _compare(before, after)

    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.change_type is ChangeType.DISPLACED
    assert obj.bbox_before.center[0] == pytest.approx(BEFORE_X * 640, abs=15)
    assert obj.bbox_after.center[0] == pytest.approx(AFTER_X * 640, abs=15)

    # The box slides sideways on the floor: almost all of the motion is along X,
    # and its size follows from the pixel shift through the pinhole model.
    shift_px = obj.bbox_after.center[0] - obj.bbox_before.center[0]
    expected_m = shift_px * obj.position_before.z / INTRINSICS.fx
    assert obj.displacement.x == pytest.approx(expected_m, rel=0.1)
    assert abs(obj.displacement.y) < 0.05 and abs(obj.displacement.z) < 0.05
    assert obj.distance_cm == pytest.approx(expected_m * 100, rel=0.1)
    assert obj.depth_change_cm == pytest.approx(60.0, abs=1.0)


def test_removed_object_is_classified_as_removed(builder):
    result = _compare(_render(builder, BEFORE_X), _render(builder))

    assert [obj.change_type for obj in result.objects] == [ChangeType.REMOVED]
    obj = result.objects[0]
    assert obj.position_before is not None and obj.position_after is None
    assert obj.bbox_after is None
    assert obj.depth_change_cm == pytest.approx(60.0, abs=1.0)
    assert result.farther_mask.any() and not result.closer_mask.any()


def test_new_object_is_classified_as_added(builder):
    result = _compare(_render(builder), _render(builder, AFTER_X))

    assert [obj.change_type for obj in result.objects] == [ChangeType.ADDED]
    assert result.objects[0].position_after is not None


def test_unchanged_scene_reports_nothing(builder):
    scene = _render(builder, BEFORE_X)
    result = _compare(scene, scene)

    assert result.objects == ()
    assert not result.farther_mask.any() and not result.closer_mask.any()
    assert result.scale_factor == pytest.approx(1.0)


def test_difference_map_is_after_minus_before(builder):
    before, after = _render(builder, BEFORE_X), _render(builder)
    result = _compare(before, after)

    vacated = box_mask(before) & ~box_mask(after)
    assert result.difference_m.shape == (480, 640)
    assert np.median(result.difference_m[vacated]) == pytest.approx(0.6, abs=0.01)


def test_change_below_threshold_is_ignored(builder):
    before, after = _render(builder, BEFORE_X), _render(builder)
    shallow = scene_depth(before, box_offset_m=0.03)
    result = _compare(before, after, depth_before=shallow, depth_after=scene_depth(after))
    assert result.objects == ()


def test_global_depth_scale_drift_is_compensated(builder):
    scene = _render(builder, BEFORE_X)
    result = _compare(scene, scene, depth_after=scene_depth(scene) * 1.08)

    assert result.scale_factor == pytest.approx(1 / 1.08, rel=1e-4)
    assert result.objects == ()


def test_without_normalization_scale_drift_reads_as_change(builder):
    scene = _render(builder, BEFORE_X)
    drifted = scene_depth(scene) * 1.08
    result = _compare(scene, scene, _service(normalize_depth_scale=False, appearance_gating=False), depth_after=drifted)
    assert result.scale_factor == 1.0
    assert result.objects


def test_depth_change_without_visual_change_is_rejected_by_gating(builder):
    scene = _render(builder, BEFORE_X)
    bumped = scene_depth(scene)
    bumped[100:200, 400:500] += 0.5

    assert _compare(scene, scene, depth_after=bumped).objects == ()
    ungated = _compare(scene, scene, _service(appearance_gating=False), depth_after=bumped)
    assert [obj.change_type for obj in ungated.objects] == [ChangeType.REMOVED]


def test_global_lighting_change_is_not_a_change(builder):
    scene = _render(builder, BEFORE_X)
    darker = np.clip(scene * 0.6, 0, 255).astype(np.uint8)
    bumped = scene_depth(scene)
    bumped[100:200, 400:500] += 0.5
    assert _compare(scene, darker, depth_before=scene_depth(scene), depth_after=bumped).objects == ()


def test_changes_outside_the_overlap_mask_are_ignored(builder):
    before, after = _render(builder, BEFORE_X), _render(builder)
    valid = np.ones((480, 640), bool)
    valid[:, : 640 // 2] = False

    assert _compare(before, after, valid_mask=valid).objects == ()


def test_small_regions_are_discarded(builder):
    scene = _render(builder)
    speckled = scene.copy()
    speckled[300:306, 500:506] = (255, 0, 0)
    depth_after = scene_depth(scene)
    depth_after[300:306, 500:506] -= 0.5
    assert _compare(scene, speckled, depth_after=depth_after).objects == ()


def test_objects_are_named_from_segmentation(builder):
    before, after = _render(builder, BEFORE_X), _render(builder, AFTER_X)
    result = _compare(
        before, after, segmentation_before=_segmentation(before), segmentation_after=_segmentation(after)
    )
    label = result.objects[0].label
    assert label.name == "caja"
    assert label.class_name == "box"
    assert label.coverage > 0.5


def test_objects_without_segmentation_are_unidentified(builder):
    result = _compare(_render(builder, BEFORE_X), _render(builder))
    assert result.objects[0].label.name == UNIDENTIFIED_LABEL
    assert not result.objects[0].label.is_identified


def test_dissimilar_regions_are_not_paired(builder):
    before = builder.render([SceneObject(center_x=BEFORE_X)])
    after = builder.render([SceneObject(center_x=AFTER_X, color=(40, 60, 200))])
    depth_after = scene_depth(after)
    blue = (after[..., 2].astype(int) - after[..., 1]) > 60
    depth_after[blue] -= 0.6

    types = sorted(obj.change_type.name for obj in _compare(before, after, depth_after=depth_after).objects)
    assert types == ["ADDED", "REMOVED"]


def test_object_ids_follow_area_order(builder):
    before = builder.render([SceneObject(center_x=0.25, size=0.25), SceneObject(center_x=0.7, size=0.12)])
    after = builder.render([])
    result = _compare(before, after)

    assert [obj.object_id for obj in result.objects] == [1, 2]
    assert result.objects[0].area_px > result.objects[1].area_px


def test_intrinsics_are_rescaled_to_the_image_resolution(builder):
    small_builder = SyntheticSceneBuilder(320, 240, seed=7)
    before, after = small_builder.render([SceneObject(center_x=BEFORE_X)]), small_builder.render([])
    result = _compare(before, after, _service(min_region_fraction=0.001))

    obj = result.objects[0]
    half = INTRINSICS.scaled_to(320, 240)
    u = obj.position_before.x * half.fx / obj.position_before.z + half.cx
    assert u == pytest.approx(BEFORE_X * 320, abs=12)


def test_mismatched_shapes_are_rejected(builder):
    scene = _render(builder)
    service = _service()
    with pytest.raises(ValueError):
        service.compare(
            ImageData(scene, "b"), ImageData(scene, "a"), _depth(scene_depth(scene)),
            _depth(np.ones((10, 10))), INTRINSICS,
        )


def test_comparison_runs_on_the_selected_device(builder):
    scene = _render(builder)
    service = _service()
    _compare(scene, scene, service)
    expected = "CPU" if not torch.cuda.is_available() else torch.cuda.get_device_name(0)
    assert service.device_name == expected


# --- Region naming ---------------------------------------------------------------


def test_labeler_requires_minimum_coverage():
    ids = np.full((10, 10), WALL_ID, np.int32)
    ids[0, :5] = BOX_ID
    mask = SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID}))
    region = np.ones((10, 10), bool)

    assert RegionLabeler(min_coverage=0.25).describe(mask, region).name == UNIDENTIFIED_LABEL
    assert RegionLabeler(min_coverage=0.05).describe(mask, region).name == "caja"


def test_labeler_ignores_structural_classes_and_unknown_names():
    ids = np.full((4, 4), WALL_ID, np.int32)
    only_structure = SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID}))
    assert RegionLabeler().describe(only_structure, np.ones((4, 4), bool)).class_name is None

    unknown = SegmentationMask(np.full((4, 4), 7, np.int32), {7: "spaceship"}, "fake", "CPU")
    assert RegionLabeler().describe(unknown, np.ones((4, 4), bool)).name == UNIDENTIFIED_LABEL


def test_labeler_tolerates_padded_class_names():
    mask = SegmentationMask(np.full((4, 4), 7, np.int32), {7: "bed "}, "fake", "CPU")
    assert RegionLabeler().describe(mask, np.ones((4, 4), bool)).name == "cama"


# --- Merging of touching regions and choice of the naming view -----------------


def _removed_with_closer_base(builder):
    """A removed box whose revealed lower half is wrongly estimated closer, as monocular models do."""
    before, after = _render(builder, BEFORE_X), _render(builder)
    depth_before = scene_depth(before)
    depth_after = scene_depth(after)
    rows = np.nonzero(box_mask(before).any(axis=1))[0]
    lower = box_mask(before).copy()
    lower[: (rows.min() + rows.max()) // 2] = False
    depth_after[lower] = depth_before[lower] - 0.2
    return before, after, depth_before, depth_after, rows


def test_touching_region_of_opposite_sign_joins_the_removed_object(builder):
    before, after, depth_before, depth_after, rows = _removed_with_closer_base(builder)
    result = _compare(before, after, depth_before=depth_before, depth_after=depth_after)

    assert [obj.change_type for obj in result.objects] == [ChangeType.REMOVED]
    box = result.objects[0].bbox_before
    assert box.y <= rows.min() + 5 and box.y + box.height >= rows.max() - 5


def test_without_merging_the_base_reads_as_a_new_object(builder):
    before, after, depth_before, depth_after, _ = _removed_with_closer_base(builder)
    result = _compare(
        before, after, _service(merge_adjacent_regions=False), depth_before=depth_before, depth_after=depth_after
    )
    assert sorted(obj.change_type.name for obj in result.objects) == ["ADDED", "REMOVED"]


def test_object_is_named_from_the_view_that_recognizes_it_best(builder):
    before, after = _render(builder, BEFORE_X), _render(builder, AFTER_X)
    chair_id = 19
    id2label = {**FAKE_ID2LABEL, chair_id: "chair"}

    # "before" sees a weak chair, "after" clearly sees a box.
    ids_before = np.full((480, 640), WALL_ID, np.int32)
    box_rows = np.nonzero(box_mask(before).any(axis=1))[0]
    ids_before[box_rows.min() : box_rows.min() + 12] = chair_id
    weak = SegmentationMask(ids_before, id2label, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID}))
    strong = _segmentation(after)

    labeler = RegionLabeler(min_coverage=0.05)
    service = SpatialComparisonService(ComparisonSettings(), labeler, DeviceManager(DevicePreference.AUTO))
    result = _compare(before, after, service, segmentation_before=weak, segmentation_after=strong)

    assert result.objects[0].label.name == "caja"


# --- Keypoint tracking of displaced objects and persistent areas ---------------


def _textured_move(builder, shift_px: int):
    """A textured box that slides ``shift_px`` to the right, less than its own width."""
    rng = np.random.default_rng(3)
    texture = rng.integers(0, 256, size=(90, 90, 3), dtype=np.uint8)
    texture[..., 0] = np.maximum(texture[..., 0], 200)  # keep it reddish so it differs from the room
    scene = builder.render([])
    before, after = scene.copy(), scene.copy()
    x, y = 250, 300
    before[y : y + 90, x : x + 90] = texture
    after[y : y + 90, x + shift_px : x + shift_px + 90] = texture
    depth_before, depth_after = scene_depth(scene), scene_depth(scene)
    depth_before[y : y + 90, x : x + 90] -= 0.6
    depth_after[y : y + 90, x + shift_px : x + shift_px + 90] -= 0.6
    return before, after, depth_before, depth_after


def test_small_move_is_measured_by_following_the_object(builder):
    shift = 24
    before, after, depth_before, depth_after = _textured_move(builder, shift)
    obj = _compare(before, after, depth_before=depth_before, depth_after=depth_after).objects[0]

    assert obj.change_type is ChangeType.DISPLACED
    expected_m = shift * obj.position_before.z / INTRINSICS.fx
    assert obj.distance_cm == pytest.approx(expected_m * 100, rel=0.2)


def test_region_centroids_overestimate_small_moves(builder):
    shift = 24
    before, after, depth_before, depth_after = _textured_move(builder, shift)
    untracked = _compare(
        before, after, _service(track_displaced_objects=False), depth_before=depth_before, depth_after=depth_after
    ).objects[0]

    expected_m = shift * untracked.position_before.z / INTRINSICS.fx
    # The vacated and occupied slivers sit on opposite edges of the box.
    assert untracked.distance_cm > 2 * expected_m * 100


def test_area_showing_the_same_object_in_both_views_is_not_a_change(builder):
    scene = _render(builder, BEFORE_X)
    bumped = scene_depth(scene)
    bumped[100:200, 400:500] += 0.5
    marked = scene.copy()
    marked[100:200, 400:500] = (90, 90, 200)  # a visual change so gating lets it through

    same_everywhere = SegmentationMask(
        np.full((480, 640), BOX_ID, np.int32), FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID})
    )
    kept = _compare(scene, marked, depth_after=bumped)
    dropped = _compare(
        scene, marked, depth_after=bumped, segmentation_before=same_everywhere, segmentation_after=same_everywhere
    )
    assert [obj.change_type for obj in kept.objects] == [ChangeType.REMOVED]
    assert dropped.objects == ()


def test_persistence_filter_can_be_disabled(builder):
    scene = _render(builder, BEFORE_X)
    bumped = scene_depth(scene)
    bumped[100:200, 400:500] += 0.5
    marked = scene.copy()
    marked[100:200, 400:500] = (90, 90, 200)
    same_everywhere = SegmentationMask(
        np.full((480, 640), BOX_ID, np.int32), FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID})
    )
    result = _compare(
        scene, marked, _service(drop_persistent_regions=False), depth_after=bumped,
        segmentation_before=same_everywhere, segmentation_after=same_everywhere,
    )
    assert [obj.change_type for obj in result.objects] == [ChangeType.REMOVED]


# --- Objects brought together and objects that turned --------------------------

RED, BLUE = (180, 60, 50), (40, 70, 200)


def _colored_depth(rgb: np.ndarray) -> np.ndarray:
    """Room depth with every red or blue box 0.6 m closer than the room behind it."""
    depth = scene_depth(rgb, box_offset_m=0.0)
    red, green, blue = (rgb[..., c].astype(np.int16) for c in range(3))
    depth[((red - green) > 70) | ((blue - red) > 90)] -= 0.6
    return depth


def test_objects_brought_together_are_two_displacements(builder):
    before = builder.render([SceneObject(center_x=0.2, color=RED), SceneObject(center_x=0.8, color=BLUE)])
    # Side by side and overlapping: one connected occupied region in "after".
    after = builder.render([SceneObject(center_x=0.45, color=RED), SceneObject(center_x=0.58, color=BLUE)])
    result = _compare(before, after, depth_before=_colored_depth(before), depth_after=_colored_depth(after))

    assert [obj.change_type for obj in result.objects] == [ChangeType.DISPLACED, ChangeType.DISPLACED]
    starts = sorted(obj.bbox_before.center[0] for obj in result.objects)
    assert starts == [pytest.approx(0.2 * 640, abs=20), pytest.approx(0.8 * 640, abs=20)]
    # Each object keeps its own footprint inside the shared region.
    ends = {round(obj.bbox_before.center[0] / 640, 1): obj.bbox_after.center[0] for obj in result.objects}
    assert ends[0.2] < ends[0.8]


def test_without_splitting_one_of_the_objects_brought_together_reads_as_removed(builder):
    before = builder.render([SceneObject(center_x=0.2, color=RED), SceneObject(center_x=0.8, color=BLUE)])
    after = builder.render([SceneObject(center_x=0.45, color=RED), SceneObject(center_x=0.58, color=BLUE)])
    result = _compare(
        before, after, _service(split_shared_regions=False),
        depth_before=_colored_depth(before), depth_after=_colored_depth(after),
    )
    # One-to-one pairing gives the shared region to one object only.
    assert sorted(obj.change_type.name for obj in result.objects) == ["DISPLACED", "REMOVED"]


def test_object_that_turned_is_paired_by_its_label(builder):
    before = builder.render([SceneObject(center_x=BEFORE_X, color=RED)])
    after = builder.render([SceneObject(center_x=AFTER_X, color=BLUE)])  # another face, other colors
    depth_before, depth_after = _colored_depth(before), _colored_depth(after)

    def boxes(rgb):
        red, green, blue = (rgb[..., c].astype(np.int16) for c in range(3))
        ids = np.where(((red - green) > 70) | ((blue - red) > 90), BOX_ID, WALL_ID).astype(np.int32)
        return SegmentationMask(ids, FAKE_ID2LABEL, "fake", "CPU", ignored_class_ids=frozenset({WALL_ID}))

    named = _compare(
        before, after, depth_before=depth_before, depth_after=depth_after,
        segmentation_before=boxes(before), segmentation_after=boxes(after),
    )
    unnamed = _compare(before, after, depth_before=depth_before, depth_after=depth_after)

    assert [obj.change_type for obj in named.objects] == [ChangeType.DISPLACED]
    assert named.objects[0].label.name == "caja"
    assert sorted(obj.change_type.name for obj in unnamed.objects) == ["ADDED", "REMOVED"]
