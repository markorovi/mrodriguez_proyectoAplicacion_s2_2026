"""Indoor scene check (RF-012): segmentation-based rejection of outdoor photos."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtTest import QTest

from config import AppSettings, DevicePreference, SegmentationSettings
from config.paths import SAMPLES_DIR
from controllers import PipelineController
from models import ImageData, NonIndoorSceneError, SceneDiffError, SegmentationMask
from models.alignment import AlignmentReport
from services.alignment import SpatialAlignmentService
from services.inference import DeviceManager
from services.segmentation import (
    ADE20K_OUTDOOR_CLASSES,
    RegionLabeler,
    SegFormerSegmenter,
    outdoor_class_ids,
)
from tests.doubles import (
    FAKE_ID2LABEL,
    SKY_ID,
    SKY_RGB,
    WALL_ID,
    FakeSegmentationModelLoader,
    OutdoorSegformerModel,
    SceneDepthEstimator,
)
from ui.image_status import ImageStatus
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

TREE_ID, PLANT_ID = 4, 17
ID2LABEL = {**FAKE_ID2LABEL, TREE_ID: "tree", PLANT_ID: "plant"}


def _mask(outdoor_rows: int = 0, class_id: int = SKY_ID, size: int = 100) -> SegmentationMask:
    """A ``size`` x ``size`` wall whose top ``outdoor_rows`` rows show ``class_id``."""
    ids = np.full((size, size), WALL_ID, np.int32)
    ids[:outdoor_rows] = class_id
    return SegmentationMask(ids, ID2LABEL, "fake", "CPU")


# ------------------------------------------------------------------ the labeler


def test_scene_without_outdoor_classes_is_indoor():
    assert RegionLabeler().validate_indoor_scene(_mask(0)) == (True, 0.0)


def test_scene_with_more_than_a_quarter_of_sky_and_trees_is_outdoor():
    mask = _mask(20)
    mask.class_ids[20:32] = TREE_ID

    is_indoor, percent = RegionLabeler().validate_indoor_scene(mask)

    assert not is_indoor
    assert percent == pytest.approx(32.0)


def test_exactly_a_quarter_of_sky_is_still_indoor():
    # A window with a view of the sky is normal in a room.
    assert RegionLabeler().validate_indoor_scene(_mask(25)) == (True, 25.0)
    assert RegionLabeler().validate_indoor_scene(_mask(26))[0] is False


def test_the_limit_can_be_changed():
    assert RegionLabeler(max_outdoor_percent=10.0).validate_indoor_scene(_mask(15))[0] is False
    with pytest.raises(ValueError):
        RegionLabeler(max_outdoor_percent=120.0)


def test_only_the_given_region_is_judged():
    mask = _mask(40)
    lower_half = np.zeros(mask.shape, bool)
    lower_half[50:] = True

    assert RegionLabeler().validate_indoor_scene(mask, lower_half) == (True, 0.0)
    assert RegionLabeler().validate_indoor_scene(mask, np.zeros(mask.shape, bool)) == (True, 0.0)


def test_potted_plants_do_not_make_a_room_outdoor():
    assert "plant" not in ADE20K_OUTDOOR_CLASSES
    assert RegionLabeler().validate_indoor_scene(_mask(60, PLANT_ID)) == (True, 0.0)


def test_outdoor_ids_are_resolved_by_name():
    id2label = {0: "wall", 2: " Sky ", 4: "tree", 17: "plant", 48: "skyscraper"}
    assert outdoor_class_ids(id2label) == frozenset({2, 4, 48})


def test_the_error_names_the_photos_in_plain_language():
    one = NonIndoorSceneError({"Después": 61.0})
    assert isinstance(one, SceneDiffError)
    assert one.views == ("Después",)
    assert str(one) == (
        "La fotografía «Después» no parece corresponder a un espacio interior. "
        "Por favor, elija imágenes tomadas dentro de una habitación o área de trabajo."
    )
    both = NonIndoorSceneError({"Antes": 40.0, "Después": 61.0})
    assert str(both).startswith("Las fotografías «Antes» y «Después» no parecen corresponder")


# ------------------------------------------------------------- the orchestrator


class _UnusedComparator:
    """Fails the test if the analysis reaches the comparison."""

    def compare(self, *args, **kwargs):
        pytest.fail("the comparison must not run for an outdoor photo")


@pytest.fixture(scope="module")
def builder() -> SyntheticSceneBuilder:
    return SyntheticSceneBuilder(640, 480, seed=7)


@pytest.fixture
def room(builder) -> ImageData:
    return ImageData(builder.render([SceneObject(center_x=0.3)]), "room.png")


def _outdoor(image: ImageData, sky_rows: int = 200) -> ImageData:
    """``image`` with its top ``sky_rows`` rows painted sky blue (42 % of 480 rows by default)."""
    pixels = image.pixels.copy()
    pixels[:sky_rows] = SKY_RGB
    return ImageData(pixels, "outdoor.png")


def _controller(settings: AppSettings | None = None, comparator=None, aligner=None) -> PipelineController:
    settings = settings or AppSettings()
    segmenter = SegFormerSegmenter(
        settings.segmentation,
        DeviceManager(DevicePreference.CPU),
        FakeSegmentationModelLoader(OutdoorSegformerModel()),
    )
    return PipelineController(
        settings,
        depth_estimator=SceneDepthEstimator(),
        segmenter=segmenter,
        comparator=comparator,
        aligner=aligner,
    )


def test_outdoor_after_photo_stops_the_analysis_before_the_comparison(room):
    with pytest.raises(NonIndoorSceneError) as error:
        _controller(comparator=_UnusedComparator()).analyze(room, _outdoor(room))

    assert error.value.views == ("Después",)
    assert error.value.outdoor_percent["Después"] > 25.0
    assert "«Después»" in str(error.value) and "«Antes»" not in str(error.value)


def test_both_outdoor_photos_are_named(room):
    outdoor = _outdoor(room)
    with pytest.raises(NonIndoorSceneError) as error:
        _controller(comparator=_UnusedComparator()).analyze(outdoor, outdoor)
    assert error.value.views == ("Antes", "Después")


def test_indoor_photos_with_a_little_sky_are_analyzed(room):
    window = _outdoor(room, sky_rows=60)  # 12.5 % of the photo
    report = _controller().analyze(room, window)
    assert report is not None


def test_the_check_can_be_turned_off(room):
    settings = AppSettings(segmentation=SegmentationSettings(check_indoor_scene=False))
    report = _controller(settings).analyze(room, _outdoor(room))
    assert report is not None


def test_without_segmentation_the_scene_is_not_checked(room):
    settings = AppSettings(segmentation=SegmentationSettings(enabled=False))
    controller = PipelineController(settings, depth_estimator=SceneDepthEstimator())
    assert controller.analyze(room, _outdoor(room)) is not None


def test_the_limit_comes_from_the_settings(room):
    settings = AppSettings(segmentation=SegmentationSettings(max_outdoor_percent=50.0))
    assert _controller(settings).analyze(room, _outdoor(room)) is not None


# ----------------------------------------------------------------- the window


def test_window_shows_the_error_and_keeps_both_photos(qapp, room, write_image):
    from ui.main_window import MainWindow, Step

    main = MainWindow(controller=_controller())
    try:
        outdoor_path = write_image(_outdoor(room).pixels, "outdoor.png")
        main.load_files([SAMPLES_DIR / "scene_before.png", outdoor_path])
        main.start_analysis()
        deadline = time.monotonic() + 60
        while main.step is Step.ANALYZING:
            assert time.monotonic() < deadline
            QTest.qWait(20)

        before, after = main.selection_panel.before_slot, main.selection_panel.after_slot
        assert main.step is Step.SELECTION
        assert "«Después» no parece corresponder a un espacio interior" in main.error_text
        # Only the rejected photo is flagged; both stay loaded and visible.
        assert before.status is ImageStatus.VALID
        assert after.status is ImageStatus.NOT_INDOOR
        assert after.status_text == "NO ES INTERIOR"
        assert "No parece un espacio interior" in after.detail_text
        assert before.acquired is not None and after.acquired is not None and after.has_preview
        assert not main.can_analyze

        # Replacing only the rejected photo clears the message and allows a new analysis.
        after.load_file(SAMPLES_DIR / "scene_after_moved.png")
        assert after.status is ImageStatus.VALID
        assert main.error_text == ""
        assert main.can_analyze
    finally:
        main.close()
        main.deleteLater()


def test_valid_badge_explains_what_was_checked_on_load(qapp):
    from ui.main_window import MainWindow

    main = MainWindow(controller=_controller())
    try:
        main.load_files([SAMPLES_DIR / "scene_before.png", SAMPLES_DIR / "scene_low_light.png"])
        before, after = main.selection_panel.before_slot, main.selection_panel.after_slot

        assert "Formato, color, resolución e iluminación correctos" in before.detail_text
        assert "Que sea un espacio interior se comprueba al analizar" in before.detail_text
        assert "Formato, color y resolución correctos" in after.detail_text
        assert "poco iluminada" in after.detail_text
    finally:
        main.close()
        main.deleteLater()


def test_worker_reports_which_photos_were_rejected(qapp, room):
    from ui.pipeline_worker import PipelineWorker

    worker = PipelineWorker(_controller(), room, _outdoor(room))
    events = []
    worker.scene_rejected.connect(lambda rejected: events.append(("rejected", sorted(rejected))))
    worker.error_occurred.connect(lambda message: events.append(("error", message[:13])))
    worker.run()

    assert events == [("rejected", ["Después"]), ("error", "La fotografía")]


# ---------------------------------------------- the check judges the original photos


class _FalseAlignmentAligner(SpatialAlignmentService):
    """Accepts a wrong homography that turns the "after" photo into something that looks like ``disguise``.

    Reproduces what the real models did with an indoor "before" and an outdoor
    "after": a handful of false matches passed RANSAC and the warped city view
    showed almost no sky, so judging the aligned view let the outdoor photo in.
    """

    def __init__(self, disguise: ImageData) -> None:
        super().__init__()
        self._disguise = disguise

    def align(self, before, after, after_depth):
        report = AlignmentReport(True, "sift", 100, 100, 11, 11, 1.0, "Imágenes alineadas con 11 correspondencias.")
        result = self.warp(before, after, after_depth, np.eye(3), report)
        return replace(result, image=ImageData(self._disguise.pixels.copy(), after.source))


def test_outdoor_after_photo_is_rejected_even_if_its_aligned_view_looks_indoor(room):
    controller = _controller(comparator=_UnusedComparator(), aligner=_FalseAlignmentAligner(disguise=room))
    with pytest.raises(NonIndoorSceneError) as error:
        controller.analyze(room, _outdoor(room))
    assert error.value.views == ("Después",)


def test_outdoor_before_photo_is_rejected_whatever_the_alignment_does(room):
    controller = _controller(comparator=_UnusedComparator(), aligner=_FalseAlignmentAligner(disguise=room))
    with pytest.raises(NonIndoorSceneError) as error:
        controller.analyze(_outdoor(room), room)
    assert error.value.views == ("Antes",)


def test_two_outdoor_photos_are_rejected_even_with_a_false_alignment(room):
    outdoor = _outdoor(room)
    controller = _controller(comparator=_UnusedComparator(), aligner=_FalseAlignmentAligner(disguise=room))
    with pytest.raises(NonIndoorSceneError) as error:
        controller.analyze(outdoor, outdoor)
    assert error.value.views == ("Antes", "Después")


def test_indoor_photos_give_the_same_result_with_and_without_the_check(builder, room):
    moved = ImageData(builder.render([SceneObject(center_x=0.65)]), "moved.png")
    checked = _controller().analyze(room, moved)
    unchecked = _controller(AppSettings(segmentation=SegmentationSettings(check_indoor_scene=False))).analyze(room, moved)
    assert [obj.summary for obj in checked.objects] == [obj.summary for obj in unchecked.objects]
    assert checked.objects and checked.objects[0].label == unchecked.objects[0].label
