"""Camera parameters for photos from any camera: calibration, EXIF focal length or field of view."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from config import AppSettings, CameraEstimationSettings, CameraIntrinsics, DevicePreference, load_settings
from controllers import PipelineController
from models import ChangeType, ImageData, IntrinsicsSource
from services.capture.file_image_loader import FileImageLoader
from services.inference import DeviceManager
from services.segmentation import SegFormerSegmenter
from tests.doubles import FakeSegmentationModelLoader, SceneDepthEstimator
from utils.camera_model import (
    horizontal_fov_deg,
    intrinsics_from_focal_35mm,
    intrinsics_from_fov,
    matches_aspect_ratio,
    resolve_intrinsics,
)
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

EXIF_IFD, FOCAL_35MM_TAG = 0x8769, 0xA405


def _image(width: int = 1920, height: int = 1080, focal_35mm=None) -> ImageData:
    return ImageData(np.zeros((height, width, 3), np.uint8), "photo.jpg", focal_length_35mm=focal_35mm)


def _jpeg_with_focal(path, focal_35mm: int, size=(800, 600)) -> None:
    exif = Image.Exif()
    exif.get_ifd(EXIF_IFD)[FOCAL_35MM_TAG] = focal_35mm
    Image.new("RGB", size, (120, 110, 100)).save(path, exif=exif.tobytes())


# ------------------------------------------------------------ estimation formulas


def test_field_of_view_gives_square_pixels_and_centered_principal_point():
    k = intrinsics_from_fov(1920, 1080, 90.0)
    assert k.fx == pytest.approx(960.0)
    assert k.fy == k.fx
    assert (k.cx, k.cy) == (960.0, 540.0)
    assert (k.reference_width, k.reference_height) == (1920, 1080)


def test_default_field_of_view_matches_the_previous_reference_camera():
    # The previous fixed intrinsics had fx = 518.86 px at 640 px wide.
    k = intrinsics_from_fov(640, 480, CameraEstimationSettings().horizontal_fov_deg)
    assert k.fx == pytest.approx(518.86, rel=0.005)


def test_35mm_equivalent_focal_length_uses_the_image_diagonal():
    # On a 3:2 image the diagonal rule reduces to fx = width * f35 / 36.
    k = intrinsics_from_focal_35mm(3600, 2400, 36.0)
    assert k.fx == pytest.approx(3600.0)
    assert horizontal_fov_deg(k) == pytest.approx(2 * math.degrees(math.atan(0.5)))


def test_35mm_focal_length_is_independent_of_orientation():
    landscape = intrinsics_from_focal_35mm(4000, 3000, 26.0)
    portrait = intrinsics_from_focal_35mm(3000, 4000, 26.0)
    assert landscape.fx == pytest.approx(portrait.fx)
    assert horizontal_fov_deg(landscape) == pytest.approx(67.4, abs=0.5)  # a typical phone main camera


@pytest.mark.parametrize(
    "call",
    [
        lambda: intrinsics_from_fov(0, 480, 60.0),
        lambda: intrinsics_from_fov(640, 480, 180.0),
        lambda: intrinsics_from_focal_35mm(640, 480, 0.0),
    ],
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_aspect_ratio_match_tolerates_rounding_only():
    calibration = CameraIntrinsics(fx=900, fy=900, cx=640, cy=360, reference_width=1280, reference_height=720)
    assert matches_aspect_ratio(calibration, 1920, 1080)
    assert matches_aspect_ratio(calibration, 1366, 768)  # 16:9 rounded
    assert not matches_aspect_ratio(calibration, 640, 480)


# ---------------------------------------------------------------- source priority


def test_a_matching_calibration_is_scaled_and_takes_priority_over_exif():
    calibration = CameraIntrinsics(fx=900, fy=905, cx=650, cy=355, reference_width=1280, reference_height=720)
    estimate = resolve_intrinsics(_image(1920, 1080, focal_35mm=26.0), calibration)

    assert estimate.source is IntrinsicsSource.CALIBRATED
    assert (estimate.intrinsics.fx, estimate.intrinsics.fy) == pytest.approx((1350.0, 1357.5))
    assert (estimate.intrinsics.cx, estimate.intrinsics.cy) == pytest.approx((975.0, 532.5))


def test_a_calibration_of_another_aspect_ratio_is_not_used():
    calibration = CameraIntrinsics()  # 640x480, 4:3
    estimate = resolve_intrinsics(_image(1920, 1080), calibration)
    assert estimate.source is IntrinsicsSource.FIELD_OF_VIEW


def test_exif_focal_length_is_used_without_calibration():
    estimate = resolve_intrinsics(_image(4000, 3000, focal_35mm=26.0))
    assert estimate.source is IntrinsicsSource.EXIF
    assert estimate.intrinsics.fx == pytest.approx(26.0 * 5000 / math.hypot(36, 24))


def test_exif_can_be_disabled():
    settings = CameraEstimationSettings(use_exif=False, horizontal_fov_deg=70.0)
    estimate = resolve_intrinsics(_image(4000, 3000, focal_35mm=26.0), settings=settings)
    assert estimate.source is IntrinsicsSource.FIELD_OF_VIEW
    assert horizontal_fov_deg(estimate.intrinsics) == pytest.approx(70.0)


def test_field_of_view_is_the_last_resort():
    estimate = resolve_intrinsics(_image(1280, 720))
    assert estimate.source is IntrinsicsSource.FIELD_OF_VIEW
    assert horizontal_fov_deg(estimate.intrinsics) == pytest.approx(CameraEstimationSettings().horizontal_fov_deg)


# --------------------------------------------------------------------- EXIF reading


def test_loader_reads_the_35mm_focal_length_from_exif(tmp_path):
    path = tmp_path / "fotó con EXIF.jpg"
    _jpeg_with_focal(path, 26)
    assert FileImageLoader(path).acquire().focal_length_35mm == 26.0


def test_loader_without_exif_has_no_focal_length(write_image):
    path = write_image(np.zeros((480, 640, 3), np.uint8), "plain.png")
    assert FileImageLoader(path).acquire().focal_length_35mm is None


def test_unknown_focal_length_in_exif_is_ignored(tmp_path):
    path = tmp_path / "zero.jpg"
    _jpeg_with_focal(path, 0)
    assert FileImageLoader(path).acquire().focal_length_35mm is None


# ------------------------------------------------------------------------ settings


def test_no_calibration_is_configured_by_default():
    settings = AppSettings()
    assert settings.camera is None
    assert settings.camera_estimation == CameraEstimationSettings()


def test_calibrated_profile_and_field_of_view_load_from_json(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({
            "camera": {"fx": 1000, "fy": 1002, "cx": 640, "cy": 360, "reference_width": 1280, "reference_height": 720},
            "camera_estimation": {"horizontal_fov_deg": 70},
        }),
        "utf-8",
    )
    settings = load_settings(path)
    assert settings.camera.fy == 1002
    assert settings.camera_estimation.horizontal_fov_deg == 70


def test_implausible_field_of_view_is_rejected():
    with pytest.raises(ValueError):
        CameraEstimationSettings(horizontal_fov_deg=5)


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def scene_pair():
    builder = SyntheticSceneBuilder(640, 480, seed=7)
    before = ImageData(builder.render([SceneObject(center_x=0.3)]), "before.png")
    after = ImageData(builder.render([SceneObject(center_x=0.65)]), "after.png")
    return before, after


def _analyze(pair, settings: AppSettings):
    segmenter = SegFormerSegmenter(
        settings.segmentation, DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    controller = PipelineController(settings, depth_estimator=SceneDepthEstimator(), segmenter=segmenter)
    return controller.analyze(*pair)


def test_report_records_the_estimated_camera_parameters(scene_pair):
    report = _analyze(scene_pair, AppSettings())
    assert report.intrinsics_source is IntrinsicsSource.FIELD_OF_VIEW
    assert report.focal_length_px == pytest.approx(518.86, rel=0.005)


def test_calibrated_profile_changes_the_measured_distance(scene_pair):
    estimated = _analyze(scene_pair, AppSettings())
    narrow = CameraIntrinsics(fx=1037.7, fy=1037.7, cx=320, cy=240)  # twice the focal length
    calibrated = _analyze(scene_pair, AppSettings(camera=narrow))

    assert calibrated.intrinsics_source is IntrinsicsSource.CALIBRATED
    assert calibrated.focal_length_px == pytest.approx(1037.7)
    lateral = [r.objects_of(ChangeType.DISPLACED)[0].displacement.x for r in (estimated, calibrated)]
    # A longer focal length means a narrower view: the same pixel shift is half the sideways distance.
    assert lateral[1] == pytest.approx(lateral[0] / 2, rel=0.05)
