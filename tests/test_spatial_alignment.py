"""Homography estimation with RANSAC and warping of images, depth and overlap mask."""

import cv2
import numpy as np
import pytest

from config import AlignmentSettings, DescriptorMatcher, FeatureDetector
from models import AlignmentError, AlignmentReport, DepthMap, ImageData, SceneDiffError
from services.alignment import SpatialAlignmentService, akaze_available, corner_displacement

# Maps "after" pixels onto "before" pixels: slight rotation, shift and perspective.
TRUE_HOMOGRAPHY = np.array([[1.02, 0.03, -12.0], [-0.02, 0.99, 8.0], [2e-5, 1e-5, 1.0]])


def _ramp_depth(height: int, width: int) -> np.ndarray:
    rows = np.linspace(4.0, 1.5, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 0.5, width, dtype=np.float32)[None, :]
    return rows + cols


def _depth(values: np.ndarray) -> DepthMap:
    return DepthMap(values=values, is_metric=True, model_name="test", device="CPU")


@pytest.fixture
def views(scene_rgb):
    """``before`` and an ``after`` view taken through the inverse of TRUE_HOMOGRAPHY."""
    height, width = scene_rgb.shape[:2]
    inverse = np.linalg.inv(TRUE_HOMOGRAPHY)
    after_rgb = cv2.warpPerspective(scene_rgb, inverse, (width, height))
    after_depth = cv2.warpPerspective(
        _ramp_depth(height, width), inverse, (width, height), flags=cv2.INTER_NEAREST
    )
    return ImageData(scene_rgb, "before"), ImageData(after_rgb, "after"), _depth(after_depth)


def _transfer_error(estimated: np.ndarray, expected: np.ndarray) -> float:
    """Mean distance (px) between where both homographies send a grid of interior points.

    The scene's keypoints lie inside the room, so the comparison stays there
    instead of extrapolating to the image corners.
    """
    xs, ys = np.meshgrid(np.linspace(100, 540, 12), np.linspace(80, 400, 9))
    grid = np.float32(np.stack([xs.ravel(), ys.ravel()], axis=1)).reshape(-1, 1, 2)
    return float(
        np.linalg.norm(
            cv2.perspectiveTransform(grid, estimated) - cv2.perspectiveTransform(grid, expected), axis=2
        ).mean()
    )


def test_recovers_known_homography(views):
    before, after, depth = views
    result = SpatialAlignmentService().align(before, after, depth)

    assert result.report.succeeded
    assert _transfer_error(result.homography, TRUE_HOMOGRAPHY) < 1.0


def test_report_counts_matches_and_inliers(views):
    report = SpatialAlignmentService().align(*views).report

    assert report.detector == "sift"
    assert report.keypoints_before > 0 and report.keypoints_after > 0
    assert report.inliers >= 10
    assert report.inliers <= report.good_matches
    assert 0.0 < report.inlier_ratio <= 1.0
    assert report.mean_reprojection_error_px < 3.0
    assert not report.used_fallback


def test_aligned_image_matches_reference_inside_overlap(views):
    before, after, depth = views
    result = SpatialAlignmentService().align(before, after, depth)

    assert result.image.pixels.shape == before.pixels.shape
    error = np.abs(result.image.pixels.astype(np.int16) - before.pixels)[result.valid_mask]
    assert error.mean() < 6.0


def test_depth_is_warped_with_the_same_homography(views):
    before, after, depth = views
    result = SpatialAlignmentService().align(before, after, depth)

    expected = _ramp_depth(before.height, before.width)
    error = np.abs(result.depth.values - expected)[result.valid_mask]
    assert result.depth.shape == (before.height, before.width)
    assert np.median(error) < 0.01
    assert result.depth.is_metric and result.depth.model_name == "test"


def test_overlap_mask_matches_the_area_the_after_view_covers(views):
    before, after, depth = views
    result = SpatialAlignmentService(AlignmentSettings(border_margin_px=0)).align(before, after, depth)

    expected = cv2.warpPerspective(
        np.ones((after.height, after.width), np.uint8), TRUE_HOMOGRAPHY, (before.width, before.height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    intersection = np.count_nonzero(result.valid_mask & expected)
    union = np.count_nonzero(result.valid_mask | expected)
    assert intersection / union > 0.97
    assert 0.8 < result.overlap_fraction < 1.0


def test_overlap_mask_is_eroded_by_the_border_margin(views):
    loose = SpatialAlignmentService(AlignmentSettings(border_margin_px=0)).align(*views)
    tight = SpatialAlignmentService(AlignmentSettings(border_margin_px=10)).align(*views)
    assert tight.overlap_fraction < loose.overlap_fraction
    assert not (tight.valid_mask & ~loose.valid_mask).any()


def test_identical_views_align_to_identity(scene_image):
    depth = _depth(_ramp_depth(scene_image.height, scene_image.width))
    result = SpatialAlignmentService().align(scene_image, scene_image, depth)

    assert _transfer_error(result.homography, np.eye(3)) < 0.5
    np.testing.assert_array_equal(result.depth.values[result.valid_mask], depth.values[result.valid_mask])


@pytest.mark.parametrize("detector", [FeatureDetector.SIFT, FeatureDetector.ORB])
@pytest.mark.parametrize("matcher", list(DescriptorMatcher))
def test_every_detector_and_matcher_recovers_the_homography(views, detector, matcher):
    service = SpatialAlignmentService(AlignmentSettings(detector=detector, matcher=matcher))
    result = service.align(*views)

    assert result.report.detector == detector.value
    assert _transfer_error(result.homography, TRUE_HOMOGRAPHY) < 3.0


@pytest.mark.skipif(not akaze_available(), reason="AKAZE requires opencv-contrib-python")
def test_akaze_recovers_the_homography(views):
    result = SpatialAlignmentService(AlignmentSettings(detector=FeatureDetector.AKAZE)).align(*views)
    assert _transfer_error(result.homography, TRUE_HOMOGRAPHY) < 3.0


@pytest.mark.skipif(akaze_available(), reason="AKAZE is available in this OpenCV build")
def test_akaze_without_extra_modules_is_rejected_up_front():
    with pytest.raises(ValueError, match="opencv-contrib-python"):
        SpatialAlignmentService(AlignmentSettings(detector=FeatureDetector.AKAZE))


def test_stricter_ratio_test_keeps_fewer_matches(views):
    strict = SpatialAlignmentService(AlignmentSettings(ratio_test=0.5)).estimate_homography(*views[:2])[1]
    loose = SpatialAlignmentService(AlignmentSettings(ratio_test=0.75)).estimate_homography(*views[:2])[1]
    assert strict.good_matches < loose.good_matches


def test_detection_on_downscaled_copies_maps_back_to_full_resolution(views):
    service = SpatialAlignmentService(AlignmentSettings(max_feature_side=480))
    homography, _ = service.estimate_homography(*views[:2])
    assert _transfer_error(homography, TRUE_HOMOGRAPHY) < 3.0


def test_featureless_view_raises_controlled_alignment_error(scene_image):
    blank = ImageData(np.full_like(scene_image.pixels, 128), "blank")
    depth = _depth(np.ones((blank.height, blank.width), np.float32))

    with pytest.raises(AlignmentError) as raised:
        SpatialAlignmentService().align(scene_image, blank, depth)

    report = raised.value.report
    assert isinstance(raised.value, SceneDiffError)
    assert not report.succeeded
    assert report.inliers == 0
    assert report.keypoints_after == 0
    assert str(raised.value) == report.message


def test_too_few_inliers_raises_alignment_error(views):
    service = SpatialAlignmentService(AlignmentSettings(min_inliers=100_000))
    with pytest.raises(AlignmentError, match="necesarios"):
        service.align(*views)


def test_unaligned_fallback_rescales_to_reference_size(scene_image):
    small = ImageData(cv2.resize(scene_image.pixels, (320, 240), interpolation=cv2.INTER_AREA), "small")
    small_depth = _depth(np.full((240, 320), 2.0, np.float32))
    reason = AlignmentReport(
        succeeded=False, detector="sift", keypoints_before=0, keypoints_after=0, good_matches=0,
        inliers=0, mean_reprojection_error_px=float("nan"), message="Sin correspondencias.",
    )

    result = SpatialAlignmentService().unaligned(scene_image, small, small_depth, reason)

    assert result.report.used_fallback and not result.report.succeeded
    assert result.image.pixels.shape == scene_image.pixels.shape
    assert result.depth.shape == (480, 640)
    assert result.valid_mask[10:-10, 10:-10].all()
    np.testing.assert_allclose(result.homography, np.diag([2.0, 2.0, 1.0]))


def test_warp_rejects_depth_of_another_resolution(views):
    before, after, _ = views
    wrong = _depth(np.ones((10, 10), np.float32))
    with pytest.raises(ValueError):
        SpatialAlignmentService().warp(before, after, wrong, np.eye(3), None)


def test_corner_displacement_measures_how_far_the_view_moves():
    shift = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -12.0], [0.0, 0.0, 1.0]])
    assert corner_displacement(shift, (640, 480)) == pytest.approx(13.0)


def test_alignment_is_reproducible(views):
    service = SpatialAlignmentService()
    first, _ = service.estimate_homography(*views[:2])
    for _ in range(3):
        again, _ = service.estimate_homography(*views[:2])
        np.testing.assert_array_equal(first, again)
