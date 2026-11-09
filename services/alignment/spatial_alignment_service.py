"""Registration of the "after" view onto the "before" view by a RANSAC homography."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import cv2
import numpy as np

from config.alignment_settings import AlignmentSettings, DescriptorMatcher, FeatureDetector
from models.alignment import AlignmentReport, AlignmentResult
from models.depth_map import DepthMap
from models.exceptions import AlignmentError
from models.image_data import ImageData

# Linear part of an acceptable homography may scale areas by at most this factor.
_MAX_AREA_SCALE = 10.0

_FLANN_KDTREE = 1
_FLANN_LSH = 6


class SpatialAlignmentService:
    """Normalizes the perspective of the "after" view to the "before" view.

    Keypoints are detected with SIFT or AKAZE on grayscale copies (downscaled
    when larger than ``max_feature_side``), paired with FLANN or brute force,
    filtered with Lowe's ratio test and fed to ``cv2.findHomography`` with
    RANSAC. The resulting homography maps "after" pixels onto the "before"
    pixel grid and is used to resample both the "after" image and its depth
    map, so every pixel of the two views refers to the same scene point.

    A planar homography is exact for a pure camera rotation or a planar
    scene; for small translations of a handheld camera it is a close
    approximation, and the depth values themselves are not re-projected.
    """

    def __init__(self, settings: AlignmentSettings | None = None) -> None:
        self._settings = settings or AlignmentSettings()
        if self._settings.detector is FeatureDetector.AKAZE and _akaze_factory() is None:
            raise ValueError(
                "The AKAZE detector requires the OpenCV extra modules (opencv-contrib-python)."
            )

    @property
    def settings(self) -> AlignmentSettings:
        return self._settings

    def align(self, before: ImageData, after: ImageData, after_depth: DepthMap) -> AlignmentResult:
        """Warp ``after`` and ``after_depth`` onto ``before``.

        Raises :class:`AlignmentError` when the views share too few reliable
        correspondences to estimate a homography.
        """
        homography, report = self.estimate_homography(before, after)
        return self.warp(before, after, after_depth, homography, report)

    def unaligned(
        self, before: ImageData, after: ImageData, after_depth: DepthMap, reason: AlignmentReport
    ) -> AlignmentResult:
        """Fallback for a fixed camera: only rescale ``after`` to the size of ``before``."""
        scale = np.diag([before.width / after.width, before.height / after.height, 1.0])
        report = replace(reason, used_fallback=True)
        return self.warp(before, after, after_depth, scale, report)

    def estimate_homography(
        self, before: ImageData, after: ImageData
    ) -> tuple[np.ndarray, AlignmentReport]:
        """Return the homography mapping ``after`` pixels to ``before`` pixels."""
        # FLANN's randomized trees and RANSAC draw from OpenCV's global
        # generator; seeding it makes the result reproducible.
        cv2.setRNGSeed(self._settings.random_seed)
        detector = self._create_detector()
        kp_before, desc_before = self._detect(detector, before)
        kp_after, desc_after = self._detect(detector, after)

        matches = self._match(desc_after, desc_before)
        stats = dict(
            detector=self._settings.detector.value,
            keypoints_before=len(kp_before),
            keypoints_after=len(kp_after),
            good_matches=len(matches),
        )
        required = self._settings.min_inliers
        if len(matches) < required:
            raise AlignmentError(
                self._failure(
                    stats,
                    0,
                    f"Las dos imágenes comparten muy pocos puntos en común ({len(matches)} de "
                    f"{required} necesarios). Tome la segunda foto desde una posición y "
                    "con un encuadre más parecidos a la primera.",
                )
            )

        src = np.float32([kp_after[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_before[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, self._settings.ransac_reprojection_threshold
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if homography is None or inliers < required:
            raise AlignmentError(
                self._failure(
                    stats,
                    inliers,
                    f"No se pudo alinear la segunda imagen con la primera: solo {inliers} "
                    f"correspondencias son geométricamente consistentes ({required} necesarias).",
                )
            )
        if not self._is_plausible(homography):
            raise AlignmentError(
                self._failure(
                    stats,
                    inliers,
                    "La alineación calculada deforma la imagen de forma irreal. "
                    "Verifique que ambas fotos muestren el mismo espacio.",
                )
            )

        keep = inlier_mask.ravel().astype(bool)
        projected = cv2.perspectiveTransform(src[keep], homography)
        error = float(np.linalg.norm(projected - dst[keep], axis=2).mean())
        report = AlignmentReport(
            succeeded=True,
            inliers=inliers,
            mean_reprojection_error_px=error,
            message=f"Imágenes alineadas con {inliers} correspondencias consistentes.",
            **stats,
        )
        return homography, report

    def warp(
        self,
        before: ImageData,
        after: ImageData,
        after_depth: DepthMap,
        homography: np.ndarray,
        report: AlignmentReport,
    ) -> AlignmentResult:
        """Resample ``after`` and its depth onto the pixel grid of ``before``."""
        if after_depth.shape != (after.height, after.width):
            raise ValueError("The depth map must have the resolution of its image.")
        size = (before.width, before.height)
        image = cv2.warpPerspective(
            after.pixels, homography, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        # Nearest neighbour keeps depth discontinuities sharp instead of
        # inventing intermediate distances along object edges.
        depth = cv2.warpPerspective(
            after_depth.values, homography, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        coverage = cv2.warpPerspective(
            np.ones((after.height, after.width), np.uint8),
            homography,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        margin = self._settings.border_margin_px
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))
            coverage = cv2.erode(coverage, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        return AlignmentResult(
            report=report,
            homography=homography,
            image=ImageData(pixels=image, source=after.source, acquired_at=after.acquired_at),
            depth=replace(after_depth, values=depth.astype(np.float32, copy=False)),
            valid_mask=coverage.astype(bool),
        )

    def _create_detector(self) -> cv2.Feature2D:
        detector = self._settings.detector
        if detector is FeatureDetector.SIFT:
            return cv2.SIFT_create(nfeatures=self._settings.max_features)
        if detector is FeatureDetector.ORB:
            return cv2.ORB_create(nfeatures=self._settings.max_features)
        return _akaze_factory()()

    def _detect(self, detector: cv2.Feature2D, image: ImageData) -> tuple[np.ndarray, np.ndarray | None]:
        """Keypoint coordinates (in full-resolution pixels) and their descriptors."""
        gray = cv2.cvtColor(image.pixels, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, self._settings.max_feature_side / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2) / scale
        return points, descriptors

    def _match(self, query: np.ndarray | None, train: np.ndarray | None) -> list[cv2.DMatch]:
        """Pairs that pass Lowe's ratio test; ``queryIdx`` indexes ``query``."""
        if query is None or train is None or len(query) < 2 or len(train) < 2:
            return []
        candidates = self._create_matcher(binary=query.dtype == np.uint8).knnMatch(query, train, k=2)
        ratio = self._settings.ratio_test
        return [
            pair[0]
            for pair in candidates
            if len(pair) == 2 and pair[0].distance <= ratio * pair[1].distance
        ]

    def _create_matcher(self, binary: bool) -> cv2.DescriptorMatcher:
        if self._settings.matcher is DescriptorMatcher.BRUTE_FORCE:
            return cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2)
        if binary:
            index = dict(algorithm=_FLANN_LSH, table_number=6, key_size=12, multi_probe_level=1)
        else:
            index = dict(algorithm=_FLANN_KDTREE, trees=5)
        return cv2.FlannBasedMatcher(index, dict(checks=50))

    @staticmethod
    def _is_plausible(homography: np.ndarray) -> bool:
        if not np.isfinite(homography).all():
            return False
        area_scale = float(np.linalg.det(homography[:2, :2]))
        return 1.0 / _MAX_AREA_SCALE <= area_scale <= _MAX_AREA_SCALE

    @staticmethod
    def _failure(stats: dict, inliers: int, message: str) -> AlignmentReport:
        return AlignmentReport(
            succeeded=False, inliers=inliers, mean_reprojection_error_px=float("nan"), message=message, **stats
        )


def akaze_available() -> bool:
    """True when the installed OpenCV build provides the AKAZE detector."""
    return _akaze_factory() is not None


def _akaze_factory():
    factory = getattr(cv2, "AKAZE_create", None)
    if factory is None and hasattr(cv2, "xfeatures2d"):
        factory = getattr(cv2.xfeatures2d, "AKAZE_create", None)
    return factory


def corner_displacement(homography: np.ndarray, size: tuple[int, int]) -> float:
    """Largest distance (px) any image corner moves under ``homography``."""
    width, height = size
    corners: Sequence = [[0, 0], [width, 0], [width, height], [0, height]]
    src = np.float32(corners).reshape(-1, 1, 2)
    return float(np.linalg.norm(cv2.perspectiveTransform(src, homography) - src, axis=2).max())
