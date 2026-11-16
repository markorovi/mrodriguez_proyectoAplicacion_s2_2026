"""Camera parameters (intrinsics) for a photo from any camera.

The pinhole model turns depth into 3-D positions with the focal lengths
``fx, fy`` and the principal point ``cx, cy``. They depend on the camera, its
resolution and its zoom, so a single fixed set only fits one camera. This
module picks, for each photo, the most precise source available:

1. a calibrated profile of the camera, when it matches the photo's aspect ratio;
2. the 35 mm-equivalent focal length stored by the camera in the EXIF data;
3. an estimate from a typical horizontal field of view.

Without a calibration, pixels are assumed square (``fx = fy``) and the
principal point at the image center, which holds closely for phones and
webcams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from config.camera_estimation_settings import CameraEstimationSettings
from config.camera_intrinsics import CameraIntrinsics
from models.camera_model import IntrinsicsSource
from models.image_data import ImageData

# Diagonal of a 36 x 24 mm frame; the 35 mm equivalence is defined on it.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

# Relative difference of aspect ratio under which a calibration still applies.
ASPECT_RATIO_TOLERANCE = 0.01


@dataclass(frozen=True)
class IntrinsicsEstimate:
    """Intrinsics for one image size and where they came from."""

    intrinsics: CameraIntrinsics
    source: IntrinsicsSource


def intrinsics_from_focal_35mm(width: int, height: int, focal_length_35mm: float) -> CameraIntrinsics:
    """Intrinsics of a ``width x height`` image whose 35 mm-equivalent focal length is known."""
    _check_size(width, height)
    if focal_length_35mm <= 0:
        raise ValueError("The 35 mm-equivalent focal length must be positive.")
    focal_px = focal_length_35mm * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
    return _centered(width, height, focal_px)


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> CameraIntrinsics:
    """Intrinsics of a ``width x height`` image with the given horizontal field of view."""
    _check_size(width, height)
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("The horizontal field of view must be in (0, 180) degrees.")
    focal_px = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    return _centered(width, height, focal_px)


def horizontal_fov_deg(intrinsics: CameraIntrinsics) -> float:
    """Horizontal field of view (degrees) of a set of intrinsics."""
    return math.degrees(2.0 * math.atan(intrinsics.reference_width / (2.0 * intrinsics.fx)))


def matches_aspect_ratio(intrinsics: CameraIntrinsics, width: int, height: int) -> bool:
    """True when an image of ``width x height`` has the aspect ratio the intrinsics were calibrated at."""
    calibrated = intrinsics.reference_width / intrinsics.reference_height
    return abs(width / height - calibrated) <= ASPECT_RATIO_TOLERANCE * calibrated


def resolve_intrinsics(
    image: ImageData,
    calibration: Optional[CameraIntrinsics] = None,
    settings: Optional[CameraEstimationSettings] = None,
) -> IntrinsicsEstimate:
    """Most precise intrinsics available for ``image``, scaled to its resolution.

    A ``calibration`` with a different aspect ratio than the image belongs to
    another camera or capture mode (or a cropped photo) and is not used.
    """
    settings = settings or CameraEstimationSettings()
    width, height = image.width, image.height
    if calibration is not None and matches_aspect_ratio(calibration, width, height):
        return IntrinsicsEstimate(calibration.scaled_to(width, height), IntrinsicsSource.CALIBRATED)
    if settings.use_exif and image.focal_length_35mm:
        return IntrinsicsEstimate(
            intrinsics_from_focal_35mm(width, height, image.focal_length_35mm), IntrinsicsSource.EXIF
        )
    return IntrinsicsEstimate(
        intrinsics_from_fov(width, height, settings.horizontal_fov_deg), IntrinsicsSource.FIELD_OF_VIEW
    )


def _centered(width: int, height: int, focal_px: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=focal_px,
        fy=focal_px,
        cx=width / 2.0,
        cy=height / 2.0,
        reference_width=width,
        reference_height=height,
    )


def _check_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Image size must be positive.")
