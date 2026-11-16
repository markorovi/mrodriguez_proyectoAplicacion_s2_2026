"""Camera calibration from photos of a printed chessboard (OpenCV).

Each photo shows the whole board; the inner corners are located with
sub-pixel precision and :func:`cv2.calibrateCamera` estimates the focal
lengths, the principal point and the lens distortion. The pipeline uses the
pinhole intrinsics only; the distortion coefficients are reported so a strong
lens distortion can be noticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from config.camera_intrinsics import CameraIntrinsics
from models.exceptions import CalibrationError

DEFAULT_PATTERN = (9, 6)
"""Inner corners per row and per column (a board of 10 x 7 squares)."""

MIN_VIEWS = 5

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of a calibration.

    ``rms_error_px`` is the mean reprojection error of the board corners;
    values under 0.5 px indicate a good calibration, above 1 px a doubtful
    one (blurry photos, a board that is not flat, too few angles).
    """

    intrinsics: CameraIntrinsics
    rms_error_px: float
    distortion: tuple[float, ...]
    used: tuple[str, ...]
    rejected: tuple[str, ...]


def find_chessboard_corners(rgb: np.ndarray, pattern: tuple[int, int] = DEFAULT_PATTERN) -> Optional[np.ndarray]:
    """Sub-pixel inner corners of the board in an RGB image, or ``None`` if the whole board is not visible."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            return None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    # The two detectors (and OpenCV versions) differ in layout; use the classic N x 1 x 2.
    return np.asarray(corners, np.float32).reshape(-1, 1, 2)


def board_points(pattern: tuple[int, int], square_size_mm: float) -> np.ndarray:
    """Coordinates of the inner corners on the board plane (``Z = 0``), in millimeters."""
    columns, rows = pattern
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_mm
    return grid


def calibrate_camera(
    images: Sequence[tuple[str, np.ndarray]],
    pattern: tuple[int, int] = DEFAULT_PATTERN,
    square_size_mm: float = 25.0,
    min_views: int = MIN_VIEWS,
) -> CalibrationResult:
    """Calibrate the camera that took ``images`` (``(name, rgb)`` pairs, all at one resolution).

    Raises :class:`CalibrationError` with a message for the user when the
    photos have different sizes or fewer than ``min_views`` show the board.
    """
    if square_size_mm <= 0:
        raise ValueError("The square size must be positive.")
    if not images:
        raise CalibrationError("No se recibieron fotos del tablero.")
    sizes = {(rgb.shape[1], rgb.shape[0]) for _, rgb in images}
    if len(sizes) > 1:
        listed = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
        raise CalibrationError(
            f"Las fotos tienen tamaños distintos ({listed}). Tómelas todas con la misma cámara y resolución."
        )
    (width, height), = sizes

    object_points, image_points, used, rejected = [], [], [], []
    template = board_points(pattern, square_size_mm)
    for name, rgb in images:
        corners = find_chessboard_corners(rgb, pattern)
        if corners is None:
            rejected.append(name)
            continue
        object_points.append(template)
        image_points.append(corners)
        used.append(name)

    if len(used) < min_views:
        raise CalibrationError(
            f"Solo se encontró el tablero completo en {len(used)} de {len(images)} fotos; se necesitan al "
            f"menos {min_views}. Revise que el tablero tenga {pattern[0]}x{pattern[1]} esquinas internas, "
            "que salga completo en cada foto y que las fotos no estén borrosas."
        )

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, (width, height), None, None)
    intrinsics = CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        reference_width=width,
        reference_height=height,
    )
    return CalibrationResult(
        intrinsics=intrinsics,
        rms_error_px=float(rms),
        distortion=tuple(float(value) for value in np.ravel(distortion)),
        used=tuple(used),
        rejected=tuple(rejected),
    )
