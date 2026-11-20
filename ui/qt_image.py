"""Conversion of NumPy pixel buffers to Qt images."""

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def to_qimage(pixels: np.ndarray) -> QImage:
    """Convert an ``H x W x 3`` RGB or ``H x W`` grayscale array to a QImage.

    Arrays that are not 8-bit are linearly rescaled to [0, 255] for display.
    The returned image owns a copy of the data.
    """
    if pixels.dtype != np.uint8:
        pixels = _rescale_to_uint8(pixels)
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]

    if pixels.ndim == 2:
        return QImage(pixels.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return QImage(pixels.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        return QImage(pixels.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
    raise ValueError(f"Unsupported pixel layout {pixels.shape}.")


def to_qpixmap(pixels: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(to_qimage(pixels))


def downscale(pixels: np.ndarray, max_side: int = 0, max_height: int = 0) -> np.ndarray:
    """A smaller copy of ``pixels`` keeping its proportions, never enlarged.

    The result fits ``max_side`` (longest side) and ``max_height`` when they
    are positive. Area interpolation averages the pixels it merges, so the
    reduced image stays smooth instead of aliased. An image that already fits
    is returned as is, without copying.
    """
    height, width = pixels.shape[:2]
    scale = 1.0
    if max_side > 0:
        scale = min(scale, max_side / max(height, width))
    if max_height > 0:
        scale = min(scale, max_height / height)
    if scale >= 1.0:
        return pixels
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)


def thumbnail_pixmap(pixels: np.ndarray, max_height: int) -> QPixmap:
    """Pixmap of ``pixels`` at most ``max_height`` tall; reduced before converting, to stay small in memory."""
    return to_qpixmap(downscale(pixels, max_height=max_height))


def _rescale_to_uint8(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float64)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return ((values - low) * (255.0 / (high - low))).astype(np.uint8)
