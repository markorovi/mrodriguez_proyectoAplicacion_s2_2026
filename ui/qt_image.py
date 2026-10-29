"""Conversion of NumPy pixel buffers to Qt images."""

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


def _rescale_to_uint8(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float64)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return ((values - low) * (255.0 / (high - low))).astype(np.uint8)
