"""Photometric statistics over RGB images."""

import numpy as np

# ITU-R BT.601 luma coefficients for R, G and B.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def mean_luminance(rgb: np.ndarray) -> float:
    """Mean perceived brightness of an ``H x W x 3`` uint8 RGB image, in [0, 255]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an H x W x 3 RGB array, got shape {rgb.shape}.")
    return float(np.tensordot(rgb.astype(np.float64), _LUMA_WEIGHTS, axes=([2], [0])).mean())
