"""Rendering of depth maps as color images."""

import cv2
import numpy as np

from models.depth_map import DepthMap


def colorize_depth(depth: DepthMap, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Return an ``H x W x 3`` uint8 RGB image of ``depth``.

    Values are min-max normalized and inverted so that near surfaces are
    bright and far surfaces are dark.
    """
    values = depth.values.astype(np.float32)
    span = float(values.max() - values.min())
    normalized = np.zeros_like(values) if span <= 0 else (values - values.min()) / span
    gray = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, colormap), cv2.COLOR_BGR2RGB)
