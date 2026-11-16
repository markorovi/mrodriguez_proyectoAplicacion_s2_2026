"""In-memory representation of an acquired image."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ImageData:
    """Pixel buffer plus provenance metadata.

    ``pixels`` follows the ``H x W x C`` layout with RGB channel order when it
    has three channels. Acquisition sources do not guarantee conformity;
    :class:`utils.image_validator.ImageValidator` is responsible for that.

    ``focal_length_35mm`` is the 35 mm-equivalent focal length (mm) that the
    camera wrote in the file's EXIF data, when present; it lets the analysis
    estimate the camera parameters of photos from any phone or camera.
    """

    pixels: np.ndarray
    source: str
    acquired_at: datetime = field(default_factory=datetime.now)
    focal_length_35mm: Optional[float] = None

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1]) if self.pixels.ndim >= 2 else 0

    @property
    def channels(self) -> int:
        return int(self.pixels.shape[2]) if self.pixels.ndim == 3 else 1

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.width, self.height
