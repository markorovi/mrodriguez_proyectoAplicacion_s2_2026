"""Outcome of registering the "after" view onto the "before" view."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.depth_map import DepthMap
from models.image_data import ImageData


@dataclass(frozen=True)
class AlignmentReport:
    """Feature matching statistics and the verdict on the estimated homography.

    ``used_fallback`` is true when no homography could be estimated and the
    views were compared unaligned (identity or plain rescaling).
    """

    succeeded: bool
    detector: str
    keypoints_before: int
    keypoints_after: int
    good_matches: int
    inliers: int
    mean_reprojection_error_px: float
    message: str
    used_fallback: bool = False

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.good_matches if self.good_matches else 0.0


@dataclass(frozen=True)
class AlignmentResult:
    """The "after" image and depth resampled onto the "before" pixel grid.

    ``homography`` maps "after" pixel coordinates to "before" pixel
    coordinates. ``valid_mask`` is true where the warped view actually covers
    the reference view; only those pixels can be compared.
    """

    report: AlignmentReport
    homography: np.ndarray
    image: ImageData
    depth: DepthMap
    valid_mask: np.ndarray

    @property
    def overlap_fraction(self) -> float:
        return float(np.count_nonzero(self.valid_mask)) / self.valid_mask.size
