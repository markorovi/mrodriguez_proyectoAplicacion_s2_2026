"""Parameters of the spatial alignment between the "before" and "after" views."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FeatureDetector(str, Enum):
    """Keypoint detector and descriptor used to find correspondences.

    SIFT and ORB ship with ``opencv-python``. Since OpenCV 5, AKAZE lives in
    the extra modules and requires ``opencv-contrib-python``.
    """

    SIFT = "sift"
    AKAZE = "akaze"
    ORB = "orb"


class DescriptorMatcher(str, Enum):
    """Strategy used to pair descriptors between the two images."""

    FLANN = "flann"
    BRUTE_FORCE = "brute_force"


class AlignmentSettings(BaseModel):
    """Feature matching, RANSAC homography and warping options."""

    model_config = ConfigDict(frozen=True)

    detector: FeatureDetector = FeatureDetector.SIFT
    matcher: DescriptorMatcher = DescriptorMatcher.FLANN
    max_features: int = Field(default=5000, gt=0, description="Keypoints kept per image (SIFT and ORB).")
    ratio_test: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description="Lowe's ratio: best match must be at most this fraction of the second best.",
    )
    ransac_reprojection_threshold: float = Field(
        default=3.0, gt=0.0, description="Maximum RANSAC reprojection error of an inlier (px)."
    )
    min_inliers: int = Field(default=10, ge=4, description="Inliers required to accept a homography.")
    max_feature_side: int = Field(
        default=1600,
        gt=0,
        description="Images are downscaled so their longest side fits this size before detection.",
    )
    border_margin_px: int = Field(
        default=4,
        ge=0,
        description="Erosion applied to the overlap mask to discard interpolated borders (px).",
    )
    random_seed: int = Field(
        default=0,
        description="Seed of the randomized FLANN trees and RANSAC, so the same pair always aligns the same way.",
    )
    fallback_to_identity: bool = Field(
        default=True,
        description="When alignment fails, compare the views unaligned (fixed camera) with a warning.",
    )
