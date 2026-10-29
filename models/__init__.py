"""Domain data structures shared across layers.

Pretrained weights are cached under ``models/weights/`` (not versioned).
"""

from models.acquired_image import AcquiredImage
from models.depth_map import DepthMap
from models.exceptions import (
    CameraError,
    DepthEstimationError,
    ImageError,
    ImageLoadError,
    InvalidImageError,
    ModelLoadError,
    SceneDiffError,
    UnsupportedImageFormatError,
)
from models.image_data import ImageData
from models.validation_report import IssueCode, ValidationIssue, ValidationReport

__all__ = [
    "AcquiredImage",
    "CameraError",
    "DepthEstimationError",
    "DepthMap",
    "ImageData",
    "ImageError",
    "ImageLoadError",
    "InvalidImageError",
    "IssueCode",
    "ModelLoadError",
    "SceneDiffError",
    "UnsupportedImageFormatError",
    "ValidationIssue",
    "ValidationReport",
]
