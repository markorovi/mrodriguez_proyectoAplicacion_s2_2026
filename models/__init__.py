"""Domain data structures shared across layers.

Pretrained weights are cached under ``models/weights/`` (not versioned).
"""

from models.acquired_image import AcquiredImage
from models.alignment import AlignmentReport, AlignmentResult
from models.comparison_report import (
    BoundingBox,
    ChangedObject,
    ChangeType,
    ComparisonReport,
    DepthComparison,
    Point3D,
)
from models.depth_map import DepthMap
from models.exceptions import (
    AlignmentError,
    CameraError,
    DepthEstimationError,
    ImageError,
    ImageLoadError,
    InvalidImageError,
    ModelLoadError,
    SceneDiffError,
    SegmentationError,
    UnsupportedImageFormatError,
)
from models.image_data import ImageData
from models.segmentation_mask import UNIDENTIFIED_LABEL, SegmentationMask, SemanticLabel
from models.validation_report import IssueCode, ValidationIssue, ValidationReport

__all__ = [
    "AcquiredImage",
    "AlignmentError",
    "AlignmentReport",
    "AlignmentResult",
    "BoundingBox",
    "CameraError",
    "ChangeType",
    "ChangedObject",
    "ComparisonReport",
    "DepthComparison",
    "DepthEstimationError",
    "DepthMap",
    "ImageData",
    "ImageError",
    "ImageLoadError",
    "InvalidImageError",
    "IssueCode",
    "ModelLoadError",
    "Point3D",
    "SceneDiffError",
    "SegmentationError",
    "SegmentationMask",
    "SemanticLabel",
    "UNIDENTIFIED_LABEL",
    "UnsupportedImageFormatError",
    "ValidationIssue",
    "ValidationReport",
]
