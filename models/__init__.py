"""Domain data structures shared across layers.

Pretrained weights are cached under ``models/weights/`` (not versioned).
"""

from models.acquired_image import AcquiredImage
from models.alignment import AlignmentReport, AlignmentResult
from models.camera_model import IntrinsicsSource
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
    CalibrationError,
    CameraError,
    ComparisonError,
    DepthEstimationError,
    ImageError,
    ImageLoadError,
    InvalidImageError,
    ModelLoadError,
    NonIndoorSceneError,
    SceneDiffError,
    SegmentationError,
    UnsupportedImageFormatError,
)
from models.image_data import ImageData
from models.segmentation_mask import UNIDENTIFIED_LABEL, SegmentationMask, SemanticLabel
from models.session_history import HistoryEntry, SessionHistory
from models.validation_report import IssueCode, ValidationIssue, ValidationReport

__all__ = [
    "AcquiredImage",
    "AlignmentError",
    "AlignmentReport",
    "AlignmentResult",
    "BoundingBox",
    "CalibrationError",
    "CameraError",
    "ComparisonError",
    "ChangeType",
    "ChangedObject",
    "ComparisonReport",
    "DepthComparison",
    "DepthEstimationError",
    "DepthMap",
    "HistoryEntry",
    "ImageData",
    "ImageError",
    "ImageLoadError",
    "IntrinsicsSource",
    "InvalidImageError",
    "IssueCode",
    "ModelLoadError",
    "NonIndoorSceneError",
    "Point3D",
    "SceneDiffError",
    "SegmentationError",
    "SegmentationMask",
    "SemanticLabel",
    "SessionHistory",
    "UNIDENTIFIED_LABEL",
    "UnsupportedImageFormatError",
    "ValidationIssue",
    "ValidationReport",
]
