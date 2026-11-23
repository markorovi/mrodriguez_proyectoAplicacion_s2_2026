"""Semantic segmentation used to name the objects that changed."""

from services.segmentation.base_segmenter import BaseSegmenter
from services.segmentation.model_loader import (
    HuggingFaceSegmentationModelLoader,
    SegmentationModelBundle,
    SegmentationModelLoader,
)
from services.segmentation.region_labeler import (
    ADE20K_OUTDOOR_CLASSES,
    ADE20K_SPANISH_NAMES,
    ADE20K_STRUCTURAL_CLASSES,
    MAX_OUTDOOR_PERCENT,
    RegionLabeler,
    outdoor_class_ids,
    structural_class_ids,
)
from services.segmentation.segformer_segmenter import SegFormerSegmenter
from services.segmentation.segmenter_factory import create_segmenter

__all__ = [
    "ADE20K_OUTDOOR_CLASSES",
    "ADE20K_SPANISH_NAMES",
    "ADE20K_STRUCTURAL_CLASSES",
    "BaseSegmenter",
    "HuggingFaceSegmentationModelLoader",
    "MAX_OUTDOOR_PERCENT",
    "RegionLabeler",
    "SegFormerSegmenter",
    "SegmentationModelBundle",
    "SegmentationModelLoader",
    "create_segmenter",
    "outdoor_class_ids",
    "structural_class_ids",
]
