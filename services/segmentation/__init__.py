"""Semantic segmentation used to name the objects that changed."""

from services.segmentation.model_loader import (
    HuggingFaceSegmentationModelLoader,
    SegmentationModelBundle,
    SegmentationModelLoader,
)
from services.segmentation.region_labeler import (
    ADE20K_SPANISH_NAMES,
    ADE20K_STRUCTURAL_CLASSES,
    RegionLabeler,
    structural_class_ids,
)
from services.segmentation.segformer_segmenter import SegFormerSegmenter

__all__ = [
    "ADE20K_SPANISH_NAMES",
    "ADE20K_STRUCTURAL_CLASSES",
    "HuggingFaceSegmentationModelLoader",
    "RegionLabeler",
    "SegFormerSegmenter",
    "SegmentationModelBundle",
    "SegmentationModelLoader",
    "structural_class_ids",
]
