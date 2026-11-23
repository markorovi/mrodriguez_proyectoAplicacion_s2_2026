"""Builds the configured semantic segmenter.

This is the single place that names a concrete segmenter class, so a newer
model can be introduced without touching the modules that consume the
segmentation (the counterpart of :mod:`services.inference.estimator_factory`).
"""

from config.segmentation_settings import SegmentationSettings
from services.segmentation.base_segmenter import BaseSegmenter
from services.segmentation.segformer_segmenter import SegFormerSegmenter


def create_segmenter(settings: SegmentationSettings | None = None) -> BaseSegmenter:
    return SegFormerSegmenter(settings or SegmentationSettings())
