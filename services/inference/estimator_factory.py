"""Builds the configured depth estimator.

This is the single place that names a concrete estimator class, so a newer
model can be introduced without touching the modules that consume depth.
"""

from config.depth_settings import DepthSettings
from services.inference.base_estimator import BaseDepthEstimator
from services.inference.depth_anything_estimator import DepthAnythingV2Estimator


def create_depth_estimator(settings: DepthSettings | None = None) -> BaseDepthEstimator:
    return DepthAnythingV2Estimator(settings or DepthSettings())
