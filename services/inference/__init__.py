"""Pretrained model inference (depth estimation) on GPU or CPU."""

from services.inference.base_estimator import BaseDepthEstimator
from services.inference.compute_device import ComputeDevice
from services.inference.depth_anything_estimator import DepthAnythingV2Estimator
from services.inference.device_manager import DeviceManager
from services.inference.estimator_factory import create_depth_estimator
from services.inference.latency_benchmark import DepthLatencyBenchmark, LatencyReport
from services.inference.model_loader import (
    DepthModelBundle,
    DepthModelLoader,
    HuggingFaceDepthModelLoader,
)

__all__ = [
    "BaseDepthEstimator",
    "ComputeDevice",
    "DepthAnythingV2Estimator",
    "DepthLatencyBenchmark",
    "DepthModelBundle",
    "DepthModelLoader",
    "DeviceManager",
    "HuggingFaceDepthModelLoader",
    "LatencyReport",
    "create_depth_estimator",
]
