"""Metric monocular depth with Depth Anything V2 (indoor checkpoints)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config.depth_settings import DepthSettings
from services.inference.base_estimator import BaseDepthEstimator
from services.inference.compute_device import ComputeDevice
from services.inference.device_manager import DeviceManager
from services.inference.model_loader import (
    DepthModelBundle,
    DepthModelLoader,
    HuggingFaceDepthModelLoader,
)


class DepthAnythingV2Estimator(BaseDepthEstimator):
    """Adapter from a Hugging Face Depth Anything V2 model to the estimator API.

    The metric indoor checkpoints predict absolute distance to the camera in
    meters. The prediction, made at the model's working resolution, is
    resized back to the input resolution with bicubic interpolation and
    clamped to non-negative values.

    On CUDA, inference runs under float16 autocast when
    ``settings.half_precision`` is enabled to reduce memory and latency.
    """

    def __init__(
        self,
        settings: DepthSettings | None = None,
        device_manager: DeviceManager | None = None,
        model_loader: DepthModelLoader | None = None,
    ) -> None:
        self._settings = settings or DepthSettings()
        self._device_manager = device_manager or DeviceManager(self._settings.device)
        self._loader = model_loader or HuggingFaceDepthModelLoader(
            self._settings.cache_dir, self._settings.local_files_only
        )
        self._device: Optional[ComputeDevice] = None
        self._bundle: Optional[DepthModelBundle] = None

    @property
    def name(self) -> str:
        return self._settings.model_id

    @property
    def is_metric(self) -> bool:
        return self._settings.is_metric

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    @property
    def device_name(self) -> str:
        return self._device.name if self._device else ""

    @property
    def compute_device(self) -> Optional[ComputeDevice]:
        """Selected hardware, including any notice for the user; None before loading."""
        return self._device

    def load(self) -> None:
        if self.is_loaded:
            return
        device = self._device_manager.select()
        self._bundle = self._loader.load(self.name, device.torch_device)
        self._device = device

    def unload(self) -> None:
        if not self.is_loaded:
            return
        self._bundle = None
        if self._device is not None:
            self._device_manager.release(self._device)

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        assert self._bundle is not None and self._device is not None
        processor, model = self._bundle
        inputs = processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device.torch_device)

        with torch.inference_mode(), self._autocast():
            predicted = model(pixel_values=pixel_values).predicted_depth

        height, width = rgb.shape[:2]
        resized = F.interpolate(
            predicted.unsqueeze(1).float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return resized[0, 0].clamp_min(0.0).cpu().numpy()

    def _autocast(self) -> AbstractContextManager:
        if self._device is not None and self._device.is_accelerated and self._settings.half_precision:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()
