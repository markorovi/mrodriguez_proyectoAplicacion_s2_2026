"""Semantic segmentation with SegFormer (ADE20K classes)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.segmentation_settings import SegmentationSettings
from services.inference.compute_device import ComputeDevice
from services.inference.device_manager import DeviceManager
from services.segmentation.base_segmenter import BaseSegmenter
from services.segmentation.model_loader import (
    HuggingFaceSegmentationModelLoader,
    SegmentationModelBundle,
    SegmentationModelLoader,
)

# Class scores are upsampled to at most this side before taking the arg-max;
# the resulting ids are then resized (nearest) to the full resolution. This
# bounds GPU memory to roughly 150 x 1024 x 1024 floats for any input size.
_MAX_LOGITS_SIDE = 1024


class SegFormerSegmenter(BaseSegmenter):
    """Adapter from a Hugging Face SegFormer model to a per-pixel class mask.

    The model predicts class scores at a quarter of its input resolution;
    they are upsampled bilinearly on the inference device and reduced with
    an arg-max there, so only the compact id map travels back to the host.
    On CUDA, inference runs under float16 autocast when
    ``settings.half_precision`` is enabled.

    Input checks, lazy loading, timing and error wrapping come from the
    :meth:`BaseSegmenter.segment` template method; like the depth estimators,
    the segmenter is a context manager that frees the model on exit.
    """

    def __init__(
        self,
        settings: SegmentationSettings | None = None,
        device_manager: DeviceManager | None = None,
        model_loader: SegmentationModelLoader | None = None,
    ) -> None:
        self._settings = settings or SegmentationSettings()
        self._device_manager = device_manager or DeviceManager(self._settings.device)
        self._loader = model_loader or HuggingFaceSegmentationModelLoader(
            self._settings.cache_dir, self._settings.local_files_only
        )
        self._device: Optional[ComputeDevice] = None
        self._bundle: Optional[SegmentationModelBundle] = None

    @property
    def name(self) -> str:
        return self._settings.model_id

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    @property
    def device_name(self) -> str:
        return self._device.name if self._device else ""

    @property
    def compute_device(self) -> Optional[ComputeDevice]:
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
            logits = model(pixel_values=pixel_values).logits

        height, width = rgb.shape[:2]
        scale = min(1.0, _MAX_LOGITS_SIDE / max(height, width))
        work_size = (max(1, round(height * scale)), max(1, round(width * scale)))
        with torch.inference_mode():
            upsampled = F.interpolate(logits.float(), size=work_size, mode="bilinear", align_corners=False)
            class_ids = upsampled.argmax(dim=1)[0].to(torch.int32).cpu().numpy()
        if work_size != (height, width):
            class_ids = cv2.resize(class_ids, (width, height), interpolation=cv2.INTER_NEAREST)
        return class_ids

    def _autocast(self) -> AbstractContextManager:
        if self._device is not None and self._device.is_accelerated and self._settings.half_precision:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _class_names(self) -> dict[int, str]:
        assert self._bundle is not None
        return {int(k): str(v) for k, v in self._bundle.model.config.id2label.items()}
