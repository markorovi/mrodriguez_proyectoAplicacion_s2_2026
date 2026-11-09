"""Semantic segmentation with SegFormer (ADE20K classes)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.segmentation_settings import SegmentationSettings
from models.exceptions import SegmentationError
from models.image_data import ImageData
from models.segmentation_mask import SegmentationMask
from services.inference.compute_device import ComputeDevice
from services.inference.device_manager import DeviceManager
from services.segmentation.model_loader import (
    HuggingFaceSegmentationModelLoader,
    SegmentationModelBundle,
    SegmentationModelLoader,
)
from services.segmentation.region_labeler import structural_class_ids

# Class scores are upsampled to at most this side before taking the arg-max;
# the resulting ids are then resized (nearest) to the full resolution. This
# bounds GPU memory to roughly 150 x 1024 x 1024 floats for any input size.
_MAX_LOGITS_SIDE = 1024


class SegFormerSegmenter:
    """Adapter from a Hugging Face SegFormer model to a per-pixel class mask.

    The model predicts class scores at a quarter of its input resolution;
    they are upsampled bilinearly on the inference device and reduced with
    an arg-max there, so only the compact id map travels back to the host.
    On CUDA, inference runs under float16 autocast when
    ``settings.half_precision`` is enabled.

    Like the depth estimators, the segmenter loads lazily on first use and
    is a context manager that frees the model on exit.
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

    def segment(self, image: ImageData) -> SegmentationMask:
        """Class id of every pixel of ``image`` at its native resolution."""
        pixels = image.pixels
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise SegmentationError("La segmentación requiere una imagen RGB de 8 bits.")
        if not self.is_loaded:
            self.load()
        assert self._bundle is not None

        start = perf_counter()
        try:
            class_ids = self._predict(pixels)
        except (RuntimeError, ValueError) as exc:
            raise SegmentationError(
                "No se pudieron identificar los objetos de la imagen. "
                "Cierre otras aplicaciones que usen la tarjeta gráfica e intente de nuevo."
            ) from exc
        elapsed = perf_counter() - start

        id2label = {int(k): str(v) for k, v in self._bundle.model.config.id2label.items()}
        return SegmentationMask(
            class_ids=class_ids,
            id2label=id2label,
            model_name=self.name,
            device=self.device_name,
            inference_seconds=elapsed,
            ignored_class_ids=structural_class_ids(id2label),
        )

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

    def __enter__(self) -> SegFormerSegmenter:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()
