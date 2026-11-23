"""Common contract for semantic segmenters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Mapping, Optional

import numpy as np

from models.exceptions import SegmentationError
from models.image_data import ImageData
from models.segmentation_mask import SegmentationMask
from services.inference.base_estimator import busy_resource
from services.inference.compute_device import ComputeDevice
from services.segmentation.region_labeler import structural_class_ids


class BaseSegmenter(ABC):
    """Strategy interface: one segmentation model behind a uniform ``segment`` call.

    :meth:`segment` is a template method, the counterpart of
    :meth:`services.inference.base_estimator.BaseDepthEstimator.estimate`. It
    guards the input before loading anything, loads the model on first use,
    times the inference, verifies the output resolution and wraps runtime
    failures (for example, running out of GPU memory) in
    :class:`SegmentationError`. Subclasses only implement :meth:`_predict`,
    :meth:`_class_names` and the model life cycle (and may override
    :meth:`_predict_with_confidence` to report how sure the model is of each
    pixel), so another segmentation
    model can replace the current one without touching the modules that use
    the mask (see :func:`services.segmentation.segmenter_factory.create_segmenter`).

    Segmenters are context managers; leaving the ``with`` block frees the
    model so the next one can use the memory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier of the underlying model."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True when the model is resident in memory."""

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Hardware the model runs on, or an empty string before loading."""

    @property
    def compute_device(self) -> Optional[ComputeDevice]:
        """Selected hardware, or None when it is not known (before loading, or a simple double)."""
        return None

    @abstractmethod
    def load(self) -> None:
        """Bring the model into memory on the selected device."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and the memory it holds."""

    @abstractmethod
    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return the ``H x W`` class-id map of an ``H x W x 3`` uint8 RGB array."""

    def _predict_with_confidence(self, rgb: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Class-id map and, when the model can tell, the ``H x W`` confidence (0-1) of each pixel.

        The default reports no confidence; a subclass whose model gives class
        probabilities overrides it so the mask carries them.
        """
        return self._predict(rgb), None

    @abstractmethod
    def _class_names(self) -> Mapping[int, str]:
        """Name of every class id the loaded model can predict."""

    def segment(self, image: ImageData) -> SegmentationMask:
        """Class id of every pixel of ``image`` at its native resolution."""
        pixels = image.pixels
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise SegmentationError("Los objetos solo se pueden identificar en una foto a color común.")
        if not self.is_loaded:
            self.load()

        start = perf_counter()
        try:
            class_ids, confidence = self._predict_with_confidence(pixels)
        except SegmentationError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise SegmentationError(
                "No se pudieron identificar los objetos de la imagen. "
                f"Cierre otras aplicaciones que usen {busy_resource(self.compute_device)} e intente de nuevo."
            ) from exc
        elapsed = perf_counter() - start

        expected = (image.height, image.width)
        if class_ids.shape != expected or (confidence is not None and confidence.shape != expected):
            raise SegmentationError(
                "No se pudieron identificar los objetos de la imagen. "
                "Intente de nuevo; si el problema continúa, reinicie la aplicación."
            )

        id2label = dict(self._class_names())
        return SegmentationMask(
            class_ids=class_ids,
            id2label=id2label,
            model_name=self.name,
            device=self.device_name,
            inference_seconds=elapsed,
            ignored_class_ids=structural_class_ids(id2label),
            confidence=confidence,
        )

    def __enter__(self) -> BaseSegmenter:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()
