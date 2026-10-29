"""Common contract for monocular depth estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from models.depth_map import DepthMap
from models.exceptions import DepthEstimationError
from models.image_data import ImageData


class BaseDepthEstimator(ABC):
    """Strategy interface: one depth model behind a uniform ``estimate`` call.

    :meth:`estimate` is a template method. It guards the input, loads the
    model on first use, times the inference, verifies the output resolution
    and wraps runtime failures (for example, running out of GPU memory) in
    :class:`DepthEstimationError`. Subclasses only implement :meth:`_predict`
    and the model life cycle.

    Estimators are context managers; leaving the ``with`` block frees the
    model so the next one can use the memory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier of the underlying model."""

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """True when the output is in meters rather than relative units."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True when the model is resident in memory."""

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Hardware the model runs on, or an empty string before loading."""

    @abstractmethod
    def load(self) -> None:
        """Bring the model into memory on the selected device."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and the memory it holds."""

    @abstractmethod
    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return an ``H x W`` depth array for an ``H x W x 3`` uint8 RGB array."""

    def estimate(self, image: ImageData) -> DepthMap:
        """Compute the depth map of ``image`` at its native resolution."""
        self._check_input(image)
        if not self.is_loaded:
            self.load()

        start = perf_counter()
        try:
            values = self._predict(image.pixels)
        except DepthEstimationError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise DepthEstimationError(
                "No se pudo calcular la profundidad de la imagen. "
                "Cierre otras aplicaciones que usen la tarjeta gráfica e intente de nuevo."
            ) from exc
        elapsed = perf_counter() - start

        if values.shape != (image.height, image.width):
            raise DepthEstimationError(
                f"El modelo devolvió un mapa de {values.shape} para una imagen "
                f"de {(image.height, image.width)}."
            )
        return DepthMap(
            values=values.astype(np.float32, copy=False),
            is_metric=self.is_metric,
            model_name=self.name,
            device=self.device_name,
            inference_seconds=elapsed,
        )

    def __enter__(self) -> BaseDepthEstimator:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    @staticmethod
    def _check_input(image: ImageData) -> None:
        pixels = image.pixels
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise DepthEstimationError(
                "La estimación de profundidad requiere una imagen RGB de 8 bits."
            )
