"""Loading of pretrained depth models and their pre-processors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Protocol

import torch

from models.exceptions import ModelLoadError


class DepthModelBundle(NamedTuple):
    """A pre-processor and its model, ready for inference."""

    processor: Any
    model: torch.nn.Module


class DepthModelLoader(Protocol):
    """Anything able to materialize a :class:`DepthModelBundle` on a device."""

    def load(self, model_id: str, device: torch.device) -> DepthModelBundle: ...


class HuggingFaceDepthModelLoader:
    """Loads checkpoints through the Hugging Face ``transformers`` Auto classes.

    Weights are downloaded once into ``cache_dir`` and reused afterwards;
    with ``local_files_only`` the network is never contacted.
    """

    def __init__(self, cache_dir: Path, local_files_only: bool = False) -> None:
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only

    def load(self, model_id: str, device: torch.device) -> DepthModelBundle:
        # Imported lazily: ``transformers`` is heavy and only needed here.
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        options = {"cache_dir": str(self._cache_dir), "local_files_only": self._local_files_only}
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, **options)
            model = AutoModelForDepthEstimation.from_pretrained(model_id, **options)
        except (OSError, ValueError) as exc:
            raise ModelLoadError(
                f"No se pudo cargar el modelo de profundidad '{model_id}'. "
                "Verifique la conexión a internet o que los pesos estén descargados."
            ) from exc
        model.to(device).eval()
        return DepthModelBundle(processor=processor, model=model)
