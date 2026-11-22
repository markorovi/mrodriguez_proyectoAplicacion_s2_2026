"""Loading of pretrained semantic segmentation models and their pre-processors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Protocol

import torch

from models.exceptions import SegmentationError


class SegmentationModelBundle(NamedTuple):
    """A pre-processor and its model, ready for inference."""

    processor: Any
    model: torch.nn.Module


class SegmentationModelLoader(Protocol):
    """Anything able to materialize a :class:`SegmentationModelBundle` on a device."""

    def load(self, model_id: str, device: torch.device) -> SegmentationModelBundle: ...


class HuggingFaceSegmentationModelLoader:
    """Loads SegFormer checkpoints through the Hugging Face ``transformers`` Auto classes."""

    def __init__(self, cache_dir: Path, local_files_only: bool = False) -> None:
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only

    def load(self, model_id: str, device: torch.device) -> SegmentationModelBundle:
        # Imported lazily: ``transformers`` is heavy and only needed here.
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        options = {"cache_dir": str(self._cache_dir), "local_files_only": self._local_files_only}
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, **options)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_id, **options)
        except (OSError, ValueError) as exc:
            raise SegmentationError(
                "No se pudo cargar el modelo que identifica los objetos. "
                "Verifique la conexión a internet o que los archivos del modelo estén descargados."
            ) from exc
        model.to(device).eval()
        return SegmentationModelBundle(processor=processor, model=model)
