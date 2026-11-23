"""Loading of pretrained semantic segmentation models and their pre-processors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Protocol

import torch

from models.exceptions import SegmentationError
from services.local_first import load_local_first

MISSING_LOCAL_FILES_MESSAGE = (
    "No se pudo cargar el modelo que identifica los objetos. "
    "Sus archivos deben estar guardados en este equipo y no se encontraron."
)
DOWNLOAD_FAILED_MESSAGE = (
    "No se pudo cargar el modelo que identifica los objetos. "
    "Verifique la conexión a internet o que los archivos del modelo estén descargados."
)


class SegmentationModelBundle(NamedTuple):
    """A pre-processor and its model, ready for inference."""

    processor: Any
    model: torch.nn.Module


class SegmentationModelLoader(Protocol):
    """Anything able to materialize a :class:`SegmentationModelBundle` on a device."""

    def load(self, model_id: str, device: torch.device) -> SegmentationModelBundle: ...


class HuggingFaceSegmentationModelLoader:
    """Loads SegFormer checkpoints through the Hugging Face ``transformers`` Auto classes.

    The files in ``cache_dir`` are always tried first, without contacting the
    network (see :func:`services.local_first.load_local_first`); they are only
    downloaded when missing and ``local_files_only`` is false.
    """

    def __init__(self, cache_dir: Path, local_files_only: bool = False) -> None:
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only

    def load(self, model_id: str, device: torch.device) -> SegmentationModelBundle:
        # Imported lazily: ``transformers`` is heavy and only needed here.
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        def attempt(local_only: bool) -> SegmentationModelBundle:
            options = {"cache_dir": str(self._cache_dir), "local_files_only": local_only}
            processor = AutoImageProcessor.from_pretrained(model_id, **options)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_id, **options)
            return SegmentationModelBundle(processor=processor, model=model)

        try:
            bundle = load_local_first(attempt, self._local_files_only)
        except (OSError, ValueError) as exc:
            message = MISSING_LOCAL_FILES_MESSAGE if self._local_files_only else DOWNLOAD_FAILED_MESSAGE
            raise SegmentationError(message) from exc
        bundle.model.to(device).eval()
        return bundle
