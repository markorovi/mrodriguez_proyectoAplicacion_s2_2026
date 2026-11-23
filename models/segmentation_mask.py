"""Per-pixel semantic classes produced by a segmentation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

UNIDENTIFIED_LABEL = "objeto no identificado"


@dataclass(frozen=True)
class SemanticLabel:
    """Name given to an image region and how much of the region supports it.

    ``name`` is the user-facing (Spanish) name; ``class_name`` is the raw
    model class, or ``None`` when the region could not be identified.
    ``confidence`` is the model's mean certainty (0-1) over the pixels of the
    dominant class, or ``None`` when the segmenter does not report it.
    """

    name: str
    class_name: str | None = None
    coverage: float = 0.0
    confidence: float | None = None

    @property
    def is_identified(self) -> bool:
        return self.class_name is not None

    @property
    def display_name(self) -> str:
        """``name`` with an initial capital, e.g. "Caja" or "Objeto no identificado"."""
        return self.name[:1].upper() + self.name[1:]


@dataclass(frozen=True)
class SegmentationMask:
    """``H x W`` array of class ids and the names of those classes.

    ``ignored_class_ids`` are scene-structure classes (wall, floor, ...) that
    never name a changed object. ``confidence``, when the segmenter reports
    it, is an ``H x W`` float array with the probability (0-1) the model gave
    to the class of each pixel.
    """

    class_ids: np.ndarray
    id2label: Mapping[int, str]
    model_name: str
    device: str
    inference_seconds: float = 0.0
    ignored_class_ids: frozenset[int] = field(default_factory=frozenset)
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.class_ids.ndim != 2:
            raise ValueError(f"Segmentation mask must be 2-D, got shape {self.class_ids.shape}.")
        if self.confidence is not None and self.confidence.shape != self.class_ids.shape:
            raise ValueError(
                f"Confidence shape {self.confidence.shape} differs from the class map {self.class_ids.shape}."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.class_ids.shape[0]), int(self.class_ids.shape[1])

    def dominant_class(self, region: np.ndarray) -> tuple[int | None, float]:
        """Most frequent non-structural class inside ``region`` and its coverage.

        Coverage is the fraction of the whole region the class occupies.
        Returns ``(None, 0.0)`` for an empty region or one made only of
        structural classes.
        """
        if region.shape != self.class_ids.shape:
            raise ValueError("Region and segmentation mask must have the same shape.")
        ids = self.class_ids[region.astype(bool)]
        if ids.size == 0:
            return None, 0.0
        counts = np.bincount(ids.astype(np.int64).ravel())
        for ignored in self.ignored_class_ids:
            if ignored < counts.size:
                counts[ignored] = 0
        if not counts.any():
            return None, 0.0
        best = int(np.argmax(counts))
        return best, float(counts[best]) / ids.size

    def class_confidence(self, region: np.ndarray, class_id: int) -> float | None:
        """Mean confidence over the pixels of ``region`` labelled ``class_id``.

        ``None`` when the mask carries no confidence or no such pixel exists.
        """
        if self.confidence is None:
            return None
        selected = region.astype(bool) & (self.class_ids == class_id)
        if not selected.any():
            return None
        return float(self.confidence[selected].mean())
