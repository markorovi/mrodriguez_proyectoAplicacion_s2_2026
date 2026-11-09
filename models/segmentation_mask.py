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
    """

    name: str
    class_name: str | None = None
    coverage: float = 0.0

    @property
    def is_identified(self) -> bool:
        return self.class_name is not None


@dataclass(frozen=True)
class SegmentationMask:
    """``H x W`` array of class ids and the names of those classes.

    ``ignored_class_ids`` are scene-structure classes (wall, floor, ...) that
    never name a changed object.
    """

    class_ids: np.ndarray
    id2label: Mapping[int, str]
    model_name: str
    device: str
    inference_seconds: float = 0.0
    ignored_class_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.class_ids.ndim != 2:
            raise ValueError(f"Segmentation mask must be 2-D, got shape {self.class_ids.shape}.")

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
