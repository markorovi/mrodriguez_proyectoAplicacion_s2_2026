"""Changed objects found between two views and the full analysis report."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from models.alignment import AlignmentReport
from models.segmentation_mask import SemanticLabel


class ChangeType(str, Enum):
    """How an object changed between the "before" and "after" views."""

    REMOVED = "RETIRADO / DESAPARICIÓN"
    DISPLACED = "DESPLAZAMIENTO"
    ADDED = "APARICIÓN"


@dataclass(frozen=True)
class Point3D:
    """Position in the camera frame of the reference view, in meters.

    X grows to the right, Y downwards and Z away from the camera.
    """

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.as_array()))


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixels of the reference ("before") image."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass(frozen=True)
class ChangedObject:
    """One object that was removed, displaced or added.

    Positions are 3-D centroids of the object's surface. ``before`` fields
    are set for removed and displaced objects, ``after`` fields for displaced
    and added objects. ``depth_change_cm`` is the mean absolute depth
    difference over the changed region(s): the size of the depth step.
    """

    object_id: int
    change_type: ChangeType
    label: SemanticLabel
    depth_change_cm: float
    area_px: int
    bbox_before: Optional[BoundingBox] = None
    bbox_after: Optional[BoundingBox] = None
    position_before: Optional[Point3D] = None
    position_after: Optional[Point3D] = None

    @property
    def displacement(self) -> Optional[Point3D]:
        """``(dX, dY, dZ)`` in meters, only for displaced objects."""
        if self.position_before is None or self.position_after is None:
            return None
        return self.position_after - self.position_before

    @property
    def distance_cm(self) -> Optional[float]:
        """Euclidean 3-D displacement in centimeters, only for displaced objects."""
        displacement = self.displacement
        return None if displacement is None else displacement.norm * 100.0

    @property
    def summary(self) -> str:
        text = f"{self.label.name}: {self.change_type.value}"
        if self.distance_cm is not None:
            text += f" ({self.distance_cm:.1f} cm)"
        return text


@dataclass(frozen=True)
class DepthComparison:
    """Pixel-level result of comparing two aligned depth maps.

    ``difference_m`` is ``D_after - D_before`` (meters, zero outside the
    overlap). ``scale_factor`` is the factor the "after" depth was multiplied
    by to match the global scale of the "before" depth.
    """

    objects: tuple[ChangedObject, ...]
    difference_m: np.ndarray
    farther_mask: np.ndarray
    closer_mask: np.ndarray
    scale_factor: float = 1.0


@dataclass(frozen=True)
class ComparisonReport:
    """Everything the end-to-end analysis produced for one pair of images.

    ``annotated_image`` is the aligned "after" view (RGB) with the changed
    objects drawn on it. ``stage_seconds`` maps each pipeline stage to its
    duration.
    """

    objects: tuple[ChangedObject, ...]
    alignment: AlignmentReport
    comparison: DepthComparison
    annotated_image: np.ndarray
    warnings: tuple[str, ...] = ()
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.objects)

    @property
    def total_seconds(self) -> float:
        return sum(self.stage_seconds.values())

    def objects_of(self, change_type: ChangeType) -> list[ChangedObject]:
        return [obj for obj in self.objects if obj.change_type is change_type]
