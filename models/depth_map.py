"""Dense per-pixel depth produced by a depth estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthMap:
    """``H x W`` float32 array of depth values.

    When ``is_metric`` is true, values are distances to the camera in meters;
    otherwise they are relative (defined up to scale and shift).
    """

    values: np.ndarray
    is_metric: bool
    model_name: str
    device: str
    inference_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"Depth map must be 2-D, got shape {self.values.shape}.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.values.shape[0]), int(self.values.shape[1])

    @property
    def min_depth(self) -> float:
        return float(np.min(self.values))

    @property
    def max_depth(self) -> float:
        return float(np.max(self.values))

    @property
    def depth_range(self) -> float:
        return self.max_depth - self.min_depth

    def depth_at(self, u: int, v: int) -> float:
        """Depth at column ``u`` and row ``v``."""
        return float(self.values[v, u])
