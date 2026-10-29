"""Test doubles for hardware and pretrained models."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from services.inference.model_loader import DepthModelBundle


class FakeVideoDevice:
    """Stands in for ``cv2.VideoCapture``; returns a fixed BGR frame."""

    def __init__(self, frame: Optional[np.ndarray], opened: bool = True) -> None:
        self.frame = frame
        self.opened = opened
        self.read_calls = 0
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        self.read_calls += 1
        if self.frame is None:
            return False, None
        return True, self.frame.copy()

    def set(self, prop_id: int, value: float) -> bool:
        self.properties[prop_id] = value
        return True

    def release(self) -> None:
        self.released = True


class FakeDepthProcessor:
    """Resizes the image to a fixed square tensor, like a real pre-processor."""

    def __init__(self, size: int = 56) -> None:
        self.size = size
        self.received: list[np.ndarray] = []

    def __call__(self, images: np.ndarray, return_tensors: str) -> dict[str, torch.Tensor]:
        self.received.append(images)
        tensor = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return {"pixel_values": F.interpolate(tensor, size=(self.size, self.size))}


class FakeDepthModel(torch.nn.Module):
    """Returns a positive "depth" derived from the pixel intensities."""

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(predicted_depth=pixel_values.mean(dim=1) * 4.0 + 0.5)


class FailingDepthModel(torch.nn.Module):
    """Simulates an out-of-memory failure during inference."""

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        raise RuntimeError("CUDA out of memory")


class FakeModelLoader:
    """Records load requests and hands out fake bundles."""

    def __init__(self, model: Optional[torch.nn.Module] = None) -> None:
        self.model = model or FakeDepthModel()
        self.processor = FakeDepthProcessor()
        self.load_calls: list[tuple[str, torch.device]] = []

    def load(self, model_id: str, device: torch.device) -> DepthModelBundle:
        self.load_calls.append((model_id, device))
        return DepthModelBundle(processor=self.processor, model=self.model.to(device))
