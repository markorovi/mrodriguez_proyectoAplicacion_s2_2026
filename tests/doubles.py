"""Test doubles for hardware and pretrained models."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from services.inference.base_estimator import BaseDepthEstimator
from services.inference.model_loader import DepthModelBundle
from services.segmentation.base_segmenter import BaseSegmenter
from services.segmentation.model_loader import SegmentationModelBundle


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


def box_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels of the red boxes drawn by ``SyntheticSceneBuilder`` (any face)."""
    red, green = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16)
    return (red - green) > 70


def scene_depth(rgb: np.ndarray, box_offset_m: float = 0.6) -> np.ndarray:
    """Plausible metric depth of a synthetic room: farther towards the top, boxes closer."""
    height, width = rgb.shape[:2]
    rows = np.arange(height, dtype=np.float32)[:, None] / height
    depth = np.broadcast_to(4.0 - 2.0 * rows, (height, width)).astype(np.float32).copy()
    depth[box_mask(rgb)] -= box_offset_m
    return depth


class SceneDepthEstimator(BaseDepthEstimator):
    """Depth estimator double that knows the geometry of the synthetic scenes."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls = 0

    @property
    def name(self) -> str:
        return "scene-depth-double"

    @property
    def is_metric(self) -> bool:
        return True

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def device_name(self) -> str:
        return "CPU" if self.loaded else ""

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        return scene_depth(rgb)


# Class ids follow the ADE20K numbering used by the SegFormer checkpoints.
WALL_ID, SKY_ID, FLOOR_ID, BOX_ID = 0, 2, 3, 41
FAKE_ID2LABEL = {i: f"class {i}" for i in range(BOX_ID + 1)}
FAKE_ID2LABEL.update({WALL_ID: "wall", SKY_ID: "sky", FLOOR_ID: "floor", BOX_ID: "box"})

# Pure sky blue: far bluer than anything in the synthetic rooms (blue - red <= 61).
SKY_RGB = (60, 150, 240)


class FakeSegformerProcessor:
    """Resizes the image to a fixed square tensor, like the SegFormer pre-processor."""

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def __call__(self, images: np.ndarray, return_tensors: str) -> dict[str, torch.Tensor]:
        tensor = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).float()
        return {"pixel_values": F.interpolate(tensor, size=(self.size, self.size), mode="bilinear")}


class FakeSegformerModel(torch.nn.Module):
    """Quarter-resolution logits: "box" on reddish pixels, "wall" elsewhere."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(id2label=dict(FAKE_ID2LABEL))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        small = F.avg_pool2d(pixel_values, 4)
        redness = small[:, 0] - small[:, 1]
        logits = torch.zeros(small.shape[0], len(FAKE_ID2LABEL), *small.shape[2:], device=small.device)
        logits[:, WALL_ID] = 70.0
        logits[:, BOX_ID] = redness
        return SimpleNamespace(logits=logits)


class OutdoorSegformerModel(FakeSegformerModel):
    """Like :class:`FakeSegformerModel`, but sees "sky" on strongly blue pixels (see ``SKY_RGB``)."""

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        output = super().forward(pixel_values)
        small = F.avg_pool2d(pixel_values, 4)
        output.logits[:, SKY_ID] = small[:, 2] - small[:, 0]
        return output


class FailingSegformerModel(FakeSegformerModel):
    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        raise RuntimeError("CUDA out of memory")


class FakeSegmentationModelLoader:
    """Records load requests and hands out fake SegFormer bundles."""

    def __init__(self, model: Optional[torch.nn.Module] = None) -> None:
        self.model = model or FakeSegformerModel()
        self.load_calls: list[tuple[str, torch.device]] = []

    def load(self, model_id: str, device: torch.device) -> SegmentationModelBundle:
        self.load_calls.append((model_id, device))
        return SegmentationModelBundle(processor=FakeSegformerProcessor(), model=self.model.to(device))


class SceneSegmenter(BaseSegmenter):
    """A segmenter that is not SegFormer: "box" on the red boxes of the synthetic scene, "wall" elsewhere.

    It implements only the :class:`BaseSegmenter` contract, so it shows that
    another segmentation model can replace SegFormer without changing the
    modules that use it. ``segment_calls`` counts the images it labelled.
    """

    def __init__(self) -> None:
        self._loaded = False
        self.load_calls = 0
        self.segment_calls = 0

    @property
    def name(self) -> str:
        return "scene-segmenter"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_name(self) -> str:
        return "CPU" if self._loaded else ""

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        self.segment_calls += 1
        class_ids = np.full(rgb.shape[:2], WALL_ID, dtype=np.int32)
        class_ids[box_mask(rgb)] = BOX_ID
        return class_ids

    def _class_names(self) -> dict[int, str]:
        return dict(FAKE_ID2LABEL)
