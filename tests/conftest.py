"""Shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from models.image_data import ImageData
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder


@pytest.fixture(scope="session")
def scene_rgb() -> np.ndarray:
    """A 640x480 synthetic indoor scene with one box on the floor."""
    return SyntheticSceneBuilder(640, 480, seed=1).render([SceneObject(center_x=0.4)])


@pytest.fixture
def scene_image(scene_rgb: np.ndarray) -> ImageData:
    return ImageData(pixels=scene_rgb.copy(), source="memory")


@pytest.fixture
def write_image(tmp_path: Path) -> Callable[..., Path]:
    """Save an array with Pillow (mode inferred from its shape) and return the path."""

    def _write(array: np.ndarray, name: str = "image.png") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
        return path

    return _write


@pytest.fixture(scope="session")
def qapp():
    """A Qt application that renders without opening windows."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
