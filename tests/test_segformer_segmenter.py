"""Semantic segmentation through the SegFormer adapter, with a fake model."""

import numpy as np
import pytest
import torch

from config import DevicePreference, SegmentationSettings
from models import ImageData, SegmentationError, SegmentationMask
from services.inference import DeviceManager
from services.segmentation import (
    ADE20K_SPANISH_NAMES,
    ADE20K_STRUCTURAL_CLASSES,
    HuggingFaceSegmentationModelLoader,
    RegionLabeler,
    SegFormerSegmenter,
    structural_class_ids,
)
from tests.doubles import BOX_ID, FLOOR_ID, WALL_ID, FailingSegformerModel, FakeSegmentationModelLoader, box_mask


def _segmenter(loader: FakeSegmentationModelLoader | None = None) -> SegFormerSegmenter:
    return SegFormerSegmenter(
        SegmentationSettings(),
        device_manager=DeviceManager(DevicePreference.CPU),
        model_loader=loader or FakeSegmentationModelLoader(),
    )


def test_mask_has_input_resolution_and_class_names(scene_image):
    mask = _segmenter().segment(scene_image)

    assert isinstance(mask, SegmentationMask)
    assert mask.shape == (scene_image.height, scene_image.width)
    assert mask.class_ids.dtype == np.int32
    assert mask.id2label[BOX_ID] == "box"
    assert mask.model_name == SegmentationSettings().model_id
    assert mask.device == "CPU"
    assert mask.inference_seconds >= 0.0


def test_mask_follows_image_content(scene_image):
    mask = _segmenter().segment(scene_image)
    box = box_mask(scene_image.pixels)

    assert np.mean(mask.class_ids[box] == BOX_ID) > 0.8
    assert np.mean(mask.class_ids[~box] == WALL_ID) > 0.95


def test_structural_classes_are_marked_as_ignored(scene_image):
    mask = _segmenter().segment(scene_image)
    assert {WALL_ID, FLOOR_ID} <= mask.ignored_class_ids
    assert BOX_ID not in mask.ignored_class_ids


def test_changed_region_is_named_in_spanish(scene_image):
    mask = _segmenter().segment(scene_image)
    label = RegionLabeler().describe(mask, box_mask(scene_image.pixels))
    assert label.name == "caja"


def test_large_images_are_segmented_at_full_resolution():
    rgb = np.zeros((1500, 2000, 3), np.uint8)
    rgb[600:900, 800:1200] = (200, 40, 40)
    mask = _segmenter().segment(ImageData(rgb, "large"))

    assert mask.shape == (1500, 2000)
    assert mask.class_ids[750, 1000] == BOX_ID
    assert mask.class_ids[100, 100] == WALL_ID


def test_model_loads_lazily_once_and_unloads():
    loader = FakeSegmentationModelLoader()
    segmenter = _segmenter(loader)
    assert not segmenter.is_loaded and segmenter.device_name == ""

    image = ImageData(np.zeros((64, 64, 3), np.uint8), "tiny")
    segmenter.segment(image)
    segmenter.segment(image)
    assert len(loader.load_calls) == 1
    assert loader.load_calls[0][0] == SegmentationSettings().model_id

    segmenter.unload()
    assert not segmenter.is_loaded


def test_context_manager_frees_the_model():
    with _segmenter() as segmenter:
        assert segmenter.is_loaded
    assert not segmenter.is_loaded


def test_non_rgb_input_is_rejected():
    with pytest.raises(SegmentationError):
        _segmenter().segment(ImageData(np.zeros((64, 64), np.uint8), "gray"))


def test_runtime_failure_is_reported_as_segmentation_error(scene_image):
    segmenter = _segmenter(FakeSegmentationModelLoader(FailingSegformerModel()))
    with pytest.raises(SegmentationError, match="tarjeta gráfica"):
        segmenter.segment(scene_image)


def test_missing_weights_raise_segmentation_error(tmp_path):
    loader = HuggingFaceSegmentationModelLoader(tmp_path, local_files_only=True)
    with pytest.raises(SegmentationError, match="segmentación"):
        loader.load("nvidia/segformer-b0-finetuned-ade-512-512", torch.device("cpu"))


def test_translations_cover_the_requested_objects():
    assert ADE20K_SPANISH_NAMES["chair"] == "silla"
    assert ADE20K_SPANISH_NAMES["table"] == "mesa"
    assert ADE20K_SPANISH_NAMES["box"] == "caja"
    assert not set(ADE20K_SPANISH_NAMES) & ADE20K_STRUCTURAL_CLASSES


def test_structural_ids_are_resolved_by_name():
    assert structural_class_ids({0: "wall", 1: "chair", 2: "floor "}) == frozenset({0, 2})
