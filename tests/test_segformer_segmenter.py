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
    with pytest.raises(SegmentationError, match="mucha memoria"):
        segmenter.segment(scene_image)


def test_missing_weights_raise_segmentation_error(tmp_path):
    loader = HuggingFaceSegmentationModelLoader(tmp_path, local_files_only=True)
    with pytest.raises(SegmentationError, match="identifica los objetos"):
        loader.load("nvidia/segformer-b0-finetuned-ade-512-512", torch.device("cpu"))


def test_translations_cover_the_requested_objects():
    assert ADE20K_SPANISH_NAMES["chair"] == "silla"
    assert ADE20K_SPANISH_NAMES["table"] == "mesa"
    assert ADE20K_SPANISH_NAMES["box"] == "caja"
    assert not set(ADE20K_SPANISH_NAMES) & ADE20K_STRUCTURAL_CLASSES


def test_structural_ids_are_resolved_by_name():
    assert structural_class_ids({0: "wall", 1: "chair", 2: "floor "}) == frozenset({0, 2})


# --- Segmenter contract (RA-003): BaseSegmenter template method and the factory -------------------

from services.segmentation import BaseSegmenter, create_segmenter  # noqa: E402


class _ProbeSegmenter(BaseSegmenter):
    """Minimal subclass whose prediction is chosen by each test; counts the loads."""

    def __init__(self, predict) -> None:
        self._predict_with = predict
        self._loaded = False
        self.load_calls = 0

    name = "probe"

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
        return self._predict_with(rgb)

    def _class_names(self) -> dict[int, str]:
        return {0: "wall", 1: "box"}


def test_factory_builds_a_segformer_segmenter_behind_the_base_contract():
    segmenter = create_segmenter()
    assert isinstance(segmenter, BaseSegmenter)
    assert isinstance(segmenter, SegFormerSegmenter)
    assert not segmenter.is_loaded


def test_base_segmenter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseSegmenter()


def test_template_rejects_non_rgb_input_without_loading():
    probe = _ProbeSegmenter(lambda rgb: np.zeros(rgb.shape[:2], np.int32))

    with pytest.raises(SegmentationError, match="foto a color"):
        probe.segment(ImageData(np.zeros((64, 64), np.uint8), "gray"))

    assert probe.load_calls == 0


def test_template_wraps_runtime_failures_in_segmentation_error():
    def failing(rgb):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(SegmentationError, match="tarjeta gráfica"):
        _ProbeSegmenter(failing).segment(ImageData(np.zeros((64, 64, 3), np.uint8), "tiny"))


def test_template_rejects_an_output_of_the_wrong_resolution():
    probe = _ProbeSegmenter(lambda rgb: np.zeros((8, 8), np.int32))

    with pytest.raises(SegmentationError, match="reinicie la aplicación"):
        probe.segment(ImageData(np.zeros((64, 64, 3), np.uint8), "tiny"))


# --- Inference failure message depends on the device (PROMPT 16) ----------------------------------

from typing import Optional  # noqa: E402

from services.inference import ComputeDevice  # noqa: E402
from tests.jargon import jargon_found  # noqa: E402

GPU_SEGMENTATION_FAILURE_MESSAGE = (
    "No se pudieron identificar los objetos de la imagen. "
    "Cierre otras aplicaciones que usen la tarjeta gráfica e intente de nuevo."
)


class _FailingOnGpuSegmenter(_ProbeSegmenter):
    """Like the probe, but on a (simulated) CUDA device and failing every inference."""

    def __init__(self) -> None:
        def failing(rgb):
            raise RuntimeError("CUDA out of memory")

        super().__init__(failing)

    @property
    def compute_device(self) -> Optional[ComputeDevice]:
        return ComputeDevice(torch.device("cuda"), "GPU de prueba")


def test_segmentation_failure_message_names_the_graphics_card_only_on_gpu(scene_image):
    cpu = _segmenter(FakeSegmentationModelLoader(FailingSegformerModel()))
    with pytest.raises(SegmentationError) as on_cpu:
        cpu.segment(scene_image)
    with pytest.raises(SegmentationError) as on_gpu:
        _FailingOnGpuSegmenter().segment(scene_image)

    cpu_text, gpu_text = str(on_cpu.value), str(on_gpu.value)
    assert "tarjeta" not in cpu_text
    assert "mucha memoria" in cpu_text
    assert gpu_text == GPU_SEGMENTATION_FAILURE_MESSAGE
    assert jargon_found(cpu_text) == [] and jargon_found(gpu_text) == []
