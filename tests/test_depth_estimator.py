"""Depth map generation through the estimator contract, with a fake model."""

import numpy as np
import pytest

from config import DepthSettings, DevicePreference
from models import DepthEstimationError, DepthMap, ImageData, ModelLoadError
from services.inference import (
    BaseDepthEstimator,
    DepthAnythingV2Estimator,
    DeviceManager,
    HuggingFaceDepthModelLoader,
    create_depth_estimator,
)
from tests.doubles import FailingDepthModel, FakeModelLoader


@pytest.fixture
def loader() -> FakeModelLoader:
    return FakeModelLoader()


def _estimator(loader: FakeModelLoader, cuda_available: bool = False) -> DepthAnythingV2Estimator:
    return DepthAnythingV2Estimator(
        DepthSettings(),
        device_manager=DeviceManager(DevicePreference.AUTO, cuda_available=lambda: cuda_available),
        model_loader=loader,
    )


def test_depth_map_matches_input_resolution(loader, scene_image):
    depth = _estimator(loader).estimate(scene_image)

    assert isinstance(depth, DepthMap)
    assert depth.shape == (scene_image.height, scene_image.width)
    assert depth.values.dtype == np.float32


def test_depth_values_are_finite_and_non_negative(loader, scene_image):
    values = _estimator(loader).estimate(scene_image).values
    assert np.isfinite(values).all()
    assert values.min() >= 0.0


def test_depth_map_carries_model_metadata(loader, scene_image):
    depth = _estimator(loader).estimate(scene_image)

    assert depth.is_metric
    assert depth.model_name == DepthSettings().model_id
    assert depth.device == "CPU"
    assert depth.inference_seconds >= 0.0


def test_depth_follows_image_content(loader, scene_image):
    depth = _estimator(loader).estimate(scene_image)
    # The fake model maps brightness to depth: the bright ceiling must come
    # out "deeper" than the dark floor.
    assert depth.depth_at(320, 10) > depth.depth_at(320, 470)


def test_processor_receives_the_original_rgb_pixels(loader, scene_image):
    _estimator(loader).estimate(scene_image)
    np.testing.assert_array_equal(loader.processor.received[0], scene_image.pixels)


def test_model_is_loaded_lazily_and_only_once(loader, scene_image):
    estimator = _estimator(loader)
    assert not estimator.is_loaded

    estimator.estimate(scene_image)
    estimator.estimate(scene_image)

    assert estimator.is_loaded
    assert len(loader.load_calls) == 1
    assert loader.load_calls[0][0] == DepthSettings().model_id


def test_context_manager_loads_and_releases_model(loader, scene_image):
    estimator = _estimator(loader)
    with estimator as active:
        assert active.is_loaded
        active.estimate(scene_image)
    assert not estimator.is_loaded


def test_cpu_fallback_notice_is_exposed(loader):
    estimator = _estimator(loader, cuda_available=False)
    estimator.load()
    assert estimator.compute_device.notice is not None


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((480, 640), dtype=np.uint8),
        np.zeros((480, 640, 4), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.float32),
    ],
)
def test_non_rgb_input_is_rejected_without_loading(loader, pixels):
    estimator = _estimator(loader)
    with pytest.raises(DepthEstimationError):
        estimator.estimate(ImageData(pixels=pixels, source="test"))
    assert loader.load_calls == []


def test_inference_failure_is_wrapped(scene_image):
    estimator = _estimator(FakeModelLoader(model=FailingDepthModel()))
    with pytest.raises(DepthEstimationError) as info:
        estimator.estimate(scene_image)
    assert isinstance(info.value.__cause__, RuntimeError)


class _WrongShapeEstimator(BaseDepthEstimator):
    name = "wrong-shape"
    is_metric = False
    is_loaded = True
    device_name = "CPU"

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        return np.zeros((10, 10), dtype=np.float32)


def test_output_resolution_mismatch_is_detected(scene_image):
    with pytest.raises(DepthEstimationError):
        _WrongShapeEstimator().estimate(scene_image)


def test_depth_map_rejects_non_2d_values():
    with pytest.raises(ValueError):
        DepthMap(values=np.zeros((2, 2, 1)), is_metric=True, model_name="m", device="CPU")


def test_factory_builds_depth_anything_estimator():
    estimator = create_depth_estimator()
    assert isinstance(estimator, DepthAnythingV2Estimator)
    assert not estimator.is_loaded


def test_hugging_face_loader_reports_missing_weights(tmp_path):
    loader = HuggingFaceDepthModelLoader(tmp_path, local_files_only=True)
    with pytest.raises(ModelLoadError):
        loader.load("depth-anything/does-not-exist", DeviceManager(DevicePreference.CPU).select().torch_device)
