"""Latency benchmark and depth colorization, with a fake model."""

import numpy as np
import pytest

from config import DepthSettings, DevicePreference
from models import DepthMap
from services.inference import DepthAnythingV2Estimator, DepthLatencyBenchmark, DeviceManager
from tests.doubles import FakeModelLoader
from utils.depth_visualization import colorize_depth


def _estimator() -> DepthAnythingV2Estimator:
    return DepthAnythingV2Estimator(
        DepthSettings(),
        device_manager=DeviceManager(DevicePreference.CPU),
        model_loader=FakeModelLoader(),
    )


def test_benchmark_reports_every_repetition(scene_image):
    report = DepthLatencyBenchmark(_estimator(), repetitions=4).run(scene_image)

    assert len(report.warm_seconds) == 4
    assert report.device_name == "CPU"
    assert report.image_size == (640, 480)
    assert report.min_seconds <= report.mean_seconds <= report.max_seconds
    assert report.pair_seconds == pytest.approx(2 * report.mean_seconds)
    assert report.load_seconds >= 0 and report.cold_seconds >= 0


def test_benchmark_requires_a_repetition():
    with pytest.raises(ValueError):
        DepthLatencyBenchmark(_estimator(), repetitions=0)


def _depth(values) -> DepthMap:
    return DepthMap(values=np.asarray(values, dtype=np.float32), is_metric=True, model_name="m", device="CPU")


def test_colorized_depth_is_rgb_at_same_resolution():
    image = colorize_depth(_depth(np.linspace(1, 4, 640 * 480).reshape(480, 640)))
    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8


def test_near_surfaces_are_brighter_than_far_ones():
    image = colorize_depth(_depth([[1.0, 5.0]]))
    near, far = image[0, 0].astype(int).sum(), image[0, 1].astype(int).sum()
    assert near > far


def test_constant_depth_does_not_divide_by_zero():
    image = colorize_depth(_depth(np.full((4, 4), 2.0)))
    assert np.all(image == image[0, 0])
