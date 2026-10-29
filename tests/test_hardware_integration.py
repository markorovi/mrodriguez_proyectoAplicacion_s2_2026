"""Integration with real hardware: CUDA GPU, physical webcam and real photos.

Opt-in, like every test that loads pretrained models:

    RUN_INTEGRATION=1 pytest tests/test_hardware_integration.py -v -s

Optional inputs:
- ``data/samples/real_before.<ext>`` and ``data/samples/real_after.<ext>``:
  a pair of real photos of the same indoor space.
- ``CAMERA_INDEX``: webcam to use. If unset, indices 0-3 are probed and the
  first one that delivers a frame is used.

Tests whose hardware or inputs are missing are skipped, not failed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from PIL import Image

from config import DepthSettings, DevicePreference
from config.image_settings import DEFAULT_EXTENSIONS
from config.paths import SAMPLES_DIR
from models import CameraError, ImageData
from services.capture import ImageAcquisitionService
from services.inference import DepthAnythingV2Estimator, DepthLatencyBenchmark
from utils.depth_visualization import colorize_depth

PIPELINE_BUDGET_SECONDS = 15.0
DEPTH_OUTPUT = SAMPLES_DIR / "depth_output_sample.png"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run tests that load pretrained models",
    ),
]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")


def _find_sample(stem: str) -> Optional[Path]:
    for ext in DEFAULT_EXTENSIONS:
        for candidate in (SAMPLES_DIR / f"{stem}{ext}", SAMPLES_DIR / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


def _real_pair() -> tuple[Path, Path]:
    before, after = _find_sample("real_before"), _find_sample("real_after")
    if before is None or after is None:
        pytest.skip("add data/samples/real_before.<ext> and real_after.<ext> to test real photos")
    return before, after


@pytest.fixture(scope="module")
def acquisition() -> ImageAcquisitionService:
    return ImageAcquisitionService()


@pytest.fixture(scope="module")
def gpu_estimator():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")
    estimator = DepthAnythingV2Estimator(DepthSettings(device=DevicePreference.CUDA))
    estimator.load()
    yield estimator
    estimator.unload()


@pytest.fixture(scope="module")
def webcam_image(acquisition) -> ImageData:
    env_index = os.environ.get("CAMERA_INDEX")
    indices = [int(env_index)] if env_index is not None else range(4)
    for index in indices:
        try:
            return acquisition.from_camera(index).image
        except CameraError:
            continue
    pytest.skip("no webcam delivered a frame")


@requires_cuda
def test_inference_runs_on_the_gpu(gpu_estimator):
    device = gpu_estimator.compute_device
    assert device.is_accelerated
    assert device.notice is None
    print(f"\nGPU: {device.name}")


@requires_cuda
def test_gpu_latency_is_within_the_pipeline_budget(gpu_estimator, acquisition):
    image = acquisition.from_file(SAMPLES_DIR / "scene_before.png").image
    report = DepthLatencyBenchmark(gpu_estimator, repetitions=10).run(image)

    print(
        f"\n{report.device_name}: primera {report.cold_seconds:.3f} s, "
        f"media {report.mean_seconds:.3f} s, par {report.pair_seconds:.3f} s"
    )
    assert report.cold_seconds + report.mean_seconds < PIPELINE_BUDGET_SECONDS
    assert report.pair_seconds < PIPELINE_BUDGET_SECONDS


@requires_cuda
def test_gpu_half_precision_agrees_with_cpu(gpu_estimator, acquisition):
    image = acquisition.from_file(SAMPLES_DIR / "scene_before.png").image
    gpu = gpu_estimator.estimate(image).values
    with DepthAnythingV2Estimator(DepthSettings(device=DevicePreference.CPU)) as cpu_estimator:
        cpu = cpu_estimator.estimate(image).values

    relative = np.abs(gpu - cpu) / np.maximum(cpu, 1e-3)
    assert float(np.median(relative)) < 0.02


@requires_cuda
def test_real_photo_pair_produces_metric_depth(gpu_estimator, acquisition):
    before_path, after_path = _real_pair()
    for path in (before_path, after_path):
        acquired = acquisition.from_file(path)
        depth = gpu_estimator.estimate(acquired.image)
        assert depth.is_metric
        assert depth.shape == (acquired.image.height, acquired.image.width)
        assert np.isfinite(depth.values).all()
        print(f"\n{path.name}: {depth.min_depth:.2f}-{depth.max_depth:.2f} m, {depth.inference_seconds:.3f} s")


@requires_cuda
def test_webcam_frame_produces_metric_depth(gpu_estimator, webcam_image):
    depth = gpu_estimator.estimate(webcam_image)
    assert depth.shape == (webcam_image.height, webcam_image.width)
    assert np.isfinite(depth.values).all()
    assert 0.0 < depth.max_depth < 20.0
    print(
        f"\n{webcam_image.source} {webcam_image.width}x{webcam_image.height}: "
        f"{depth.min_depth:.2f}-{depth.max_depth:.2f} m, {depth.inference_seconds:.3f} s"
    )


@requires_cuda
def test_depth_output_sample_is_saved(gpu_estimator, acquisition):
    source = _find_sample("real_before") or SAMPLES_DIR / "scene_before.png"
    image = acquisition.from_file(source).image
    colored = colorize_depth(gpu_estimator.estimate(image))

    Image.fromarray(colored).save(DEPTH_OUTPUT)

    assert DEPTH_OUTPUT.is_file()
    with Image.open(DEPTH_OUTPUT) as saved:
        assert saved.size == (image.width, image.height)
