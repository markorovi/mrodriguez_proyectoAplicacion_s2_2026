"""Integration with real hardware: CUDA GPU, physical webcam and real photos.

Opt-in, like every test that loads pretrained models:

    RUN_INTEGRATION=1 pytest tests/test_hardware_integration.py -v -s

Optional inputs:
- ``data/samples/real_before<suffix>.<ext>`` and ``real_after<suffix>.<ext>``:
  pairs of real photos of the same indoor space (see :mod:`tests.real_pairs`).
- ``CAMERA_INDEX``: webcam to use. If unset, indices 0-3 are probed and the
  first one that delivers a frame is used.

Tests whose hardware or inputs are missing are skipped, not failed.

The colored depth map is computed from the synthetic ``scene_before.png`` only,
never from a personal photo, and is written to pytest's temporary directory.
With ``UPDATE_SAMPLES=1`` it replaces ``data/samples/depth_output_sample.png``
(see :mod:`tests.sample_outputs`).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image

from config import DepthSettings, DevicePreference
from config.paths import SAMPLES_DIR
from models import CameraError, ImageData
from services.capture import ImageAcquisitionService
from services.inference import DepthAnythingV2Estimator, DepthLatencyBenchmark
from tests.real_pairs import real_pair_params
from tests.sample_outputs import output_path
from utils.depth_visualization import colorize_depth

PIPELINE_BUDGET_SECONDS = 15.0
DEPTH_OUTPUT = "depth_output_sample.png"
# The versioned depth sample always comes from the synthetic scene, never from a personal photo.
DEPTH_SAMPLE_SOURCE = SAMPLES_DIR / "scene_before.png"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run tests that load pretrained models",
    ),
]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")


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
@pytest.mark.parametrize("pair", real_pair_params())
def test_real_photo_pair_produces_metric_depth(gpu_estimator, acquisition, pair):
    for path in (pair.before, pair.after):
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
def test_depth_output_sample_is_saved(gpu_estimator, acquisition, tmp_path):
    image = acquisition.from_file(DEPTH_SAMPLE_SOURCE).image
    colored = colorize_depth(gpu_estimator.estimate(image))

    destination = output_path(DEPTH_OUTPUT, tmp_path)
    Image.fromarray(colored).save(destination)

    assert destination.is_file()
    with Image.open(destination) as saved:
        assert saved.size == (image.width, image.height)
