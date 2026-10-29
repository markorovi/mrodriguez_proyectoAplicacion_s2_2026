"""End-to-end depth estimation with the real pretrained checkpoint.

Downloads the weights on first run (~100 MB), so it is opt-in:

    RUN_INTEGRATION=1 pytest -m integration
"""

import os

import numpy as np
import pytest

from config import DepthSettings
from config.paths import SAMPLES_DIR
from services.capture import ImageAcquisitionService
from services.inference import create_depth_estimator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run tests that load pretrained models",
    ),
]


@pytest.fixture(scope="module")
def depth_of_sample():
    image = ImageAcquisitionService().from_file(SAMPLES_DIR / "scene_before.png").image
    with create_depth_estimator(DepthSettings()) as estimator:
        return estimator.estimate(image)


def test_real_model_produces_metric_depth_at_input_resolution(depth_of_sample):
    assert depth_of_sample.shape == (480, 640)
    assert depth_of_sample.is_metric
    assert np.isfinite(depth_of_sample.values).all()


def test_real_model_depth_is_in_indoor_range(depth_of_sample):
    assert 0.05 < depth_of_sample.min_depth
    assert depth_of_sample.max_depth < 20.0


def test_real_model_perceives_back_wall_behind_foreground_floor(depth_of_sample):
    values = depth_of_sample.values
    back_wall = np.median(values[120:200, 180:260])
    near_floor = np.median(values[440:480, 260:380])
    assert near_floor < back_wall
