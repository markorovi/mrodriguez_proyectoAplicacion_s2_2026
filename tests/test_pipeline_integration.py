"""End-to-end analysis with the real pretrained models (depth and segmentation).

Opt-in, like every test that loads pretrained models:

    RUN_INTEGRATION=1 pytest tests/test_pipeline_integration.py -v

Uses the synthetic sample pair in ``data/samples`` and writes the annotated
result of the "moved" pair to ``data/samples/comparison_output_sample.png``.
Every real pair found by :mod:`tests.real_pairs` (``real_before<suffix>`` /
``real_after<suffix>``) is analyzed too, checked against the expectations in
``data/samples/real_pairs.json`` and written to
``real_comparison_output<suffix>.png``.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest
import torch
from PIL import Image

from config.paths import SAMPLES_DIR
from controllers import PipelineController
from models import ChangeType
from tests.real_pairs import real_pair_params, unmatched_real_images

PIPELINE_BUDGET_SECONDS = 15.0
COMPARISON_OUTPUT = SAMPLES_DIR / "comparison_output_sample.png"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run tests that load pretrained models",
    ),
]


@pytest.fixture(scope="module")
def controller():
    with PipelineController() as pipeline:
        yield pipeline


def _analyze(controller, after_name: str):
    return controller.analyze_files(SAMPLES_DIR / "scene_before.png", SAMPLES_DIR / after_name)


def test_moved_box_is_a_single_displacement(controller):
    report = _analyze(controller, "scene_after_moved.png")
    Image.fromarray(report.annotated_image).save(COMPARISON_OUTPUT)

    assert report.alignment.succeeded
    assert [obj.change_type for obj in report.objects] == [ChangeType.DISPLACED]
    obj = report.objects[0]
    # The sample box moves from 35 % to 65 % of the image width.
    assert obj.bbox_before.center[0] == pytest.approx(0.35 * 640, abs=30)
    assert obj.bbox_after.center[0] == pytest.approx(0.65 * 640, abs=30)
    assert obj.displacement.x > 0.2
    assert 20.0 < obj.distance_cm < 150.0


def test_removed_box_is_a_single_removal(controller):
    report = _analyze(controller, "scene_after_removed.png")

    assert [obj.change_type for obj in report.objects] == [ChangeType.REMOVED]
    assert report.objects[0].bbox_before.center[0] == pytest.approx(0.35 * 640, abs=30)


def test_identical_views_have_no_changes(controller):
    assert not _analyze(controller, "scene_before.png").has_changes


def test_same_scene_in_low_light_has_no_changes(controller):
    report = _analyze(controller, "scene_low_light.png")
    assert not report.has_changes
    assert report.warnings


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")
def test_models_run_on_the_gpu_within_budget(controller):
    report = _analyze(controller, "scene_after_moved.png")

    assert controller._estimator.device_name == torch.cuda.get_device_name(0)
    assert controller._segmenter.device_name == torch.cuda.get_device_name(0)
    assert controller._comparator.device_name == torch.cuda.get_device_name(0)
    assert report.total_seconds < PIPELINE_BUDGET_SECONDS


@pytest.mark.parametrize("pair", real_pair_params())
def test_real_photo_pair(controller, pair):
    report = controller.analyze_files(pair.before, pair.after)
    Image.fromarray(report.annotated_image).save(pair.output_path)
    print(f"\n{pair.name}: " + ("; ".join(obj.summary for obj in report.objects) or "sin cambios"))

    assert report.alignment.succeeded, report.alignment.message
    assert report.has_changes, "a real pair is expected to show at least one change"
    assert report.total_seconds < PIPELINE_BUDGET_SECONDS

    expected = pair.expected
    if expected.changes is not None:
        assert Counter(obj.change_type for obj in report.objects) == Counter(expected.changes)
    if expected.label is not None:
        assert report.objects[0].label.name == expected.label
    if expected.distance_cm is not None:
        displaced = [
            obj for obj in report.objects_of(ChangeType.DISPLACED)
            if expected.distance_label is None or obj.label.name == expected.distance_label
        ]
        assert displaced, f"no displaced object named {expected.distance_label!r}"
        measured = displaced[0].distance_cm
        print(f"{pair.name}: {measured:.1f} cm calculados, {expected.distance_cm:.1f} cm medidos")
        assert measured == pytest.approx(expected.distance_cm, rel=expected.distance_tolerance)


def test_every_real_photo_has_a_partner():
    assert unmatched_real_images() == [], "rename them as real_before<suffix> / real_after<suffix>"
