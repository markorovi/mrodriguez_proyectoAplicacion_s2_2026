"""Discovery of real photo pairs and their declared expectations."""

import json

import pytest

from models import ChangeType
from tests.real_pairs import find_real_pairs, unmatched_real_images


@pytest.fixture
def samples(tmp_path):
    def _touch(*names):
        for name in names:
            (tmp_path / name).write_bytes(b"")
        return tmp_path

    return _touch


def test_pairs_are_matched_by_suffix_across_extensions(samples):
    directory = samples(
        "real_before.jpg", "real_after.jpg", "real_before_2.png", "real_after_2.JPG", "scene_before.png"
    )
    pairs = find_real_pairs(directory)

    assert [pair.name for pair in pairs] == ["real", "real_2"]
    assert (pairs[1].before.name, pairs[1].after.name) == ("real_before_2.png", "real_after_2.JPG")


def test_incomplete_pairs_are_left_out_and_reported(samples):
    directory = samples("real_before.jpg", "real_after.jpg", "real_before_3.jpg", "real_afer_4.jpg")

    assert [pair.name for pair in find_real_pairs(directory)] == ["real"]
    assert unmatched_real_images(directory) == ["real_before_3.jpg"]


def test_non_image_files_are_ignored(samples):
    directory = samples("real_before.txt", "real_after.txt")
    assert find_real_pairs(directory) == []
    assert unmatched_real_images(directory) == []


def test_output_path_follows_the_pair_suffix(samples):
    directory = samples("real_before.jpg", "real_after.jpg", "real_before_2.jpg", "real_after_2.jpg")
    outputs = [pair.output_path.name for pair in find_real_pairs(directory)]
    assert outputs == ["real_comparison_output.png", "real_comparison_output_2.png"]


def test_expectations_are_read_from_json(samples):
    directory = samples("real_before.jpg", "real_after.jpg", "real_before_2.jpg", "real_after_2.jpg")
    (directory / "real_pairs.json").write_text(
        json.dumps(
            {"real_2": {"changes": ["DESPLAZAMIENTO", "REMOVED"], "label": "silla", "distance_cm": 45,
                        "distance_label": "mesa"}}
        ),
        encoding="utf-8",
    )
    first, second = find_real_pairs(directory)

    assert first.expected.changes is None and first.expected.label is None
    assert second.expected.changes == (ChangeType.DISPLACED, ChangeType.REMOVED)
    assert second.expected.label == "silla"
    assert second.expected.distance_cm == 45
    assert second.expected.distance_label == "mesa"
    assert first.expected.distance_label is None
    assert second.expected.distance_tolerance == 0.3
