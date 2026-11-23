"""Known-displacement measurement script (RF-005 evidence), with synthetic scenes and doubles.

Runs without the personal photos: the end-to-end case writes synthetic
``pos1`` ... ``pos3`` into pytest's temporary directory.
"""

import json

import pytest
from PIL import Image

from config import AppSettings
from config.paths import SAMPLES_DIR
from controllers import PipelineController
from models import ChangedObject, ChangeType, Point3D, SemanticLabel
from scripts.measure_known_displacement import (
    AMBIGUOUS,
    MEASURED,
    NO_MEASUREMENT,
    PairResult,
    by_k,
    classify_pair,
    expected_cm,
    main,
    pair_indices,
    summarize,
)
from tests.doubles import SceneDepthEstimator, SceneSegmenter
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder


def _object(change_type: ChangeType, distance_m: float = 0.0, name: str = "caja") -> ChangedObject:
    moved = change_type is ChangeType.DISPLACED
    return ChangedObject(
        object_id=1,
        change_type=change_type,
        label=SemanticLabel(name, "box", 1.0),
        depth_change_cm=10.0,
        area_px=500,
        position_before=Point3D(0.0, 0.0, 2.0),
        position_after=Point3D(distance_m, 0.0, 2.0) if moved else None,
    )


def _measured(before: int, after: int, measured_cm: float, step: float = 5.6) -> PairResult:
    return PairResult(before, after, expected_cm(step, before, after), MEASURED, measured_cm=measured_cm)


def test_pairs_are_consecutive_or_all_in_a_fixed_order():
    assert pair_indices(7, "consecutive") == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    every = pair_indices(7, "all")
    assert len(every) == 21
    assert every[:7] == [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3)]
    assert every[-1] == (6, 7)
    assert all(i < j for i, j in every)


def test_expected_displacement_grows_with_the_steps_between_photos():
    values = [expected_cm(5.6, 1, j) for j in range(2, 8)]
    assert values == pytest.approx([5.6, 11.2, 16.8, 22.4, 28.0, 33.6])
    assert expected_cm(5.6, 3, 5) == pytest.approx(11.2)


def test_summary_statistics_use_only_the_measured_pairs():
    results = [
        _measured(1, 2, 6.6),   # +1.0 cm, +17.86 %
        _measured(2, 3, 4.6),   # -1.0 cm, -17.86 %
        _measured(1, 3, 14.2),  # +3.0 cm, +26.79 %
        PairResult(3, 4, 5.6, NO_MEASUREMENT),
    ]
    summary = summarize(results)

    assert (summary.pairs, summary.measured) == (4, 3)
    assert summary.bias_cm == pytest.approx(1.0)
    assert summary.std_cm == pytest.approx(2.0)
    assert summary.mae_cm == pytest.approx(5.0 / 3.0)
    assert summary.mape_pct == pytest.approx((100 / 5.6 + 100 / 5.6 + 300 / 11.2) / 3)
    assert summary.max_abs_error_cm == pytest.approx(3.0)

    rows = {row.k: row for row in by_k(results, 5.6)}
    assert (rows[1].pairs, rows[1].measured) == (3, 2)
    assert rows[1].mean_rel_error_pct == pytest.approx(0.0)
    assert rows[1].mean_abs_rel_error_pct == pytest.approx(100 / 5.6)
    assert rows[2].expected_cm == pytest.approx(11.2)
    assert rows[2].mean_rel_error_pct == pytest.approx(300 / 11.2)


def test_pair_without_a_displaced_object_has_no_measurement():
    result = classify_pair([_object(ChangeType.REMOVED)], 1, 2, 5.6)

    assert result.status == NO_MEASUREMENT
    assert result.measured_cm is None and result.error_cm is None
    assert result.candidates == ()
    assert [c.change_type for c in result.other_objects] == [ChangeType.REMOVED.value]
    assert summarize([result]).measured == 0


def test_pair_with_several_displaced_objects_is_ambiguous_and_lists_them():
    objects = [_object(ChangeType.DISPLACED, 0.05, "caja"), _object(ChangeType.DISPLACED, 0.30, "silla")]
    result = classify_pair(objects, 1, 2, 5.6)

    assert result.status == AMBIGUOUS
    assert result.measured_cm is None and result.label is None
    assert [(c.label, round(c.distance_cm, 1)) for c in result.candidates] == [("caja", 5.0), ("silla", 30.0)]


def test_single_displaced_object_is_the_measurement():
    result = classify_pair([_object(ChangeType.DISPLACED, 0.06), _object(ChangeType.ADDED)], 2, 3, 5.6)

    assert result.status == MEASURED
    assert result.measured_cm == pytest.approx(6.0)
    assert result.error_cm == pytest.approx(0.4)
    assert result.rel_error_pct == pytest.approx(40 / 5.6)
    assert result.label == "caja" and result.change_type == ChangeType.DISPLACED.value


def _samples_snapshot():
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in SAMPLES_DIR.iterdir())


def test_script_measures_synthetic_photos_without_writing_into_the_samples(tmp_path, capsys):
    photos = tmp_path / "fotos"
    photos.mkdir()
    builder = SyntheticSceneBuilder(640, 480, seed=7)
    for n, x in enumerate((0.3, 0.5, 0.7), start=1):
        Image.fromarray(builder.render([SceneObject(center_x=x)])).save(photos / f"pos{n}.png")
    before = _samples_snapshot()
    out = tmp_path / "salida" / "medicion.json"

    def controller():
        return PipelineController(AppSettings(), depth_estimator=SceneDepthEstimator(), segmenter=SceneSegmenter())

    code = main(
        ["--folder", str(photos), "--count", "3", "--step-cm", "10", "--json", str(out)], controller
    )

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [(p["before"], p["after"]) for p in data["pairs"]] == [(1, 2), (1, 3), (2, 3)]
    assert [p["expected_cm"] for p in data["pairs"]] == pytest.approx([10.0, 20.0, 10.0])
    assert all(p["status"] == MEASURED and p["label"] == "caja" for p in data["pairs"])
    assert data["summary_all"]["measured"] == 3 and data["summary_consecutive"]["pairs"] == 2
    assert "no son mediciones independientes" in capsys.readouterr().out
    assert _samples_snapshot() == before


def test_json_inside_the_samples_folder_is_refused(tmp_path, capsys):
    target = SAMPLES_DIR / "no_debe_existir.json"
    code = main(["--folder", str(tmp_path), "--count", "2", "--step-cm", "19.8", "--json", str(target)])

    assert code == 1
    assert not target.exists()
    assert "data/samples" in capsys.readouterr().err


def test_missing_photo_is_reported(tmp_path, capsys):
    assert main(["--folder", str(tmp_path), "--count", "2", "--step-cm", "19.8"]) == 1
    assert "pos1" in capsys.readouterr().err


# --- The step has no default (PROMPT 24): 5.6 cm was the spacer, the box moved 19.8 cm ---------------


def test_step_is_required(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--folder", str(tmp_path), "--count", "2"])

    assert stopped.value.code == 2
    assert "--step-cm" in capsys.readouterr().err


@pytest.mark.parametrize("step", ["0", "-5.6"])
def test_step_must_be_positive(tmp_path, capsys, step):
    with pytest.raises(SystemExit) as stopped:
        main(["--folder", str(tmp_path), "--count", "2", f"--step-cm={step}"])

    assert stopped.value.code == 2
    assert "mayor que cero" in capsys.readouterr().err


def test_step_used_is_printed_and_saved(tmp_path, capsys):
    photos = tmp_path / "fotos"
    photos.mkdir()
    builder = SyntheticSceneBuilder(640, 480, seed=7)
    for n, x in enumerate((0.3, 0.6), start=1):
        Image.fromarray(builder.render([SceneObject(center_x=x)])).save(photos / f"pos{n}.png")
    out = tmp_path / "medicion.json"

    def controller():
        return PipelineController(AppSettings(), depth_estimator=SceneDepthEstimator(), segmenter=SceneSegmenter())

    code = main(["--folder", str(photos), "--count", "2", "--step-cm", "19.8", "--json", str(out)], controller)

    assert code == 0
    assert "Paso real usado (--step-cm): 19.8 cm" in capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["step_cm"] == 19.8
    assert data["pairs"][0]["expected_cm"] == pytest.approx(19.8)
