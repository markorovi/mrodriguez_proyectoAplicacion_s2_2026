"""Main window, analysis worker and result screen, rendered offscreen with model doubles."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from PIL import Image
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu, QToolBar

from config import AppSettings, DevicePreference, SegmentationSettings
from config.paths import SAMPLES_DIR
from controllers import STAGES, PipelineController
from models import ChangeType, ComparisonReport, DepthEstimationError
from models.alignment import AlignmentReport
from models.comparison_report import BoundingBox, ChangedObject, DepthComparison, Point3D
from models.segmentation_mask import UNIDENTIFIED_LABEL, SemanticLabel
from services.inference import DeviceManager
from services.segmentation import SegFormerSegmenter
from tests.doubles import FakeSegmentationModelLoader, SceneDepthEstimator
from tests.jargon import JARGON
from ui import report_text
from ui.banner import BannerKind
from ui.image_status import ImageStatus
from ui.image_viewer import ZoomableImageView
from ui.main_window import APP_TITLE, MainWindow, Step
from ui.pipeline_worker import UNEXPECTED_ERROR_MESSAGE, PipelineWorker
from ui.report_widget import ReportWidget
from utils.change_visualization import _caption

BEFORE = SAMPLES_DIR / "scene_before.png"
MOVED = SAMPLES_DIR / "scene_after_moved.png"
LOW_LIGHT = SAMPLES_DIR / "scene_low_light.png"



def _controller(cls=PipelineController) -> PipelineController:
    segmenter = SegFormerSegmenter(
        SegmentationSettings(), DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    return cls(AppSettings(), depth_estimator=SceneDepthEstimator(), segmenter=segmenter)


class FailingController(PipelineController):
    error: Exception = DepthEstimationError("No se pudo calcular la profundidad de la imagen.")

    def analyze(self, before, after, progress=None):
        if progress:
            progress("acquisition")
        raise self.error


class CrashingController(FailingController):
    error = RuntimeError("CUDA out of memory")


@pytest.fixture
def window(qapp):
    windows = []

    def build(cls=PipelineController) -> MainWindow:
        widget = MainWindow(controller=_controller(cls))
        windows.append(widget)
        return widget

    yield build
    for widget in windows:
        widget.close()
        widget.deleteLater()


def _wait_for(condition, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the window did not reach the expected state in time"
        QTest.qWait(20)


def _analyze(main: MainWindow, before=BEFORE, after=MOVED) -> None:
    main.load_files([before, after])
    main.start_analysis()
    _wait_for(lambda: main.step is not Step.ANALYZING)


def _changed(kind: ChangeType, number: int = 1, label: str = "caja", moved_m: float = 0.633) -> ChangedObject:
    box = BoundingBox(10, 10, 20, 20)
    start = Point3D(0.0, 0.0, 2.0)
    return ChangedObject(
        object_id=number,
        change_type=kind,
        label=SemanticLabel(label, "box" if label != UNIDENTIFIED_LABEL else None, 0.9),
        depth_change_cm=12.0,
        area_px=400,
        bbox_before=None if kind is ChangeType.ADDED else box,
        bbox_after=None if kind is ChangeType.REMOVED else BoundingBox(40, 10, 20, 20),
        position_before=start if kind is ChangeType.DISPLACED else None,
        position_after=Point3D(moved_m, 0.0, 2.0) if kind is ChangeType.DISPLACED else None,
    )


def _report(objects=(), warnings=(), intrinsics_source=None) -> ComparisonReport:
    shape = (48, 64)
    return ComparisonReport(
        objects=tuple(objects),
        alignment=AlignmentReport(True, "sift", 100, 100, 50, 40, 0.8, "Imágenes alineadas."),
        comparison=DepthComparison((), np.zeros(shape, np.float32), np.zeros(shape, bool), np.zeros(shape, bool)),
        annotated_image=np.full((*shape, 3), 90, np.uint8),
        warnings=tuple(warnings),
        stage_seconds={"depth": 0.1, "comparison": 0.05},
        intrinsics_source=intrinsics_source,
        focal_length_px=None if intrinsics_source is None else 1556.6,
    )


# ------------------------------------------------------------------ main window


def test_window_starts_on_the_selection_step(window):
    main = window()
    assert main.windowTitle() == APP_TITLE
    assert main.step is Step.SELECTION
    assert not main.can_analyze
    assert main.selection_panel.before_slot.status is ImageStatus.EMPTY
    assert main.error_text == ""


def test_two_valid_photos_enable_the_analysis(window):
    main = window()
    main.load_files([BEFORE, MOVED])

    assert main.selection_panel.before_slot.status is ImageStatus.VALID
    assert main.selection_panel.after_slot.status is ImageStatus.VALID
    assert main.can_analyze


def test_one_photo_at_a_time_fills_the_next_empty_panel(window):
    main = window()
    main.load_files([BEFORE])
    assert main.selection_panel.before_slot.acquired is not None
    assert main.selection_panel.after_slot.acquired is None
    assert not main.can_analyze

    main.load_files([MOVED])
    assert main.selection_panel.after_slot.acquired.image.source.endswith("scene_after_moved.png")
    assert main.can_analyze


def test_low_light_photo_shows_a_yellow_warning_and_can_be_analyzed(window):
    main = window()
    main.load_files([BEFORE, LOW_LIGHT])

    assert "poco iluminada" in main.selection_warning_text
    assert main.selection_warning_text.startswith("Después:")
    assert main.can_analyze


def test_analysis_runs_in_the_background_and_ends_on_the_report(window):
    main = window()
    main.load_files([BEFORE, MOVED])
    main.start_analysis()

    # The call returns at once: the window is free while the worker runs.
    assert main.step is Step.ANALYZING
    assert not main.can_analyze
    assert main.progress_text

    _wait_for(lambda: main.step is not Step.ANALYZING)
    assert main.step is Step.REPORT
    assert main.progress_value == len(STAGES)
    result = main.report_widget
    assert result.report is not None and result.report.has_changes
    assert result.headline_kind is BannerKind.INFO
    assert result.headline_text == report_text.headline(result.report)
    assert any("Objeto desplazado" in card and "Caja" in card and " cm" in card for card in result.card_texts)
    assert result.timing_text.startswith("Análisis completado en")
    assert result.viewer.has_image
    assert result.can_export


def test_same_photo_twice_shows_the_no_changes_banner(window):
    main = window()
    _analyze(main, BEFORE, BEFORE)

    assert main.step is Step.REPORT
    assert main.report_widget.headline_text == "No se detectaron cambios en la escena"
    assert main.report_widget.headline_kind is BannerKind.SUCCESS
    assert main.report_widget.card_texts == []


def test_failed_analysis_returns_to_the_selection_with_a_message(window):
    main = window(FailingController)
    _analyze(main)

    assert main.step is Step.SELECTION
    assert "No se pudo calcular la profundidad" in main.error_text
    # The photos are kept, so the user can try again.
    assert main.can_analyze


def test_unexpected_failure_shows_a_friendly_message(window):
    main = window(CrashingController)
    _analyze(main)

    assert UNEXPECTED_ERROR_MESSAGE in main.error_text
    assert "CUDA" not in main.error_text


def test_changing_a_photo_hides_the_previous_error(window):
    main = window(FailingController)
    _analyze(main)
    assert main.error_text

    main.load_files([MOVED])
    assert main.error_text == ""


def test_new_analysis_clears_the_photos_and_the_result(window):
    main = window()
    _analyze(main)
    main.new_analysis()

    assert main.step is Step.SELECTION
    assert main.selection_panel.before_slot.status is ImageStatus.EMPTY
    assert main.selection_panel.after_slot.status is ImageStatus.EMPTY
    assert main.report_widget.report is None
    assert not main.report_widget.can_export


def test_loading_photos_from_the_report_starts_a_new_analysis(window):
    main = window()
    _analyze(main)
    main.load_files([MOVED])

    assert main.step is Step.SELECTION
    assert main.selection_panel.before_slot.acquired.image.source.endswith("scene_after_moved.png")
    assert main.selection_panel.after_slot.acquired is None


def test_the_window_can_analyze_twice_with_the_same_models(window):
    main = window()
    _analyze(main)
    main.new_analysis()
    _analyze(main, BEFORE, BEFORE)

    assert main.step is Step.REPORT
    assert not main.report_widget.report.has_changes


# ----------------------------------------------------------------------- worker


def test_worker_reports_each_stage_then_the_report(qapp):
    controller = _controller()
    acquired = controller.acquisition.from_file(BEFORE), controller.acquisition.from_file(MOVED)
    worker = PipelineWorker(controller, *acquired)
    stages, reports, errors = [], [], []
    worker.progress_updated.connect(stages.append)
    worker.report_ready.connect(reports.append)
    worker.error_occurred.connect(errors.append)

    worker.run()  # Same thread: the connections are direct.

    assert tuple(stages) == STAGES
    assert len(reports) == 1 and isinstance(reports[0], ComparisonReport)
    assert errors == []


def test_worker_accepts_file_paths(qapp):
    worker = PipelineWorker(_controller(), BEFORE, MOVED)
    reports = []
    worker.report_ready.connect(reports.append)
    worker.run()
    assert reports[0].has_changes


def test_worker_turns_a_missing_file_into_a_message(qapp, tmp_path):
    worker = PipelineWorker(_controller(), BEFORE, tmp_path / "missing.png")
    errors = []
    worker.error_occurred.connect(errors.append)
    worker.run()
    assert errors and "No se encontró el archivo" in errors[0]


# ------------------------------------------------------------------ result view


def test_report_widget_shows_one_card_per_object(qapp):
    widget = ReportWidget()
    objects = [
        _changed(ChangeType.DISPLACED, 1),
        _changed(ChangeType.REMOVED, 2, "silla"),
        _changed(ChangeType.ADDED, 3, UNIDENTIFIED_LABEL),
    ]
    widget.show_report(_report(objects))

    assert widget.headline_text == "Se detectaron 3 cambios en la escena"
    displaced, removed, added = widget.card_texts
    assert "Objeto desplazado" in displaced and "Caja" in displaced and "Desplazamiento: 63.3 cm" in displaced
    assert "Objeto retirado" in removed and "Silla" in removed and " cm" not in removed
    assert "Objeto nuevo ingresado" in added and "Objeto no identificado" in added
    assert widget.timing_text == "Análisis completado en 0.15 s."


def test_first_analysis_reports_the_preparation_time_apart(qapp):
    widget = ReportWidget()
    report = _report()
    report.stage_seconds.update(loading=9.71)
    widget.show_report(report)

    assert widget.timing_text.startswith("Análisis completado en 0.15 s.")
    assert "preparación inicial del sistema tomó 9.7 s más" in widget.timing_text
    assert "solo ocurre en el primer análisis" in widget.timing_text
    assert widget.timing_details_text == (
        "Tiempo de cada etapa: distancias 0.10 s, búsqueda de cambios 0.05 s, preparación inicial 9.71 s."
    )


def test_image_captions_use_the_words_of_the_report():
    obj = _changed(ChangeType.DISPLACED)
    assert _caption(obj) == "#1 Caja · Objeto desplazado · 63.3 cm"
    assert _caption(_changed(ChangeType.REMOVED, 2, "silla")) == "#2 Silla · Objeto retirado"


def test_report_widget_shows_warnings_as_banners(qapp):
    widget = ReportWidget()
    warning = "Imagen 'después': La imagen está poco iluminada; el resultado podría no ser confiable."
    widget.show_report(_report([_changed(ChangeType.REMOVED)], [warning]))
    assert widget.warning_texts == [warning]

    widget.show_report(_report())
    assert widget.warning_texts == []
    assert widget.headline_text == report_text.NO_CHANGES_MESSAGE


def test_report_is_exported_as_json_and_png(qapp, tmp_path):
    widget = ReportWidget()
    widget.show_report(_report([_changed(ChangeType.DISPLACED)], ["Aviso de prueba."]))

    summary, image = widget.export_to(tmp_path / "resultado.json")

    assert image == tmp_path / "resultado.png" and image.stat().st_size > 0
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["hay_cambios"] is True
    assert data["cambios"][0]["tipo"] == "Objeto desplazado"
    assert data["cambios"][0]["categoria"] == "Caja"
    assert data["cambios"][0]["distancia_cm"] == 63.3
    assert data["advertencias"] == ["Aviso de prueba."]


def test_report_is_exported_as_text_by_default(qapp, tmp_path):
    widget = ReportWidget()
    widget.show_report(_report())

    summary, image = widget.export_to(tmp_path / "resultado")

    assert summary.suffix == ".txt" and image.suffix == ".png"
    assert "No se detectaron cambios en la escena." in summary.read_text(encoding="utf-8")


def test_only_the_annotated_image_is_saved_at_its_original_size(qapp, tmp_path):
    widget = ReportWidget()
    report = _report([_changed(ChangeType.DISPLACED)])
    report.annotated_image[10:20, 30:40] = (200, 40, 40)
    widget.show_report(report)
    widget.viewer.zoom_in()  # The on-screen zoom must not change the saved file.

    saved = widget.save_image_to(tmp_path / "resultado.png")

    assert saved == tmp_path / "resultado.png"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["resultado.png"]
    with Image.open(saved) as image:
        assert image.size == (64, 48)
        assert np.array_equal(np.asarray(image.convert("RGB")), report.annotated_image)


@pytest.mark.parametrize(("typed", "expected"), [("resultado", "resultado.png"), ("foto.jpg", "foto.png")])
def test_saved_image_is_always_png(qapp, tmp_path, typed, expected):
    widget = ReportWidget()
    widget.show_report(_report())
    assert widget.save_image_to(tmp_path / typed) == tmp_path / expected


def test_image_can_be_saved_only_when_there_is_a_result(qapp):
    widget = ReportWidget()
    assert not widget.can_save_image
    widget.show_report(_report())
    assert widget.can_save_image
    widget.clear()
    assert not widget.can_save_image


def test_save_image_menu_option_is_enabled_only_on_the_result(window):
    main = window()
    assert not main.can_save_image
    _analyze(main)
    assert main.can_save_image
    main.new_analysis()
    assert not main.can_save_image


def test_user_facing_texts_avoid_technical_terms():
    report = _report([_changed(kind) for kind in ChangeType])
    texts = [report_text.to_text(report), *report_text.STAGE_DESCRIPTIONS.values()]
    texts += [report_text.describe(obj) for obj in report.objects]
    for text in texts:
        assert not any(term in text.lower() for term in JARGON), text


def test_viewer_zooms_and_fits_the_image(qapp):
    viewer = ZoomableImageView()
    viewer.resize(400, 300)
    viewer.set_image(np.zeros((480, 640, 3), np.uint8))
    fitted = viewer.zoom

    viewer.zoom_in()
    assert viewer.zoom > fitted
    viewer.actual_size()
    assert viewer.zoom == pytest.approx(1.0)
    viewer.fit_to_window()
    assert viewer.zoom == pytest.approx(fitted)


# ------------------------------------------------------------ camera parameters


def test_camera_note_explains_how_distances_were_measured(qapp):
    from models import IntrinsicsSource

    report = _report([_changed(ChangeType.DISPLACED)], intrinsics_source=IntrinsicsSource.FIELD_OF_VIEW)
    widget = ReportWidget()
    widget.show_report(report)

    assert "estimación estándar de la cámara" in widget.camera_note_text
    assert "estimación estándar de la cámara" in report_text.to_text(report)
    camera = report_text.to_dict(report)["camara"]
    assert camera == {"origen_parametros": "estimada", "distancia_focal_px": 1556.6}
    assert not any(term in widget.camera_note_text.lower() for term in JARGON)

    widget.clear()
    assert widget.camera_note_text == ""


def test_camera_note_is_hidden_when_no_distance_was_measured(qapp):
    from models import IntrinsicsSource

    for objects in ((), [_changed(ChangeType.REMOVED)]):
        report = _report(objects, intrinsics_source=IntrinsicsSource.EXIF)
        assert report_text.camera_note(report) == ""
        widget = ReportWidget()
        widget.show_report(report)
        assert widget.camera_note_text == ""


@pytest.mark.parametrize("source, words", [("CALIBRATED", "calibración"), ("EXIF", "guardados en la foto")])
def test_camera_note_names_the_source(source, words):
    from models import IntrinsicsSource

    report = _report([_changed(ChangeType.DISPLACED)], intrinsics_source=IntrinsicsSource[source])
    assert words in report_text.camera_note(report)


# ------------------------------------------------------------- session history


def test_each_successful_analysis_enters_the_history_newest_first(window):
    main = window()
    _analyze(main)
    _analyze(main, after=BEFORE)

    entries = main.history.entries
    assert [entry.number for entry in entries] == [2, 1]
    assert (entries[1].before_name, entries[1].after_name) == ("scene_before.png", "scene_after_moved.png")
    assert (entries[0].before_name, entries[0].after_name) == ("scene_before.png", "scene_before.png")
    assert entries[1].report.has_changes and not entries[0].report.has_changes


def test_a_failed_analysis_does_not_enter_the_history(window):
    main = window(FailingController)
    _analyze(main)
    assert main.step is Step.SELECTION
    assert main.history.is_empty


def test_new_analysis_keeps_the_history(window):
    main = window()
    _analyze(main)
    main.new_analysis()
    assert main.step is Step.SELECTION
    assert len(main.history) == 1


def test_history_window_explains_when_there_is_nothing_yet(window):
    dialog = window().history_dialog()
    assert dialog.entry_titles == []
    assert dialog.shown_entry is None
    assert dialog.empty_message_text == report_text.HISTORY_EMPTY_MESSAGE


def test_history_window_lists_newest_first_and_shows_the_chosen_report(window):
    main = window()
    _analyze(main)
    _analyze(main, after=BEFORE)
    dialog = main.history_dialog()

    assert len(dialog.entry_titles) == 2
    assert dialog.entry_titles[0].startswith("#2 · ")
    assert dialog.entry_titles[1].endswith("scene_before.png → scene_after_moved.png")
    assert dialog.empty_message_text == ""

    assert dialog.shown_entry.number == 2
    assert dialog.report_widget.headline_text == report_text.NO_CHANGES_MESSAGE
    assert dialog.report_widget.card_texts == []

    dialog.select(1)
    first = main.history.entries[1]
    assert dialog.shown_entry is first
    assert dialog.report_widget.headline_text == report_text.headline(first.report)
    assert len(dialog.report_widget.card_texts) == len(first.report.objects) == 1
    assert len(main.history) == 2  # Browsing never changes the history.


def test_history_is_unavailable_only_while_analyzing(window):
    main = window()
    assert main.can_open_history

    main.load_files([BEFORE, MOVED])
    main.start_analysis()
    assert main.step is Step.ANALYZING
    assert not main.can_open_history

    _wait_for(lambda: main.step is not Step.ANALYZING)
    assert main.step is Step.REPORT
    assert main.can_open_history


def test_history_texts_avoid_technical_terms(window):
    main = window()
    _analyze(main)
    entry = main.history.latest
    texts = [report_text.HISTORY_EMPTY_MESSAGE, report_text.history_title(entry), report_text.history_summary(entry)]
    for text in texts:
        assert not any(term in text.lower() for term in JARGON), text


# ------------------------------------------------------------------ user manual


def _actions_by_text(main: MainWindow) -> dict:
    return {action.text(): action for action in main.findChildren(QAction)}


def test_manual_is_in_the_help_menu_and_the_toolbar_with_f1(window):
    main = window()
    manual = _actions_by_text(main)["Manual de usuario"]
    assert manual.shortcut().toString() == "F1"

    help_menu = next(m for m in main.menuBar().findChildren(QMenu) if m.title() == "A&yuda")
    assert [a.text() for a in help_menu.actions()] == ["Manual de usuario", "Acerca de"]
    toolbar_texts = [a.text() for a in main.findChildren(QToolBar)[0].actions() if a.text()]
    assert toolbar_texts[-2:] == ["Manual de usuario", "Acerca de"]


def test_manual_is_available_at_every_step(window):
    main = window()
    manual = _actions_by_text(main)["Manual de usuario"]
    assert manual.isEnabled()

    main.load_files([BEFORE, MOVED])
    main.start_analysis()
    assert main.step is Step.ANALYZING
    assert manual.isEnabled()

    _wait_for(lambda: main.step is not Step.ANALYZING)
    assert main.step is Step.REPORT
    assert manual.isEnabled()


def test_manual_dialog_builds_without_blocking(window):
    dialog = window().manual_dialog()
    assert len(dialog.section_titles) == 8
    assert dialog.current_section == "para-que-sirve"
    dialog.deleteLater()


def test_no_two_actions_share_a_shortcut(window):
    shortcuts = [
        action.shortcut().toString() for action in window().findChildren(QAction) if not action.shortcut().isEmpty()
    ]
    assert len(shortcuts) == len(set(shortcuts)), shortcuts


# --------------------------------------------------------------- example pairs


def _examples_menu(main: MainWindow) -> QMenu:
    file_menu = next(m for m in main.menuBar().findChildren(QMenu) if m.title() == "&Archivo")
    return next(a.menu() for a in file_menu.actions() if a.menu() is not None)


def test_examples_menu_lists_every_example_of_the_manifest(window):
    from services.capture.example_catalog import load_examples

    main = window()
    menu = _examples_menu(main)
    entries = load_examples().entries
    assert menu.title() == "Cargar ejemplo" and menu.isEnabled()
    assert [a.text() for a in menu.actions()] == [entry.title for entry in entries]
    assert [a.statusTip() for a in menu.actions()] == [entry.description for entry in entries]


def test_choosing_an_example_loads_both_panels_without_analyzing(window):
    from services.capture.example_catalog import load_examples

    main = window()
    first = load_examples().entries[0]
    _examples_menu(main).actions()[0].trigger()

    panel = main.selection_panel
    assert panel.before_slot.acquired.image.source == str(first.before_path)
    assert panel.after_slot.acquired.image.source == str(first.after_path)
    assert main.step is Step.SELECTION and main.can_analyze
    assert main.history.is_empty  # Nothing was analyzed.


def test_examples_menu_follows_the_loading_rules_during_an_analysis(window):
    main = window()
    menu = _examples_menu(main)
    main.load_files([BEFORE, MOVED])
    main.start_analysis()
    assert not menu.isEnabled()
    _wait_for(lambda: main.step is not Step.ANALYZING)
    assert menu.isEnabled()


def test_examples_menu_is_disabled_without_examples(window, monkeypatch):
    from services.capture.example_catalog import ExampleCatalog

    monkeypatch.setattr("ui.main_window.load_examples", lambda: ExampleCatalog(skipped=("sin manifiesto",)))
    main = window()
    menu = _examples_menu(main)
    assert menu.title() == "Cargar ejemplo (no hay ejemplos disponibles)"
    assert not menu.isEnabled() and menu.actions() == []
    main.new_analysis()  # Moving through the steps must not re-enable an empty menu.
    assert not menu.isEnabled()


# ------------------------------------------------------------ original photos


def _photo(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 120, np.uint8)


def test_result_shows_both_original_photos_as_thumbnails(qapp):
    from ui.report_widget import ORIGINAL_THUMBNAIL_HEIGHT

    widget = ReportWidget()
    widget.show_report(_report([_changed(ChangeType.DISPLACED)]), originals=(_photo(1920, 1080), _photo(480, 640)))

    assert widget.originals_visible
    assert widget.original_captions == ["Antes", "Después"]
    landscape, portrait = widget.original_sizes
    assert landscape.height() == ORIGINAL_THUMBNAIL_HEIGHT and portrait.height() == ORIGINAL_THUMBNAIL_HEIGHT
    assert landscape.width() == pytest.approx(ORIGINAL_THUMBNAIL_HEIGHT * 1920 / 1080, abs=1)
    assert portrait.width() == pytest.approx(ORIGINAL_THUMBNAIL_HEIGHT * 480 / 640, abs=1)

    widget.clear()
    assert not widget.originals_visible and widget.original_sizes == []


def test_small_photos_are_not_enlarged_and_no_strip_without_photos(qapp):
    widget = ReportWidget()
    widget.show_report(_report(), originals=(_photo(200, 100), _photo(200, 100)))
    assert [(s.width(), s.height()) for s in widget.original_sizes] == [(200, 100), (200, 100)]

    widget.show_report(_report())
    assert not widget.originals_visible


def test_window_result_shows_the_analyzed_photos_and_new_analysis_clears_them(window):
    main = window()
    _analyze(main)
    report = main.report_widget
    assert main.step is Step.REPORT
    assert report.originals_visible and report.original_captions == ["Antes", "Después"]
    assert [s.width() for s in report.original_sizes] == [213, 213]  # 640x480 photos at 160 px high

    main.new_analysis()
    assert not report.originals_visible


def test_failed_analysis_shows_no_original_photos(window):
    main = window(FailingController)
    _analyze(main)
    assert main.step is Step.SELECTION
    assert not main.report_widget.originals_visible


def test_history_keeps_reduced_originals_and_shows_them(window):
    from ui.main_window import HISTORY_THUMBNAIL_MAX_SIDE

    main = window()
    _analyze(main)
    entry = main.history.latest
    assert entry.originals is not None
    for pixels in entry.originals:
        assert max(pixels.shape[:2]) <= HISTORY_THUMBNAIL_MAX_SIDE
        assert pixels.shape[1] / pixels.shape[0] == pytest.approx(640 / 480, rel=0.01)

    dialog = main.history_dialog()
    assert dialog.report_widget.originals_visible
    assert dialog.report_widget.original_captions == ["Antes", "Después"]
    dialog.deleteLater()


def test_history_entry_without_originals_hides_the_strip(qapp):
    from models import SessionHistory
    from ui.history_dialog import HistoryDialog

    history = SessionHistory()
    history.add(_report(), "a.png", "b.png")
    dialog = HistoryDialog(history)
    assert not dialog.report_widget.originals_visible
    dialog.deleteLater()


def test_downscale_keeps_proportions_and_never_enlarges():
    from ui.qt_image import downscale

    reduced = downscale(_photo(1920, 1080), max_side=320)
    assert reduced.shape[:2] == (180, 320)
    small = _photo(100, 50)
    assert downscale(small, max_side=320) is small
    assert downscale(_photo(480, 640), max_height=160).shape[:2] == (160, 120)


# --- Small fixes (PROMPT 16) ----------------------------------------------------------------------


def test_new_photo_replaces_the_photo_rejected_as_not_indoor(window):
    from ui.image_status import ImageStatus

    main = window()
    main.load_files([BEFORE, MOVED])
    before, after = main.selection_panel.before_slot, main.selection_panel.after_slot
    # The analysis flags only "Antes" as an outdoor scene; the photo stays visible until replaced.
    main.selection_panel.mark_not_indoor({"Antes": 90.0})
    assert before.status is ImageStatus.NOT_INDOOR and before.acquired is not None

    main.load_files([LOW_LIGHT])

    assert before.acquired.image.source.endswith("scene_low_light.png")
    assert before.status is ImageStatus.LOW_LIGHT
    assert after.acquired.image.source.endswith("scene_after_moved.png")
    assert main.can_analyze


def test_report_export_error_belongs_to_the_domain_error_family():
    from models import SceneDiffError
    from ui.report_export import ReportExportError

    assert issubclass(ReportExportError, SceneDiffError)
