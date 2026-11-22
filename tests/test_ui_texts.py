"""Every text the interface shows avoids technical terms (RNF-001).

The windows are built with the model doubles of ``test_main_window`` and
walked widget by widget in each state they can reach: empty, with photos,
with every image status, analyzing, with and without a result, and with an
empty or a filled history. Besides the widget properties, the test records
what the code writes while it runs: every ``QLabel.setText``, every status
bar message, and the titles and texts handed to message boxes and to the
file dialogs (which are replaced, never opened).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PySide6.QtCore import QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QFileDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QWidget,
)

from config import AppSettings
from controllers import STAGES
from models import ChangeType, NonIndoorSceneError, SessionHistory
from models.camera_model import IntrinsicsSource
from tests.doubles import FakeVideoDevice
from tests.jargon import jargon_found
from tests.test_main_window import (
    BEFORE,
    LOW_LIGHT,
    MOVED,
    CrashingController,
    FailingController,
    _actions_by_text,
    _analyze,
    _changed,
    _report,
    _wait_for,
    window,  # noqa: F401  (fixture)
)
from ui.history_dialog import HistoryDialog
from ui.image_status import ImageStatus
from ui import report_text
from ui.main_window import Step
from ui.manual_dialog import ManualDialog
from ui.report_widget import ReportWidget

_PROPERTIES = ("text", "toolTip", "statusTip", "whatsThis", "placeholderText", "windowTitle", "title")


class _Texts:
    """Collects every user-facing text with where it came from, and checks them for jargon."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, str, str]] = []  # (state, origin, property, text)
        self.open_files: list[str] = []
        self.save_path: str = ""

    def add(self, state: str, origin: str, prop: str, text: object) -> None:
        if isinstance(text, str) and text.strip():
            self.records.append((state, origin, prop, text))

    def crawl(self, root: QWidget, state: str) -> None:
        """Read the text properties of ``root`` and of every widget and action inside it."""
        for obj in (root, *root.findChildren(QObject)):
            if not isinstance(obj, (QWidget, QAction)):
                continue
            origin = f"{type(obj).__name__} '{obj.objectName()}'"
            for prop in _PROPERTIES:
                if prop == "title" and not isinstance(obj, (QMenu, QGroupBox)):
                    continue
                if prop == "text" and not isinstance(obj, (QLabel, QAbstractButton, QAction)):
                    continue
                getter = getattr(obj, prop, None)
                if callable(getter):
                    self.add(state, origin, prop, getter())
            if isinstance(obj, QTextEdit):
                self.add(state, origin, "toPlainText", obj.toPlainText())
            if isinstance(obj, QListWidget):
                for row in range(obj.count()):
                    item = obj.item(row)
                    self.add(state, origin, f"item[{row}].text", item.text())
                    self.add(state, origin, f"item[{row}].toolTip", item.toolTip())

    def assert_clean(self) -> None:
        assert self.records, "no text was collected"
        failures = [
            f"[{state}] {origin} {prop}: {text!r} -> {terms}"
            for state, origin, prop, text in self.records
            if (terms := jargon_found(text))
        ]
        assert not failures, "\n".join(failures)


@pytest.fixture
def texts(qapp, monkeypatch) -> _Texts:
    """A :class:`_Texts` that also records what the code writes while the test runs.

    Patched before any window is built, so the signal connections made in the
    constructors already point to the recorders.
    """
    collected = _Texts()

    set_label_text = QLabel.setText

    def record_label(label, text):
        collected.add("ejecución", f"{type(label).__name__} '{label.objectName()}'", "setText", text)
        set_label_text(label, text)

    show_message = QStatusBar.showMessage

    def record_status(bar, text, timeout=0):
        collected.add("ejecución", "QStatusBar", "showMessage", text)
        show_message(bar, text, timeout)

    def record_box(kind):
        def box(parent, title, text, *args, **kwargs):
            collected.add("ejecución", f"QMessageBox.{kind}", "title", title)
            collected.add("ejecución", f"QMessageBox.{kind}", "text", text)
            return QMessageBox.StandardButton.Ok

        return box

    def record_dialog(kind, answer):
        def dialog(parent, caption="", directory="", filter="", *args, **kwargs):
            collected.add("ejecución", f"QFileDialog.{kind}", "caption", caption)
            # The suggested file name is shown in the dialog; the folders above it are the user's own.
            collected.add("ejecución", f"QFileDialog.{kind}", "suggested name", Path(directory).name)
            for name in filter.split(";;"):
                collected.add("ejecución", f"QFileDialog.{kind}", "filter", name)
            return answer(), filter.split(";;")[0]

        return dialog

    monkeypatch.setattr(QLabel, "setText", record_label)
    monkeypatch.setattr(QStatusBar, "showMessage", record_status)
    for kind in ("warning", "about", "information", "critical"):
        monkeypatch.setattr(QMessageBox, kind, record_box(kind))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", record_dialog("getOpenFileNames", lambda: collected.open_files))
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", record_dialog("getOpenFileName", lambda: (collected.open_files or [""])[0])
    )
    monkeypatch.setattr(QFileDialog, "getSaveFileName", record_dialog("getSaveFileName", lambda: collected.save_path))
    return collected


def _camera(monkeypatch, frame: Optional[np.ndarray], opened: bool = True) -> None:
    """Make the next capture use a fake camera instead of real hardware."""
    monkeypatch.setattr(
        "services.capture.camera_capture.cv2.VideoCapture", lambda index: FakeVideoDevice(frame, opened=opened)
    )


class _RejectingController(FailingController):
    """The analysis rejects both photos as outdoor scenes, as the segmentation would."""

    error = NonIndoorSceneError({"Antes": 81.0, "Después": 64.0})


class _SlowController(FailingController):
    """Keeps the analysis running until ``release`` is set, so the window can be closed meanwhile."""

    release = threading.Event()

    def analyze(self, before, after, progress=None):
        self.release.wait(timeout=10)
        return super().analyze(before, after, progress)


# ------------------------------------------------------------------------ tests


def test_static_texts_of_every_window(texts, window):
    main = window()
    texts.crawl(main, "ventana inicial")
    texts.crawl(ReportWidget(), "resultado vacío")
    texts.crawl(HistoryDialog(SessionHistory()), "historial vacío")
    texts.crawl(ManualDialog(AppSettings()), "manual")
    texts.assert_clean()


def test_every_image_status_and_panel_state(texts, window, monkeypatch, tmp_path, write_image):
    main = window()
    panel = main.selection_panel
    seen = set()

    def visit(state: str) -> None:
        seen.update({panel.before_slot.status, panel.after_slot.status})
        texts.crawl(main, state)

    visit("paneles vacíos")
    panel.before_slot.load_file(BEFORE)
    visit("foto válida")
    panel.after_slot.load_file(LOW_LIGHT)
    visit("foto con poca luz (aviso)")
    panel.after_slot.load_file(write_image(np.zeros((240, 320, 3), np.uint8), "pequena.png"))
    visit("resolución insuficiente")
    panel.after_slot.load_file(write_image(np.zeros((480, 640), np.uint8), "gris.png"))
    visit("foto en blanco y negro")
    panel.after_slot.load_file(tmp_path / "foto.gif")
    visit("formato no compatible")
    panel.after_slot.load_file(tmp_path / "no_existe.png")
    visit("archivo inexistente")
    damaged = tmp_path / "danada.png"
    damaged.write_bytes(b"no es una imagen")
    panel.after_slot.load_file(damaged)
    visit("archivo dañado")
    _camera(monkeypatch, frame=None, opened=False)
    panel.after_slot.capture()
    visit("cámara no disponible")
    _camera(monkeypatch, frame=None)
    panel.after_slot.capture()
    visit("cámara sin imagen")
    _camera(monkeypatch, frame=np.full((720, 1280, 3), 120, np.uint8))
    panel.after_slot.capture()
    visit("foto capturada")

    rejected = window(_RejectingController)
    _analyze(rejected)
    seen.update({rejected.selection_panel.before_slot.status, rejected.selection_panel.after_slot.status})
    texts.crawl(rejected, "fotos rechazadas por no ser interiores")

    panel.clear()
    visit("paneles vaciados")
    assert seen == set(ImageStatus)
    texts.assert_clean()


def test_analysis_result_and_history_states(texts, window):
    main = window()
    texts.crawl(main.history_dialog(), "historial vacío desde la ventana")

    main.load_files([BEFORE, MOVED])
    main.start_analysis()
    texts.crawl(main, "analizando")  # Before any queued progress signal is processed.
    _wait_for(lambda: main.step is not Step.ANALYZING)
    assert main.step is Step.REPORT
    texts.crawl(main, "resultado con cambios")

    _analyze(main, BEFORE, LOW_LIGHT)
    texts.crawl(main, "resultado con avisos")
    _analyze(main, BEFORE, BEFORE)
    assert not main.report_widget.report.has_changes
    texts.crawl(main, "resultado sin cambios")

    history = main.history_dialog()
    for row in range(len(main.history)):
        history.select(row)
        texts.crawl(history, f"historial, entrada {row}")

    # Every kind of change, every way the distances were measured, with the original photos.
    widget = ReportWidget()
    photos = (np.full((48, 64, 3), 90, np.uint8), np.full((48, 64, 3), 120, np.uint8))
    for source in (*IntrinsicsSource, None):
        report = _report([_changed(kind, number) for number, kind in enumerate(ChangeType, 1)], (), source)
        widget.show_report(report, originals=photos)
        texts.crawl(widget, f"tarjetas de cada tipo de cambio, cámara {source}")
    widget.clear()
    texts.crawl(widget, "resultado vaciado")

    main.new_analysis()
    texts.crawl(main, "nuevo análisis")
    texts.assert_clean()


def test_status_bar_message_boxes_and_file_dialogs(texts, window, tmp_path):
    main = window()
    actions = _actions_by_text(main)

    texts.open_files = [str(BEFORE), str(MOVED)]
    actions["Cargar fotos..."].trigger()
    texts.open_files = [str(BEFORE)]
    before_buttons = main.selection_panel.before_slot.findChildren(QPushButton)
    next(button for button in before_buttons if button.text() == "Cargar archivo...").click()
    assert main.can_analyze

    main.start_analysis()
    _wait_for(lambda: main.step is not Step.ANALYZING)
    texts.save_path = str(tmp_path / "reporte.txt")
    actions["Exportar..."].trigger()
    texts.save_path = str(tmp_path / "imagen.png")
    actions["Guardar imagen..."].trigger()
    texts.save_path = str(tmp_path / "no_existe" / "reporte.txt")
    actions["Exportar..."].trigger()
    texts.save_path = str(tmp_path / "no_existe" / "imagen.png")
    actions["Guardar imagen..."].trigger()
    assert sum(1 for _, origin, _, _ in texts.records if origin == "QMessageBox.warning") == 4  # Title and text, twice.

    history = main.history_dialog()
    texts.save_path = str(tmp_path / "desde_historial.txt")
    history.report_widget.export_with_dialog()
    texts.crawl(history, "historial tras exportar")

    actions["Acerca de"].trigger()
    actions["Nuevo análisis"].trigger()

    for controller in (FailingController, CrashingController):
        failed = window(controller)
        _analyze(failed)
        texts.crawl(failed, f"análisis fallido ({controller.__name__})")

    closing = window(_SlowController)
    closing.load_files([BEFORE, MOVED])
    closing.start_analysis()
    threading.Timer(0.3, _SlowController.release.set).start()
    closing.close()  # Waits for the running analysis and says so in the status bar.

    statuses = [text for _, origin, _, text in texts.records if origin == "QStatusBar"]
    assert any("Terminando" in text for text in statuses)
    assert any(text.startswith("Reporte guardado") for text in statuses)
    texts.assert_clean()


def test_progress_label_shows_every_stage_description(texts, window):
    main = window()
    main.load_files([BEFORE, MOVED])
    main.start_analysis()
    _wait_for(lambda: main.step is not Step.ANALYZING)
    shown = {text for _, origin, prop, text in texts.records if prop == "setText"}
    # The first analysis loads the models, so every stage, "loading" included, is announced.
    assert {report_text.stage_description(stage) for stage in STAGES} <= shown
    texts.assert_clean()


def test_main_window_without_examples(texts, window, monkeypatch):
    from services.capture.example_catalog import ExampleCatalog

    monkeypatch.setattr("ui.main_window.load_examples", lambda: ExampleCatalog(skipped=("sin manifiesto",)))
    texts.crawl(window(), "sin ejemplos")
    texts.assert_clean()
