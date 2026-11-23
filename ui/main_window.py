"""Main application window: choose two photos, analyze them and read the result."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config.paths import SAMPLES_DIR
from config.settings import AppSettings
from controllers.pipeline_controller import STAGES, PipelineController
from models.comparison_report import ComparisonReport
from models.image_data import ImageData
from models.session_history import SessionHistory
from services.capture.example_catalog import ExampleEntry, load_examples
from ui import report_text, theme
from ui.banner import Banner, BannerKind
from ui.history_dialog import HistoryDialog
from ui.manual_dialog import ManualDialog
from ui.image_selection_panel import ImageSelectionPanel
from ui.image_slot_widget import ImageSlotWidget
from ui.pipeline_worker import PipelineWorker
from ui.qt_image import downscale
from ui.report_widget import ReportWidget

APP_TITLE = "Detección de cambios en escena"

# Longest side of the copies of the photos kept in the session history (about 0.35 MB per pair at 16:9).
HISTORY_THUMBNAIL_MAX_SIDE = 320

ABOUT_TEXT = (
    "<h3>Detección de cambios en escena</h3>"
    "<p>Compara dos fotografías del mismo espacio interior, tomadas en momentos distintos, "
    "e indica qué objetos se retiraron, cuáles se movieron (y cuántos centímetros) "
    "y cuáles aparecieron.</p>"
    "<p>Para mejores resultados, tome ambas fotos desde la misma posición y con una "
    "iluminación parecida.</p>"
    "<p>Las fotos se procesan en este equipo y no se envían a internet.</p>"
)


class Step(Enum):
    """The three steps of an analysis, one page each."""

    SELECTION = 0
    ANALYZING = 1
    REPORT = 2


class MainWindow(QMainWindow):
    """Guides the user through selection, analysis and result in three interactions.

    The analysis runs on a :class:`PipelineWorker` thread; the window only
    talks to the :class:`PipelineController` (through the worker and its
    acquisition service) and to the domain models it returns.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        controller: Optional[PipelineController] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller or PipelineController(settings)
        self._settings = self._controller.settings
        self._worker: Optional[PipelineWorker] = None
        self._step = Step.SELECTION
        # One history per window: it lives as long as the session and is never saved (RF-014).
        self._history = SessionHistory()
        self._analyzed_sources: tuple[str, str] = ("", "")
        # The photos being analyzed, by reference: the panels already hold them, so nothing is copied.
        self._analyzed_images: Optional[tuple[ImageData, ImageData]] = None

        self.setWindowTitle(APP_TITLE)
        self._fit_to_screen()
        self._build_actions()
        self._build_pages()
        self._set_step(Step.SELECTION)
        self.statusBar().showMessage("Elija la foto de referencia (Antes) y la foto actual (Después).")

    # ----------------------------------------------------------------- state

    @property
    def step(self) -> Step:
        return self._step

    @property
    def controller(self) -> PipelineController:
        return self._controller

    @property
    def selection_panel(self) -> ImageSelectionPanel:
        return self._panel

    @property
    def report_widget(self) -> ReportWidget:
        return self._report

    @property
    def history(self) -> SessionHistory:
        return self._history

    @property
    def can_open_history(self) -> bool:
        return self._history_action.isEnabled()

    @property
    def can_analyze(self) -> bool:
        return self._analyze.isEnabled()

    @property
    def is_analyzing(self) -> bool:
        return self._step is Step.ANALYZING

    @property
    def error_text(self) -> str:
        return self._error.text() if not self._error.isHidden() else ""

    @property
    def selection_warning_text(self) -> str:
        return self._selection_warning.text() if not self._selection_warning.isHidden() else ""

    @property
    def progress_text(self) -> str:
        return self._stage.text()

    @property
    def progress_value(self) -> int:
        return self._progress.value()

    # --------------------------------------------------------------- actions

    def start_analysis(self) -> None:
        """Step 2: analyze the selected pair on a background thread."""
        pair = self._panel.selected_pair()
        if pair is None or self.is_analyzing:
            return
        if self._worker is not None:
            self._worker.wait()  # The previous run is only finishing its thread.

        self._error.hide()
        self._progress.setValue(0)
        self._stage.setText(report_text.stage_description(STAGES[0]))
        self._analyzed_sources = (pair[0].image.source, pair[1].image.source)
        self._analyzed_images = (pair[0].image, pair[1].image)
        worker = PipelineWorker(self._controller, *pair, parent=self)
        worker.progress_updated.connect(self._on_progress)
        worker.report_ready.connect(self._on_report)
        worker.scene_rejected.connect(self._panel.mark_not_indoor)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(lambda: self._on_worker_finished(worker))
        self._worker = worker
        self._set_step(Step.ANALYZING)
        self.statusBar().showMessage("Analizando las fotos...")
        worker.start()

    def new_analysis(self) -> None:
        """Discard the photos and the result and go back to step 1."""
        if self.is_analyzing:
            return
        self._panel.clear()
        self._report.clear()
        self._analyzed_images = None
        self._error.hide()
        self._set_step(Step.SELECTION)
        self.statusBar().showMessage("Elija la foto de referencia (Antes) y la foto actual (Después).")

    def load_files(self, paths: Sequence[Union[str, Path]]) -> None:
        """Load one photo into the next empty panel, or two photos as "before" and "after"."""
        if self.is_analyzing or not paths:
            return
        self._back_to_selection()
        if len(paths) >= 2:
            self._panel.before_slot.load_file(paths[0])
            self._panel.after_slot.load_file(paths[1])
        else:
            self._next_slot().load_file(paths[0])

    def load_example(self, entry: ExampleEntry) -> None:
        """Load an example pair as "before" and "after", exactly like choosing both files by hand."""
        self.load_files([entry.before_path, entry.after_path])

    def capture_photo(self) -> None:
        """Capture with the selected camera into the next empty panel."""
        if self.is_analyzing:
            return
        self._back_to_selection()
        self._next_slot().capture()

    def export_report(self) -> None:
        if self._step is Step.REPORT:
            self._report.export_with_dialog()

    def save_image(self) -> None:
        if self._step is Step.REPORT:
            self._report.save_image_with_dialog()

    def history_dialog(self) -> HistoryDialog:
        """Build the session history window without showing it."""
        return HistoryDialog(self._history, parent=self)

    def open_history(self) -> None:
        """Show the comparisons of this session (CU-03); not available while analyzing."""
        if self.is_analyzing:
            return
        self.history_dialog().exec()

    def manual_dialog(self) -> ManualDialog:
        """Build the user manual for the settings this window runs with, without showing it."""
        return ManualDialog(self._settings, parent=self)

    def open_manual(self) -> None:
        """Show the user manual; available at any step, also while an analysis runs."""
        self.manual_dialog().exec()

    @property
    def can_save_image(self) -> bool:
        return self._save_image_action.isEnabled()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Terminando el análisis en curso antes de cerrar...")
            self._worker.wait()
        self._controller.close()
        super().closeEvent(event)

    # ------------------------------------------------------- worker signals

    def _on_progress(self, stage: str) -> None:
        if stage in STAGES:
            self._progress.setValue(STAGES.index(stage))
        self._stage.setText(report_text.stage_description(stage))

    def _on_report(self, report: ComparisonReport) -> None:
        self._progress.setValue(len(STAGES))
        originals = None
        if self._analyzed_images is not None:
            originals = (self._analyzed_images[0].pixels, self._analyzed_images[1].pixels)
        # Only successful comparisons enter the history; a failed one has no report to review.
        # The history keeps reduced copies of the photos, not the full-size ones.
        history_originals = None if originals is None else tuple(
            downscale(pixels, max_side=HISTORY_THUMBNAIL_MAX_SIDE) for pixels in originals
        )
        self._history.add(report, *self._analyzed_sources, originals=history_originals)
        self._report.show_report(report, originals=originals)
        self._set_step(Step.REPORT)
        self.statusBar().showMessage(report_text.headline(report))

    def _on_error(self, message: str) -> None:
        self._error.show_message(f"No se pudo completar el análisis. {message}", BannerKind.ERROR)
        self._error.show()
        # A failed analysis has no result: nothing of it, the original photos included, stays on screen.
        self._report.clear()
        self._analyzed_images = None
        self._set_step(Step.SELECTION)
        self.statusBar().showMessage("El análisis no se completó.")

    def _on_worker_finished(self, worker: PipelineWorker) -> None:
        # A late signal from a previous run must not drop the current worker.
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    # --------------------------------------------------------------- helpers

    def _set_step(self, step: Step) -> None:
        self._step = step
        self._pages.setCurrentIndex(step.value)
        busy = step is Step.ANALYZING
        for action in (self._new_action, self._load_action, self._capture_action):
            action.setEnabled(not busy)
        # Loading an example is one more way of loading photos, so it follows the same rule.
        self._examples_menu.setEnabled(bool(self._example_actions) and not busy)
        self._export_action.setEnabled(step is Step.REPORT)
        self._save_image_action.setEnabled(step is Step.REPORT)
        # Enabled even with an empty history, so the user is told there is nothing yet.
        self._history_action.setEnabled(not busy)
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        self._analyze.setEnabled(self._panel.is_ready and not self.is_analyzing)
        warnings = [
            f"{slot.title()}: {warning}"
            for slot in (self._panel.before_slot, self._panel.after_slot)
            if slot.acquired is not None
            for warning in slot.acquired.warnings
        ]
        self._selection_warning.setText("\n".join(warnings))
        self._selection_warning.setVisible(bool(warnings))

    def _on_selection_changed(self) -> None:
        self._error.hide()
        self._refresh_selection()

    def _back_to_selection(self) -> None:
        if self._step is Step.REPORT:
            self.new_analysis()

    def _next_slot(self) -> ImageSlotWidget:
        """The first panel without a usable photo (empty, or rejected as not indoor); else "after".

        A photo rejected by the analysis stays visible but unusable, so a new photo replaces it.
        """
        for slot in (self._panel.before_slot, self._panel.after_slot):
            if slot.acquired is None or not slot.status.is_usable:
                return slot
        return self._panel.after_slot

    def _choose_files(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in self._settings.image.allowed_extensions)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Cargar fotos (elija una, o dos: primero la de Antes y luego la de Después)",
            str(SAMPLES_DIR),
            f"Imágenes ({extensions})",
        )
        self.load_files(paths)

    def _fit_to_screen(self) -> None:
        """1280 x 820 when it fits; otherwise 90 % of the available screen."""
        width, height = 1280, 820
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.9))
            height = min(height, int(available.height() * 0.9))
        self.resize(width, height)

    def _about(self) -> None:
        QMessageBox.about(self, f"Acerca de {APP_TITLE}", ABOUT_TEXT)

    # ---------------------------------------------------------------- layout

    def _build_actions(self) -> None:
        self._new_action = QAction("Nuevo análisis", self)
        self._new_action.setShortcut(QKeySequence.StandardKey.New)
        self._new_action.setStatusTip("Descartar las fotos y el resultado y empezar de nuevo.")
        self._new_action.triggered.connect(self.new_analysis)

        self._load_action = QAction("Cargar fotos...", self)
        self._load_action.setShortcut(QKeySequence.StandardKey.Open)
        self._load_action.setStatusTip("Abrir una o dos fotos guardadas en el equipo.")
        self._load_action.triggered.connect(self._choose_files)

        self._capture_action = QAction("Capturar de cámara", self)
        self._capture_action.setStatusTip("Tomar una foto con la cámara elegida.")
        self._capture_action.triggered.connect(self.capture_photo)

        self._save_image_action = QAction("Guardar imagen...", self)
        self._save_image_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_image_action.setStatusTip("Guardar solo la imagen con los cambios marcados, en PNG.")
        self._save_image_action.triggered.connect(self.save_image)

        self._export_action = QAction("Exportar...", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+E"))
        self._export_action.setStatusTip("Guardar la imagen del resultado y un resumen.")
        self._export_action.triggered.connect(self.export_report)

        self._history_action = QAction("Historial...", self)
        self._history_action.setShortcut(QKeySequence("Ctrl+H"))
        self._history_action.setStatusTip("Ver los resultados de las comparaciones de esta sesión.")
        self._history_action.triggered.connect(self.open_history)

        # Not part of _set_step: the manual stays available at every step.
        self._manual_action = QAction("Manual de usuario", self)
        self._manual_action.setShortcut(QKeySequence("F1"))
        self._manual_action.setStatusTip("Abrir el manual de usuario.")
        self._manual_action.triggered.connect(self.open_manual)

        about_action = QAction("Acerca de", self)
        about_action.triggered.connect(self._about)
        quit_action = QAction("Salir", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&Archivo")
        file_menu.addAction(self._new_action)
        file_menu.addAction(self._load_action)
        self._build_examples_menu(file_menu)
        for action in (
            self._capture_action,
            self._save_image_action,
            self._export_action,
            self._history_action,
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        help_menu = self.menuBar().addMenu("A&yuda")
        help_menu.addAction(self._manual_action)
        help_menu.addAction(about_action)

        toolbar = QToolBar("Acciones principales")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for action in (
            self._new_action,
            self._load_action,
            self._capture_action,
            self._export_action,
            self._history_action,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self._manual_action)
        toolbar.addAction(about_action)
        self.addToolBar(toolbar)

    def _build_examples_menu(self, file_menu: QMenu) -> None:
        """Submenu with one action per example pair of the manifest, in its order."""
        catalog = load_examples()
        self._examples_menu = file_menu.addMenu("Cargar ejemplo")
        self._example_actions: list[QAction] = []
        for entry in catalog.entries:
            action = QAction(entry.title, self)
            action.setToolTip(entry.description)
            action.setStatusTip(entry.description)
            action.triggered.connect(lambda _checked=False, entry=entry: self.load_example(entry))
            self._examples_menu.addAction(action)
            self._example_actions.append(action)
        if not self._example_actions:
            self._examples_menu.setTitle("Cargar ejemplo (no hay ejemplos disponibles)")
            self._examples_menu.setEnabled(False)

    def _build_pages(self) -> None:
        self._pages = QStackedWidget()
        self._pages.addWidget(self._selection_page())
        self._pages.addWidget(self._analysis_page())
        self._pages.addWidget(self._report_page())
        self.setCentralWidget(self._pages)

    def _selection_page(self) -> QWidget:
        self._panel = ImageSelectionPanel(self._controller.acquisition, self._settings.capture.camera_index)
        self._panel.changed.connect(self._on_selection_changed)
        self._error = Banner(kind=BannerKind.ERROR)
        self._error.hide()
        self._selection_warning = Banner(kind=BannerKind.WARNING)
        self._selection_warning.hide()

        self._analyze = QPushButton("Analizar diferencias")
        self._analyze.setMinimumHeight(40)
        self._analyze.setStyleSheet("font-weight:bold; padding:0 24px;")
        self._analyze.setToolTip("Disponible cuando las dos fotos son válidas.")
        self._analyze.clicked.connect(self.start_analysis)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._analyze)

        page, layout = _page("Paso 1 de 3 · Elija las dos fotos")
        layout.addWidget(self._error)
        layout.addWidget(self._panel)
        layout.addWidget(self._selection_warning)
        layout.addStretch()
        layout.addLayout(actions)
        return page

    def _analysis_page(self) -> QWidget:
        self._progress = QProgressBar()
        self._progress.setRange(0, len(STAGES))
        self._progress.setTextVisible(False)
        self._progress.setMinimumWidth(420)
        self._stage = QLabel()
        self._stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self._stage.font()
        font.setPointSizeF(font.pointSizeF() * 1.2)
        self._stage.setFont(font)
        hint = QLabel("La primera vez puede tardar unos segundos más mientras el sistema se prepara.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color:{theme.TEXT_MUTED};")

        page, layout = _page("Paso 2 de 3 · Analizando las diferencias")
        layout.addStretch()
        layout.addWidget(self._stage)
        layout.addWidget(self._progress, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(hint)
        layout.addStretch()
        return page

    def _report_page(self) -> QWidget:
        self._report = ReportWidget()
        self._report.exported.connect(self.statusBar().showMessage)
        again = QPushButton("Nuevo análisis")
        again.clicked.connect(self.new_analysis)
        actions = QHBoxLayout()
        actions.addWidget(again)
        actions.addStretch()

        page, layout = _page("Paso 3 de 3 · Resultado")
        layout.addWidget(self._report, stretch=1)
        layout.addLayout(actions)
        return page


def _page(title: str) -> tuple[QWidget, QVBoxLayout]:
    page = QWidget()
    layout = QVBoxLayout(page)
    heading = QLabel(title)
    font = heading.font()
    font.setPointSizeF(font.pointSizeF() * 1.4)
    font.setBold(True)
    heading.setFont(font)
    layout.addWidget(heading)
    return page, layout
