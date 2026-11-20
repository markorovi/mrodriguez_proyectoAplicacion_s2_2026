"""Result screen: the annotated image, the two original photos and a plain-language summary."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from models.comparison_report import ChangedObject, ComparisonReport
from ui import report_text, theme
from ui.banner import Banner, BannerKind
from ui.image_viewer import ZoomableImageView
from ui.qt_image import thumbnail_pixmap
from ui.report_export import ReportExportError, export_report, save_annotated_image

_TEXT_FILTER = "Resumen en texto (*.txt)"
_JSON_FILTER = "Resumen en JSON (*.json)"
_PNG_FILTER = "Imagen PNG (*.png)"

# Height of the original photos shown next to the result: small enough to leave the
# annotated image most of the space, large enough to recognize the scene.
ORIGINAL_THUMBNAIL_HEIGHT = 160
ORIGINAL_CAPTIONS = ("Antes", "Después")


class ReportWidget(QWidget):
    """Left: zoomable annotated image and the two original photos. Right: summary, cards, timings, export.

    The original photos are shown as small thumbnails ("Antes", "Después")
    under the annotated image, so the inputs, the highlighted change and the
    text are on screen at the same time. ``exported`` carries a confirmation
    message after the report or the image is saved.
    """

    exported = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._report: Optional[ComparisonReport] = None
        self._cards: list[QFrame] = []
        self._warnings: list[Banner] = []
        self._build()
        self.clear()

    @property
    def report(self) -> Optional[ComparisonReport]:
        return self._report

    @property
    def viewer(self) -> ZoomableImageView:
        return self._viewer

    @property
    def headline_text(self) -> str:
        return self._headline.text()

    @property
    def headline_kind(self) -> BannerKind:
        return self._headline.kind

    @property
    def warning_texts(self) -> list[str]:
        return [banner.text() for banner in self._warnings]

    @property
    def card_texts(self) -> list[str]:
        """Text of each object card, its lines joined by newlines."""
        return [
            "\n".join(label.text() for label in card.findChildren(QLabel) if label.text())
            for card in self._cards
        ]

    @property
    def timing_text(self) -> str:
        return self._timing.text()

    @property
    def camera_note_text(self) -> str:
        return self._camera_note.text() if not self._camera_note.isHidden() else ""

    @property
    def timing_details_text(self) -> str:
        """Per-stage breakdown, shown as the tooltip of the timing line."""
        return self._timing.toolTip()

    @property
    def can_export(self) -> bool:
        return self._export.isEnabled()

    @property
    def can_save_image(self) -> bool:
        return self._save_image.isEnabled()

    @property
    def originals_visible(self) -> bool:
        return not self._originals.isHidden()

    @property
    def original_captions(self) -> list[str]:
        return [caption.text() for caption in self._original_captions] if self.originals_visible else []

    @property
    def original_sizes(self) -> list[QSize]:
        """Size of each original thumbnail on screen (empty when none is shown)."""
        if not self.originals_visible:
            return []
        return [label.pixmap().size() for label in self._original_images]

    def show_report(
        self, report: ComparisonReport, originals: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """Show ``report`` and, when given, the two input photos ("before", "after") as thumbnails.

        Only reduced copies of ``originals`` are kept, never the full photos.
        """
        self.clear()
        self._report = report
        self._viewer.set_image(report.annotated_image)
        if originals is not None:
            for label, pixels in zip(self._original_images, originals):
                label.setPixmap(thumbnail_pixmap(pixels, ORIGINAL_THUMBNAIL_HEIGHT))
            self._originals.show()
        kind = BannerKind.INFO if report.has_changes else BannerKind.SUCCESS
        self._headline.show_message(report_text.headline(report), kind)
        self._headline.show()

        for warning in report.warnings:
            banner = Banner(warning, BannerKind.WARNING)
            self._warnings.append(banner)
            self._warnings_box.addWidget(banner)
        for obj in report.objects:
            card = _object_card(obj)
            self._cards.append(card)
            self._cards_box.addWidget(card)

        note = report_text.camera_note(report)
        self._camera_note.setText(note)
        self._camera_note.setVisible(bool(note))
        self._timing.setText(report_text.timing_summary(report))
        self._timing.setToolTip(f"Tiempo de cada etapa: {report_text.timing_details(report)}.")
        self._export.setEnabled(True)
        self._save_image.setEnabled(True)

    def clear(self) -> None:
        self._report = None
        for widget in (*self._warnings, *self._cards):
            widget.setParent(None)
            widget.deleteLater()
        self._warnings.clear()
        self._cards.clear()
        self._viewer.set_image(None)
        for label in self._original_images:
            label.clear()
        self._originals.hide()
        self._headline.hide()
        self._camera_note.clear()
        self._camera_note.hide()
        self._timing.clear()
        self._timing.setToolTip("")
        self._export.setEnabled(False)
        self._save_image.setEnabled(False)

    def export_to(self, summary_path: Union[str, Path]) -> list[Path]:
        """Save the summary (.txt or .json) and the annotated PNG next to it."""
        if self._report is None:
            raise ReportExportError("Todavía no hay un resultado para exportar.")
        return export_report(self._report, summary_path)

    def export_with_dialog(self) -> None:
        if self._report is None:
            return
        suggested = Path.home() / f"reporte_cambios_{datetime.now():%Y%m%d_%H%M}.txt"
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Exportar reporte", str(suggested), f"{_TEXT_FILTER};;{_JSON_FILTER}"
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() not in (".txt", ".json"):
            target = target.with_suffix(".json" if chosen_filter == _JSON_FILTER else ".txt")
        try:
            summary, image = self.export_to(target)
        except ReportExportError as error:
            QMessageBox.warning(self, "Exportar reporte", str(error))
            return
        self.exported.emit(f"Reporte guardado: {summary.name} y {image.name}, en {summary.parent}.")

    def save_image_to(self, path: Union[str, Path]) -> Path:
        """Save only the annotated image as PNG, at its original resolution (not the on-screen zoom)."""
        if self._report is None:
            raise ReportExportError("Todavía no hay un resultado para guardar.")
        return save_annotated_image(self._report, path)

    def save_image_with_dialog(self) -> None:
        if self._report is None:
            return
        suggested = Path.home() / f"resultado_cambios_{datetime.now():%Y%m%d_%H%M}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen", str(suggested), _PNG_FILTER)
        if not path:
            return
        try:
            saved = self.save_image_to(path)
        except ReportExportError as error:
            QMessageBox.warning(self, "Guardar imagen", str(error))
            return
        self.exported.emit(f"Imagen guardada: {saved.name}, en {saved.parent}.")

    def _build(self) -> None:
        self._viewer = ZoomableImageView()
        zoom_row = QHBoxLayout()
        for text, tip, slot in (
            ("Ajustar", "Ver la imagen completa", self._viewer.fit_to_window),
            ("100 %", "Ver la imagen a su tamaño real", self._viewer.actual_size),
            ("Acercar", "También puede usar la rueda del ratón", self._viewer.zoom_in),
            ("Alejar", "También puede usar la rueda del ratón", self._viewer.zoom_out),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            zoom_row.addWidget(button)
        zoom_row.addStretch()

        image_side = QWidget()
        image_layout = QVBoxLayout(image_side)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self._viewer, stretch=1)
        image_layout.addLayout(zoom_row)
        image_layout.addWidget(self._build_originals())

        self._headline = Banner()
        font = self._headline.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        font.setBold(True)
        self._headline.setFont(font)
        self._warnings_box = QVBoxLayout()
        self._cards_box = QVBoxLayout()
        self._cards_box.setSpacing(8)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self._headline)
        content_layout.addLayout(self._warnings_box)
        content_layout.addLayout(self._cards_box)
        content_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)

        self._camera_note = QLabel()
        self._camera_note.setWordWrap(True)
        self._camera_note.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self._timing = QLabel()
        self._timing.setWordWrap(True)
        self._timing.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self._save_image = QPushButton("Guardar imagen...")
        self._save_image.setToolTip("Guarda solo la imagen con los cambios marcados, en PNG y a su tamaño original.")
        self._save_image.clicked.connect(self.save_image_with_dialog)
        self._export = QPushButton("Exportar reporte...")
        self._export.setToolTip("Guarda la imagen con los cambios marcados y un resumen en texto o JSON.")
        self._export.clicked.connect(self.export_with_dialog)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self._save_image)
        buttons.addWidget(self._export)

        summary_side = QWidget()
        summary_layout = QVBoxLayout(summary_side)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(scroll, stretch=1)
        summary_layout.addWidget(self._camera_note)
        summary_layout.addWidget(self._timing)
        summary_layout.addLayout(buttons)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(image_side)
        splitter.addWidget(summary_side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)


    def _build_originals(self) -> QWidget:
        """Strip "Fotos originales" with the "before" and "after" thumbnails side by side."""
        self._originals = QWidget()
        title = QLabel("Fotos originales")
        title.setStyleSheet("font-weight:bold;")
        row = QHBoxLayout()
        self._original_images: list[QLabel] = []
        self._original_captions: list[QLabel] = []
        for caption_text in ORIGINAL_CAPTIONS:
            image = QLabel()
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setMinimumHeight(ORIGINAL_THUMBNAIL_HEIGHT)
            image.setStyleSheet(f"background:{theme.IMAGE_BACKGROUND}; border-radius:4px;")
            caption = QLabel(caption_text)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setStyleSheet(f"color:{theme.TEXT_MUTED};")
            column = QVBoxLayout()
            column.addWidget(image)
            column.addWidget(caption)
            row.addLayout(column)
            self._original_images.append(image)
            self._original_captions.append(caption)
        layout = QVBoxLayout(self._originals)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(title)
        layout.addLayout(row)
        self._originals.hide()
        return self._originals


def _object_card(obj: ChangedObject) -> QFrame:
    card = QFrame()
    card.setObjectName("objectCard")
    card.setStyleSheet(
        f"QFrame#objectCard {{ background:{theme.SURFACE}; border:1px solid {theme.BORDER};"
        " border-radius:4px; }"
        f" QLabel {{ color:{theme.TEXT}; background:transparent; }}"
    )
    title = QLabel(f"#{obj.object_id} · {obj.change_type.title}")
    title.setStyleSheet("font-weight:bold; font-size:11pt;")
    name = QLabel(f"Objeto: {report_text.category(obj)}")
    message = QLabel(report_text.describe(obj))
    message.setWordWrap(True)
    message.setStyleSheet(f"color:{theme.TEXT_MUTED};")

    layout = QVBoxLayout(card)
    layout.addWidget(title)
    layout.addWidget(name)
    distance = report_text.distance_text(obj)
    if distance:
        layout.addWidget(QLabel(distance))
    layout.addWidget(message)
    return card
