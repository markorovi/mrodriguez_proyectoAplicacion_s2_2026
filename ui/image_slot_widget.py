"""Panel to choose, preview and validate one image of the comparison pair."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.paths import SAMPLES_DIR
from models.acquired_image import AcquiredImage
from models.exceptions import InvalidImageError, SceneDiffError
from models.image_data import ImageData
from services.capture.acquisition_service import ImageAcquisitionService
from ui import theme
from ui.image_status import ImageStatus, Severity, status_for_error, status_for_report
from ui.qt_image import to_qpixmap

_BADGE_COLORS = {
    Severity.NONE: theme.SURFACE_RAISED,
    Severity.OK: "#3d5a44",
    Severity.WARNING: "#5e5131",
    Severity.ERROR: "#633a36",
}


class ImageSlotWidget(QGroupBox):
    """Loads an image from disk or the camera through the acquisition service.

    Validation runs as part of every acquisition. The panel shows a
    thumbnail (also for rejected images, when they could be decoded), a
    status badge and the user-facing messages. ``changed`` is emitted after
    every attempt, successful or not.
    """

    changed = Signal()

    THUMBNAIL_SIZE = QSize(320, 240)

    def __init__(
        self,
        title: str,
        service: ImageAcquisitionService,
        camera_index: Callable[[], int],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title, parent)
        self._title = title
        self._service = service
        self._camera_index = camera_index
        self._acquired: Optional[AcquiredImage] = None
        self._status = ImageStatus.EMPTY
        self._build()
        self._render_empty()

    @property
    def acquired(self) -> Optional[AcquiredImage]:
        """The accepted image, or None if nothing usable has been loaded."""
        return self._acquired

    @property
    def status(self) -> ImageStatus:
        return self._status

    @property
    def status_text(self) -> str:
        return self._badge.text()

    @property
    def detail_text(self) -> str:
        return self._detail.text()

    @property
    def has_preview(self) -> bool:
        pixmap = self._thumbnail.pixmap()
        return pixmap is not None and not pixmap.isNull()

    def load_file(self, path: Union[str, Path]) -> None:
        self._attempt(lambda: self._service.from_file(path))

    def capture(self) -> None:
        self._attempt(lambda: self._service.from_camera(self._camera_index()))

    def mark_not_indoor(self, outdoor_percent: float) -> None:
        """Flag the loaded image as rejected by the analysis because it shows an outdoor scene.

        The image stays loaded and visible, but is no longer usable until it is replaced.
        """
        if self._acquired is None:
            return
        self._set_status(ImageStatus.NOT_INDOOR)
        self._detail.setText(
            f"{self._describe(self._acquired.image)}\n"
            f"No parece un espacio interior: cerca del {outdoor_percent:.0f} % de la foto muestra "
            "elementos de exterior (cielo, árboles, edificios...). Reemplácela por una foto "
            "tomada dentro de una habitación o área de trabajo."
        )
        self.changed.emit()

    def clear(self) -> None:
        """Forget the current image and return to the empty state."""
        self._acquired = None
        self._thumbnail.clear()
        self._render_empty()
        self.changed.emit()

    def _build(self) -> None:
        self._thumbnail = QLabel()
        self._thumbnail.setFixedSize(self.THUMBNAIL_SIZE)
        self._thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail.setStyleSheet(
            f"background:{theme.IMAGE_BACKGROUND}; color:{theme.TEXT_MUTED}; border-radius:4px;"
        )

        self._badge = QLabel()
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setMinimumHeight(48)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        browse = QPushButton("Cargar archivo...")
        browse.clicked.connect(self._browse)
        capture = QPushButton("Capturar")
        capture.clicked.connect(self.capture)

        buttons = QHBoxLayout()
        buttons.addWidget(browse)
        buttons.addWidget(capture)

        layout = QVBoxLayout(self)
        layout.addWidget(self._thumbnail, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._badge)
        layout.addWidget(self._detail)
        layout.addLayout(buttons)

    def _browse(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in self._service.validator.settings.allowed_extensions)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Seleccionar imagen: {self._title}",
            str(SAMPLES_DIR),
            f"Imágenes ({extensions})",
        )
        if path:
            self.load_file(path)

    def _attempt(self, acquire: Callable[[], AcquiredImage]) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            acquired = acquire()
        except SceneDiffError as error:
            self._render_failure(error)
        else:
            self._render_success(acquired)
        finally:
            QApplication.restoreOverrideCursor()
        self.changed.emit()

    def _render_success(self, acquired: AcquiredImage) -> None:
        self._acquired = acquired
        self._set_status(status_for_report(acquired.report))
        self._set_preview(acquired.image)
        self._detail.setText(
            "\n".join([self._describe(acquired.image), _checked_on_load(self._status), *acquired.warnings])
        )

    def _render_failure(self, error: SceneDiffError) -> None:
        self._acquired = None
        self._set_status(status_for_error(error))
        rejected = error.image if isinstance(error, InvalidImageError) else None
        if rejected is not None:
            self._set_preview(rejected)
            self._detail.setText(f"{self._describe(rejected)}\n{error}")
        else:
            self._thumbnail.clear()
            self._thumbnail.setText("Sin vista previa")
            self._detail.setText(str(error))

    def _render_empty(self) -> None:
        self._set_status(ImageStatus.EMPTY)
        self._thumbnail.setText("Sin imagen")
        self._detail.setText("Cargue un archivo o capture desde la cámara.")

    def _set_preview(self, image: ImageData) -> None:
        pixmap = to_qpixmap(image.pixels).scaled(
            self.THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail.setPixmap(pixmap)

    def _set_status(self, status: ImageStatus) -> None:
        self._status = status
        self._badge.setText(status.label)
        self._badge.setStyleSheet(
            f"background:{_BADGE_COLORS[status.severity]}; color:{theme.TEXT}; font-weight:bold;"
            "padding:4px 8px; border-radius:4px;"
        )

    @staticmethod
    def _describe(image: ImageData) -> str:
        name = Path(image.source).name if not image.source.startswith("camera:") else image.source
        return f"{name} | {image.width}x{image.height}"


def _checked_on_load(status: ImageStatus) -> str:
    """What an accepted image passed, and what is only checked later by the analysis."""
    checked = "Formato, color y resolución correctos" if status is ImageStatus.LOW_LIGHT else (
        "Formato, color, resolución e iluminación correctos"
    )
    return f"{checked}. Que sea un espacio interior se comprueba al analizar."
