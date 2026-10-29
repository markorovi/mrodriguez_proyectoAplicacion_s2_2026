"""Dialog to choose the "before" and "after" images of a comparison."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.settings import AppSettings
from models.acquired_image import AcquiredImage
from services.capture.acquisition_service import ImageAcquisitionService
from ui.image_slot_widget import ImageSlotWidget
from utils.image_validator import ImageValidator


class ImageSelectionDialog(QDialog):
    """Two image panels plus a camera selector.

    "Continuar" is enabled only when both images are usable (valid, or valid
    with a lighting warning). After the dialog is accepted, the pair is
    available through :meth:`selected_pair`.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        service: Optional[ImageAcquisitionService] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings or AppSettings()
        self._service = service or ImageAcquisitionService(
            ImageValidator(self._settings.image), self._settings.capture
        )
        self.setWindowTitle(f"{self._settings.app_name} - Selección de imágenes")
        self._build()
        self._refresh()

    @property
    def before_slot(self) -> ImageSlotWidget:
        return self._before

    @property
    def after_slot(self) -> ImageSlotWidget:
        return self._after

    @property
    def camera_index(self) -> int:
        return self._camera.value()

    @camera_index.setter
    def camera_index(self, index: int) -> None:
        self._camera.setValue(index)

    @property
    def can_continue(self) -> bool:
        return self._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()

    def selected_pair(self) -> Optional[tuple[AcquiredImage, AcquiredImage]]:
        before, after = self._before.acquired, self._after.acquired
        if before is None or after is None:
            return None
        return before, after

    def _build(self) -> None:
        instructions = QLabel(
            "Seleccione la imagen de referencia (Antes) y la imagen actual (Después). "
            "Tome ambas desde la misma posición y con iluminación similar. "
            f"Resolución mínima: {self._settings.image.min_width}x{self._settings.image.min_height}."
        )
        instructions.setWordWrap(True)

        self._camera = QSpinBox()
        self._camera.setRange(0, 9)
        self._camera.setValue(self._settings.capture.camera_index)
        self._camera.setToolTip("Número de la cámara conectada que se usará al pulsar \"Capturar\".")

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Cámara:"))
        camera_row.addWidget(self._camera)
        camera_row.addStretch()

        self._before = ImageSlotWidget("Antes", self._service, self._camera.value)
        self._after = ImageSlotWidget("Después", self._service, self._camera.value)
        self._before.changed.connect(self._refresh)
        self._after.changed.connect(self._refresh)

        slots = QHBoxLayout()
        slots.addWidget(self._before)
        slots.addWidget(self._after)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continuar")
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(instructions)
        layout.addLayout(camera_row)
        layout.addLayout(slots)
        layout.addWidget(self._buttons)

    def _refresh(self) -> None:
        ready = self._before.status.is_usable and self._after.status.is_usable
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ready)
