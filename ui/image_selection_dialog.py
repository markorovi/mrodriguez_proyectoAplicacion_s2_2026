"""Dialog to choose the "before" and "after" images of a comparison."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from config.settings import AppSettings
from models.acquired_image import AcquiredImage
from services.capture.acquisition_service import ImageAcquisitionService
from ui.image_selection_panel import ImageSelectionPanel
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
        return self._panel.before_slot

    @property
    def after_slot(self) -> ImageSlotWidget:
        return self._panel.after_slot

    @property
    def camera_index(self) -> int:
        return self._panel.camera_index

    @camera_index.setter
    def camera_index(self, index: int) -> None:
        self._panel.camera_index = index

    @property
    def can_continue(self) -> bool:
        return self._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()

    def selected_pair(self) -> Optional[tuple[AcquiredImage, AcquiredImage]]:
        return self._panel.selected_pair()

    def _build(self) -> None:
        self._panel = ImageSelectionPanel(self._service, self._settings.capture.camera_index)
        self._panel.changed.connect(self._refresh)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continuar")
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._panel)
        layout.addWidget(self._buttons)

    def _refresh(self) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._panel.is_ready)
