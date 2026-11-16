"""The "before" and "after" image panels with the camera selector."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

from config.image_settings import ImageSettings
from models.acquired_image import AcquiredImage
from services.capture.acquisition_service import ImageAcquisitionService
from ui.image_slot_widget import ImageSlotWidget


class ImageSelectionPanel(QWidget):
    """Lets the user load or capture the pair of images to compare.

    Shared by :class:`ui.image_selection_dialog.ImageSelectionDialog` and
    :class:`ui.main_window.MainWindow`. ``changed`` is emitted whenever either
    image changes; :attr:`is_ready` tells whether both are usable (valid, or
    valid with a lighting warning).
    """

    changed = Signal()

    def __init__(
        self,
        service: ImageAcquisitionService,
        camera_index: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._build(service.validator.settings, camera_index)

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
    def is_ready(self) -> bool:
        return self._before.status.is_usable and self._after.status.is_usable

    def selected_pair(self) -> Optional[tuple[AcquiredImage, AcquiredImage]]:
        before, after = self._before.acquired, self._after.acquired
        if before is None or after is None:
            return None
        return before, after

    def mark_not_indoor(self, outdoor_percent: dict[str, float]) -> None:
        """Flag the photos the analysis rejected as outdoor, keyed by panel title ("Antes", "Después")."""
        for slot in (self._before, self._after):
            if slot.title() in outdoor_percent:
                slot.mark_not_indoor(outdoor_percent[slot.title()])

    def clear(self) -> None:
        self._before.clear()
        self._after.clear()

    def _build(self, image_settings: ImageSettings, camera_index: int) -> None:
        instructions = QLabel(
            "Seleccione la imagen de referencia (Antes) y la imagen actual (Después). "
            "Tome ambas desde la misma posición y con iluminación similar. "
            f"Resolución mínima: {image_settings.min_width}x{image_settings.min_height}."
        )
        instructions.setWordWrap(True)

        self._camera = QSpinBox()
        self._camera.setRange(0, 9)
        self._camera.setValue(camera_index)
        self._camera.setToolTip("Número de la cámara conectada que se usará al pulsar \"Capturar\".")

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Cámara:"))
        camera_row.addWidget(self._camera)
        camera_row.addStretch()

        self._before = ImageSlotWidget("Antes", self._service, self._camera.value)
        self._after = ImageSlotWidget("Después", self._service, self._camera.value)
        self._before.changed.connect(self.changed)
        self._after.changed.connect(self.changed)

        slots = QHBoxLayout()
        slots.addWidget(self._before)
        slots.addWidget(self._after)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(instructions)
        layout.addLayout(camera_row)
        layout.addLayout(slots)
