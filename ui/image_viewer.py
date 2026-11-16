"""Image view with zoom (mouse wheel or buttons) and fit-to-window."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

from ui import theme
from ui.qt_image import to_qpixmap

_ZOOM_STEP = 1.25
_MIN_ZOOM = 0.05
_MAX_ZOOM = 16.0


class ZoomableImageView(QGraphicsView):
    """Shows one image; the wheel zooms around the cursor and dragging pans.

    The image starts fitted to the view and stays fitted when the view is
    resized, until the user zooms.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(theme.IMAGE_BACKGROUND))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._fitted = True

    @property
    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def set_image(self, pixels: Optional[np.ndarray]) -> None:
        self._item.setPixmap(QPixmap() if pixels is None else to_qpixmap(pixels))
        self._scene.setSceneRect(self._item.boundingRect())
        self.fit_to_window()

    def fit_to_window(self) -> None:
        self._fitted = True
        if self.has_image:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self) -> None:
        self._fitted = False
        self.resetTransform()

    def zoom_in(self) -> None:
        self._scale_by(_ZOOM_STEP)

    def zoom_out(self) -> None:
        self._scale_by(1.0 / _ZOOM_STEP)

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            self._scale_by(_ZOOM_STEP**steps)
        event.accept()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self.fit_to_window()

    def _scale_by(self, factor: float) -> None:
        if not self.has_image:
            return
        target = min(_MAX_ZOOM, max(_MIN_ZOOM, self.zoom * factor))
        self._fitted = False
        self.scale(target / self.zoom, target / self.zoom)
