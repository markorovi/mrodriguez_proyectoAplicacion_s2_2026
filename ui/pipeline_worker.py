"""Runs an analysis on a background thread so the window stays responsive."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import QObject, QThread, Signal

from controllers.pipeline_controller import ImageInput, PipelineController
from models.exceptions import NonIndoorSceneError, SceneDiffError

ImageSource = Union[ImageInput, str, Path]

UNEXPECTED_ERROR_MESSAGE = (
    "Ocurrió un problema inesperado durante el análisis. "
    "Intente de nuevo; si el problema continúa, reinicie la aplicación."
)

_log = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Calls :meth:`PipelineController.analyze` (or ``analyze_files`` for two paths) in :meth:`run`.

    Besides the ``started`` and ``finished`` signals every ``QThread`` has,
    it emits ``progress_updated`` with the key of each stage as it starts
    (see ``controllers.STAGES``), then exactly one of ``report_ready`` (with
    the :class:`ComparisonReport`) or ``error_occurred`` (with a message for
    the user). When a photo is rejected because it is not an indoor scene,
    ``scene_rejected`` comes first, with the share of outdoor scenery of
    each rejected photo keyed by its name ("Antes", "Después"). Signals
    reach the window through queued connections, so their slots run on the
    interface thread.

    The controller keeps its models loaded between analyses and is not
    meant to run two analyses at once; start a new worker only after the
    previous one finished.
    """

    progress_updated = Signal(str)
    report_ready = Signal(object)
    scene_rejected = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        controller: PipelineController,
        before: ImageSource,
        after: ImageSource,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._before = before
        self._after = after

    def run(self) -> None:
        try:
            if _is_path(self._before) and _is_path(self._after):
                report = self._controller.analyze_files(self._before, self._after, self.progress_updated.emit)
            else:
                before, after = (self._loaded(source) for source in (self._before, self._after))
                report = self._controller.analyze(before, after, self.progress_updated.emit)
        except SceneDiffError as error:
            if isinstance(error, NonIndoorSceneError):
                self.scene_rejected.emit(dict(error.outdoor_percent))
            self.error_occurred.emit(str(error))
        except Exception:  # Never let a failure escape the thread silently.
            _log.exception("Unexpected failure while analyzing the images")
            self.error_occurred.emit(UNEXPECTED_ERROR_MESSAGE)
        else:
            self.report_ready.emit(report)

    def _loaded(self, source: ImageSource) -> ImageInput:
        return self._controller.acquisition.from_file(source) if _is_path(source) else source


def _is_path(source: ImageSource) -> bool:
    return isinstance(source, (str, Path))
