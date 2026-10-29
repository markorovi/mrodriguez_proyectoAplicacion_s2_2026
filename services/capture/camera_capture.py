"""Captures still frames from a conventional RGB camera (webcam or phone)."""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from config.capture_settings import CaptureSettings
from models.exceptions import CameraError
from models.image_data import ImageData
from services.capture.image_source import ImageSource


class VideoDevice(Protocol):
    """Subset of ``cv2.VideoCapture`` used by :class:`CameraCapture`."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def release(self) -> None: ...


VideoDeviceFactory = Callable[[int], VideoDevice]


class CameraCapture(ImageSource):
    """Opens the camera, lets exposure settle, grabs one frame and releases it.

    The device is held only for the duration of :meth:`acquire`, so other
    applications can use the camera between the "before" and "after" shots.
    ``device_factory`` allows replacing OpenCV with a test double.
    """

    def __init__(
        self,
        settings: CaptureSettings | None = None,
        device_factory: VideoDeviceFactory | None = None,
    ) -> None:
        self._settings = settings or CaptureSettings()
        self._device_factory = device_factory or cv2.VideoCapture

    @property
    def description(self) -> str:
        return f"camera:{self._settings.camera_index}"

    def acquire(self) -> ImageData:
        device = self._open()
        try:
            for _ in range(self._settings.warmup_frames):
                device.read()
            ok, frame = device.read()
        finally:
            device.release()

        if not ok or frame is None:
            raise CameraError("La cámara no entregó ninguna imagen. Intente de nuevo.")
        return ImageData(pixels=self._to_rgb(frame), source=self.description)

    def _open(self) -> VideoDevice:
        device = self._device_factory(self._settings.camera_index)
        if not device.isOpened():
            device.release()
            raise CameraError(
                f"No se pudo acceder a la cámara {self._settings.camera_index}. "
                "Verifique que esté conectada y que ninguna otra aplicación la esté usando."
            )
        device.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
        device.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
        return device

    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
