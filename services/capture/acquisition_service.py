"""Single entry point to obtain validated images from any source."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from config.capture_settings import CaptureSettings
from models.acquired_image import AcquiredImage
from services.capture.camera_capture import CameraCapture, VideoDeviceFactory
from services.capture.file_image_loader import FileImageLoader
from services.capture.image_source import ImageSource
from utils.image_validator import ImageValidator


class ImageAcquisitionService:
    """Acquires an image from a source and enforces the acceptance criteria.

    Raises :class:`models.exceptions.ImageError` subclasses when the image
    cannot be obtained or fails a blocking check; non-blocking issues are
    returned as warnings in the :class:`AcquiredImage` report.
    """

    def __init__(
        self,
        validator: ImageValidator | None = None,
        capture_settings: CaptureSettings | None = None,
        device_factory: VideoDeviceFactory | None = None,
    ) -> None:
        self._validator = validator or ImageValidator()
        self._capture_settings = capture_settings or CaptureSettings()
        self._device_factory = device_factory

    @property
    def validator(self) -> ImageValidator:
        return self._validator

    def acquire(self, source: ImageSource) -> AcquiredImage:
        image = source.acquire()
        report = self._validator.ensure_valid(image)
        return AcquiredImage(image=image, report=report)

    def from_file(self, path: Union[str, Path]) -> AcquiredImage:
        return self.acquire(FileImageLoader(path, self._validator))

    def from_camera(self, camera_index: Optional[int] = None) -> AcquiredImage:
        """Capture from the configured camera, or from ``camera_index`` if given."""
        settings = self._capture_settings
        if camera_index is not None:
            settings = settings.model_copy(update={"camera_index": camera_index})
        return self.acquire(CameraCapture(settings, self._device_factory))
