"""Acceptance checks for input images (format, channels, resolution, lighting)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from config.image_settings import ImageSettings
from models.exceptions import InvalidImageError, UnsupportedImageFormatError
from models.image_data import ImageData
from models.validation_report import IssueCode, ValidationReport
from utils.image_statistics import mean_luminance


class ImageValidator:
    """Verifies that an image is suitable for depth estimation.

    Blocking checks: 8-bit pixel depth, exactly three RGB channels and a
    minimum resolution. The resolution check is orientation independent, so
    a 480x640 portrait photo satisfies a 640x480 minimum. Non-blocking check:
    mean luminance below the configured threshold produces a warning.
    """

    def __init__(self, settings: ImageSettings | None = None) -> None:
        self._settings = settings or ImageSettings()

    @property
    def settings(self) -> ImageSettings:
        return self._settings

    def check_file_format(self, path: Union[str, Path]) -> None:
        """Raise :class:`UnsupportedImageFormatError` if the extension is not accepted."""
        suffix = Path(path).suffix.lower()
        if suffix not in self._settings.allowed_extensions:
            accepted = ", ".join(self._settings.allowed_extensions)
            raise UnsupportedImageFormatError(
                f"El formato '{suffix or 'sin extensión'}' no es compatible. "
                f"Formatos aceptados: {accepted}."
            )

    def validate(self, image: ImageData) -> ValidationReport:
        """Run every check and return the collected errors and warnings."""
        report = ValidationReport()
        pixels = image.pixels

        if pixels.dtype != np.uint8:
            report.add_error(
                IssueCode.BIT_DEPTH,
                "La imagen debe tener una profundidad de color de 8 bits por canal.",
            )

        if pixels.ndim != 3 or pixels.shape[2] != 3:
            report.add_error(IssueCode.CHANNELS, "La imagen debe ser a color (RGB, tres canales).")

        self._check_resolution(image, report)

        if report.is_valid:
            self._check_illumination(pixels, report)
        return report

    def ensure_valid(self, image: ImageData) -> ValidationReport:
        """Validate and raise :class:`InvalidImageError` when a blocking check fails."""
        report = self.validate(image)
        if not report.is_valid:
            raise InvalidImageError(report, image)
        return report

    def _check_resolution(self, image: ImageData, report: ValidationReport) -> None:
        min_long = max(self._settings.min_width, self._settings.min_height)
        min_short = min(self._settings.min_width, self._settings.min_height)
        long_side = max(image.width, image.height)
        short_side = min(image.width, image.height)
        if long_side < min_long or short_side < min_short:
            report.add_error(
                IssueCode.RESOLUTION,
                f"La resolución {image.width}x{image.height} es insuficiente; "
                f"se requiere al menos {self._settings.min_width}x{self._settings.min_height}."
            )

    def _check_illumination(self, pixels: np.ndarray, report: ValidationReport) -> None:
        if mean_luminance(pixels) < self._settings.low_light_threshold:
            report.add_warning(
                IssueCode.LOW_LIGHT,
                "La imagen está poco iluminada; el resultado podría no ser confiable."
            )
