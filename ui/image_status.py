"""User-facing status of a selected image, derived from validation results."""

from __future__ import annotations

from enum import Enum

from models.exceptions import (
    CameraError,
    InvalidImageError,
    SceneDiffError,
    UnsupportedImageFormatError,
)
from models.validation_report import IssueCode, ValidationReport


class Severity(str, Enum):
    NONE = "none"
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class ImageStatus(Enum):
    """Badge shown next to each image preview."""

    EMPTY = ("SIN IMAGEN", Severity.NONE)
    VALID = ("VÁLIDA", Severity.OK)
    LOW_LIGHT = ("ADVERTENCIA LUZ", Severity.WARNING)
    RESOLUTION_ERROR = ("ERROR RESOLUCIÓN", Severity.ERROR)
    COLOR_ERROR = ("ERROR COLOR", Severity.ERROR)
    FORMAT_ERROR = ("ERROR FORMATO", Severity.ERROR)
    FILE_ERROR = ("ERROR ARCHIVO", Severity.ERROR)
    CAMERA_ERROR = ("ERROR CÁMARA", Severity.ERROR)

    def __init__(self, label: str, severity: Severity) -> None:
        self.label = label
        self.severity = severity

    @property
    def is_usable(self) -> bool:
        """True when the image may proceed to analysis (possibly with a warning)."""
        return self.severity in (Severity.OK, Severity.WARNING)


def status_for_report(report: ValidationReport) -> ImageStatus:
    """Map a validation report to the most relevant status."""
    if report.has(IssueCode.RESOLUTION):
        return ImageStatus.RESOLUTION_ERROR
    if report.has(IssueCode.CHANNELS) or report.has(IssueCode.BIT_DEPTH):
        return ImageStatus.COLOR_ERROR
    if report.has(IssueCode.LOW_LIGHT):
        return ImageStatus.LOW_LIGHT
    return ImageStatus.VALID


def status_for_error(error: SceneDiffError) -> ImageStatus:
    """Map an acquisition failure to a status."""
    if isinstance(error, InvalidImageError):
        return status_for_report(error.report)
    if isinstance(error, UnsupportedImageFormatError):
        return ImageStatus.FORMAT_ERROR
    if isinstance(error, CameraError):
        return ImageStatus.CAMERA_ERROR
    # ImageLoadError and any other acquisition failure.
    return ImageStatus.FILE_ERROR
