"""Domain exception hierarchy.

Every error raised intentionally by the application derives from
:class:`SceneDiffError`, so callers (orchestration, UI) can handle the
whole family with a single ``except`` clause. Messages are written for the
end user and avoid technical jargon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.image_data import ImageData
    from models.validation_report import ValidationReport


class SceneDiffError(Exception):
    """Base class for all application errors."""


class ImageError(SceneDiffError):
    """Base class for problems with an input image."""


class ImageLoadError(ImageError):
    """The image file is missing, unreadable or corrupt."""


class UnsupportedImageFormatError(ImageError):
    """The image file has an extension that is not accepted."""


class InvalidImageError(ImageError):
    """The image was decoded but does not meet the acceptance criteria.

    The rejected ``image`` is kept so callers can still show it to the user.
    """

    def __init__(self, report: ValidationReport, image: ImageData | None = None) -> None:
        self.report = report
        self.image = image
        super().__init__(" ".join(report.errors) or "La imagen no es válida.")


class CameraError(SceneDiffError):
    """The camera could not be opened or failed to deliver a frame."""


class DepthEstimationError(SceneDiffError):
    """Depth inference could not be completed."""


class ModelLoadError(DepthEstimationError):
    """The pretrained model or its weights could not be loaded."""
