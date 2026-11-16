"""Domain exception hierarchy.

Every error raised intentionally by the application derives from
:class:`SceneDiffError`, so callers (orchestration, UI) can handle the
whole family with a single ``except`` clause. Messages are written for the
end user and avoid technical jargon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.alignment import AlignmentReport
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


class AlignmentError(SceneDiffError):
    """The "after" view could not be registered onto the "before" view.

    ``report`` holds the matching statistics that led to the rejection.
    """

    def __init__(self, report: AlignmentReport) -> None:
        self.report = report
        super().__init__(report.message)


class SegmentationError(SceneDiffError):
    """Semantic segmentation could not be completed or its model loaded."""


class CalibrationError(SceneDiffError):
    """The camera could not be calibrated from the given chessboard photos."""


class NonIndoorSceneError(SceneDiffError):
    """One or both photos do not show an indoor space.

    ``outdoor_percent`` maps the name of each rejected photo ("Antes",
    "Después") to the share of it (0-100) that shows open-air scenery.
    The message names those photos so the user can replace only them.
    """

    def __init__(self, outdoor_percent: dict[str, float]) -> None:
        self.outdoor_percent = dict(outdoor_percent)
        names = " y ".join(f"«{name}»" for name in self.outdoor_percent)
        if len(self.outdoor_percent) == 1:
            subject = f"La fotografía {names} no parece corresponder"
        else:
            subject = f"Las fotografías {names} no parecen corresponder"
        super().__init__(
            f"{subject} a un espacio interior. Por favor, elija imágenes tomadas "
            "dentro de una habitación o área de trabajo."
        )

    @property
    def views(self) -> tuple[str, ...]:
        return tuple(self.outdoor_percent)
