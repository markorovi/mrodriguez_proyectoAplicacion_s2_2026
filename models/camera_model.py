"""Where the camera parameters used to measure distances came from."""

from __future__ import annotations

from enum import Enum


class IntrinsicsSource(str, Enum):
    """How the focal length and principal point of the "before" photo were obtained.

    Listed from the most to the least precise: a calibration of the camera
    with a chessboard, the focal length the camera wrote in the photo (EXIF)
    and, when neither is available, an estimate from a typical field of view.
    """

    CALIBRATED = "calibrada"
    EXIF = "exif"
    FIELD_OF_VIEW = "estimada"

    @property
    def description(self) -> str:
        """Plain-language origin of the parameters, for the report."""
        return _DESCRIPTIONS[self]


_DESCRIPTIONS = {
    IntrinsicsSource.CALIBRATED: "la calibración de la cámara",
    IntrinsicsSource.EXIF: "los datos de la cámara guardados en la foto",
    IntrinsicsSource.FIELD_OF_VIEW: "una estimación estándar de la cámara",
}
