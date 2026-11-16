"""How to obtain the camera parameters when no calibration matches the photo."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Horizontal field of view of the camera the project used as reference
# (fx = 518.86 px at 640 px wide); typical of webcams and phone main cameras.
DEFAULT_HORIZONTAL_FOV_DEG = 63.3


class CameraEstimationSettings(BaseModel):
    """Fallbacks used when no calibrated profile (``AppSettings.camera``) applies.

    With ``use_exif``, a photo that carries its 35 mm-equivalent focal length
    is measured with it. Otherwise (webcam frames, edited photos) the focal
    length is derived from ``horizontal_fov_deg``, with square pixels and the
    principal point at the image center.
    """

    model_config = ConfigDict(frozen=True)

    use_exif: bool = Field(default=True, description="Use the focal length stored in the photo's EXIF data.")
    horizontal_fov_deg: float = Field(
        default=DEFAULT_HORIZONTAL_FOV_DEG,
        gt=10.0,
        lt=170.0,
        description="Horizontal field of view assumed for cameras without calibration or EXIF data.",
    )
