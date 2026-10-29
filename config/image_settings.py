"""Acceptance criteria for input images."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class ImageSettings(BaseModel):
    """Minimum quality requirements an input image must satisfy."""

    model_config = ConfigDict(frozen=True)

    min_width: int = Field(default=640, gt=0)
    min_height: int = Field(default=480, gt=0)
    allowed_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    low_light_threshold: float = Field(
        default=40.0,
        ge=0.0,
        le=255.0,
        description="Mean luminance (0-255) below which the image is flagged as poorly lit.",
    )

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value
        )
        if not normalized:
            raise ValueError("At least one image extension must be allowed.")
        return normalized
