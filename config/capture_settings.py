"""Parameters for acquiring frames from a connected RGB camera."""

from pydantic import BaseModel, ConfigDict, Field


class CaptureSettings(BaseModel):
    """Camera device selection and acquisition behaviour."""

    model_config = ConfigDict(frozen=True)

    camera_index: int = Field(default=0, ge=0, description="OS index of the video device.")
    frame_width: int = Field(default=1280, gt=0, description="Requested capture width (px).")
    frame_height: int = Field(default=720, gt=0, description="Requested capture height (px).")
    warmup_frames: int = Field(
        default=5,
        ge=0,
        description="Frames discarded before the capture so exposure can settle.",
    )
