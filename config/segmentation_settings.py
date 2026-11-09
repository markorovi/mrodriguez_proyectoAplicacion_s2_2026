"""Configuration of the semantic segmentation model."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from config.depth_settings import DevicePreference
from config.paths import WEIGHTS_DIR


class SegmentationSettings(BaseModel):
    """Model selection and runtime options for semantic segmentation."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=True, description="Name changed objects; when false they are reported as unidentified."
    )
    model_id: str = Field(
        default="nvidia/segformer-b4-finetuned-ade-512-512",
        description=(
            "Hugging Face SegFormer checkpoint (ADE20K classes) or local path. B4 recognizes "
            "everyday objects in real photos that B0/B2 confuse with the furniture under them."
        ),
    )
    device: DevicePreference = DevicePreference.AUTO
    half_precision: bool = Field(default=True, description="Use float16 autocast when running on CUDA.")
    cache_dir: Path = WEIGHTS_DIR
    local_files_only: bool = Field(
        default=False, description="Never reach the network; load weights from the cache only."
    )
    min_label_coverage: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fraction of a region a class must cover to name the object after it.",
    )
