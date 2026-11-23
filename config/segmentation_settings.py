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
        default=False,
        description=(
            "Weights are always loaded from the cache first, without the network. When they are "
            "missing, true fails right away and false downloads them into the cache."
        ),
    )
    min_label_coverage: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fraction of a region a class must cover to name the object after it.",
    )
    min_label_confidence: float = Field(
        default=0.76,
        ge=0.0,
        le=1.0,
        description=(
            "Mean model confidence (0-1) a changed object's class needs for its name to be shown; "
            "below it the object is shown as unidentified. Only the displayed name depends on it. "
            "Measured on the three real pairs: wrong names reached 0.72 at most, right ones 0.80 "
            "at least (0 turns the rule off)."
        ),
    )
    check_indoor_scene: bool = Field(
        default=True,
        description="Reject a photo whose segmentation shows mostly open-air scenery (sky, trees, grass...).",
    )
    max_outdoor_percent: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Largest share of a photo (percent) outdoor classes may cover in an indoor scene.",
    )
