"""Configuration of the monocular depth estimation model."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from config.paths import WEIGHTS_DIR


class DevicePreference(str, Enum):
    """Where inference should run."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class DepthModelVariant(str, Enum):
    """Depth Anything V2 checkpoints fine-tuned for metric indoor depth."""

    SMALL = "small"
    BASE = "base"
    LARGE = "large"

    @property
    def model_id(self) -> str:
        return f"depth-anything/Depth-Anything-V2-Metric-Indoor-{self.value.capitalize()}-hf"


class DepthSettings(BaseModel):
    """Model selection and runtime options for depth inference."""

    model_config = ConfigDict(frozen=True)

    variant: DepthModelVariant = DepthModelVariant.SMALL
    model_id_override: Optional[str] = Field(
        default=None,
        description="Explicit Hugging Face model id or local path; takes precedence over 'variant'.",
    )
    is_metric: bool = Field(
        default=True,
        description="Whether the checkpoint outputs metric depth in meters (true for every variant).",
    )
    device: DevicePreference = DevicePreference.AUTO
    half_precision: bool = Field(
        default=True,
        description="Use float16 autocast when running on CUDA.",
    )
    cache_dir: Path = WEIGHTS_DIR
    local_files_only: bool = Field(
        default=False,
        description="Never reach the network; load weights from the cache only.",
    )

    @property
    def model_id(self) -> str:
        return self.model_id_override or self.variant.model_id
