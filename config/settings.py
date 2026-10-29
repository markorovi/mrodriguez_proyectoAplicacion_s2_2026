"""Root application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from config.camera_intrinsics import CameraIntrinsics
from config.capture_settings import CaptureSettings
from config.depth_settings import DepthSettings
from config.image_settings import ImageSettings


class AppSettings(BaseModel):
    """Aggregates every configuration section of the application."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "Scene Change Detector"
    image: ImageSettings = Field(default_factory=ImageSettings)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    depth: DepthSettings = Field(default_factory=DepthSettings)
    camera: CameraIntrinsics = Field(default_factory=CameraIntrinsics)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> AppSettings:
        """Load settings from a JSON file; missing keys keep their defaults."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_settings(path: Optional[Union[str, Path]] = None) -> AppSettings:
    """Return settings from ``path`` if given, otherwise the defaults."""
    return AppSettings.from_json_file(path) if path is not None else AppSettings()
