"""Pinhole camera intrinsic parameters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CameraIntrinsics(BaseModel):
    """Focal lengths and principal point of a pinhole camera, in pixels.

    The parameters are tied to the resolution they were calibrated at
    (``reference_width`` x ``reference_height``). Use :meth:`scaled_to` to
    obtain the equivalent parameters for an image of a different size.
    """

    model_config = ConfigDict(frozen=True)

    fx: float = Field(default=518.86, gt=0, description="Horizontal focal length (px).")
    fy: float = Field(default=519.47, gt=0, description="Vertical focal length (px).")
    cx: float = Field(default=325.58, ge=0, description="Principal point, x (px).")
    cy: float = Field(default=253.74, ge=0, description="Principal point, y (px).")
    reference_width: int = Field(default=640, gt=0)
    reference_height: int = Field(default=480, gt=0)

    def scaled_to(self, width: int, height: int) -> CameraIntrinsics:
        """Return the intrinsics rescaled to an image of ``width`` x ``height``."""
        if width <= 0 or height <= 0:
            raise ValueError("Target resolution must be positive.")
        sx = width / self.reference_width
        sy = height / self.reference_height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            reference_width=width,
            reference_height=height,
        )
