"""Pinhole camera geometry on torch tensors (runs on the tensors' device)."""

from __future__ import annotations

import torch

from config.camera_intrinsics import CameraIntrinsics


def backproject_pixels(
    u: torch.Tensor, v: torch.Tensor, z: torch.Tensor, intrinsics: CameraIntrinsics
) -> torch.Tensor:
    """3-D camera-frame points of pixels ``(u, v)`` seen at depth ``z``.

    Applies the inverse pinhole model ``X = (u - cx) Z / fx``,
    ``Y = (v - cy) Z / fy``, ``Z = z``. Inputs broadcast together; the result
    has their shape plus a trailing axis of size 3 holding ``(X, Y, Z)``.
    """
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return torch.stack(torch.broadcast_tensors(x, y, z), dim=-1)


def backproject_depth(depth: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor:
    """``H x W x 3`` point map of an ``H x W`` depth map (meters)."""
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2-D, got shape {tuple(depth.shape)}.")
    height, width = depth.shape
    v = torch.arange(height, device=depth.device, dtype=depth.dtype).unsqueeze(1)
    u = torch.arange(width, device=depth.device, dtype=depth.dtype).unsqueeze(0)
    return backproject_pixels(u, v, depth, intrinsics)
