"""Procedural indoor scenes for demos and tests.

Renders a simple one-point-perspective room (ceiling, walls, plank floor and
a wall frame) with optional box-shaped objects resting on the floor. Moving or
removing a box between two renders yields controlled "before"/"after" pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

Color = tuple[int, int, int]


@dataclass(frozen=True)
class SceneObject:
    """A box placed on the floor; coordinates are normalized to the image size."""

    center_x: float
    base_y: float = 0.85
    size: float = 0.18
    color: Color = (180, 60, 50)


class SyntheticSceneBuilder:
    """Renders RGB ``height x width x 3`` uint8 images of a synthetic room."""

    _CEILING: Color = (225, 225, 220)
    _BACK_WALL: Color = (200, 190, 170)
    _SIDE_WALL: Color = (180, 170, 150)
    _FLOOR: Color = (140, 100, 65)
    _PLANK_LINE: Color = (110, 78, 50)
    _FRAME: Color = (60, 70, 90)

    def __init__(self, width: int = 640, height: int = 480, seed: int = 0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Scene dimensions must be positive.")
        self._width = width
        self._height = height
        self._seed = seed

    def render(self, objects: Sequence[SceneObject] = (), brightness: float = 1.0) -> np.ndarray:
        """Draw the room, then each object, and apply a global brightness factor."""
        canvas = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._draw_room(canvas)
        for obj in objects:
            self._draw_box(canvas, obj)
        return self._finish(canvas, brightness)

    def _px(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x * self._width)), int(round(y * self._height))

    def _vanishing_point(self) -> np.ndarray:
        return np.array(self._px(0.5, 0.42), dtype=np.float64)

    def _draw_room(self, canvas: np.ndarray) -> None:
        tl, tr = self._px(0.25, 0.21), self._px(0.75, 0.21)
        bl, br = self._px(0.25, 0.63), self._px(0.75, 0.63)
        w, h = self._width, self._height
        self._fill(canvas, [(0, 0), (w, 0), tr, tl], self._CEILING)
        self._fill(canvas, [(0, h), bl, br, (w, h)], self._FLOOR)
        # Planks are drawn before the walls so the walls clip them to the floor.
        self._draw_floor_planks(canvas, bl, br)
        self._fill(canvas, [(0, 0), tl, bl, (0, h)], self._SIDE_WALL)
        self._fill(canvas, [(w, 0), (w, h), br, tr], self._SIDE_WALL)
        self._fill(canvas, [tl, tr, br, bl], self._BACK_WALL)
        cv2.rectangle(canvas, self._px(0.42, 0.29), self._px(0.58, 0.45), self._FRAME, -1)
        cv2.rectangle(canvas, self._px(0.44, 0.31), self._px(0.56, 0.43), (150, 170, 190), -1)

    def _draw_floor_planks(self, canvas: np.ndarray, bl: tuple[int, int], br: tuple[int, int]) -> None:
        for t in np.linspace(0.0, 1.0, 9):
            top = (int(bl[0] + t * (br[0] - bl[0])), bl[1])
            bottom = (int(-0.5 * self._width + t * 2.0 * self._width), self._height)
            cv2.line(canvas, top, bottom, self._PLANK_LINE, 2, cv2.LINE_AA)

    def _draw_box(self, canvas: np.ndarray, obj: SceneObject) -> None:
        box_w = obj.size * self._width
        box_h = 0.8 * box_w
        cx, base = obj.center_x * self._width, obj.base_y * self._height
        front = np.array(
            [[cx - box_w / 2, base - box_h], [cx + box_w / 2, base - box_h],
             [cx + box_w / 2, base], [cx - box_w / 2, base]],
            dtype=np.float64,
        )
        back = front + (self._vanishing_point() - front) * 0.18
        top = [front[0], front[1], back[1], back[0]]
        side_idx = (1, 2) if cx < self._width / 2 else (0, 3)
        side = [front[side_idx[0]], back[side_idx[0]], back[side_idx[1]], front[side_idx[1]]]

        self._fill(canvas, side, self._shade(obj.color, 0.7))
        self._fill(canvas, top, self._shade(obj.color, 1.25))
        self._fill(canvas, front, obj.color)

    def _finish(self, canvas: np.ndarray, brightness: float) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        noisy = canvas.astype(np.float64) + rng.normal(0.0, 4.0, canvas.shape)
        return np.clip(noisy * brightness, 0, 255).astype(np.uint8)

    @staticmethod
    def _fill(canvas: np.ndarray, points: Sequence, color: Color) -> None:
        polygon = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(canvas, [polygon], color, cv2.LINE_AA)

    @staticmethod
    def _shade(color: Color, factor: float) -> Color:
        r, g, b = (int(min(255, c * factor)) for c in color)
        return r, g, b
