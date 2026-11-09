"""Rendering of detected changes on top of an image."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.comparison_report import BoundingBox, ChangedObject, ChangeType

Color = tuple[int, int, int]

CHANGE_COLORS: dict[ChangeType, Color] = {
    ChangeType.REMOVED: (220, 50, 50),
    ChangeType.DISPLACED: (40, 180, 80),
    ChangeType.ADDED: (50, 120, 230),
}
PREVIOUS_POSITION_COLOR: Color = (255, 165, 0)

# Resolved by Pillow against the platform font directories (Windows, Linux, macOS).
_FONT_CANDIDATES = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf")


def draw_changes(rgb: np.ndarray, objects: Iterable[ChangedObject]) -> np.ndarray:
    """Return a copy of ``rgb`` with every changed object outlined and captioned.

    Removed objects are boxed at their former position, added objects at
    their new position. A displaced object shows its former position in
    orange, its current position in the color of its change type and an
    arrow between the two. Captions read ``#id label - type [distance]``.
    """
    canvas = np.ascontiguousarray(rgb.copy())
    thickness = max(2, round(max(canvas.shape[:2]) / 400))
    captions: list[tuple[BoundingBox, str, Color]] = []

    for obj in objects:
        color = CHANGE_COLORS[obj.change_type]
        current = obj.bbox_after or obj.bbox_before
        assert current is not None
        if obj.change_type is ChangeType.DISPLACED and obj.bbox_before is not None:
            _draw_box(canvas, obj.bbox_before, PREVIOUS_POSITION_COLOR, thickness)
            cv2.arrowedLine(
                canvas,
                _int_point(obj.bbox_before.center),
                _int_point(current.center),
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.08,
            )
        _draw_box(canvas, current, color, thickness)
        captions.append((current, _caption(obj), color))

    return _draw_captions(canvas, captions, font_size=max(12, round(max(canvas.shape[:2]) / 45)))


def _caption(obj: ChangedObject) -> str:
    text = f"#{obj.object_id} {obj.label.name} - {obj.change_type.value}"
    if obj.distance_cm is not None:
        text += f" {obj.distance_cm:.1f} cm"
    return text


def _draw_box(canvas: np.ndarray, box: BoundingBox, color: Color, thickness: int) -> None:
    cv2.rectangle(canvas, (box.x, box.y), (box.x + box.width, box.y + box.height), color, thickness)


def _int_point(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def _draw_captions(
    canvas: np.ndarray, captions: list[tuple[BoundingBox, str, Color]], font_size: int
) -> np.ndarray:
    """Draw each caption next to its box without covering earlier captions.

    Text is drawn with Pillow because OpenCV fonts lack accented letters.
    """
    if not captions:
        return canvas
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    placed: list[tuple[int, int, int, int]] = []
    for box, text, color in captions:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        size = (right - left + 6, bottom - top + 6)
        x, y = _free_spot(box, size, placed, image.size)
        placed.append((x, y, x + size[0], y + size[1]))
        draw.rectangle(placed[-1], fill=color)
        draw.text((x + 3 - left, y + 3 - top), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _free_spot(
    box: BoundingBox,
    size: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Top-left corner for a caption: above the box, else below it, else stacked lower."""
    width, height = size
    image_width, image_height = image_size
    x = int(np.clip(box.x, 0, max(0, image_width - width)))
    candidates = [box.y - height, box.y + box.height]
    candidates += [box.y + box.height + step * height for step in range(1, 12)]
    for y in candidates:
        y = int(np.clip(y, 0, max(0, image_height - height)))
        rect = (x, y, x + width, y + height)
        if not any(_overlap(rect, other) for other in placed):
            return x, y
    return x, int(np.clip(candidates[0], 0, max(0, image_height - height)))


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@lru_cache(maxsize=8)
def _font(size: int) -> Optional[ImageFont.ImageFont]:
    """A system TrueType font with Latin accents; Pillow's default as a last resort."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except (OSError, TypeError):
        return ImageFont.load_default()
