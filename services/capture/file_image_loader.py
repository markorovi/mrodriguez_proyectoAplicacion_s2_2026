"""Loads images stored on the local filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from models.exceptions import ImageLoadError
from models.image_data import ImageData
from services.capture.image_source import ImageSource
from utils.image_validator import ImageValidator

# Colour modes that carry RGB information and can be converted losslessly
# (apart from dropping alpha). Grayscale and high-bit-depth modes are kept
# as decoded so the validator can reject them explicitly.
_RGB_CONVERTIBLE_MODES = frozenset({"RGB", "RGBA", "RGBX", "P", "PA", "CMYK", "YCbCr"})


class FileImageLoader(ImageSource):
    """Decodes an image file into an RGB :class:`ImageData`.

    Uses Pillow, which handles non-ASCII paths on every platform, applies the
    EXIF orientation written by phone cameras and detects truncated files.
    """

    def __init__(self, path: Union[str, Path], validator: ImageValidator | None = None) -> None:
        self._path = Path(path)
        self._validator = validator or ImageValidator()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def description(self) -> str:
        return str(self._path)

    def acquire(self) -> ImageData:
        self._validator.check_file_format(self._path)
        if not self._path.is_file():
            raise ImageLoadError(f"No se encontró el archivo de imagen '{self._path}'.")
        return ImageData(pixels=self._decode(), source=self.description)

    def _decode(self) -> np.ndarray:
        try:
            with Image.open(self._path) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                if image.mode in _RGB_CONVERTIBLE_MODES and image.mode != "RGB":
                    image = image.convert("RGB")
                return np.array(image)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageLoadError(
                f"El archivo '{self._path.name}' está dañado o no es una imagen válida."
            ) from exc
