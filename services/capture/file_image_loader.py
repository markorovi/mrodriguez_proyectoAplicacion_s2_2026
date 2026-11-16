"""Loads images stored on the local filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

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

_EXIF_IFD = 0x8769
_FOCAL_LENGTH_IN_35MM_FILM = 0xA405


class FileImageLoader(ImageSource):
    """Decodes an image file into an RGB :class:`ImageData`.

    Uses Pillow, which handles non-ASCII paths on every platform, applies the
    EXIF orientation written by phone cameras and detects truncated files.
    It also keeps the 35 mm-equivalent focal length from the EXIF data, when
    the camera wrote it, so distances can be measured with the camera's own
    parameters.
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
        pixels, focal_length_35mm = self._decode()
        return ImageData(pixels=pixels, source=self.description, focal_length_35mm=focal_length_35mm)

    def _decode(self) -> tuple[np.ndarray, Optional[float]]:
        try:
            with Image.open(self._path) as image:
                image.load()
                focal_length_35mm = _focal_length_35mm(image)
                image = ImageOps.exif_transpose(image)
                if image.mode in _RGB_CONVERTIBLE_MODES and image.mode != "RGB":
                    image = image.convert("RGB")
                return np.array(image), focal_length_35mm
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageLoadError(
                f"El archivo '{self._path.name}' está dañado o no es una imagen válida."
            ) from exc


def _focal_length_35mm(image: Image.Image) -> Optional[float]:
    """35 mm-equivalent focal length from the EXIF data, or ``None`` if absent or unusable.

    The equivalence is defined on the image diagonal, so it does not depend on
    the photo's orientation. Malformed metadata is ignored rather than
    rejecting an otherwise valid photo.
    """
    try:
        value = image.getexif().get_ifd(_EXIF_IFD).get(_FOCAL_LENGTH_IN_35MM_FILM)
        focal = float(value) if value is not None else 0.0
    except (TypeError, ValueError, AttributeError, KeyError, OSError):
        return None
    return focal if focal > 0.0 else None
