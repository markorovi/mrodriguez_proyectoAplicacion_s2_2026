"""Loading images from disk and acquiring validated images."""

import numpy as np
import pytest
from PIL import Image

from config.paths import SAMPLES_DIR
from models import (
    AcquiredImage,
    ImageLoadError,
    InvalidImageError,
    UnsupportedImageFormatError,
)
from services.capture import FileImageLoader, ImageAcquisitionService

SAMPLE_NAMES = [
    "scene_before.png",
    "scene_after_moved.png",
    "scene_after_removed.png",
    "scene_low_light.png",
]


def test_png_round_trip_is_lossless(write_image, scene_rgb):
    image = FileImageLoader(write_image(scene_rgb, "scene.png")).acquire()

    np.testing.assert_array_equal(image.pixels, scene_rgb)
    assert image.pixels.dtype == np.uint8
    assert image.resolution == (640, 480)


def test_jpeg_is_decoded_as_rgb(tmp_path, scene_rgb):
    path = tmp_path / "scene.jpg"
    Image.fromarray(scene_rgb).save(path, quality=95)

    image = FileImageLoader(path).acquire()

    assert image.pixels.shape == (480, 640, 3)
    # Lossy, but channel order must be preserved (R and B differ widely here).
    decoded_means = image.pixels.reshape(-1, 3).mean(axis=0)
    original_means = scene_rgb.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(decoded_means, original_means, atol=2.0)


def test_source_records_the_file_path(write_image, scene_rgb):
    path = write_image(scene_rgb)
    assert FileImageLoader(path).acquire().source == str(path)


def test_paths_with_non_ascii_characters_are_supported(write_image, scene_rgb):
    path = write_image(scene_rgb, "imágenes de prueba/escena_añadida.png")
    assert FileImageLoader(path).acquire().pixels.shape == (480, 640, 3)


def test_rgba_is_converted_to_rgb(write_image, scene_rgb):
    rgba = np.dstack([scene_rgb, np.full(scene_rgb.shape[:2], 255, np.uint8)])
    image = FileImageLoader(write_image(rgba, "alpha.png")).acquire()
    np.testing.assert_array_equal(image.pixels, scene_rgb)


def test_grayscale_is_kept_single_channel(write_image):
    image = FileImageLoader(write_image(np.zeros((480, 640), np.uint8), "gray.png")).acquire()
    assert image.pixels.ndim == 2


def test_missing_file_raises_load_error(tmp_path):
    with pytest.raises(ImageLoadError, match="No se encontró"):
        FileImageLoader(tmp_path / "missing.png").acquire()


def test_directory_is_not_an_image(tmp_path):
    folder = tmp_path / "folder.png"
    folder.mkdir()
    with pytest.raises(ImageLoadError):
        FileImageLoader(folder).acquire()


def test_corrupt_file_raises_load_error(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises(ImageLoadError, match="dañado"):
        FileImageLoader(path).acquire()


def test_truncated_file_raises_load_error(write_image, scene_rgb):
    path = write_image(scene_rgb, "truncated.png")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageLoadError):
        FileImageLoader(path).acquire()


def test_unsupported_extension_is_rejected_before_reading(tmp_path):
    with pytest.raises(UnsupportedImageFormatError):
        FileImageLoader(tmp_path / "animation.gif").acquire()


def test_acquisition_service_returns_validated_image(write_image, scene_rgb):
    acquired = ImageAcquisitionService().from_file(write_image(scene_rgb))

    assert isinstance(acquired, AcquiredImage)
    assert acquired.report.is_valid
    assert acquired.warnings == []


def test_acquisition_service_rejects_low_resolution(write_image, scene_rgb):
    small = scene_rgb[:240, :320].copy()
    with pytest.raises(InvalidImageError):
        ImageAcquisitionService().from_file(write_image(small, "small.png"))


def test_acquisition_service_rejects_grayscale(write_image):
    with pytest.raises(InvalidImageError, match="RGB"):
        ImageAcquisitionService().from_file(write_image(np.zeros((480, 640), np.uint8), "gray.png"))


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_bundled_samples_are_valid_inputs(name):
    acquired = ImageAcquisitionService().from_file(SAMPLES_DIR / name)
    assert acquired.image.resolution == (640, 480)


def test_bundled_low_light_sample_triggers_warning():
    acquired = ImageAcquisitionService().from_file(SAMPLES_DIR / "scene_low_light.png")
    assert len(acquired.warnings) == 1
