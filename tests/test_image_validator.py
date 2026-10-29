import numpy as np
import pytest

from config import ImageSettings
from models import ImageData, InvalidImageError, UnsupportedImageFormatError
from utils import ImageValidator, mean_luminance


@pytest.fixture
def validator() -> ImageValidator:
    return ImageValidator()


def _image(height: int, width: int, *, channels: int = 3, value: int = 128, dtype=np.uint8) -> ImageData:
    shape = (height, width) if channels == 1 else (height, width, channels)
    return ImageData(pixels=np.full(shape, value, dtype=dtype), source="test")


def test_valid_scene_passes_without_warnings(validator, scene_image):
    report = validator.validate(scene_image)
    assert report.is_valid
    assert report.warnings == []


def test_exact_minimum_resolution_is_accepted(validator):
    assert validator.validate(_image(480, 640)).is_valid


def test_portrait_orientation_is_accepted(validator):
    assert validator.validate(_image(640, 480)).is_valid


@pytest.mark.parametrize(("height", "width"), [(479, 640), (480, 639), (240, 320)])
def test_resolution_below_minimum_is_rejected(validator, height, width):
    report = validator.validate(_image(height, width))
    assert not report.is_valid
    assert "resolución" in report.errors[0]


@pytest.mark.parametrize("channels", [1, 4])
def test_non_rgb_images_are_rejected(validator, channels):
    report = validator.validate(_image(480, 640, channels=channels))
    assert not report.is_valid
    assert any("RGB" in error for error in report.errors)


def test_non_8_bit_images_are_rejected(validator):
    report = validator.validate(_image(480, 640, dtype=np.uint16))
    assert any("8 bits" in error for error in report.errors)


def test_dark_image_produces_warning_not_error(validator):
    report = validator.validate(_image(480, 640, value=10))
    assert report.is_valid
    assert len(report.warnings) == 1


def test_low_light_threshold_is_configurable():
    strict = ImageValidator(ImageSettings(low_light_threshold=200))
    assert strict.validate(_image(480, 640, value=150)).warnings


def test_ensure_valid_raises_with_report(validator):
    with pytest.raises(InvalidImageError) as info:
        validator.ensure_valid(_image(100, 100))
    assert not info.value.report.is_valid
    assert "resolución" in str(info.value)


@pytest.mark.parametrize("name", ["photo.png", "photo.JPG", "photo.jpeg", "dir/photo.webp"])
def test_supported_file_formats(validator, name):
    validator.check_file_format(name)


@pytest.mark.parametrize("name", ["photo.gif", "notes.txt", "no_extension"])
def test_unsupported_file_formats(validator, name):
    with pytest.raises(UnsupportedImageFormatError):
        validator.check_file_format(name)


def test_mean_luminance_uses_perceptual_weights():
    pure_green = np.zeros((2, 2, 3), dtype=np.uint8)
    pure_green[..., 1] = 255
    assert mean_luminance(pure_green) == pytest.approx(0.587 * 255)


def test_mean_luminance_rejects_non_rgb():
    with pytest.raises(ValueError):
        mean_luminance(np.zeros((4, 4), dtype=np.uint8))
