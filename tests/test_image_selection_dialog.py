"""Image selection dialog, rendered offscreen."""

import numpy as np
import pytest

from config.paths import SAMPLES_DIR
from services.capture import ImageAcquisitionService
from tests.doubles import FakeVideoDevice
from ui.image_selection_dialog import ImageSelectionDialog
from ui.image_status import ImageStatus
from ui.qt_image import to_qimage


@pytest.fixture
def dialog(qapp):
    widget = ImageSelectionDialog()
    yield widget
    widget.deleteLater()


def _camera_dialog(frame, opened=True):
    requested = []

    def factory(index):
        requested.append(index)
        return FakeVideoDevice(frame, opened=opened)

    service = ImageAcquisitionService(device_factory=factory)
    return ImageSelectionDialog(service=service), requested


def test_starts_empty_and_blocks_continue(dialog):
    assert dialog.before_slot.status is ImageStatus.EMPTY
    assert dialog.after_slot.status is ImageStatus.EMPTY
    assert not dialog.can_continue
    assert dialog.selected_pair() is None


def test_two_valid_images_enable_continue(dialog):
    dialog.before_slot.load_file(SAMPLES_DIR / "scene_before.png")
    dialog.after_slot.load_file(SAMPLES_DIR / "scene_after_moved.png")

    assert dialog.before_slot.status_text == "VÁLIDA"
    assert dialog.after_slot.status_text == "VÁLIDA"
    assert dialog.before_slot.has_preview
    assert dialog.can_continue
    before, after = dialog.selected_pair()
    assert before.image.source.endswith("scene_before.png")
    assert after.image.source.endswith("scene_after_moved.png")


def test_low_light_shows_warning_but_allows_continue(dialog):
    dialog.before_slot.load_file(SAMPLES_DIR / "scene_before.png")
    dialog.after_slot.load_file(SAMPLES_DIR / "scene_low_light.png")

    assert dialog.after_slot.status_text == "ADVERTENCIA LUZ"
    assert "poco iluminada" in dialog.after_slot.detail_text
    assert dialog.can_continue


def test_low_resolution_shows_error_with_preview(dialog, write_image, scene_rgb):
    dialog.before_slot.load_file(SAMPLES_DIR / "scene_before.png")
    dialog.after_slot.load_file(write_image(scene_rgb[:240, :320].copy(), "small.png"))

    assert dialog.after_slot.status_text == "ERROR RESOLUCIÓN"
    assert dialog.after_slot.has_preview
    assert "320x240" in dialog.after_slot.detail_text
    assert dialog.after_slot.acquired is None
    assert not dialog.can_continue
    assert dialog.selected_pair() is None


def test_grayscale_shows_color_error(dialog, write_image):
    dialog.before_slot.load_file(write_image(np.zeros((480, 640), np.uint8), "gray.png"))
    assert dialog.before_slot.status is ImageStatus.COLOR_ERROR


def test_corrupt_file_shows_file_error_without_preview(dialog, tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    dialog.before_slot.load_file(corrupt)

    assert dialog.before_slot.status is ImageStatus.FILE_ERROR
    assert not dialog.before_slot.has_preview


def test_unsupported_format_shows_format_error(dialog, tmp_path):
    dialog.before_slot.load_file(tmp_path / "animation.gif")
    assert dialog.before_slot.status is ImageStatus.FORMAT_ERROR


def test_replacing_a_valid_image_with_an_invalid_one_blocks_continue(dialog, write_image, scene_rgb):
    dialog.before_slot.load_file(SAMPLES_DIR / "scene_before.png")
    dialog.after_slot.load_file(SAMPLES_DIR / "scene_after_moved.png")
    assert dialog.can_continue

    dialog.after_slot.load_file(write_image(scene_rgb[:200, :200].copy(), "tiny.png"))
    assert not dialog.can_continue


def test_capture_uses_the_selected_camera(qapp):
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    dialog, requested = _camera_dialog(frame)
    dialog.camera_index = 1

    dialog.before_slot.capture()

    assert requested == [1]
    assert dialog.before_slot.status is ImageStatus.VALID
    assert "camera:1" in dialog.before_slot.detail_text


def test_unavailable_camera_shows_camera_error(qapp):
    dialog, _ = _camera_dialog(frame=None, opened=False)
    dialog.after_slot.capture()
    assert dialog.after_slot.status_text == "ERROR CÁMARA"


@pytest.mark.parametrize(
    ("pixels", "size"),
    [
        (np.zeros((48, 64, 3), np.uint8), (64, 48)),
        (np.zeros((48, 64), np.uint8), (64, 48)),
        (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64), (64, 48)),
        (np.zeros((48, 64, 4), np.uint8), (64, 48)),
    ],
)
def test_to_qimage_preserves_size(qapp, pixels, size):
    image = to_qimage(pixels)
    assert (image.width(), image.height()) == size


def test_to_qimage_preserves_rgb_order(qapp):
    pixels = np.zeros((2, 2, 3), np.uint8)
    pixels[..., 0] = 255
    color = to_qimage(pixels).pixelColor(0, 0)
    assert (color.red(), color.green(), color.blue()) == (255, 0, 0)


def test_to_qimage_rejects_unknown_layout(qapp):
    with pytest.raises(ValueError):
        to_qimage(np.zeros((4, 4, 2), np.uint8))
