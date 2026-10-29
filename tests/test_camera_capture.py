"""Frame acquisition from an RGB camera, using a fake video device."""

import cv2
import numpy as np
import pytest

from config import CaptureSettings
from models import CameraError, InvalidImageError
from services.capture import CameraCapture, ImageAcquisitionService
from tests.doubles import FakeVideoDevice


def _bgr_frame(height: int = 720, width: int = 1280) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 255  # Blue in OpenCV's BGR order.
    frame[..., 2] = 40
    return frame


def _capture(device: FakeVideoDevice, **settings) -> CameraCapture:
    return CameraCapture(CaptureSettings(**settings), device_factory=lambda index: device)


def test_frame_is_converted_from_bgr_to_rgb():
    image = _capture(FakeVideoDevice(_bgr_frame())).acquire()

    assert image.pixels.shape == (720, 1280, 3)
    assert tuple(image.pixels[0, 0]) == (40, 0, 255)


def test_warmup_frames_are_discarded_and_device_released():
    device = FakeVideoDevice(_bgr_frame())
    _capture(device, warmup_frames=3).acquire()

    assert device.read_calls == 4
    assert device.released


def test_requested_resolution_is_applied():
    device = FakeVideoDevice(_bgr_frame())
    _capture(device, frame_width=1920, frame_height=1080).acquire()

    assert device.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1920
    assert device.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 1080


def test_camera_index_is_forwarded_and_described():
    requested = []

    def factory(index):
        requested.append(index)
        return FakeVideoDevice(_bgr_frame())

    capture = CameraCapture(CaptureSettings(camera_index=2), device_factory=factory)
    image = capture.acquire()

    assert requested == [2]
    assert image.source == "camera:2"


def test_unavailable_camera_raises_and_releases():
    device = FakeVideoDevice(_bgr_frame(), opened=False)
    with pytest.raises(CameraError, match="No se pudo acceder"):
        _capture(device).acquire()
    assert device.released


def test_failed_read_raises_and_releases():
    device = FakeVideoDevice(frame=None)
    with pytest.raises(CameraError):
        _capture(device).acquire()
    assert device.released


def test_acquisition_service_validates_camera_frames():
    service = ImageAcquisitionService(device_factory=lambda index: FakeVideoDevice(_bgr_frame()))
    acquired = service.from_camera()
    assert acquired.image.resolution == (1280, 720)


def test_acquisition_service_rejects_low_resolution_camera():
    small = _bgr_frame(240, 320)
    service = ImageAcquisitionService(device_factory=lambda index: FakeVideoDevice(small))
    with pytest.raises(InvalidImageError):
        service.from_camera()
