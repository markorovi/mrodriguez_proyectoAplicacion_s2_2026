"""Indoor scene check (RF-012) with the real pretrained models.

The unit tests in ``test_indoor_validation.py`` use a fake segmentation that
sees sky on blue pixels; these tests run the real one on versioned photos:

    RUN_INTEGRATION=1 pytest tests/test_indoor_real_integration.py -v

``no_interior_01.png`` is an outdoor photo (a city at night) kept only as the
negative case of this check; it is not offered as an example in the
application. The metadata test runs always, without the models.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

from config import AppSettings, DepthSettings, SegmentationSettings
from config.paths import SAMPLES_DIR
from controllers import PipelineController
from models import NonIndoorSceneError
from tests.jargon import JARGON

OUTDOOR_PHOTO = SAMPLES_DIR / "no_interior_01.png"
GPS_IFD = 0x8825

_skip_without_models = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run tests that load pretrained models",
)


def needs_models(test):
    """Integration marks for the tests that load the real models (the metadata test does not)."""
    return pytest.mark.integration(_skip_without_models(test))


@pytest.fixture(scope="module")
def controller():
    # The weights must already be in the local cache: these tests never reach the network.
    settings = AppSettings(
        depth=DepthSettings(local_files_only=True),
        segmentation=SegmentationSettings(local_files_only=True),
    )
    with PipelineController(settings) as pipeline:
        yield pipeline


@needs_models
def test_outdoor_photo_is_rejected_with_a_plain_message(controller):
    with pytest.raises(NonIndoorSceneError) as rejected:
        controller.analyze_files(SAMPLES_DIR / "scene_before.png", OUTDOOR_PHOTO)

    error = rejected.value
    assert list(error.outdoor_percent) == ["Después"]
    assert error.outdoor_percent["Después"] > SegmentationSettings().max_outdoor_percent
    message = str(error)
    print(f"\nMensaje: {message}\nExterior: {error.outdoor_percent['Después']:.1f} %")
    assert "«Después»" in message and "espacio interior" in message
    assert not [term for term in JARGON if term in message.lower()]


@needs_models
def test_outdoor_before_photo_is_rejected_and_only_it_is_named(controller):
    # The symmetric case: the outdoor photo as the reference and a real indoor photo after it.
    with pytest.raises(NonIndoorSceneError) as rejected:
        controller.analyze_files(OUTDOOR_PHOTO, SAMPLES_DIR / "ejemplo_01_b.jpg")

    error = rejected.value
    assert list(error.outdoor_percent) == ["Antes"]
    assert error.outdoor_percent["Antes"] > SegmentationSettings().max_outdoor_percent
    message = str(error)
    print(f"\nMensaje: {message}\nExterior: {error.outdoor_percent['Antes']:.1f} %")
    assert "«Antes»" in message and "«Después»" not in message
    assert not [term for term in JARGON if term in message.lower()]


@needs_models
@pytest.mark.parametrize(
    "before, after",
    [("scene_before.png", "scene_after_moved.png"), ("ejemplo_01_a.jpg", "ejemplo_01_b.jpg")],
)
def test_indoor_photos_are_accepted(controller, before, after):
    report = controller.analyze_files(SAMPLES_DIR / before, SAMPLES_DIR / after)
    assert report.alignment is not None  # The analysis reached the end: no NonIndoorSceneError.


def test_outdoor_photo_carries_no_metadata():
    with Image.open(OUTDOOR_PHOTO) as image:
        exif = image.getexif()
        assert len(exif) == 0
        assert len(exif.get_ifd(GPS_IFD)) == 0
        assert "exif" not in image.info and "icc_profile" not in image.info
        assert not [key for key, value in image.info.items() if isinstance(value, str)]
