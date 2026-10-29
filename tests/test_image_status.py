"""Mapping of validation results and acquisition errors to UI status badges."""

import numpy as np
import pytest

from models import (
    CameraError,
    ImageData,
    ImageLoadError,
    InvalidImageError,
    IssueCode,
    UnsupportedImageFormatError,
    ValidationReport,
)
from ui.image_status import ImageStatus, status_for_error, status_for_report
from utils import ImageValidator


def _report(errors=(), warnings=()) -> ValidationReport:
    report = ValidationReport()
    for code in errors:
        report.add_error(code, "error")
    for code in warnings:
        report.add_warning(code, "warning")
    return report


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (_report(), ImageStatus.VALID),
        (_report(warnings=[IssueCode.LOW_LIGHT]), ImageStatus.LOW_LIGHT),
        (_report(errors=[IssueCode.RESOLUTION]), ImageStatus.RESOLUTION_ERROR),
        (_report(errors=[IssueCode.CHANNELS]), ImageStatus.COLOR_ERROR),
        (_report(errors=[IssueCode.BIT_DEPTH]), ImageStatus.COLOR_ERROR),
        (_report(errors=[IssueCode.CHANNELS, IssueCode.RESOLUTION]), ImageStatus.RESOLUTION_ERROR),
    ],
)
def test_status_for_report(report, expected):
    assert status_for_report(report) is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UnsupportedImageFormatError("x"), ImageStatus.FORMAT_ERROR),
        (ImageLoadError("x"), ImageStatus.FILE_ERROR),
        (CameraError("x"), ImageStatus.CAMERA_ERROR),
        (InvalidImageError(_report(errors=[IssueCode.RESOLUTION])), ImageStatus.RESOLUTION_ERROR),
    ],
)
def test_status_for_error(error, expected):
    assert status_for_error(error) is expected


def test_required_labels():
    assert ImageStatus.VALID.label == "VÁLIDA"
    assert ImageStatus.LOW_LIGHT.label == "ADVERTENCIA LUZ"
    assert ImageStatus.RESOLUTION_ERROR.label == "ERROR RESOLUCIÓN"


def test_only_valid_and_warning_states_are_usable():
    usable = {status for status in ImageStatus if status.is_usable}
    assert usable == {ImageStatus.VALID, ImageStatus.LOW_LIGHT}


def test_validator_reports_issue_codes():
    report = ImageValidator().validate(
        ImageData(pixels=np.zeros((100, 100), dtype=np.uint16), source="test")
    )
    assert report.has(IssueCode.RESOLUTION)
    assert report.has(IssueCode.CHANNELS)
    assert report.has(IssueCode.BIT_DEPTH)
    assert not report.has(IssueCode.LOW_LIGHT)


def test_invalid_image_error_keeps_the_rejected_image():
    image = ImageData(pixels=np.zeros((100, 100, 3), dtype=np.uint8), source="test")
    with pytest.raises(InvalidImageError) as info:
        ImageValidator().ensure_valid(image)
    assert info.value.image is image
