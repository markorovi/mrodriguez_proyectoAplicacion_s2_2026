"""Chessboard calibration recovers the parameters of a virtual camera."""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from config import CameraIntrinsics, load_settings
from models import CalibrationError
from scripts import calibrate_camera as cli
from utils.camera_calibration import board_points, calibrate_camera, find_chessboard_corners

PATTERN = (9, 6)  # inner corners of a 10 x 7 board
SQUARE_MM = 25.0
WIDTH, HEIGHT = 960, 540
TRUE_K = np.array([[820.0, 0, 470.0], [0, 820.0, 280.0], [0, 0, 1]])
TEXTURE_PX_PER_MM = 4

# (rotation about x, rotation about y, rotation about z) in degrees and (tx, ty, tz) in mm.
POSES = [
    ((0, 0, 0), (-110, -70, 600)),
    ((20, 0, 5), (-130, -60, 650)),
    ((-20, 10, 0), (-90, -90, 620)),
    ((0, 25, -5), (-150, -80, 700)),
    ((0, -25, 8), (-60, -60, 680)),
    ((15, 20, 0), (-200, -120, 750)),
    ((-15, -20, 3), (-20, -20, 720)),
    ((25, -10, -8), (-120, -150, 800)),
]


def _render_view(rotation_deg, translation_mm) -> np.ndarray:
    """RGB photo of the board taken by the virtual camera at the given pose."""
    squares_x, squares_y = PATTERN[0] + 1, PATTERN[1] + 1
    margin = 1  # white quiet zone of one square around the board
    s = int(SQUARE_MM * TEXTURE_PX_PER_MM)
    texture = np.full(((squares_y + 2 * margin) * s, (squares_x + 2 * margin) * s), 255, np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y, x = (row + margin) * s, (col + margin) * s
                texture[y:y + s, x:x + s] = 0

    rotation, _ = cv2.Rodrigues(np.radians(np.array(rotation_deg, float)))
    # Texture pixel -> board mm, with the first inner corner at the origin.
    to_board = np.array([[1 / TEXTURE_PX_PER_MM, 0, -(margin + 1) * SQUARE_MM],
                         [0, 1 / TEXTURE_PX_PER_MM, -(margin + 1) * SQUARE_MM],
                         [0, 0, 1]])
    homography = TRUE_K @ np.column_stack([rotation[:, 0], rotation[:, 1], translation_mm]) @ to_board
    gray = cv2.warpPerspective(texture, homography, (WIDTH, HEIGHT), flags=cv2.INTER_AREA, borderValue=200)
    return np.dstack([gray] * 3)


@pytest.fixture(scope="module")
def views() -> list[tuple[str, np.ndarray]]:
    return [(f"view_{i}.png", _render_view(r, t)) for i, (r, t) in enumerate(POSES)]


def test_board_corners_are_found_in_every_view(views):
    for name, rgb in views:
        corners = find_chessboard_corners(rgb, PATTERN)
        assert corners is not None, name
        assert corners.shape == (PATTERN[0] * PATTERN[1], 1, 2)


def test_calibration_recovers_the_virtual_camera(views):
    result = calibrate_camera(views, PATTERN, SQUARE_MM)
    k = result.intrinsics

    assert len(result.used) == len(views) and not result.rejected
    assert (k.reference_width, k.reference_height) == (WIDTH, HEIGHT)
    assert k.fx == pytest.approx(TRUE_K[0, 0], rel=0.02)
    assert k.fy == pytest.approx(TRUE_K[1, 1], rel=0.02)
    assert k.cx == pytest.approx(TRUE_K[0, 2], abs=10)
    assert k.cy == pytest.approx(TRUE_K[1, 2], abs=10)
    assert result.rms_error_px < 0.5


def test_photos_without_the_board_are_reported_and_skipped(views):
    blank = ("blank.png", np.full((HEIGHT, WIDTH, 3), 128, np.uint8))
    result = calibrate_camera([*views, blank], PATTERN, SQUARE_MM)
    assert result.rejected == ("blank.png",)


def test_too_few_views_raise_a_user_message(views):
    with pytest.raises(CalibrationError, match="se necesitan al menos 5"):
        calibrate_camera(views[:3], PATTERN, SQUARE_MM)


def test_mixed_resolutions_are_rejected(views):
    small = ("small.png", cv2.resize(views[0][1], (640, 360)))
    with pytest.raises(CalibrationError, match="tamaños distintos"):
        calibrate_camera([*views, small], PATTERN, SQUARE_MM)


def test_board_points_are_spaced_by_the_square_size():
    points = board_points((3, 2), 25.0)
    assert points.shape == (6, 3)
    assert points[1].tolist() == [25.0, 0.0, 0.0]
    assert points[3].tolist() == [0.0, 25.0, 0.0]


def test_command_line_saves_a_profile_the_application_can_load(tmp_path, views):
    folder = tmp_path / "tablero cámara"
    folder.mkdir()
    for name, rgb in views:
        Image.fromarray(rgb).save(folder / name)
    output = tmp_path / "settings.json"
    output.write_text(json.dumps({"capture": {"camera_index": 1}}), "utf-8")

    assert cli.main([str(folder), "--output", str(output)]) == 0

    settings = load_settings(output)
    assert settings.capture.camera_index == 1  # existing keys are kept
    assert isinstance(settings.camera, CameraIntrinsics)
    assert settings.camera.fx == pytest.approx(TRUE_K[0, 0], rel=0.02)
    assert (settings.camera.reference_width, settings.camera.reference_height) == (WIDTH, HEIGHT)


def test_command_line_reports_failure_without_writing(tmp_path, capsys):
    folder = tmp_path / "vacía"
    folder.mkdir()
    output = tmp_path / "settings.json"

    assert cli.main([str(folder), "--output", str(output)]) == 1
    assert "No se recibieron fotos" in capsys.readouterr().out
    assert not output.exists()
