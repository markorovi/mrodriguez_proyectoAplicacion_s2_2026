"""Calibrate a camera with photos of a printed chessboard.

Usage:
    # 1. (Optional) take the photos with a connected camera, pressing Enter for each one
    python -m scripts.calibrate_camera FOLDER --capture 15 [--camera 0]

    # 2. Calibrate from the photos in FOLDER and save the profile in a settings file
    python -m scripts.calibrate_camera FOLDER [--columns 9] [--rows 6] [--square-mm 25]
                                              [--output settings.json]

Use the resulting file with ``python -m ui settings.json``. Print a board of
10 x 7 squares (9 x 6 inner corners), stick it to something flat and measure
the side of one square. Take 15-20 photos with the same camera, resolution
and zoom used for the analysis, showing the whole board at different angles,
distances and positions (corners and edges of the frame included).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config.capture_settings import CaptureSettings
from config.image_settings import ImageSettings
from models.exceptions import SceneDiffError
from services.capture.acquisition_service import ImageAcquisitionService
from services.capture.file_image_loader import FileImageLoader
from utils.camera_calibration import CalibrationResult, calibrate_camera
from utils.camera_model import horizontal_fov_deg
from utils.image_validator import ImageValidator

GOOD_RMS_PX = 0.5
DOUBTFUL_RMS_PX = 1.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibración de la cámara con un tablero de ajedrez.")
    parser.add_argument("folder", type=Path, help="Carpeta con las fotos del tablero.")
    parser.add_argument("--columns", type=int, default=9, help="Esquinas internas por fila (por defecto 9).")
    parser.add_argument("--rows", type=int, default=6, help="Esquinas internas por columna (por defecto 6).")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Lado de un cuadro, en mm.")
    parser.add_argument("--output", type=Path, default=Path("settings.json"), help="Archivo de configuración.")
    parser.add_argument("--capture", type=int, default=0, metavar="N", help="Tomar N fotos con la cámara.")
    parser.add_argument("--camera", type=int, default=0, help="Número de la cámara para --capture.")
    return parser.parse_args(argv)


def capture_photos(folder: Path, count: int, camera_index: int) -> None:
    """Save ``count`` frames of the camera in ``folder``, one each time the user presses Enter."""
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    service = ImageAcquisitionService(capture_settings=CaptureSettings(camera_index=camera_index))
    print(f"Se tomarán {count} fotos con la cámara {camera_index}. Mueva el tablero entre una y otra.")
    for number in range(1, count + 1):
        input(f"[{number}/{count}] Coloque el tablero y pulse Enter...")
        image = service.from_camera().image
        path = folder / f"tablero_{number:02d}.png"
        Image.fromarray(image.pixels).save(path)
        print(f"    Guardada {path.name} ({image.width}x{image.height})")


def load_photos(folder: Path) -> list[tuple[str, np.ndarray]]:
    extensions = ImageSettings().allowed_extensions
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
    validator = ImageValidator()
    return [(path.name, FileImageLoader(path, validator).acquire().pixels) for path in paths]


def save_profile(result: CalibrationResult, output: Path) -> None:
    """Write the calibrated intrinsics under ``"camera"``, keeping the rest of an existing settings file."""
    settings = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    k = result.intrinsics
    settings["camera"] = {
        "fx": round(k.fx, 2),
        "fy": round(k.fy, 2),
        "cx": round(k.cx, 2),
        "cy": round(k.cy, 2),
        "reference_width": k.reference_width,
        "reference_height": k.reference_height,
    }
    output.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_result(result: CalibrationResult) -> None:
    k = result.intrinsics
    print(f"Fotos usadas:        {len(result.used)}")
    if result.rejected:
        print(f"Sin tablero visible: {len(result.rejected)} ({', '.join(result.rejected)})")
    print(f"Resolución:          {k.reference_width}x{k.reference_height}")
    print(f"fx, fy:              {k.fx:.1f}, {k.fy:.1f} px")
    print(f"cx, cy:              {k.cx:.1f}, {k.cy:.1f} px")
    print(f"Campo de visión:     {horizontal_fov_deg(k):.1f}° horizontal")
    print(f"Distorsión:          {', '.join(f'{d:.3f}' for d in result.distortion)}")
    quality = (
        "buena" if result.rms_error_px < GOOD_RMS_PX
        else "aceptable" if result.rms_error_px < DOUBTFUL_RMS_PX
        else "dudosa: repita con fotos más nítidas y más ángulos"
    )
    print(f"Error de calibración: {result.rms_error_px:.3f} px ({quality})")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.capture > 0:
            capture_photos(args.folder, args.capture, args.camera)
        if not args.folder.is_dir():
            print(f"No existe la carpeta '{args.folder}'.")
            return 1
        photos = load_photos(args.folder)
        result = calibrate_camera(photos, (args.columns, args.rows), args.square_mm)
    except SceneDiffError as error:
        print(f"Error: {error}")
        return 1

    print_result(result)
    save_profile(result, args.output)
    print(f"Perfil guardado en '{args.output}'. Úselo con: python -m ui {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
