"""Saving an analysis report: the annotated image plus a text or JSON summary."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Union

from models.comparison_report import ComparisonReport
from ui import report_text
from ui.qt_image import to_qimage

SUMMARY_FORMATS = (".txt", ".json")

# A name typed with another image extension still gets a PNG ("foto.jpg" -> "foto.png").
_OTHER_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif")


class ReportExportError(Exception):
    """The report files could not be written; the message is meant for the user."""


def export_report(report: ComparisonReport, summary_path: Union[str, Path]) -> list[Path]:
    """Write the summary (``.txt`` or ``.json``) and, next to it, the annotated image as PNG.

    Returns the written paths: the summary first, then the image.
    """
    summary_path = Path(summary_path)
    if summary_path.suffix.lower() not in SUMMARY_FORMATS:
        summary_path = summary_path.with_suffix(".txt")
    image_path = summary_path.with_suffix(".png")

    created_at = datetime.now()
    if summary_path.suffix.lower() == ".json":
        content = json.dumps(report_text.to_dict(report, created_at), ensure_ascii=False, indent=2) + "\n"
    else:
        content = report_text.to_text(report, created_at)

    try:
        summary_path.write_text(content, encoding="utf-8")
        saved = to_qimage(report.annotated_image).save(str(image_path), "PNG")
    except OSError as exc:
        raise ReportExportError(
            f"No se pudo guardar el reporte en '{summary_path.parent}'. "
            "Verifique que la carpeta exista y que tenga permiso para escribir en ella."
        ) from exc
    if not saved:
        raise ReportExportError(f"No se pudo guardar la imagen del resultado en '{image_path.name}'.")
    return [summary_path, image_path]


def save_annotated_image(report: ComparisonReport, path: Union[str, Path]) -> Path:
    """Write only the annotated image, as PNG at its original resolution; returns the written path."""
    image_path = Path(path)
    if image_path.suffix.lower() in _OTHER_IMAGE_SUFFIXES:
        image_path = image_path.with_suffix(".png")
    elif image_path.suffix.lower() != ".png":
        image_path = image_path.with_name(image_path.name + ".png")
    if not to_qimage(report.annotated_image).save(str(image_path), "PNG"):
        raise ReportExportError(
            f"No se pudo guardar la imagen en '{image_path.parent}'. "
            "Verifique que la carpeta exista y que tenga permiso para escribir en ella."
        )
    return image_path
