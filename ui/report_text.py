"""Plain-language wording of an analysis report, shared by the screen and the exported files."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from models.camera_model import IntrinsicsSource
from models.comparison_report import BoundingBox, ChangedObject, ChangeType, ComparisonReport
from models.session_history import HistoryEntry

NO_CHANGES_MESSAGE = "No se detectaron cambios en la escena"

HISTORY_EMPTY_MESSAGE = (
    "Aún no hay comparaciones en esta sesión. Cuando analice un par de fotos, aparecerá aquí."
)

_CAMERA_NOTES: dict[IntrinsicsSource, str] = {
    IntrinsicsSource.CALIBRATED: "Las distancias se calcularon con la calibración de la cámara.",
    IntrinsicsSource.EXIF: "Las distancias se calcularon con los datos de la cámara guardados en la foto.",
    IntrinsicsSource.FIELD_OF_VIEW: (
        "Las distancias se calcularon con una estimación estándar de la cámara, "
        "porque la foto no trae sus datos; pueden variar unos centímetros."
    ),
}

# Friendly name of each pipeline stage (keys of controllers.STAGES).
STAGE_DESCRIPTIONS: dict[str, str] = {
    "acquisition": "Revisando las fotos",
    "loading": "Preparando el sistema (solo la primera vez)",
    "depth": "Calculando las distancias en cada foto",
    "alignment": "Haciendo coincidir las dos fotos",
    "segmentation": "Reconociendo los objetos",
    "comparison": "Buscando los cambios",
    "rendering": "Preparando el resultado",
}

_STAGE_NAMES: dict[str, str] = {
    "acquisition": "revisión de las fotos",
    "loading": "preparación inicial",
    "depth": "distancias",
    "alignment": "coincidencia de las fotos",
    "segmentation": "reconocimiento de objetos",
    "comparison": "búsqueda de cambios",
    "rendering": "dibujo del resultado",
}


def headline(report: ComparisonReport) -> str:
    count = len(report.objects)
    if count == 0:
        return NO_CHANGES_MESSAGE
    return "Se detectó 1 cambio en la escena" if count == 1 else f"Se detectaron {count} cambios en la escena"


def history_title(entry: HistoryEntry) -> str:
    """Line that identifies a comparison of the session, e.g. "#2 · 14:05 · antes.jpg → después.jpg"."""
    return f"#{entry.number} · {entry.created_at:%H:%M} · {entry.before_name} → {entry.after_name}"


def history_summary(entry: HistoryEntry) -> str:
    """What that comparison found, in the words of the report's headline."""
    return headline(entry.report)


def category(obj: ChangedObject) -> str:
    """Object category with an initial capital, e.g. "Caja" or "Objeto no identificado"."""
    return obj.label.display_name


def distance_text(obj: ChangedObject) -> Optional[str]:
    distance = obj.distance_cm
    return None if distance is None else f"Desplazamiento: {distance:.1f} cm"


def describe(obj: ChangedObject) -> str:
    """One or two sentences that explain the change and where to find it in the image."""
    if obj.change_type is ChangeType.DISPLACED:
        moved = f"se movió unos {obj.distance_cm:.1f} cm" if obj.distance_cm is not None else "se movió"
        return (
            f"Este objeto cambió de lugar: {moved}. Su posición anterior está marcada "
            "en naranja y la actual en verde, unidas por una flecha."
        )
    if obj.change_type is ChangeType.REMOVED:
        return "Este objeto ya no está en la escena. El lugar donde estaba está marcado en rojo."
    return "Este objeto no estaba en la foto de referencia. Está marcado en azul."


def camera_note(report: ComparisonReport) -> str:
    """Which camera parameters the distances used, only when some distance was measured."""
    if report.intrinsics_source is None or not report.objects_of(ChangeType.DISPLACED):
        return ""
    return _CAMERA_NOTES[report.intrinsics_source]


def stage_description(stage: str) -> str:
    return STAGE_DESCRIPTIONS.get(stage, "Analizando")


def timing_summary(report: ComparisonReport) -> str:
    """Total analysis time; the one-time preparation of the system is reported apart."""
    if not report.stage_seconds:
        return ""
    loading = report.stage_seconds.get("loading", 0.0)
    text = f"Análisis completado en {report.total_seconds - loading:.2f} s."
    if loading >= 0.05:  # Too short to mention when the models were nearly ready.
        text += f" La preparación inicial del sistema tomó {loading:.1f} s más; solo ocurre en el primer análisis."
    return text


def timing_details(report: ComparisonReport) -> str:
    """Time of each stage, e.g. "distancias 0.07 s, búsqueda de cambios 0.10 s"."""
    return ", ".join(
        f"{_STAGE_NAMES.get(stage, stage)} {seconds:.2f} s" for stage, seconds in report.stage_seconds.items()
    )


def to_text(report: ComparisonReport, created_at: Optional[datetime] = None) -> str:
    """Summary of the report as plain text, for people."""
    created_at = created_at or datetime.now()
    lines = [
        "Reporte de cambios en la escena",
        f"Fecha: {created_at:%d/%m/%Y %H:%M}",
        "",
        f"{headline(report)}.",
    ]
    for number, obj in enumerate(report.objects, start=1):
        lines += ["", f"{number}. {obj.change_type.title}: {category(obj)}"]
        distance = distance_text(obj)
        if distance:
            lines.append(f"   {distance}")
        lines.append(f"   {describe(obj)}")
    note = camera_note(report)
    if note:
        lines += ["", note]
    if report.warnings:
        lines += ["", "Advertencias:"]
        lines += [f"- {warning}" for warning in report.warnings]
    timing = timing_summary(report)
    if timing:
        lines += ["", timing, f"Detalle: {timing_details(report)}."]
    return "\n".join(lines) + "\n"


def to_dict(report: ComparisonReport, created_at: Optional[datetime] = None) -> dict[str, Any]:
    """Summary of the report as JSON-serializable data, for other programs."""
    created_at = created_at or datetime.now()
    return {
        "fecha": created_at.isoformat(timespec="seconds"),
        "hay_cambios": report.has_changes,
        "resumen": headline(report),
        "cambios": [
            {
                "numero": number,
                "tipo": obj.change_type.title,
                "categoria": category(obj),
                "distancia_cm": None if obj.distance_cm is None else round(obj.distance_cm, 1),
                "descripcion": describe(obj),
                "zona_antes": _box(obj.bbox_before),
                "zona_despues": _box(obj.bbox_after),
            }
            for number, obj in enumerate(report.objects, start=1)
        ],
        "advertencias": list(report.warnings),
        "camara": {
            "origen_parametros": None if report.intrinsics_source is None else report.intrinsics_source.value,
            "distancia_focal_px": None if report.focal_length_px is None else round(report.focal_length_px, 1),
        },
        "tiempos_s": {stage: round(seconds, 3) for stage, seconds in report.stage_seconds.items()},
        "tiempo_total_s": round(report.total_seconds, 3),
    }


def _box(box: Optional[BoundingBox]) -> Optional[dict[str, int]]:
    if box is None:
        return None
    return {"x": box.x, "y": box.y, "ancho": box.width, "alto": box.height}
