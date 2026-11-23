"""Measure a known displacement with the real pipeline (evidence for RF-005).

Usage:
    python -m scripts.measure_known_displacement [--folder CARPETA] [--step-cm 5.6]
                                                 [--count 7] [--pairs {all,consecutive}]
                                                 [--json RUTA]

The folder holds photos ``pos1`` ... ``posN`` (any accepted image extension)
of the same scene, taken from the same place, in which one object moved
``step_cm`` between consecutive photos. Every pair ``pos_i`` -> ``pos_j``
(``i < j``) is analyzed with one :class:`PipelineController`, so the models
load once, and the distance measured for the displaced object is compared with
the expected ``step_cm * (j - i)``.

The measured object is the only DESPLAZAMIENTO object of the report. A pair
with none is recorded as "sin medición" and one with several as "ambiguo",
listing every candidate; the script never chooses among them.

Nothing is written next to the photos and no image is saved. With ``--json``
the per-pair results and the summaries are also written to that file, which
must be outside ``data/samples``. Run it without network access, e.g. with
``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``: the models load from the
saved weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from config.image_settings import DEFAULT_EXTENSIONS
from config.paths import SAMPLES_DIR
from config.settings import AppSettings
from controllers.pipeline_controller import PipelineController
from models.comparison_report import ChangedObject, ChangeType

DEFAULT_STEP_CM = 5.6
DEFAULT_COUNT = 7
PHOTO_PREFIX = "pos"

MEASURED = "medido"
NO_MEASUREMENT = "sin medición"
AMBIGUOUS = "ambiguo"

WARNING = (
    "Advertencia: los pares reutilizan las mismas fotos, así que no son mediciones "
    "independientes; y el error no incluye la incertidumbre de la medición física "
    "del desplazamiento entre fotos."
)


class MeasurementError(Exception):
    """The photos or the arguments do not allow the measurement."""


@dataclass(frozen=True)
class Candidate:
    """A changed object found in a pair: its type, name and distance (cm, displacements only)."""

    change_type: str
    label: str
    distance_cm: Optional[float]


@dataclass(frozen=True)
class PairResult:
    """Expected and measured displacement for the pair ``pos{before}`` -> ``pos{after}``."""

    before: int
    after: int
    expected_cm: float
    status: str
    measured_cm: Optional[float] = None
    label: Optional[str] = None
    change_type: Optional[str] = None
    candidates: tuple[Candidate, ...] = ()
    other_objects: tuple[Candidate, ...] = ()
    intrinsics_source: Optional[str] = None
    focal_length_px: Optional[float] = None

    @property
    def k(self) -> int:
        return self.after - self.before

    @property
    def error_cm(self) -> Optional[float]:
        """Signed error: measured minus expected."""
        return None if self.measured_cm is None else self.measured_cm - self.expected_cm

    @property
    def abs_error_cm(self) -> Optional[float]:
        return None if self.error_cm is None else abs(self.error_cm)

    @property
    def rel_error_pct(self) -> Optional[float]:
        """Signed error as a percentage of the expected displacement."""
        return None if self.error_cm is None else 100.0 * self.error_cm / self.expected_cm

    def as_dict(self) -> dict:
        data = asdict(self)
        data.update(k=self.k, error_cm=self.error_cm, abs_error_cm=self.abs_error_cm, rel_error_pct=self.rel_error_pct)
        return data


@dataclass(frozen=True)
class Summary:
    """Error statistics over the measured pairs of a group (cm, and % for MAPE)."""

    pairs: int
    measured: int
    bias_cm: Optional[float] = None
    std_cm: Optional[float] = None
    mae_cm: Optional[float] = None
    mape_pct: Optional[float] = None
    max_abs_error_cm: Optional[float] = None


@dataclass(frozen=True)
class KRow:
    """Mean relative errors of the pairs whose photos are ``k`` steps apart."""

    k: int
    expected_cm: float
    pairs: int
    measured: int
    mean_rel_error_pct: Optional[float] = None
    mean_abs_rel_error_pct: Optional[float] = None


@dataclass
class Measurement:
    """Every pair and the summaries the script reports."""

    step_cm: float
    count: int
    pairs_mode: str
    results: list[PairResult] = field(default_factory=list)

    @property
    def consecutive(self) -> list[PairResult]:
        return [r for r in self.results if r.k == 1]

    def as_dict(self) -> dict:
        return {
            "step_cm": self.step_cm,
            "count": self.count,
            "pairs_mode": self.pairs_mode,
            "pairs": [r.as_dict() for r in self.results],
            "summary_consecutive": asdict(summarize(self.consecutive)),
            "summary_all": asdict(summarize(self.results)),
            "by_k": [asdict(row) for row in by_k(self.results, self.step_cm)],
            "warning": WARNING,
        }


# ------------------------------------------------------------------ pairs and expectations


def pair_indices(count: int, mode: str = "all") -> list[tuple[int, int]]:
    """``(i, j)`` pairs (1-based, ``i < j``) in a fixed order: by ``i``, then by ``j``."""
    if count < 2:
        raise MeasurementError("Se necesitan al menos dos fotos.")
    if mode == "consecutive":
        return [(i, i + 1) for i in range(1, count)]
    if mode == "all":
        return [(i, j) for i in range(1, count) for j in range(i + 1, count + 1)]
    raise MeasurementError(f"Modo de pares desconocido: {mode!r}.")


def expected_cm(step_cm: float, before: int, after: int) -> float:
    """Displacement between ``pos{before}`` and ``pos{after}`` when each step moves ``step_cm``."""
    return step_cm * (after - before)


def find_photos(folder: Path, count: int) -> list[Path]:
    """``pos1`` ... ``pos{count}`` in ``folder``; each must exist exactly once (any accepted extension)."""
    photos = []
    for n in range(1, count + 1):
        stem = f"{PHOTO_PREFIX}{n}"
        matches = sorted(
            p for p in folder.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in DEFAULT_EXTENSIONS
        )
        if not matches:
            raise MeasurementError(f"No se encontró {stem} en la carpeta indicada.")
        if len(matches) > 1:
            raise MeasurementError(f"Hay más de un archivo {stem} en la carpeta indicada.")
        photos.append(matches[0])
    return photos


# ------------------------------------------------------------------ one pair


def _candidate(obj: ChangedObject) -> Candidate:
    return Candidate(obj.change_type.value, obj.label.name, obj.distance_cm)


def classify_pair(
    objects: Sequence[ChangedObject],
    before: int,
    after: int,
    step_cm: float,
    intrinsics_source: Optional[str] = None,
    focal_length_px: Optional[float] = None,
) -> PairResult:
    """Result of one pair from the objects of its report; never picks among several displacements."""
    displaced = [obj for obj in objects if obj.change_type is ChangeType.DISPLACED]
    others = tuple(_candidate(obj) for obj in objects if obj.change_type is not ChangeType.DISPLACED)
    common = dict(
        before=before,
        after=after,
        expected_cm=expected_cm(step_cm, before, after),
        other_objects=others,
        intrinsics_source=intrinsics_source,
        focal_length_px=focal_length_px,
    )
    if len(displaced) == 1:
        obj = displaced[0]
        return PairResult(
            status=MEASURED,
            measured_cm=obj.distance_cm,
            label=obj.label.name,
            change_type=obj.change_type.value,
            candidates=(_candidate(obj),),
            **common,
        )
    status = NO_MEASUREMENT if not displaced else AMBIGUOUS
    return PairResult(status=status, candidates=tuple(_candidate(obj) for obj in displaced), **common)


# ------------------------------------------------------------------ statistics


def summarize(results: Sequence[PairResult]) -> Summary:
    """Bias, spread and size of the error over the pairs that have a measurement."""
    measured = [r for r in results if r.status == MEASURED and r.measured_cm is not None]
    if not measured:
        return Summary(pairs=len(results), measured=0)
    errors = [r.error_cm for r in measured]
    return Summary(
        pairs=len(results),
        measured=len(measured),
        bias_cm=statistics.fmean(errors),
        std_cm=statistics.stdev(errors) if len(errors) > 1 else None,
        mae_cm=statistics.fmean(abs(e) for e in errors),
        mape_pct=statistics.fmean(abs(r.rel_error_pct) for r in measured),
        max_abs_error_cm=max(abs(e) for e in errors),
    )


def by_k(results: Sequence[PairResult], step_cm: float) -> list[KRow]:
    """Mean signed and absolute relative error (%) for each separation ``k = j - i``."""
    rows = []
    for k in sorted({r.k for r in results}):
        group = [r for r in results if r.k == k]
        measured = [r.rel_error_pct for r in group if r.status == MEASURED and r.rel_error_pct is not None]
        rows.append(
            KRow(
                k=k,
                expected_cm=step_cm * k,
                pairs=len(group),
                measured=len(measured),
                mean_rel_error_pct=statistics.fmean(measured) if measured else None,
                mean_abs_rel_error_pct=statistics.fmean(abs(v) for v in measured) if measured else None,
            )
        )
    return rows


# ------------------------------------------------------------------ running


def run(
    controller: PipelineController,
    photos: Sequence[Path],
    step_cm: float,
    pairs: Sequence[tuple[int, int]],
) -> list[PairResult]:
    """Analyze each pair with ``controller``; every photo is loaded once."""
    loaded = {}
    results = []
    for before, after in pairs:
        for n in (before, after):
            if n not in loaded:
                loaded[n] = controller.acquisition.from_file(photos[n - 1])
        report = controller.analyze(loaded[before], loaded[after])
        source = report.intrinsics_source.value if report.intrinsics_source is not None else None
        results.append(classify_pair(report.objects, before, after, step_cm, source, report.focal_length_px))
    return results


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mide con el análisis real un desplazamiento conocido (RF-005).")
    parser.add_argument("--folder", type=Path, default=SAMPLES_DIR, help="Carpeta con pos1 ... posN.")
    parser.add_argument("--step-cm", type=float, default=DEFAULT_STEP_CM, help="Desplazamiento entre fotos consecutivas.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Número de fotos (pos1 ... posN).")
    parser.add_argument("--pairs", choices=["all", "consecutive"], default="all")
    parser.add_argument("--json", type=Path, default=None, help="Archivo JSON de salida, fuera de data/samples.")
    args = parser.parse_args(list(argv))
    if args.step_cm <= 0:
        parser.error("--step-cm debe ser positivo.")
    return args


def measure(
    args: argparse.Namespace, controller_factory: Callable[[], PipelineController] | None = None
) -> Measurement:
    """Run every pair requested by ``args``; the photos are only read."""
    if args.json is not None and _inside(args.json, SAMPLES_DIR):
        raise MeasurementError("El archivo JSON no puede quedar dentro de data/samples.")
    photos = find_photos(args.folder, args.count)
    pairs = pair_indices(args.count, args.pairs)
    factory = controller_factory or (lambda: PipelineController(AppSettings()))
    with factory() as controller:
        results = run(controller, photos, args.step_cm, pairs)
    return Measurement(args.step_cm, args.count, args.pairs, results)


# ------------------------------------------------------------------ output


def _num(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def format_report(measurement: Measurement) -> str:
    lines = [
        f"Desplazamiento conocido: {measurement.step_cm} cm por paso, {measurement.count} fotos, "
        f"pares: {measurement.pairs_mode} ({len(measurement.results)}).",
        "",
        "| Par | k | Esperado (cm) | Medido (cm) | Error (cm) | |Error| (cm) | Error rel. (%) | Estado | Etiqueta | Tipo | Cámara |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in measurement.results:
        camera = f"{r.intrinsics_source} f={_num(r.focal_length_px, 1)} px" if r.intrinsics_source else "—"
        lines.append(
            f"| pos{r.before}->pos{r.after} | {r.k} | {_num(r.expected_cm)} | {_num(r.measured_cm)} | "
            f"{_num(r.error_cm)} | {_num(r.abs_error_cm)} | {_num(r.rel_error_pct, 1)} | {r.status} | "
            f"{r.label or '—'} | {r.change_type or '—'} | {camera} |"
        )
        if r.status != MEASURED:
            found = "; ".join(f"{c.label} ({_num(c.distance_cm)} cm)" for c in r.candidates) or "ninguno"
            others = "; ".join(f"{c.label}: {c.change_type}" for c in r.other_objects) or "ninguno"
            lines.append(f"|   candidatos: {found}; otros cambios: {others} | | | | | | | | | | |")

    for title, group in (("consecutivos", measurement.consecutive), ("todos", measurement.results)):
        s = summarize(group)
        lines += [
            "",
            f"Resumen ({title}): n={s.measured} de {s.pairs}, sesgo={_num(s.bias_cm)} cm, "
            f"desviación estándar={_num(s.std_cm)} cm, MAE={_num(s.mae_cm)} cm, "
            f"MAPE={_num(s.mape_pct, 1)} %, error máximo={_num(s.max_abs_error_cm)} cm",
        ]
    lines += ["", "| k | Esperado (cm) | Pares | Medidos | Error rel. medio (%) | Error rel. absoluto medio (%) |", "|---|---|---|---|---|---|"]
    for row in by_k(measurement.results, measurement.step_cm):
        lines.append(
            f"| {row.k} | {_num(row.expected_cm)} | {row.pairs} | {row.measured} | "
            f"{_num(row.mean_rel_error_pct, 1)} | {_num(row.mean_abs_rel_error_pct, 1)} |"
        )
    lines += ["", WARNING]
    return "\n".join(lines)


def main(argv: Sequence[str], controller_factory: Callable[[], PipelineController] | None = None) -> int:
    args = parse_args(argv)
    try:
        measurement = measure(args, controller_factory)
    except MeasurementError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(format_report(measurement))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(measurement.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
