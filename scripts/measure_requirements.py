"""Reproducible evidence for the performance and portability requirements.

Covers RR-001 (a pair in at most 15 s on the GPU), RR-002 (640x480 input and a
sound depth map), RF-009 (inference on the GPU, seen by a hardware monitor),
RF-010 (CPU fallback with a notice to the user) and RA-002 (the same analysis
completes with and without a GPU).

Usage:
    python -m scripts.measure_requirements                    # GPU and CPU, writes docs/evidencia/
    python -m scripts.measure_requirements --device cuda      # only one device
    python -m scripts.measure_requirements --out-dir OTHER    # write the evidence elsewhere

Each device is measured in a fresh process, so the first analysis is a real
cold start that includes loading the models. The CPU process is started with
CUDA_VISIBLE_DEVICES=-1 and the automatic device preference, as on a machine
without a graphics card (an empty value does not hide the GPU on Windows: CUDA
treats it as unset). The CPU process also records whether CUDA was really
hidden, so a run that silently used the GPU cannot pass as a CPU run. The models are loaded from the local cache only; the
measurement never reaches the network. The generated documents contain only
relative paths and pair names: no user name, no absolute paths, no pixels and
no names of other processes using the GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from config.paths import PROJECT_ROOT, SAMPLES_DIR

BUDGET_SECONDS = 15.0
MIN_GPU_MEMORY_INCREASE_MIB = 100.0
SAMPLE_INTERVAL_S = 0.2
BASELINE_SAMPLES = 5
STAGE_ORDER = ("acquisition", "loading", "depth", "alignment", "segmentation", "comparison", "rendering")
STAGE_NAMES = {
    "acquisition": "Adquisición",
    "loading": "Carga de modelos",
    "depth": "Profundidad",
    "alignment": "Alineación",
    "segmentation": "Segmentación",
    "comparison": "Comparación",
    "rendering": "Dibujo",
}
DEFAULT_OUT_DIR = Path("docs") / "evidencia"
REGENERATE_COMMAND = "python -m scripts.measure_requirements"
SMI_QUERY = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
# An invalid device index hides every GPU; an empty value is ignored by CUDA on Windows.
HIDE_ALL_GPUS = "-1"

# (pair name, before, after, expected result recorded in the progress log)
SYNTHETIC_PAIRS = (
    ("sintético: desplazamiento", "scene_before.png", "scene_after_moved.png", "un desplazamiento de unos 51 cm"),
    ("sintético: retiro", "scene_before.png", "scene_after_removed.png", "un objeto retirado"),
    ("sintético: poca luz", "scene_before.png", "scene_low_light.png", "sin cambios, con aviso de poca luz"),
)
REAL_PAIR_SUFFIXES = ("", "_2", "_3")
WARM_RUNS = {"cuda": {"sintetico": 10, "real": 3}, "cpu": {"sintetico": 3, "real": 0}}


# ------------------------------------------------------------------ statistics


def stats(values: Sequence[float]) -> Optional[dict[str, float]]:
    """Mean, minimum and maximum of ``values``; ``None`` when there are none."""
    if not values:
        return None
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def stage_stats(stage_runs: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Statistics of each stage over several analyses (stages absent from a run count as missing)."""
    stages = [stage for stage in STAGE_ORDER if any(stage in run for run in stage_runs)]
    return {stage: stats([run[stage] for run in stage_runs if stage in run]) for stage in stages}


# ---------------------------------------------------------------- nvidia-smi


def parse_smi_samples(text: str) -> list[tuple[float, float]]:
    """``(utilization %, memory MiB)`` per GPU line of the nvidia-smi CSV query; malformed lines are ignored."""
    samples = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            samples.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return samples


def query_smi(run: Callable[..., Any] = subprocess.run) -> Optional[tuple[float, float]]:
    """One reading of the first GPU, or ``None`` when nvidia-smi is not available."""
    try:
        result = run(SMI_QUERY, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    samples = parse_smi_samples(result.stdout)
    return samples[0] if samples else None


def baseline(readings: Sequence[Optional[tuple[float, float]]]) -> Optional[dict[str, float]]:
    """Median idle utilization and memory over the valid readings; ``None`` if there are none."""
    valid = [reading for reading in readings if reading is not None]
    if not valid:
        return None
    return {
        "util_median": statistics.median(r[0] for r in valid),
        "mem_median_mib": statistics.median(r[1] for r in valid),
        "samples": len(valid),
    }


class GpuMonitor:
    """Samples nvidia-smi on a background thread and keeps the maxima."""

    def __init__(self, query: Callable[[], Optional[tuple[float, float]]] = query_smi,
                 interval: float = SAMPLE_INTERVAL_S) -> None:
        self._query = query
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[tuple[float, float]] = []

    def __enter__(self) -> GpuMonitor:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            reading = self._query()
            if reading is not None:
                self.samples.append(reading)
            self._stop.wait(self._interval)

    @property
    def max_util(self) -> Optional[float]:
        return max((s[0] for s in self.samples), default=None)

    @property
    def max_mem_mib(self) -> Optional[float]:
        return max((s[1] for s in self.samples), default=None)


# ------------------------------------------------------------------ evaluation


@dataclass(frozen=True)
class Verdict:
    requirement: str
    status: str  # "CUMPLE", "NO CUMPLE" or "NO EVALUADO"
    evidence: str


def _synthetic(pairs: Sequence[dict]) -> list[dict]:
    return [pair for pair in pairs if pair.get("kind") == "sintetico"]


def _completed_synthetic(child: Optional[dict]) -> bool:
    if not child:
        return False
    synthetic = _synthetic(child.get("pairs", []))
    return len(synthetic) == len(SYNTHETIC_PAIRS) and all(not pair.get("error") for pair in synthetic)


def evaluate_rr001(cuda: Optional[dict]) -> Verdict:
    if not cuda:
        return Verdict("RR-001", "NO EVALUADO", "No se midió en GPU.")
    synthetic = _synthetic(cuda.get("pairs", []))
    if not _completed_synthetic(cuda):
        return Verdict("RR-001", "NO CUMPLE", "Alguna escena sintética no se completó en GPU.")
    cold = max(pair["cold"]["total"] for pair in synthetic)
    warm = max(pair["warm"]["total"]["max"] for pair in synthetic if pair["warm"]["total"])
    ok = cold <= BUDGET_SECONDS and warm <= BUDGET_SECONDS
    return Verdict(
        "RR-001",
        "CUMPLE" if ok else "NO CUMPLE",
        f"Máximo en frío {cold:.2f} s (incluye la carga de modelos) y en caliente {warm:.2f} s, "
        f"frente al límite de {BUDGET_SECONDS:.0f} s (3 escenas sintéticas 640x480).",
    )


def evaluate_rf009(cuda: Optional[dict]) -> Verdict:
    if not cuda:
        return Verdict("RF-009", "NO EVALUADO", "No se midió en GPU.")
    gpu = cuda.get("gpu", {})
    torch_mib = gpu.get("torch_max_allocated_mib") or 0.0
    if not gpu.get("smi_available"):
        ok = torch_mib > 0
        return Verdict(
            "RF-009",
            "CUMPLE" if ok else "NO CUMPLE",
            f"PyTorch reservó como máximo {torch_mib:.0f} MiB en la GPU; monitoreo externo no disponible.",
        )
    increase = gpu.get("mem_increase_mib") or 0.0
    ok = increase >= MIN_GPU_MEMORY_INCREASE_MIB and torch_mib > 0
    base = gpu.get("baseline") or {}
    return Verdict(
        "RF-009",
        "CUMPLE" if ok else "NO CUMPLE",
        f"nvidia-smi: memoria de la GPU de {base.get('mem_median_mib', 0):.0f} MiB en reposo a "
        f"{gpu.get('max_mem_mib', 0):.0f} MiB durante el análisis (+{increase:.0f} MiB; umbral "
        f"{MIN_GPU_MEMORY_INCREASE_MIB:.0f} MiB); utilización máxima {gpu.get('max_util', 0):.0f} % "
        f"(reposo {base.get('util_median', 0):.0f} %); PyTorch reservó como máximo {torch_mib:.0f} MiB.",
    )


def evaluate_rf010(cpu: Optional[dict]) -> Verdict:
    if not cpu:
        return Verdict("RF-010", "NO EVALUADO", "No se midió sin GPU.")
    resolved = cpu.get("resolved_devices", {}).get("depth", "")
    completed = _completed_synthetic(cpu)
    notice = cpu.get("cpu_notice_in_warnings", False)
    gpu_hidden = cpu.get("cuda_available") is False
    ok = resolved.upper() == "CPU" and completed and notice and gpu_hidden
    return Verdict(
        "RF-010",
        "CUMPLE" if ok else "NO CUMPLE",
        f"GPU oculta al proceso: {'sí' if gpu_hidden else 'no'}; "
        f"dispositivo resuelto: {resolved or 'desconocido'}; escenas sintéticas completadas: "
        f"{'sí' if completed else 'no'}; aviso en las advertencias del reporte: {'sí' if notice else 'no'}"
        + (f" («{cpu.get('cpu_notice')}»)." if notice else "."),
    )


def evaluate_ra002(cuda: Optional[dict], cpu: Optional[dict]) -> Verdict:
    if not cuda or not cpu:
        return Verdict("RA-002", "NO EVALUADO", "Requiere medir en GPU y sin GPU.")
    gpu_hidden = cpu.get("cuda_available") is False
    ok = _completed_synthetic(cuda) and _completed_synthetic(cpu) and gpu_hidden
    if ok:
        evidence = "Las 3 escenas sintéticas se completaron en GPU y en el entorno sin GPU, en procesos separados."
    elif not gpu_hidden:
        evidence = "El proceso «sin GPU» todavía veía la GPU, así que no prueba un entorno puramente CPU."
    else:
        evidence = "Las 3 escenas sintéticas no se completaron en los dos entornos."
    return Verdict("RA-002", "CUMPLE" if ok else "NO CUMPLE", evidence)


def depth_checks_pass(depth: Optional[dict]) -> bool:
    """Shape 480x640, every value finite and a positive, non-degenerate range."""
    if not depth:
        return False
    return (
        tuple(depth.get("input_size", ())) == (640, 480)
        and tuple(depth.get("shape", ())) == (480, 640)
        and depth.get("finite") is True
        and depth.get("min", 0) > 0
        and depth.get("max", 0) > depth.get("min", 0)
    )


def evaluate_rr002(children: Sequence[dict]) -> Verdict:
    checked = [
        depth
        for child in children
        for pair in _synthetic(child.get("pairs", []))
        for depth in pair.get("depth", [])
    ]
    if not checked:
        return Verdict("RR-002", "NO EVALUADO", "No hay mapas de profundidad de las escenas sintéticas.")
    ok = all(depth_checks_pass(depth) for depth in checked)
    low = min(d.get("min", 0) for d in checked)
    high = max(d.get("max", 0) for d in checked)
    return Verdict(
        "RR-002",
        "CUMPLE" if ok else "NO CUMPLE",
        f"{len(checked)} mapas de profundidad de entradas 640x480: forma 480x640, valores finitos y "
        f"positivos, entre {low:.2f} m y {high:.2f} m.",
    )


def evaluate(results: dict) -> list[Verdict]:
    cuda, cpu = results.get("cuda"), results.get("cpu")
    return [
        evaluate_rr001(cuda),
        evaluate_rr002([child for child in (cuda, cpu) if child]),
        evaluate_rf009(cuda),
        evaluate_rf010(cpu),
        evaluate_ra002(cuda, cpu),
    ]


# -------------------------------------------------------------------- markdown


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _stage_table(pairs: Sequence[dict]) -> list[str]:
    lines = [
        "| Par | Resolución | Análisis | Total (s) | " + " | ".join(STAGE_NAMES[s] for s in STAGE_ORDER) + " |",
        "|---|---|---|---|" + "---|" * len(STAGE_ORDER),
    ]
    for pair in pairs:
        size = "x".join(str(v) for v in pair.get("size", ()))
        if pair.get("error"):
            lines.append(f"| {pair['name']} | {size} | error | — | " + " | ".join("—" for _ in STAGE_ORDER) + " |")
            continue
        cold = pair["cold"]
        lines.append(
            f"| {pair['name']} | {size} | frío{' (con carga)' if cold.get('includes_loading') else ''} | "
            f"{_fmt(cold['total'])} | " + " | ".join(_fmt(cold["stages"].get(s)) for s in STAGE_ORDER) + " |"
        )
        warm = pair.get("warm", {})
        if warm.get("total"):
            total = warm["total"]
            lines.append(
                f"| {pair['name']} | {size} | caliente (media de {warm['count']}; mín–máx) | "
                f"{_fmt(total['mean'])} ({_fmt(total['min'])}–{_fmt(total['max'])}) | "
                + " | ".join(_fmt((warm["stages"].get(s) or {}).get("mean")) for s in STAGE_ORDER)
                + " |"
            )
    return lines


def render_markdown(results: dict, verdicts: Sequence[Verdict]) -> str:
    meta = results.get("meta", {})
    lines = [
        "# Evidencia de rendimiento y portabilidad",
        "",
        f"Generado el {meta.get('date', '—')} sobre el commit `{meta.get('commit', '—')}`"
        + (" (con cambios locales sin commit)" if meta.get("dirty") else "")
        + ".",
        "",
        f"Para regenerarlo: `{REGENERATE_COMMAND}`",
        "",
        "> Medido en una sola máquina, con la GPU compartida con el escritorio: otras aplicaciones "
        "usan parte de su memoria y de su capacidad. Los tiempos son indicativos de este equipo.",
        "",
        "## Requisitos",
        "",
        "| Requisito | Resultado | Dato que lo sustenta |",
        "|---|---|---|",
    ]
    lines += [f"| {v.requirement} | **{v.status}** | {v.evidence} |" for v in verdicts]

    env = results.get("environment", {})
    lines += ["", "## Entorno", "", "| Elemento | Valor |", "|---|---|"]
    labels = {
        "gpu": "GPU", "driver": "Controlador NVIDIA", "torch": "PyTorch", "cuda": "CUDA (PyTorch)",
        "python": "Python", "os": "Sistema operativo", "cpu": "Procesador",
    }
    lines += [f"| {label} | {env.get(key) or '—'} |" for key, label in labels.items()]

    cuda = results.get("cuda")
    if cuda:
        gpu = cuda.get("gpu", {})
        base = gpu.get("baseline")
        lines += ["", "## GPU en reposo y durante la medición (nvidia-smi)", ""]
        if gpu.get("smi_available") and base:
            lines += [
                "| Medida | Reposo (mediana de "
                f"{base['samples']} lecturas) | Máximo durante la medición |",
                "|---|---|---|",
                f"| Utilización | {_fmt(base['util_median'], 0)} % | {_fmt(gpu.get('max_util'), 0)} % |",
                f"| Memoria usada | {_fmt(base['mem_median_mib'], 0)} MiB | {_fmt(gpu.get('max_mem_mib'), 0)} MiB |",
                "",
                f"Aumento de memoria sobre el reposo: {_fmt(gpu.get('mem_increase_mib'), 0)} MiB. "
                f"Memoria máxima reservada por PyTorch: {_fmt(gpu.get('torch_max_allocated_mib'), 0)} MiB. "
                f"Lecturas cada {SAMPLE_INTERVAL_S} s ({gpu.get('samples', 0)} en total): pueden perder picos "
                "de utilización de decenas de milisegundos.",
            ]
        else:
            lines.append(
                "Monitoreo externo no disponible (nvidia-smi). Memoria máxima reservada por PyTorch: "
                f"{_fmt(gpu.get('torch_max_allocated_mib'), 0)} MiB."
            )

    for key, title in (("cuda", "GPU"), ("cpu", "CPU (sin GPU)")):
        child = results.get(key)
        if not child:
            continue
        pairs = child.get("pairs", [])
        lines += ["", f"## Tiempos en {title}", ""]
        devices = child.get("resolved_devices", {})
        lines.append(
            "Dispositivos resueltos: " + ", ".join(f"{k} {v}" for k, v in devices.items()) + "."
        )
        if child.get("cpu_notice"):
            lines.append(f"Aviso mostrado a la persona usuaria: «{child['cpu_notice']}»")
        lines += ["", "### Escenas sintéticas (640x480)", ""] + _stage_table(_synthetic(pairs))
        real = [pair for pair in pairs if pair.get("kind") == "real"]
        if real:
            lines += ["", "### Pares de fotos reales (1920x1080, informativo)", ""] + _stage_table(real)

    lines += ["", "## Observaciones", ""]
    lines += [f"- {note}" for note in results.get("observations", [])] or ["- Sin observaciones."]
    return "\n".join(lines) + "\n"


def observations(results: dict) -> list[str]:
    """Results compared with what the progress log recorded; informative, never a failure."""
    notes = []
    for key, title in (("cuda", "GPU"), ("cpu", "CPU")):
        child = results.get(key)
        if not child:
            continue
        for pair in _synthetic(child.get("pairs", [])):
            if pair.get("error"):
                notes.append(f"{title}, {pair['name']}: error ({pair['error']}).")
                continue
            found = "; ".join(pair.get("changes", [])) or "sin cambios"
            notes.append(f"{title}, {pair['name']}: se esperaba {pair['expected']}; se obtuvo: {found}.")
        if child.get("cold_note"):
            notes.append(f"{title}: {child['cold_note']}")
    return notes


# --------------------------------------------------------------- parent mode


def child_command(device: str, json_out: Path) -> list[str]:
    return [sys.executable, "-m", "scripts.measure_requirements", "--device", device, "--json-out", str(json_out)]


def child_env(device: str, base: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Environment of a measurement process: offline, and without any visible GPU for the CPU run."""
    env = dict(os.environ if base is None else base)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = HIDE_ALL_GPUS
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_parent(devices: Sequence[str], out_dir: Path, run: Callable[..., Any] = subprocess.run) -> dict:
    """Measure each device in its own process and write rendimiento.json and rendimiento.md."""
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for device in devices:
            json_out = Path(scratch) / f"{device}.json"
            print(f"Midiendo en {device} (proceso nuevo)...", flush=True)
            completed = run(child_command(device, json_out), env=child_env(device), cwd=str(PROJECT_ROOT), check=False)
            if completed.returncode == 0 and json_out.is_file():
                results[device] = json.loads(json_out.read_text(encoding="utf-8"))
            else:
                results[device] = {"pairs": [], "error": f"el proceso terminó con código {completed.returncode}"}
    results["environment"] = merged_environment(results)
    results["meta"] = metadata()
    results["observations"] = observations(results)
    verdicts = evaluate(results)
    results["requirements"] = [vars(v) for v in verdicts]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rendimiento.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "rendimiento.md").write_text(render_markdown(results, verdicts), encoding="utf-8")
    return results


def merged_environment(results: dict) -> dict:
    env: dict[str, Any] = {}
    for key in ("cpu", "cuda"):  # The GPU process knows the GPU, so it wins.
        env.update({k: v for k, v in (results.get(key) or {}).get("environment", {}).items() if v})
    return env


def metadata(run: Callable[..., Any] = subprocess.run) -> dict:
    def git(*args: str) -> str:
        try:
            out = run(["git", *args], capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False)
            return out.stdout.strip()
        except OSError:
            return ""

    return {
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": git("rev-parse", "--short", "HEAD") or "—",
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


# ---------------------------------------------------------------- child mode


def environment() -> dict[str, Optional[str]]:
    import torch

    driver = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        driver = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or None,
    }


def real_pairs(directory: Path = SAMPLES_DIR) -> list[tuple[str, Path, Path]]:
    """The validation photo pairs ``real_before<suffix>`` / ``real_after<suffix>`` that exist (JPG)."""
    pairs = []
    for suffix in REAL_PAIR_SUFFIXES:
        before, after = directory / f"real_before{suffix}.jpg", directory / f"real_after{suffix}.jpg"
        if before.is_file() and after.is_file():
            pairs.append((f"real{suffix}", before, after))
    return pairs


def public_pair(record: dict) -> dict:
    """What the evidence may say about a pair: real photos are personal, so only their name and timings."""
    if record.get("kind") != "real":
        return record
    return {key: value for key, value in record.items() if key not in ("changes", "files", "expected")}


def sanitize(text: str) -> str:
    """Remove absolute paths of the project and the user's home folder from a message."""
    for path, replacement in ((PROJECT_ROOT, "."), (Path.home(), "~")):
        for form in {str(path), path.as_posix()}:
            text = text.replace(form, replacement)
    return text


def _depth_record(depth_map: Any, input_size: tuple[int, int]) -> dict:
    import numpy as np

    values = depth_map.values
    finite = bool(np.isfinite(values).all())
    return {
        "input_size": list(input_size),
        "shape": list(values.shape),
        "finite": finite,
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def run_child(device: str, json_out: Path) -> int:
    import torch

    from config.depth_settings import DepthSettings, DevicePreference
    from config.segmentation_settings import SegmentationSettings
    from config.settings import AppSettings
    from controllers.pipeline_controller import PipelineController
    from services.inference.device_manager import CPU_FALLBACK_NOTICE
    from services.inference.estimator_factory import create_depth_estimator

    smi_baseline = None
    if device == "cuda":
        readings = []
        for _ in range(BASELINE_SAMPLES):  # Before any model touches the GPU.
            readings.append(query_smi())
            time.sleep(SAMPLE_INTERVAL_S)
        smi_baseline = baseline(readings)

    # Automatic preference on both runs: the CPU run must resolve to the CPU by itself.
    settings = AppSettings(
        depth=DepthSettings(device=DevicePreference.AUTO, local_files_only=True),
        segmentation=SegmentationSettings(device=DevicePreference.AUTO, local_files_only=True),
    )
    estimator = create_depth_estimator(settings.depth)
    recorded: list[Any] = []
    original_estimate = estimator.estimate

    def recording_estimate(image):  # Keeps the depth maps of the last analysis for RR-002.
        depth_map = original_estimate(image)
        recorded.append(depth_map)
        return depth_map

    estimator.estimate = recording_estimate  # type: ignore[method-assign]
    controller = PipelineController(settings, depth_estimator=estimator)

    child: dict[str, Any] = {
        "device_requested": device,
        "cuda_available": torch.cuda.is_available(),
        "pairs": [],
        "environment": environment(),
    }
    notices_seen: set[str] = set()
    pair_plan = [("sintetico", name, SAMPLES_DIR / b, SAMPLES_DIR / a, expected)
                 for name, b, a, expected in SYNTHETIC_PAIRS]
    pair_plan += [("real", name, before, after, None) for name, before, after in real_pairs()]

    monitor = GpuMonitor() if device == "cuda" and smi_baseline else None
    if monitor:
        monitor.__enter__()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for kind, name, before, after, expected in pair_plan:
            record: dict[str, Any] = {"name": name, "kind": kind, "files": [before.name, after.name]}
            if expected:
                record["expected"] = expected
            try:
                pair = (controller.acquisition.from_file(before), controller.acquisition.from_file(after))
                record["size"] = list(pair[0].image.resolution)
                cold_has_loading = not controller.models_loaded
                recorded.clear()
                cold = controller.analyze(*pair)
                notices_seen.update(w for w in cold.warnings if w == CPU_FALLBACK_NOTICE)
                record["cold"] = {
                    "total": cold.total_seconds,
                    "stages": dict(cold.stage_seconds),
                    "includes_loading": cold_has_loading,
                }
                if kind == "sintetico":
                    record["depth"] = [_depth_record(d, pair[i].image.resolution) for i, d in enumerate(recorded[:2])]
                warm_reports = [controller.analyze(*pair) for _ in range(WARM_RUNS[device][kind])]
                record["warm"] = {
                    "count": len(warm_reports),
                    "total": stats([r.total_seconds for r in warm_reports]),
                    "stages": stage_stats([r.stage_seconds for r in warm_reports]),
                }
                last = warm_reports[-1] if warm_reports else cold
                record["changes"] = [obj.summary for obj in last.objects]
                record["warnings"] = len(last.warnings)
            except Exception as exc:  # Recorded, so the other pairs and the report still complete.
                record["error"] = sanitize(f"{type(exc).__name__}: {exc}")
            child["pairs"].append(public_pair(record))
            print(f"  {name}: {'error' if record.get('error') else 'listo'}", flush=True)
    finally:
        if monitor:
            monitor.__exit__(None, None, None)

    child["resolved_devices"] = controller.device_names
    child["cpu_notice"] = controller.device_notice
    child["cpu_notice_in_warnings"] = bool(controller.device_notice) and controller.device_notice in notices_seen
    if device == "cuda":
        child["gpu"] = {
            "smi_available": smi_baseline is not None,
            "baseline": smi_baseline,
            "samples": len(monitor.samples) if monitor else 0,
            "max_util": monitor.max_util if monitor else None,
            "max_mem_mib": monitor.max_mem_mib if monitor else None,
            "mem_increase_mib": (monitor.max_mem_mib - smi_baseline["mem_median_mib"])
            if monitor and monitor.max_mem_mib is not None else None,
            "torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
        }
    child["cold_note"] = (
        "solo el primer análisis del proceso incluye la carga de los modelos; los «frío» de los pares "
        "siguientes son el primer análisis de cada par con los modelos ya en memoria."
    )
    controller.close()
    json_out.write_text(json.dumps(child, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------- main


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidencia de rendimiento y portabilidad.")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Medir solo este dispositivo.")
    parser.add_argument("--json-out", type=Path, help=argparse.SUPPRESS)  # Child mode.
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Carpeta de los documentos.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.json_out is not None:
        if args.device is None:
            raise SystemExit("--json-out requiere --device.")
        return run_child(args.device, args.json_out)
    start = time.perf_counter()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    results = run_parent([args.device] if args.device else ["cuda", "cpu"], out_dir)
    for item in results["requirements"]:
        print(f"{item['requirement']}: {item['status']}")
    print(f"Documentos en {out_dir.name}/ ({time.perf_counter() - start:.0f} s en total).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
