"""Measure the complete analysis of a pair of photos on the CPU and/or the GPU.

Usage:
    python -m scripts.benchmark_pipeline [--devices cpu cuda] [--repetitions N]
                                         [--pair BEFORE AFTER ...]

By default it measures the synthetic sample pair (640x480) and, if present,
the real pair ``real_before_2`` / ``real_after_2`` (1920x1080) on every
available device. For each device it reports the first analysis (which
loads the models), the mean of ``N`` analyses with the models in memory,
the time of each stage and the changes found, so CPU and GPU results can
be checked against each other.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch

from config.depth_settings import DepthSettings, DevicePreference
from config.paths import SAMPLES_DIR
from config.segmentation_settings import SegmentationSettings
from config.settings import AppSettings
from controllers.pipeline_controller import STAGES, PipelineController
from models.comparison_report import ComparisonReport

PIPELINE_BUDGET_SECONDS = 15.0


@dataclass(frozen=True)
class PairTiming:
    """Timings of one pair of photos on one device."""

    name: str
    image_size: tuple[int, int]
    first_seconds: Optional[float]
    first_stages: dict[str, float]
    warm_seconds: tuple[float, ...]
    warm_stage_means: dict[str, float]
    changes: tuple[str, ...]

    @property
    def warm_mean(self) -> float:
        return statistics.fmean(self.warm_seconds) if self.warm_seconds else float("nan")


@dataclass
class DeviceTiming:
    """Timings of every pair on one device."""

    device: str
    device_names: dict[str, str] = field(default_factory=dict)
    pairs: list[PairTiming] = field(default_factory=list)
    peak_gpu_mib: Optional[float] = None
    notice: str = ""


def settings_for(device: DevicePreference) -> AppSettings:
    """Default settings with both models forced onto ``device``."""
    return AppSettings(depth=DepthSettings(device=device), segmentation=SegmentationSettings(device=device))


def measure_pair(
    controller: PipelineController,
    name: str,
    before: Path,
    after: Path,
    repetitions: int,
) -> PairTiming:
    """Analyze ``before``/``after`` once and then ``repetitions`` times with the models in memory.

    The first analysis includes loading the models only while they are not
    yet in memory; ``first_seconds`` is ``None`` when they already were.
    """
    pair = (controller.acquisition.from_file(before), controller.acquisition.from_file(after))
    cold = not controller.models_loaded
    first = controller.analyze(*pair)
    warm = [controller.analyze(*pair) for _ in range(repetitions)]
    width, height = pair[0].image.resolution
    return PairTiming(
        name=name,
        image_size=(width, height),
        first_seconds=first.total_seconds if cold else None,
        first_stages=dict(first.stage_seconds) if cold else {},
        warm_seconds=tuple(report.total_seconds for report in warm),
        warm_stage_means=_stage_means(warm),
        changes=_changes(warm[-1] if warm else first),
    )


def measure_device(
    controller: PipelineController,
    device: str,
    pairs: Sequence[tuple[str, Path, Path]],
    repetitions: int,
) -> DeviceTiming:
    """Measure every pair with one controller (the first pair pays the model loading)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timing = DeviceTiming(device=device)
    for name, before, after in pairs:
        timing.pairs.append(measure_pair(controller, name, before, after, repetitions))
    timing.device_names = controller.device_names
    timing.notice = controller.device_notice
    if torch.cuda.is_available() and "CPU" not in timing.device_names.get("depth", "CPU"):
        timing.peak_gpu_mib = torch.cuda.max_memory_allocated() / 2**20
    return timing


def default_pairs() -> list[tuple[str, Path, Path]]:
    pairs = [("sintético", SAMPLES_DIR / "scene_before.png", SAMPLES_DIR / "scene_after_moved.png")]
    real_before, real_after = SAMPLES_DIR / "real_before_2.jpg", SAMPLES_DIR / "real_after_2.jpg"
    if real_before.is_file() and real_after.is_file():
        pairs.append(("real_2", real_before, real_after))
    return pairs


def default_devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def print_device(timing: DeviceTiming) -> None:
    print(f"\n=== {timing.device.upper()}: {timing.device_names.get('depth', '?')} ===")
    if timing.notice:
        print(f"Aviso: {timing.notice}")
    for pair in timing.pairs:
        width, height = pair.image_size
        print(f"\nPar {pair.name} ({width}x{height})")
        if pair.first_seconds is not None:
            loading = pair.first_stages.get("loading", 0.0)
            print(f"  Primer análisis:  {pair.first_seconds:.2f} s (carga de modelos {loading:.2f} s)")
        if pair.warm_seconds:
            print(
                f"  Con modelos en memoria: media {pair.warm_mean:.2f} s, mín {min(pair.warm_seconds):.2f} s,"
                f" máx {max(pair.warm_seconds):.2f} s ({len(pair.warm_seconds)} análisis)"
            )
            stages = ", ".join(
                f"{stage} {pair.warm_stage_means[stage]:.2f} s" for stage in STAGES if stage in pair.warm_stage_means
            )
            print(f"  Por etapa:        {stages}")
        print(f"  Cambios:          {'; '.join(pair.changes) or 'sin cambios'}")
    if timing.peak_gpu_mib is not None:
        print(f"\nMemoria GPU pico:  {timing.peak_gpu_mib:.0f} MiB")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medición del análisis completo en CPU y GPU.")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--pair", nargs=2, action="append", type=Path, metavar=("ANTES", "DESPUES"), help="Par de fotos a medir."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    pairs = (
        [(f"{before.stem} / {after.stem}", before, after) for before, after in args.pair]
        if args.pair
        else default_pairs()
    )
    for device in args.devices or default_devices():
        with PipelineController(settings_for(DevicePreference(device))) as controller:
            print_device(measure_device(controller, device, pairs, args.repetitions))
    print(f"\nLímite de RR-001: {PIPELINE_BUDGET_SECONDS:.0f} s por par en GPU.")
    return 0


def _stage_means(reports: Sequence[ComparisonReport]) -> dict[str, float]:
    stages = {stage for report in reports for stage in report.stage_seconds}
    return {stage: statistics.fmean(r.stage_seconds.get(stage, 0.0) for r in reports) for stage in stages}


def _changes(report: ComparisonReport) -> tuple[str, ...]:
    return tuple(obj.summary for obj in report.objects)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
