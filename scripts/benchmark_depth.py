"""Measure depth inference latency on the current hardware.

Usage:
    python -m scripts.benchmark_depth [--image PATH] [--variant small|base|large]
                                      [--device auto|cuda|cpu] [--repetitions N]
                                      [--full-precision]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from config.depth_settings import DepthModelVariant, DepthSettings, DevicePreference
from config.paths import SAMPLES_DIR
from services.capture import ImageAcquisitionService
from services.inference import DepthAnythingV2Estimator, DepthLatencyBenchmark, LatencyReport

PIPELINE_BUDGET_SECONDS = 15.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth inference latency benchmark.")
    parser.add_argument("--image", type=Path, default=SAMPLES_DIR / "scene_before.png")
    parser.add_argument("--variant", choices=[v.value for v in DepthModelVariant], default="small")
    parser.add_argument("--device", choices=[d.value for d in DevicePreference], default="auto")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--full-precision", action="store_true", help="Disable float16 on CUDA.")
    return parser.parse_args(argv)


def print_report(report: LatencyReport, half_precision: bool) -> None:
    width, height = report.image_size
    precision = "float16 (mixta)" if half_precision and "CPU" not in report.device_name else "float32"
    print(f"Modelo:              {report.model_name}")
    print(f"Dispositivo:         {report.device_name}")
    print(f"Precisión:           {precision}")
    print(f"Imagen:              {width}x{height}")
    print(f"Carga del modelo:    {report.load_seconds:.3f} s")
    print(f"Primera inferencia:  {report.cold_seconds:.3f} s")
    print(
        f"Inferencia estable:  media {report.mean_seconds:.3f} s, mín {report.min_seconds:.3f} s,"
        f" máx {report.max_seconds:.3f} s ({len(report.warm_seconds)} repeticiones)"
    )
    share = 100.0 * report.pair_seconds / PIPELINE_BUDGET_SECONDS
    print(
        f"Profundidad por par: {report.pair_seconds:.3f} s "
        f"({share:.1f} % del límite de {PIPELINE_BUDGET_SECONDS:.0f} s para el par completo)"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    settings = DepthSettings(
        variant=DepthModelVariant(args.variant),
        device=DevicePreference(args.device),
        half_precision=not args.full_precision,
    )
    image = ImageAcquisitionService().from_file(args.image).image
    estimator = DepthAnythingV2Estimator(settings)
    try:
        report = DepthLatencyBenchmark(estimator, args.repetitions).run(image)
    finally:
        estimator.unload()

    print_report(report, settings.half_precision)
    if estimator.compute_device and estimator.compute_device.notice:
        print(f"Aviso: {estimator.compute_device.notice}")
    if torch.cuda.is_available() and estimator.compute_device and estimator.compute_device.is_accelerated:
        print(f"Memoria GPU pico:    {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
