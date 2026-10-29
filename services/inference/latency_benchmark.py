"""Measurement of depth inference latency."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from models.image_data import ImageData
from services.inference.base_estimator import BaseDepthEstimator


@dataclass(frozen=True)
class LatencyReport:
    """Timings, in seconds, of one estimator on one image.

    ``cold_seconds`` is the first inference after loading, which includes
    one-off costs such as CUDA kernel selection. ``warm_seconds`` are the
    subsequent, steady-state inferences. Each timing includes moving the
    result back to host memory, so GPU work is fully accounted for.
    """

    model_name: str
    device_name: str
    image_size: tuple[int, int]
    load_seconds: float
    cold_seconds: float
    warm_seconds: tuple[float, ...]

    @property
    def mean_seconds(self) -> float:
        return mean(self.warm_seconds)

    @property
    def min_seconds(self) -> float:
        return min(self.warm_seconds)

    @property
    def max_seconds(self) -> float:
        return max(self.warm_seconds)

    @property
    def pair_seconds(self) -> float:
        """Steady-state depth time for a "before"/"after" pair."""
        return 2.0 * self.mean_seconds


class DepthLatencyBenchmark:
    """Loads an estimator and times repeated inferences on the same image."""

    def __init__(self, estimator: BaseDepthEstimator, repetitions: int = 10) -> None:
        if repetitions < 1:
            raise ValueError("At least one timed repetition is required.")
        self._estimator = estimator
        self._repetitions = repetitions

    def run(self, image: ImageData) -> LatencyReport:
        start = perf_counter()
        self._estimator.load()
        load_seconds = perf_counter() - start

        cold = self._estimator.estimate(image).inference_seconds
        warm = tuple(
            self._estimator.estimate(image).inference_seconds for _ in range(self._repetitions)
        )
        return LatencyReport(
            model_name=self._estimator.name,
            device_name=self._estimator.device_name,
            image_size=image.resolution,
            load_seconds=load_seconds,
            cold_seconds=cold,
            warm_seconds=warm,
        )
