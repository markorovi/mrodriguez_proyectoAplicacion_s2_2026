"""Chooses between GPU and CPU for model inference."""

from __future__ import annotations

from typing import Callable

import torch

from config.depth_settings import DevicePreference
from services.inference.compute_device import ComputeDevice

CPU_FALLBACK_NOTICE = (
    "No se detectó una tarjeta gráfica compatible; el análisis se ejecutará "
    "en el procesador y podría tardar más de lo habitual."
)


class DeviceManager:
    """Prioritizes CUDA when available and falls back to the CPU otherwise.

    A CUDA preference on a machine without a usable GPU degrades to the CPU
    with a user notice instead of failing, so the application stays portable.
    """

    def __init__(
        self,
        preference: DevicePreference = DevicePreference.AUTO,
        cuda_available: Callable[[], bool] = torch.cuda.is_available,
    ) -> None:
        self._preference = preference
        self._cuda_available = cuda_available

    def select(self) -> ComputeDevice:
        if self._preference is DevicePreference.CPU:
            return ComputeDevice(torch.device("cpu"), "CPU")
        if self._cuda_available():
            return ComputeDevice(torch.device("cuda"), torch.cuda.get_device_name(0))
        return ComputeDevice(torch.device("cpu"), "CPU", notice=CPU_FALLBACK_NOTICE)

    @staticmethod
    def release(device: ComputeDevice) -> None:
        """Return cached GPU memory to the driver after a model is freed."""
        if device.is_accelerated:
            torch.cuda.empty_cache()
