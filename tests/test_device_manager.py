import torch

from config import DevicePreference
from services.inference import DeviceManager
from services.inference.device_manager import CPU_FALLBACK_NOTICE


def test_cpu_preference_is_honoured_without_notice():
    device = DeviceManager(DevicePreference.CPU, cuda_available=lambda: True).select()
    assert device.torch_device.type == "cpu"
    assert not device.is_accelerated
    assert device.notice is None


def test_auto_without_gpu_falls_back_to_cpu_with_notice():
    device = DeviceManager(DevicePreference.AUTO, cuda_available=lambda: False).select()
    assert device.torch_device.type == "cpu"
    assert device.notice == CPU_FALLBACK_NOTICE


def test_cuda_preference_without_gpu_degrades_gracefully():
    device = DeviceManager(DevicePreference.CUDA, cuda_available=lambda: False).select()
    assert device.torch_device.type == "cpu"
    assert device.notice is not None


def test_auto_with_gpu_selects_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Test GPU")
    device = DeviceManager(DevicePreference.AUTO, cuda_available=lambda: True).select()
    assert device.torch_device.type == "cuda"
    assert device.is_accelerated
    assert device.name == "Test GPU"
    assert device.notice is None


def test_release_on_cpu_is_a_no_op():
    manager = DeviceManager(DevicePreference.CPU)
    manager.release(manager.select())
