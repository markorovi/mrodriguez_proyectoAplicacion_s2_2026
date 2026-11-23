"""Pretrained models are loaded from the local files first; the network is only a fallback.

The unit tests replace ``from_pretrained`` with a recorder and never touch the
network or real weights. The integration test loads the real models with the
default settings, offline mode switched off and every socket call blocked, and
requires zero connection attempts:

    RUN_INTEGRATION=1 pytest tests/test_model_loading_local_first.py -v
"""

from __future__ import annotations

import os
import socket
from typing import Optional

import pytest
import torch
import transformers

from config import DepthSettings, DevicePreference, SegmentationSettings
from models import ModelLoadError, SegmentationError
from services.inference import DepthAnythingV2Estimator, DeviceManager, HuggingFaceDepthModelLoader
from services.segmentation import SegFormerSegmenter
from services.local_first import DISABLE_CONVERSION_VARIABLE
from services.segmentation.model_loader import HuggingFaceSegmentationModelLoader
from tests.jargon import jargon_found

# (loader class, transformers class of the model, domain error) for depth and segmentation.
LOADERS = [
    pytest.param(HuggingFaceDepthModelLoader, "AutoModelForDepthEstimation", ModelLoadError, id="profundidad"),
    pytest.param(
        HuggingFaceSegmentationModelLoader, "AutoModelForSemanticSegmentation", SegmentationError, id="segmentacion"
    ),
]


class _FakeHub:
    """Stands in for ``from_pretrained`` and records the ``local_files_only`` of every call."""

    def __init__(self, missing_locally: bool = False, error: Optional[Exception] = None) -> None:
        self.missing_locally = missing_locally
        self.error = error
        self.processor_calls: list[bool] = []
        self.model_calls: list[bool] = []
        # Value of the conversion switch seen by each from_pretrained call of the processor.
        self.conversion_switch: list[Optional[str]] = []

    def processor(self, model_id: str, **options) -> object:
        self.processor_calls.append(options["local_files_only"])
        self.conversion_switch.append(os.environ.get(DISABLE_CONVERSION_VARIABLE))
        if self.error is not None:
            raise self.error
        return object()

    def model(self, model_id: str, **options) -> torch.nn.Module:
        # The model is the part missing from the cache: the processor alone is not enough.
        self.model_calls.append(options["local_files_only"])
        if self.missing_locally and options["local_files_only"]:
            raise OSError("not in the local cache")
        return torch.nn.Linear(1, 1)


@pytest.fixture
def hub(monkeypatch):
    def install(model_class: str, **kwargs) -> _FakeHub:
        fake = _FakeHub(**kwargs)
        monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", fake.processor)
        monkeypatch.setattr(getattr(transformers, model_class), "from_pretrained", fake.model)
        return fake

    return install


@pytest.mark.parametrize(("loader_class", "model_class", "error"), LOADERS)
def test_local_weights_are_loaded_with_a_single_offline_call(hub, tmp_path, loader_class, model_class, error):
    fake = hub(model_class)

    loader_class(tmp_path, local_files_only=False).load("modelo", torch.device("cpu"))

    assert fake.processor_calls == [True]
    assert fake.model_calls == [True]


@pytest.mark.parametrize(("loader_class", "model_class", "error"), LOADERS)
def test_missing_weights_are_downloaded_when_allowed(hub, tmp_path, loader_class, model_class, error):
    fake = hub(model_class, missing_locally=True)

    loader_class(tmp_path, local_files_only=False).load("modelo", torch.device("cpu"))

    # Processor and model are retried together, the second time allowing the download.
    assert fake.processor_calls == [True, False]
    assert fake.model_calls == [True, False]


@pytest.mark.parametrize(("loader_class", "model_class", "error"), LOADERS)
def test_missing_weights_fail_without_retry_when_only_local_files_are_allowed(
    hub, tmp_path, loader_class, model_class, error
):
    fake = hub(model_class, missing_locally=True)

    with pytest.raises(error):
        loader_class(tmp_path, local_files_only=True).load("modelo", torch.device("cpu"))

    assert fake.processor_calls == [True]
    assert fake.model_calls == [True]


@pytest.mark.parametrize(("loader_class", "model_class", "error"), LOADERS)
def test_value_error_is_not_retried(hub, tmp_path, loader_class, model_class, error):
    fake = hub(model_class, error=ValueError("incompatible checkpoint"))

    with pytest.raises(error):
        loader_class(tmp_path, local_files_only=False).load("modelo", torch.device("cpu"))

    assert fake.processor_calls == [True]
    assert fake.model_calls == []


def test_local_only_message_asks_for_the_files_on_this_computer(hub, tmp_path):
    messages = []
    for loader_class, model_class, error in [p.values for p in LOADERS]:
        hub(model_class, missing_locally=True)
        with pytest.raises(error) as raised:
            loader_class(tmp_path, local_files_only=True).load("modelo", torch.device("cpu"))
        messages.append(str(raised.value))

    for text in messages:
        assert "internet" not in text.lower(), text
        assert "este equipo" in text, text
        assert jargon_found(text) == [], text


def _absent(monkeypatch, name: str) -> None:
    """Make ``name`` absent now and absent again after the test, whatever the test leaves."""
    monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv(name)


def test_conversion_check_is_off_while_loading_and_absent_again_afterwards(hub, tmp_path, monkeypatch):
    _absent(monkeypatch, DISABLE_CONVERSION_VARIABLE)
    fake = hub("AutoModelForDepthEstimation")

    HuggingFaceDepthModelLoader(tmp_path, local_files_only=False).load("modelo", torch.device("cpu"))

    assert fake.conversion_switch == ["1"]
    assert DISABLE_CONVERSION_VARIABLE not in os.environ


def test_conversion_switch_set_by_the_user_is_respected(hub, tmp_path, monkeypatch):
    monkeypatch.setenv(DISABLE_CONVERSION_VARIABLE, "0")
    fake = hub("AutoModelForSemanticSegmentation")

    HuggingFaceSegmentationModelLoader(tmp_path, local_files_only=False).load("modelo", torch.device("cpu"))

    assert fake.conversion_switch == ["0"]
    assert os.environ[DISABLE_CONVERSION_VARIABLE] == "0"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run tests that load pretrained models",
)
def test_saved_weights_load_with_default_settings_without_any_connection_attempt(monkeypatch):
    import huggingface_hub.constants

    # Offline mode off, as on a normal start of the application: only the code may avoid the network.
    # The code itself must turn the safetensors conversion check off: the environment does not help.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", DISABLE_CONVERSION_VARIABLE):
        _absent(monkeypatch, name)
    # The library reads the variables once, at import time; reset the value it keeps.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)

    attempts: list[str] = []

    def blocked(name: str):
        def refuse(*args, **kwargs):
            attempts.append(name)
            raise OSError(f"network blocked by the test ({name})")

        return refuse

    monkeypatch.setattr(socket, "getaddrinfo", blocked("getaddrinfo"))
    monkeypatch.setattr(socket, "create_connection", blocked("create_connection"))
    monkeypatch.setattr(socket.socket, "connect", blocked("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", blocked("socket.connect_ex"))

    depth = DepthAnythingV2Estimator(DepthSettings(), DeviceManager(DevicePreference.AUTO))
    segmenter = SegFormerSegmenter(SegmentationSettings(), DeviceManager(DevicePreference.AUTO))
    try:
        depth.load()
        segmenter.load()
        assert depth.is_loaded and segmenter.is_loaded
    finally:
        depth.unload()
        segmenter.unload()

    print(f"\nIntentos de conexion: {len(attempts)} {attempts[:5]}")
    assert attempts == [], f"{len(attempts)} connection attempts: {attempts[:5]}"
