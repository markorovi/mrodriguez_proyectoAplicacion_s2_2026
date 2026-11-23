"""Domain messages that reach the user avoid technical terms (RNF-001).

Each message is produced through the code path that shows it to the user (a
failed check, an exception, a report warning), never copied into the test,
so a later rewording is checked as it really reads.
"""

from __future__ import annotations

import re
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from config import (
    AlignmentSettings,
    AppSettings,
    CaptureSettings,
    DepthSettings,
    DevicePreference,
    SegmentationSettings,
)
from controllers import PipelineController
from models import ComparisonReport, ImageData, SceneDiffError
from services.alignment.spatial_alignment_service import SpatialAlignmentService
from services.capture.camera_capture import CameraCapture
from services.capture.file_image_loader import FileImageLoader
from services.inference import DepthAnythingV2Estimator, DeviceManager, HuggingFaceDepthModelLoader
from services.segmentation import SegFormerSegmenter
from services.segmentation.model_loader import HuggingFaceSegmentationModelLoader
from tests.doubles import (
    FailingDepthModel,
    FailingSegformerModel,
    FakeModelLoader,
    FakeSegmentationModelLoader,
    FakeVideoDevice,
    SceneDepthEstimator,
)
from tests.jargon import jargon_found
from tests.test_session_history import _report
from ui.pipeline_worker import PipelineWorker
from ui.report_export import ReportExportError, export_report, save_annotated_image
from ui.report_widget import ReportWidget
from utils.image_validator import ImageValidator
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

# --- The detector itself -------------------------------------------------------------------------


def test_jargon_is_found_ignoring_case_and_accents():
    assert jargon_found("Se usó una homografía.") == ["homografía"]
    assert jargon_found("SE USÓ UNA HOMOGRAFIA.") == ["homografía"]
    assert jargon_found("Puntos geometricamente consistentes en la GPU") == ["geométricamente", "gpu"]
    assert jargon_found("Las fotos se compararon sin problemas.") == []


# --- Scenes and helpers --------------------------------------------------------------------------

_BUILDER = SyntheticSceneBuilder(640, 480, seed=7)


def _room(center_x: float | None = 0.3, brightness: float = 1.0) -> ImageData:
    objects = [] if center_x is None else [SceneObject(center_x=center_x)]
    return ImageData(_BUILDER.render(objects, brightness=brightness), "memoria")


def _controller(settings: AppSettings | None = None, estimator=None, aligner=None) -> PipelineController:
    settings = settings or AppSettings()
    segmenter = SegFormerSegmenter(
        settings.segmentation, DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    return PipelineController(
        settings, depth_estimator=estimator or SceneDepthEstimator(), aligner=aligner, segmenter=segmenter
    )


def _raised(action) -> str:
    """Message of the error ``action`` raises for the user."""
    with pytest.raises((SceneDiffError, ReportExportError)) as raised:
        action()
    return str(raised.value)


def _warning(report: ComparisonReport, fragment: str) -> str:
    matches = [text for text in report.warnings if fragment in text]
    assert len(matches) == 1, report.warnings
    return matches[0]


def _validation_errors(pixels: np.ndarray) -> str:
    report = ImageValidator().validate(ImageData(pixels, "memoria"))
    assert report.errors
    return " ".join(report.errors)


def _depth(image: ImageData):
    return SceneDepthEstimator().estimate(image)


class _NonMetricEstimator(SceneDepthEstimator):
    @property
    def is_metric(self) -> bool:
        return False


class _WrongShapeEstimator(SceneDepthEstimator):
    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        return np.ones((10, 10), np.float32)


class _LowOverlapAligner(SpatialAlignmentService):
    """Real alignment whose result covers only a third of the "before" view."""

    def align(self, before, after, after_depth):
        aligned = super().align(before, after, after_depth)
        mask = np.zeros_like(aligned.valid_mask)
        mask[:, : mask.shape[1] // 3] = True
        return replace(aligned, valid_mask=mask)


class _CrashingController:
    """Controller whose analysis fails with an error the application does not expect."""

    def analyze(self, before, after, progress=None):
        raise RuntimeError("CUDA error: device-side assert triggered")


# --- One producer per message (numbered as in the RNF-001 inventory) -----------------------------


def _m01_unaligned_fallback() -> str:
    settings = AppSettings(alignment=AlignmentSettings(min_inliers=100_000))
    return _warning(_controller(settings).analyze(_room(0.3), _room(None)), "sin alinear")


def _m02_low_overlap() -> str:
    report = _controller(aligner=_LowOverlapAligner()).analyze(_room(0.3), _room(0.65))
    return _warning(report, "aparece en ambas imágenes")


def _m03_depth_not_metric() -> str:
    report = _controller(estimator=_NonMetricEstimator()).analyze(_room(0.3), _room(0.65))
    return _warning(report, "metros")


def _m04_cpu_fallback_notice() -> str:
    notice = DeviceManager(DevicePreference.AUTO, cuda_available=lambda: False).select().notice
    assert notice
    return notice


def _m05_photo_prefix_in_warnings() -> str:
    dark = _room(0.3, brightness=0.12)
    report = _controller().analyze(dark, ImageData(dark.pixels.copy(), "b"))
    named = [text for text in report.warnings if text.startswith("Imagen '")]
    assert len(named) == 2
    return " ".join(named)


def _m06_few_matches() -> str:
    service = SpatialAlignmentService(AlignmentSettings(min_inliers=100_000))
    after = _room(0.65)
    return _raised(lambda: service.align(_room(0.3), after, _depth(after)))


def _m07_few_consistent_matches() -> str:
    before, after = _room(0.3), _room(0.65)
    _, aligned = SpatialAlignmentService().estimate_homography(before, after)
    # Enough matches to try the alignment, one consistent match short of accepting it.
    assert aligned.good_matches > aligned.inliers
    service = SpatialAlignmentService(AlignmentSettings(min_inliers=aligned.inliers + 1))
    return _raised(lambda: service.align(before, after, _depth(after)))


def _m08_unrealistic_alignment() -> str:
    before = _room(0.3)
    # The centre of the room enlarged 3.5 times: aligning it needs a shrink far beyond any real shot.
    h, w = int(480 / 3.5), int(640 / 3.5)
    y, x = (480 - h) // 2, (640 - w) // 2
    zoomed = ImageData(cv2.resize(before.pixels[y : y + h, x : x + w], (640, 480), interpolation=cv2.INTER_CUBIC), "z")
    return _raised(lambda: SpatialAlignmentService().align(before, zoomed, _depth(zoomed)))


def _m09_unsupported_format() -> str:
    return _raised(lambda: ImageValidator().check_file_format("foto.gif"))


def _m10_bit_depth() -> str:
    return _validation_errors(np.zeros((480, 640, 3), np.uint16))


def _m11_not_color() -> str:
    return _validation_errors(np.zeros((480, 640), np.uint8))


def _m12_low_resolution() -> str:
    return _validation_errors(np.zeros((240, 320, 3), np.uint8))


def _m13_low_light() -> str:
    report = ImageValidator().validate(_room(0.3, brightness=0.12))
    assert report.is_valid and report.warnings
    return " ".join(report.warnings)


def _m14_camera_gave_no_image() -> str:
    device = FakeVideoDevice(frame=None)
    return _raised(CameraCapture(CaptureSettings(), device_factory=lambda index: device).acquire)


def _m15_camera_unavailable() -> str:
    device = FakeVideoDevice(np.zeros((720, 1280, 3), np.uint8), opened=False)
    return _raised(CameraCapture(CaptureSettings(), device_factory=lambda index: device).acquire)


def _m16_file_not_found(tmp_path) -> str:
    return _raised(FileImageLoader(tmp_path / "no_existe.png").acquire)


def _m17_damaged_file(tmp_path) -> str:
    path = tmp_path / "danada.png"
    path.write_bytes(b"esto no es una imagen")
    return _raised(FileImageLoader(path).acquire)


def _m18_depth_inference_failed() -> str:
    estimator = DepthAnythingV2Estimator(
        DepthSettings(),
        device_manager=DeviceManager(DevicePreference.CPU),
        model_loader=FakeModelLoader(model=FailingDepthModel()),
    )
    return _raised(lambda: estimator.estimate(_room(0.3)))


def _m19_depth_size_mismatch() -> str:
    return _raised(lambda: _WrongShapeEstimator().estimate(_room(0.3)))


def _m20_depth_needs_color() -> str:
    return _raised(lambda: SceneDepthEstimator().estimate(ImageData(np.zeros((480, 640), np.uint8), "gris")))


def _m21_depth_model_missing(tmp_path) -> str:
    loader = HuggingFaceDepthModelLoader(tmp_path, local_files_only=True)
    return _raised(lambda: loader.load(DepthSettings().model_id, torch.device("cpu")))


def _m22_segmentation_model_missing(tmp_path) -> str:
    loader = HuggingFaceSegmentationModelLoader(tmp_path, local_files_only=True)
    return _raised(lambda: loader.load(SegmentationSettings().model_id, torch.device("cpu")))


def _unreachable_hub():
    """Every ``from_pretrained`` fails as with missing files and no connection; nothing is downloaded."""
    import transformers

    return patch.object(transformers.AutoImageProcessor, "from_pretrained", side_effect=OSError("sin conexión"))


def _m30_depth_model_download_failed(tmp_path) -> str:
    loader = HuggingFaceDepthModelLoader(tmp_path, local_files_only=False)
    with _unreachable_hub():
        return _raised(lambda: loader.load(DepthSettings().model_id, torch.device("cpu")))


def _m31_segmentation_model_download_failed(tmp_path) -> str:
    loader = HuggingFaceSegmentationModelLoader(tmp_path, local_files_only=False)
    with _unreachable_hub():
        return _raised(lambda: loader.load(SegmentationSettings().model_id, torch.device("cpu")))


def _m32_segmentation_size_mismatch() -> str:
    from tests.doubles import SceneSegmenter

    class _WrongSizeSegmenter(SceneSegmenter):
        def _predict(self, rgb: np.ndarray) -> np.ndarray:
            return np.zeros((8, 8), np.int32)

    return _raised(lambda: _WrongSizeSegmenter().segment(ImageData(np.zeros((480, 640, 3), np.uint8), "foto")))


def _segmenter(loader: FakeSegmentationModelLoader) -> SegFormerSegmenter:
    return SegFormerSegmenter(SegmentationSettings(), DeviceManager(DevicePreference.CPU), loader)


def _m23_segmentation_needs_color() -> str:
    gray = ImageData(np.zeros((480, 640), np.uint8), "gris")
    return _raised(lambda: _segmenter(FakeSegmentationModelLoader()).segment(gray))


def _m24_segmentation_failed() -> str:
    segmenter = _segmenter(FakeSegmentationModelLoader(FailingSegformerModel()))
    return _raised(lambda: segmenter.segment(_room(0.3)))


def _m25_report_not_saved(tmp_path) -> str:
    return _raised(lambda: export_report(_report(), tmp_path / "no_existe" / "reporte.txt"))


def _m26_report_image_not_saved(tmp_path) -> str:
    (tmp_path / "reporte.png").mkdir()  # The summary is written; its image cannot be.
    return _raised(lambda: export_report(_report(), tmp_path / "reporte.txt"))


def _m27_image_not_saved(tmp_path) -> str:
    return _raised(lambda: save_annotated_image(_report(), tmp_path / "no_existe" / "resultado.png"))


def _m28_nothing_to_export(tmp_path) -> str:
    widget = ReportWidget()
    try:
        texts = (
            _raised(lambda: widget.export_to(tmp_path / "reporte.txt")),
            _raised(lambda: widget.save_image_to(tmp_path / "resultado.png")),
        )
    finally:
        widget.deleteLater()
    return " ".join(texts)


def _m29_unexpected_failure() -> str:
    worker = PipelineWorker(_CrashingController(), _room(0.3), _room(0.65))
    messages: list[str] = []
    worker.error_occurred.connect(messages.append)
    worker.run()  # On this thread: the signal reaches ``messages`` directly.
    assert len(messages) == 1
    return messages[0]


MESSAGES = {
    1: _m01_unaligned_fallback,
    2: _m02_low_overlap,
    3: _m03_depth_not_metric,
    4: _m04_cpu_fallback_notice,
    5: _m05_photo_prefix_in_warnings,
    6: _m06_few_matches,
    7: _m07_few_consistent_matches,
    8: _m08_unrealistic_alignment,
    9: _m09_unsupported_format,
    10: _m10_bit_depth,
    11: _m11_not_color,
    12: _m12_low_resolution,
    13: _m13_low_light,
    14: _m14_camera_gave_no_image,
    15: _m15_camera_unavailable,
    16: _m16_file_not_found,
    17: _m17_damaged_file,
    18: _m18_depth_inference_failed,
    19: _m19_depth_size_mismatch,
    20: _m20_depth_needs_color,
    21: _m21_depth_model_missing,
    22: _m22_segmentation_model_missing,
    23: _m23_segmentation_needs_color,
    24: _m24_segmentation_failed,
    25: _m25_report_not_saved,
    26: _m26_report_image_not_saved,
    27: _m27_image_not_saved,
    28: _m28_nothing_to_export,
    29: _m29_unexpected_failure,
    30: _m30_depth_model_download_failed,
    31: _m31_segmentation_model_download_failed,
    32: _m32_segmentation_size_mismatch,
}

_NEEDS_TMP_PATH = {16, 17, 21, 22, 25, 26, 27, 28, 30, 31}
_NEEDS_QT = {28, 29}


@pytest.mark.parametrize("number", list(MESSAGES), ids=[f"{n:02d}-{f.__name__[5:]}" for n, f in MESSAGES.items()])
def test_domain_message_avoids_technical_terms(number, request, tmp_path):
    if number in _NEEDS_QT:
        request.getfixturevalue("qapp")
    produce = MESSAGES[number]
    text = produce(tmp_path) if number in _NEEDS_TMP_PATH else produce()
    assert text.strip()
    assert jargon_found(text) == [], text


def test_depth_size_mismatch_hides_internal_sizes():
    # Message 19: the user is told what happened, not the sizes the model worked with.
    text = _m19_depth_size_mismatch()
    assert jargon_found(text) == [], text
    assert "mapa" not in text.lower(), text
    assert not re.search(r"\(\d+,\s*\d+\)", text), text
