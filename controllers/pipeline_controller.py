"""End-to-end orchestration of the scene change analysis."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator, Optional, Union

from config.settings import AppSettings
from models.acquired_image import AcquiredImage
from models.alignment import AlignmentResult
from models.comparison_report import ComparisonReport
from models.depth_map import DepthMap
from models.exceptions import AlignmentError, NonIndoorSceneError
from models.image_data import ImageData
from models.segmentation_mask import SegmentationMask
from services.alignment.spatial_alignment_service import SpatialAlignmentService
from services.capture.acquisition_service import ImageAcquisitionService
from services.comparison.spatial_comparison_service import SpatialComparisonService
from services.inference.base_estimator import BaseDepthEstimator
from services.inference.device_manager import DeviceManager
from services.inference.estimator_factory import create_depth_estimator
from services.segmentation.base_segmenter import BaseSegmenter
from services.segmentation.region_labeler import RegionLabeler
from services.segmentation.segmenter_factory import create_segmenter
from utils.camera_model import resolve_intrinsics
from utils.change_visualization import draw_changes
from utils.image_validator import ImageValidator

ImageInput = Union[AcquiredImage, ImageData]
ProgressCallback = Callable[[str], None]

# Pipeline stages in execution order; each key is reported to the progress
# callback when the stage starts and timed in ``ComparisonReport.stage_seconds``.
# "loading" only runs (and only appears) while a model is not yet in memory.
STAGES = ("acquisition", "loading", "depth", "alignment", "segmentation", "comparison", "rendering")

LOW_OVERLAP_FRACTION = 0.5


class PipelineController:
    """Single entry point that turns a "before"/"after" pair into a report.

    Stages, in order: acquisition and validation, monocular metric depth of
    both views, spatial alignment of "after" onto "before", semantic
    segmentation of both views, 3-D comparison and classification, and
    rendering of the annotated result. Each collaborator can be injected,
    which keeps the controller independent of concrete models and testable
    without GPU or network.

    Models are loaded on the first analysis, in a "loading" stage of their
    own so the timing of the other stages reflects the analysis alone, and
    kept in memory for the next ones; :meth:`close` (or leaving a ``with``
    block) frees them. Errors
    raised by the stages derive from :class:`models.exceptions.SceneDiffError`.
    A failed alignment does not stop the analysis when
    ``settings.alignment.fallback_to_identity`` is set: the views are then
    compared unaligned and the report carries a warning.

    After the segmentation, a photo dominated by open-air scenery (sky,
    trees, grass...) stops the analysis with
    :class:`models.exceptions.NonIndoorSceneError` (RF-012), unless
    ``settings.segmentation.check_indoor_scene`` is off.

    Distances are measured with the camera parameters of the "before" photo:
    the calibrated profile ``settings.camera`` when it matches, otherwise the
    focal length in the photo's EXIF data or an estimate
    (:func:`utils.camera_model.resolve_intrinsics`).

    An optional ``progress`` callback receives the key of each stage in
    :data:`STAGES` as it starts; it runs on the caller's thread.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        acquisition: ImageAcquisitionService | None = None,
        depth_estimator: BaseDepthEstimator | None = None,
        aligner: SpatialAlignmentService | None = None,
        segmenter: BaseSegmenter | None = None,
        comparator: SpatialComparisonService | None = None,
    ) -> None:
        self._settings = settings or AppSettings()
        s = self._settings
        self._acquisition = acquisition or ImageAcquisitionService(ImageValidator(s.image), s.capture)
        self._estimator = depth_estimator or create_depth_estimator(s.depth)
        self._aligner = aligner or SpatialAlignmentService(s.alignment)
        if segmenter is None and s.segmentation.enabled:
            segmenter = create_segmenter(s.segmentation)
        self._segmenter = segmenter if s.segmentation.enabled else None
        self._labeler = RegionLabeler(
            s.segmentation.min_label_coverage, max_outdoor_percent=s.segmentation.max_outdoor_percent
        )
        self._comparator = comparator or SpatialComparisonService(
            s.comparison, self._labeler, DeviceManager(s.depth.device)
        )

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def models_loaded(self) -> bool:
        """True when every model the analysis needs is already in memory."""
        return self._estimator.is_loaded and (self._segmenter is None or self._segmenter.is_loaded)

    @property
    def acquisition(self) -> ImageAcquisitionService:
        """Service that loads and validates images with this controller's settings."""
        return self._acquisition

    @property
    def device_names(self) -> dict[str, str]:
        """Hardware each stage ran on in the last analysis ("" before the first one)."""
        names = {"depth": self._estimator.device_name, "comparison": self._comparator.device_name}
        if self._segmenter is not None:
            names["segmentation"] = self._segmenter.device_name
        return names

    @property
    def device_notice(self) -> str:
        """Notice for the user when the analysis fell back to the CPU ("" otherwise)."""
        device = getattr(self._estimator, "compute_device", None)
        return getattr(device, "notice", None) or ""

    def analyze_files(
        self,
        before: Union[str, Path],
        after: Union[str, Path],
        progress: Optional[ProgressCallback] = None,
    ) -> ComparisonReport:
        """Load, validate and compare two image files."""
        timed = _Stages(progress)
        with timed("acquisition"):
            acquired = (self._acquisition.from_file(before), self._acquisition.from_file(after))
        return self._run(*acquired, timed)

    def analyze(
        self, before: ImageInput, after: ImageInput, progress: Optional[ProgressCallback] = None
    ) -> ComparisonReport:
        """Compare two images already in memory (validated here if needed)."""
        timed = _Stages(progress)
        with timed("acquisition"):
            acquired = (self._validated(before), self._validated(after))
        return self._run(*acquired, timed)

    def close(self) -> None:
        """Release the models and the GPU memory they hold."""
        self._estimator.unload()
        if self._segmenter is not None:
            self._segmenter.unload()

    def __enter__(self) -> PipelineController:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _validated(self, image: ImageInput) -> AcquiredImage:
        if isinstance(image, AcquiredImage):
            return image
        return AcquiredImage(image=image, report=self._acquisition.validator.ensure_valid(image))

    def _run(self, before: AcquiredImage, after: AcquiredImage, timed: _Stages) -> ComparisonReport:
        warnings = [f"Imagen 'antes': {text}" for text in before.warnings]
        warnings += [f"Imagen 'después': {text}" for text in after.warnings]

        if not self.models_loaded:
            with timed("loading"):
                self._estimator.load()
                if self._segmenter is not None:
                    self._segmenter.load()

        with timed("depth"):
            depth_before = self._estimator.estimate(before.image)
            depth_after = self._estimator.estimate(after.image)
        warnings += self._depth_warnings(depth_before)

        with timed("alignment"):
            aligned = self._align(before.image, after.image, depth_after, warnings)

        with timed("segmentation"):
            segmentation_before = self._segment(before.image)
            segmentation_after = self._segment(aligned.image)
            self._check_indoor(segmentation_before, after.image)

        with timed("comparison"):
            camera = resolve_intrinsics(before.image, self._settings.camera, self._settings.camera_estimation)
            comparison = self._comparator.compare(
                before.image,
                aligned.image,
                depth_before,
                aligned.depth,
                camera.intrinsics,
                aligned.valid_mask,
                segmentation_before,
                segmentation_after,
            )

        with timed("rendering"):
            annotated = draw_changes(aligned.image.pixels, comparison.objects)

        return ComparisonReport(
            objects=comparison.objects,
            alignment=aligned.report,
            comparison=comparison,
            annotated_image=annotated,
            warnings=tuple(warnings),
            stage_seconds=timed.seconds,
            intrinsics_source=camera.source,
            focal_length_px=camera.intrinsics.fx,
        )

    def _align(
        self, before: ImageData, after: ImageData, depth_after: DepthMap, warnings: list[str]
    ) -> AlignmentResult:
        try:
            aligned = self._aligner.align(before, after, depth_after)
        except AlignmentError as exc:
            if not self._settings.alignment.fallback_to_identity:
                raise
            warnings.append(
                f"{exc.report.message} Las imágenes se compararon sin alinear, "
                "asumiendo que la cámara no se movió."
            )
            return self._aligner.unaligned(before, after, depth_after, exc.report)

        if aligned.overlap_fraction < LOW_OVERLAP_FRACTION:
            warnings.append(
                f"Solo el {aligned.overlap_fraction:.0%} de la escena aparece en ambas imágenes; "
                "los cambios fuera de esa zona no se pueden detectar."
            )
        return aligned

    def _segment(self, image: ImageData) -> Optional[SegmentationMask]:
        return self._segmenter.segment(image) if self._segmenter is not None else None

    def _check_indoor(self, segmentation_before: Optional[SegmentationMask], after_photo: ImageData) -> None:
        """Raise :class:`NonIndoorSceneError` naming each photo that shows an outdoor scene.

        Both photos are judged whole and as taken. The "after" photo gets a
        segmentation of its own instead of reusing the one of its aligned
        view: a wrong alignment between unrelated photos (a room and a street)
        can warp the outdoor view into something that looks like a room, and
        the aligned view would then let the outdoor photo in. The aligned
        segmentation still feeds the comparison and the object names. Needs
        the segmentation, so it is skipped when segmentation is disabled.
        """
        if not self._settings.segmentation.check_indoor_scene or self._segmenter is None:
            return
        outdoor = {}
        for name, mask in (("Antes", segmentation_before), ("Después", self._segment(after_photo))):
            if mask is None:
                continue
            is_indoor, percent = self._labeler.validate_indoor_scene(mask)
            if not is_indoor:
                outdoor[name] = percent
        if outdoor:
            raise NonIndoorSceneError(outdoor)

    def _depth_warnings(self, depth: DepthMap) -> list[str]:
        warnings = []
        if not depth.is_metric:
            warnings.append(
                "El modelo de profundidad no entrega distancias en metros; "
                "las medidas reportadas son aproximadas."
            )
        notice = getattr(getattr(self._estimator, "compute_device", None), "notice", None)
        if notice:
            warnings.append(notice)
        return warnings


class _Stages:
    """Announces each stage to the progress callback and records its duration."""

    def __init__(self, progress: Optional[ProgressCallback]) -> None:
        self._progress = progress
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if self._progress is not None:
            self._progress(stage)
        start = perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] = perf_counter() - start
