"""End-to-end orchestration of the scene change analysis."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator, Optional, Union

from config.settings import AppSettings
from models.acquired_image import AcquiredImage
from models.alignment import AlignmentResult
from models.comparison_report import ComparisonReport
from models.depth_map import DepthMap
from models.exceptions import AlignmentError
from models.image_data import ImageData
from models.segmentation_mask import SegmentationMask
from services.alignment.spatial_alignment_service import SpatialAlignmentService
from services.capture.acquisition_service import ImageAcquisitionService
from services.comparison.spatial_comparison_service import SpatialComparisonService
from services.inference.base_estimator import BaseDepthEstimator
from services.inference.device_manager import DeviceManager
from services.inference.estimator_factory import create_depth_estimator
from services.segmentation.region_labeler import RegionLabeler
from services.segmentation.segformer_segmenter import SegFormerSegmenter
from utils.change_visualization import draw_changes
from utils.image_validator import ImageValidator

ImageInput = Union[AcquiredImage, ImageData]

LOW_OVERLAP_FRACTION = 0.5


class PipelineController:
    """Single entry point that turns a "before"/"after" pair into a report.

    Stages, in order: acquisition and validation, monocular metric depth of
    both views, spatial alignment of "after" onto "before", semantic
    segmentation of both views, 3-D comparison and classification, and
    rendering of the annotated result. Each collaborator can be injected,
    which keeps the controller independent of concrete models and testable
    without GPU or network.

    Models are loaded on the first analysis and kept in memory for the next
    ones; :meth:`close` (or leaving a ``with`` block) frees them. Errors
    raised by the stages derive from :class:`models.exceptions.SceneDiffError`.
    A failed alignment does not stop the analysis when
    ``settings.alignment.fallback_to_identity`` is set: the views are then
    compared unaligned and the report carries a warning.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        acquisition: ImageAcquisitionService | None = None,
        depth_estimator: BaseDepthEstimator | None = None,
        aligner: SpatialAlignmentService | None = None,
        segmenter: SegFormerSegmenter | None = None,
        comparator: SpatialComparisonService | None = None,
    ) -> None:
        self._settings = settings or AppSettings()
        s = self._settings
        self._acquisition = acquisition or ImageAcquisitionService(ImageValidator(s.image), s.capture)
        self._estimator = depth_estimator or create_depth_estimator(s.depth)
        self._aligner = aligner or SpatialAlignmentService(s.alignment)
        if segmenter is None and s.segmentation.enabled:
            segmenter = SegFormerSegmenter(s.segmentation)
        self._segmenter = segmenter if s.segmentation.enabled else None
        self._comparator = comparator or SpatialComparisonService(
            s.comparison,
            RegionLabeler(s.segmentation.min_label_coverage),
            DeviceManager(s.depth.device),
        )

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def analyze_files(self, before: Union[str, Path], after: Union[str, Path]) -> ComparisonReport:
        """Load, validate and compare two image files."""
        timings: dict[str, float] = {}
        with _timed(timings, "acquisition"):
            acquired = (self._acquisition.from_file(before), self._acquisition.from_file(after))
        return self._run(*acquired, timings)

    def analyze(self, before: ImageInput, after: ImageInput) -> ComparisonReport:
        """Compare two images already in memory (validated here if needed)."""
        timings: dict[str, float] = {}
        with _timed(timings, "acquisition"):
            acquired = (self._validated(before), self._validated(after))
        return self._run(*acquired, timings)

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

    def _run(self, before: AcquiredImage, after: AcquiredImage, timings: dict[str, float]) -> ComparisonReport:
        warnings = [f"Imagen 'antes': {text}" for text in before.warnings]
        warnings += [f"Imagen 'después': {text}" for text in after.warnings]

        with _timed(timings, "depth"):
            depth_before = self._estimator.estimate(before.image)
            depth_after = self._estimator.estimate(after.image)
        warnings += self._depth_warnings(depth_before)

        with _timed(timings, "alignment"):
            aligned = self._align(before.image, after.image, depth_after, warnings)

        with _timed(timings, "segmentation"):
            segmentation_before = self._segment(before.image)
            segmentation_after = self._segment(aligned.image)

        with _timed(timings, "comparison"):
            comparison = self._comparator.compare(
                before.image,
                aligned.image,
                depth_before,
                aligned.depth,
                self._settings.camera,
                aligned.valid_mask,
                segmentation_before,
                segmentation_after,
            )

        with _timed(timings, "rendering"):
            annotated = draw_changes(aligned.image.pixels, comparison.objects)

        return ComparisonReport(
            objects=comparison.objects,
            alignment=aligned.report,
            comparison=comparison,
            annotated_image=annotated,
            warnings=tuple(warnings),
            stage_seconds=timings,
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


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        timings[stage] = perf_counter() - start
