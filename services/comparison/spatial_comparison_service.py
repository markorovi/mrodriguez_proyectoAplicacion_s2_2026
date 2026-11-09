"""3-D metrology of the changes between two aligned depth maps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.camera_intrinsics import CameraIntrinsics
from config.comparison_settings import ComparisonSettings
from models.comparison_report import (
    BoundingBox,
    ChangedObject,
    ChangeType,
    DepthComparison,
    Point3D,
)
from models.depth_map import DepthMap
from models.image_data import ImageData
from models.segmentation_mask import SegmentationMask
from services.inference.compute_device import ComputeDevice
from services.inference.device_manager import DeviceManager
from services.segmentation.region_labeler import RegionLabeler
from utils.geometry import backproject_depth

_HIST_BINS = (30, 32)
_HIST_RANGES = (0, 180, 0, 256)
_TRACK_RATIO = 0.75
_MIN_TRACK_MOTION_PX = 4.0
# Back-projection scores range 0-255; below this a pixel resembles no object.
_MIN_BACKPROJECTION = 12.0


@dataclass(frozen=True)
class _Region:
    """A connected area where the depth grew (``farther``) or shrank."""

    mask: np.ndarray
    bbox: BoundingBox
    area: int
    mean_change_m: float
    centroid: Point3D


@dataclass
class _Candidate:
    """Regions explaining one object: where it was (``before``) and where it is (``after``).

    The first region of each side is the primary one and gives the object's
    3-D position; further regions were merged in because they touch it.
    """

    change_type: ChangeType
    before: list[_Region] = field(default_factory=list)
    after: list[_Region] = field(default_factory=list)

    @property
    def regions(self) -> list[_Region]:
        return [*self.before, *self.after]


@dataclass
class _Views:
    """Both aligned views with their colors and camera-frame point maps."""

    before: ImageData
    after: ImageData
    points_before: torch.Tensor
    points_after: torch.Tensor
    difference: torch.Tensor

    @cached_property
    def hsv_before(self) -> np.ndarray:
        return cv2.cvtColor(self.before.pixels, cv2.COLOR_RGB2HSV)

    @cached_property
    def hsv_after(self) -> np.ndarray:
        return cv2.cvtColor(self.after.pixels, cv2.COLOR_RGB2HSV)


class SpatialComparisonService:
    """Finds, measures and classifies the objects that changed between two views.

    Inputs are the reference depth ``D_a`` and the "after" depth ``D_d'``
    already aligned onto the reference pixel grid. The service:

    1. optionally rescales ``D_d'`` by the median of ``D_a / D_d'`` over the
       overlap, removing the global scale drift between two independent
       monocular estimates (changes cover a minority of pixels, so the median
       is dominated by the static scene);
    2. computes ``dD = D_d' - D_a`` and thresholds ``|dD|`` at
       ``change_threshold_m``. With ``appearance_gating``, a pixel only
       counts when the aligned images also differ in color there: a monocular
       model re-estimates the whole room from each photo, so unchanged
       surfaces drift by several centimeters while their pixels stay the
       same. Each sign is then cleaned with a morphological opening and
       closing and regions smaller than ``min_region_fraction`` of the image
       are dropped;
    3. back-projects every pixel to the camera frame with the pinhole model
       and summarizes each region by the median 3-D point of its surface;
    4. pairs a region that got farther (vacated in "after") with a region
       that got closer (occupied in "after") when their colors match, which
       is a DISPLACED object; unpaired vacated regions are REMOVED objects
       (the background became visible) and unpaired occupied regions are
       ADDED objects. With ``merge_adjacent_regions``, an unpaired region
       that touches an object's footprint and looks like that object in the
       view where the object is visible becomes part of it instead of a new
       one: the revealed base of a removed object can be estimated slightly
       closer, which would otherwise read as a phantom addition. With
       ``drop_persistent_regions``, a removal or addition is discarded when
       the segmentation sees the same thing over its area in both views (a
       bed is still a bed): its depth step is an estimation artifact, and such
       regions are never offered as partners when splitting or pairing. With ``pair_by_label``, unexplained regions named as the
       same kind of object in their respective views are paired as a
       displacement: an object that turned shows other faces and colors;
    5. with ``track_displaced_objects``, measures a displacement by following
       the object itself: SIFT keypoints inside the object's area are matched
       between the aligned views, the static ones are discarded and the
       median 3-D motion of the rest is the displacement. When an object
       moves less than its own size, the vacated and occupied regions are
       thin slivers on opposite edges and their centroids are about one
       object width apart whatever the actual motion. Region centroids remain
       the fallback when too few keypoints follow the object.

    Objects that end up touching (a mug placed on a box) form a single
    occupied region, and objects that were touching form a single vacated
    one, which one-to-one pairing cannot explain. With
    ``split_shared_regions``, such a region is divided by color: the
    hue-saturation histogram of each unpaired region on the other side is
    back-projected onto it and every pixel goes to the object whose colors
    explain it best. Each part that resembles its object becomes that
    object's footprint; what no object explains stays an addition or removal.

    Tensor work (difference, scale, thresholds, color gating and
    back-projection) runs on
    the device chosen by ``device_manager`` (CUDA when available);
    morphology and connected components run in OpenCV.
    """

    def __init__(
        self,
        settings: ComparisonSettings | None = None,
        labeler: RegionLabeler | None = None,
        device_manager: DeviceManager | None = None,
    ) -> None:
        self._settings = settings or ComparisonSettings()
        self._labeler = labeler or RegionLabeler()
        self._device_manager = device_manager or DeviceManager()
        self._device: Optional[ComputeDevice] = None

    @property
    def settings(self) -> ComparisonSettings:
        return self._settings

    @property
    def device_name(self) -> str:
        return self._device.name if self._device else ""

    def compare(
        self,
        before: ImageData,
        after: ImageData,
        depth_before: DepthMap,
        depth_after: DepthMap,
        intrinsics: CameraIntrinsics,
        valid_mask: Optional[np.ndarray] = None,
        segmentation_before: Optional[SegmentationMask] = None,
        segmentation_after: Optional[SegmentationMask] = None,
    ) -> DepthComparison:
        """Compare two depth maps that share the pixel grid of ``before``."""
        shape = (before.height, before.width)
        for name, item_shape in (
            ("after image", (after.height, after.width)),
            ("before depth", depth_before.shape),
            ("after depth", depth_after.shape),
        ):
            if item_shape != shape:
                raise ValueError(f"The {name} has shape {item_shape}; expected {shape}.")
        if valid_mask is None:
            valid_mask = np.ones(shape, dtype=bool)
        elif valid_mask.shape != shape:
            raise ValueError("The valid mask must have the shape of the images.")
        if (intrinsics.reference_width, intrinsics.reference_height) != (before.width, before.height):
            intrinsics = intrinsics.scaled_to(before.width, before.height)

        device = self._select_device().torch_device
        d_before = torch.from_numpy(np.ascontiguousarray(depth_before.values, np.float32)).to(device)
        d_after = torch.from_numpy(np.ascontiguousarray(depth_after.values, np.float32)).to(device)
        valid = torch.from_numpy(np.ascontiguousarray(valid_mask, bool)).to(device)
        valid &= torch.isfinite(d_before) & torch.isfinite(d_after) & (d_before > 0) & (d_after > 0)

        scale = self._scale_factor(d_before, d_after, valid)
        d_after = d_after * scale
        difference = torch.where(valid, d_after - d_before, torch.zeros_like(d_before))

        threshold = self._settings.change_threshold_m
        farther, closer = difference >= threshold, difference <= -threshold
        if self._settings.appearance_gating:
            looks_different = self._appearance_change(before, after, device)
            farther, closer = farther & looks_different, closer & looks_different
        farther = self._clean(farther.cpu().numpy())
        closer = self._clean(closer.cpu().numpy())

        points_before = backproject_depth(d_before, intrinsics)
        points_after = backproject_depth(d_after, intrinsics)
        vacated = self._regions(farther, difference, points_before)
        occupied = self._regions(closer, difference, points_after)

        views = _Views(before, after, points_before, points_after, difference)
        objects = self._classify(vacated, occupied, views, segmentation_before, segmentation_after)
        return DepthComparison(
            objects=tuple(objects),
            difference_m=difference.cpu().numpy(),
            farther_mask=farther,
            closer_mask=closer,
            scale_factor=scale,
        )

    def _select_device(self) -> ComputeDevice:
        if self._device is None:
            self._device = self._device_manager.select()
        return self._device

    def _scale_factor(self, d_before: torch.Tensor, d_after: torch.Tensor, valid: torch.Tensor) -> float:
        if not self._settings.normalize_depth_scale or not bool(valid.any()):
            return 1.0
        ratio = float(torch.median(d_before[valid] / d_after[valid]))
        return ratio if np.isfinite(ratio) and ratio > 0 else 1.0

    def _appearance_change(self, before: ImageData, after: ImageData, device: torch.device) -> torch.Tensor:
        """Pixels whose smoothed, gain-compensated color differs, dilated to tolerate small misalignment."""
        size = self._settings.appearance_blur_px
        images = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(image.pixels)) for image in (before, after)]
        ).to(device)
        smoothed = F.avg_pool2d(
            images.permute(0, 3, 1, 2).float(), size, stride=1, padding=size // 2, count_include_pad=False
        )
        # Per-channel gain compensation: a global exposure or lighting change
        # must not read as a change of every pixel.
        flat = smoothed.flatten(2)
        gain = flat[0].median(dim=1).values / flat[1].median(dim=1).values.clamp_min(1.0)
        compensated = (smoothed[1] * gain[:, None, None]).clamp(0.0, 255.0)
        changed = (smoothed[0] - compensated).abs().amax(dim=0) >= self._settings.appearance_threshold
        reach = self._settings.morphology_kernel_px | 1
        dilated = F.max_pool2d(changed[None, None].float(), reach, stride=1, padding=reach // 2)
        return dilated[0, 0, : changed.shape[0], : changed.shape[1]].bool()

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """Opening removes speckles; closing fills small holes inside objects."""
        size = self._settings.morphology_kernel_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned.astype(bool)

    def _regions(self, mask: np.ndarray, difference: torch.Tensor, points: torch.Tensor) -> list[_Region]:
        min_area = self._settings.min_region_fraction * mask.size
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        labels_t = torch.from_numpy(labels).to(points.device)
        regions = []
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < min_area:
                continue
            selected = labels_t == index
            centroid = torch.median(points[selected], dim=0).values.tolist()
            regions.append(
                _Region(
                    mask=labels == index,
                    bbox=BoundingBox(x, y, w, h),
                    area=area,
                    mean_change_m=float(difference[selected].mean()),
                    centroid=Point3D(*centroid),
                )
            )
        return regions

    def _classify(
        self,
        vacated: list[_Region],
        occupied: list[_Region],
        views: _Views,
        segmentation_before: Optional[SegmentationMask],
        segmentation_after: Optional[SegmentationMask],
    ) -> list[ChangedObject]:
        segmented = segmentation_before is not None and segmentation_after is not None
        filtering = self._settings.drop_persistent_regions and segmented
        # Regions showing the same thing in both views are estimation
        # artifacts. They are kept until the end (a slightly moved object or a
        # poorly segmented one can look persistent and still pair correctly),
        # but never offered as a partner to another object.
        artifacts = {
            id(r) for r in (*vacated, *occupied)
            if filtering and self._persists([r], segmentation_before, segmentation_after)
        }

        pairs = dict(self._pair(vacated, occupied, views))  # vacated index -> occupied index
        split_candidates: list[_Candidate] = []
        remainders: list[tuple[_Region, bool]] = []
        divided: set[int] = set()
        if self._settings.split_shared_regions:
            split_candidates, remainders, divided = self._split_shared(vacated, occupied, pairs, views, artifacts)
        candidates = [_Candidate(ChangeType.DISPLACED, [vacated[i]], [occupied[j]]) for i, j in pairs.items()]
        candidates += split_candidates

        # A divided region is replaced by its parts and its unexplained rest.
        used = {id(region) for candidate in candidates for region in candidate.regions} | divided
        leftovers = [(region, True) for region in vacated if id(region) not in used]
        leftovers += [(region, False) for region in occupied if id(region) not in used]
        leftovers += remainders
        if self._settings.pair_by_label and segmented:
            label_pairs, leftovers = self._pair_by_label(
                leftovers, segmentation_before, segmentation_after, artifacts
            )
            candidates += label_pairs
        # Strongest evidence first, so it becomes the primary region of its
        # object: estimation artifacts are weaker depth steps than real objects.
        leftovers.sort(key=lambda item: abs(item[0].mean_change_m) * item[0].area, reverse=True)
        reach = self._merge_reach(views.before)
        for region, is_vacated in leftovers:
            side = self._touching_side(region, candidates, reach, views) if reach else None
            if side is not None:
                side.append(region)
            elif is_vacated:
                candidates.append(_Candidate(ChangeType.REMOVED, before=[region]))
            else:
                candidates.append(_Candidate(ChangeType.ADDED, after=[region]))

        if filtering:
            candidates = [
                c for c in candidates
                if c.change_type is ChangeType.DISPLACED
                or not self._persists(c.regions, segmentation_before, segmentation_after)
            ]
        found = [self._changed(c, views, segmentation_before, segmentation_after) for c in candidates]
        found.sort(key=lambda obj: obj.area_px, reverse=True)
        return [replace(obj, object_id=number) for number, obj in enumerate(found, start=1)]

    def _persists(
        self, regions: list[_Region], segmentation_before: SegmentationMask, segmentation_after: SegmentationMask
    ) -> bool:
        """True when both views see the same thing over the area of ``regions``.

        Either the same object class names the area in both views, or most of
        its pixels keep their class (for example a wall and bed edge that the
        depth model re-estimated). A real change shows an object in one view
        and what was behind it in the other.
        """
        area = _union_mask(regions)
        before = self._labeler.describe(segmentation_before, area)
        after = self._labeler.describe(segmentation_after, area)
        if before.is_identified and after.is_identified and before.class_name == after.class_name:
            return True
        agreement = np.mean(segmentation_before.class_ids[area] == segmentation_after.class_ids[area])
        return bool(agreement >= self._settings.persistence_agreement)

    def _pair_by_label(
        self,
        leftovers: list[tuple[_Region, bool]],
        segmentation_before: SegmentationMask,
        segmentation_after: SegmentationMask,
        artifacts: set[int],
    ) -> tuple[list[_Candidate], list[tuple[_Region, bool]]]:
        """Pair unexplained regions that the segmentation names as the same kind of object.

        Catches an object that turned while moving and shows other faces, so
        its colors no longer match. A vacated region is named from "before"
        (where the object was visible) and an occupied one from "after".
        """
        named = []
        for region, is_vacated in leftovers:
            label = self._labeler.describe(segmentation_before if is_vacated else segmentation_after, region.mask)
            usable = label.is_identified and id(region) not in artifacts
            named.append(label.class_name if usable else None)

        options = []
        for i, (old, old_vacated) in enumerate(leftovers):
            for j, (new, new_vacated) in enumerate(leftovers):
                if not old_vacated or new_vacated or named[i] is None or named[i] != named[j]:
                    continue
                ratio = max(old.area, new.area) / min(old.area, new.area)
                if ratio <= self._settings.pairing_max_area_ratio:
                    options.append((ratio, i, j))

        candidates, used = [], set()
        for _, i, j in sorted(options):
            if i not in used and j not in used:
                candidates.append(_Candidate(ChangeType.DISPLACED, [leftovers[i][0]], [leftovers[j][0]]))
                used.update((i, j))
        return candidates, [item for k, item in enumerate(leftovers) if k not in used]

    def _merge_reach(self, image: ImageData) -> int:
        """Largest gap (px) between two regions that still makes them one object."""
        if not self._settings.merge_adjacent_regions:
            return 0
        return max(1, round(self._settings.merge_gap_fraction * float(np.hypot(image.width, image.height))))

    def _touching_side(
        self, region: _Region, candidates: list[_Candidate], reach: int, views: _Views
    ) -> Optional[list[_Region]]:
        """The footprint list (``before`` or ``after``) of the object ``region`` belongs to.

        The region must touch a region of that footprint and resemble it in
        the view where that footprint shows the object: "before" for the old
        footprint, "after" for the new one.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        grown = cv2.dilate(region.mask.astype(np.uint8), kernel).astype(bool)
        for candidate in candidates:
            for side, hsv in ((candidate.before, views.hsv_before), (candidate.after, views.hsv_after)):
                for other in side:
                    if not (_boxes_near(region.bbox, other.bbox, reach) and (grown & other.mask).any()):
                        continue
                    similarity = cv2.compareHist(
                        _histogram(hsv, region.mask), _histogram(hsv, other.mask), cv2.HISTCMP_CORREL
                    )
                    if similarity >= self._settings.pairing_min_similarity:
                        return side
        return None

    def _split_shared(
        self,
        vacated: list[_Region],
        occupied: list[_Region],
        pairs: dict[int, int],
        views: _Views,
        artifacts: set[int],
    ) -> tuple[list[_Candidate], list[tuple[_Region, bool]], set[int]]:
        """Divide regions shared by several objects among them.

        Occupied regions are tried first (objects brought together), then
        vacated ones (objects that were together). ``pairs`` is updated in
        place when a split replaces a one-to-one pair. Returns the new
        displacements, the unexplained rests (region, is_vacated) and the ids
        of the regions that were divided.
        """
        candidates: list[_Candidate] = []
        remainders: list[tuple[_Region, bool]] = []
        divided: set[int] = set()
        consumed_vacated: set[int] = set()
        consumed_occupied: set[int] = set()

        for shared_in_after in (True, False):
            shared_side, other_side = (occupied, vacated) if shared_in_after else (vacated, occupied)
            for k in sorted(range(len(shared_side)), key=lambda n: shared_side[n].area, reverse=True):
                if k in (consumed_occupied if shared_in_after else consumed_vacated):
                    continue
                if id(shared_side[k]) in artifacts:
                    continue
                if shared_in_after:
                    current = [i for i, j in pairs.items() if j == k]
                    free = [i for i in range(len(vacated)) if i not in pairs and i not in consumed_vacated]
                else:
                    current = [pairs[k]] if k in pairs else []
                    paired_occupied = set(pairs.values())
                    free = [j for j in range(len(occupied)) if j not in paired_occupied and j not in consumed_occupied]
                free = [n for n in free if id(other_side[n]) not in artifacts]
                if not free:
                    continue
                sources = current + free
                parts = self._divide(shared_side[k], [other_side[n] for n in sources], views, shared_in_after)
                accepted = {sources[n]: part for n, part in parts.items()}
                # A region already paired one-to-one is only re-divided when it
                # demonstrably holds more than one object.
                if not accepted or (current and len(accepted) < 2):
                    continue

                for n in current:
                    if shared_in_after:
                        del pairs[n]
                    else:
                        pairs.pop(k, None)
                explained = np.zeros_like(shared_side[k].mask)
                for n, part in accepted.items():
                    piece = self._region_from_mask(part, views, in_after=shared_in_after)
                    other = other_side[n]
                    if shared_in_after:
                        candidates.append(_Candidate(ChangeType.DISPLACED, [other], [piece]))
                        consumed_vacated.add(n)
                    else:
                        candidates.append(_Candidate(ChangeType.DISPLACED, [piece], [other]))
                        consumed_occupied.add(n)
                    explained |= part
                (consumed_occupied if shared_in_after else consumed_vacated).add(k)
                divided.add(id(shared_side[k]))

                rest = _largest_component(self._clean(shared_side[k].mask & ~explained))
                if rest.sum() >= self._settings.min_region_fraction * rest.size:
                    remainders.append((self._region_from_mask(rest, views, in_after=shared_in_after), not shared_in_after))

        # Consumed one-to-one partners are no longer leftovers.
        for i in consumed_vacated:
            pairs.pop(i, None)
        for j in consumed_occupied:
            for i in [i for i, jj in pairs.items() if jj == j]:
                del pairs[i]
        return candidates, remainders, divided

    def _divide(
        self, shared: _Region, objects: list[_Region], views: _Views, shared_in_after: bool
    ) -> dict[int, np.ndarray]:
        """Parts of ``shared`` explained by each of ``objects`` (index -> full-size mask)."""
        shared_hsv = views.hsv_after if shared_in_after else views.hsv_before
        object_hsv = views.hsv_before if shared_in_after else views.hsv_after
        box = shared.bbox
        window = (slice(box.y, box.y + box.height), slice(box.x, box.x + box.width))
        crop = np.ascontiguousarray(shared_hsv[window])
        inside = shared.mask[window]

        scores = []
        for obj in objects:
            hist = cv2.calcHist([object_hsv], [0, 1], obj.mask.astype(np.uint8), list(_HIST_BINS), list(_HIST_RANGES))
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            projected = cv2.calcBackProject([crop], [0, 1], hist, list(_HIST_RANGES), 1).astype(np.float32)
            scores.append(cv2.GaussianBlur(projected, (0, 0), 3))
        scores = np.stack(scores)
        owner = scores.argmax(axis=0)
        owned = inside & (scores.max(axis=0) >= _MIN_BACKPROJECTION)

        min_area = self._settings.min_region_fraction * shared.mask.size
        parts = {}
        for n, obj in enumerate(objects):
            part = np.zeros_like(shared.mask)
            part[window] = _largest_component(self._clean(owned & (owner == n)))
            area = int(part.sum())
            if area < min_area or max(area, obj.area) / min(area, obj.area) > self._settings.pairing_max_area_ratio:
                continue
            similarity = cv2.compareHist(
                _histogram(object_hsv, obj.mask), _histogram(shared_hsv, part), cv2.HISTCMP_CORREL
            )
            if similarity >= self._settings.pairing_min_similarity:
                parts[n] = part
        return parts

    def _region_from_mask(self, mask: np.ndarray, views: _Views, in_after: bool) -> _Region:
        points = views.points_after if in_after else views.points_before
        rows, cols = np.nonzero(mask)
        selected = torch.from_numpy(mask).to(points.device)
        return _Region(
            mask=mask,
            bbox=BoundingBox(
                int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)
            ),
            area=int(mask.sum()),
            mean_change_m=float(views.difference[selected].mean()),
            centroid=Point3D(*torch.median(points[selected], dim=0).values.tolist()),
        )

    def _pair(
        self, vacated: list[_Region], occupied: list[_Region], views: _Views
    ) -> list[tuple[int, int]]:
        """Greedy one-to-one pairing by color similarity of the object's two appearances."""
        if not vacated or not occupied:
            return []
        hists_before = [_histogram(views.hsv_before, region.mask) for region in vacated]
        hists_after = [_histogram(views.hsv_after, region.mask) for region in occupied]

        candidates = []
        for i, old in enumerate(vacated):
            for j, new in enumerate(occupied):
                ratio = max(old.area, new.area) / min(old.area, new.area)
                if ratio > self._settings.pairing_max_area_ratio:
                    continue
                similarity = cv2.compareHist(hists_before[i], hists_after[j], cv2.HISTCMP_CORREL)
                if similarity >= self._settings.pairing_min_similarity:
                    candidates.append((similarity, i, j))

        pairs, used_i, used_j = [], set(), set()
        for _, i, j in sorted(candidates, reverse=True):
            if i not in used_i and j not in used_j:
                pairs.append((i, j))
                used_i.add(i)
                used_j.add(j)
        return pairs

    def _changed(
        self,
        candidate: _Candidate,
        views: _Views,
        segmentation_before: Optional[SegmentationMask],
        segmentation_after: Optional[SegmentationMask],
    ) -> ChangedObject:
        regions = candidate.regions
        area = sum(region.area for region in regions)
        magnitude = sum(abs(region.mean_change_m) * region.area for region in regions) / area

        # The object is visible in "before" over its old footprint and in
        # "after" over its new one; it is named from whichever view the
        # segmentation recognizes it best.
        labels = []
        if candidate.before:
            labels.append(self._labeler.describe(segmentation_before, _union_mask(candidate.before)))
        if candidate.after:
            labels.append(self._labeler.describe(segmentation_after, _union_mask(candidate.after)))
        label = max(labels, key=lambda item: (item.is_identified, item.coverage))

        position_before = candidate.before[0].centroid if candidate.before else None
        position_after = candidate.after[0].centroid if candidate.after else None
        if candidate.change_type is ChangeType.DISPLACED and self._settings.track_displaced_objects:
            tracked = self._track(candidate, views)
            if tracked is not None:
                position_before, position_after = tracked

        return ChangedObject(
            object_id=0,
            change_type=candidate.change_type,
            label=label,
            depth_change_cm=magnitude * 100.0,
            area_px=area,
            bbox_before=_union_bbox(candidate.before),
            bbox_after=_union_bbox(candidate.after),
            position_before=position_before,
            position_after=position_after,
        )

    def _track(self, candidate: _Candidate, views: _Views) -> Optional[tuple[Point3D, Point3D]]:
        """Positions of a displaced object from keypoints that move with it, or None."""
        area = _union_bbox(candidate.regions)
        margin_x, margin_y = round(0.15 * area.width), round(0.15 * area.height)
        x0, y0 = max(0, area.x - margin_x), max(0, area.y - margin_y)
        x1 = min(views.before.width, area.x + area.width + margin_x)
        y1 = min(views.before.height, area.y + area.height + margin_y)

        sift = cv2.SIFT_create()
        found = []
        for image in (views.before, views.after):
            gray = cv2.cvtColor(np.ascontiguousarray(image.pixels[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 2:
                return None
            found.append((np.float32([kp.pt for kp in keypoints]) + (x0, y0), descriptors))
        (points_b, desc_b), (points_a, desc_a) = found

        pairs = [
            pair[0]
            for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_b, desc_a, k=2)
            if len(pair) == 2 and pair[0].distance <= _TRACK_RATIO * pair[1].distance
        ]
        if len(pairs) < self._settings.min_object_matches:
            return None
        src = points_b[[m.queryIdx for m in pairs]]
        dst = points_a[[m.trainIdx for m in pairs]]
        motion = dst - src

        # Background keypoints barely move once the views are aligned.
        moving = np.linalg.norm(motion, axis=1) >= max(_MIN_TRACK_MOTION_PX, 0.03 * np.hypot(x1 - x0, y1 - y0))
        if moving.sum() < self._settings.min_object_matches:
            return None
        keep = _largest_consistent_group(motion, moving)
        if keep.sum() < self._settings.min_object_matches:
            return None

        before_3d = _sample(views.points_before, src[keep])
        after_3d = _sample(views.points_after, dst[keep])
        start = torch.median(before_3d, dim=0).values
        shift = torch.median(after_3d - before_3d, dim=0).values
        return Point3D(*start.tolist()), Point3D(*(start + shift).tolist())


def _histogram(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalized hue-saturation histogram of the pixels under ``mask``."""
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), list(_HIST_BINS), list(_HIST_RANGES))
    return cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)


def _union_mask(regions: list[_Region]) -> np.ndarray:
    mask = np.zeros_like(regions[0].mask)
    for region in regions:
        mask |= region.mask
    return mask


def _union_bbox(regions: list[_Region]) -> Optional[BoundingBox]:
    if not regions:
        return None
    x0 = min(r.bbox.x for r in regions)
    y0 = min(r.bbox.y for r in regions)
    x1 = max(r.bbox.x + r.bbox.width for r in regions)
    y1 = max(r.bbox.y + r.bbox.height for r in regions)
    return BoundingBox(x0, y0, x1 - x0, y1 - y0)


def _boxes_near(a: BoundingBox, b: BoundingBox, gap: int) -> bool:
    """Cheap pre-check: the boxes overlap once one is grown by ``gap`` pixels."""
    return (
        a.x - gap <= b.x + b.width
        and b.x <= a.x + a.width + gap
        and a.y - gap <= b.y + b.height
        and b.y <= a.y + a.height + gap
    )


def _largest_consistent_group(motion: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Mask of the biggest set of ``candidates`` whose 2-D motions agree.

    Each candidate's motion is tried as the reference; the one with the most
    neighbours within 25 % of its length (at least a few pixels) wins.
    Mismatched keypoints scatter, while those on the moving object share one
    motion, so they form the largest group.
    """
    best = np.zeros(len(motion), dtype=bool)
    for reference in motion[candidates]:
        tolerance = max(_MIN_TRACK_MOTION_PX, 0.25 * float(np.linalg.norm(reference)))
        group = candidates & (np.linalg.norm(motion - reference, axis=1) <= tolerance)
        if group.sum() > best.sum():
            best = group
    return best


def _sample(points: torch.Tensor, pixels: np.ndarray) -> torch.Tensor:
    """3-D points of an ``H x W x 3`` point map at sub-pixel positions (nearest pixel)."""
    height, width = points.shape[:2]
    cols = np.clip(np.rint(pixels[:, 0]), 0, width - 1).astype(np.int64)
    rows = np.clip(np.rint(pixels[:, 1]), 0, height - 1).astype(np.int64)
    index = torch.from_numpy(rows * width + cols).to(points.device)
    return points.reshape(-1, 3)[index]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """The biggest connected blob of ``mask``: the body of an object, without stray pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
