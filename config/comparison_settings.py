"""Parameters of the depth comparison and change classification."""

from pydantic import BaseModel, ConfigDict, Field


class ComparisonSettings(BaseModel):
    """Thresholds that turn a depth difference into changed objects."""

    model_config = ConfigDict(frozen=True)

    change_threshold_m: float = Field(
        default=0.05, gt=0.0, description="Minimum absolute depth difference counted as a change (m)."
    )
    normalize_depth_scale: bool = Field(
        default=True,
        description=(
            "Rescale the 'after' depth by the median depth ratio over the overlap, compensating "
            "the global scale drift between two independent monocular estimates."
        ),
    )
    appearance_gating: bool = Field(
        default=True,
        description=(
            "Accept a depth change only where the aligned images also differ in color; "
            "monocular estimates drift on surfaces that did not change, the pixels there do not."
        ),
    )
    appearance_threshold: float = Field(
        default=25.0,
        ge=0.0,
        le=255.0,
        description="Minimum per-channel difference (0-255) of the smoothed images counted as a visual change.",
    )
    appearance_blur_px: int = Field(
        default=5, ge=1, description="Box filter size applied to both images before comparing colors (px)."
    )
    morphology_kernel_px: int = Field(
        default=5, ge=1, description="Diameter of the opening/closing structuring element (px)."
    )
    min_region_fraction: float = Field(
        default=0.0015,
        gt=0.0,
        lt=1.0,
        description="Smallest changed region kept, as a fraction of the image area.",
    )
    pairing_min_similarity: float = Field(
        default=0.5,
        ge=-1.0,
        le=1.0,
        description="Minimum color-histogram correlation to treat a vacated and an occupied region as the same object.",
    )
    merge_adjacent_regions: bool = Field(
        default=True,
        description="Attach an unpaired region that touches an object's footprint to that object.",
    )
    merge_gap_fraction: float = Field(
        default=0.01,
        ge=0.0,
        lt=1.0,
        description="Largest gap between regions of one object, as a fraction of the image diagonal.",
    )
    drop_persistent_regions: bool = Field(
        default=True,
        description="Discard a removal or addition whose area shows the same thing in both views.",
    )
    persistence_agreement: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="Fraction of an area's pixels keeping their class in both views that marks it as unchanged.",
    )
    split_shared_regions: bool = Field(
        default=True,
        description="Divide by color a region shared by several objects (a mug placed on a box) among them.",
    )
    pair_by_label: bool = Field(
        default=True,
        description="Pair leftover regions recognized as the same kind of object (an object that turned while moving).",
    )
    pair_leftover_by_tracking: bool = Field(
        default=True,
        description=(
            "Pair the only unexplained vacated region with the only unexplained, non-persistent occupied "
            "region when keypoints that move with the object confirm it (an object whose colors change "
            "between views)."
        ),
    )
    track_displaced_objects: bool = Field(
        default=True,
        description="Measure displacements by matching keypoints of the object instead of region centroids.",
    )
    min_object_matches: int = Field(
        default=5, ge=3, description="Keypoints that must follow an object to trust the tracked displacement."
    )
    pairing_max_area_ratio: float = Field(
        default=3.0,
        ge=1.0,
        description="Largest area ratio between the two regions of a displaced object.",
    )
