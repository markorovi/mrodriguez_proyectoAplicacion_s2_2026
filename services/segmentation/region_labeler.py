"""Names changed regions in Spanish from the classes of a segmentation mask."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from models.segmentation_mask import UNIDENTIFIED_LABEL, SegmentationMask, SemanticLabel

# ADE20K class names (as published with the SegFormer checkpoints) of movable
# objects, translated to the names shown to the user.
ADE20K_SPANISH_NAMES: Mapping[str, str] = {
    "chair": "silla",
    "armchair": "sillón",
    "swivel chair": "silla giratoria",
    "seat": "asiento",
    "stool": "taburete",
    "bench": "banca",
    "sofa": "sofá",
    "ottoman": "puf",
    "table": "mesa",
    "coffee table": "mesa de centro",
    "desk": "escritorio",
    "pool table": "mesa de billar",
    "box": "caja",
    "case": "vitrina",
    "chest of drawers": "cómoda",
    "cabinet": "gabinete",
    "wardrobe": "ropero",
    "bookcase": "librero",
    "shelf": "estante",
    "bed": "cama",
    "cradle": "cuna",
    "cushion": "cojín",
    "pillow": "almohada",
    "blanket": "cobija",
    "lamp": "lámpara",
    "sconce": "aplique de pared",
    "plant": "planta",
    "flower": "flor",
    "vase": "florero",
    "pot": "maceta",
    "basket": "canasta",
    "bag": "bolso",
    "bottle": "botella",
    "glass": "vaso",
    "plate": "plato",
    "tray": "bandeja",
    "book": "libro",
    "computer": "computadora",
    "monitor": "monitor",
    "screen": "pantalla",
    "crt screen": "monitor",
    "television receiver": "televisor",
    "fan": "ventilador",
    "clock": "reloj",
    "ashcan": "basurero",
    "barrel": "barril",
    "plaything": "juguete",
    "ball": "pelota",
    "towel": "toalla",
    "apparel": "ropa",
    "painting": "cuadro",
    "poster": "póster",
    "mirror": "espejo",
    "rug": "alfombra",
    "curtain": "cortina",
    "person": "persona",
    "animal": "animal",
    "bicycle": "bicicleta",
    "microwave": "microondas",
    "refrigerator": "refrigerador",
    "oven": "horno",
    "stove": "cocina",
    "washer": "lavadora",
    "dishwasher": "lavavajillas",
    "radiator": "radiador",
    "sculpture": "escultura",
    "bulletin board": "pizarra de anuncios",
    "food": "comida",
}

# Scene structure: surfaces revealed when an object leaves, never the object itself.
ADE20K_STRUCTURAL_CLASSES: frozenset[str] = frozenset(
    {
        "wall", "building", "sky", "floor", "tree", "ceiling", "road", "grass",
        "sidewalk", "earth", "mountain", "water", "sea", "field", "sand", "river",
        "hill", "land", "path", "runway", "dirt track", "stairs", "stairway", "step",
        "windowpane", "door", "screen door", "column", "fence", "railing",
        "bannister", "house", "skyscraper", "tower", "bridge", "pier", "lake",
        "waterfall", "swimming pool", "stage", "base", "countertop", "counter",
        "kitchen island", "blind", "fireplace",
    }
)


# Open-air scenery: a photo largely covered by these was not taken indoors.
# "plant" is left out on purpose: potted plants are common indoor objects.
ADE20K_OUTDOOR_CLASSES: frozenset[str] = frozenset(
    {
        "sky", "tree", "grass", "earth", "mountain", "sea", "field", "building",
        "skyscraper", "road", "sidewalk", "sand", "river", "hill", "land", "rock",
        "palm", "lake", "waterfall", "water", "house", "path",
    }
)

# Largest share of a photo (percent) that outdoor classes may cover in an indoor scene.
MAX_OUTDOOR_PERCENT = 25.0


def normalize_class_name(name: str) -> str:
    return name.strip().lower()


def structural_class_ids(id2label: Mapping[int, str]) -> frozenset[int]:
    """Ids of ``id2label`` whose class is part of the scene structure."""
    return _ids_of(id2label, ADE20K_STRUCTURAL_CLASSES)


def outdoor_class_ids(id2label: Mapping[int, str]) -> frozenset[int]:
    """Ids of ``id2label`` whose class belongs to open-air scenery.

    Ids are resolved by name, like :func:`structural_class_ids`, so they
    follow the numbering of whichever checkpoint produced the mask.
    """
    return _ids_of(id2label, ADE20K_OUTDOOR_CLASSES)


def _ids_of(id2label: Mapping[int, str], names: frozenset[str]) -> frozenset[int]:
    return frozenset(class_id for class_id, name in id2label.items() if normalize_class_name(name) in names)


class RegionLabeler:
    """Gives a changed region the name of the object class that dominates it.

    A region is named only when its most frequent non-structural class
    covers at least ``min_coverage`` of it and that class has a known
    translation; otherwise it is an unidentified object.

    Names meant for display (``describe(..., apply_confidence=True)``) are
    also withheld when the model's mean confidence in that class is below
    ``min_label_confidence``: a small object the model does not know gets the
    closest of its classes (a mug read as a bag), and showing it as
    unidentified is better than a wrong name. Callers that compare regions
    (persistence, pairing by label) keep the default and are not affected.

    It also tells whether a whole photo shows an indoor scene
    (:meth:`validate_indoor_scene`).
    """

    def __init__(
        self,
        min_coverage: float = 0.25,
        names: Mapping[str, str] = ADE20K_SPANISH_NAMES,
        max_outdoor_percent: float = MAX_OUTDOOR_PERCENT,
        min_label_confidence: float = 0.76,
    ) -> None:
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("Minimum coverage must be in (0, 1].")
        if not 0.0 <= max_outdoor_percent <= 100.0:
            raise ValueError("Maximum outdoor percentage must be in [0, 100].")
        if not 0.0 <= min_label_confidence <= 1.0:
            raise ValueError("Minimum label confidence must be in [0, 1].")
        self._min_coverage = min_coverage
        self._names = {normalize_class_name(key): value for key, value in names.items()}
        self._max_outdoor_percent = max_outdoor_percent
        self._min_label_confidence = min_label_confidence

    def validate_indoor_scene(
        self, mask: SegmentationMask, region: Optional[np.ndarray] = None
    ) -> tuple[bool, float]:
        """``(is_indoor, outdoor_percent)`` for the photo segmented in ``mask``.

        ``outdoor_percent`` is the share (0-100) of the pixels, or of the
        pixels inside ``region`` when given, that show open-air scenery
        (:data:`ADE20K_OUTDOOR_CLASSES`). The scene counts as indoor while
        that share is at most ``max_outdoor_percent``; a window showing the
        sky or a tree does not make a room an outdoor scene.
        """
        ids = mask.class_ids if region is None else mask.class_ids[region.astype(bool)]
        if ids.size == 0:
            return True, 0.0
        outdoor = np.isin(ids, np.fromiter(outdoor_class_ids(mask.id2label), dtype=np.int64))
        percent = 100.0 * float(np.count_nonzero(outdoor)) / ids.size
        return percent <= self._max_outdoor_percent, percent

    def describe(
        self, mask: Optional[SegmentationMask], region: np.ndarray, apply_confidence: bool = False
    ) -> SemanticLabel:
        """Name of the object class that dominates ``region``.

        When the mask carries the model's confidence, the label reports its
        mean over the pixels of that class. With ``apply_confidence`` (only
        for the name shown to the user), a class below
        ``min_label_confidence`` gives an unidentified object; the default
        leaves the result as it is without the confidence rule.
        """
        if mask is None:
            return SemanticLabel(UNIDENTIFIED_LABEL)
        class_id, coverage = mask.dominant_class(region)
        if class_id is None or coverage < self._min_coverage:
            return SemanticLabel(UNIDENTIFIED_LABEL, coverage=coverage)
        confidence = mask.class_confidence(region, class_id)
        class_name = normalize_class_name(mask.id2label.get(class_id, ""))
        spanish = self._names.get(class_name)
        doubtful = apply_confidence and confidence is not None and confidence < self._min_label_confidence
        if spanish is None or doubtful:
            return SemanticLabel(UNIDENTIFIED_LABEL, coverage=coverage, confidence=confidence)
        return SemanticLabel(spanish, class_name=class_name, coverage=coverage, confidence=confidence)
