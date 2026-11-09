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


def normalize_class_name(name: str) -> str:
    return name.strip().lower()


def structural_class_ids(id2label: Mapping[int, str]) -> frozenset[int]:
    """Ids of ``id2label`` whose class is part of the scene structure."""
    return frozenset(
        class_id
        for class_id, name in id2label.items()
        if normalize_class_name(name) in ADE20K_STRUCTURAL_CLASSES
    )


class RegionLabeler:
    """Gives a changed region the name of the object class that dominates it.

    A region is named only when its most frequent non-structural class
    covers at least ``min_coverage`` of it and that class has a known
    translation; otherwise it is an unidentified object.
    """

    def __init__(
        self, min_coverage: float = 0.25, names: Mapping[str, str] = ADE20K_SPANISH_NAMES
    ) -> None:
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("Minimum coverage must be in (0, 1].")
        self._min_coverage = min_coverage
        self._names = {normalize_class_name(key): value for key, value in names.items()}

    def describe(self, mask: Optional[SegmentationMask], region: np.ndarray) -> SemanticLabel:
        if mask is None:
            return SemanticLabel(UNIDENTIFIED_LABEL)
        class_id, coverage = mask.dominant_class(region)
        if class_id is None or coverage < self._min_coverage:
            return SemanticLabel(UNIDENTIFIED_LABEL, coverage=coverage)
        class_name = normalize_class_name(mask.id2label.get(class_id, ""))
        spanish = self._names.get(class_name)
        if spanish is None:
            return SemanticLabel(UNIDENTIFIED_LABEL, coverage=coverage)
        return SemanticLabel(spanish, class_name=class_name, coverage=coverage)
