"""Discovery of the real photo pairs placed in ``data/samples``.

A pair is ``real_before<suffix>.<ext>`` plus ``real_after<suffix>.<ext>``,
for example ``real_before.jpg``/``real_after.jpg`` or
``real_before_2.jpg``/``real_after_2.jpg``. Any accepted image extension
works. Each pair is named after its suffix: ``real``, ``real_2``, ...

The expected outcome of a pair can be declared in
``data/samples/real_pairs.json``, keyed by pair name::

    {
      "real_2": {
        "changes": ["DESPLAZAMIENTO"],
        "label": "silla",
        "distance_cm": 45.0,
        "distance_label": "silla",
        "distance_tolerance": 0.3
      }
    }

``changes`` lists the expected change types (``ChangeType`` values or
names), ``label`` the expected name of the largest changed object and
``distance_cm`` the displacement measured by hand, checked within
``distance_tolerance`` (relative, 0.3 by default) against the displaced
object named ``distance_label`` (the largest displaced object when omitted).
Every field is optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from config.image_settings import DEFAULT_EXTENSIONS
from config.paths import SAMPLES_DIR
from models import ChangeType

EXPECTATIONS_FILE = SAMPLES_DIR / "real_pairs.json"
_BEFORE, _AFTER = "real_before", "real_after"


@dataclass(frozen=True)
class RealPairExpectation:
    changes: Optional[tuple[ChangeType, ...]] = None
    label: Optional[str] = None
    distance_cm: Optional[float] = None
    distance_label: Optional[str] = None
    distance_tolerance: float = 0.3


@dataclass(frozen=True)
class RealPair:
    name: str
    before: Path
    after: Path
    expected: RealPairExpectation = field(default_factory=RealPairExpectation)

    @property
    def output_path(self) -> Path:
        """Where the annotated result of this pair is written."""
        return SAMPLES_DIR / f"{self.name.replace('real', 'real_comparison_output', 1)}.png"


def find_real_pairs(directory: Path = SAMPLES_DIR) -> list[RealPair]:
    """Complete pairs found in ``directory``, ordered by name."""
    images = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    }
    expectations = _load_expectations(directory / EXPECTATIONS_FILE.name)
    pairs = []
    for stem, before in images.items():
        if not stem.startswith(_BEFORE):
            continue
        suffix = stem[len(_BEFORE):]
        after = images.get(_AFTER + suffix)
        if after is not None:
            name = "real" + suffix
            pairs.append(RealPair(name, before, after, expectations.get(name, RealPairExpectation())))
    return sorted(pairs, key=lambda pair: pair.name)


def unmatched_real_images(directory: Path = SAMPLES_DIR) -> list[str]:
    """Names of ``real_before*``/``real_after*`` images that have no partner."""
    stems = {
        path.stem: path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    }
    lonely = []
    for stem, filename in stems.items():
        for own, other in ((_BEFORE, _AFTER), (_AFTER, _BEFORE)):
            if stem.startswith(own) and other + stem[len(own):] not in stems:
                lonely.append(filename)
    return sorted(lonely)


def real_pair_params() -> list:
    """``pytest.param`` per pair, or a single skipped one when there are none."""
    pairs = find_real_pairs()
    if not pairs:
        return [
            pytest.param(
                None,
                id="no-real-pairs",
                marks=pytest.mark.skip(reason="add data/samples/real_before.<ext> and real_after.<ext>"),
            )
        ]
    return [pytest.param(pair, id=pair.name) for pair in pairs]


def _load_expectations(path: Path) -> dict[str, RealPairExpectation]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: _expectation(entry) for name, entry in raw.items()}


def _expectation(entry: dict) -> RealPairExpectation:
    changes = entry.get("changes")
    return RealPairExpectation(
        changes=None if changes is None else tuple(_change_type(item) for item in changes),
        label=entry.get("label"),
        distance_cm=entry.get("distance_cm"),
        distance_label=entry.get("distance_label"),
        distance_tolerance=entry.get("distance_tolerance", 0.3),
    )


def _change_type(value: str) -> ChangeType:
    try:
        return ChangeType(value)
    except ValueError:
        return ChangeType[value]
