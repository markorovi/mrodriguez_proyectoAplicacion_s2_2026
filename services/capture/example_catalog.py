"""Catalog of the example photo pairs shipped with the application, read from a manifest.

The manifest (``data/samples/examples.json``) lists pairs of image files that
live next to it, so the application can be shown and checked without the
user's own photos and without network access. Reading the catalog never
fails: a broken manifest or entry is skipped and the reason is kept for the
caller, because a missing example must not prevent the application from
starting. Files are only checked for existence; they are decoded when loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Optional

from config.paths import SAMPLES_DIR

MANIFEST_NAME = "examples.json"
ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg")
_REQUIRED_FIELDS = ("id", "title", "description", "before", "after")


@dataclass(frozen=True)
class ExampleEntry:
    """One example pair; both paths are absolute and point to existing files."""

    id: str
    title: str
    description: str
    before_path: Path
    after_path: Path


@dataclass(frozen=True)
class ExampleCatalog:
    """Valid examples in manifest order, and why each discarded one was left out."""

    entries: tuple[ExampleEntry, ...] = ()
    skipped: tuple[str, ...] = field(default_factory=tuple)


def load_examples(manifest_path: Optional[Path] = None, samples_dir: Optional[Path] = None) -> ExampleCatalog:
    """Read the manifest; file names are resolved against ``samples_dir`` (the samples folder by default)."""
    samples_dir = Path(samples_dir) if samples_dir is not None else SAMPLES_DIR
    manifest_path = Path(manifest_path) if manifest_path is not None else samples_dir / MANIFEST_NAME

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ExampleCatalog(skipped=(f"No existe el manifiesto de ejemplos '{manifest_path.name}'.",))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ExampleCatalog(skipped=(f"El manifiesto de ejemplos no se pudo leer: {exc}.",))

    raw_entries = data.get("examples") if isinstance(data, dict) else None
    if not isinstance(raw_entries, list):
        return ExampleCatalog(skipped=("El manifiesto de ejemplos no tiene una lista 'examples'.",))

    entries: list[ExampleEntry] = []
    skipped: list[str] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_entries, start=1):
        entry, reason = _parse_entry(raw, samples_dir)
        if entry is not None and entry.id in seen_ids:
            entry, reason = None, f"el identificador '{entry.id}' está repetido"
        if entry is None:
            skipped.append(f"Ejemplo {position} omitido: {reason}.")
            continue
        seen_ids.add(entry.id)
        entries.append(entry)
    return ExampleCatalog(entries=tuple(entries), skipped=tuple(skipped))


def _parse_entry(raw: Any, samples_dir: Path) -> tuple[Optional[ExampleEntry], str]:
    if not isinstance(raw, dict):
        return None, "no es un objeto"
    missing = [name for name in _REQUIRED_FIELDS if not isinstance(raw.get(name), str) or not raw[name].strip()]
    if missing:
        return None, f"faltan campos: {', '.join(missing)}"
    paths = []
    for side in ("before", "after"):
        path, reason = _resolve(raw[side], samples_dir)
        if path is None:
            return None, f"'{side}' {reason}"
        paths.append(path)
    return ExampleEntry(raw["id"], raw["title"], raw["description"], *paths), ""


def _resolve(name: str, samples_dir: Path) -> tuple[Optional[Path], str]:
    """Path of a bare file name inside ``samples_dir``; anything that could leave the folder is refused."""
    if "/" in name or "\\" in name or ".." in name or PurePath(name).is_absolute() or ":" in name:
        return None, f"debe ser solo un nombre de archivo, sin carpetas: {name!r}"
    if PurePath(name).suffix.lower() not in ALLOWED_SUFFIXES:
        return None, f"tiene una extensión no admitida: {name!r}"
    path = (samples_dir / name).resolve()
    if not path.is_file():
        return None, f"no existe el archivo {name!r}"
    return path, ""
