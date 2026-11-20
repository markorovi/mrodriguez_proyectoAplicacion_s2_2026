"""Catalog of the example pairs (data/samples/examples.json), without Qt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from config.paths import SAMPLES_DIR
from services.capture.example_catalog import MANIFEST_NAME, load_examples
from tests.real_pairs import find_real_pairs

GPS_IFD = 0x8825


def _write_manifest(folder: Path, examples) -> Path:
    path = folder / MANIFEST_NAME
    path.write_text(json.dumps({"version": 1, "examples": examples}, ensure_ascii=False), encoding="utf-8")
    return path


def _entry(entry_id: str = "uno", before: str = "a.png", after: str = "b.png") -> dict:
    return {"id": entry_id, "title": "Título", "description": "Descripción", "before": before, "after": after}


@pytest.fixture
def folder(tmp_path) -> Path:
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    return tmp_path


# ------------------------------------------------------------- shipped manifest


def test_shipped_manifest_loads_every_example():
    catalog = load_examples()
    assert catalog.skipped == ()
    assert len(catalog.entries) >= 1
    for entry in catalog.entries:
        assert entry.before_path.is_file() and entry.after_path.is_file()
        assert entry.before_path.parent == SAMPLES_DIR.resolve()


def test_examples_are_not_validation_pairs_nor_outdoor_photos():
    names = [p.name for e in load_examples().entries for p in (e.before_path, e.after_path)]
    assert not [name for name in names if name.startswith(("real_", "exterior"))]
    # The integration tests discover pairs by name; the examples must not join them.
    discovered = {p.name for pair in find_real_pairs() for p in (pair.before, pair.after)}
    assert not discovered & set(names)


def test_example_images_carry_no_metadata():
    paths = {p for e in load_examples().entries for p in (e.before_path, e.after_path)}
    for path in sorted(paths):
        with Image.open(path) as image:
            exif = image.getexif()
            assert len(exif) == 0, path.name
            assert len(exif.get_ifd(GPS_IFD)) == 0, path.name
            assert "exif" not in image.info and "icc_profile" not in image.info, path.name
            assert not [key for key, value in image.info.items() if isinstance(value, str)], path.name


# ------------------------------------------------------------- invalid manifests


def test_missing_manifest_gives_no_examples_and_a_reason(tmp_path):
    catalog = load_examples(samples_dir=tmp_path)
    assert catalog.entries == ()
    assert len(catalog.skipped) == 1 and "No existe" in catalog.skipped[0]


def test_invalid_json_gives_a_reason(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text("{ esto no es json", encoding="utf-8")
    catalog = load_examples(samples_dir=tmp_path)
    assert catalog.entries == ()
    assert "no se pudo leer" in catalog.skipped[0]


@pytest.mark.parametrize("name", ["../a.png", "sub/a.png", "sub\\a.png", "C:/a.png", "a.gif"])
def test_names_that_could_leave_the_folder_or_are_not_images_are_skipped(folder, name):
    _write_manifest(folder, [_entry(before=name), _entry("dos")])
    catalog = load_examples(samples_dir=folder)
    assert [e.id for e in catalog.entries] == ["dos"]
    assert len(catalog.skipped) == 1 and "Ejemplo 1 omitido" in catalog.skipped[0]


def test_missing_file_and_missing_fields_are_skipped(folder):
    incomplete = _entry("tres")
    del incomplete["title"]
    _write_manifest(folder, [_entry(after="no_existe.png"), incomplete, _entry("dos")])
    catalog = load_examples(samples_dir=folder)
    assert [e.id for e in catalog.entries] == ["dos"]
    assert "no existe el archivo" in catalog.skipped[0]
    assert "faltan campos: title" in catalog.skipped[1]


def test_duplicate_id_keeps_the_first(folder):
    _write_manifest(folder, [_entry("uno"), _entry("uno", before="b.png", after="a.png")])
    catalog = load_examples(samples_dir=folder)
    assert [e.before_path.name for e in catalog.entries] == ["a.png"]
    assert "repetido" in catalog.skipped[0]
