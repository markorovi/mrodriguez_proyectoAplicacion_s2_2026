"""User manual: content built from the settings, HTML and the dialog that shows it."""

from __future__ import annotations

import re

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QMenu, QPushButton

from config import AppSettings, DevicePreference, ImageSettings, SegmentationSettings
from controllers import PipelineController
from models import SessionHistory
from services.inference import DeviceManager
from services.segmentation import SegFormerSegmenter
from tests.doubles import FakeSegmentationModelLoader, SceneDepthEstimator
from tests.test_main_window import JARGON
from tests.test_session_history import _report
from ui import manual_text
from ui.main_window import MainWindow
from ui.manual_dialog import ManualDialog
from ui.manual_text import ManualSection, build_manual, manual_html

TITLES = [
    "Para qué sirve",
    "Primeros pasos",
    "Cómo tomar las fotos",
    "Requisitos de las fotos",
    "Cómo leer el resultado",
    "Historial de la sesión",
    "Avisos y errores frecuentes",
    "Limitaciones",
]


def _text(sections: tuple[ManualSection, ...]) -> str:
    return "\n".join(
        "\n".join((section.title, *section.paragraphs, *section.bullets)) for section in sections
    )


def _section(sections: tuple[ManualSection, ...], title: str) -> ManualSection:
    return next(section for section in sections if section.title == title)


def test_manual_has_the_eight_sections_in_order_with_unique_anchors():
    sections = build_manual()
    assert [section.title for section in sections] == TITLES
    anchors = [section.anchor for section in sections]
    assert len(set(anchors)) == len(anchors)
    assert all(re.fullmatch(r"[a-z0-9-]+", anchor) for anchor in anchors)
    assert all(section.paragraphs or section.bullets for section in sections)


def test_photo_taking_section_has_at_least_six_guidelines():
    assert len(_section(build_manual(), "Cómo tomar las fotos").bullets) >= 6


def test_figures_come_from_the_settings():
    default = _text(build_manual())
    assert "640x480" in default
    assert "menor que 40 en una escala de 0" in default
    assert "25 % de elementos de exterior" in default
    assert "1280x720" in default
    assert "50 % de la escena" in default

    custom = AppSettings(
        image=ImageSettings(min_width=1024, min_height=768, low_light_threshold=55.0),
        segmentation=SegmentationSettings(max_outdoor_percent=30.0),
    )
    text = _text(build_manual(custom))
    assert "1024x768" in text and "640x480" not in text
    assert "menor que 55 en una escala de 0" in text
    assert "30 % de elementos de exterior" in text

    without_check = _text(build_manual(AppSettings(segmentation=SegmentationSettings(check_indoor_scene=False))))
    assert "elementos de exterior" not in without_check
    assert "no parece un espacio interior" not in without_check


def test_history_limit_in_the_manual_matches_the_real_history():
    history = SessionHistory()
    for _ in range(history.max_entries + 3):
        history.add(_report(), "a.png", "b.png")
    kept = len(history)
    assert kept == manual_text.max_history_entries()
    assert f"como máximo los {kept} análisis" in _text(build_manual())


def test_html_has_an_anchor_and_an_index_link_per_section():
    sections = build_manual()
    page = manual_html(sections)
    for section in sections:
        assert f'<a name="{section.anchor}">' in page
        assert f'<a href="#{section.anchor}">' in page
    assert "{" not in page and "}" not in page


def test_manual_avoids_technical_terms_and_requirement_codes():
    text = _text(build_manual()).lower()
    assert not [term for term in JARGON if term in text]
    assert not re.search(r"\b(rf|ris|rr|rnf)-", text)


@pytest.fixture
def real_window(qapp):
    segmenter = SegFormerSegmenter(
        SegmentationSettings(), DeviceManager(DevicePreference.CPU), FakeSegmentationModelLoader()
    )
    window = MainWindow(controller=PipelineController(depth_estimator=SceneDepthEstimator(), segmenter=segmenter))
    yield window
    window.close()
    window.deleteLater()


def test_names_and_shortcuts_cited_in_first_steps_exist_in_the_window(real_window):
    actions = {action.text(): action.shortcut().toString() for action in real_window.findChildren(QAction)}
    for name, shortcut in manual_text.ACTION_SHORTCUTS.items():
        assert name in actions, name
        assert actions[name] == (shortcut or ""), name
    buttons = {button.text() for button in real_window.findChildren(QPushButton)}
    labels = {label.text() for label in real_window.findChildren(QLabel)}
    assert {manual_text.LOAD_FILE_BUTTON, manual_text.CAPTURE_BUTTON, manual_text.ANALYZE_BUTTON} <= buttons
    assert manual_text.CAMERA_LABEL in labels

    first_steps = _text((_section(build_manual(), "Primeros pasos"),))
    for name in (*manual_text.ACTION_SHORTCUTS, manual_text.LOAD_FILE_BUTTON, manual_text.ANALYZE_BUTTON):
        if name == "Capturar de cámara":
            continue  # The steps point to the panel button «Capturar» instead.
        assert f"«{name}»" in first_steps, name


def test_dialog_lists_the_sections_and_moves_between_them(qapp):
    dialog = ManualDialog(AppSettings())
    assert dialog.windowTitle() == "Manual de usuario"
    assert dialog.section_titles == TITLES
    assert dialog.current_section == build_manual()[0].anchor
    assert dialog.browser.isReadOnly() and not dialog.browser.openLinks()

    last = build_manual()[-1].anchor
    dialog.show_section(last)
    assert dialog.current_section == last
    with pytest.raises(ValueError):
        dialog.show_section("no-existe")
    dialog.deleteLater()


def test_manual_mentions_the_examples_menu_that_exists_in_the_window(real_window):
    first_steps = _text((_section(build_manual(), "Primeros pasos"),))
    assert f"«{manual_text.EXAMPLES_MENU}»" in first_steps
    menu_titles = {menu.title() for menu in real_window.menuBar().findChildren(QMenu)}
    assert manual_text.EXAMPLES_MENU in menu_titles
