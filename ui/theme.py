"""Dark color palette of the application.

A few grays carry the interface; the only accents are three muted tones
that tell success, warning and error apart. The annotated image keeps its
own colors, which the report text refers to.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BACKGROUND = "#1b1b1b"
SURFACE = "#242424"
SURFACE_RAISED = "#2c2c2c"
BORDER = "#3a3a3a"
TEXT = "#e4e4e4"
TEXT_MUTED = "#a6a6a6"
IMAGE_BACKGROUND = "#141414"

SUCCESS = "#6f9c7a"
WARNING = "#b3954f"
ERROR = "#b5645c"
NEUTRAL = "#7a7a7a"


def apply_dark_theme(app: QApplication) -> None:
    """Fusion style with the dark palette, so native widgets match the custom ones."""
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE,
        QPalette.ColorRole.AlternateBase: SURFACE_RAISED,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: SURFACE_RAISED,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.ToolTipBase: SURFACE_RAISED,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_MUTED,
        QPalette.ColorRole.Highlight: "#4a4a4a",
        QPalette.ColorRole.HighlightedText: TEXT,
        QPalette.ColorRole.Link: TEXT,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6a6a6a"))
    app.setPalette(palette)
