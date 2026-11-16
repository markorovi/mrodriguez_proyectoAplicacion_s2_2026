"""Message strip used for results, warnings and errors."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from PySide6.QtWidgets import QLabel, QWidget

from ui import theme


class BannerKind(str, Enum):
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# Every banner is a dark surface with light text; only the side stripe changes.
_STRIPES = {
    BannerKind.SUCCESS: theme.SUCCESS,
    BannerKind.INFO: theme.NEUTRAL,
    BannerKind.WARNING: theme.WARNING,
    BannerKind.ERROR: theme.ERROR,
}


class Banner(QLabel):
    """A word-wrapped message marked with the tone of its :class:`BannerKind`."""

    def __init__(
        self, text: str = "", kind: BannerKind = BannerKind.INFO, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.show_message(text, kind)

    @property
    def kind(self) -> BannerKind:
        return self._kind

    def show_message(self, text: str, kind: BannerKind) -> None:
        self._kind = kind
        self.setText(text)
        self.setStyleSheet(
            f"background:{theme.SURFACE_RAISED}; color:{theme.TEXT}; border:1px solid {theme.BORDER};"
            f"border-left:4px solid {_STRIPES[kind]}; border-radius:4px; padding:8px 10px;"
        )
