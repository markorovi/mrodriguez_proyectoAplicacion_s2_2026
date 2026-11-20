"""User manual window: the list of sections and the text of the manual, side by side."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from config.settings import AppSettings
from ui import theme
from ui.manual_text import build_manual, manual_html

# The document has its own style sheet: rich text does not inherit the widget palette for headings.
_DOCUMENT_STYLE = f"""
body {{ color: {theme.TEXT}; }}
h1 {{ color: {theme.TEXT}; font-size: 18pt; }}
h2 {{ color: {theme.TEXT}; font-size: 14pt; margin-top: 18px;
      border-bottom: 1px solid {theme.BORDER}; }}
p, li {{ color: {theme.TEXT}; font-size: 10.5pt; }}
a {{ color: {theme.TEXT_MUTED}; }}
"""


class ManualDialog(QDialog):
    """Shows the manual built for ``settings``, so its figures match the running configuration.

    Links of the index jump to their section; any other link is ignored,
    because the manual is meant to work without internet access.
    """

    def __init__(self, settings: Optional[AppSettings] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sections = build_manual(settings)
        self._current = ""
        self.setWindowTitle("Manual de usuario")
        self.resize(self._initial_size(parent))
        self._build()
        self.show_section(self._sections[0].anchor)

    @property
    def section_titles(self) -> list[str]:
        return [section.title for section in self._sections]

    @property
    def current_section(self) -> str:
        return self._current

    @property
    def browser(self) -> QTextBrowser:
        return self._browser

    def show_section(self, anchor: str) -> None:
        """Scroll the text to the section ``anchor`` and select it in the list."""
        anchors = [section.anchor for section in self._sections]
        if anchor not in anchors:
            raise ValueError(f"The manual has no section {anchor!r}.")
        self._current = anchor
        self._browser.scrollToAnchor(anchor)
        row = anchors.index(anchor)
        if self._list.currentRow() != row:
            self._list.blockSignals(True)  # Avoid scrolling twice when the list itself changed.
            self._list.setCurrentRow(row)
            self._list.blockSignals(False)

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._sections):
            self.show_section(self._sections[row].anchor)

    def _on_link(self, url: QUrl) -> None:
        anchor = url.fragment()
        if not url.scheme() and anchor in {section.anchor for section in self._sections}:
            self.show_section(anchor)

    def _build(self) -> None:
        self._list = QListWidget()
        self._list.setMinimumWidth(280)
        self._list.addItems(self.section_titles)
        self._list.currentRowChanged.connect(self._on_row_changed)

        self._browser = QTextBrowser()
        self._browser.setReadOnly(True)
        self._browser.setOpenLinks(False)
        self._browser.setOpenExternalLinks(False)
        self._browser.anchorClicked.connect(self._on_link)
        self._browser.document().setDefaultStyleSheet(_DOCUMENT_STYLE)
        self._browser.setHtml(manual_html(self._sections))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._list)
        splitter.addWidget(self._browser)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        buttons = QDialogButtonBox()
        close = buttons.addButton("Cerrar", QDialogButtonBox.ButtonRole.RejectRole)
        close.clicked.connect(self.reject)
        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(footer)

    @staticmethod
    def _initial_size(parent: Optional[QWidget]) -> QSize:
        """Most of the parent window when there is one, so long sections stay readable."""
        if parent is not None and parent.width() > 0:
            return QSize(int(parent.width() * 0.9), int(parent.height() * 0.9))
        return QSize(1100, 700)
