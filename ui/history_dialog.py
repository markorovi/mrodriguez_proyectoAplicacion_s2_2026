"""Session history window: the comparisons of the session and the report of the chosen one (CU-03)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from models.session_history import HistoryEntry, SessionHistory
from ui import report_text
from ui.banner import Banner, BannerKind
from ui.report_widget import ReportWidget


class HistoryDialog(QDialog):
    """Lists the comparisons of the session, newest first, and shows the report of the selected one.

    It only reads the history: opening it or moving through the list never
    changes it. With an empty history it shows a message instead of the list
    (CU-03, flow A1). The report pane is a regular :class:`ReportWidget`, so
    an older result can still be exported or saved as an image from here.
    """

    def __init__(self, history: SessionHistory, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entries = history.entries
        self.setWindowTitle("Historial de la sesión")
        self.resize(self._initial_size(parent))
        self._build()
        if self._entries:
            self.select(0)

    @property
    def entry_titles(self) -> list[str]:
        return [report_text.history_title(entry) for entry in self._entries]

    @property
    def shown_entry(self) -> Optional[HistoryEntry]:
        row = self._list.currentRow()
        return self._entries[row] if 0 <= row < len(self._entries) else None

    @property
    def empty_message_text(self) -> str:
        return self._empty.text() if not self._entries else ""

    @property
    def report_widget(self) -> ReportWidget:
        return self._report

    def select(self, index: int) -> None:
        """Show the report of the entry at ``index`` (0 is the newest)."""
        if not 0 <= index < len(self._entries):
            raise IndexError(f"There is no history entry at position {index}.")
        self._list.setCurrentRow(index)

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._entries):
            entry = self._entries[row]
            self._report.show_report(entry.report, originals=entry.originals)

    def _build(self) -> None:
        self._list = QListWidget()
        self._list.setMinimumWidth(280)
        self._list.setWordWrap(True)
        for entry in self._entries:
            item = QListWidgetItem(f"{report_text.history_title(entry)}\n{report_text.history_summary(entry)}")
            item.setSizeHint(QSize(0, 48))
            self._list.addItem(item)
        self._list.currentRowChanged.connect(self._on_row_changed)

        self._report = ReportWidget()
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._report.exported.connect(self._status.setText)

        report_side = QWidget()
        report_layout = QVBoxLayout(report_side)
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_layout.addWidget(self._report, stretch=1)
        report_layout.addWidget(self._status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._list)
        splitter.addWidget(report_side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._empty = Banner(report_text.HISTORY_EMPTY_MESSAGE, BannerKind.INFO)

        buttons = QDialogButtonBox()
        close = buttons.addButton("Cerrar", QDialogButtonBox.ButtonRole.RejectRole)
        close.clicked.connect(self.reject)
        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self._empty)
        layout.addWidget(splitter, stretch=1)
        if not self._entries:
            layout.addStretch()  # Keeps the message at the top instead of stretching it.
        layout.addLayout(footer)
        self._empty.setVisible(not self._entries)
        splitter.setVisible(bool(self._entries))

    @staticmethod
    def _initial_size(parent: Optional[QWidget]) -> QSize:
        """Most of the parent window when there is one, so the report image stays legible."""
        if parent is not None and parent.width() > 0:
            return QSize(int(parent.width() * 0.9), int(parent.height() * 0.9))
        return QSize(1100, 700)
