"""In-memory history of the comparisons made during one session of the application (RF-014)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.comparison_report import ComparisonReport

CAMERA_SOURCE_PREFIX = "camera:"


@dataclass(frozen=True)
class HistoryEntry:
    """One successful comparison of the session.

    ``number`` counts the comparisons of the session from 1 and is never
    reused, so it still identifies an entry after older ones are dropped.
    """

    number: int
    created_at: datetime
    before_name: str
    after_name: str
    report: ComparisonReport


@dataclass
class SessionHistory:
    """Comparisons of the active session, newest first, kept in memory only.

    The history is not persisted: it disappears when the application closes
    (CU-03, flow A2). It keeps at most ``max_entries`` reports and drops the
    oldest beyond that, because each report holds its annotated image and the
    comparison arrays: about 19 MB for a pair of 1920x1080 photos, so an
    unbounded history would grow without limit over a long session.
    """

    max_entries: int = 20
    _entries: deque[HistoryEntry] = field(init=False, repr=False)
    _next_number: int = field(init=False, default=1, repr=False)

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("The history must keep at least one entry.")
        self._entries = deque(maxlen=self.max_entries)

    def add(self, report: ComparisonReport, before_source: str, after_source: str) -> HistoryEntry:
        """Record a finished comparison and return its entry."""
        entry = HistoryEntry(
            number=self._next_number,
            created_at=datetime.now(),
            before_name=display_name(before_source),
            after_name=display_name(after_source),
            report=report,
        )
        self._next_number += 1
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        """Every kept entry, from the newest to the oldest."""
        return tuple(reversed(self._entries))

    @property
    def latest(self) -> Optional[HistoryEntry]:
        return self._entries[-1] if self._entries else None

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def __len__(self) -> int:
        return len(self._entries)


def display_name(source: str) -> str:
    """Short name of an image source: the file name, or "Cámara N" for a camera capture."""
    if source.startswith(CAMERA_SOURCE_PREFIX):
        return f"Cámara {source[len(CAMERA_SOURCE_PREFIX):]}"
    return Path(source).name
