"""In-memory history of the comparisons of a session (RF-014), without Qt."""

import numpy as np
import pytest

from models import ComparisonReport, DepthComparison, HistoryEntry, SessionHistory
from models.alignment import AlignmentReport


def _report(changes: int = 0) -> ComparisonReport:
    """Smallest valid report; ``changes`` only tags it so entries can be told apart."""
    shape = (4, 6)
    return ComparisonReport(
        objects=(),
        alignment=AlignmentReport(True, "sift", 10, 10, 8, 8, 0.5, f"par {changes}"),
        comparison=DepthComparison((), np.zeros(shape, np.float32), np.zeros(shape, bool), np.zeros(shape, bool)),
        annotated_image=np.zeros((*shape, 3), np.uint8),
    )


def test_history_starts_empty():
    history = SessionHistory()
    assert history.is_empty
    assert len(history) == 0
    assert history.entries == ()
    assert history.latest is None


def test_entries_are_numbered_from_one():
    history = SessionHistory()
    numbers = [history.add(_report(i), "a.png", "b.png").number for i in range(3)]
    assert numbers == [1, 2, 3]
    assert isinstance(history.latest, HistoryEntry)
    assert history.latest.number == 3


def test_entries_go_from_newest_to_oldest():
    history = SessionHistory()
    first = history.add(_report(1), "a.png", "b.png")
    second = history.add(_report(2), "c.png", "d.png")
    assert history.entries == (second, first)
    assert history.entries[0].report.alignment.message == "par 2"


def test_oldest_entry_is_dropped_beyond_the_limit_and_numbers_are_not_reused():
    history = SessionHistory(max_entries=2)
    for i in range(3):
        history.add(_report(i), "a.png", "b.png")
    assert len(history) == 2
    assert [entry.number for entry in history.entries] == [3, 2]


def test_file_sources_keep_only_the_file_name(tmp_path):
    entry = SessionHistory().add(_report(), str(tmp_path / "fotos" / "antes del cambio ñ.jpg"), "C:/x/después.png")
    assert entry.before_name == "antes del cambio ñ.jpg"
    assert entry.after_name == "después.png"


def test_camera_sources_are_named_in_spanish():
    entry = SessionHistory().add(_report(), "camera:1", "camera:0")
    assert (entry.before_name, entry.after_name) == ("Cámara 1", "Cámara 0")


def test_history_must_keep_at_least_one_entry():
    with pytest.raises(ValueError):
        SessionHistory(max_entries=0)
