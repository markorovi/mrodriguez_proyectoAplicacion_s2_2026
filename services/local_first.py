"""Loading pretrained files from this computer first, downloading only when they are missing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")

# transformers reads this variable on every from_pretrained call and treats "1" as active.
DISABLE_CONVERSION_VARIABLE = "DISABLE_SAFETENSORS_CONVERSION"


def load_local_first(load: Callable[[bool], T], local_files_only: bool) -> T:
    """Return ``load(True)``; when the local files are missing, maybe ``load(False)``.

    ``load`` receives the ``local_files_only`` flag to pass to every
    ``from_pretrained`` call and must load the pre-processor and the model
    together, so a partial local copy is downloaded again as a whole.

    The first attempt never contacts the network. Only an :class:`OSError`
    (files not found in the cache) leads to a second attempt that may
    download, and only when ``local_files_only`` is false. Any other error,
    such as a :class:`ValueError` from an incompatible checkpoint, propagates
    without retrying.
    """
    with _conversion_check_disabled():
        try:
            return load(True)
        except OSError:
            if local_files_only:
                raise
        return load(False)


@contextmanager
def _conversion_check_disabled() -> Iterator[None]:
    """Turn off transformers' automatic safetensors conversion while models load.

    When cached weights are in the old .bin format, transformers starts a
    background request asking whether a safetensors conversion exists, even
    with local_files_only. The application never needs that conversion and the
    request breaks the local mode. transformers decides whether to start it
    inside each from_pretrained call, so the variable only needs to be active
    during the load: it is set here (not at import), removed afterwards if it
    was not there before, and a value set by the user is left untouched.
    """
    was_set = DISABLE_CONVERSION_VARIABLE in os.environ
    os.environ.setdefault(DISABLE_CONVERSION_VARIABLE, "1")
    try:
        yield
    finally:
        if not was_set:
            os.environ.pop(DISABLE_CONVERSION_VARIABLE, None)
