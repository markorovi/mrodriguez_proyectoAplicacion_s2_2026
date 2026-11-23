"""Where tests write the images they produce, so a test run never changes the repository.

By default every output goes to a temporary directory of pytest. Only the
versioned samples listed in :data:`VERSIONED_SAMPLES` can be written into
``data/samples``, and only when ``UPDATE_SAMPLES=1`` is set on purpose:

    UPDATE_SAMPLES=1 RUN_INTEGRATION=1 pytest tests/test_hardware_integration.py -k depth_output

The annotated results of the personal photo pairs are not in the list, so they
always stay in the temporary directory, whatever ``UPDATE_SAMPLES`` says.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

from config.paths import SAMPLES_DIR

UPDATE_SAMPLES_VARIABLE = "UPDATE_SAMPLES"

# The only generated files that are versioned in data/samples.
VERSIONED_SAMPLES = frozenset({"comparison_output_sample.png", "depth_output_sample.png"})


def output_path(name: str, tmp_dir: Path, environ: Optional[Mapping[str, str]] = None) -> Path:
    """Path where a test should write the output file ``name``.

    ``tmp_dir`` is the test's temporary directory (``tmp_path``). The result is
    ``data/samples/<name>`` only for a versioned sample with ``UPDATE_SAMPLES=1``.
    """
    if Path(name).name != name:
        raise ValueError(f"Expected a bare file name, got {name!r}.")
    environ = os.environ if environ is None else environ
    if environ.get(UPDATE_SAMPLES_VARIABLE) == "1" and name in VERSIONED_SAMPLES:
        return SAMPLES_DIR / name
    return Path(tmp_dir) / name
