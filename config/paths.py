"""Well-known filesystem locations of the project."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
WEIGHTS_DIR: Path = PROJECT_ROOT / "models" / "weights"
SAMPLES_DIR: Path = PROJECT_ROOT / "data" / "samples"
