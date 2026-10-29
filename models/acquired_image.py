"""An image that passed validation, together with its validation outcome."""

from dataclasses import dataclass

from models.image_data import ImageData
from models.validation_report import ValidationReport


@dataclass(frozen=True)
class AcquiredImage:
    """Validated image ready for inference; ``report`` may still hold warnings."""

    image: ImageData
    report: ValidationReport

    @property
    def warnings(self) -> list[str]:
        return list(self.report.warnings)
