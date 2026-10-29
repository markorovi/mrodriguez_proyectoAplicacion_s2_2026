"""Outcome of validating an input image."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IssueCode(str, Enum):
    """Machine-readable category of a validation finding."""

    BIT_DEPTH = "bit_depth"
    CHANNELS = "channels"
    RESOLUTION = "resolution"
    LOW_LIGHT = "low_light"


@dataclass(frozen=True)
class ValidationIssue:
    """A single finding: its category and a message for the end user."""

    code: IssueCode
    message: str


@dataclass
class ValidationReport:
    """Blocking errors and non-blocking warnings found during validation.

    An image with errors must not be processed. Warnings (for example, low
    illumination) allow processing but must be surfaced to the user because
    they can compromise the reliability of the result.
    """

    error_issues: list[ValidationIssue] = field(default_factory=list)
    warning_issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.error_issues

    @property
    def errors(self) -> list[str]:
        return [issue.message for issue in self.error_issues]

    @property
    def warnings(self) -> list[str]:
        return [issue.message for issue in self.warning_issues]

    def add_error(self, code: IssueCode, message: str) -> None:
        self.error_issues.append(ValidationIssue(code, message))

    def add_warning(self, code: IssueCode, message: str) -> None:
        self.warning_issues.append(ValidationIssue(code, message))

    def has(self, code: IssueCode) -> bool:
        """True if any error or warning has the given code."""
        return any(issue.code is code for issue in (*self.error_issues, *self.warning_issues))
