"""Description of the hardware selected to run inference."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ComputeDevice:
    """A torch device plus the information the user should be told about it.

    ``notice`` is set when the selection degrades the experience, for example
    when no GPU is available and inference falls back to the CPU.
    """

    torch_device: torch.device
    name: str
    notice: Optional[str] = None

    @property
    def is_accelerated(self) -> bool:
        return self.torch_device.type == "cuda"
