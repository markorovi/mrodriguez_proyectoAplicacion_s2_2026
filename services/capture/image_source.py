"""Abstraction over anything that can deliver an RGB image."""

from abc import ABC, abstractmethod

from models.image_data import ImageData


class ImageSource(ABC):
    """A provider of images (file on disk, connected camera, ...).

    Implementations decode or capture the image and return it in RGB order.
    They do not judge its quality; that is the validator's job.
    """

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable identifier of the source (path, device index, ...)."""

    @abstractmethod
    def acquire(self) -> ImageData:
        """Return a new image from this source.

        Raises:
            models.exceptions.ImageError: the image cannot be obtained.
        """
