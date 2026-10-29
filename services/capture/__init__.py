"""Image acquisition from disk or from a connected RGB camera."""

from services.capture.acquisition_service import ImageAcquisitionService
from services.capture.camera_capture import CameraCapture, VideoDevice
from services.capture.file_image_loader import FileImageLoader
from services.capture.image_source import ImageSource

__all__ = [
    "CameraCapture",
    "FileImageLoader",
    "ImageAcquisitionService",
    "ImageSource",
    "VideoDevice",
]
