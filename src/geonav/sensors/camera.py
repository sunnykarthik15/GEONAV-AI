"""Camera sensor interface and frame data representation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Optional


@dataclass
class CameraFrame:
    """Represents a single optical camera frame.

    Attributes:
        timestamp: Monotonic capture timestamp in seconds (must be non-negative).
        frame_data: Raw image or frame container (must not be None).
    """

    timestamp: float
    frame_data: Any

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a float or integer")
        if math.isnan(self.timestamp) or math.isinf(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"timestamp must be a finite non-negative number, got {self.timestamp}")
        if self.frame_data is None:
            raise ValueError("frame_data must not be None")


class CameraSensor(ABC):
    """Abstract base class representing a camera sensor data source."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check whether the camera sensor is currently connected."""
        raise NotImplementedError

    @abstractmethod
    def capture_frame(self) -> Optional[CameraFrame]:
        """Capture and return the latest camera frame if available."""
        raise NotImplementedError


class DatasetCameraSensor(CameraSensor):
    """Camera sensor adapter providing frames sequentially from a dataset loader."""

    def __init__(self, loader: Any) -> None:
        """Initialize adapter with a dataset camera loader or iterable of DatasetImage.

        Args:
            loader: Iterable or sequence of DatasetImage instances.
        """
        self._loader = loader
        self._iterator: Optional[Any] = None
        self._connected: bool = True

    def is_connected(self) -> bool:
        """Check whether sensor is connected."""
        return self._connected

    def connect(self) -> None:
        """Connect sensor and initialize iterator."""
        self._connected = True
        self._iterator = iter(self._loader)

    def disconnect(self) -> None:
        """Disconnect sensor."""
        self._connected = False

    def capture_frame(self) -> Optional[CameraFrame]:
        """Capture the next sequential frame from the dataset.

        Returns:
            Optional[CameraFrame]: The next CameraFrame, or None if exhausted/disconnected.
        """
        if not self._connected:
            return None
        if self._iterator is None:
            self._iterator = iter(self._loader)
        try:
            item = next(self._iterator)
            if hasattr(item, "to_camera_frame"):
                return item.to_camera_frame()
            if isinstance(item, CameraFrame):
                return item
            raise TypeError(f"Unexpected item type from camera dataset loader: {type(item)}")
        except StopIteration:
            return None

