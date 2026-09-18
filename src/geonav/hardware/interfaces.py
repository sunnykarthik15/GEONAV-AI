"""Abstract interface definitions for GEONAV-AI hardware abstraction layer.

These interfaces define the contracts that real hardware drivers must implement
to plug into the GEONAV-AI VIO pipeline. The existing EuRoC dataset adapters
implement these same interfaces for offline validation.

NOTE: No physical UAV hardware is currently available. These interfaces are
designed for future integration with physical sensors and flight controllers.
Motor control, autonomous flight, and navigation commands are explicitly
out of scope and are NOT part of these interfaces.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
import numpy as np

from geonav.sensors.camera import CameraFrame
from geonav.sensors.imu import IMUSample
from geonav.hardware.nav_output import NavigationMessage


class CameraSource(ABC):
    """Abstract camera data source interface.

    Implementations provide an iterable stream of camera frames.
    All implementations must:
        - Produce frames in chronological order.
        - Provide timestamps in seconds.
        - Produce grayscale or color frames as numpy arrays.

    Unit conventions:
        - timestamp: seconds (float)
        - frame_data: numpy array (H, W) or (H, W, 3)
    """

    @abstractmethod
    def frames(self) -> Iterator[CameraFrame]:
        """Yield camera frames in chronological order.

        Yields:
            CameraFrame: Next available camera frame.
        """
        ...

    @abstractmethod
    def camera_matrix(self) -> Optional[np.ndarray]:
        """Return the 3x3 camera intrinsic matrix K, or None if unavailable.

        Returns:
            np.ndarray: K matrix (3, 3), or None.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the source to the beginning of the stream."""
        ...


class IMUSource(ABC):
    """Abstract IMU data source interface.

    Implementations provide an iterable stream of IMU samples.
    All implementations must:
        - Produce samples in chronological order.
        - Provide timestamps in seconds.
        - Report accelerometer readings in m/s².
        - Report gyroscope readings in rad/s.

    Unit conventions:
        - timestamp: seconds (float)
        - linear_acceleration: m/s² (3-element vector, body frame)
        - angular_velocity: rad/s (3-element vector, body frame)
    """

    @abstractmethod
    def samples(self) -> Iterator[IMUSample]:
        """Yield IMU samples in chronological order.

        Yields:
            IMUSample: Next available IMU measurement.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the source to the beginning of the stream."""
        ...


class NavigationOutput(ABC):
    """Abstract navigation state output interface.

    Implementations deliver estimated navigation states to downstream
    consumers (logging, display, flight controller, etc.).

    Unit conventions (enforced by NavigationMessage):
        - position: meters (3-element vector)
        - velocity: m/s (3-element vector)
        - orientation: unit quaternion [w, x, y, z]
        - timestamp: seconds (float)
    """

    @abstractmethod
    def publish(self, message: NavigationMessage) -> None:
        """Deliver a navigation state message to downstream consumers.

        Args:
            message: NavigationMessage containing position, velocity, orientation,
                     tracking status, confidence, and health flags.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the output channel."""
        ...


class UAVIntegrationAdapter(ABC):
    """Abstract future UAV integration adapter interface.

    This interface is a placeholder for future integration with physical
    flight controllers (e.g., via MAVLink, ROS, or custom protocols).

    IMPORTANT: This interface is NOT implemented and NOT tested with any
    physical UAV hardware. It exists to document the intended integration
    point for future development.

    PROHIBITED:
        - Motor control commands
        - Autonomous flight decisions
        - Navigation waypoint sending
        - Any commands that could affect vehicle motion

    PERMITTED:
        - Publishing estimated navigation state to the flight controller
        - Requesting current sensor availability
        - Reporting health status
    """

    @abstractmethod
    def send_navigation_state(self, message: NavigationMessage) -> None:
        """Transmit estimated navigation state to the UAV autopilot.

        Args:
            message: Current navigation state estimate.

        Note:
            This method sends information TO the flight controller.
            It does NOT send control commands or affect vehicle motion.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check whether the adapter has an active connection to the flight controller.

        Returns:
            bool: True if connected, False otherwise.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the flight controller."""
        ...
