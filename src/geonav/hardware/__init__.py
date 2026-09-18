"""Hardware abstraction layer for GEONAV-AI.

This package provides abstract interfaces for camera, IMU, and navigation
output that decouple the VIO pipeline from specific hardware drivers.

Architecture:
    CameraSource / IMUSource   → Sensor ingestion interfaces
    NavigationOutput           → Navigation state delivery interface
    UAVIntegrationAdapter      → Future flight-controller interface (not yet implemented)
    HealthMonitor              → Pipeline health state machine
    NavigationMessage          → Typed navigation state output

Current implementations:
    EurocCameraSource    → Wraps EurocDataset (dataset validation)
    EurocIMUSource       → Wraps EurocDataset (dataset validation)
    ConsoleNavigationOutput → Prints navigation messages to stdout

Future implementations (documented, not yet built):
    RealCameraDriver     → Physical camera via OpenCV/GStreamer
    RealIMUDriver        → Physical IMU via serial/USB
    MAVLinkAdapter       → MAVLink-based UAV flight controller interface
"""

from geonav.hardware.interfaces import (
    CameraSource,
    IMUSource,
    NavigationOutput,
    UAVIntegrationAdapter,
)
from geonav.hardware.health import HealthState, HealthMonitor, SystemHealth
from geonav.hardware.nav_output import NavigationMessage, ConsoleNavigationOutput
from geonav.hardware.euroc_adapter import EurocCameraSource, EurocIMUSource

__all__ = [
    "CameraSource",
    "IMUSource",
    "NavigationOutput",
    "UAVIntegrationAdapter",
    "HealthState",
    "HealthMonitor",
    "SystemHealth",
    "NavigationMessage",
    "ConsoleNavigationOutput",
    "EurocCameraSource",
    "EurocIMUSource",
]
