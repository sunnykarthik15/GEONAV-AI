from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from geonav.state.state import NavigationState


def test_navigation_state_creation() -> None:
    """Test valid creation of NavigationState."""
    state = NavigationState(
        timestamp=10.0,
        position=(1.0, 2.0, -3.0),
        velocity=(0.5, -0.2, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    assert state.timestamp == 10.0
    assert state.position == (1.0, 2.0, -3.0)
    assert state.velocity == (0.5, -0.2, 0.0)
    assert state.orientation == (1.0, 0.0, 0.0, 0.0)


def test_navigation_state_timestamp_validation() -> None:
    """Test timestamp validation on NavigationState."""
    with pytest.raises(ValueError, match="finite non-negative"):
        NavigationState(
            timestamp=-1.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    with pytest.raises(TypeError, match="float or integer"):
        NavigationState(
            timestamp="invalid",  # type: ignore
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )


def test_navigation_state_position_validation() -> None:
    """Test that position vector must contain exactly 3 finite values."""
    with pytest.raises(ValueError, match="exactly 3 values"):
        NavigationState(
            timestamp=0.0,
            position=(1.0, 2.0),  # type: ignore
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="exactly 3 values"):
        NavigationState(
            timestamp=0.0,
            position=(1.0, 2.0, 3.0, 4.0),  # type: ignore
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="finite numbers"):
        NavigationState(
            timestamp=0.0,
            position=(1.0, float("inf"), 3.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )


def test_navigation_state_velocity_validation() -> None:
    """Test that velocity vector must contain exactly 3 finite values."""
    with pytest.raises(ValueError, match="exactly 3 values"):
        NavigationState(
            timestamp=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0,),  # type: ignore
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="finite numbers"):
        NavigationState(
            timestamp=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, None, 0.0),  # type: ignore
            orientation=(1.0, 0.0, 0.0, 0.0),
        )


def test_navigation_state_orientation_validation() -> None:
    """Test that orientation quaternion must contain exactly 4 finite values."""
    with pytest.raises(ValueError, match="exactly 4 values"):
        NavigationState(
            timestamp=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0),  # type: ignore
        )

    with pytest.raises(ValueError, match="exactly 4 values"):
        NavigationState(
            timestamp=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0, 0.0),  # type: ignore
        )

    with pytest.raises(ValueError, match="finite numbers"):
        NavigationState(
            timestamp=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, float("nan"), 0.0),
        )
