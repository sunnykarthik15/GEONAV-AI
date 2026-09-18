"""Unit tests for GEONAV-AI."""

from pathlib import Path
import sys

# Ensure src directory is in sys.path for test discovery and execution
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
