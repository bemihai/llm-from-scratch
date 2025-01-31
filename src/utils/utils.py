"""Utils"""
from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root folder as a Path."""
    return Path(__file__).parent.parent.parent