"""Canonical AlloyModelForge Python API.

The implementation remains in :mod:`alloyforge` for source and project-file
compatibility. New applications may import from ``alloymodelforge`` while
existing applications continue to work unchanged.
"""

from alloyforge import (
    BuildResult,
    ModelSpecification,
    build_model,
    estimate_model,
    load_project,
    load_specification,
)

__all__ = [
    "BuildResult",
    "ModelSpecification",
    "build_model",
    "estimate_model",
    "load_project",
    "load_specification",
]

__version__ = "0.2.0"
