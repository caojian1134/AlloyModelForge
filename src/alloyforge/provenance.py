from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform

import numpy as np
from ase import Atoms


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def structure_fingerprint(atoms: Atoms) -> str:
    """Return a deterministic digest of the complete generated structure."""
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.cell.array, dtype="<f8").tobytes())
    digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
    digest.update(np.round(atoms.positions, 10).astype("<f8").tobytes())
    digest.update("\0".join(atoms.get_chemical_symbols()).encode("utf-8"))
    for name in sorted(atoms.arrays):
        if name in {"numbers", "positions"}:
            continue
        values = np.ascontiguousarray(atoms.arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def runtime_provenance() -> dict:
    dependencies = {
        name: package_version(name)
        for name in ("ase", "numpy", "scipy", "pydantic", "icet", "sqsgenerator", "ovito")
    }
    return {
        "alloy_model_forge_version": (
            package_version("alloy-model-forge")
            or package_version("alloyforge")
            or "0.2.0+source"
        ),
        # Kept for reading and comparing projects produced before the public rename.
        "alloyforge_version": (
            package_version("alloy-model-forge")
            or package_version("alloyforge")
            or "0.2.0+source"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
