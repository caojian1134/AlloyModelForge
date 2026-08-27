from __future__ import annotations

import hashlib
import json

import numpy as np

from alloyforge import ModelSpecification, build_model


SPECIFICATION = {
    "name": "AlloyModelForge reproducibility baseline",
    "random_seed": 20260825,
    "box": {"lengths": [18, 18, 18], "periodic": [True, True, True]},
    "phases": [{
        "name": "FCC",
        "structure": "fcc",
        "lattice_parameter": 3.6,
        "fraction": 1.0,
        "composition": {"Co": 0.5, "Ni": 0.5},
    }],
    "grains": {"number": 3, "orientations": "random"},
    "chemistry": {"method": "random"},
    "analysis": {"ovito_enabled": False, "methods": []},
    "cleanup": {"minimum_distance_factor": 0.6, "maximum_deleted_fraction": 0.2},
}

EXPECTED = {
    "sha256": "96b3b5a0860682320679c70617c7134e7179aea5f7fbee6dbbadcf18c6f49753",
    "atom_count": 459,
    "minimum_distance": 1.554171787371,
    "element_counts": {"Co": 230, "Ni": 229},
    "validation_status": "WARNING",
}


def fingerprint() -> dict:
    result = build_model(ModelSpecification.model_validate(SPECIFICATION))
    digest = hashlib.sha256()
    digest.update(np.round(result.atoms.positions, 8).astype("<f8").tobytes())
    digest.update("\0".join(result.atoms.get_chemical_symbols()).encode())
    for name in ("grain_id", "phase_id", "region_type"):
        digest.update(np.asarray(result.atoms.arrays[name], dtype="<i4").tobytes())
    return {
        "sha256": digest.hexdigest(),
        "atom_count": len(result.atoms),
        "minimum_distance": round(result.validation["minimum_distance"], 12),
        "element_counts": result.metadata["element_counts"],
        "validation_status": result.validation["status"],
    }


if __name__ == "__main__":
    result = fingerprint()
    print(json.dumps(result, sort_keys=True))
    mismatches = {
        key: {"expected": expected, "actual": result[key]}
        for key, expected in EXPECTED.items() if result[key] != expected
    }
    if mismatches:
        raise SystemExit(f"Reproducibility baseline mismatch: {json.dumps(mismatches)}")
