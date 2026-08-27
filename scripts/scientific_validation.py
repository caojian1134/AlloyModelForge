from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from alloyforge import ModelSpecification, build_model


REFERENCE_CASES = [
    ("FCC", "fcc", 3.6, [7.2, 7.2, 7.2], 32, 3.6 / math.sqrt(2), 12),
    ("BCC", "bcc", 2.87, [5.74, 5.74, 5.74], 16, 2.87 * math.sqrt(3) / 2, 8),
    ("HCP", "hcp", 2.95, [8.85, 3 * math.sqrt(3) * 2.95, 3 * 2.95 * 1.633],
     108, 2.95, 12),
]


def reference_case(name, structure, a, lengths, expected_atoms, expected_distance, expected_cn):
    spec = ModelSpecification.model_validate({
        "name": f"scientific-reference-{name}",
        "box": {"lengths": lengths, "periodic": [True, True, True]},
        "phases": [{"name": name, "structure": structure, "lattice_parameter": a,
                    "c_over_a": 1.633, "fraction": 1, "composition": {"Ti": 1}}],
        "grains": {"number": 1, "orientations": "identity"},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    coordination = result.validation["structure_statistics"]["coordination"][name]
    checks = {
        "validation_pass": result.validation["status"] == "PASS",
        "atom_count": len(result.atoms) == expected_atoms,
        "nearest_distance": abs(result.validation["minimum_distance"] - expected_distance) < 1e-8,
        "coordination": abs(coordination["mean"] - expected_cn) < 1e-8,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "atom_count": len(result.atoms), "minimum_distance": result.validation["minimum_distance"],
            "coordination": coordination, "structure_sha256": result.metadata["structure_sha256"],
            "provenance": result.metadata["provenance"]}


def sqs_case(method: str):
    spec = ModelSpecification.model_validate({
        "name": f"scientific-{method}-NiCo",
        "random_seed": 3,
        "box": {"lengths": [7.2, 7.2, 7.2], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Ni": .5, "Co": .5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": method, "steps": 2000, "neighbor_shells": 2,
                      "exclude_boundaries": False},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    try:
        result = build_model(spec)
    except RuntimeError as exc:
        if "not installed" in str(exc):
            return {"passed": None, "skipped": str(exc)}
        raise
    before = result.metadata["sro_before"]["total_objective"]
    after = result.metadata["sro_after"]["total_objective"]
    return {"passed": after <= before, "before": before, "after": after,
            "composition": result.metadata["element_counts"],
            "engine": result.metadata["sro_after"].get("engine", "alloy-model-forge-native"),
            "engine_version": result.metadata["sro_after"].get("engine_version"),
            "structure_sha256": result.metadata["structure_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("scientific-validation.json"))
    args = parser.parse_args()
    cases = {name: reference_case(*case) for case in REFERENCE_CASES for name in [case[0]]}
    cases.update({method: sqs_case(method) for method in ("native_sqs", "sqsgenerator", "icet")})
    failed = [name for name, report in cases.items() if report["passed"] is False]
    report = {"status": "FAIL" if failed else "PASS", "failed": failed, "cases": cases}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
