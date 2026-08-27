"""Generate numerical and visual OVITO gold-standard validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk

from alloyforge import ModelSpecification, build_model
from alloyforge.analysis import analyze_with_ovito


CASES = [
    ("FCC", "fcc", "Ni", 3.52, (4, 4, 4), 0.99),
    ("BCC", "bcc", "Fe", 2.87, (5, 5, 5), 0.99),
    ("HCP", "hcp", "Ti", 2.95, (5, 5, 4), 0.98),
]
COLORS = {0: "#777777", 1: "#2f80ed", 2: "#27ae60", 3: "#eb5757", 4: "#9b51e0"}


def make_spec(name: str, structure: str, element: str, lattice: float) -> ModelSpecification:
    return ModelSpecification.model_validate({
        "name": f"OVITO gold standard {name}",
        "phases": [{"name": name, "structure": structure, "lattice_parameter": lattice,
                    "fraction": 1, "composition": {element: 1}}],
        "grains": {"number": 1, "orientations": "identity"},
        "analysis": {"ovito_enabled": True, "methods": ["cna", "ptm"],
                     "rdf_enabled": False},
    })


def render(atoms, values: np.ndarray, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    sample = np.arange(len(atoms))
    if len(sample) > 5000:
        sample = sample[:: max(1, len(sample) // 5000)]
    ax.scatter(*atoms.positions[sample].T, c=[COLORS.get(int(v), "black") for v in values[sample]],
               s=11, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_box_aspect(atoms.cell.lengths())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/ovito-gold"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, structure, element, lattice, repeat, threshold in CASES:
        kwargs = {"a": lattice}
        if structure == "hcp":
            kwargs["c"] = lattice * 1.633
        atoms = bulk(element, structure, **kwargs).repeat(repeat)
        report = analyze_with_ovito(atoms, make_spec(name, structure, element, lattice))
        if not report["available"]:
            results[name] = {"passed": None, "message": report.get("message")}
            continue
        expected_count = report["methods"]["cna"]["counts"].get(name, 0)
        fraction = expected_count / len(atoms)
        ptm_count = report["methods"]["ptm"]["counts"].get(name, 0)
        ptm_fraction = ptm_count / len(atoms)
        passed = fraction >= threshold and ptm_fraction >= threshold
        image = args.output / f"{name.lower()}-ovito-cna.png"
        render(atoms, atoms.arrays["cna_type"], image, f"{name}: OVITO CNA classifications")
        results[name] = {
            "passed": passed, "atom_count": len(atoms), "threshold": threshold,
            "cna_expected_fraction": fraction, "ptm_expected_fraction": ptm_fraction,
            "cna_counts": report["methods"]["cna"]["counts"],
            "ptm_counts": report["methods"]["ptm"]["counts"],
            "image": image.name,
        }
    complex_specs = {
        "FCC_POLYCRYSTAL": {
            "name": "OVITO gold polycrystal", "random_seed": 41,
            "box": {"lengths": [25.2, 25.2, 25.2], "periodic": [True, True, True]},
            "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                        "fraction": 1, "composition": {"Ni": 1}}],
            "grains": {"number": 4, "orientations": "random"},
            "cleanup": {"boundary_width": 3.2, "maximum_deleted_fraction": .25,
                        "maximum_boundary_fraction": .8},
            "analysis": {"ovito_enabled": True, "methods": ["cna", "ptm"],
                         "rdf_enabled": False},
        },
        "FCC_BCC_DUAL_PHASE": {
            "name": "OVITO gold dual phase", "random_seed": 12,
            "box": {"lengths": [25.2, 25.2, 25.2], "periodic": [True, True, True]},
            "phases": [
                {"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                 "fraction": .5, "composition": {"Ni": 1}},
                {"name": "BCC", "structure": "bcc", "lattice_parameter": 2.87,
                 "fraction": .5, "composition": {"Fe": 1}},
            ],
            "grains": {"number": 2, "orientations": "identity"},
            "cleanup": {"boundary_width": 3.2, "maximum_deleted_fraction": .25,
                        "maximum_boundary_fraction": .8, "maximum_phase_fraction_deviation": .2},
            "analysis": {"ovito_enabled": True, "methods": ["cna", "ptm"],
                         "rdf_enabled": False},
        },
    }
    for name, spec_data in complex_specs.items():
        result = build_model(ModelSpecification.model_validate(spec_data))
        values = result.atoms.arrays["cna_type"]
        interior = result.atoms.arrays["region_type"] == 0
        phase_ids = result.atoms.arrays["phase_id"]
        if name == "FCC_POLYCRYSTAL":
            expected = values == 1
        else:
            expected = ((phase_ids == 0) & (values == 1)) | ((phase_ids == 1) & (values == 3))
        interior_fraction = float(np.mean(expected[interior])) if np.any(interior) else 0.0
        boundary = ~interior
        boundary_fraction = float(np.mean(expected[boundary])) if np.any(boundary) else 0.0
        passed = interior_fraction >= 0.60 and interior_fraction >= boundary_fraction
        image = args.output / f"{name.lower()}-ovito-cna.png"
        render(result.atoms, values, image, f"{name}: OVITO CNA classifications")
        results[name] = {
            "passed": passed, "atom_count": len(result.atoms),
            "interior_expected_fraction": interior_fraction,
            "boundary_expected_fraction": boundary_fraction,
            "criterion": "interior >= 0.60 and interior >= boundary",
            "cna_counts": result.metadata["ovito_analysis"]["methods"]["cna"]["counts"],
            "region_counts": result.validation["geometry_quality"]["regions"],
            "image": image.name,
        }
    failed = [name for name, item in results.items() if item["passed"] is False]
    payload = {"status": "FAIL" if failed else "PASS", "failed": failed, "cases": results}
    (args.output / "ovito-gold-standard.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
