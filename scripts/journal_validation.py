"""Build and audit journal-grade AlloyModelForge validation models.

The suite intentionally combines independent structural identification (OVITO),
geometric quality, chemistry/SRO targets, file round-trips, and visual inspection
artifacts.  It exits non-zero when any predeclared scientific criterion fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from alloyforge import ModelSpecification, build_model


ROOT = Path(__file__).resolve().parents[1]
PHASE_COLORS = ["#2f80ed", "#eb5757", "#27ae60", "#9b51e0", "#f2994a"]
REGION_COLORS = ["#bdbdbd", "#f2c94c", "#eb5757", "#9b51e0", "#56ccf2"]
CNA_COLORS = ["#777777", "#2f80ed", "#27ae60", "#eb5757", "#9b51e0"]


def _load_example(name: str) -> dict:
    return yaml.safe_load((ROOT / "examples" / name).read_text(encoding="utf-8"))


def _cases() -> dict[str, dict]:
    single = {
        "name": "Journal reference FCC",
        "random_seed": 101,
        "box": {"lengths": [14.4, 14.4, 14.4], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Ni": 1}}],
        "grains": {"number": 1, "orientations": "identity"},
        "analysis": {"ovito_enabled": True, "methods": ["cna", "ptm"],
                     "rdf_enabled": True},
        "output": {"formats": ["extxyz", "lammps-data", "vasp", "cif"]},
    }
    cases = {
        "single_fcc": single,
        "rectangular_fcc": _load_example("fcc_rectangular_cell.yaml"),
        "fcc_polycrystal": _load_example("fcc_polycrystal.yaml"),
        "fcc_bcc_dual_phase": _load_example("fcc_bcc_dual_phase.yaml"),
        "spherical_precipitate": _load_example("fcc_bcc_spherical_precipitate.yaml"),
        "region_constrained_sqs": _load_example("fcc_polycrystal_region_sqs.yaml"),
        "core_shell_particle": _load_example("fcc_bcc_hcp_core_shell.yaml"),
        "directional_core_shell_sqs": _load_example(
            "fcc_bcc_hcp_core_shell_directional_sqs.yaml"
        ),
    }
    for data in cases.values():
        data.setdefault("analysis", {})
        data["analysis"].update({"ovito_enabled": True, "methods": ["cna", "ptm"]})
        data.setdefault("output", {})["formats"] = ["extxyz", "lammps-data", "vasp", "cif"]
    return cases


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def _render(atoms, values: np.ndarray, colors: list[str], path: Path, title: str,
            labels: dict[int, str] | None = None, mask: np.ndarray | None = None) -> None:
    sample = np.flatnonzero(mask) if mask is not None else np.arange(len(atoms))
    if len(sample) > 8000:
        sample = sample[:: max(1, len(sample) // 8000)]
    fig = plt.figure(figsize=(7.2, 6.2), dpi=180)
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        *atoms.positions[sample].T,
        c=[colors[int(value) % len(colors)] for value in values[sample]],
        s=10, alpha=.9, depthshade=False, linewidths=0,
    )
    axis.set(title=title, xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")
    axis.set_box_aspect(atoms.cell.lengths())
    axis.view_init(elev=22, azim=35)
    if labels:
        present = sorted(set(map(int, values[sample])))
        axis.legend(
            handles=[Patch(color=colors[value % len(colors)], label=labels.get(value, str(value)))
                     for value in present],
            loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _check(checks: list[dict], name: str, value, criterion: str, passed: bool) -> None:
    checks.append({"name": name, "value": value, "criterion": criterion, "passed": bool(passed)})


def _expected_cna(atoms, spec: ModelSpecification) -> np.ndarray:
    mapping = {"fcc": 1, "hcp": 2, "bcc": 3}
    phase_ids = atoms.arrays["phase_id"]
    return np.asarray([mapping.get(spec.phases[int(pid)].structure, 0) for pid in phase_ids])


def _audit(case_name: str, result, elapsed: float, case_dir: Path) -> dict:
    atoms, spec = result.atoms, result.specification
    checks: list[dict] = []
    validation = result.validation
    geometry = validation["geometry_quality"]
    spacing = geometry["spacing"]
    ovito = result.metadata.get("ovito_analysis", {})
    cna = atoms.arrays.get("cna_type", np.zeros(len(atoms), dtype=int))
    ptm = atoms.arrays.get("ptm_type", np.zeros(len(atoms), dtype=int))
    expected = _expected_cna(atoms, spec)
    interior = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=int)) == 0
    boundary = ~interior

    _check(checks, "no_validation_failures", len(validation["failures"]), "= 0",
           not validation["failures"])
    _check(checks, "minimum_distance", spacing["minimum"], "> 0.5 × smallest phase NN",
           spacing["minimum"] > .5 * min(
               phase.lattice_parameter / np.sqrt(2) if phase.structure == "fcc"
               else phase.lattice_parameter * np.sqrt(3) / 2 if phase.structure == "bcc"
               else min(phase.lattice_parameter, phase.lattice_parameter * phase.c_over_a)
               for phase in spec.phases
           ))
    _check(checks, "overdense_fraction", spacing["overdense_fraction"], "<= 0.10",
           spacing["overdense_fraction"] <= .10)
    low_density_interior = geometry["local_density"].get("low_density_fraction_interior")
    _check(checks, "low_density_fraction_interior", low_density_interior,
           "<= 0.30", low_density_interior is not None and low_density_interior <= .30)
    low_density_boundary = geometry["local_density"].get("low_density_fraction_boundary")
    if low_density_boundary is not None:
        _check(checks, "boundary_density_disorder", low_density_boundary - low_density_interior,
               ">= 0", low_density_boundary >= low_density_interior)
    _check(checks, "ovito_available", ovito.get("available"), "is true", ovito.get("available") is True)

    cna_interior = _fraction(cna[interior] == expected[interior]) if np.any(interior) else 0.0
    ptm_interior = _fraction(ptm[interior] == expected[interior]) if np.any(interior) else 0.0
    cna_boundary = _fraction(cna[boundary] == expected[boundary]) if np.any(boundary) else 0.0
    threshold = .98 if case_name in {"single_fcc", "rectangular_fcc"} else .55
    _check(checks, "cna_expected_interior", cna_interior, f">= {threshold:.2f}",
           cna_interior >= threshold)
    _check(checks, "ptm_expected_interior", ptm_interior, f">= {threshold:.2f}",
           ptm_interior >= threshold)
    if np.any(boundary):
        _check(checks, "interior_more_crystalline_than_boundary",
               cna_interior - cna_boundary, ">= 0", cna_interior >= cna_boundary)

    for phase_name, phase_report in validation["phases"].items():
        _check(checks, f"phase_fraction_{phase_name}", phase_report["volume_fraction_deviation"],
               f"<= {spec.cleanup.maximum_phase_fraction_deviation}",
               phase_report["volume_fraction_deviation"] <= spec.cleanup.maximum_phase_fraction_deviation)
        target = next(phase.composition for phase in spec.phases if phase.name == phase_name)
        max_error = max(abs(phase_report["composition"].get(e, 0) - c) for e, c in target.items())
        _check(checks, f"composition_{phase_name}", max_error,
               f"<= {spec.cleanup.maximum_composition_deviation}",
               max_error <= spec.cleanup.maximum_composition_deviation)

    roundtrip = result.metadata.get("output_roundtrip", {})
    for format_name, report in roundtrip.items():
        _check(checks, f"roundtrip_{format_name}", report.get("passed"), "is true",
               report.get("passed") is True)

    if case_name == "spherical_precipitate":
        inclusion_count = int(np.count_nonzero(atoms.arrays.get("inclusion_id", -np.ones(len(atoms))) >= 0))
        _check(checks, "explicit_inclusion_atoms", inclusion_count, "> 0", inclusion_count > 0)
        _check(checks, "two_phases_present", validation["phase_count"], "= 2",
               validation["phase_count"] == 2)

    if case_name == "rectangular_fcc":
        lengths = atoms.cell.lengths()
        _check(checks, "rectangular_cell_lengths", lengths.tolist(), "= [14.4, 18.0, 21.6]",
               np.allclose(lengths, [14.4, 18.0, 21.6]))
        _check(checks, "rectangular_cell_atom_count", len(atoms), "= 480", len(atoms) == 480)

    if case_name == "core_shell_particle":
        center = atoms.cell.lengths() / 2
        delta = atoms.positions - center
        box = atoms.cell.lengths()
        delta -= box * np.round(delta / box)
        radius = np.linalg.norm(delta, axis=1)
        phase_ids = atoms.arrays["phase_id"]
        core, shell, matrix = phase_ids == 2, phase_ids == 1, phase_ids == 0
        _check(checks, "three_phases_present", validation["phase_count"], "= 3",
               validation["phase_count"] == 3)
        _check(checks, "core_inside_radius", float(np.max(radius[core])), "<= 4.0 Å",
               np.any(core) and np.max(radius[core]) <= 4.0 + 1e-8)
        _check(checks, "shell_annulus", [float(np.min(radius[shell])), float(np.max(radius[shell]))],
               "4.0 < r <= 7.5 Å", np.any(shell) and np.min(radius[shell]) > 4.0 - 1e-8
               and np.max(radius[shell]) <= 7.5 + 1e-8)
        _check(checks, "matrix_outside_shell", float(np.min(radius[matrix])), "> 7.5 Å",
               np.any(matrix) and np.min(radius[matrix]) > 7.5 - 1e-8)

    if case_name == "region_constrained_sqs":
        before = result.metadata["sro_before"]
        after = result.metadata["sro_after"]
        _check(checks, "optimization_objective", after["optimization_objective"],
               f"< {before['optimization_objective']}",
               after["optimization_objective"] < before["optimization_objective"])
        for name, before_region in before["region_constraints"].items():
            after_region = after["region_constraints"][name]
            _check(checks, f"region_composition_{name}", after_region["composition_l1_error"],
                   f"< {before_region['composition_l1_error']}",
                   after_region["composition_l1_error"] < before_region["composition_l1_error"])

    if case_name == "directional_core_shell_sqs":
        before = result.metadata["sro_before"]
        after = result.metadata["sro_after"]
        directional_name = "core-to-shell-contact"
        triplet_name = "matrix-angular-randomness"
        before_directional = before["directional_sro_constraints"][directional_name]
        after_directional = after["directional_sro_constraints"][directional_name]
        before_triplet = before["triplet_constraints"][triplet_name]
        after_triplet = after["triplet_constraints"][triplet_name]
        _check(checks, "directional_edges", after_directional["edge_count"], "> 0",
               after_directional["edge_count"] > 0)
        _check(checks, "directional_objective", after_directional["objective"],
               f"<= {before_directional['objective']}",
               after_directional["objective"] <= before_directional["objective"])
        _check(checks, "triplet_motifs", after_triplet["motif_count"], "> 0",
               after_triplet["motif_count"] > 0)
        _check(checks, "triplet_objective", after_triplet["objective"],
               f"<= {before_triplet['objective']}",
               after_triplet["objective"] <= before_triplet["objective"])

    phase_image = case_dir / "view-phase.png"
    region_image = case_dir / "view-region.png"
    cna_image = case_dir / "view-ovito-cna.png"
    element_image = case_dir / "view-elements.png"
    grain_image = case_dir / "view-grains.png"
    interface_image = case_dir / "view-interface-only.png"
    section_image = case_dir / "view-section-phase.png"
    phase_labels = {i: phase.name for i, phase in enumerate(spec.phases)}
    region_labels = {0: "Interior", 1: "Grain boundary", 2: "Phase boundary",
                     3: "Triple junction", 4: "Mixed boundary"}
    structure_labels = {0: "Other", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO"}
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    element_names = sorted(set(symbols.tolist()))
    element_ids = np.asarray([element_names.index(symbol) for symbol in symbols], dtype=int)
    _render(atoms, element_ids, PHASE_COLORS + REGION_COLORS, element_image,
            f"{case_name}: chemical elements",
            {i: symbol for i, symbol in enumerate(element_names)})
    _render(atoms, atoms.arrays["grain_id"], PHASE_COLORS + REGION_COLORS, grain_image,
            f"{case_name}: grain identity",
            {int(value): f"Grain {int(value)}" for value in np.unique(atoms.arrays["grain_id"])})
    _render(atoms, atoms.arrays["phase_id"], PHASE_COLORS, phase_image,
            f"{case_name}: phase identity", phase_labels)
    _render(atoms, atoms.arrays.get("primary_region", atoms.arrays["region_type"]),
            REGION_COLORS, region_image, f"{case_name}: microstructural regions", region_labels)
    _render(atoms, cna, CNA_COLORS, cna_image, f"{case_name}: OVITO CNA", structure_labels)
    _render(atoms, atoms.arrays.get("primary_region", atoms.arrays["region_type"]),
            REGION_COLORS, interface_image, f"{case_name}: interfaces only", region_labels,
            atoms.arrays["region_type"] > 0)
    _render(atoms, atoms.arrays["phase_id"], PHASE_COLORS, section_image,
            f"{case_name}: x-half section", phase_labels,
            atoms.positions[:, 0] <= atoms.cell.lengths()[0] / 2)

    return {
        "passed": all(item["passed"] for item in checks),
        "atom_count": len(atoms),
        "elapsed_seconds": elapsed,
        "validation_status": validation["status"],
        "warnings": validation["warnings"],
        "structure_sha256": result.metadata["structure_sha256"],
        "cna_counts": dict(Counter(map(int, cna))),
        "ptm_counts": dict(Counter(map(int, ptm))),
        "cna_expected_interior": cna_interior,
        "cna_expected_boundary": cna_boundary,
        "checks": checks,
        "files": [path.name for path in case_dir.iterdir() if path.is_file()],
        "images": [element_image.name, grain_image.name, phase_image.name, region_image.name,
                   cna_image.name, interface_image.name, section_image.name],
    }


def _summary_plot(reports: dict[str, dict], path: Path) -> None:
    names = list(reports)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    axes[0].bar(x - .18, [reports[n]["cna_expected_interior"] for n in names], .36,
                label="Interior")
    axes[0].bar(x + .18, [reports[n]["cna_expected_boundary"] for n in names], .36,
                label="Boundary/interface")
    axes[0].set(ylabel="Correct OVITO CNA fraction", ylim=(0, 1.05), xticks=x,
                xticklabels=names, title="Independent structural identification")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(x, [reports[n]["elapsed_seconds"] for n in names], color="#2f80ed")
    axes[1].set(ylabel="Build and analysis time (s)", xticks=x, xticklabels=names,
                title="End-to-end validation runtime")
    axes[1].tick_params(axis="x", rotation=25); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("validation-output/journal-validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    rows = []
    for name, data in _cases().items():
        case_dir = args.output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        spec = ModelSpecification.model_validate(data)
        started = time.perf_counter()
        result = build_model(spec)
        result.write_outputs(case_dir)
        elapsed = time.perf_counter() - started
        reports[name] = _audit(name, result, elapsed, case_dir)
        for check in reports[name]["checks"]:
            rows.append({"case": name, **check})
    with (args.output / "scientific-checks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "name", "value", "criterion", "passed"])
        writer.writeheader(); writer.writerows(rows)
    _summary_plot(reports, args.output / "journal-validation-overview.png")
    failed = [name for name, report in reports.items() if not report["passed"]]
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "failed": failed,
        "case_count": len(reports),
        "passed_checks": sum(row["passed"] for row in rows),
        "total_checks": len(rows),
        "cases": reports,
    }
    (args.output / "journal-validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in
                      ("status", "failed", "case_count", "passed_checks", "total_checks")}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
