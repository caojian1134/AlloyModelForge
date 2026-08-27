"""Build the fixed journal golden dataset and one unified validation report."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from alloyforge import ModelSpecification, build_model
from alloyforge.provenance import runtime_provenance
from alloyforge.rendering import render_with_ovito
from journal_validation import _audit, _cases, _summary_plot


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _performance_cases(repeats: list[int], seed: int) -> list[dict]:
    rows = []
    for repeat in repeats:
        length = 3.6 * repeat
        spec = ModelSpecification.model_validate({
            "name": f"performance-fcc-r{repeat}", "random_seed": seed,
            "box": {"lengths": [length] * 3, "periodic": [True] * 3},
            "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                        "fraction": 1, "composition": {"Ni": 1}}],
            "grains": {"number": 1, "orientations": "identity"},
            "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
            "output": {"formats": ["extxyz"]},
        })
        tracemalloc.start()
        started = time.perf_counter()
        result = build_model(spec)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({
            "repeat": repeat, "atom_count": len(result.atoms), "seconds": elapsed,
            "atoms_per_second": len(result.atoms) / max(elapsed, 1e-12),
            "python_peak_memory_mb": peak / 1024 ** 2,
            "validation_status": result.validation["status"],
            "structure_sha256": result.metadata["structure_sha256"],
        })
    return rows


def _performance_plot(rows: list[dict], path: Path) -> dict:
    atoms = np.asarray([row["atom_count"] for row in rows], dtype=float)
    seconds = np.asarray([row["seconds"] for row in rows], dtype=float)
    memory = np.asarray([row["python_peak_memory_mb"] for row in rows], dtype=float)
    fit = np.polyfit(atoms, seconds, 1)
    prediction = np.polyval(fit, atoms)
    denominator = np.sum((seconds - np.mean(seconds)) ** 2)
    r_squared = 1.0 - np.sum((seconds - prediction) ** 2) / denominator if denominator else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)
    axes[0].plot(atoms, seconds, "o-", color="#176b52")
    axes[0].plot(atoms, prediction, "--", color="#9c783c", label=f"linear fit R²={r_squared:.3f}")
    axes[0].set(xlabel="Atom count", ylabel="Build time (s)", title="Construction scaling")
    axes[0].legend()
    axes[1].plot(atoms, memory, "o-", color="#4169b1")
    axes[1].set(xlabel="Atom count", ylabel="Peak Python allocation (MB)",
                title="Measured Python memory scaling")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"runtime_linear_fit_slope": float(fit[0]), "runtime_linear_fit_intercept": float(fit[1]),
            "runtime_fit_r_squared": float(r_squared), "memory_metric": "tracemalloc Python allocations"}


def _radial_plot(inclusions: dict, path: Path) -> bool:
    if not inclusions:
        return False
    fig, axis = plt.subplots(figsize=(7, 4.5), dpi=180)
    for name, report in inclusions.items():
        profile = report["radial_phase_profile"]
        radius = [(row["r_inner"] + row["r_outer"]) / 2 for row in profile]
        fraction = [np.nan if row["target_phase_fraction"] is None else row["target_phase_fraction"]
                    for row in profile]
        axis.plot(radius, fraction, marker="o", label=name)
    axis.set(xlabel="Radius from specified center (Å)", ylabel="Target phase atom fraction",
             ylim=(-.03, 1.03), title="Core-shell radial phase profile")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _grain_plot(grains: dict, path: Path) -> bool:
    if not grains:
        return False
    ids = list(grains)
    diameters = [grains[key]["equivalent_sphere_diameter"] for key in ids]
    boundary = [grains[key]["boundary_fraction"] for key in ids]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=180)
    axes[0].bar(ids, diameters, color="#176b52")
    axes[0].set(xlabel="Grain ID", ylabel="Equivalent diameter (Å)", title="Grain size distribution")
    axes[1].bar(ids, boundary, color="#d58532")
    axes[1].set(xlabel="Grain ID", ylabel="Boundary atom fraction", ylim=(0, 1),
                title="Per-grain interface fraction")
    for axis in axes: axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def _markdown(payload: dict) -> str:
    lines = [
        "# AlloyModelForge journal golden benchmark report", "",
        f"**Overall status:** {payload['status']}", "",
        f"Generated: `{payload['generated_at']}`  ",
        f"Python: `{payload['environment']['python']}`  ",
        f"Platform: `{payload['environment']['platform']}`", "",
        "## Acceptance summary", "",
        f"- Golden cases: {payload['case_count']}",
        f"- Scientific checks: {payload['passed_checks']} / {payload['total_checks']} passed",
        f"- Deterministic rebuilds: {payload['deterministic_matches']} / {payload['case_count']} matched",
        f"- OVITO images: {payload['ovito_image_count']}",
        f"- Runtime linear-fit R²: {payload['performance_fit']['runtime_fit_r_squared']:.4f}", "",
        "![Benchmark overview](benchmark-overview.png)", "",
        "![Performance scaling](performance-scaling.png)", "",
        "## Case matrix", "",
        "| Case | Atoms | Validation | Checks | Deterministic | Time (s) |", "|---|---:|---|---:|---|---:|",
    ]
    for name, report in payload["cases"].items():
        lines.append(f"| {name} | {report['atom_count']} | {report['validation_status']} | "
                     f"{report['passed_check_count']}/{report['check_count']} | "
                     f"{'yes' if report['deterministic_match'] else 'no'} | {report['elapsed_seconds']:.3f} |")
    lines += ["", "## Structural validity metrics", "",
              "| Case | Minimum spacing (Å) | Overdense fraction | Boundary fraction | Populated grains | Empty grains |",
              "|---|---:|---:|---:|---:|---|"]
    for name, report in payload["cases"].items():
        metric = report["scientific_metrics"]
        lines.append(f"| {name} | {metric['minimum_spacing']:.4f} | {metric['overdense_fraction']:.4f} | "
                     f"{metric['boundary_fraction']:.4f} | {metric['populated_grains']} | "
                     f"{metric['empty_grains'] or 'none'} |")
    lines += ["", "## Scientific interpretation", "",
              "CNA/PTM values describe the unrelaxed initial geometry. Interface atoms classified as Other are not automatically construction errors. Acceptance is based on the predeclared geometric, chemical, round-trip and region-specific criteria.", ""]
    if payload.get("core_shell_radial_plot"):
        lines += ["## Core-shell radial validity", "",
                  "| Region | Atoms | Max extent (Å) | Center offset (Å) | Components | Largest component |",
                  "|---|---:|---:|---:|---:|---:|"]
        core_metrics = payload["cases"]["core_shell_particle"]["core_shell_metrics"]
        for name, metric in core_metrics.items():
            lines.append(f"| {name} | {metric['atom_count']} | {metric['maximum_radial_extent']:.4f} | "
                         f"{metric['center_of_atoms_offset']:.4f} | {metric['connected_component_count']} | "
                         f"{metric['largest_component_fraction']:.3f} |")
        lines += ["", "![Core-shell radial profile](core-shell-radial-profile.png)", ""]
    if payload.get("grain_statistics_plot"):
        lines += ["## Polycrystal statistics", "", "![Grain statistics](grain-statistics.png)", ""]
    lines += ["## OVITO publication images", ""]
    for image in payload["ovito_images"]:
        lines += [f"### {image['case']} — {image['mode']} / {image['filter']}", "",
                  f"![{image['case']} {image['mode']}]({image['path']})", ""]
    lines += ["## Reproduction", "", "```powershell",
              "conda run -n alloymodelforge-dev python scripts/golden_benchmark.py --output validation-output/golden-benchmark",
              "```", "", "See `docs/PUBLICATION_WORKFLOW.md` for reporting requirements and limitations."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/golden-benchmark"))
    parser.add_argument("--render-size", nargs=2, type=int, default=[1200, 900])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load((ROOT / "benchmarks" / "golden_cases.yaml").read_text(encoding="utf-8"))
    reports, all_checks, ovito_images = {}, [], []
    radial_inclusions, representative_grains = {}, {}
    started_all = time.perf_counter()
    for name, data in _cases().items():
        case_dir = args.output / "cases" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        spec = ModelSpecification.model_validate(data)
        started = time.perf_counter(); result = build_model(spec); elapsed = time.perf_counter() - started
        result.write_outputs(case_dir)
        audit = _audit(name, result, elapsed, case_dir)
        repeat = build_model(spec)
        deterministic = repeat.metadata["structure_sha256"] == result.metadata["structure_sha256"]
        audit["deterministic_match"] = deterministic
        audit["check_count"] = len(audit["checks"])
        audit["passed_check_count"] = sum(item["passed"] for item in audit["checks"])
        geometry = result.validation["geometry_quality"]
        grain_stats = geometry["grain_statistics"]
        audit["scientific_metrics"] = {
            "minimum_spacing": geometry["spacing"]["minimum"],
            "overdense_fraction": geometry["spacing"]["overdense_fraction"],
            "boundary_fraction": result.validation["boundary_fraction"],
            "populated_grains": grain_stats["populated_count"],
            "empty_grains": grain_stats["empty_grain_ids"],
            "equivalent_diameter": grain_stats["equivalent_diameter"],
            "orientation_statistics": result.metadata["orientation_analysis"]["statistics"],
        }
        if name == "core_shell_particle":
            audit["core_shell_metrics"] = geometry["inclusions"]
        reports[name] = audit
        all_checks.extend({"case": name, **item} for item in audit["checks"])
        if name == "core_shell_particle": radial_inclusions = result.validation["geometry_quality"]["inclusions"]
        if name == "fcc_polycrystal": representative_grains = result.validation["geometry_quality"]["grains"]
        render_specs = [("phase", "all")]
        if name in {"single_fcc", "rectangular_fcc", "fcc_polycrystal", "fcc_bcc_dual_phase"}:
            render_specs.append(("cna", "half-x" if name != "single_fcc" else "all"))
        if name == "core_shell_particle": render_specs = [("inclusion", "half-x"), ("phase", "half-x")]
        for mode, filter_mode in render_specs:
            image_path = case_dir / f"ovito-{mode}-{filter_mode}.png"
            render_report = render_with_ovito(case_dir / "model.extxyz", image_path,
                                              color_mode=mode, filter_mode=filter_mode,
                                              width=args.render_size[0], height=args.render_size[1], dpi=300)
            ovito_images.append({"case": name, "mode": mode, "filter": filter_mode,
                                 "path": image_path.relative_to(args.output).as_posix(),
                                 "atom_count": render_report["atom_count"]})
    performance_rows = _performance_cases(manifest["performance"]["repeats"],
                                          manifest["performance"]["random_seed"])
    performance_fit = _performance_plot(performance_rows, args.output / "performance-scaling.png")
    _write_csv(args.output / "performance.csv", performance_rows)
    _write_csv(args.output / "scientific-checks.csv", all_checks)
    radial_plot = _radial_plot(radial_inclusions, args.output / "core-shell-radial-profile.png")
    grain_plot = _grain_plot(representative_grains, args.output / "grain-statistics.png")
    _summary_plot(reports, args.output / "benchmark-overview.png")
    failed = [name for name, report in reports.items() if not report["passed"]]
    deterministic_matches = sum(report["deterministic_match"] for report in reports.values())
    criteria = manifest["acceptance"]
    status = "PASS" if (not failed and deterministic_matches == len(reports)
                               and performance_fit["runtime_fit_r_squared"] >= criteria["performance_fit_min_r_squared"]
                               and len(ovito_images) > 0) else "FAIL"
    payload = {
        "schema_version": manifest["schema_version"], "dataset_name": manifest["dataset_name"],
        "status": status, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.perf_counter() - started_all,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "provenance": runtime_provenance()},
        "case_count": len(reports), "failed_cases": failed,
        "passed_checks": sum(row["passed"] for row in all_checks), "total_checks": len(all_checks),
        "deterministic_matches": deterministic_matches, "ovito_image_count": len(ovito_images),
        "performance_fit": performance_fit, "performance": performance_rows,
        "core_shell_radial_plot": radial_plot, "grain_statistics_plot": grain_plot,
        "ovito_images": ovito_images, "cases": reports,
    }
    (args.output / "golden-benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    (args.output / "golden-manifest.json").write_text(json.dumps({
        "dataset": manifest, "case_hashes": {name: report["structure_sha256"] for name, report in reports.items()},
        "status": status,
    }, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "case_count", "passed_checks",
                                                    "total_checks", "deterministic_matches",
                                                    "ovito_image_count", "elapsed_seconds")}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
