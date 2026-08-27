"""Create reproducible SQS benchmark tables and publication-ready plots."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

from alloyforge import ModelSpecification, build_model


METHOD_MODULES = {"sqsgenerator": "sqsgenerator", "icet": "icet"}


def case(method: str, seed: int, repeats: int, steps: int) -> dict:
    length = 3.6 * repeats
    spec = ModelSpecification.model_validate({
        "name": f"benchmark-{method}-r{repeats}-s{seed}", "random_seed": seed,
        "box": {"lengths": [length] * 3, "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .25, "Cr": .25,
                                                     "Fe": .25, "Ni": .25}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": method, "steps": steps, "neighbor_shells": 2,
                      "exclude_boundaries": False},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    started = time.perf_counter()
    result = build_model(spec)
    elapsed = time.perf_counter() - started
    before = result.metadata["sro_before"]["total_objective"]
    after = result.metadata["sro_after"]["total_objective"]
    row = {
        "method": method, "seed": seed, "repeats": repeats, "atom_count": len(result.atoms),
        "steps": steps, "seconds": elapsed, "objective_before": before,
        "objective_after": after, "improvement": before - after,
        "structure_sha256": result.metadata["structure_sha256"], "status": "PASS",
    }
    phase_shells = result.metadata["sro_after"]["phases"]["FCC"]
    for shell in phase_shells:
        values = [abs(value) for value in shell["warren_cowley"].values() if value is not None]
        row[f"shell_{shell['shell']}_mean_abs_wc"] = sum(values) / len(values) if values else 0.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output/publication"))
    parser.add_argument("--methods", nargs="+", default=["native_sqs", "sqsgenerator", "icet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 29])
    parser.add_argument("--repeats", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for method in args.methods:
        module = METHOD_MODULES.get(method)
        if module and importlib.util.find_spec(module) is None:
            skipped.append({"method": method, "reason": f"{module} is not installed"})
            continue
        for repeats in args.repeats:
            for seed in args.seeds:
                try:
                    rows.append(case(method, seed, repeats, args.steps))
                except Exception as exc:
                    rows.append({"method": method, "seed": seed, "repeats": repeats,
                                 "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    fields = sorted({key for row in rows for key in row})
    with (args.output / "sqs-benchmark.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    passed = [row for row in rows if row.get("status") == "PASS"]
    if passed:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
        methods = sorted({row["method"] for row in passed})
        for method in methods:
            selected = [row for row in passed if row["method"] == method]
            sizes = sorted({row["atom_count"] for row in selected})
            objective = [sum(r["objective_after"] for r in selected if r["atom_count"] == n) /
                         sum(r["atom_count"] == n for r in selected) for n in sizes]
            runtime = [sum(r["seconds"] for r in selected if r["atom_count"] == n) /
                       sum(r["atom_count"] == n for r in selected) for n in sizes]
            axes[0].plot(sizes, objective, marker="o", label=method)
            axes[1].plot(sizes, runtime, marker="o", label=method)
        axes[0].set(xlabel="Atom count", ylabel="Final pair objective", title="SQS quality")
        axes[1].set(xlabel="Atom count", ylabel="Runtime (s)", title="SQS runtime")
        for axis in axes: axis.grid(alpha=.25); axis.legend()
        fig.tight_layout(); fig.savefig(args.output / "sqs-quality-runtime.png", bbox_inches="tight")
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(6.5, 4.5), dpi=180)
        for method in methods:
            selected = [row for row in passed if row["method"] == method]
            shell_numbers = sorted(
                int(key.split("_")[1]) for key in selected[0] if key.startswith("shell_")
            )
            values = [
                sum(row[f"shell_{shell}_mean_abs_wc"] for row in selected) / len(selected)
                for shell in shell_numbers
            ]
            axis.plot(shell_numbers, values, marker="o", label=method)
        axis.set(xlabel="Neighbor shell", ylabel="Mean |Warren–Cowley α|",
                 title="Residual short-range order")
        axis.set_xticks(shell_numbers); axis.grid(alpha=.25); axis.legend()
        fig.tight_layout(); fig.savefig(args.output / "sro-by-neighbor-shell.png", bbox_inches="tight")
        plt.close(fig)
    summary = {
        "status": "FAIL" if any(row.get("status") == "FAIL" for row in rows) else "PASS",
        "completed_cases": len(passed), "failed_cases": len(rows) - len(passed),
        "skipped_backends": skipped, "configuration": vars(args) | {"output": str(args.output)},
    }
    (args.output / "benchmark-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if summary["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
