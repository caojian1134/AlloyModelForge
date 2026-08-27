from __future__ import annotations

from importlib.metadata import version
import math

import numpy as np
from ase.build import bulk

from ..chemistry import calculate_sro_report
from ..config import ModelSpecification
from ..chemistry import _ideal_shell_distances
from ..progress import CancellationToken, ProgressCallback, ProgressEvent


def _require_periodic_reference(spec: ModelSpecification, backend: str) -> None:
    if len(spec.phases) != 1 or spec.grains.number != 1:
        raise ValueError(
            f"{backend} requires one globally periodic phase and one grain. "
            "Use chemistry.method=native_sqs for polycrystal or multiphase geometry."
        )
    if not all(spec.box.periodic):
        raise ValueError(f"{backend} requires periodic boundaries in all directions")
    if spec.grains.orientations != "identity":
        raise ValueError(
            f"{backend} requires an identity/commensurate reference lattice; "
            "an arbitrary rotated lattice is not commensurate with this box"
        )


def _run_sqsgenerator(atoms, spec: ModelSpecification, token: CancellationToken):
    try:
        import sqsgenerator
    except ImportError as exc:
        raise RuntimeError(
            "The sqsgenerator backend is not installed. Install alloy-model-forge[sqs]."
        ) from exc
    _require_periodic_reference(spec, "sqsgenerator")
    token.raise_if_cancelled()
    symbols = atoms.get_chemical_symbols()
    config = {
        "structure": {
            "lattice": atoms.cell.array.tolist(),
            "coords": atoms.get_scaled_positions(wrap=True).tolist(),
            "species": symbols,
        },
        "composition": {
            element: symbols.count(element) for element in sorted(set(symbols))
        },
        "iterations": spec.chemistry.steps,
        "shell_weights": {
            shell: 1.0 / shell for shell in range(1, spec.chemistry.neighbor_shells + 1)
        },
        "max_results_per_objective": 1,
        "seed": [spec.random_seed],
        "thread_config": [1],
    }
    parsed = sqsgenerator.parse_config(config)
    if type(parsed).__name__ == "ParseError":
        raise RuntimeError(f"sqsgenerator configuration error at {parsed.key}: {parsed.msg}")
    pack = sqsgenerator.optimize(parsed)
    result = pack.best()
    optimized = sqsgenerator.to_ase(result.structure())
    if len(optimized) != len(atoms):
        raise RuntimeError("sqsgenerator returned a structure with a different atom count")
    atoms.set_chemical_symbols(optimized.get_chemical_symbols())
    objective = result.objective() if callable(result.objective) else result.objective
    return {
        "engine": "sqsgenerator",
        "engine_version": ".".join(map(str, sqsgenerator.__version__)),
        "objective": float(objective),
        "iterations": spec.chemistry.steps,
    }


def _run_icet(atoms, spec: ModelSpecification, token: CancellationToken):
    try:
        from icet import ClusterSpace
        from icet.tools.structure_generation import generate_sqs_from_supercells
    except ImportError as exc:
        raise RuntimeError(
            "The icet backend is not installed. Install alloy-model-forge[icet] in an environment "
            "with a supported icet wheel or C++ build toolchain."
        ) from exc
    _require_periodic_reference(spec, "icet")
    token.raise_if_cancelled()
    phase = spec.phases[0]
    a = phase.lattice_parameter
    if phase.structure == "fcc":
        ideal = [a / math.sqrt(2), a, a * math.sqrt(1.5), a * math.sqrt(2), a * math.sqrt(2.5)]
        primitive = bulk(next(iter(phase.composition)), "fcc", a=a)
    elif phase.structure == "bcc":
        ideal = [a * math.sqrt(3) / 2, a, a * math.sqrt(2), a * math.sqrt(11) / 2, a * math.sqrt(3)]
        primitive = bulk(next(iter(phase.composition)), "bcc", a=a)
    elif phase.structure == "hcp":
        c = a * phase.c_over_a
        ideal = _ideal_shell_distances(phase, spec.chemistry.neighbor_shells)
        primitive = bulk(next(iter(phase.composition)), "hcp", a=a, c=c)
    else:
        raise ValueError("The icet backend currently supports fcc, bcc, and hcp phases")
    # In icet each item in ``cutoffs`` denotes a different cluster order
    # (pairs, triplets, ...), not a successive neighbor shell.  SQS matching
    # here is deliberately pair-only, with one cutoff enclosing every shell
    # requested by AlloyModelForge.
    pair_cutoff = float(ideal[spec.chemistry.neighbor_shells - 1] * 1.08)
    cluster_space = ClusterSpace(
        primitive, cutoffs=[pair_cutoff], chemical_symbols=sorted(phase.composition)
    )
    optimized = generate_sqs_from_supercells(
        cluster_space=cluster_space,
        supercells=[atoms],
        target_concentrations=phase.composition,
        n_steps=spec.chemistry.steps,
        random_seed=spec.random_seed,
    )
    atoms.set_chemical_symbols(optimized.get_chemical_symbols())
    return {
        "engine": "icet",
        "engine_version": version("icet"),
        "cutoffs": [pair_cutoff],
        "matched_pair_shells": spec.chemistry.neighbor_shells,
        "iterations": spec.chemistry.steps,
    }


def optimize_with_backend(
    atoms,
    spec: ModelSpecification,
    rng,
    progress_callback: ProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
):
    """Run the requested external engine; never silently substitute another optimizer."""
    del rng
    token = cancellation_token or CancellationToken()
    before = calculate_sro_report(atoms, spec)
    if progress_callback:
        progress_callback(ProgressEvent("sqs_optimize", 0.80, f"Running {spec.chemistry.method}"))
    if spec.chemistry.method == "sqsgenerator":
        backend_report = _run_sqsgenerator(atoms, spec, token)
    elif spec.chemistry.method == "icet":
        backend_report = _run_icet(atoms, spec, token)
    else:
        raise ValueError(f"Unsupported external SQS backend: {spec.chemistry.method}")
    token.raise_if_cancelled()
    after = calculate_sro_report(atoms, spec)
    before["engine"] = backend_report["engine"]
    after.update(backend_report)
    return before, after
