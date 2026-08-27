from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math

import numpy as np
from ase import Atoms
from ase.io import read
from scipy.spatial import cKDTree
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .config import (
    DirectionalSROConstraintSpecification, ModelSpecification, PhaseSpecification,
    RegionConstraintSpecification, SpatialSelectorSpecification,
    TripletConstraintSpecification,
)
from .progress import CancellationToken, ProgressCallback, ProgressEvent


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _ideal_shell_distances(phase: PhaseSpecification, number_of_shells: int) -> list[float]:
    """Return ideal coordination-shell radii for a phase reference cell."""
    a = phase.lattice_parameter
    if phase.structure == "fcc":
        return [a / math.sqrt(2), a, a * math.sqrt(1.5), a * math.sqrt(2), a * math.sqrt(2.5)][
            :number_of_shells
        ]
    if phase.structure == "bcc":
        return [a * math.sqrt(3) / 2, a, a * math.sqrt(2), a * math.sqrt(11) / 2, a * math.sqrt(3)][
            :number_of_shells
        ]
    if phase.structure == "hcp":
        # Enumerate the orthorhombic HCP conventional cell used by builder.py.
        # This remains correct when c/a is not the ideal close-packed value.
        basis = np.array([[0, 0, 0], [.5, .5, 0], [0, 2 / 3, .5], [.5, 1 / 6, .5]])
        cell = np.diag([a, math.sqrt(3) * a, phase.c_over_a * a])
        distances = []
        origin = basis[0] @ cell
        for shift in np.ndindex(5, 5, 5):
            translated = (basis + np.asarray(shift) - 2) @ cell
            distances.extend(np.linalg.norm(translated - origin, axis=1).tolist())
        ideal: list[float] = []
        for distance in sorted(value for value in distances if value > 1e-8):
            if not ideal or abs(distance - ideal[-1]) > max(1e-5, ideal[-1] * 1e-4):
                ideal.append(float(distance))
                if len(ideal) == number_of_shells:
                    break
        return ideal
    template = read(phase.template_path)
    basis = template.get_scaled_positions(wrap=True)
    cell = template.cell.array
    distances = []
    for left in basis:
        for right in basis:
            for shift in np.ndindex(3, 3, 3):
                delta = (right + np.asarray(shift) - 1 - left) @ cell
                distance = float(np.linalg.norm(delta))
                if distance > 1e-8:
                    distances.append(distance)
    ideal = []
    for distance in sorted(distances):
        if not ideal or abs(distance - ideal[-1]) > max(1e-5, ideal[-1] * 1e-4):
            ideal.append(distance)
    return ideal[:number_of_shells]


def _shell_edges(
    atoms: Atoms,
    indices: np.ndarray,
    phase: PhaseSpecification,
    number_of_shells: int,
) -> list[np.ndarray]:
    """Build first/second-shell edge lists for one phase under PBC."""
    if len(indices) < 2:
        return [np.empty((0, 2), dtype=np.int32) for _ in range(number_of_shells)]
    ideal = _ideal_shell_distances(phase, number_of_shells)
    upper = [
        (ideal[i] + ideal[i + 1]) / 2 if i + 1 < len(ideal) else ideal[i] * 1.12
        for i in range(len(ideal))
    ]
    positions = atoms.positions[indices]
    tree = cKDTree(positions, boxsize=atoms.cell.lengths())
    pairs = tree.query_pairs(upper[-1], output_type="ndarray")
    if len(pairs) == 0:
        return [np.empty((0, 2), dtype=np.int32) for _ in ideal]
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    box = atoms.cell.lengths()
    delta -= box * np.round(delta / box)
    distances = np.linalg.norm(delta, axis=1)
    shells: list[np.ndarray] = []
    lower = 0.0
    for limit in upper:
        selected = pairs[(distances > lower) & (distances <= limit)]
        shells.append(indices[selected].astype(np.int32, copy=False))
        lower = limit
    return shells


def _counts_and_targets(
    symbols: np.ndarray,
    edges: np.ndarray,
    elements: list[str],
    concentrations: dict[str, float],
    population_counts: dict[str, int] | None = None,
):
    keys = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
    lookup = {key: i for i, key in enumerate(keys)}
    counts = np.zeros(len(keys), dtype=np.int64)
    for left, right in edges:
        counts[lookup[_pair_key(symbols[left], symbols[right])]] += 1
    targets = np.empty(len(keys), dtype=float)
    for i, (a, b) in enumerate(keys):
        if population_counts is None:
            probability = concentrations[a] ** 2 if a == b else 2 * concentrations[a] * concentrations[b]
        else:
            population = sum(population_counts.values())
            denominator = population * (population - 1)
            if denominator <= 0:
                probability = 0.0
            elif a == b:
                probability = population_counts.get(a, 0) * (population_counts.get(a, 0) - 1) / denominator
            else:
                probability = 2 * population_counts.get(a, 0) * population_counts.get(b, 0) / denominator
        targets[i] = len(edges) * probability
    return keys, lookup, counts, targets


def _integer_target_counts(size: int, composition: dict[str, float]) -> dict[str, int]:
    """Largest-remainder projection of a fractional composition onto finite sites."""
    raw = {element: size * fraction for element, fraction in composition.items()}
    counts = {element: int(np.floor(value)) for element, value in raw.items()}
    remaining = size - sum(counts.values())
    order = sorted(raw, key=lambda element: (raw[element] - counts[element], element), reverse=True)
    for element in order[:remaining]:
        counts[element] += 1
    return counts


def _touching_shell_edges(
    atoms: Atoms, region_indices: np.ndarray, phase_indices: np.ndarray,
    phase: PhaseSpecification, number_of_shells: int,
) -> list[np.ndarray]:
    """Return unique phase bonds with at least one endpoint in the target region."""
    shells = _shell_edges(atoms, phase_indices, phase, number_of_shells)
    membership = np.zeros(len(atoms), dtype=bool)
    membership[region_indices] = True
    return [edges[membership[edges[:, 0]] | membership[edges[:, 1]]] for edges in shells]


def _crossing_shell_edges(
    atoms: Atoms, region_indices: np.ndarray, phase_indices: np.ndarray,
    phase: PhaseSpecification, number_of_shells: int,
) -> list[np.ndarray]:
    """Return phase bonds with exactly one endpoint in the target region."""
    shells = _shell_edges(atoms, phase_indices, phase, number_of_shells)
    membership = np.zeros(len(atoms), dtype=bool)
    membership[region_indices] = True
    return [edges[membership[edges[:, 0]] ^ membership[edges[:, 1]]] for edges in shells]


def _objective(counts: np.ndarray, targets: np.ndarray) -> float:
    """Dimensionless mean squared relative residual over pair/species classes."""
    if not len(counts):
        return 0.0
    scale = np.maximum(targets, 1.0)
    return float(np.mean(((counts - targets) / scale) ** 2))


def _constraint_shell_edges(
    atoms: Atoms, indices: np.ndarray, phase_indices: np.ndarray,
    phase: PhaseSpecification, number_of_shells: int, pair_scope: str,
) -> list[np.ndarray]:
    if pair_scope == "touching":
        return _touching_shell_edges(atoms, indices, phase_indices, phase, number_of_shells)
    if pair_scope == "crossing":
        return _crossing_shell_edges(atoms, indices, phase_indices, phase, number_of_shells)
    return _shell_edges(atoms, indices, phase, number_of_shells)


def _apply_target_pair_factors(
    targets: np.ndarray, keys: list[tuple[str, str]],
    constraint: RegionConstraintSpecification, shell_number: int,
) -> np.ndarray:
    """Scale finite-random pair targets by explicit, auditable factors."""
    factors = constraint.target_pair_factors.get(shell_number, {})
    if not factors:
        return targets
    result = targets.copy()
    known = {f"{a}-{b}": index for index, (a, b) in enumerate(keys)}
    for pair, factor in factors.items():
        normalized = "-".join(sorted(pair.split("-", 1)))
        if normalized not in known:
            raise ValueError(
                f"Unknown pair {pair!r} in constraint {constraint.name!r}; "
                f"available pairs are {sorted(known)}"
            )
        result[known[normalized]] *= factor
    target_sum = float(np.sum(targets))
    adjusted_sum = float(np.sum(result))
    if target_sum > 0 and adjusted_sum <= 0:
        raise ValueError(f"Pair factors for constraint {constraint.name!r} eliminate every target pair")
    if adjusted_sum > 0:
        result *= target_sum / adjusted_sum
    return result


def _directional_shell_edges(
    atoms: Atoms, source_indices: np.ndarray, target_indices: np.ndarray,
    reference_phase: PhaseSpecification, shell_number: int,
) -> np.ndarray:
    """Build oriented source-to-target edges for one geometric shell."""
    if not len(source_indices) or not len(target_indices):
        return np.empty((0, 2), dtype=np.int32)
    union = np.union1d(source_indices, target_indices)
    distances = _ideal_shell_distances(reference_phase, shell_number + 1)
    center = distances[shell_number - 1]
    lower = 0.0 if shell_number == 1 else (distances[shell_number - 2] + center) / 2
    upper = ((center + distances[shell_number]) / 2
             if len(distances) > shell_number else center * 1.12)
    positions = atoms.positions[union]
    tree = cKDTree(positions, boxsize=atoms.cell.lengths())
    pairs = tree.query_pairs(upper, output_type="ndarray")
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int32)
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    box = atoms.cell.lengths()
    delta -= box * np.round(delta / box)
    keep = np.linalg.norm(delta, axis=1) > lower
    pairs = union[pairs[keep]]
    source_mask = np.zeros(len(atoms), dtype=bool); source_mask[source_indices] = True
    target_mask = np.zeros(len(atoms), dtype=bool); target_mask[target_indices] = True
    directed = []
    for left, right in pairs:
        if source_mask[left] and target_mask[right]:
            directed.append((left, right))
        if source_mask[right] and target_mask[left]:
            directed.append((right, left))
    return np.asarray(directed, dtype=np.int32).reshape((-1, 2))


def _directional_counts_targets(
    symbols: np.ndarray, edges: np.ndarray, source_elements: list[str],
    target_elements: list[str], source_concentrations: dict[str, float],
    target_concentrations: dict[str, float], target_sro: dict[str, float],
):
    keys = [(a, b) for a in source_elements for b in target_elements]
    lookup = {key: index for index, key in enumerate(keys)}
    counts = np.zeros(len(keys), dtype=np.int64)
    for left, right in edges:
        counts[lookup[(symbols[left], symbols[right])]] += 1
    targets = np.asarray([
        len(edges) * source_concentrations[a] * target_concentrations[b]
        for a, b in keys
    ], dtype=float)
    for source in source_elements:
        row = [lookup[(source, target)] for target in target_elements]
        baseline_sum = float(np.sum(targets[row]))
        for target in target_elements:
            alpha = target_sro.get(f"{source}->{target}", 0.0)
            targets[lookup[(source, target)]] *= 1.0 - alpha
        adjusted_sum = float(np.sum(targets[row]))
        if baseline_sum > 0 and adjusted_sum <= 0:
            raise ValueError(f"Directional SRO eliminates all targets from source element {source}")
        if adjusted_sum > 0:
            targets[row] *= baseline_sum / adjusted_sum
    return keys, lookup, counts, targets


def _directional_sro_values(
    counts: np.ndarray, keys: list[tuple[str, str]],
    target_concentrations: dict[str, float],
) -> dict[str, float | None]:
    row_totals: dict[str, int] = {}
    for (source, _), count in zip(keys, counts):
        row_totals[source] = row_totals.get(source, 0) + int(count)
    values = {}
    for (source, target), count in zip(keys, counts):
        denominator = row_totals[source]
        concentration = target_concentrations.get(target, 0.0)
        values[f"{source}->{target}"] = (
            None if denominator == 0 or concentration == 0 else
            round(1.0 - (int(count) / denominator) / concentration, 8)
        )
    return values


def _selector_elements(spec: ModelSpecification, selector: SpatialSelectorSpecification) -> list[str]:
    phases = ([next(p for p in spec.phases if p.name == selector.phase)]
              if selector.phase else spec.phases)
    return sorted(set().union(*(phase.composition for phase in phases)))


def _selector_reference_composition(
    spec: ModelSpecification, selector: SpatialSelectorSpecification,
) -> dict[str, float]:
    if selector.phase is None:
        raise ValueError("Directional SRO selectors require explicit phases")
    return dict(next(phase.composition for phase in spec.phases if phase.name == selector.phase))


def calculate_directional_sro_report(atoms: Atoms, spec: ModelSpecification) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    report = {}
    for constraint in spec.chemistry.directional_sro_constraints:
        source = _region_indices(atoms, spec, constraint.source)
        target = _region_indices(atoms, spec, constraint.target)
        reference = (next(p for p in spec.phases if p.name == constraint.source.phase)
                     if constraint.source.phase else spec.phases[0])
        source_elements = _selector_elements(spec, constraint.source)
        target_elements = _selector_elements(spec, constraint.target)
        edges = _directional_shell_edges(atoms, source, target, reference, constraint.shell)
        source_c = _selector_reference_composition(spec, constraint.source)
        target_c = _selector_reference_composition(spec, constraint.target)
        keys, _, counts, targets = _directional_counts_targets(
            symbols, edges, source_elements, target_elements, source_c, target_c,
            constraint.target_sro,
        )
        report[constraint.name] = {
            "source": constraint.source.model_dump(), "target": constraint.target.model_dump(),
            "reference_source_composition": source_c,
            "reference_target_composition": target_c,
            "achieved_source_composition": _actual_concentrations(symbols, source, source_elements),
            "achieved_target_composition": _actual_concentrations(symbols, target, target_elements),
            "shell": constraint.shell, "edge_count": int(len(edges)),
            "target_sro": constraint.target_sro,
            "achieved_sro": _directional_sro_values(counts, keys, target_c),
            "objective": _objective(counts, targets),
            "pair_counts": {f"{a}->{b}": int(count) for (a, b), count in zip(keys, counts)},
            "target_pair_counts": {f"{a}->{b}": float(value) for (a, b), value in zip(keys, targets)},
        }
    return report


def _angular_triplets(
    atoms: Atoms, center_indices: np.ndarray, phase_indices: np.ndarray,
    phase: PhaseSpecification, shell: int, angle_bins: list[float],
) -> np.ndarray:
    """Enumerate center-distinguished angular triplets for one neighbor shell."""
    edges = _shell_edges(atoms, phase_indices, phase, shell)[shell - 1]
    neighbors: list[list[int]] = [[] for _ in range(len(atoms))]
    for left, right in edges:
        neighbors[left].append(int(right)); neighbors[right].append(int(left))
    box = atoms.cell.lengths()
    motifs = []
    for center in center_indices:
        local = neighbors[int(center)]
        for first_index, first in enumerate(local):
            vector_first = atoms.positions[first] - atoms.positions[center]
            vector_first -= box * np.round(vector_first / box)
            for second in local[first_index + 1:]:
                vector_second = atoms.positions[second] - atoms.positions[center]
                vector_second -= box * np.round(vector_second / box)
                cosine = np.dot(vector_first, vector_second) / (
                    np.linalg.norm(vector_first) * np.linalg.norm(vector_second)
                )
                angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
                angle_bin = int(np.searchsorted(angle_bins, angle - 1e-8, side="left"))
                if angle_bin < len(angle_bins):
                    motifs.append((int(center), first, second, angle_bin))
    return np.asarray(motifs, dtype=np.int32).reshape((-1, 4))


def _triplet_probability(
    center: str, first: str, second: str, population_counts: dict[str, int],
) -> float:
    population = sum(population_counts.values())
    if population < 3:
        return 0.0
    center_count = population_counts.get(center, 0)
    first_count = population_counts.get(first, 0) - int(first == center)
    second_count = population_counts.get(second, 0) - int(second == center)
    denominator = population * (population - 1) * (population - 2)
    if first == second:
        numerator = center_count * first_count * (first_count - 1)
    else:
        numerator = center_count * 2 * first_count * second_count
    return max(float(numerator / denominator), 0.0)


def _triplet_counts_targets(
    symbols: np.ndarray, motifs: np.ndarray, elements: list[str],
    angle_bin_count: int, population_counts: dict[str, int],
):
    keys = [
        (angle_bin, center, first, second)
        for angle_bin in range(angle_bin_count)
        for center in elements
        for first_index, first in enumerate(elements)
        for second in elements[first_index:]
    ]
    lookup = {key: index for index, key in enumerate(keys)}
    counts = np.zeros(len(keys), dtype=np.int64)
    geometry_counts = np.bincount(
        motifs[:, 3] if len(motifs) else np.asarray([], dtype=int),
        minlength=angle_bin_count,
    )
    for center, first, second, angle_bin in motifs:
        neighbor_pair = sorted((symbols[first], symbols[second]))
        counts[lookup[(int(angle_bin), symbols[center], *neighbor_pair)]] += 1
    targets = np.asarray([
        geometry_counts[angle_bin] * _triplet_probability(
            center, first, second, population_counts
        )
        for angle_bin, center, first, second in keys
    ], dtype=float)
    return keys, lookup, counts, targets


def calculate_triplet_report(atoms: Atoms, spec: ModelSpecification) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    phase_ids = atoms.arrays["phase_id"]
    report = {}
    for constraint in spec.chemistry.triplet_constraints:
        centers = _region_indices(atoms, spec, constraint.selector)
        phase_id = (next(i for i, p in enumerate(spec.phases)
                         if p.name == constraint.selector.phase)
                    if constraint.selector.phase else
                    (int(phase_ids[centers[0]]) if len(centers) else 0))
        centers = centers[phase_ids[centers] == phase_id]
        phase_indices = np.flatnonzero(phase_ids == phase_id)
        phase = spec.phases[phase_id]
        elements = sorted(phase.composition)
        motifs = _angular_triplets(
            atoms, centers, phase_indices, phase, constraint.shell,
            constraint.angle_bins,
        )
        keys, _, counts, targets = _triplet_counts_targets(
            symbols, motifs, elements, len(constraint.angle_bins),
            _population_counts(symbols, phase_indices, elements),
        )
        bin_reports = []
        for angle_bin, upper in enumerate(constraint.angle_bins):
            selected = [index for index, key in enumerate(keys) if key[0] == angle_bin]
            bin_reports.append({
                "angle_upper_deg": upper,
                "motif_count": int(np.count_nonzero(motifs[:, 3] == angle_bin)) if len(motifs) else 0,
                "objective": _objective(counts[selected], targets[selected]),
            })
        report[constraint.name] = {
            "selector": constraint.selector.model_dump(), "shell": constraint.shell,
            "angle_bins": constraint.angle_bins, "motif_count": int(len(motifs)),
            "objective": _objective(counts, targets), "bins": bin_reports,
            "target_model": "finite_fixed_composition_without_replacement_triplets",
        }
    return report


def _actual_concentrations(
    symbols: np.ndarray, indices: np.ndarray, elements: list[str]
) -> dict[str, float]:
    if not len(indices):
        return {element: 0.0 for element in elements}
    return {
        element: float(np.count_nonzero(symbols[indices] == element) / len(indices))
        for element in elements
    }


def _population_counts(symbols: np.ndarray, indices: np.ndarray, elements: list[str]) -> dict[str, int]:
    return {element: int(np.count_nonzero(symbols[indices] == element)) for element in elements}


def _region_indices(
    atoms: Atoms, spec: ModelSpecification,
    constraint: RegionConstraintSpecification | SpatialSelectorSpecification,
) -> np.ndarray:
    mask = np.ones(len(atoms), dtype=bool)
    if constraint.phase is not None:
        phase_id = next(i for i, phase in enumerate(spec.phases) if phase.name == constraint.phase)
        mask &= atoms.arrays["phase_id"] == phase_id
    if constraint.grain_ids is not None:
        mask &= np.isin(atoms.arrays["grain_id"], constraint.grain_ids)
    primary = atoms.arrays.get("primary_region")
    region = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=np.int8))
    if constraint.region == "interior":
        mask &= primary == 0 if primary is not None else region == 0
    elif constraint.region == "grain_boundary":
        mask &= primary == 1 if primary is not None else (region & 1) > 0
    elif constraint.region == "phase_boundary":
        mask &= primary == 2 if primary is not None else (region & 2) > 0
    elif constraint.region == "triple_junction":
        mask &= primary == 3 if primary is not None else (
            atoms.arrays.get("is_triple_junction", np.zeros(len(atoms), dtype=np.int8)) > 0
        )
    elif constraint.region == "mixed_boundary":
        mask &= primary == 4 if primary is not None else region == 3
    return np.flatnonzero(mask)


def calculate_region_constraint_report(atoms: Atoms, spec: ModelSpecification) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    report = {}
    for constraint in spec.chemistry.region_constraints:
        indices = _region_indices(atoms, spec, constraint)
        if constraint.phase is not None:
            phase = next(phase for phase in spec.phases if phase.name == constraint.phase)
            elements = sorted(phase.composition)
        else:
            elements = sorted(set(symbols[indices].tolist()))
        actual = _actual_concentrations(symbols, indices, elements)
        item = {
            "site_count": int(len(indices)),
            "region": constraint.region,
            "phase": constraint.phase,
            "grain_ids": constraint.grain_ids,
            "pair_scope": constraint.pair_scope,
            "target_composition": constraint.target_composition,
            "achieved_composition": actual,
            "composition_l1_error": (
                float(sum(abs(actual.get(e, 0.0) - value) for e, value in constraint.target_composition.items()))
                if constraint.target_composition else None
            ),
            "shells": [],
        }
        if constraint.target_composition:
            integer_target = _integer_target_counts(len(indices), constraint.target_composition)
            phase_pool = (np.flatnonzero(atoms.arrays["phase_id"] == next(
                i for i, p in enumerate(spec.phases) if p.name == constraint.phase
            )) if constraint.phase is not None else np.arange(len(atoms)))
            available = _population_counts(symbols, phase_pool, sorted(constraint.target_composition))
            item["integer_target_counts"] = integer_target
            item["minimum_rounding_l1_error"] = float(sum(
                abs(integer_target[e] / max(len(indices), 1) - fraction)
                for e, fraction in constraint.target_composition.items()
            ))
            item["availability_feasible"] = all(integer_target[e] <= available.get(e, 0)
                                                for e in integer_target)
        if len(indices) >= 2:
            phase_ids = np.unique(atoms.arrays["phase_id"][indices])
            if len(phase_ids) == 1:
                phase = spec.phases[int(phase_ids[0])]
                concentrations = constraint.target_composition or actual
                concentrations = {element: concentrations.get(element, 0.0) for element in elements}
                phase_indices = np.flatnonzero(atoms.arrays["phase_id"] == int(phase_ids[0]))
                shell_edges = _constraint_shell_edges(
                    atoms, indices, phase_indices, phase,
                    spec.chemistry.neighbor_shells, constraint.pair_scope,
                )
                for shell_number, edges in enumerate(shell_edges, 1):
                    valid_elements = sorted(concentrations)
                    if not len(edges) or not valid_elements:
                        continue
                    keys, _, counts, targets = _counts_and_targets(
                        symbols, edges, valid_elements, concentrations,
                        _integer_target_counts(len(indices), concentrations),
                    )
                    targets = _apply_target_pair_factors(
                        targets, keys, constraint, shell_number
                    )
                    item["shells"].append({
                        "shell": shell_number,
                        "pairs": int(len(edges)),
                        "objective": round(_objective(counts, targets), 8),
                        "target_pair_factors": constraint.target_pair_factors.get(shell_number, {}),
                        "warren_cowley": _warren_cowley(
                            symbols, edges, indices, valid_elements, concentrations
                        ),
                    })
        report[constraint.name] = item
    return report


def calculate_constraint_feasibility(atoms: Atoms, spec: ModelSpecification) -> dict:
    regions = [(constraint, _region_indices(atoms, spec, constraint))
               for constraint in spec.chemistry.region_constraints]
    overlaps = []
    for i, (left, left_indices) in enumerate(regions):
        for right, right_indices in regions[i + 1:]:
            overlap = int(np.intersect1d(left_indices, right_indices, assume_unique=True).size)
            if overlap:
                overlaps.append({"first": left.name, "second": right.name,
                                 "overlap_sites": overlap,
                                 "potential_conflict": bool(
                                     left.target_composition and right.target_composition and
                                     left.target_composition != right.target_composition)})
    reports = calculate_region_constraint_report(atoms, spec)
    joint = _joint_composition_feasibility(atoms, spec)
    return {
        "regions": {name: {key: value for key, value in report.items() if key in {
            "site_count", "integer_target_counts", "minimum_rounding_l1_error",
            "availability_feasible", "pair_scope"}}
                    for name, report in reports.items()},
        "overlaps": overlaps,
        "all_availability_feasible": all(
            report.get("availability_feasible", True) for report in reports.values()
        ),
        "joint_composition": joint,
        "all_jointly_feasible": all(item["exactly_feasible"] for item in joint.values()),
    }


def _joint_composition_feasibility(atoms: Atoms, spec: ModelSpecification) -> dict:
    """Prove exact regional composition feasibility using a compact MILP.

    Atoms with identical region-membership signatures are grouped, so the
    number of integer variables scales with region overlaps rather than atoms.
    Pair objectives are deliberately excluded from this composition proof.
    """
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    phase_ids = atoms.arrays["phase_id"]
    result: dict[str, dict] = {}
    for phase_id, phase in enumerate(spec.phases):
        phase_indices = np.flatnonzero(phase_ids == phase_id)
        constraints = [
            c for c in spec.chemistry.region_constraints
            if c.target_composition is not None and c.phase in (None, phase.name)
        ]
        if not constraints:
            result[phase.name] = {
                "status": "not_applicable", "exactly_feasible": True,
                "variable_count": 0, "constraint_count": 0,
            }
            continue
        elements = sorted(phase.composition)
        memberships = []
        for constraint in constraints:
            mask = np.zeros(len(atoms), dtype=bool)
            mask[_region_indices(atoms, spec, constraint)] = True
            memberships.append(mask[phase_indices])
        signatures: dict[tuple[bool, ...], int] = {}
        for local_index in range(len(phase_indices)):
            signature = tuple(bool(mask[local_index]) for mask in memberships)
            signatures[signature] = signatures.get(signature, 0) + 1
        groups = list(signatures.items())
        variable_count = len(groups) * len(elements)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs: list[float] = []

        def add_equation(coefficients: list[tuple[int, float]], value: int) -> None:
            row = len(rhs)
            for column, coefficient in coefficients:
                rows.append(row); cols.append(column); data.append(coefficient)
            rhs.append(float(value))

        for group_index, (_, size) in enumerate(groups):
            add_equation([(group_index * len(elements) + e, 1.0)
                          for e in range(len(elements))], size)
        phase_counts = _population_counts(symbols, phase_indices, elements)
        for element_index, element in enumerate(elements):
            add_equation([(group_index * len(elements) + element_index, 1.0)
                          for group_index in range(len(groups))], phase_counts[element])
        for constraint_index, constraint in enumerate(constraints):
            region_size = sum(size for signature, size in groups if signature[constraint_index])
            targets = _integer_target_counts(region_size, constraint.target_composition or {})
            for element_index, element in enumerate(elements):
                add_equation([
                    (group_index * len(elements) + element_index, 1.0)
                    for group_index, (signature, _) in enumerate(groups)
                    if signature[constraint_index]
                ], targets.get(element, 0))
        matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), variable_count)).tocsr()
        solved = milp(
            c=np.zeros(variable_count), integrality=np.ones(variable_count),
            bounds=Bounds(np.zeros(variable_count), np.repeat(
                [size for _, size in groups], len(elements)
            )),
            constraints=LinearConstraint(matrix, np.asarray(rhs), np.asarray(rhs)),
            options={"time_limit": 10.0},
        )
        feasible = bool(solved.success)
        result[phase.name] = {
            "status": "exactly_feasible" if feasible else "infeasible",
            "exactly_feasible": feasible,
            "solver": "scipy_milp_highs",
            "solver_status": int(solved.status),
            "membership_group_count": len(groups),
            "variable_count": variable_count,
            "constraint_count": len(rhs),
            "region_constraints": [constraint.name for constraint in constraints],
        }
    return result


def calculate_native_optimization_objective(
    atoms: Atoms, spec: ModelSpecification, with_components: bool = False,
) -> float | dict:
    """Recompute the exact weighted objective used by the native optimizer."""
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    phase_ids = atoms.arrays["phase_id"]
    boundary = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=np.int8)) > 0
    total = 0.0
    components: list[dict] = []
    for phase_id, phase in enumerate(spec.phases):
        phase_indices = np.flatnonzero(phase_ids == phase_id)
        phase_constraints = [c for c in spec.chemistry.region_constraints if c.phase in (None, phase.name)]
        base_indices = phase_indices
        if spec.chemistry.exclude_boundaries and not phase_constraints:
            interior = phase_indices[~boundary[phase_indices]]
            if len(interior) >= max(20, len(phase.composition) * 4):
                base_indices = interior
        elements = sorted(phase.composition)
        concentrations = _actual_concentrations(symbols, base_indices, elements)
        for shell_index, edges in enumerate(
            _shell_edges(atoms, base_indices, phase, spec.chemistry.neighbor_shells)
        ):
            if len(edges):
                _, _, counts, targets = _counts_and_targets(
                    symbols, edges, elements, concentrations,
                    _population_counts(symbols, base_indices, elements),
                )
                raw = _objective(counts, targets)
                weighted = raw / (shell_index + 1)
                total += weighted
                components.append({"kind": "phase_pair", "phase": phase.name,
                                   "shell": shell_index + 1, "raw": raw,
                                   "weight": 1.0 / (shell_index + 1),
                                   "weighted": weighted, "pair_count": int(len(edges))})
        for constraint in phase_constraints:
            indices = _region_indices(atoms, spec, constraint)
            indices = indices[phase_ids[indices] == phase_id]
            if len(indices) < 2:
                continue
            target = constraint.target_composition or _actual_concentrations(symbols, indices, elements)
            target = {element: target.get(element, 0.0) for element in elements}
            if constraint.pair_weight:
                for shell_index, edges in enumerate(_constraint_shell_edges(
                    atoms, indices, phase_indices, phase,
                    spec.chemistry.neighbor_shells, constraint.pair_scope,
                )):
                    if len(edges):
                        keys, _, counts, targets = _counts_and_targets(
                            symbols, edges, elements, target,
                            _integer_target_counts(len(indices), target),
                        )
                        targets = _apply_target_pair_factors(
                            targets, keys, constraint, shell_index + 1
                        )
                        raw = _objective(counts, targets)
                        weight = constraint.pair_weight / (shell_index + 1)
                        weighted = weight * raw
                        total += weighted
                        components.append({"kind": "region_pair", "phase": phase.name,
                                           "region_constraint": constraint.name,
                                           "pair_scope": constraint.pair_scope,
                                           "shell": shell_index + 1, "raw": raw,
                                           "weight": weight, "weighted": weighted,
                                           "pair_count": int(len(edges))})
            if constraint.target_composition and constraint.composition_weight:
                counts = np.asarray([np.count_nonzero(symbols[indices] == e) for e in elements])
                targets = np.asarray([len(indices) * target[e] for e in elements])
                raw = _objective(counts, targets)
                weighted = constraint.composition_weight * raw
                total += weighted
                components.append({"kind": "region_composition", "phase": phase.name,
                                   "region_constraint": constraint.name, "raw": raw,
                                   "weight": constraint.composition_weight,
                                   "weighted": weighted, "site_count": int(len(indices))})
    directional = calculate_directional_sro_report(atoms, spec)
    for constraint in spec.chemistry.directional_sro_constraints:
        item = directional[constraint.name]
        weighted = constraint.weight * item["objective"]
        total += weighted
        components.append({
            "kind": "directional_sro", "constraint": constraint.name,
            "shell": constraint.shell, "raw": item["objective"],
            "weight": constraint.weight, "weighted": weighted,
            "edge_count": item["edge_count"],
        })
    triplets = calculate_triplet_report(atoms, spec)
    for constraint in spec.chemistry.triplet_constraints:
        item = triplets[constraint.name]
        weighted = constraint.weight * item["objective"]
        total += weighted
        components.append({
            "kind": "angular_triplet", "constraint": constraint.name,
            "shell": constraint.shell, "raw": item["objective"],
            "weight": constraint.weight, "weighted": weighted,
            "motif_count": item["motif_count"],
        })
    if with_components:
        return {
            "definition": "weighted_mean_squared_relative_residual",
            "normalization": "mean_over_species_classes_with_target_floor_1",
            "total": float(total), "components": components,
        }
    return float(total)


def _warren_cowley(
    symbols: np.ndarray,
    edges: np.ndarray,
    indices: np.ndarray,
    elements: list[str],
    concentrations: dict[str, float],
) -> dict[str, float | None]:
    """Return directional Warren-Cowley alpha_ij for one neighbor shell."""
    degrees = {element: 0 for element in elements}
    directed = {(a, b): 0 for a in elements for b in elements}
    for left, right in edges:
        a, b = symbols[left], symbols[right]
        degrees[a] += 1
        degrees[b] += 1
        directed[(a, b)] += 1
        directed[(b, a)] += 1
    result: dict[str, float | None] = {}
    for a in elements:
        for b in elements:
            if degrees[a] == 0 or concentrations[b] == 0:
                result[f"{a}->{b}"] = None
            else:
                probability = directed[(a, b)] / degrees[a]
                result[f"{a}->{b}"] = round(1.0 - probability / concentrations[b], 8)
    return result


def calculate_sro_report(
    atoms: Atoms,
    spec: ModelSpecification,
    exclude_boundaries: bool | None = None,
) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    phase_ids = atoms.arrays["phase_id"]
    boundary = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=np.int8)) > 0
    exclude = spec.chemistry.exclude_boundaries if exclude_boundaries is None else exclude_boundaries
    report: dict = {"method": spec.chemistry.method, "exclude_boundaries": exclude, "phases": {}}
    for phase_id, phase in enumerate(spec.phases):
        indices = np.flatnonzero(phase_ids == phase_id)
        if exclude:
            interior = indices[~boundary[indices]]
            if len(interior) >= max(20, len(phase.composition) * 4):
                indices = interior
        shells = _shell_edges(atoms, indices, phase, spec.chemistry.neighbor_shells)
        concentrations = _actual_concentrations(symbols, indices, sorted(phase.composition))
        phase_report = []
        for shell_index, edges in enumerate(shells, 1):
            elements = sorted(phase.composition)
            # Keep the conventional thermodynamic-limit SRO report for direct
            # comparison with external SQS packages. The native optimization
            # objective separately uses finite-population targets.
            keys, _, counts, targets = _counts_and_targets(symbols, edges, elements, concentrations)
            deviations = {
                f"{a}-{b}": round(float((count - target) / max(target, 1.0)), 6)
                for (a, b), count, target in zip(keys, counts, targets)
            }
            phase_report.append(
                {
                    "shell": shell_index,
                    "pairs": int(len(edges)),
                    "objective": round(_objective(counts, targets), 8),
                    "normalized_rmse": round(
                        float(np.sqrt(np.mean(((counts - targets) / np.maximum(targets, 1.0)) ** 2))), 8
                    ),
                    "relative_pair_deviation": deviations,
                    "warren_cowley": _warren_cowley(
                        symbols, edges, indices, elements, concentrations
                    ),
                }
            )
        report["phases"][phase.name] = phase_report
    report["total_objective"] = round(
        sum(s["objective"] for phase_report in report["phases"].values() for s in phase_report), 8
    )
    report["target_model"] = "thermodynamic_limit_random_alloy"
    return report


@dataclass
class _ShellState:
    edges: np.ndarray
    adjacency: list[list[int]]
    keys: list[tuple[str, str]]
    lookup: dict[tuple[str, str], int]
    counts: np.ndarray
    targets: np.ndarray
    weight: float
    directed: bool = False

    def pair_key(self, left: str, right: str) -> tuple[str, str]:
        return (left, right) if self.directed else _pair_key(left, right)


@dataclass
class _CompositionState:
    membership: np.ndarray
    elements: list[str]
    lookup: dict[str, int]
    counts: np.ndarray
    targets: np.ndarray
    weight: float

    def objective(self, counts: np.ndarray | None = None) -> float:
        values = self.counts if counts is None else counts
        return _objective(values, self.targets)


@dataclass
class _TripletState:
    motifs: np.ndarray
    adjacency: list[list[int]]
    lookup: dict[tuple[int, str, str, str], int]
    counts: np.ndarray
    targets: np.ndarray
    weight: float

    def motif_key(self, symbols: np.ndarray, motif: np.ndarray) -> tuple[int, str, str, str]:
        center, first, second, angle_bin = motif
        neighbors = sorted((symbols[first], symbols[second]))
        return int(angle_bin), symbols[center], neighbors[0], neighbors[1]


def _optimize_native_sqs_once(
    atoms: Atoms,
    spec: ModelSpecification,
    rng: np.random.Generator,
    progress_callback: ProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[dict, dict]:
    """Pair-correlation SQS optimizer for single crystals and polycrystals.

    It preserves the exact composition of each phase and performs simulated
    annealing swaps to match ideal random pair correlations in requested
    neighbor shells. Grain-boundary sites may be excluded from the objective.
    """
    token = cancellation_token or CancellationToken()
    before = calculate_sro_report(atoms, spec)
    before["region_constraints"] = calculate_region_constraint_report(atoms, spec)
    before["directional_sro_constraints"] = calculate_directional_sro_report(atoms, spec)
    before["triplet_constraints"] = calculate_triplet_report(atoms, spec)
    feasibility = calculate_constraint_feasibility(atoms, spec)
    before["constraint_feasibility"] = feasibility
    if not feasibility["all_availability_feasible"]:
        raise ValueError("One or more region composition targets require more atoms than are available in the phase")
    if (spec.chemistry.require_exact_region_composition and
            not feasibility["all_jointly_feasible"]):
        raise ValueError("Regional composition targets are not jointly feasible at exact integer counts")
    before["optimization_objective"] = round(
        calculate_native_optimization_objective(atoms, spec), 8
    )
    before["optimization_objective_breakdown"] = calculate_native_optimization_objective(
        atoms, spec, with_components=True
    )
    before["optimization_target_model"] = "finite_fixed_composition_without_replacement"
    before["optimizer"] = "alloy_model_forge_native_sqs_v4"
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    phase_ids = atoms.arrays["phase_id"]
    boundary = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=np.int8)) > 0
    total_steps = spec.chemistry.steps
    completed_steps = 0
    history: list[dict] = []
    phase_summaries: list[dict] = []
    for phase_id, phase in enumerate(spec.phases):
        selectable = np.flatnonzero(phase_ids == phase_id)
        objective_indices = selectable
        phase_constraints = [
            constraint for constraint in spec.chemistry.region_constraints
            if constraint.phase in (None, phase.name)
        ]
        if spec.chemistry.exclude_boundaries and not phase_constraints:
            interior = selectable[~boundary[selectable]]
            if len(interior) >= max(20, len(phase.composition) * 4):
                selectable = interior
                objective_indices = interior
        if len(selectable) < 2 or len(set(symbols[selectable])) < 2:
            continue
        shell_edges = _shell_edges(atoms, objective_indices, phase, spec.chemistry.neighbor_shells)
        states: list[_ShellState] = []
        triplet_states: list[_TripletState] = []
        composition_states: list[_CompositionState] = []
        elements = sorted(phase.composition)
        concentrations = _actual_concentrations(symbols, objective_indices, elements)
        for shell_index, edges in enumerate(shell_edges):
            if not len(edges):
                continue
            keys, lookup, counts, targets = _counts_and_targets(
                symbols, edges, elements, concentrations,
                _population_counts(symbols, objective_indices, elements),
            )
            adjacency: list[list[int]] = [[] for _ in range(len(atoms))]
            for edge_index, (left, right) in enumerate(edges):
                adjacency[left].append(edge_index)
                adjacency[right].append(edge_index)
            states.append(_ShellState(edges, adjacency, keys, lookup, counts, targets, 1.0 / (shell_index + 1)))
        for constraint in phase_constraints:
            indices = _region_indices(atoms, spec, constraint)
            indices = indices[phase_ids[indices] == phase_id]
            if len(indices) < 2:
                continue
            concentrations = constraint.target_composition or _actual_concentrations(
                symbols, indices, elements
            )
            if not set(concentrations).issubset(elements):
                raise ValueError(
                    f"Region constraint {constraint.name!r} uses elements outside phase {phase.name}"
                )
            region_shells = _constraint_shell_edges(
                atoms, indices, selectable, phase,
                spec.chemistry.neighbor_shells, constraint.pair_scope,
            )
            for shell_index, edges in enumerate(region_shells):
                if not len(edges) or constraint.pair_weight == 0:
                    continue
                keys, lookup, counts, targets = _counts_and_targets(
                    symbols, edges, elements, {e: concentrations.get(e, 0.0) for e in elements},
                    _integer_target_counts(len(indices), {e: concentrations.get(e, 0.0) for e in elements}),
                )
                targets = _apply_target_pair_factors(
                    targets, keys, constraint, shell_index + 1
                )
                adjacency = [[] for _ in range(len(atoms))]
                for edge_index, (left, right) in enumerate(edges):
                    adjacency[left].append(edge_index)
                    adjacency[right].append(edge_index)
                states.append(_ShellState(
                    edges, adjacency, keys, lookup, counts, targets,
                    constraint.pair_weight / (shell_index + 1),
                ))
            if constraint.target_composition and constraint.composition_weight > 0:
                membership = np.zeros(len(atoms), dtype=bool)
                membership[indices] = True
                counts = np.asarray([np.count_nonzero(symbols[indices] == e) for e in elements], dtype=np.int64)
                targets = np.asarray([len(indices) * constraint.target_composition.get(e, 0.0) for e in elements])
                composition_states.append(_CompositionState(
                    membership, elements, {e: i for i, e in enumerate(elements)},
                    counts, targets, constraint.composition_weight,
                ))
        for constraint in spec.chemistry.directional_sro_constraints:
            involved_phases = {constraint.source.phase, constraint.target.phase}
            if None not in involved_phases and phase.name not in involved_phases:
                continue
            source = _region_indices(atoms, spec, constraint.source)
            target = _region_indices(atoms, spec, constraint.target)
            reference = (next(p for p in spec.phases if p.name == constraint.source.phase)
                         if constraint.source.phase else phase)
            edges = _directional_shell_edges(
                atoms, source, target, reference, constraint.shell
            )
            if not len(edges):
                continue
            source_elements = _selector_elements(spec, constraint.source)
            target_elements = _selector_elements(spec, constraint.target)
            keys, lookup, counts, targets = _directional_counts_targets(
                symbols, edges, source_elements, target_elements,
                _selector_reference_composition(spec, constraint.source),
                _selector_reference_composition(spec, constraint.target),
                constraint.target_sro,
            )
            adjacency = [[] for _ in range(len(atoms))]
            for edge_index, (left, right) in enumerate(edges):
                adjacency[left].append(edge_index); adjacency[right].append(edge_index)
            states.append(_ShellState(
                edges, adjacency, keys, lookup, counts, targets,
                constraint.weight, directed=True,
            ))
        for constraint in spec.chemistry.triplet_constraints:
            if constraint.selector.phase not in (None, phase.name):
                continue
            centers = _region_indices(atoms, spec, constraint.selector)
            centers = centers[phase_ids[centers] == phase_id]
            motifs = _angular_triplets(
                atoms, centers, selectable, phase, constraint.shell,
                constraint.angle_bins,
            )
            if not len(motifs):
                continue
            _, lookup, counts, targets = _triplet_counts_targets(
                symbols, motifs, elements, len(constraint.angle_bins),
                _population_counts(symbols, selectable, elements),
            )
            adjacency = [[] for _ in range(len(atoms))]
            for motif_index, motif in enumerate(motifs):
                for atom_index in motif[:3]:
                    adjacency[int(atom_index)].append(motif_index)
            triplet_states.append(_TripletState(
                motifs, adjacency, lookup, counts, targets, constraint.weight
            ))
        if not states and not composition_states and not triplet_states:
            continue
        current = sum(state.weight * _objective(state.counts, state.targets) for state in states)
        current += sum(state.weight * state.objective() for state in composition_states)
        current += sum(state.weight * _objective(state.counts, state.targets)
                       for state in triplet_states)
        initial_objective = current
        best_objective = current
        best_step = 0
        best_symbols = symbols[selectable].copy()
        attempted = 0
        accepted = 0
        block_attempted = 0
        block_accepted = 0
        temperature_factor = 1.0
        last_improvement_attempt = 0
        stop_reason = "step_budget"
        phase_steps = max(100, total_steps // len(spec.phases))
        initial_temperature = max(spec.chemistry.random_temperature * current, 1e-6)
        history.append({
            "phase": phase.name, "phase_id": phase_id, "step": 0,
            "temperature": initial_temperature, "current_objective": current,
            "best_objective": best_objective, "attempted": attempted,
            "accepted": accepted, "acceptance_rate": 0.0,
            "accepted_move": None, "delta_objective": None,
        })
        for step in range(phase_steps):
            if step % 250 == 0:
                token.raise_if_cancelled()
            left = right = None
            # Region-directed proposals exchange an overrepresented species
            # inside a constrained region with an underrepresented one outside.
            if composition_states and rng.random() < 0.5:
                state = composition_states[int(rng.integers(len(composition_states)))]
                residual = state.counts - state.targets
                excess = [state.elements[i] for i in np.flatnonzero(residual > 0.5)]
                deficit = [state.elements[i] for i in np.flatnonzero(residual < -0.5)]
                if excess and deficit:
                    inside = selectable[state.membership[selectable] & np.isin(symbols[selectable], excess)]
                    outside = selectable[~state.membership[selectable] & np.isin(symbols[selectable], deficit)]
                    if len(inside) and len(outside):
                        left, right = int(rng.choice(inside)), int(rng.choice(outside))
            if left is None:
                left = int(rng.choice(selectable))
                different = selectable[symbols[selectable] != symbols[left]]
                if not len(different):
                    continue
                right = int(rng.choice(different))
            attempted += 1
            block_attempted += 1
            changes = []
            delta_objective = 0.0
            for state in states:
                affected = set(state.adjacency[left]) | set(state.adjacency[right])
                old_indices: list[int] = []
                new_indices: list[int] = []
                local_delta = np.zeros_like(state.counts)
                for edge_index in affected:
                    a, b = state.edges[edge_index]
                    old_idx = state.lookup[state.pair_key(symbols[a], symbols[b])]
                    sa = symbols[right] if a == left else symbols[left] if a == right else symbols[a]
                    sb = symbols[right] if b == left else symbols[left] if b == right else symbols[b]
                    new_idx = state.lookup[state.pair_key(sa, sb)]
                    if old_idx != new_idx:
                        local_delta[old_idx] -= 1
                        local_delta[new_idx] += 1
                    old_indices.append(old_idx); new_indices.append(new_idx)
                old_value = _objective(state.counts, state.targets)
                new_value = _objective(state.counts + local_delta, state.targets)
                delta_objective += state.weight * (new_value - old_value)
                changes.append((state, local_delta))
            triplet_changes = []
            for state in triplet_states:
                affected = set(state.adjacency[left]) | set(state.adjacency[right])
                local_delta = np.zeros_like(state.counts)
                swapped = symbols.copy()
                swapped[left], swapped[right] = swapped[right], swapped[left]
                for motif_index in affected:
                    motif = state.motifs[motif_index]
                    old_idx = state.lookup[state.motif_key(symbols, motif)]
                    new_idx = state.lookup[state.motif_key(swapped, motif)]
                    if old_idx != new_idx:
                        local_delta[old_idx] -= 1; local_delta[new_idx] += 1
                delta_objective += state.weight * (
                    _objective(state.counts + local_delta, state.targets)
                    - _objective(state.counts, state.targets)
                )
                triplet_changes.append((state, local_delta))
            composition_changes = []
            for state in composition_states:
                local_delta = np.zeros_like(state.counts)
                if state.membership[left] != state.membership[right]:
                    inside = left if state.membership[left] else right
                    outside = right if inside == left else left
                    local_delta[state.lookup[symbols[inside]]] -= 1
                    local_delta[state.lookup[symbols[outside]]] += 1
                delta_objective += state.weight * (
                    state.objective(state.counts + local_delta) - state.objective()
                )
                composition_changes.append((state, local_delta))
            temperature = (initial_temperature * temperature_factor *
                           max(1e-4, (1 - step / phase_steps) ** 2))
            accepted_move = (
                delta_objective <= 0
                or rng.random() < math.exp(-delta_objective / temperature)
            )
            if accepted_move:
                symbols[left], symbols[right] = symbols[right], symbols[left]
                for state, local_delta in changes:
                    state.counts += local_delta
                for state, local_delta in composition_changes:
                    state.counts += local_delta
                for state, local_delta in triplet_changes:
                    state.counts += local_delta
                current += delta_objective
                accepted += 1
                block_accepted += 1
                if current < best_objective - 1e-12:
                    best_objective = current
                    best_step = step + 1
                    best_symbols = symbols[selectable].copy()
                    last_improvement_attempt = attempted
            completed_steps += 1
            if (
                (step + 1) % spec.chemistry.history_interval == 0
                or step + 1 == phase_steps
            ):
                history.append({
                    "phase": phase.name, "phase_id": phase_id, "step": step + 1,
                    "temperature": temperature, "current_objective": current,
                    "best_objective": best_objective, "attempted": attempted,
                    "accepted": accepted,
                    "acceptance_rate": accepted / attempted if attempted else 0.0,
                    "accepted_move": accepted_move,
                    "delta_objective": delta_objective,
                })
            if progress_callback and step % max(100, phase_steps // 100) == 0:
                progress_callback(
                    ProgressEvent(
                        "sqs_optimize",
                        0.76 + 0.09 * min(completed_steps / total_steps, 1.0),
                        f"Optimizing SQS correlations: phase {phase_id + 1}, step {step}/{phase_steps}",
                    )
                )
            if spec.chemistry.adaptive_temperature and block_attempted >= 100:
                block_rate = block_accepted / block_attempted
                target_rate = spec.chemistry.target_acceptance_rate
                temperature_factor *= float(np.clip(
                    np.exp(0.5 * (target_rate - block_rate)), 0.75, 1.35
                ))
                temperature_factor = float(np.clip(temperature_factor, 1e-3, 1e3))
                block_attempted = block_accepted = 0
            if best_objective <= spec.chemistry.objective_tolerance:
                stop_reason = "objective_tolerance"
                break
            patience = spec.chemistry.early_stop_patience
            if patience is not None and attempted - last_improvement_attempt >= patience:
                stop_reason = "no_improvement_patience"
                break
        terminal_objective = current
        symbols[selectable] = best_symbols
        phase_summaries.append({
            "phase": phase.name, "phase_id": phase_id,
            "initial_objective": initial_objective,
            "terminal_objective": terminal_objective,
            "best_objective": best_objective, "best_step": best_step,
            "attempted": attempted, "accepted": accepted,
            "acceptance_rate": accepted / attempted if attempted else 0.0,
            "restored_historical_best": best_objective < terminal_objective - 1e-12,
            "stop_reason": stop_reason,
            "adaptive_temperature": spec.chemistry.adaptive_temperature,
            "final_temperature_factor": temperature_factor,
        })
    atoms.set_chemical_symbols(symbols.tolist())
    after = calculate_sro_report(atoms, spec)
    after["region_constraints"] = calculate_region_constraint_report(atoms, spec)
    after["directional_sro_constraints"] = calculate_directional_sro_report(atoms, spec)
    after["triplet_constraints"] = calculate_triplet_report(atoms, spec)
    after["constraint_feasibility"] = calculate_constraint_feasibility(atoms, spec)
    after["optimization_objective"] = round(
        calculate_native_optimization_objective(atoms, spec), 8
    )
    after["optimization_objective_breakdown"] = calculate_native_optimization_objective(
        atoms, spec, with_components=True
    )
    after["optimization_target_model"] = "finite_fixed_composition_without_replacement"
    after["optimizer"] = "alloy_model_forge_native_sqs_v4"
    after["optimization_history"] = history
    after["optimization_summary"] = {
        "history_interval": spec.chemistry.history_interval,
        "phases": phase_summaries,
        "returned_state": "historical_best",
    }
    return before, after


def calculate_exact_native_optimum(
    atoms: Atoms, spec: ModelSpecification, max_configurations: int = 1_000_000,
) -> dict:
    """Exhaustively enumerate a small single-phase composition as a gold standard."""
    if len(spec.phases) != 1:
        raise ValueError("Exact native-SQS benchmarking currently requires one phase")
    original = atoms.get_chemical_symbols()
    counts = Counter(original)
    configurations = math.factorial(len(atoms))
    for count in counts.values():
        configurations //= math.factorial(count)
    if configurations > max_configurations:
        raise ValueError(
            f"Exact benchmark requires {configurations} configurations, exceeding "
            f"the limit {max_configurations}"
        )
    elements = sorted(counts)
    working = [""] * len(atoms)
    best = math.inf
    best_symbols: list[str] | None = None
    optimum_degeneracy = 0
    evaluated = 0

    def visit(position: int) -> None:
        nonlocal best, best_symbols, optimum_degeneracy, evaluated
        if position == len(working):
            atoms.set_chemical_symbols(working)
            value = float(calculate_native_optimization_objective(atoms, spec))
            evaluated += 1
            if value < best - 1e-14:
                best = value
                best_symbols = working.copy()
                optimum_degeneracy = 1
            elif abs(value - best) <= 1e-14:
                optimum_degeneracy += 1
            return
        for element in elements:
            if counts[element] == 0:
                continue
            counts[element] -= 1
            working[position] = element
            visit(position + 1)
            counts[element] += 1

    try:
        visit(0)
    finally:
        atoms.set_chemical_symbols(original)
    return {
        "status": "completed", "method": "exhaustive_multiset_enumeration",
        "site_count": len(atoms), "configuration_count": configurations,
        "evaluated": evaluated, "optimal_objective": best,
        "optimum_degeneracy": optimum_degeneracy,
        "one_optimal_assignment": best_symbols,
    }


def _attach_exact_benchmark(
    initial: Atoms, spec: ModelSpecification, after: dict,
) -> None:
    limit = spec.chemistry.exact_benchmark_max_sites
    if limit is None:
        return
    if len(initial) > limit:
        after["exact_benchmark"] = {
            "status": "skipped", "reason": "site_limit",
            "site_count": len(initial), "max_sites": limit,
        }
        return
    exact = calculate_exact_native_optimum(
        initial.copy(), spec, spec.chemistry.exact_benchmark_max_configurations
    )
    achieved = float(after["optimization_objective"])
    exact["achieved_objective"] = achieved
    exact["absolute_optimality_gap"] = achieved - exact["optimal_objective"]
    exact["relative_optimality_gap"] = (
        (achieved - exact["optimal_objective"]) / max(abs(exact["optimal_objective"]), 1e-15)
    )
    after["exact_benchmark"] = exact


def optimize_native_sqs(
    atoms: Atoms,
    spec: ModelSpecification,
    rng: np.random.Generator,
    progress_callback: ProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[dict, dict]:
    """Run one or more independent native-SQS anneals and return the best."""
    if spec.chemistry.restarts == 1:
        initial = atoms.copy()
        before, after = _optimize_native_sqs_once(
            atoms, spec, rng, progress_callback, cancellation_token
        )
        after["optimization_summary"]["restarts"] = {
            "count": 1, "selected": 0,
            "objectives": [after["optimization_objective"]],
            "mean": after["optimization_objective"], "std": 0.0,
        }
        _attach_exact_benchmark(initial, spec, after)
        return before, after

    token = cancellation_token or CancellationToken()
    initial = atoms.copy()
    runs: list[tuple[int, Atoms, dict, dict]] = []
    for restart in range(spec.chemistry.restarts):
        token.raise_if_cancelled()
        derived_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        candidate = initial.copy()

        def restart_progress(event: ProgressEvent, index: int = restart) -> None:
            if progress_callback:
                progress_callback(ProgressEvent(
                    event.stage, event.fraction,
                    f"Restart {index + 1}/{spec.chemistry.restarts}: {event.message}",
                ))

        before, after = _optimize_native_sqs_once(
            candidate, spec, np.random.default_rng(derived_seed),
            restart_progress, token,
        )
        runs.append((derived_seed, candidate, before, after))
    selected = min(range(len(runs)), key=lambda index: runs[index][3]["optimization_objective"])
    _, winner, before, after = runs[selected]
    atoms.set_chemical_symbols(winner.get_chemical_symbols())
    objectives = np.asarray([run[3]["optimization_objective"] for run in runs], dtype=float)
    after["optimization_summary"]["restarts"] = {
        "count": len(runs), "selected": selected,
        "derived_seeds": [run[0] for run in runs],
        "objectives": objectives.tolist(),
        "minimum": float(np.min(objectives)),
        "mean": float(np.mean(objectives)),
        "std": float(np.std(objectives)),
        "near_best_fraction": float(np.mean(
            objectives <= np.min(objectives) + max(1e-12, 0.01 * abs(np.min(objectives)))
        )),
    }
    _attach_exact_benchmark(initial, spec, after)
    return before, after
