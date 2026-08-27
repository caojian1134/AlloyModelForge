from __future__ import annotations

import json
import math
import csv
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import ModelSpecification, PhaseSpecification
from .orientation import (
    minimum_misorientation_deg, random_rotation_matrix, rotation_angle_deg,
    validate_rotation_matrix,
)
from .progress import CancellationToken, ProgressCallback, ProgressEvent


def _emit(callback: ProgressCallback | None, stage: str, fraction: float, message: str):
    if callback:
        callback(ProgressEvent(stage, float(fraction), message))


def _basis(structure: str) -> np.ndarray:
    if structure == "fcc":
        return np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    if structure == "bcc":
        return np.array([[0, 0, 0], [.5, .5, .5]])
    if structure == "hcp":
        # Orthorhombic HCP conventional cell.
        return np.array([[0, 0, 0], [.5, .5, 0], [0, 2 / 3, .5], [.5, 1 / 6, .5]])
    raise ValueError(f"Unsupported structure: {structure}")


def _unit_cell(phase: PhaseSpecification) -> tuple[np.ndarray, np.ndarray]:
    if phase.structure in {"fcc", "bcc"}:
        return np.eye(3) * phase.lattice_parameter, _basis(phase.structure)
    if phase.structure == "hcp":
        cell = np.diag([
            phase.lattice_parameter,
            math.sqrt(3) * phase.lattice_parameter,
            phase.c_over_a * phase.lattice_parameter,
        ])
        return cell, _basis("hcp")
    template = read(phase.template_path)
    if len(template) == 0 or abs(np.linalg.det(template.cell.array)) < 1e-10:
        raise ValueError(f"Custom template {phase.template_path!r} has no atoms or valid cell")
    return np.asarray(template.cell.array, dtype=float), template.get_scaled_positions(wrap=True)


def _atomic_volume(phase: PhaseSpecification) -> float:
    cell, basis = _unit_cell(phase)
    return abs(float(np.linalg.det(cell))) / len(basis)


def _nearest_ideal_distance(phase: PhaseSpecification) -> float:
    cell, basis = _unit_cell(phase)
    points = []
    for shift in np.ndindex(3, 3, 3):
        points.append((basis + np.asarray(shift) - 1) @ cell)
    points = np.concatenate(points)
    origin = basis[0] @ cell
    distances = np.linalg.norm(points - origin, axis=1)
    return float(np.min(distances[distances > 1e-8]))


def _first_shell_cutoff(phase: PhaseSpecification) -> float:
    """Choose a cutoff between the first and second coordination shells."""
    a = phase.lattice_parameter
    if phase.structure == "fcc":
        first, second = a / math.sqrt(2), a
    elif phase.structure == "bcc":
        first, second = a * math.sqrt(3) / 2, a
    elif phase.structure == "hcp":
        first, second = a, a * phase.c_over_a
    else:
        return 1.20 * _nearest_ideal_distance(phase)
    return (first + second) / 2


def _crystallographic_rotation(x_direction, z_direction) -> np.ndarray:
    x_axis = np.asarray(x_direction, dtype=float)
    z_axis = np.asarray(z_direction, dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def _commensurability_report(
    spec: ModelSpecification,
    orientations: list[np.ndarray],
    grain_phases: np.ndarray,
    box: np.ndarray | None = None,
) -> dict[str, Any]:
    box = np.asarray(spec.box.lengths) if box is None else np.asarray(box)
    periodic = np.asarray(spec.box.periodic)
    grains = []
    for grain_id, rotation in enumerate(orientations):
        phase_id = int(grain_phases[grain_id])
        phase = spec.phases[phase_id]
        unit_cell, _ = _unit_cell(phase)
        rotated_cell = unit_cell @ rotation.T
        residuals = []
        for axis in np.flatnonzero(periodic):
            translation = np.zeros(3)
            translation[axis] = box[axis]
            coefficients = translation @ np.linalg.inv(rotated_cell)
            residuals.append(float(np.max(np.abs(coefficients - np.rint(coefficients)))))
        grains.append({"grain_id": grain_id, "maximum_integer_residual": max(residuals, default=0.0)})
    return {
        "strictly_commensurate": all(g["maximum_integer_residual"] < 1e-8 for g in grains),
        "grains": grains,
        "note": "Strict cell commensurability is primarily meaningful for a single crystal",
    }


def _snap_nearly_commensurate_box(
    spec: ModelSpecification, orientation: np.ndarray, box: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Snap small decimal-rounding errors in a periodic single-crystal box."""
    if spec.grains.number != 1 or len(spec.phases) != 1:
        return box, {"applied": False, "adjustments": []}
    unit_cell, _ = _unit_cell(spec.phases[0])
    rotated_cell = unit_cell @ orientation.T
    inverse = np.linalg.inv(rotated_cell)
    snapped = box.copy()
    adjustments = []
    for axis, is_periodic in enumerate(spec.box.periodic):
        if not is_periodic:
            continue
        translation = np.zeros(3)
        translation[axis] = box[axis]
        coefficients = translation @ inverse
        integers = np.rint(coefficients)
        reconstructed = integers @ rotated_cell
        residual = np.linalg.norm(reconstructed - translation)
        if residual <= max(1e-6, box[axis] * 1e-6):
            snapped[axis] = reconstructed[axis]
            if abs(snapped[axis] - box[axis]) > 1e-12:
                adjustments.append({
                    "axis": int(axis), "requested": float(box[axis]),
                    "actual": float(snapped[axis]), "difference": float(snapped[axis] - box[axis]),
                })
    return snapped, {"applied": bool(adjustments), "adjustments": adjustments}


def _nearest_seed(
    points: np.ndarray,
    seeds: np.ndarray,
    box: np.ndarray,
    periodic: np.ndarray | None = None,
    chunk=20_000,
):
    periodic = np.ones(3, dtype=bool) if periodic is None else np.asarray(periodic, dtype=bool)
    result = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        delta = block[:, None, :] - seeds[None, :, :]
        delta[:, :, periodic] -= box[periodic] * np.round(
            delta[:, :, periodic] / box[periodic]
        )
        result[start:start + len(block)] = np.argmin(np.einsum("bij,bij->bi", delta, delta), axis=1)
    return result


def _integer_counts(total: int, composition: dict[str, float]) -> dict[str, int]:
    elements = list(composition)
    raw = np.array([composition[e] * total for e in elements])
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:remainder]] += 1
    return dict(zip(elements, base.tolist()))


def _estimate_grain_volumes(
    seeds: np.ndarray, box: np.ndarray, periodic: np.ndarray, resolution: int = 28
) -> np.ndarray:
    """Deterministically estimate periodic Voronoi-cell volumes on a regular grid."""
    axes = [(np.arange(resolution) + 0.5) * length / resolution for length in box]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    owners = _nearest_seed(points, seeds, box, periodic)
    counts = np.bincount(owners, minlength=len(seeds)).astype(float)
    return counts / counts.sum()


def _phase_by_grain(
    spec: ModelSpecification, grain_volumes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Assign whole Voronoi grains to phases while matching target volume fractions."""
    inclusion_names = {inclusion.phase for inclusion in spec.inclusions}
    matrix_phase_ids = [
        phase_id for phase_id, phase in enumerate(spec.phases)
        if phase.name not in inclusion_names
    ]
    phase_count = len(matrix_phase_ids)
    targets = np.asarray([spec.phases[i].fraction for i in matrix_phase_ids], dtype=float)
    targets /= targets.sum()
    values = np.full(spec.grains.number, -1, dtype=np.int16)
    assigned = np.zeros(phase_count, dtype=float)
    # Start each phase with one small grain; this guarantees that no requested
    # phase can silently disappear when integer grain allocation is coarse.
    grain_order = np.argsort(grain_volumes)
    phase_order = np.argsort(targets)
    for grain_id, phase_id in zip(grain_order[:phase_count], phase_order):
        values[grain_id] = matrix_phase_ids[phase_id]
        assigned[phase_id] += grain_volumes[grain_id]
    remaining = grain_order[phase_count:]
    # Largest remaining cells first gives the greedy fit room to correct with
    # progressively smaller cells.
    remaining = remaining[np.argsort(-grain_volumes[remaining])]
    for grain_id in remaining:
        scores = []
        for phase_id in range(phase_count):
            candidate = assigned.copy()
            candidate[phase_id] += grain_volumes[grain_id]
            scores.append(float(np.sum((candidate - targets) ** 2 / np.maximum(targets, 1e-12))))
        best = np.flatnonzero(np.isclose(scores, np.min(scores)))
        phase_id = int(rng.choice(best))
        values[grain_id] = matrix_phase_ids[phase_id]
        assigned[phase_id] += grain_volumes[grain_id]
    return values


def _generate_oriented_candidates(
    seed: np.ndarray,
    rotation: np.ndarray,
    phase: PhaseSpecification,
    box: np.ndarray,
    periodic: np.ndarray,
) -> np.ndarray:
    # Each periodic seed image owns the part of a boundary-crossing grain next
    # to it. Building the same oriented lattice around those images makes the
    # Voronoi microstructure continuous across periodic faces.
    unit_cell, basis = _unit_cell(phase)
    shortest_vector = float(np.min(np.linalg.norm(unit_cell, axis=1)))
    radius = float(np.linalg.norm(box) / 2 + shortest_vector)
    extent = int(math.ceil(radius / shortest_vector)) + 2
    grid = np.indices((2 * extent + 1,) * 3, dtype=np.int32).reshape(3, -1).T - extent
    fractional = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    relative = (fractional @ unit_cell) @ rotation.T
    inside = np.all(np.abs(relative) <= radius + shortest_vector, axis=1)
    relative = relative[inside]
    shifts = [(-1, 0, 1) if periodic[axis] else (0,) for axis in range(3)]
    images = np.stack(np.meshgrid(*shifts, indexing="ij"), axis=-1).reshape(-1, 3)
    chunks = []
    for image in images:
        positions = seed + image * box + relative
        in_primary_cell = np.all((positions >= 0.0) & (positions < box), axis=1)
        # Keep only the points for which this is the nearest periodic image of
        # the seed. Otherwise several translated, non-commensurate lattices
        # would be superimposed in the same Voronoi cell.
        nearest_image = np.zeros_like(positions, dtype=np.int8)
        nearest_image[:, periodic] = np.rint(
            (positions[:, periodic] - seed[periodic]) / box[periodic]
        ).astype(np.int8)
        belongs_to_image = np.all(nearest_image == image, axis=1)
        selected = in_primary_cell & belongs_to_image
        if np.any(selected):
            chunks.append(positions[selected])
    positions = np.concatenate(chunks) if chunks else np.empty((0, 3))
    # Remove exact duplicates within the candidate lattice.
    keys = np.round(positions, decimals=8)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return positions[np.sort(unique)]


def _inclusion_rotation(inclusion) -> np.ndarray:
    if inclusion.orientation_matrix is not None:
        rotation = np.asarray(inclusion.orientation_matrix, dtype=float)
    elif inclusion.euler_angles_deg is not None:
        rotation = Rotation.from_euler("ZXZ", inclusion.euler_angles_deg, degrees=True).as_matrix()
    else:
        rotation = np.eye(3)
    validate_rotation_matrix(rotation)
    return rotation


def _inclusion_mask(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    shape: str,
    rotation: np.ndarray,
    box: np.ndarray,
    periodic_axes: np.ndarray,
    use_periodic_images: bool,
) -> np.ndarray:
    delta = points - center
    wrap = periodic_axes if use_periodic_images else np.zeros(3, dtype=bool)
    delta[:, wrap] -= box[wrap] * np.round(delta[:, wrap] / box[wrap])
    local = delta @ rotation
    if shape == "sphere":
        return np.linalg.norm(local, axis=1) <= size[0]
    if shape == "ellipsoid":
        return np.sum((local / size) ** 2, axis=1) <= 1.0
    if shape == "box":
        return np.all(np.abs(local) <= size, axis=1)
    if shape == "cylinder":
        return (np.linalg.norm(local[:, :2], axis=1) <= size[0]) & (
            np.abs(local[:, 2]) <= size[2]
        )
    if shape == "slab":
        return np.abs(local[:, 2]) <= size[2]
    raise ValueError(f"Unsupported inclusion shape: {shape}")


def _apply_inclusion_geometry(
    positions: np.ndarray,
    grain_ids: np.ndarray,
    phase_ids: np.ndarray,
    seeds: np.ndarray,
    spec: ModelSpecification,
    box: np.ndarray,
    periodic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    reports = []
    for inclusion_id, inclusion in enumerate(spec.inclusions):
        phase_id = next(
            i for i, phase in enumerate(spec.phases) if phase.name == inclusion.phase
        )
        phase = spec.phases[phase_id]
        center = np.asarray(inclusion.center, dtype=float)
        if inclusion.fractional_center:
            center = center * box
        if np.any(center < 0) or np.any(center >= box):
            raise ValueError(f"Inclusion {inclusion.name!r} center lies outside the box")
        size = np.asarray(inclusion.size, dtype=float)
        rotation = _inclusion_rotation(inclusion)
        remove = _inclusion_mask(
            positions, center, size, inclusion.shape, rotation, box, periodic,
            inclusion.periodic,
        )
        removed = int(np.count_nonzero(remove))
        positions = positions[~remove]
        grain_ids = grain_ids[~remove]
        phase_ids = phase_ids[~remove]
        candidates = _generate_oriented_candidates(
            center, rotation, phase, box, periodic if inclusion.periodic else np.zeros(3, bool)
        )
        selected = _inclusion_mask(
            candidates, center, size, inclusion.shape, rotation, box, periodic,
            inclusion.periodic,
        )
        candidates = candidates[selected]
        owners = _nearest_seed(candidates, seeds, box, periodic)
        positions = np.concatenate([positions, candidates])
        grain_ids = np.concatenate([grain_ids, owners.astype(np.int32)])
        phase_ids = np.concatenate([
            phase_ids, np.full(len(candidates), phase_id, dtype=np.int16)
        ])
        reports.append({
            "id": inclusion_id,
            "name": inclusion.name,
            "phase": inclusion.phase,
            "shape": inclusion.shape,
            "center": center.tolist(),
            "size": size.tolist(),
            "matrix_atoms_removed": removed,
            "inclusion_candidates": int(len(candidates)),
            "orientation": rotation.tolist(),
        })
    return positions, grain_ids, phase_ids, reports


def _classify_inclusions(atoms: Atoms, spec: ModelSpecification) -> np.ndarray:
    values = np.full(len(atoms), -1, dtype=np.int32)
    box = atoms.cell.lengths()
    periodic = np.asarray(atoms.pbc, dtype=bool)
    for inclusion_id, inclusion in enumerate(spec.inclusions):
        phase_id = next(i for i, phase in enumerate(spec.phases) if phase.name == inclusion.phase)
        center = np.asarray(inclusion.center, dtype=float)
        if inclusion.fractional_center:
            center *= box
        inside = _inclusion_mask(
            atoms.positions, center, np.asarray(inclusion.size), inclusion.shape,
            _inclusion_rotation(inclusion), box, periodic, inclusion.periodic,
        )
        values[inside & (atoms.arrays["phase_id"] == phase_id)] = inclusion_id
    return values


def _remove_overlaps(
    positions: np.ndarray,
    grain_ids: np.ndarray,
    phase_ids: np.ndarray,
    box: np.ndarray,
    cutoff: float,
    maximum_fraction: float,
    periodic: np.ndarray,
    rng: np.random.Generator,
):
    if len(positions) == 0:
        return positions, grain_ids, phase_ids, {
            "deleted_atoms": 0, "deleted_fraction": 0.0,
            "by_grain": {}, "by_phase": {}, "strategy": "balanced-conflict-graph",
        }
    tree = cKDTree(positions, boxsize=np.where(periodic, box, 0.0))
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    delete = np.zeros(len(positions), dtype=bool)
    grain_population = {int(g): int(np.count_nonzero(grain_ids == g)) for g in np.unique(grain_ids)}
    phase_population = {int(p): int(np.count_nonzero(phase_ids == p)) for p in np.unique(phase_ids)}
    grain_deleted = {key: 0 for key in grain_population}
    phase_deleted = {key: 0 for key in phase_population}
    if len(pairs):
        degree = np.bincount(pairs.ravel(), minlength=len(positions))
        max_degree = max(int(degree.max()), 1)
        # Resolve the close-contact graph while balancing deletion fractions
        # across grains and phases. High-conflict sites are preferred only when
        # doing so does not repeatedly penalize the same microstructural group.
        for pair_index in rng.permutation(len(pairs)):
            left, right = pairs[pair_index]
            if delete[left] or delete[right]:
                continue
            scores = []
            for candidate in (int(left), int(right)):
                grain = int(grain_ids[candidate]); phase = int(phase_ids[candidate])
                grain_fraction = grain_deleted[grain] / max(grain_population[grain], 1)
                phase_fraction = phase_deleted[phase] / max(phase_population[phase], 1)
                conflict_gain = degree[candidate] / max_degree
                scores.append(grain_fraction + phase_fraction - 0.25 * conflict_gain)
            if abs(scores[0] - scores[1]) < 1e-12:
                victim = int(rng.choice((left, right)))
            else:
                victim = int(left if scores[0] < scores[1] else right)
            delete[victim] = True
            grain_deleted[int(grain_ids[victim])] += 1
            phase_deleted[int(phase_ids[victim])] += 1
    deleted = int(delete.sum())
    if deleted / len(positions) > maximum_fraction:
        raise RuntimeError(
            f"Overlap cleanup would delete {deleted / len(positions):.1%} of atoms; "
            "reduce minimum_distance_factor or revise the model"
        )
    keep = ~delete
    report = {
        "strategy": "balanced-conflict-graph",
        "candidate_pairs": int(len(pairs)),
        "deleted_atoms": deleted,
        "deleted_fraction": float(deleted / len(positions)),
        "by_grain": {
            str(key): {"before": grain_population[key], "deleted": grain_deleted[key],
                       "deleted_fraction": grain_deleted[key] / max(grain_population[key], 1)}
            for key in sorted(grain_population)
        },
        "by_phase": {
            str(key): {"before": phase_population[key], "deleted": phase_deleted[key],
                       "deleted_fraction": phase_deleted[key] / max(phase_population[key], 1)}
            for key in sorted(phase_population)
        },
    }
    return positions[keep], grain_ids[keep], phase_ids[keep], report


def _assign_species(
    phase_ids: np.ndarray,
    spec: ModelSpecification,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, int]]:
    symbols = np.empty(len(phase_ids), dtype=object)
    totals: dict[str, int] = {}
    for phase_id, phase in enumerate(spec.phases):
        indices = np.flatnonzero(phase_ids == phase_id)
        counts = _integer_counts(len(indices), phase.composition)
        phase_symbols: list[str] = []
        for element, count in counts.items():
            phase_symbols.extend([element] * count)
            totals[element] = totals.get(element, 0) + count
        rng.shuffle(phase_symbols)
        symbols[indices] = phase_symbols
    return symbols.tolist(), totals


def _classify_boundaries(atoms: Atoms, boundary_width: float) -> np.ndarray:
    if len(atoms) == 0:
        return np.empty(0, dtype=np.int8)
    positions = atoms.positions
    box = atoms.cell.lengths()
    tree = cKDTree(positions, boxsize=np.where(atoms.pbc, box, 0.0))
    pairs = tree.query_pairs(boundary_width, output_type="ndarray")
    region = np.zeros(len(atoms), dtype=np.int8)
    grain_ids = atoms.arrays["grain_id"]
    phase_ids = atoms.arrays["phase_id"]
    for left, right in pairs:
        different_grain = grain_ids[left] != grain_ids[right]
        different_phase = phase_ids[left] != phase_ids[right]
        if different_phase:
            region[left] |= 2
            region[right] |= 2
        elif different_grain:
            region[left] |= 1
            region[right] |= 1
    return region


def _set_region_arrays(atoms: Atoms, boundary_width: float) -> None:
    region = _classify_boundaries(atoms, boundary_width)
    atoms.set_array("region_type", region)
    atoms.set_array("is_grain_boundary", ((region & 1) > 0).astype(np.int8))
    atoms.set_array("is_phase_boundary", ((region & 2) > 0).astype(np.int8))
    triple = np.zeros(len(atoms), dtype=np.int8)
    if len(atoms):
        tree = cKDTree(
            atoms.positions, boxsize=np.where(atoms.pbc, atoms.cell.lengths(), 0.0)
        )
        grain_ids = atoms.arrays["grain_id"]
        for index in np.flatnonzero(region):
            neighbors = tree.query_ball_point(atoms.positions[index], boundary_width)
            if len(np.unique(grain_ids[neighbors])) >= 3:
                triple[index] = 1
    atoms.set_array("is_triple_junction", triple)
    # Mutually exclusive primary class. Precedence is triple junction, mixed
    # grain/phase boundary, phase boundary, grain boundary, then interior.
    primary = np.zeros(len(atoms), dtype=np.int8)
    primary[(region & 1) > 0] = 1
    primary[(region & 2) > 0] = 2
    primary[(region & 3) == 3] = 4
    primary[triple > 0] = 3
    atoms.set_array("primary_region", primary)


def _apply_boundary_segregation(
    atoms: Atoms, spec: ModelSpecification, rng: np.random.Generator
) -> list[dict[str, Any]]:
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    reports = []
    for request in spec.chemistry.segregation:
        try:
            phase_id = next(i for i, phase in enumerate(spec.phases) if phase.name == request.phase)
        except StopIteration as exc:
            raise ValueError(f"Unknown segregation phase: {request.phase}") from exc
        phase = spec.phases[phase_id]
        if not set(request.target_composition).issubset(phase.composition):
            raise ValueError(
                f"Segregation target for {phase.name} contains elements absent from the phase"
            )
        primary_code = 1 if request.region == "grain_boundary" else 2
        boundary = np.flatnonzero(
            (atoms.arrays["phase_id"] == phase_id)
            & (atoms.arrays["primary_region"] == primary_code)
        )
        interior = np.flatnonzero(
            (atoms.arrays["phase_id"] == phase_id)
            & (atoms.arrays["primary_region"] == 0)
        )
        full_target = {
            element: request.target_composition.get(element, 0.0)
            for element in phase.composition
        }
        target = _integer_counts(len(boundary), full_target)
        for element, desired in target.items():
            current = int(np.count_nonzero(symbols[boundary] == element))
            deficit = desired - current
            if deficit <= 0:
                continue
            donors = interior[symbols[interior] == element]
            excess_sites = []
            for other, other_target in target.items():
                excess = int(np.count_nonzero(symbols[boundary] == other)) - other_target
                if excess > 0:
                    excess_sites.extend(boundary[symbols[boundary] == other][:excess].tolist())
            swaps = min(deficit, len(donors), len(excess_sites))
            if swaps:
                selected_donors = rng.choice(donors, swaps, replace=False)
                selected_boundary = rng.choice(np.asarray(excess_sites), swaps, replace=False)
                symbols[selected_donors], symbols[selected_boundary] = (
                    symbols[selected_boundary].copy(), symbols[selected_donors].copy()
                )
        achieved = {
            element: float(np.count_nonzero(symbols[boundary] == element) / len(boundary))
            if len(boundary) else 0.0
            for element in full_target
        }
        reports.append({
            "phase": request.phase, "region": request.region,
            "site_count": int(len(boundary)), "target": request.target_composition,
            "achieved": achieved,
        })
    atoms.set_chemical_symbols(symbols.tolist())
    return reports


def _apply_defects(
    atoms: Atoms, spec: ModelSpecification, rng: np.random.Generator
) -> dict[str, Any]:
    report: dict[str, Any] = {"vacancies": [], "interstitials": []}
    for request in spec.defects.vacancies:
        if request.phase is None:
            candidates = np.arange(len(atoms))
        else:
            try:
                phase_id = next(i for i, phase in enumerate(spec.phases) if phase.name == request.phase)
            except StopIteration as exc:
                raise ValueError(f"Unknown vacancy phase: {request.phase}") from exc
            candidates = np.flatnonzero(atoms.arrays["phase_id"] == phase_id)
        count = int(round(len(candidates) * request.fraction))
        selected = np.sort(rng.choice(candidates, count, replace=False)) if count else np.empty(0, int)
        del atoms[selected]
        report["vacancies"].append({
            "phase": request.phase, "fraction": request.fraction, "removed": int(count)
        })
    for request in spec.defects.interstitials:
        position = np.asarray(request.position, dtype=float)
        if request.fractional:
            position = position @ atoms.cell.array
        if np.any(position < 0) or np.any(position >= atoms.cell.lengths()):
            raise ValueError(f"Interstitial position lies outside the orthogonal box: {position}")
        if len(atoms):
            delta = atoms.positions - position
            periodic = np.asarray(atoms.pbc, dtype=bool)
            box = atoms.cell.lengths()
            delta[:, periodic] -= box[periodic] * np.round(delta[:, periodic] / box[periodic])
            nearest = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
            grain_id = int(atoms.arrays["grain_id"][nearest])
            phase_id = int(atoms.arrays["phase_id"][nearest])
        else:
            grain_id = phase_id = -1
        atoms.append(request.element)
        atoms.positions[-1] = position
        atoms.arrays["grain_id"][-1] = grain_id
        atoms.arrays["phase_id"][-1] = phase_id
        if "inclusion_id" in atoms.arrays:
            atoms.arrays["inclusion_id"][-1] = -1
        report["interstitials"].append({
            "element": request.element, "position": position.tolist(),
            "grain_id": grain_id, "phase_id": phase_id,
        })
    return report


def _structural_statistics(atoms: Atoms, spec: ModelSpecification) -> dict[str, Any]:
    if not len(atoms):
        return {"number_density": 0.0, "coordination": {}, "rdf": None}
    volume = abs(float(np.linalg.det(atoms.cell.array)))
    tree = cKDTree(
        atoms.positions, boxsize=np.where(atoms.pbc, atoms.cell.lengths(), 0.0)
    )
    phase_ids = atoms.arrays["phase_id"]
    coordination: dict[str, Any] = {}
    for phase_id, phase in enumerate(spec.phases):
        indices = np.flatnonzero(phase_ids == phase_id)
        if not len(indices):
            continue
        cutoff = _first_shell_cutoff(phase)
        neighbors = tree.query_ball_point(atoms.positions[indices], cutoff, return_length=True) - 1
        coordination[phase.name] = {
            "cutoff": float(cutoff),
            "mean": float(np.mean(neighbors)),
            "minimum": int(np.min(neighbors)),
            "maximum": int(np.max(neighbors)),
        }
    rdf = None
    if spec.analysis.rdf_enabled and len(atoms) > 1:
        sample = np.arange(len(atoms))
        if len(sample) > 20_000:
            sample = np.linspace(0, len(atoms) - 1, 20_000, dtype=int)
        positions = atoms.positions[sample]
        sample_tree = cKDTree(
            positions, boxsize=np.where(atoms.pbc, atoms.cell.lengths(), 0.0)
        )
        pairs = sample_tree.query_pairs(spec.analysis.rdf_cutoff, output_type="ndarray")
        if len(pairs):
            delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            periodic = np.asarray(atoms.pbc, dtype=bool)
            box = atoms.cell.lengths()
            delta[:, periodic] -= box[periodic] * np.round(delta[:, periodic] / box[periodic])
            distances = np.linalg.norm(delta, axis=1)
        else:
            distances = np.empty(0)
        counts, edges = np.histogram(
            distances, bins=spec.analysis.rdf_bins, range=(0, spec.analysis.rdf_cutoff)
        )
        shell_volume = 4 * math.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
        density = len(sample) / volume
        expected = len(sample) * density * shell_volume / 2
        values = np.divide(counts, expected, out=np.zeros_like(expected), where=expected > 0)
        rdf = {
            "sampled_atoms": int(len(sample)),
            "radius": ((edges[:-1] + edges[1:]) / 2).tolist(),
            "g_r": values.tolist(),
        }
    return {
        "number_density": float(len(atoms) / volume),
        "coordination": coordination,
        "rdf": rdf,
    }


def _orientation_and_misorientation_report(
    atoms: Atoms, spec: ModelSpecification, orientations: list[np.ndarray] | np.ndarray,
    boundary_width: float,
) -> dict[str, Any]:
    grain_ids = atoms.arrays["grain_id"]
    phase_ids = atoms.arrays["phase_id"]
    grains = {}
    grain_phase = {}
    for grain_id, matrix in enumerate(orientations):
        indices = np.flatnonzero(grain_ids == grain_id)
        phase_id = int(np.bincount(phase_ids[indices]).argmax()) if len(indices) else -1
        grain_phase[grain_id] = phase_id
        rotation = Rotation.from_matrix(matrix)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
            euler = rotation.as_euler("ZXZ", degrees=True).tolist()
        grains[str(grain_id)] = {
            "phase_id": phase_id,
            "phase": spec.phases[phase_id].name if phase_id >= 0 else None,
            "orientation_matrix": np.asarray(matrix).tolist(),
            "quaternion_xyzw": rotation.as_quat().tolist(),
            "euler_zxz_deg": euler,
        }
    adjacent: dict[tuple[int, int], int] = {}
    if len(atoms) > 1:
        tree = cKDTree(
            atoms.positions, boxsize=np.where(atoms.pbc, atoms.cell.lengths(), 0.0)
        )
        for left, right in tree.query_pairs(boundary_width, output_type="ndarray"):
            first, second = int(grain_ids[left]), int(grain_ids[right])
            if first == second:
                continue
            key = tuple(sorted((first, second)))
            adjacent[key] = adjacent.get(key, 0) + 1
    boundaries = []
    for (first, second), contacts in sorted(adjacent.items()):
        left_phase, right_phase = grain_phase[first], grain_phase[second]
        raw = rotation_angle_deg(orientations[first] @ orientations[second].T)
        if left_phase == right_phase and left_phase >= 0:
            structure = spec.phases[left_phase].structure
            reduced = minimum_misorientation_deg(
                orientations[first], orientations[second], structure
            )
        else:
            structure = None
            reduced = None
        boundaries.append({
            "grain_i": first, "grain_j": second, "contact_edges": contacts,
            "phase_i": left_phase, "phase_j": right_phase,
            "raw_rotation_angle_deg": raw,
            "symmetry_reduced_misorientation_deg": reduced,
            "symmetry": structure,
        })
    reduced_values = [item["symmetry_reduced_misorientation_deg"] for item in boundaries
                      if item["symmetry_reduced_misorientation_deg"] is not None]
    return {
        "euler_convention": "intrinsic ZXZ, degrees",
        "quaternion_convention": "x, y, z, w",
        "grains": grains,
        "adjacent_grain_pairs": boundaries,
        "statistics": {
            "adjacent_pair_count": len(boundaries),
            "mean_misorientation_deg": float(np.mean(reduced_values)) if reduced_values else None,
            "minimum_misorientation_deg": float(np.min(reduced_values)) if reduced_values else None,
            "maximum_misorientation_deg": float(np.max(reduced_values)) if reduced_values else None,
        },
    }


def estimate_model(spec: ModelSpecification) -> dict[str, Any]:
    volume = float(np.prod(spec.box.lengths))
    density = sum(
        phase.fraction / _atomic_volume(phase)
        for phase in spec.phases
    )
    estimated_atoms = int(round(volume * density))
    mean_grain_diameter = (6 * volume / (math.pi * spec.grains.number)) ** (1 / 3)
    return {
        "estimated_atoms": estimated_atoms,
        "estimated_memory_mb": round(estimated_atoms * 180 / 1024 ** 2, 1),
        "mean_grain_diameter": round(mean_grain_diameter, 2),
        "warnings": ["Large browser previews will be sampled"] if estimated_atoms > 100_000 else [],
    }


@dataclass
class BuildResult:
    atoms: Atoms
    specification: ModelSpecification
    metadata: dict[str, Any]
    validation: dict[str, Any]

    def _roundtrip_report(self, path: Path, format_name: str) -> dict[str, Any]:
        read_kwargs = {"format": "lammps-data", "atom_style": "atomic"} if format_name == "lammps-data" else {}
        try:
            restored = read(path, **read_kwargs)
        except Exception as exc:
            return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        original_counts = dict(sorted(Counter(self.atoms.get_chemical_symbols()).items()))
        restored_counts = dict(sorted(Counter(restored.get_chemical_symbols()).items()))
        atom_count = len(restored) == len(self.atoms)
        composition = restored_counts == original_counts
        cell = bool(np.allclose(restored.cell.array, self.atoms.cell.array, atol=2e-5))
        # Compare wrapped fractional coordinate sets; VASP sorts by species.
        def coordinate_rows(atoms: Atoms) -> list[tuple]:
            scaled = atoms.get_scaled_positions(wrap=True)
            return sorted(
                (symbol, *np.round(position, 6).tolist())
                for symbol, position in zip(atoms.get_chemical_symbols(), scaled)
            )
        coordinates = (
            bool(np.allclose(restored.positions, self.atoms.positions, atol=2e-6))
            if format_name == "extxyz"
            else coordinate_rows(restored) == coordinate_rows(self.atoms)
        )
        scientific_arrays = [
            name for name in ("grain_id", "phase_id", "region_type", "primary_region",
                              "is_grain_boundary", "is_phase_boundary", "is_triple_junction",
                              "inclusion_id", "cna_type", "ptm_type")
            if name in restored.arrays
        ]
        required = ["atom_count", "composition", "cell", "coordinates"]
        checks = {"atom_count": atom_count, "composition": composition,
                  "cell": cell, "coordinates": coordinates}
        return {
            "passed": all(checks[key] for key in required),
            "checks": checks,
            "original_composition": original_counts,
            "restored_composition": restored_counts,
            "preserved_scientific_arrays": scientific_arrays,
            "lost_scientific_arrays": sorted(
                set(self.atoms.arrays) - set(restored.arrays) - {"numbers", "positions"}
            ),
        }

    def write_outputs(self, directory: str | Path) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        name = "model"
        formats = self.specification.output.formats
        written_formats: list[tuple[str, Path]] = []
        if "extxyz" in formats:
            path = directory / f"{name}.extxyz"
            write(path, self.atoms, format="extxyz")
            paths.append(path)
            written_formats.append(("extxyz", path))
        if "lammps-data" in formats:
            path = directory / f"{name}.data"
            write(path, self.atoms, format="lammps-data", atom_style="atomic", masses=True)
            paths.append(path)
            written_formats.append(("lammps-data", path))
        if "vasp" in formats:
            path = directory / "POSCAR"
            write(path, self.atoms, format="vasp", sort=True, direct=True)
            paths.append(path)
            written_formats.append(("vasp", path))
        if "cif" in formats:
            path = directory / f"{name}.cif"
            write(path, self.atoms, format="cif")
            paths.append(path)
            written_formats.append(("cif", path))
        roundtrip = {
            format_name: self._roundtrip_report(path, format_name)
            for format_name, path in written_formats
        }
        self.validation["output_roundtrip"] = roundtrip
        roundtrip_path = directory / "output-roundtrip.json"
        roundtrip_path.write_text(
            json.dumps(roundtrip, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.append(roundtrip_path)
        history = (self.metadata.get("sro_after") or {}).get("optimization_history", [])
        if history:
            history_path = directory / "optimization-history.csv"
            with history_path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            paths.append(history_path)
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps({**self.metadata, "validation": self.validation}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(metadata_path)
        specification_path = directory / "specification.yaml"
        specification_path.write_text(
            yaml.safe_dump(
                self.specification.model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        paths.append(specification_path)
        validation_path = directory / "validation.json"
        validation_path.write_text(
            json.dumps(self.validation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.append(validation_path)
        return paths

    def write_project(self, directory: str | Path) -> list[Path]:
        """Write a self-contained AlloyModelForge project directory."""
        paths = self.write_outputs(directory)
        model_path = Path(directory) / "model.extxyz"
        if not model_path.exists():
            write(model_path, self.atoms, format="extxyz")
            paths.insert(0, model_path)
        return paths


def load_project(directory: str | Path) -> BuildResult:
    """Load a previously written AlloyModelForge project without rebuilding it."""
    directory = Path(directory)
    spec = ModelSpecification.model_validate(
        yaml.safe_load((directory / "specification.yaml").read_text(encoding="utf-8"))
    )
    atoms = read(directory / "model.extxyz")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    metadata.pop("validation", None)
    return BuildResult(atoms, spec, metadata, validation)


def validate_model(
    atoms: Atoms,
    spec: ModelSpecification,
    deleted_atoms: int,
    ovito_report: dict | None = None,
    commensurability: dict | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if len(atoms) == 0:
        failures.append("The model contains no atoms")
        minimum_distance = None
    elif len(atoms) == 1:
        minimum_distance = None
    else:
        tree = cKDTree(
            atoms.positions, boxsize=np.where(atoms.pbc, atoms.cell.lengths(), 0.0)
        )
        distances, _ = tree.query(atoms.positions, k=2)
        minimum_distance = float(np.min(distances[:, 1]))
        if minimum_distance < 0.4:
            failures.append("The model contains a severe atomic overlap")
    actual_grains = int(len(np.unique(atoms.arrays["grain_id"]))) if len(atoms) else 0
    if actual_grains != spec.grains.number:
        warnings.append(f"Expected {spec.grains.number} grains but found {actual_grains} populated grains")
    if spec.grains.orientations == "random" and any(spec.box.periodic):
        warnings.append(
            "Arbitrary crystal rotations are generally non-commensurate with an orthogonal "
            "periodic box; relax the initial structure and inspect periodic seams"
        )
        if spec.grains.number == 1 and ovito_report and ovito_report.get("available"):
            cna = ovito_report.get("methods", {}).get("cna", {}).get("counts", {})
            expected = spec.phases[0].structure.upper()
            recognized = cna.get(expected, 0) / max(len(atoms), 1)
            if recognized < 0.90:
                failures.append(
                    f"Only {recognized:.1%} of the rotated periodic single crystal is "
                    f"recognized as {expected}; use identity orientation or a commensurate cell"
                )
    elif (
        spec.grains.number == 1
        and any(spec.box.periodic)
        and commensurability
        and not commensurability["strictly_commensurate"]
    ):
        warnings.append(
            "The single-crystal lattice is not strictly commensurate with the periodic box"
        )
    phase_ids = atoms.arrays.get("phase_id", np.empty(0, dtype=np.int16))
    actual_phase_ids = set(map(int, np.unique(phase_ids)))
    missing_phases = [
        phase.name for phase_id, phase in enumerate(spec.phases) if phase_id not in actual_phase_ids
    ]
    if missing_phases:
        failures.append(f"Requested phases contain no atoms: {', '.join(missing_phases)}")
    phase_report: dict[str, Any] = {}
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    total_atoms = max(len(atoms), 1)
    for phase_id, phase in enumerate(spec.phases):
        indices = np.flatnonzero(phase_ids == phase_id)
        atom_fraction = len(indices) / total_atoms
        # Convert atom counts to an approximate volume using each ideal phase's
        # atomic volume. This is the relevant comparison to requested phase.fraction.
        atomic_volume = _atomic_volume(phase)
        composition = {
            element: float(np.count_nonzero(symbols[indices] == element) / len(indices))
            if len(indices) else 0.0
            for element in phase.composition
        }
        phase_report[phase.name] = {
            "atom_count": int(len(indices)),
            "atom_fraction": float(atom_fraction),
            "estimated_volume": float(len(indices) * atomic_volume),
            "composition": composition,
        }
        if len(indices):
            worst = max(abs(composition[e] - target) for e, target in phase.composition.items())
            if worst > spec.cleanup.maximum_composition_deviation:
                warnings.append(
                    f"Phase {phase.name} composition deviation {worst:.3f} exceeds "
                    f"{spec.cleanup.maximum_composition_deviation:.3f}"
                )
    total_estimated_volume = sum(item["estimated_volume"] for item in phase_report.values())
    for phase in spec.phases:
        item = phase_report[phase.name]
        volume_fraction = (
            item["estimated_volume"] / total_estimated_volume if total_estimated_volume else 0.0
        )
        item["estimated_volume_fraction"] = float(volume_fraction)
        item["requested_volume_fraction"] = float(phase.fraction)
        deviation = abs(volume_fraction - phase.fraction)
        item["volume_fraction_deviation"] = float(deviation)
        if item["atom_count"] and deviation > spec.cleanup.maximum_phase_fraction_deviation:
            warnings.append(
                f"Phase {phase.name} volume-fraction deviation {deviation:.3f} exceeds "
                f"{spec.cleanup.maximum_phase_fraction_deviation:.3f}"
            )
    region = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=np.int8))
    boundary_atoms = int(np.count_nonzero(region))
    boundary_fraction = float(boundary_atoms / len(atoms)) if len(atoms) else 0.0
    if boundary_fraction > spec.cleanup.maximum_boundary_fraction:
        warnings.append(
            f"Boundary/interface atoms occupy {boundary_fraction:.1%} of the model, exceeding "
            f"{spec.cleanup.maximum_boundary_fraction:.1%}; enlarge the grains, reduce "
            "cleanup.boundary_width, or treat bulk-property statistics with caution"
        )
    statistics = _structural_statistics(atoms, spec)
    from .analysis import analyze_microstructure_geometry
    geometry_quality = analyze_microstructure_geometry(atoms, spec)
    spacing = geometry_quality.get("spacing", {})
    if spacing.get("overdense_fraction", 0.0) > 0.01:
        warnings.append(
            f"{spacing['overdense_fraction']:.1%} of atoms have unusually short nearest-neighbor spacing"
        )
    if spacing.get("void_adjacent_fraction", 0.0) > 0.02:
        warnings.append(
            f"{spacing['void_adjacent_fraction']:.1%} of atoms are adjacent to unusually large local gaps"
        )
    status = "FAIL" if failures else ("WARNING" if warnings else "PASS")
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "atom_count": len(atoms),
        "grain_count": actual_grains,
        "phase_count": len(actual_phase_ids),
        "requested_phase_count": len(spec.phases),
        "phases": phase_report,
        "minimum_distance": minimum_distance,
        "deleted_atoms": deleted_atoms,
        "boundary_atoms": boundary_atoms,
        "boundary_fraction": boundary_fraction,
        "geometry_quality": geometry_quality,
        "structure_statistics": statistics,
    }


def build_model(
    spec: ModelSpecification,
    progress_callback: ProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> BuildResult:
    token = cancellation_token or CancellationToken()
    rng = np.random.default_rng(spec.random_seed)
    box = np.asarray(spec.box.lengths, dtype=float)
    periodic = np.asarray(spec.box.periodic, dtype=bool)
    token.raise_if_cancelled()
    _emit(progress_callback, "prepare", 0.03, "Preparing grain seeds and orientations")

    if spec.grains.seeds:
        seeds = np.asarray(spec.grains.seeds, dtype=float)
        if np.all((seeds >= 0) & (seeds <= 1)):
            seeds = seeds * box
        seeds %= box
    else:
        seeds = rng.random((spec.grains.number, 3)) * box
    if spec.grains.orientations == "random":
        orientations = [random_rotation_matrix(rng) for _ in range(spec.grains.number)]
    elif spec.grains.orientations == "identity":
        orientations = [np.eye(3) for _ in range(spec.grains.number)]
    elif spec.grains.orientation_matrices is not None:
        orientations = [np.asarray(value, dtype=float) for value in spec.grains.orientation_matrices]
    elif spec.grains.euler_angles_deg is not None:
        orientations = [
            Rotation.from_euler("ZXZ", value, degrees=True).as_matrix()
            for value in spec.grains.euler_angles_deg or []
        ]
    else:
        orientations = [
            _crystallographic_rotation(value.x_direction, value.z_direction)
            for value in spec.grains.crystallographic_orientations or []
        ]
    for matrix in orientations:
        validate_rotation_matrix(matrix)
    box, box_snap = _snap_nearly_commensurate_box(spec, orientations[0], box)
    seeds %= box
    grain_volumes = _estimate_grain_volumes(seeds, box, periodic)
    grain_phases = _phase_by_grain(spec, grain_volumes, rng)
    commensurability = _commensurability_report(spec, orientations, grain_phases, box)
    commensurability["box_snap"] = box_snap

    all_positions: list[np.ndarray] = []
    all_grain_ids: list[np.ndarray] = []
    all_phase_ids: list[np.ndarray] = []
    for grain_id, (seed, rotation) in enumerate(zip(seeds, orientations)):
        token.raise_if_cancelled()
        phase_id = int(grain_phases[grain_id])
        phase = spec.phases[phase_id]
        lattice_seed = (
            np.zeros(3)
            if spec.grains.number == 1 and commensurability["strictly_commensurate"]
            else seed
        )
        candidates = _generate_oriented_candidates(
            lattice_seed, rotation, phase, box, periodic
        )
        owners = _nearest_seed(candidates, seeds, box, periodic)
        selected = candidates[owners == grain_id]
        all_positions.append(selected)
        all_grain_ids.append(np.full(len(selected), grain_id, dtype=np.int32))
        all_phase_ids.append(np.full(len(selected), phase_id, dtype=np.int16))
        _emit(
            progress_callback,
            "fill_grains",
            0.08 + 0.52 * (grain_id + 1) / spec.grains.number,
            f"Filled grain {grain_id + 1}/{spec.grains.number}",
        )

    positions = np.concatenate(all_positions) if all_positions else np.empty((0, 3))
    grain_ids = np.concatenate(all_grain_ids) if all_grain_ids else np.empty(0, dtype=np.int32)
    phase_ids = np.concatenate(all_phase_ids) if all_phase_ids else np.empty(0, dtype=np.int16)
    positions, grain_ids, phase_ids, inclusion_report = _apply_inclusion_geometry(
        positions, grain_ids, phase_ids, seeds, spec, box, periodic
    )

    token.raise_if_cancelled()
    _emit(progress_callback, "cleanup", 0.66, "Removing severe interface overlaps")
    nearest = min(_nearest_ideal_distance(phase) for phase in spec.phases)
    positions, grain_ids, phase_ids, overlap_cleanup = _remove_overlaps(
        positions,
        grain_ids,
        phase_ids,
        box,
        cutoff=nearest * spec.cleanup.minimum_distance_factor,
        maximum_fraction=spec.cleanup.maximum_deleted_fraction,
        periodic=periodic,
        rng=rng,
    )

    token.raise_if_cancelled()
    _emit(progress_callback, "chemistry", 0.70, "Assigning alloy species")
    symbols, element_totals = _assign_species(phase_ids, spec, rng)

    atoms = Atoms(symbols=symbols, positions=positions, cell=box, pbc=spec.box.periodic)
    atoms.set_array("grain_id", grain_ids)
    atoms.set_array("phase_id", phase_ids)
    atoms.set_array("inclusion_id", _classify_inclusions(atoms, spec))
    defect_report = _apply_defects(atoms, spec, rng)
    _emit(progress_callback, "boundaries", 0.74, "Classifying grain and phase boundaries")
    _set_region_arrays(atoms, spec.cleanup.boundary_width)
    segregation_report = _apply_boundary_segregation(atoms, spec, rng)

    sro_before = None
    sro_after = None
    if spec.chemistry.method == "native_sqs":
        from .chemistry import optimize_native_sqs

        _emit(progress_callback, "sqs_optimize", 0.75, "Optimizing pair correlations for SQS")
        sro_before, sro_after = optimize_native_sqs(
            atoms, spec, rng, progress_callback, token
        )
    elif spec.chemistry.method in {"icet", "sqsgenerator"}:
        from .backends.sqs import optimize_with_backend

        _emit(progress_callback, "sqs_optimize", 0.78, f"Running {spec.chemistry.method} SQS backend")
        sro_before, sro_after = optimize_with_backend(atoms, spec, rng, progress_callback, token)

    ovito_report = None
    if spec.analysis.ovito_enabled:
        from .analysis import analyze_with_ovito

        _emit(progress_callback, "structure_analysis", 0.91, "Running OVITO CNA/PTM analysis")
        ovito_report = analyze_with_ovito(atoms, spec)

    orientation_report = _orientation_and_misorientation_report(
        atoms, spec, orientations, spec.cleanup.boundary_width
    )

    token.raise_if_cancelled()
    _emit(progress_callback, "validate", 0.94, "Validating model geometry and composition")
    validation = validate_model(
        atoms, spec, overlap_cleanup["deleted_atoms"], ovito_report, commensurability
    )
    validation["orientation_analysis"] = orientation_report
    from .provenance import runtime_provenance, structure_fingerprint

    metadata = {
        "name": spec.name,
        "random_seed": spec.random_seed,
        "atom_count": len(atoms),
        "grain_count": spec.grains.number,
        "phase_names": [p.name for p in spec.phases],
        "element_counts": {
            element: atoms.get_chemical_symbols().count(element)
            for element in sorted(set(atoms.get_chemical_symbols()))
        },
        "defects": defect_report,
        "segregation": segregation_report,
        "inclusions": inclusion_report,
        "overlap_cleanup": overlap_cleanup,
        "grain_phase_ids": grain_phases.tolist(),
        "grain_volume_fractions": grain_volumes.tolist(),
        "grain_seeds": seeds.tolist(),
        "grain_orientations": [matrix.tolist() for matrix in orientations],
        "orientation_analysis": orientation_report,
        "commensurability": commensurability,
        "sro_before": sro_before,
        "sro_after": sro_after,
        "ovito_analysis": ovito_report,
        "structure_sha256": structure_fingerprint(atoms),
        "provenance": runtime_provenance(),
        "specification": spec.model_dump(mode="json"),
    }
    _emit(progress_callback, "completed", 1.0, "Model completed")
    return BuildResult(atoms, spec, metadata, validation)
