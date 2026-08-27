from __future__ import annotations

from collections import Counter
import warnings

import numpy as np
from ase import Atoms

from .config import ModelSpecification


STRUCTURE_NAMES = {
    0: "Other", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO",
    5: "SC", 6: "Cubic diamond", 7: "Hexagonal diamond", 8: "Graphene",
}


def _counts(values: np.ndarray) -> dict[str, int]:
    return {STRUCTURE_NAMES.get(int(k), f"Type {k}"): int(v) for k, v in sorted(Counter(values).items())}


def _counts_by_region(atoms: Atoms, values: np.ndarray) -> dict[str, dict[str, int]]:
    region = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=int))
    return {"interior": _counts(values[region == 0]), "interface": _counts(values[region > 0])}


def _ovito_compatible_atoms(atoms: Atoms) -> Atoms:
    """Convert ASE integer arrays unsupported by OVITO (e.g. int16)."""
    compatible = atoms.copy()
    for name, values in list(compatible.arrays.items()):
        if values.dtype.kind in "iu" and values.dtype not in (
            np.dtype(np.int8), np.dtype(np.int32), np.dtype(np.int64)
        ):
            compatible.arrays[name] = values.astype(np.int32)
    return compatible


def analyze_with_ovito(atoms: Atoms, spec: ModelSpecification) -> dict:
    """Run OVITO structure identification and persist results in ASE arrays.

    OVITO is optional so a model remains buildable in a lightweight install.
    The returned report explicitly records unavailability instead of inventing
    classifications or silently failing.
    """
    report: dict = {"available": False, "requested": list(spec.analysis.methods), "methods": {}}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
            from ovito.io.ase import ase_to_ovito
            from ovito.modifiers import CommonNeighborAnalysisModifier, PolyhedralTemplateMatchingModifier
            from ovito.pipeline import Pipeline, StaticSource
    except (ImportError, OSError) as exc:
        report["message"] = f"OVITO unavailable: {type(exc).__name__}: {exc}"
        return report

    report["available"] = True
    source_atoms = _ovito_compatible_atoms(atoms)
    if "cna" in spec.analysis.methods:
        pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(source_atoms)))
        modifier = CommonNeighborAnalysisModifier(
            mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
        )
        pipeline.modifiers.append(modifier)
        data = pipeline.compute()
        values = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        atoms.set_array("cna_type", values)
        report["methods"]["cna"] = {"mode": "adaptive_cutoff", "counts": _counts(values),
                                      "counts_by_region": _counts_by_region(atoms, values)}

    if "ptm" in spec.analysis.methods:
        pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(source_atoms)))
        modifier = PolyhedralTemplateMatchingModifier(
            output_orientation=True,
            output_rmsd=True,
            rmsd_cutoff=spec.analysis.ptm_rmsd_cutoff,
        )
        pipeline.modifiers.append(modifier)
        data = pipeline.compute()
        values = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        atoms.set_array("ptm_type", values)
        method_report = {
            "rmsd_cutoff": spec.analysis.ptm_rmsd_cutoff,
            "counts": _counts(values),
            "counts_by_region": _counts_by_region(atoms, values),
        }
        if "RMSD" in data.particles:
            rmsd = np.asarray(data.particles["RMSD"], dtype=float)
            atoms.set_array("ptm_rmsd", rmsd)
            method_report["mean_rmsd"] = float(np.mean(rmsd))
        if "Orientation" in data.particles:
            atoms.set_array("ptm_orientation", np.asarray(data.particles["Orientation"], dtype=float))
        report["methods"]["ptm"] = method_report
    return report


def analyze_microstructure_geometry(atoms: Atoms, spec: ModelSpecification) -> dict:
    """Quantify region populations, local spacing, grains, and periodic seams."""
    if not len(atoms):
        return {"atom_count": 0, "regions": {}, "grains": {}, "spacing": {}}
    from scipy.spatial import cKDTree

    positions = atoms.positions
    lengths = atoms.cell.lengths()
    periodic = np.asarray(atoms.pbc, dtype=bool)
    tree = cKDTree(positions, boxsize=np.where(periodic, lengths, 0.0))
    distances, _ = tree.query(positions, k=min(2, len(atoms)))
    nearest = distances[:, -1] if len(atoms) > 1 else np.asarray([np.nan])
    region = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=np.int8))
    triple = atoms.arrays.get("is_triple_junction", np.zeros(len(atoms), dtype=np.int8)) > 0
    labels = {
        "interior": region == 0,
        "grain_boundary": (region & 1) > 0,
        "phase_boundary": (region & 2) > 0,
        "triple_junction": triple,
    }
    regions = {
        name: {"count": int(np.count_nonzero(mask)), "fraction": float(np.mean(mask))}
        for name, mask in labels.items()
    }
    primary = atoms.arrays.get("primary_region", np.zeros(len(atoms), dtype=np.int8))
    primary_names = {0: "interior", 1: "grain_boundary", 2: "phase_boundary",
                     3: "triple_junction", 4: "mixed_boundary"}
    primary_regions = {
        primary_names[code]: {
            "count": int(np.count_nonzero(primary == code)),
            "fraction": float(np.mean(primary == code)),
        }
        for code in primary_names
    }
    grain_ids = atoms.arrays["grain_id"]
    phase_ids = atoms.arrays["phase_id"]
    grains = {}
    box_volume = float(abs(np.linalg.det(atoms.cell.array)))
    for grain_id in sorted(map(int, np.unique(grain_ids))):
        mask = grain_ids == grain_id
        grains[str(grain_id)] = {
            "atom_count": int(np.count_nonzero(mask)),
            "atom_fraction": float(np.mean(mask)),
            "estimated_volume": float(np.mean(mask) * box_volume),
            "equivalent_sphere_diameter": float(
                (6.0 * np.mean(mask) * box_volume / np.pi) ** (1.0 / 3.0)
            ),
            "centroid": np.mean(positions[mask], axis=0).tolist(),
            "phase_ids": sorted(map(int, np.unique(phase_ids[mask]))),
            "boundary_fraction": float(np.mean(region[mask] > 0)),
        }
    grain_diameters = [value["equivalent_sphere_diameter"] for value in grains.values()]
    grain_statistics = {
        "requested_count": int(spec.grains.number),
        "populated_count": len(grains),
        "empty_grain_ids": sorted(set(range(spec.grains.number)) - set(map(int, grains))),
        "equivalent_diameter": {
            "minimum": float(np.min(grain_diameters)), "mean": float(np.mean(grain_diameters)),
            "maximum": float(np.max(grain_diameters)), "std": float(np.std(grain_diameters)),
        } if grain_diameters else {},
    }

    inclusion_ids = atoms.arrays.get("inclusion_id", np.full(len(atoms), -1, dtype=int))
    inclusion_quality = {}
    for inclusion_id, inclusion in enumerate(spec.inclusions):
        mask = inclusion_ids == inclusion_id
        center = np.asarray(inclusion.center, dtype=float)
        if inclusion.fractional_center:
            center = center @ atoms.cell.array
        delta = positions - center
        for axis in range(3):
            if periodic[axis]:
                delta[:, axis] -= np.rint(delta[:, axis] / lengths[axis]) * lengths[axis]
        distance = np.linalg.norm(delta, axis=1)
        radial_edges = np.linspace(0.0, max(float(np.max(distance[mask])) if np.any(mask) else 0.0,
                                            max(inclusion.size)), 9)
        phase_index = next(i for i, phase in enumerate(spec.phases) if phase.name == inclusion.phase)
        radial_profile = []
        for lower, upper in zip(radial_edges[:-1], radial_edges[1:]):
            shell = (distance >= lower) & (distance < upper)
            radial_profile.append({
                "r_inner": float(lower), "r_outer": float(upper),
                "atom_count": int(np.count_nonzero(shell)),
                "target_phase_fraction": float(np.mean(phase_ids[shell] == phase_index)) if np.any(shell) else None,
            })
        element_counts = Counter(atoms.get_chemical_symbols()[i] for i in np.flatnonzero(mask))
        total = int(np.count_nonzero(mask))
        achieved_composition = ({key: value / total for key, value in sorted(element_counts.items())}
                                if total else {})
        target_composition = spec.phases[phase_index].composition
        composition_error = {element: achieved_composition.get(element, 0.0) - fraction
                             for element, fraction in target_composition.items()}
        component_count, largest_component_fraction = 0, None
        if total:
            subset = positions[mask]
            subset_tree = cKDTree(subset, boxsize=lengths if np.all(periodic) else None)
            nearest_subset, _ = subset_tree.query(subset, k=min(2, total))
            cutoff = 1.35 * float(np.median(nearest_subset[:, -1])) if total > 1 else 0.0
            parent = np.arange(total)
            def find(value):
                while parent[value] != value:
                    parent[value] = parent[parent[value]]
                    value = parent[value]
                return value
            for first, second in subset_tree.query_pairs(cutoff):
                root_first, root_second = find(first), find(second)
                if root_first != root_second:
                    parent[root_second] = root_first
            component_sizes = Counter(find(i) for i in range(total))
            component_count = len(component_sizes)
            largest_component_fraction = max(component_sizes.values()) / total
        inclusion_quality[inclusion.name] = {
            "inclusion_id": inclusion_id, "shape": inclusion.shape,
            "center": center.tolist(), "specified_size": list(inclusion.size),
            "atom_count": total, "phase_id": phase_index,
            "composition": achieved_composition, "target_composition": target_composition,
            "composition_error": composition_error,
            "connected_component_count": component_count,
            "largest_component_fraction": largest_component_fraction,
            "maximum_radial_extent": float(np.max(distance[mask])) if total else None,
            "center_of_atoms_offset": float(np.linalg.norm(np.mean(delta[mask], axis=0))) if total else None,
            "touches_periodic_half_box": bool(total and np.any(np.max(np.abs(delta[mask]), axis=0) >= lengths / 2)),
            "radial_phase_profile": radial_profile,
        }
    seam_counts = {}
    seam_width = spec.cleanup.boundary_width
    for axis, axis_name in enumerate("xyz"):
        if not periodic[axis]:
            seam_counts[axis_name] = {"periodic": False, "near_faces": 0}
            continue
        near = (positions[:, axis] < seam_width) | (positions[:, axis] > lengths[axis] - seam_width)
        seam_counts[axis_name] = {
            "periodic": True,
            "near_faces": int(np.count_nonzero(near)),
            "boundary_fraction_near_faces": float(np.mean(region[near] > 0)) if np.any(near) else 0.0,
        }
    finite = nearest[np.isfinite(nearest)]
    spacing = {
        "minimum": float(np.min(finite)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "maximum": float(np.max(finite)) if len(finite) else None,
        "p01": float(np.quantile(finite, 0.01)) if len(finite) else None,
        "p50": float(np.quantile(finite, 0.50)) if len(finite) else None,
        "p99": float(np.quantile(finite, 0.99)) if len(finite) else None,
    }
    if len(finite):
        # Normalize anomaly thresholds by phase. A global median systematically
        # misclassifies the naturally different nearest-neighbor distances of
        # FCC/BCC/HCP phases as overlaps or voids.
        overdense = np.zeros(len(atoms), dtype=bool)
        void_adjacent = np.zeros(len(atoms), dtype=bool)
        spacing_by_phase = {}
        for phase_id in map(int, np.unique(phase_ids)):
            mask = (phase_ids == phase_id) & np.isfinite(nearest)
            phase_nearest = nearest[mask]
            reference_mask = mask & (region == 0)
            if np.count_nonzero(reference_mask) < 10:
                reference_mask = mask
            phase_median = max(float(np.median(nearest[reference_mask])), 1e-12)
            overdense[mask] = phase_nearest < 0.70 * phase_median
            void_adjacent[mask] = phase_nearest > 1.45 * phase_median
            spacing_by_phase[str(phase_id)] = {
                "atom_count": int(np.count_nonzero(mask)),
                "median": phase_median,
                "reference": "interior" if np.any(mask & (region == 0)) and
                np.count_nonzero(mask & (region == 0)) >= 10 else "all_phase_atoms",
                "overdense_fraction": float(np.mean(overdense[mask])),
                "void_adjacent_fraction": float(np.mean(void_adjacent[mask])),
            }
        spacing["normalization"] = "per_phase_interior_median_with_fallback"
        spacing["by_phase"] = spacing_by_phase
        spacing["overdense_fraction"] = float(np.mean(overdense))
        spacing["void_adjacent_fraction"] = float(np.mean(void_adjacent))
    neighbor_order = min(13, len(atoms))
    local_density: dict[str, Any]
    if neighbor_order >= 2:
        neighbor_distances, _ = tree.query(positions, k=neighbor_order)
        radius = neighbor_distances[:, -1]
        k = neighbor_order - 1
        density = k / np.maximum((4.0 * np.pi / 3.0) * radius ** 3, 1e-12)
        atoms.set_array("local_number_density", density.astype(float))
        median_density = float(np.median(density))
        low_density = np.zeros(len(atoms), dtype=bool)
        density_by_phase = {}
        for phase_id in map(int, np.unique(phase_ids)):
            mask = phase_ids == phase_id
            phase_density = density[mask]
            reference_mask = mask & (region == 0)
            if np.count_nonzero(reference_mask) < 10:
                reference_mask = mask
            phase_median = max(float(np.median(density[reference_mask])), 1e-12)
            low_density[mask] = phase_density < 0.60 * phase_median
            density_by_phase[str(phase_id)] = {
                "atom_count": int(np.count_nonzero(mask)),
                "minimum": float(np.min(phase_density)),
                "mean": float(np.mean(phase_density)),
                "median": phase_median,
                "reference": "interior" if np.count_nonzero(mask & (region == 0)) >= 10
                else "all_phase_atoms",
                "maximum": float(np.max(phase_density)),
                "low_density_fraction": float(np.mean(low_density[mask])),
            }
        local_density = {
            "neighbor_order": k,
            "normalization": "per_phase_interior_median_with_fallback",
            "by_phase": density_by_phase,
            "minimum": float(np.min(density)), "mean": float(np.mean(density)),
            "maximum": float(np.max(density)), "median": median_density,
            "p01": float(np.quantile(density, .01)),
            "p99": float(np.quantile(density, .99)),
            "low_density_atom_count": int(np.count_nonzero(low_density)),
            "low_density_fraction": float(np.mean(low_density)),
            "low_density_fraction_interior": (
                float(np.mean(low_density[region == 0])) if np.any(region == 0) else None
            ),
            "low_density_fraction_boundary": (
                float(np.mean(low_density[region > 0])) if np.any(region > 0) else None
            ),
            "largest_nearest_neighbor_gap": float(np.max(finite)) if len(finite) else None,
            "void_candidate_indices": np.flatnonzero(low_density)[:100].astype(int).tolist(),
        }
    else:
        local_density = {"neighbor_order": 0}
    return {
        "atom_count": len(atoms),
        "regions": regions,
        "primary_regions": primary_regions,
        "region_overlap_rule": {
            "exclusive_precedence": ["triple_junction", "mixed_boundary",
                                     "phase_boundary", "grain_boundary", "interior"],
            "constraint_selectors_use_primary_region": True,
        },
        "grains": grains,
        "grain_statistics": grain_statistics,
        "inclusions": inclusion_quality,
        "spacing": spacing,
        "local_density": local_density,
        "periodic_seams": seam_counts,
    }
