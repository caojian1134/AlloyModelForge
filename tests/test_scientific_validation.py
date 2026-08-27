import math

import pytest

from alloyforge import ModelSpecification, build_model
from alloyforge.chemistry import (
    _apply_target_pair_factors, _counts_and_targets, _crossing_shell_edges,
    _ideal_shell_distances, _integer_target_counts, _shell_edges,
    _touching_shell_edges, calculate_constraint_feasibility,
)
import numpy as np
from ase.build import bulk
from alloyforge.config import PhaseSpecification, RegionConstraintSpecification


@pytest.mark.parametrize(
    "structure,a,lengths,expected_atoms,expected_nearest,expected_coordination",
    [
        ("fcc", 3.6, [7.2, 7.2, 7.2], 32, 3.6 / math.sqrt(2), 12),
        ("bcc", 2.87, [5.74, 5.74, 5.74], 16, 2.87 * math.sqrt(3) / 2, 8),
        ("hcp", 2.95, [8.85, 3 * math.sqrt(3) * 2.95, 3 * 2.95 * 1.633],
         108, 2.95, 12),
    ],
)
def test_reference_crystals_have_exact_geometry(
    structure, a, lengths, expected_atoms, expected_nearest, expected_coordination
):
    spec = ModelSpecification.model_validate({
        "name": f"reference-{structure}",
        "box": {"lengths": lengths, "periodic": [True, True, True]},
        "phases": [{"name": structure.upper(), "structure": structure,
                    "lattice_parameter": a, "c_over_a": 1.633,
                    "fraction": 1, "composition": {"Ti" if structure == "hcp" else "Fe": 1}}],
        "grains": {"number": 1, "orientations": "identity"},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    phase = result.validation["structure_statistics"]["coordination"][structure.upper()]
    assert result.validation["status"] == "PASS"
    assert len(result.atoms) == expected_atoms
    assert result.validation["deleted_atoms"] == 0
    assert result.validation["minimum_distance"] == pytest.approx(expected_nearest, abs=1e-8)
    assert phase["mean"] == pytest.approx(expected_coordination)
    assert phase["minimum"] == expected_coordination
    assert phase["maximum"] == expected_coordination


@pytest.mark.parametrize("seed", [1, 7, 29])
def test_native_sqs_improves_multiple_independent_seeds(seed):
    spec = ModelSpecification.model_validate({
        "random_seed": seed,
        "box": {"lengths": [7.2, 7.2, 7.2], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .25, "Cr": .25,
                                                    "Fe": .25, "Ni": .25}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": "native_sqs", "steps": 2000,
                      "neighbor_shells": 2, "exclude_boundaries": False},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    assert result.metadata["element_counts"] == {"Co": 8, "Cr": 8, "Fe": 8, "Ni": 8}
    assert result.metadata["sro_after"]["total_objective"] <= result.metadata["sro_before"]["total_objective"]


def test_structure_fingerprint_detects_seed_change():
    base = {
        "box": {"lengths": [14.4, 14.4, 14.4], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Ni": .5, "Co": .5}}],
        "grains": {"number": 3, "orientations": "random"},
        "cleanup": {"minimum_distance_factor": 0.6, "maximum_deleted_fraction": 0.25},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    }
    first = build_model(ModelSpecification.model_validate({**base, "random_seed": 11}))
    second = build_model(ModelSpecification.model_validate({**base, "random_seed": 12}))
    assert first.metadata["structure_sha256"] != second.metadata["structure_sha256"]


def test_region_constrained_sqs_improves_boundary_composition_and_preserves_global_counts():
    spec = ModelSpecification.model_validate({
        "name": "region-constrained-polycrystal",
        "random_seed": 17,
        "box": {"lengths": [18.0, 18.0, 18.0], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 4, "orientations": "identity"},
        "chemistry": {
            "method": "native_sqs", "steps": 6000, "neighbor_shells": 2,
            "exclude_boundaries": False,
            "region_constraints": [{
                "name": "Ni-rich-GB", "phase": "FCC", "region": "grain_boundary",
                "target_composition": {"Co": .25, "Ni": .75},
                "composition_weight": 30, "pair_weight": 1,
            }],
        },
        "cleanup": {"boundary_width": 3.0, "maximum_deleted_fraction": .25},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    before = result.metadata["sro_before"]["region_constraints"]["Ni-rich-GB"]
    after = result.metadata["sro_after"]["region_constraints"]["Ni-rich-GB"]
    assert before["site_count"] > 0
    assert after["composition_l1_error"] < before["composition_l1_error"]
    assert result.metadata["sro_after"]["optimization_objective"] < result.metadata["sro_before"]["optimization_objective"]
    assert result.metadata["element_counts"] == {
        symbol: result.atoms.get_chemical_symbols().count(symbol) for symbol in ("Co", "Ni")
    }
    quality = result.validation["geometry_quality"]
    assert quality["regions"]["grain_boundary"]["count"] > 0
    assert len(quality["grains"]) == 4
    assert set(quality["periodic_seams"]) == {"x", "y", "z"}


def test_composition_only_region_constraint_is_optimized():
    spec = ModelSpecification.model_validate({
        "random_seed": 17,
        "box": {"lengths": [18.0, 18.0, 18.0], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 4, "orientations": "identity"},
        "chemistry": {
            "method": "native_sqs", "steps": 4000, "neighbor_shells": 2,
            "exclude_boundaries": False,
            "region_constraints": [{
                "name": "composition-only", "phase": "FCC", "region": "grain_boundary",
                "target_composition": {"Co": .25, "Ni": .75},
                "composition_weight": 30, "pair_weight": 0,
            }],
        },
        "cleanup": {"boundary_width": 3.0, "maximum_deleted_fraction": .25},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    before = result.metadata["sro_before"]["region_constraints"]["composition-only"]
    after = result.metadata["sro_after"]["region_constraints"]["composition-only"]
    assert after["composition_l1_error"] < before["composition_l1_error"]
    counts = result.metadata["element_counts"]
    assert sum(counts.values()) == len(result.atoms)
    assert abs(counts["Co"] - counts["Ni"]) <= 1


def test_multiphase_target_composition_requires_explicit_phase():
    with pytest.raises(ValueError, match="must name a phase"):
        ModelSpecification.model_validate({
            "phases": [
                {"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                 "fraction": .5, "composition": {"Ni": 1}},
                {"name": "BCC", "structure": "bcc", "lattice_parameter": 2.9,
                 "fraction": .5, "composition": {"Fe": 1}},
            ],
            "grains": {"number": 2},
            "chemistry": {"method": "native_sqs", "region_constraints": [{
                "name": "ambiguous", "region": "all",
                "target_composition": {"Ni": .5, "Fe": .5},
            }]},
        })


def test_nonideal_hcp_shells_are_derived_from_reference_cell():
    phase = PhaseSpecification(
        name="HCP", structure="hcp", lattice_parameter=3.0,
        c_over_a=1.4, fraction=1, composition={"Ti": 1},
    )
    shells = _ideal_shell_distances(phase, 3)
    interlayer = math.sqrt(phase.lattice_parameter ** 2 / 3 +
                           (phase.lattice_parameter * phase.c_over_a) ** 2 / 4)
    assert shells[0] == pytest.approx(interlayer)
    assert shells[1] == pytest.approx(phase.lattice_parameter)
    assert shells == sorted(shells)


def test_finite_population_pair_targets_use_sampling_without_replacement():
    symbols = np.asarray(["A", "A", "B", "B"], dtype=object)
    edges = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    keys, _, _, finite = _counts_and_targets(
        symbols, edges, ["A", "B"], {"A": .5, "B": .5}, {"A": 2, "B": 2}
    )
    targets = dict(zip(keys, finite))
    assert targets[("A", "A")] == pytest.approx(1.0)
    assert targets[("B", "B")] == pytest.approx(1.0)
    assert targets[("A", "B")] == pytest.approx(4.0)
    assert _integer_target_counts(7, {"A": .5, "B": .5}) in (
        {"A": 4, "B": 3}, {"A": 3, "B": 4}
    )


def test_touching_region_pairs_include_neighbors_outside_region():
    atoms = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    phase = PhaseSpecification(name="FCC", structure="fcc", lattice_parameter=3.6,
                               fraction=1, composition={"Ni": 1})
    all_indices = np.arange(len(atoms))
    region = np.asarray([0], dtype=int)
    internal = _shell_edges(atoms, region, phase, 1)[0]
    touching = _touching_shell_edges(atoms, region, all_indices, phase, 1)[0]
    assert len(internal) == 0
    assert len(touching) == 12


def test_crossing_region_pairs_exclude_internal_bonds():
    atoms = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    phase = PhaseSpecification(name="FCC", structure="fcc", lattice_parameter=3.6,
                               fraction=1, composition={"Ni": 1})
    all_indices = np.arange(len(atoms))
    region = np.asarray([0, 1], dtype=int)
    crossing = _crossing_shell_edges(atoms, region, all_indices, phase, 1)[0]
    touching = _touching_shell_edges(atoms, region, all_indices, phase, 1)[0]
    assert len(crossing) < len(touching)
    assert all((left in region) != (right in region) for left, right in crossing)


def test_pair_propensity_factors_preserve_shell_pair_total():
    constraint = RegionConstraintSpecification(
        name="ordering", target_pair_factors={1: {"Co-Ni": 2.0}}
    )
    keys = [("Co", "Co"), ("Co", "Ni"), ("Ni", "Ni")]
    targets = np.asarray([10.0, 20.0, 10.0])
    adjusted = _apply_target_pair_factors(targets, keys, constraint, 1)
    assert adjusted.sum() == pytest.approx(targets.sum())
    assert adjusted[1] / adjusted[0] == pytest.approx(4.0)


def test_joint_composition_milp_detects_overlapping_conflict():
    spec = ModelSpecification.model_validate({
        "box": {"lengths": [7.2] * 3, "periodic": [True] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 4, "orientations": "identity"},
        "chemistry": {"method": "random", "region_constraints": [
            {"name": "left-Co-rich", "phase": "FCC", "region": "all",
             "grain_ids": [0, 1], "target_composition": {"Co": .9375, "Ni": .0625}},
            {"name": "right-Co-rich", "phase": "FCC", "region": "all",
             "grain_ids": [1, 2], "target_composition": {"Co": .9375, "Ni": .0625}},
        ]},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    atoms = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms.arrays["phase_id"] = np.zeros(32, dtype=np.int32)
    atoms.arrays["region_type"] = np.zeros(32, dtype=np.int8)
    atoms.arrays["primary_region"] = np.zeros(32, dtype=np.int8)
    # Four 8-site membership groups. Each 16-site region independently needs
    # 15 Co (available globally: 20), but their overlap has only 8 sites, so
    # satisfying both would require at least 22 Co.
    atoms.arrays["grain_id"] = np.repeat(np.arange(4), 8).astype(np.int32)
    atoms.set_chemical_symbols(["Co"] * 20 + ["Ni"] * 12)
    feasibility = calculate_constraint_feasibility(atoms, spec)
    assert feasibility["all_availability_feasible"] is True
    assert feasibility["all_jointly_feasible"] is False
    assert feasibility["joint_composition"]["FCC"]["status"] == "infeasible"


def test_native_sqs_multistart_reports_distribution_and_selects_minimum():
    spec = ModelSpecification.model_validate({
        "random_seed": 31,
        "box": {"lengths": [7.2] * 3, "periodic": [True] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": "native_sqs", "steps": 300,
                      "neighbor_shells": 2, "restarts": 3,
                      "exclude_boundaries": False},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    report = build_model(spec).metadata["sro_after"]
    restarts = report["optimization_summary"]["restarts"]
    assert restarts["count"] == 3
    assert restarts["objectives"][restarts["selected"]] == pytest.approx(
        min(restarts["objectives"])
    )
    assert report["optimization_objective_breakdown"]["definition"] == (
        "weighted_mean_squared_relative_residual"
    )


def test_exact_small_cell_benchmark_reports_optimality_gap():
    spec = ModelSpecification.model_validate({
        "random_seed": 4,
        "box": {"lengths": [3.6] * 3, "periodic": [True] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": "native_sqs", "steps": 200,
                      "neighbor_shells": 1, "exclude_boundaries": False,
                      "exact_benchmark_max_sites": 4},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    exact = build_model(spec).metadata["sro_after"]["exact_benchmark"]
    assert exact["status"] == "completed"
    assert exact["configuration_count"] == 6
    assert exact["evaluated"] == 6
    assert exact["absolute_optimality_gap"] >= -1e-8


def test_directional_sro_constraint_is_optimized_across_boundary_to_interior():
    spec = ModelSpecification.model_validate({
        "random_seed": 23,
        "box": {"lengths": [18.0] * 3, "periodic": [True] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 4, "orientations": "identity"},
        "chemistry": {
            "method": "native_sqs", "steps": 3000, "neighbor_shells": 1,
            "exclude_boundaries": False,
            "directional_sro_constraints": [{
                "name": "gb-to-bulk", "shell": 1, "weight": 8,
                "source": {"phase": "FCC", "region": "grain_boundary"},
                "target": {"phase": "FCC", "region": "interior"},
                "target_sro": {"Co->Ni": -0.35, "Ni->Co": -0.35},
            }],
        },
        "cleanup": {"boundary_width": 3.0, "maximum_deleted_fraction": .25},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    before = result.metadata["sro_before"]["directional_sro_constraints"]["gb-to-bulk"]
    after = result.metadata["sro_after"]["directional_sro_constraints"]["gb-to-bulk"]
    assert before["edge_count"] > 0
    assert after["objective"] <= before["objective"]
    assert result.metadata["sro_after"]["optimization_objective"] < result.metadata["sro_before"]["optimization_objective"]
    assert result.metadata["sro_after"]["optimization_summary"]["phases"][0]["best_objective"] == pytest.approx(
        result.metadata["sro_after"]["optimization_objective"], rel=1e-6, abs=1e-8
    )


def test_angular_triplet_constraint_builds_motifs_and_improves_objective():
    spec = ModelSpecification.model_validate({
        "random_seed": 18,
        "box": {"lengths": [7.2] * 3, "periodic": [True] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {
            "method": "native_sqs", "steps": 2500, "neighbor_shells": 1,
            "exclude_boundaries": False,
            "triplet_constraints": [{
                "name": "fcc-angular-random", "selector": {"phase": "FCC", "region": "all"},
                "shell": 1, "angle_bins": [60, 90, 120, 180], "weight": 10,
            }],
        },
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    before = result.metadata["sro_before"]["triplet_constraints"]["fcc-angular-random"]
    after = result.metadata["sro_after"]["triplet_constraints"]["fcc-angular-random"]
    assert before["motif_count"] > 0
    assert after["objective"] <= before["objective"]
    assert after["target_model"] == "finite_fixed_composition_without_replacement_triplets"
    assert result.metadata["sro_after"]["optimization_summary"]["phases"][0]["best_objective"] == pytest.approx(
        result.metadata["sro_after"]["optimization_objective"], rel=1e-6, abs=1e-8
    )


def test_native_sqs_reports_feasibility_and_early_stopping():
    spec = ModelSpecification.model_validate({
        "random_seed": 9,
        "box": {"lengths": [7.2, 7.2, 7.2], "periodic": [True] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Co": .5, "Ni": .5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": "native_sqs", "steps": 2000, "objective_tolerance": 1e9,
                      "region_constraints": [{"name": "whole", "phase": "FCC", "region": "all",
                                              "target_composition": {"Co": .5, "Ni": .5},
                                              "pair_scope": "touching"}]},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    report = build_model(spec).metadata["sro_after"]
    assert report["constraint_feasibility"]["all_availability_feasible"] is True
    assert report["region_constraints"]["whole"]["pair_scope"] == "touching"
    assert report["optimization_summary"]["phases"][0]["stop_reason"] == "objective_tolerance"
