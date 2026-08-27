from pathlib import Path
import json

import numpy as np
import pytest
from ase.io import read
from ase.io import write
from ase.build import bulk

from alloyforge import (
    ModelSpecification, build_model, estimate_model, load_project, load_specification,
)


def simple_spec(grains=3):
    return ModelSpecification.model_validate(
        {
            "name": "test",
            "random_seed": 13,
            "box": {"lengths": [18, 18, 18], "periodic": [True, True, True]},
            "phases": [
                {
                    "name": "fcc",
                    "structure": "fcc",
                    "lattice_parameter": 3.6,
                    "fraction": 1.0,
                    "composition": {"Ni": 0.5, "Co": 0.5},
                }
            ],
            "grains": {"number": grains, "orientations": "random"},
            "cleanup": {"minimum_distance_factor": 0.6, "maximum_deleted_fraction": 0.2},
        }
    )


def test_estimate_is_positive():
    assert estimate_model(simple_spec())["estimated_atoms"] > 0


def test_build_polycrystal_is_reproducible():
    first = build_model(simple_spec())
    second = build_model(simple_spec())
    assert len(first.atoms) > 0
    assert len(first.atoms) == len(second.atoms)
    assert np.allclose(first.atoms.positions, second.atoms.positions)
    assert first.metadata["structure_sha256"] == second.metadata["structure_sha256"]
    assert first.metadata["provenance"]["alloyforge_version"]
    assert first.metadata["provenance"]["alloy_model_forge_version"]
    assert first.metadata["provenance"]["dependencies"]["ase"]
    assert set(first.atoms.get_chemical_symbols()) == {"Ni", "Co"}
    assert "grain_id" in first.atoms.arrays
    assert "phase_id" in first.atoms.arrays
    assert "region_type" in first.atoms.arrays
    assert "is_grain_boundary" in first.atoms.arrays
    assert "is_phase_boundary" in first.atoms.arrays
    assert "is_triple_junction" in first.atoms.arrays


def test_native_sqs_preserves_composition_and_improves_pair_objective():
    data = simple_spec(grains=1).model_dump()
    data["chemistry"].update({"method": "native_sqs", "neighbor_shells": 2,
                              "steps": 3000, "exclude_boundaries": False,
                              "random_temperature": 0.02})
    data["analysis"].update({"ovito_enabled": False, "methods": []})
    spec = ModelSpecification.model_validate(data)
    result = build_model(spec)
    symbols = result.atoms.get_chemical_symbols()
    assert symbols.count("Ni") + symbols.count("Co") == len(symbols)
    assert abs(symbols.count("Ni") - symbols.count("Co")) <= 1
    assert result.metadata["sro_after"]["total_objective"] <= result.metadata["sro_before"]["total_objective"]
    assert "warren_cowley" in result.metadata["sro_after"]["phases"]["fcc"][0]


def test_multiphase_build_populates_every_phase_and_reports_fractions():
    spec = ModelSpecification.model_validate({
        "random_seed": 7,
        "box": {"lengths": [18, 18, 18], "periodic": [True, True, True]},
        "phases": [
            {"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
             "fraction": 0.6, "composition": {"Ni": 1}},
            {"name": "BCC", "structure": "bcc", "lattice_parameter": 2.9,
             "fraction": 0.4, "composition": {"Fe": 1}},
        ],
        "grains": {"number": 4, "orientations": "identity"},
        "analysis": {"ovito_enabled": False, "methods": []},
    })
    result = build_model(spec)
    assert result.validation["phase_count"] == 2
    density = result.validation["geometry_quality"]["local_density"]
    assert density["normalization"] == "per_phase_interior_median_with_fallback"
    assert len(density["by_phase"]) == 2
    assert density["low_density_fraction_interior"] is not None
    assert density["low_density_fraction_boundary"] is not None
    assert all(item["atom_count"] > 0 for item in result.validation["phases"].values())
    assert set(result.atoms.get_chemical_symbols()) == {"Ni", "Fe"}


def test_sqsgenerator_backend_is_real_and_preserves_composition():
    pytest.importorskip("sqsgenerator")
    spec = ModelSpecification.model_validate({
        "random_seed": 3,
        "box": {"lengths": [7.2, 7.2, 7.2], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Ni": 0.5, "Co": 0.5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": "sqsgenerator", "steps": 2000,
                      "neighbor_shells": 2, "exclude_boundaries": False},
        "analysis": {"ovito_enabled": False, "methods": []},
    })
    result = build_model(spec)
    assert result.metadata["sro_after"]["engine"] == "sqsgenerator"
    assert result.metadata["sro_after"]["engine_version"]
    assert result.metadata["sro_after"]["total_objective"] <= result.metadata["sro_before"]["total_objective"]
    assert result.atoms.get_chemical_symbols().count("Ni") == 16
    assert result.atoms.get_chemical_symbols().count("Co") == 16


def test_icet_backend_is_real_and_matches_requested_pair_shells():
    pytest.importorskip("icet")
    spec = ModelSpecification.model_validate({
        "random_seed": 3,
        "box": {"lengths": [7.2, 7.2, 7.2], "periodic": [True, True, True]},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Ni": 0.5, "Co": 0.5}}],
        "grains": {"number": 1, "orientations": "identity"},
        "chemistry": {"method": "icet", "steps": 2000,
                      "neighbor_shells": 2, "exclude_boundaries": False},
        "analysis": {"ovito_enabled": False, "methods": []},
    })
    result = build_model(spec)
    report = result.metadata["sro_after"]
    assert report["engine"] == "icet"
    assert report["engine_version"]
    assert report["matched_pair_shells"] == 2
    assert report["total_objective"] <= result.metadata["sro_before"]["total_objective"]
    assert result.atoms.get_chemical_symbols().count("Ni") == 16
    assert result.atoms.get_chemical_symbols().count("Co") == 16


def test_extxyz_output_round_trip_preserves_scientific_arrays(tmp_path):
    result = build_model(simple_spec(grains=3))
    result.write_outputs(tmp_path)
    restored = read(tmp_path / "model.extxyz")
    assert len(restored) == len(result.atoms)
    assert restored.get_chemical_symbols() == result.atoms.get_chemical_symbols()
    for name in ("grain_id", "phase_id", "region_type", "is_grain_boundary",
                 "is_phase_boundary", "is_triple_junction", "primary_region",
                 "local_number_density", "inclusion_id"):
        if result.atoms.arrays[name].dtype.kind == "f":
            assert np.allclose(restored.arrays[name], result.atoms.arrays[name])
        else:
            assert np.array_equal(restored.arrays[name], result.atoms.arrays[name])


def test_all_output_formats_round_trip_and_report_metadata_loss(tmp_path):
    data = simple_spec(grains=1).model_dump(mode="json")
    data["grains"]["orientations"] = "identity"
    data["output"]["formats"] = ["extxyz", "lammps-data", "vasp", "cif"]
    result = build_model(ModelSpecification.model_validate(data))
    result.write_outputs(tmp_path)
    report = json.loads((tmp_path / "output-roundtrip.json").read_text())
    assert all(item["passed"] for item in report.values())
    assert "grain_id" in report["extxyz"]["preserved_scientific_arrays"]
    for format_name in ("lammps-data", "vasp", "cif"):
        assert "grain_id" in report[format_name]["lost_scientific_arrays"]


def test_native_sqs_records_full_history_and_returns_historical_best(tmp_path):
    data = simple_spec(grains=1).model_dump(mode="json")
    data["grains"]["orientations"] = "identity"
    data["chemistry"].update({"method": "native_sqs", "steps": 300,
                              "history_interval": 1, "exclude_boundaries": False})
    result = build_model(ModelSpecification.model_validate(data))
    report = result.metadata["sro_after"]
    history = report["optimization_history"]
    summary = report["optimization_summary"]["phases"][0]
    assert len(history) > 2
    assert report["optimization_summary"]["returned_state"] == "historical_best"
    assert summary["best_objective"] <= summary["terminal_objective"] + 1e-12
    result.write_outputs(tmp_path)
    assert (tmp_path / "optimization-history.csv").exists()


def test_primary_regions_are_exclusive_and_orientation_report_is_complete():
    result = build_model(simple_spec(grains=3))
    primary = result.atoms.arrays["primary_region"]
    assert set(np.unique(primary)).issubset({0, 1, 2, 3, 4})
    counts = result.validation["geometry_quality"]["primary_regions"]
    assert sum(item["count"] for item in counts.values()) == len(result.atoms)
    orientation = result.metadata["orientation_analysis"]
    assert len(orientation["grains"]) == 3
    assert orientation["statistics"]["adjacent_pair_count"] > 0
    assert all("symmetry_reduced_misorientation_deg" in pair
               for pair in orientation["adjacent_grain_pairs"])
    assert "local_number_density" in result.atoms.arrays
    assert result.validation["geometry_quality"]["local_density"]["neighbor_order"] > 0
    grain_quality = result.validation["geometry_quality"]["grain_statistics"]
    assert grain_quality["populated_count"] == 3
    assert grain_quality["empty_grain_ids"] == []
    assert grain_quality["equivalent_diameter"]["mean"] > 0
    cleanup = result.metadata["overlap_cleanup"]
    assert cleanup["strategy"] == "balanced-conflict-graph"
    assert sum(item["deleted"] for item in cleanup["by_grain"].values()) == cleanup["deleted_atoms"]


def test_hcp_commensurate_single_crystal_is_recognized_by_ptm():
    a = 2.95
    c = a * 1.633
    spec = ModelSpecification.model_validate({
        "box": {"lengths": [3 * a, 3 * np.sqrt(3) * a, 3 * c],
                "periodic": [True, True, True]},
        "phases": [{"name": "HCP", "structure": "hcp", "lattice_parameter": a,
                    "c_over_a": 1.633, "fraction": 1, "composition": {"Ti": 1}}],
        "grains": {"number": 1, "orientations": "identity"},
        "analysis": {"ovito_enabled": True, "methods": ["ptm"],
                     "ptm_rmsd_cutoff": 0.15},
    })
    result = build_model(spec)
    assert len(result.atoms) == 108
    assert result.metadata["commensurability"]["strictly_commensurate"] is True
    assert result.metadata["ovito_analysis"]["methods"]["ptm"]["counts"] == {"HCP": 108}


def test_explicit_euler_orientation_and_defects_are_reproducible():
    spec = ModelSpecification.model_validate({
        "random_seed": 9,
        "box": {"lengths": [10.8, 10.8, 10.8], "periodic": [False] * 3},
        "phases": [{"name": "FCC", "structure": "fcc", "lattice_parameter": 3.6,
                    "fraction": 1, "composition": {"Ni": 1}}],
        "grains": {"number": 1, "orientations": "explicit",
                   "euler_angles_deg": [[30, 20, 10]]},
        "defects": {"vacancies": [{"fraction": 0.05, "phase": "FCC"}],
                    "interstitials": [{"element": "C", "position": [0.5, 0.5, 0.5]}]},
        "analysis": {"ovito_enabled": False, "methods": []},
    })
    first = build_model(spec)
    second = build_model(spec)
    assert np.allclose(first.atoms.positions, second.atoms.positions)
    assert first.metadata["defects"]["vacancies"][0]["removed"] > 0
    assert first.atoms.get_chemical_symbols().count("C") == 1


def test_project_directory_round_trip(tmp_path):
    result = build_model(simple_spec())
    result.write_project(tmp_path)
    restored = load_project(tmp_path)
    assert len(restored.atoms) == len(result.atoms)
    assert restored.validation == result.validation
    assert restored.specification.model_dump() == result.specification.model_dump()


def test_custom_cif_template_can_be_used_as_a_phase(tmp_path):
    template = bulk("Si", "diamond", a=5.43, cubic=True)
    template_path = tmp_path / "diamond.cif"
    write(template_path, template)
    spec = ModelSpecification.model_validate({
        "box": {"lengths": [10.86, 10.86, 10.86], "periodic": [True] * 3},
        "phases": [{"name": "diamond", "structure": "custom",
                    "template_path": str(template_path), "lattice_parameter": 5.43,
                    "fraction": 1, "composition": {"Si": 1}}],
        "grains": {"number": 1, "orientations": "identity"},
        "analysis": {"ovito_enabled": False, "methods": []},
    })
    result = build_model(spec)
    assert len(result.atoms) == 64
    assert result.validation["status"] == "PASS"


def test_boundary_segregation_preserves_global_composition():
    data = simple_spec(grains=4).model_dump()
    data["chemistry"]["segregation"] = [{
        "phase": "fcc", "region": "grain_boundary",
        "target_composition": {"Ni": 0.8, "Co": 0.2},
    }]
    spec = ModelSpecification.model_validate(data)
    result = build_model(spec)
    report = result.metadata["segregation"][0]
    assert report["site_count"] > 0
    assert report["achieved"]["Ni"] > 0.5
    symbols = result.atoms.get_chemical_symbols()
    assert abs(symbols.count("Ni") - symbols.count("Co")) <= 1


@pytest.mark.parametrize("shape,size", [
    ("sphere", [5.0, 5.0, 5.0]),
    ("ellipsoid", [5.0, 3.5, 2.5]),
    ("box", [4.0, 3.0, 2.0]),
    ("cylinder", [4.0, 4.0, 3.0]),
    ("slab", [1.0, 1.0, 2.0]),
])
def test_structural_inclusion_replaces_matrix_lattice(shape, size):
    spec = ModelSpecification.model_validate({
        "random_seed": 11,
        "box": {"lengths": [21.6, 21.6, 21.6], "periodic": [True] * 3},
        "phases": [
            {"name": "matrix", "structure": "fcc", "lattice_parameter": 3.6,
             "fraction": 0.9, "composition": {"Ni": 1}},
            {"name": "precipitate", "structure": "bcc", "lattice_parameter": 2.9,
             "fraction": 0.1, "composition": {"Fe": 1}},
        ],
        "grains": {"number": 1, "orientations": "identity"},
        "inclusions": [{"name": "p1", "phase": "precipitate", "shape": shape,
                        "center": [0.5, 0.5, 0.5], "size": size}],
        "cleanup": {"minimum_distance_factor": 0.6, "maximum_deleted_fraction": 0.3},
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
    })
    result = build_model(spec)
    assert result.validation["phase_count"] == 2
    assert np.count_nonzero(result.atoms.arrays["inclusion_id"] == 0) > 0
    assert result.validation["boundary_atoms"] > 0
    assert set(result.atoms.get_chemical_symbols()) == {"Ni", "Fe"}
    assert result.metadata["inclusions"][0]["matrix_atoms_removed"] > 0


def test_nested_inclusions_form_distinct_core_and_shell():
    spec = load_specification("examples/fcc_bcc_hcp_core_shell.yaml")
    result = build_model(spec)
    center = np.asarray(spec.box.lengths) / 2
    delta = result.atoms.positions - center
    box = np.asarray(spec.box.lengths)
    delta -= box * np.round(delta / box)
    radius = np.linalg.norm(delta, axis=1)
    phase = result.atoms.arrays["phase_id"]
    core = phase == 2
    shell = phase == 1
    matrix = phase == 0
    assert np.count_nonzero(core) > 0
    assert np.count_nonzero(shell) > 0
    assert np.count_nonzero(matrix) > 0
    assert np.max(radius[core]) <= 4.0 + 1e-8
    assert np.min(radius[shell]) > 4.0 - 1e-8
    assert np.max(radius[shell]) <= 7.5 + 1e-8
    assert np.min(radius[matrix]) > 7.5 - 1e-8
    assert set(result.atoms.get_chemical_symbols()) == {"Ni", "Fe", "Ti"}
    inclusion_quality = result.validation["geometry_quality"]["inclusions"]
    assert inclusion_quality["outer BCC particle"]["atom_count"] > 0
    assert inclusion_quality["inner HCP core"]["atom_count"] > 0
    assert inclusion_quality["inner HCP core"]["composition"] == {"Ti": 1.0}
    assert inclusion_quality["inner HCP core"]["maximum_radial_extent"] <= 4.0 + 1e-8
    assert len(inclusion_quality["outer BCC particle"]["radial_phase_profile"]) == 8


def test_commensurate_rectangular_cell_has_exact_fcc_geometry():
    spec = load_specification("examples/fcc_rectangular_cell.yaml")
    result = build_model(spec)
    assert np.allclose(result.atoms.cell.lengths(), [14.4, 18.0, 21.6])
    assert len(result.atoms) == 4 * 4 * 5 * 6
    assert result.validation["status"] == "PASS"
    assert result.validation["minimum_distance"] == pytest.approx(3.6 / np.sqrt(2))
    assert result.metadata["commensurability"]["strictly_commensurate"] is True
    assert result.metadata["ovito_analysis"]["methods"]["cna"]["counts"] == {"FCC": 480}
    assert result.metadata["ovito_analysis"]["methods"]["ptm"]["counts"] == {"FCC": 480}


def test_phase_fraction_mismatch_is_reported_with_scientific_default():
    spec = ModelSpecification.model_validate({
        "box": {"lengths": [21.6, 21.6, 21.6], "periodic": [True] * 3},
        "phases": [
            {"name": "matrix", "structure": "fcc", "lattice_parameter": 3.6,
             "fraction": 0.9, "composition": {"Ni": 1}},
            {"name": "precipitate", "structure": "bcc", "lattice_parameter": 2.9,
             "fraction": 0.1, "composition": {"Fe": 1}},
        ],
        "grains": {"number": 1, "orientations": "identity"},
        "inclusions": [{"name": "too-small", "phase": "precipitate", "shape": "sphere",
                        "center": [0.5, 0.5, 0.5], "size": [5, 5, 5]}],
        "analysis": {"ovito_enabled": False, "methods": [], "rdf_enabled": False},
        "cleanup": {"minimum_distance_factor": 0.6, "maximum_deleted_fraction": 0.3},
    })
    result = build_model(spec)
    assert result.validation["status"] == "WARNING"
    assert any("volume-fraction deviation" in warning
               for warning in result.validation["warnings"])


def test_boundary_fraction_is_reported_and_warned_when_excessive():
    result = build_model(simple_spec(grains=4))
    expected = result.validation["boundary_atoms"] / len(result.atoms)
    assert result.validation["boundary_fraction"] == pytest.approx(expected)
    assert expected > result.specification.cleanup.maximum_boundary_fraction
    assert any("Boundary/interface atoms occupy" in warning
               for warning in result.validation["warnings"])


def test_spherical_precipitate_example_matches_fraction_and_geometry():
    path = Path(__file__).parents[1] / "examples" / "fcc_bcc_spherical_precipitate.yaml"
    result = build_model(load_specification(path))
    phase = result.validation["phases"]["BCC precipitate"]
    assert phase["volume_fraction_deviation"] <= 0.02
    mask = result.atoms.arrays["inclusion_id"] == 0
    box = result.atoms.cell.lengths()
    displacement = result.atoms.positions[mask] - box / 2
    displacement -= box * np.round(displacement / box)
    assert np.linalg.norm(displacement, axis=1).max() <= 6.4 + 1e-8
    assert result.validation["status"] == "PASS"


def test_native_sqs_example_improves_sro_and_retains_fcc_lattice():
    path = Path(__file__).parents[1] / "examples" / "fcc_cocrfeni_native_sqs.yaml"
    result = build_model(load_specification(path))
    before = result.metadata["sro_before"]["total_objective"]
    after = result.metadata["sro_after"]["total_objective"]
    assert after <= before
    assert set(result.atoms.get_chemical_symbols()) == {"Co", "Cr", "Fe", "Ni"}
    counts = result.metadata["ovito_analysis"]["methods"]["ptm"]["counts"]
    assert counts == {"FCC": len(result.atoms)}


def test_native_sqs_progress_never_moves_backwards():
    data = simple_spec(grains=1).model_dump()
    data["grains"]["orientations"] = "identity"
    data["chemistry"].update({"method": "native_sqs", "steps": 300,
                               "exclude_boundaries": False})
    data["analysis"].update({"ovito_enabled": False, "methods": [],
                             "rdf_enabled": False})
    fractions = []
    build_model(ModelSpecification.model_validate(data),
                progress_callback=lambda event: fractions.append(event.fraction))
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
