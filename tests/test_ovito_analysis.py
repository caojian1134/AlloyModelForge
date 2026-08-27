from ase.build import bulk
import pytest

from alloyforge import ModelSpecification
from alloyforge.analysis import analyze_with_ovito


def _spec(structure: str, element: str, lattice: float):
    return ModelSpecification.model_validate({
        "phases": [{"name": structure.upper(), "structure": structure,
                    "lattice_parameter": lattice, "fraction": 1,
                    "composition": {element: 1}}],
        "grains": {"number": 1},
        "analysis": {"ovito_enabled": True, "methods": ["cna", "ptm"],
                     "ptm_rmsd_cutoff": 0.15},
    })


@pytest.mark.parametrize("structure,element,lattice,expected,count", [
    ("fcc", "Ni", 3.52, "FCC", 256),
    ("bcc", "Fe", 2.87, "BCC", 128),
])
def test_ovito_cna_and_ptm_identify_perfect_crystals(
    structure, element, lattice, expected, count
):
    atoms = bulk(element, structure, a=lattice, cubic=True).repeat((4, 4, 4))
    report = analyze_with_ovito(atoms, _spec(structure, element, lattice))
    assert report["available"] is True
    assert report["methods"]["cna"]["counts"] == {expected: count}
    assert report["methods"]["ptm"]["counts"] == {expected: count}
    assert "cna_type" in atoms.arrays
    assert "ptm_type" in atoms.arrays
    assert "ptm_rmsd" in atoms.arrays
    assert "ptm_orientation" in atoms.arrays
    assert report["methods"]["ptm"]["mean_rmsd"] < 1e-6
