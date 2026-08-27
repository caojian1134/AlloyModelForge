import pytest

from alloyforge.config import ModelSpecification


def test_rejects_invalid_phase_fraction():
    with pytest.raises(ValueError):
        ModelSpecification.model_validate(
            {
                "phases": [
                    {
                        "name": "fcc",
                        "structure": "fcc",
                        "lattice_parameter": 3.6,
                        "fraction": 0.8,
                        "composition": {"Ni": 1.0},
                    }
                ]
            }
        )


def test_rejects_more_phases_than_grains():
    with pytest.raises(ValueError, match="at least the number of matrix phases"):
        ModelSpecification.model_validate({
            "phases": [
                {"name": "A", "structure": "fcc", "lattice_parameter": 3.6,
                 "fraction": 0.5, "composition": {"Ni": 1}},
                {"name": "B", "structure": "bcc", "lattice_parameter": 2.9,
                 "fraction": 0.5, "composition": {"Fe": 1}},
            ],
            "grains": {"number": 1},
        })
