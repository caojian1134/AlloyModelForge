from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

from alloyforge.rendering import COLOR_MODES, FILTER_MODES, _colors, _filtered_atoms, render_with_ovito


@pytest.fixture()
def annotated_model(tmp_path):
    atoms = bulk("Ni", "fcc", a=3.52, cubic=True).repeat((3, 2, 2))
    count = len(atoms)
    index = np.arange(count)
    atoms.set_array("grain_id", (index % 4).astype(np.int32))
    atoms.set_array("phase_id", (index % 2).astype(np.int32))
    atoms.set_array("inclusion_id", ((index % 3) - 1).astype(np.int32))
    region = np.zeros(count, dtype=np.int32)
    region[index % 4 == 0] = 1
    region[index % 5 == 0] |= 2
    atoms.set_array("region_type", region)
    atoms.set_array("primary_region", np.minimum(region, 4).astype(np.int32))
    atoms.set_array("is_triple_junction", (index % 11 == 0).astype(np.int32))
    atoms.set_array("cna_type", np.ones(count, dtype=np.int32))
    atoms.set_array("ptm_type", np.ones(count, dtype=np.int32))
    path = tmp_path / "rectangular.extxyz"
    write(path, atoms)
    return path, atoms


@pytest.mark.parametrize("mode", sorted(COLOR_MODES))
def test_all_color_modes_produce_rgb_values(annotated_model, mode):
    _, atoms = annotated_model
    colors = _colors(atoms, mode)
    assert colors.shape == (len(atoms), 3)
    assert np.all((colors >= 0) & (colors <= 1))


@pytest.mark.parametrize("mode", sorted(FILTER_MODES))
def test_all_filters_are_defined_and_bounded(annotated_model, mode):
    _, atoms = annotated_model
    selected = _filtered_atoms(atoms, mode)
    assert 0 < len(selected) <= len(atoms)
    if mode == "half-x":
        assert np.all(selected.positions[:, 0] <= atoms.cell.lengths()[0] / 2)


@pytest.mark.ovito_render
def test_real_ovito_tachyon_render_of_rectangular_section(annotated_model, tmp_path):
    pytest.importorskip("ovito")
    model, atoms = annotated_model
    output = tmp_path / "section.png"
    report = render_with_ovito(
        model, output, color_mode="cna", filter_mode="half-x",
        projection="perspective", width=640, height=480, transparent=True,
    )
    assert output.is_file() and output.stat().st_size > 5_000
    assert report["renderer"] == "OVITO Tachyon"
    assert report["atom_count"] < len(atoms)
    from matplotlib import image as mpimg
    image = mpimg.imread(output)
    assert image.shape[:2] == (480, 640)
    assert image.shape[2] == 4


def test_invalid_render_options_fail_explicitly(annotated_model, tmp_path):
    model, _ = annotated_model
    with pytest.raises(ValueError, match="Unknown color mode"):
        render_with_ovito(model, tmp_path / "bad.png", color_mode="invented")
    with pytest.raises(ValueError, match="Image dimensions"):
        render_with_ovito(model, tmp_path / "bad.png", width=100)
