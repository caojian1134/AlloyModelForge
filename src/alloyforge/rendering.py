from __future__ import annotations

import shutil
import subprocess
import sys
import warnings
import json
from pathlib import Path

import numpy as np
from ase.data.colors import jmol_colors
from ase.io import read

from .analysis import STRUCTURE_NAMES, _ovito_compatible_atoms


COLOR_MODES = {"element", "grain", "phase", "inclusion", "region", "cna", "ptm"}
FILTER_MODES = {
    "all", "interior", "boundary", "grain-boundary", "phase-boundary",
    "triple-junction", "half-x", "half-y", "half-z",
}
PROJECTIONS = {"perspective", "orthographic"}
_PALETTE = np.asarray([
    (0.09, .42, .32), (.84, .52, .20), (.36, .47, .71), (.63, .33, .45),
    (.50, .64, .23), (.45, .35, .63), (.71, .30, .25), (.24, .55, .60),
    (.61, .47, .24), (.34, .39, .37),
])
_STRUCTURE_COLORS = {
    0: (.94, .94, .94), 1: (.10, .90, .20), 2: (.92, .05, .20),
    3: (.08, .22, .90), 4: (.66, .30, .72),
}
_REGION_COLORS = {
    0: (.55, .57, .56), 1: (.18, .61, .45), 2: (.88, .64, .22),
    3: (.25, .41, .69), 4: (.63, .33, .61),
}


def _filtered_atoms(atoms, filter_mode: str):
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"Unknown filter mode: {filter_mode}")
    if filter_mode == "all":
        return atoms.copy()
    region = atoms.arrays.get("region_type", np.zeros(len(atoms), dtype=int))
    triple = atoms.arrays.get("is_triple_junction", np.zeros(len(atoms), dtype=int))
    if filter_mode == "interior":
        mask = region == 0
    elif filter_mode == "boundary":
        mask = region > 0
    elif filter_mode == "grain-boundary":
        mask = (region & 1) > 0
    elif filter_mode == "phase-boundary":
        mask = (region & 2) > 0
    elif filter_mode == "triple-junction":
        mask = triple > 0
    else:
        axis = {"half-x": 0, "half-y": 1, "half-z": 2}[filter_mode]
        mask = atoms.positions[:, axis] <= atoms.cell.lengths()[axis] / 2
    return atoms[np.asarray(mask, dtype=bool)]


def _colors(atoms, color_mode: str) -> np.ndarray:
    if color_mode not in COLOR_MODES:
        raise ValueError(f"Unknown color mode: {color_mode}")
    if color_mode == "element":
        return jmol_colors[np.asarray(atoms.numbers, dtype=int)]
    array_name = {
        "grain": "grain_id", "phase": "phase_id", "inclusion": "inclusion_id",
        "region": "primary_region", "cna": "cna_type", "ptm": "ptm_type",
    }[color_mode]
    values = np.asarray(atoms.arrays.get(array_name, np.zeros(len(atoms), dtype=int)), dtype=int)
    if color_mode in {"cna", "ptm"}:
        return np.asarray([_STRUCTURE_COLORS.get(int(v), (.75, .30, .26)) for v in values])
    if color_mode == "region":
        return np.asarray([_REGION_COLORS.get(int(v), (.75, .30, .26)) for v in values])
    return _PALETTE[np.mod(values, len(_PALETTE))]


def render_with_ovito(
    model: str | Path,
    output: str | Path,
    *,
    color_mode: str = "element",
    filter_mode: str = "all",
    projection: str = "orthographic",
    width: int = 1600,
    height: int = 1200,
    transparent: bool = False,
    particle_radius: float = 0.65,
    show_cell: bool = True,
    show_legend: bool = True,
    dpi: int = 300,
) -> dict:
    """Render a complete AlloyModelForge model with OVITO's Tachyon renderer."""
    if projection not in PROJECTIONS:
        raise ValueError(f"Unknown projection: {projection}")
    if not (320 <= width <= 7680 and 320 <= height <= 7680):
        raise ValueError("Image dimensions must be between 320 and 7680 pixels")
    if not (0.05 <= particle_radius <= 5.0):
        raise ValueError("Particle radius must be between 0.05 and 5.0")
    model_path, output_path = Path(model), Path(output)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atoms = _filtered_atoms(read(model_path), filter_mode)
    if not len(atoms):
        raise ValueError(f"Filter '{filter_mode}' selected no atoms")
    colors = _colors(atoms, color_mode)
    atoms = _ovito_compatible_atoms(atoms)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
            from ovito.io.ase import ase_to_ovito
            from ovito.pipeline import Pipeline, StaticSource
            from ovito.vis import TachyonRenderer, Viewport
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"OVITO rendering is unavailable: {exc}") from exc
    data = ase_to_ovito(atoms)
    data.particles_.create_property("Color", data=colors)
    data.particles.vis.radius = float(particle_radius)
    if data.cell is not None:
        data.cell.vis.enabled = show_cell
        data.cell.vis.line_width = 0.1
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    try:
        viewport = Viewport(type=(
            Viewport.Type.Ortho if projection == "orthographic" else Viewport.Type.Perspective
        ))
        camera = {
            "half-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "half-y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            "half-z": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        }.get(filter_mode, ((1.0, -1.0, -0.8), (0.0, 0.0, 1.0)))
        viewport.camera_dir, viewport.camera_up = camera
        viewport.zoom_all(size=(width, height))
        renderer = TachyonRenderer(ambient_occlusion=True, shadows=True, antialiasing=True)
        viewport.render_image(
            filename=str(output_path), size=(width, height), renderer=renderer,
            alpha=transparent, background=(1.0, 1.0, 1.0),
        )
    finally:
        pipeline.remove_from_scene()
    legend = _add_legend(output_path, atoms, color_mode, dpi, model_path) if show_legend else []
    return {
        "output": str(output_path), "atom_count": len(atoms), "color_mode": color_mode,
        "filter_mode": filter_mode, "projection": projection, "width": width,
        "height": height, "transparent": transparent, "renderer": "OVITO Tachyon",
        "show_cell": show_cell, "dpi": dpi, "legend": legend,
    }


def _add_legend(path: Path, atoms, color_mode: str, dpi: int, model_path: Path) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont

    if color_mode == "element":
        values = np.asarray(atoms.get_chemical_symbols())
        entries = [(name, int(np.count_nonzero(values == name)), jmol_colors[atoms.numbers[np.flatnonzero(values == name)[0]]])
                   for name in sorted(set(values))]
    else:
        array_name = {"grain": "grain_id", "phase": "phase_id", "inclusion": "inclusion_id",
                      "region": "primary_region", "cna": "cna_type", "ptm": "ptm_type"}[color_mode]
        values = np.asarray(atoms.arrays.get(array_name, np.zeros(len(atoms), dtype=int)), dtype=int)
        names = STRUCTURE_NAMES if color_mode in {"cna", "ptm"} else {}
        specification_path = model_path.parent / "specification.json"
        yaml_path = model_path.parent / "specification.yaml"
        specification = None
        if specification_path.is_file():
            specification = json.loads(specification_path.read_text(encoding="utf-8"))
        elif yaml_path.is_file():
            import yaml
            specification = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if specification:
            if color_mode == "phase":
                names = {index: phase["name"] for index, phase in enumerate(specification.get("phases", []))}
            elif color_mode == "inclusion":
                names = {-1: "matrix"} | {index: item["name"] for index, item in enumerate(specification.get("inclusions", []))}
        if color_mode == "region":
            names = {0: "interior", 1: "grain boundary", 2: "phase boundary",
                     3: "triple junction", 4: "mixed boundary"}
        entries = []
        for value in sorted(map(int, np.unique(values))):
            color = (_STRUCTURE_COLORS.get(value, (.75, .30, .26)) if color_mode in {"cna", "ptm"}
                     else _REGION_COLORS.get(value, (.75, .30, .26)) if color_mode == "region"
                     else _PALETTE[value % len(_PALETTE)])
            label = names.get(value, f"{color_mode} {value}")
            entries.append((label, int(np.count_nonzero(values == value)), color))
    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=max(11, image.width // 100))
    entries = entries[:18]
    line_height = max(19, image.height // 35)
    panel_width = max(190, image.width // 4)
    panel_height = 18 + line_height * len(entries)
    x0, y0 = image.width - panel_width - 14, 14
    draw.rounded_rectangle((x0, y0, image.width - 14, y0 + panel_height), radius=8,
                           fill=(255, 255, 255, 220), outline=(45, 45, 45, 180))
    total = max(len(atoms), 1)
    result = []
    for index, (label, count, color) in enumerate(entries):
        y = y0 + 10 + index * line_height
        rgb = tuple(int(255 * float(v)) for v in color)
        draw.rectangle((x0 + 10, y + 2, x0 + 24, y + 16), fill=rgb + (255,), outline=(30, 30, 30, 255))
        draw.text((x0 + 31, y), f"{label}: {count} ({count / total:.1%})", fill=(20, 20, 20, 255), font=font)
        result.append({"label": label, "count": count, "fraction": count / total, "rgb": rgb})
    image.save(path, dpi=(dpi, dpi))
    return result


def find_ovito_desktop() -> Path | None:
    candidates = [shutil.which("ovito"), shutil.which("ovito.exe")]
    env_root = Path(sys.prefix)
    candidates += [env_root / "Library" / "bin" / "ovito.exe", env_root / "Scripts" / "ovito.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def open_in_ovito(model: str | Path) -> Path:
    model_path = Path(model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    executable = find_ovito_desktop()
    if executable is None:
        raise RuntimeError("OVITO Desktop executable was not found in the active Conda environment or PATH")
    subprocess.Popen([str(executable), str(model_path)], cwd=str(model_path.parent))
    return executable
