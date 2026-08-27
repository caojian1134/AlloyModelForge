from __future__ import annotations

import json
from pathlib import Path

import typer

from ase.io import read
from ase.visualize import view as ase_view

from .analysis import analyze_with_ovito
from .builder import build_model, estimate_model, load_project, validate_model
from .config import load_specification
from .rendering import open_in_ovito, render_with_ovito

app = typer.Typer(
    help="High-throughput construction and validation of atomistic alloy models for MD and DFT"
)


@app.command()
def build(config: Path, output: Path = Path("output")):
    spec = load_specification(config)
    result = build_model(spec, lambda event: typer.echo(f"[{event.fraction:5.1%}] {event.message}"))
    paths = result.write_outputs(output)
    typer.echo(f"Validation: {result.validation['status']}")
    for path in paths:
        typer.echo(str(path))


@app.command()
def estimate(config: Path):
    typer.echo(json.dumps(estimate_model(load_specification(config)), indent=2))


@app.command("validate")
def validate_command(model: Path, config: Path):
    """Validate an existing model against its AlloyModelForge YAML specification."""
    atoms = read(model)
    report = validate_model(atoms, load_specification(config), deleted_atoms=0)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "FAIL":
        raise typer.Exit(code=1)


@app.command("analyze")
def analyze_command(model: Path, config: Path):
    """Run optional OVITO CNA/PTM analysis on an existing model."""
    atoms = read(model)
    report = analyze_with_ovito(atoms, load_specification(config))
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("view")
def view_command(model: Path):
    """Open an existing structure in ASE's installed graphical viewer."""
    ase_view(read(model))


@app.command("render-ovito")
def render_ovito_command(
    model: Path,
    output: Path = Path("ovito-render.png"),
    color_by: str = "element",
    filter_by: str = "all",
    projection: str = "orthographic",
    width: int = 1600,
    height: int = 1200,
    transparent: bool = False,
    particle_radius: float = 0.65,
    show_cell: bool = True,
    show_legend: bool = True,
    dpi: int = 300,
):
    """Create a publication-quality image using OVITO's Tachyon renderer."""
    report = render_with_ovito(
        model, output, color_mode=color_by, filter_mode=filter_by,
        projection=projection, width=width, height=height,
        transparent=transparent, particle_radius=particle_radius,
        show_cell=show_cell, show_legend=show_legend, dpi=dpi,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("open-ovito")
def open_ovito_command(model: Path):
    """Open a model in OVITO Desktop when the executable is installed."""
    executable = open_in_ovito(model)
    typer.echo(f"Opened with {executable}")


@app.command("project-info")
def project_info(directory: Path):
    """Inspect a saved AlloyModelForge project directory."""
    project = load_project(directory)
    typer.echo(json.dumps({
        "name": project.specification.name,
        "atoms": len(project.atoms),
        "validation": project.validation,
    }, ensure_ascii=False, indent=2))


@app.command()
def web(host: str = "127.0.0.1", port: int = 8765):
    from alloyforge_web.app import run_server

    run_server(host, port, open_browser=True)


if __name__ == "__main__":
    app()
