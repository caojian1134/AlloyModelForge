from __future__ import annotations

import asyncio
import argparse
import json
import os
import threading
import webbrowser
import shutil
from importlib.resources import files
from pathlib import Path
import yaml

from ase.io import read
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from alloyforge import ModelSpecification, estimate_model

from .jobs import LocalJobManager
from .render_jobs import render_manager


def _data_root() -> Path:
    configured = os.environ.get("ALLOYMODELFORGE_DATA_DIR") or os.environ.get(
        "ALLOYFORGE_DATA_DIR"
    )
    return Path(configured) if configured else Path.home() / ".alloymodelforge" / "jobs"


manager = LocalJobManager(_data_root())
app = FastAPI(title="AlloyModelForge Web", version="0.2.0")
static_dir = Path(str(files("alloyforge_web").joinpath("static")))
project_root = Path(__file__).resolve().parents[2]
PRESETS = {
    "single_fcc": ("入门：FCC 单晶", "Beginner: FCC single crystal", "fcc_cocrfeni_native_sqs.yaml"),
    "rectangular_fcc": ("标准长方体 FCC", "Standard rectangular FCC", "fcc_rectangular_cell.yaml"),
    "polycrystal": ("FCC 多晶", "FCC polycrystal", "fcc_polycrystal.yaml"),
    "dual_phase": ("FCC/BCC 双相", "FCC/BCC dual phase", "fcc_bcc_dual_phase.yaml"),
    "precipitate": ("球形析出相", "Spherical precipitate", "fcc_bcc_spherical_precipitate.yaml"),
    "core_shell": ("核壳颗粒", "Core-shell particle", "fcc_bcc_hcp_core_shell.yaml"),
    "region_sqs": ("区域约束 SQS", "Region-constrained SQS", "fcc_polycrystal_region_sqs.yaml"),
    "directional_sqs": ("方向性/三体 SQS", "Directional/triplet SQS", "fcc_bcc_hcp_core_shell_directional_sqs.yaml"),
}


@app.middleware("http")
async def prevent_stale_frontend(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


class JobRequest(BaseModel):
    specification: dict


class SpecificationTextRequest(BaseModel):
    text: str


class OvitoRenderRequest(BaseModel):
    color_mode: str = "element"
    filter_mode: str = "all"
    projection: str = "orthographic"
    width: int = Field(1600, ge=320, le=7680)
    height: int = Field(1200, ge=320, le=7680)
    transparent: bool = False
    particle_radius: float = Field(0.65, ge=0.05, le=5.0)
    show_cell: bool = True
    show_legend: bool = True
    dpi: int = Field(300, ge=72, le=1200)


@app.post("/api/specification/parse")
def parse_specification(request: SpecificationTextRequest):
    try:
        data = yaml.safe_load(request.text)
        if isinstance(data, dict) and "specification" in data:
            data = data["specification"]
        return ModelSpecification.model_validate(data).model_dump(mode="json")
    except (yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        detail = (
            exc.errors(include_url=False, include_context=False)
            if isinstance(exc, ValidationError) else str(exc)
        )
        raise HTTPException(422, detail=detail) from exc


@app.post("/api/specification/export", response_class=PlainTextResponse)
def export_specification(request: JobRequest):
    try:
        spec = ModelSpecification.model_validate(request.specification)
    except ValidationError as exc:
        raise HTTPException(
            422, detail=exc.errors(include_url=False, include_context=False)
        ) from exc
    return yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False)


@app.get("/api/health")
def health():
    usage = shutil.disk_usage(_data_root())
    return {"status": "ok", "disk_free_bytes": usage.free, "data_root": str(_data_root())}


@app.get("/api/presets")
def list_presets():
    return [{"id": key, "name_zh": value[0], "name_en": value[1]}
            for key, value in PRESETS.items()]


@app.get("/api/presets/{preset_id}")
def get_preset(preset_id: str):
    preset = PRESETS.get(preset_id)
    if not preset:
        raise HTTPException(404, "Preset not found")
    path = project_root / "examples" / preset[2]
    try:
        return ModelSpecification.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        ).model_dump(mode="json")
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise HTTPException(500, f"Preset is invalid: {preset_id}") from exc


@app.post("/api/estimate")
def estimate(request: JobRequest):
    try:
        spec = ModelSpecification.model_validate(request.specification)
    except ValidationError as exc:
        raise HTTPException(
            422, detail=exc.errors(include_url=False, include_context=False)
        ) from exc
    return estimate_model(spec)


@app.post("/api/jobs")
def create_job(request: JobRequest):
    try:
        spec = ModelSpecification.model_validate(request.specification)
        estimate = estimate_model(spec)
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors(include_url=False, include_context=False)) from exc
    maximum_atoms = int(
        os.environ.get("ALLOYMODELFORGE_MAX_ATOMS")
        or os.environ.get("ALLOYFORGE_MAX_ATOMS", "5000000")
    )
    if estimate["estimated_atoms"] > maximum_atoms:
        raise HTTPException(413, f"Estimated atom count exceeds configured limit ({maximum_atoms:,})")
    required = max(100_000_000, int(estimate["estimated_memory_mb"] * 1024 * 1024 * 3))
    if shutil.disk_usage(_data_root()).free < required:
        raise HTTPException(507, "Insufficient free disk space for this model")
    job_id = manager.submit(request.specification)
    return {"job_id": job_id, "status": "queued", "websocket_url": f"/ws/jobs/{job_id}"}


@app.get("/api/jobs")
def list_jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(409, "Job is not running")
    return {"status": "cancelling"}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if not manager.delete(job_id):
        raise HTTPException(409, "Job is running or does not exist")
    return {"status": "deleted", "recoverable": False}


@app.get("/api/jobs/{job_id}/files")
def list_files(job_id: str):
    directory = manager.output_directory(job_id)
    if not directory:
        raise HTTPException(404, "Job not found")
    return [p.name for p in directory.iterdir() if p.is_file() and p.name != "jobs.sqlite3"]


@app.get("/api/jobs/{job_id}/files/{filename}")
def download_file(job_id: str, filename: str):
    directory = manager.output_directory(job_id)
    if not directory:
        raise HTTPException(404, "Job not found")
    path = (directory / Path(filename).name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=path.name)


@app.get("/api/jobs/{job_id}/preview")
def preview(job_id: str, limit: int = 60_000):
    directory = manager.output_directory(job_id)
    path = directory / "model.extxyz" if directory else None
    if not path or not path.exists():
        raise HTTPException(404, "Preview is not available")
    atoms = read(path)
    count = len(atoms)
    if count > limit:
        region = atoms.arrays.get("region_type")
        boundary = list(map(int, (region.nonzero()[0] if region is not None else [])))
        budget = max(0, limit - min(len(boundary), limit // 2))
        rng = __import__("numpy").random.default_rng(42)
        remaining = __import__("numpy").setdiff1d(__import__("numpy").arange(count), boundary)
        sampled = rng.choice(remaining, min(budget, len(remaining)), replace=False)
        indices = __import__("numpy").unique(__import__("numpy").concatenate([boundary[: limit // 2], sampled]))
        atoms = atoms[indices]
    grain = atoms.arrays.get("grain_id", [0] * len(atoms))
    phase = atoms.arrays.get("phase_id", [0] * len(atoms))
    region = atoms.arrays.get("region_type", [0] * len(atoms))
    primary = atoms.arrays.get("primary_region", [0] * len(atoms))
    triple = atoms.arrays.get("is_triple_junction", [0] * len(atoms))
    inclusion = atoms.arrays.get("inclusion_id", [-1] * len(atoms))
    cna = atoms.arrays.get("cna_type", [0] * len(atoms))
    ptm = atoms.arrays.get("ptm_type", [0] * len(atoms))
    return {
        "total_atoms": count,
        "displayed_atoms": len(atoms),
        "box": atoms.cell.lengths().tolist(),
        "atoms": [
            {
                "elem": atom.symbol, "x": atom.x, "y": atom.y, "z": atom.z,
                "grain": int(grain[i]), "phase": int(phase[i]), "region": int(region[i]),
                "primary": int(primary[i]), "triple": int(triple[i]),
                "inclusion": int(inclusion[i]),
                "cna": int(cna[i]), "ptm": int(ptm[i]),
            }
            for i, atom in enumerate(atoms)
        ],
    }


@app.get("/api/jobs/{job_id}/analysis")
def analysis_report(job_id: str):
    directory = manager.output_directory(job_id)
    path = directory / "metadata.json" if directory else None
    if not path or not path.exists():
        raise HTTPException(404, "Analysis report is not available")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return {
        "sro_before": metadata.get("sro_before"),
        "sro_after": metadata.get("sro_after"),
        "ovito": metadata.get("ovito_analysis"),
        "validation": metadata.get("validation"),
        "commensurability": metadata.get("commensurability"),
        "defects": metadata.get("defects"),
    }


@app.post("/api/jobs/{job_id}/render-ovito")
def render_ovito(job_id: str, request: OvitoRenderRequest):
    directory = manager.output_directory(job_id)
    model = directory / "model.extxyz" if directory else None
    if not model or not model.exists():
        raise HTTPException(404, "Model is not available")
    options = {
        "color_mode": request.color_mode, "filter_mode": request.filter_mode,
        "projection": request.projection, "width": request.width, "height": request.height,
        "transparent": request.transparent, "particle_radius": request.particle_radius,
        "show_cell": request.show_cell, "show_legend": request.show_legend, "dpi": request.dpi,
    }
    render = render_manager.submit(model, directory, options)
    return {**render, "status_url": f"/api/jobs/{job_id}/renders/{render['render_id']}"}


@app.get("/api/jobs/{job_id}/renders/{render_id}")
def render_status(job_id: str, render_id: str):
    directory = manager.output_directory(job_id)
    if not directory:
        raise HTTPException(404, "Job not found")
    render = render_manager.get(render_id)
    if not render:
        raise HTTPException(404, "Render job not found")
    result = dict(render)
    if result["status"] == "completed":
        result["url"] = f"/api/jobs/{job_id}/files/{result['output']}"
    return result


@app.post("/api/jobs/{job_id}/open-ovito")
def open_ovito(job_id: str):
    from alloyforge.rendering import open_in_ovito

    directory = manager.output_directory(job_id)
    model = directory / "model.extxyz" if directory else None
    if not model or not model.exists():
        raise HTTPException(404, "Model is not available")
    try:
        executable = open_in_ovito(model)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"status": "opened", "executable": str(executable)}


@app.websocket("/ws/jobs/{job_id}")
async def job_socket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if not manager.get(job_id):
        await websocket.send_json({"type": "failed", "message": "Job not found"})
        await websocket.close(code=4404)
        return
    sequence = 0
    try:
        while True:
            events = manager.events_after(job_id, sequence)
            for event in events:
                sequence = event["sequence"]
                await websocket.send_json(event)
            job = manager.get(job_id)
            if job and job["status"] in {"completed", "failed", "cancelled"} and not events:
                await websocket.send_json({"type": "snapshot", "job": job, "sequence": sequence})
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def run_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    import uvicorn

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run("alloyforge_web.app:app", host=host, port=port, workers=1)


def main():
    parser = argparse.ArgumentParser(description="Run the local AlloyModelForge web application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open the system browser automatically",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
