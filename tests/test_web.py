import sys
from pathlib import Path

from fastapi.testclient import TestClient

from alloyforge_web.app import app


def test_web_entrypoint_honors_host_port_and_browser_arguments(monkeypatch):
    import alloyforge_web.app as web_app

    called = {}
    monkeypatch.setattr(web_app, "run_server", lambda **kwargs: called.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["alloy-model-forge-web", "--host", "127.0.0.1",
                                      "--port", "8766", "--no-browser"])
    web_app.main()
    assert called == {"host": "127.0.0.1", "port": 8766, "open_browser": False}


def test_health_and_home():
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        response = client.get("/")
        assert response.status_code == 200
        assert "AlloyModelForge" in response.text


def test_project_specification_parse_and_export_round_trip():
    text = """
phases:
  - {name: matrix, structure: fcc, lattice_parameter: 3.6, fraction: 0.9,
     composition: {Ni: 1.0}}
  - {name: precipitate, structure: bcc, lattice_parameter: 2.9, fraction: 0.1,
     composition: {Fe: 1.0}}
grains: {number: 1, orientations: identity}
inclusions:
  - {name: p1, phase: precipitate, shape: sphere, center: [0.5, 0.5, 0.5],
     size: [5, 5, 5]}
"""
    with TestClient(app) as client:
        parsed = client.post("/api/specification/parse", json={"text": text})
        assert parsed.status_code == 200
        assert parsed.json()["inclusions"][0]["shape"] == "sphere"
        exported = client.post(
            "/api/specification/export", json={"specification": parsed.json()}
        )
        assert exported.status_code == 200
        reparsed = client.post("/api/specification/parse", json={"text": exported.text})
        assert reparsed.status_code == 200
        assert reparsed.json() == parsed.json()


def test_beginner_presets_cover_basic_and_advanced_structures():
    with TestClient(app) as client:
        catalog = client.get("/api/presets")
        assert catalog.status_code == 200
        ids = {item["id"] for item in catalog.json()}
        assert {"single_fcc", "rectangular_fcc", "polycrystal", "dual_phase",
                "precipitate", "core_shell", "region_sqs", "directional_sqs"} <= ids
        for preset_id in ids:
            response = client.get(f"/api/presets/{preset_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["phases"]
            assert sum(phase["fraction"] for phase in data["phases"]) == 1.0
        advanced = client.get("/api/presets/directional_sqs").json()
        assert advanced["chemistry"]["directional_sro_constraints"]
        assert advanced["chemistry"]["triplet_constraints"]


def test_web_ovito_render_endpoint_uses_complete_model(monkeypatch, tmp_path):
    import alloyforge_web.app as web_app

    (tmp_path / "model.extxyz").write_text("model placeholder", encoding="utf-8")
    monkeypatch.setattr(web_app.manager, "output_directory", lambda job_id: tmp_path)
    captured = {}
    def fake_submit(model, output_dir, options):
        captured.update(model=Path(model), output_dir=Path(output_dir), **options)
        return {"render_id": "render123", "status": "queued", "output": "ovito-render123.png"}
    monkeypatch.setattr(web_app.render_manager, "submit", fake_submit)
    with TestClient(app) as client:
        response = client.post("/api/jobs/example/render-ovito", json={
            "color_mode": "phase", "filter_mode": "half-z",
            "projection": "orthographic", "width": 800, "height": 600,
        })
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["render_id"] == "render123"
    assert captured["model"] == tmp_path / "model.extxyz"
    assert captured["color_mode"] == "phase"
    assert captured["filter_mode"] == "half-z"
