from __future__ import annotations

import time

from alloyforge_web.render_jobs import RenderJobManager


def _wait(manager, render_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = manager.get(render_id)
        if record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.01)
    raise AssertionError("render did not finish")


def test_render_jobs_are_unique_and_complete(monkeypatch, tmp_path):
    import alloyforge.rendering as rendering

    model = tmp_path / "model.extxyz"
    model.write_text("placeholder", encoding="utf-8")

    def fake_render(model, output, **options):
        output.write_bytes(b"png")
        return {"atom_count": 10, "output": str(output), **options}

    monkeypatch.setattr(rendering, "render_with_ovito", fake_render)
    manager = RenderJobManager()
    first = manager.submit(model, tmp_path, {"color_mode": "phase"})
    second = manager.submit(model, tmp_path, {"color_mode": "grain"})
    assert first["render_id"] != second["render_id"]
    assert first["output"] != second["output"]
    assert _wait(manager, first["render_id"])["status"] == "completed"
    assert _wait(manager, second["render_id"])["status"] == "completed"


def test_render_failure_is_reported(monkeypatch, tmp_path):
    import alloyforge.rendering as rendering

    model = tmp_path / "model.extxyz"
    model.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(rendering, "render_with_ovito",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("renderer failed")))
    manager = RenderJobManager()
    record = manager.submit(model, tmp_path, {})
    finished = _wait(manager, record["render_id"])
    assert finished["status"] == "failed"
    assert "renderer failed" in finished["error"]
