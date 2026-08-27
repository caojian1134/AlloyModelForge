from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RenderJobManager:
    """Serialize OVITO scene access and expose non-blocking render status."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovito-render")
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}

    def submit(self, model: Path, output_dir: Path, options: dict) -> dict:
        render_id = uuid.uuid4().hex[:12]
        output = output_dir / f"ovito-{render_id}.png"
        record = {
            "render_id": render_id, "status": "queued", "created_at": _now(),
            "updated_at": _now(), "output": output.name, "options": options,
        }
        with self._lock:
            self._jobs[render_id] = record
        self._executor.submit(self._run, render_id, model, output, options)
        return dict(record)

    def _run(self, render_id: str, model: Path, output: Path, options: dict):
        from alloyforge.rendering import render_with_ovito
        self._update(render_id, status="running")
        try:
            report = render_with_ovito(model, output, **options)
            self._update(render_id, status="completed", report=report)
        except Exception as exc:
            self._update(render_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _update(self, render_id: str, **values):
        with self._lock:
            self._jobs[render_id].update(values, updated_at=_now())

    def get(self, render_id: str) -> dict | None:
        with self._lock:
            value = self._jobs.get(render_id)
            return dict(value) if value else None


render_manager = RenderJobManager()
