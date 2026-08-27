from __future__ import annotations

import json
import multiprocessing as mp
import queue
import sqlite3
import threading
import traceback
import uuid
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker(job_id: str, spec_data: dict, output_dir: str, event_queue, cancel_event):
    from alloyforge.builder import build_model
    from alloyforge.config import ModelSpecification
    from alloyforge.progress import BuildCancelled, CancellationToken

    def progress(event):
        event_queue.put({"type": "progress", **event.as_dict()})

    try:
        spec = ModelSpecification.model_validate(spec_data)
        result = build_model(spec, progress, CancellationToken(cancel_event))
        files = result.write_outputs(output_dir)
        event_queue.put(
            {
                "type": "completed",
                "fraction": 1.0,
                "stage": "completed",
                "message": "Model completed",
                "files": [Path(path).name for path in files],
                "summary": result.validation,
            }
        )
    except BuildCancelled:
        event_queue.put({"type": "cancelled", "message": "Task cancelled"})
    except Exception as exc:
        event_queue.put(
            {
                "type": "failed",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


@dataclass
class ActiveJob:
    process: mp.Process
    queue: Any
    cancel_event: Any
    monitor: threading.Thread


class LocalJobManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "jobs.sqlite3"
        self._lock = threading.RLock()
        self._active: dict[str, ActiveJob] = {}
        self._ctx = mp.get_context("spawn")
        self._initialize_database()
        self._mark_interrupted_jobs()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT,
                    config_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, sequence)
                );
                """
            )

    def _mark_interrupted_jobs(self):
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='failed', error='Application stopped during build', updated_at=? "
                "WHERE status IN ('queued','running','cancelling')",
                (_now(),),
            )

    def submit(self, spec_data: dict) -> str:
        from alloyforge.config import ModelSpecification

        spec = ModelSpecification.model_validate(spec_data)
        job_id = uuid.uuid4().hex[:12]
        output_dir = self.root / job_id
        output_dir.mkdir(parents=True)
        (output_dir / "specification.json").write_text(
            json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id, spec.name, "queued", "queued", 0.0, "Waiting to start",
                    json.dumps(spec.model_dump(mode="json")), str(output_dir), _now(), _now(), None,
                ),
            )
        event_queue = self._ctx.Queue()
        cancel_event = self._ctx.Event()
        process = self._ctx.Process(
            target=_worker,
            args=(job_id, spec.model_dump(mode="json"), str(output_dir), event_queue, cancel_event),
            name=f"alloy-model-forge-{job_id}",
        )
        monitor = threading.Thread(
            target=self._monitor, args=(job_id, process, event_queue), daemon=True
        )
        with self._lock:
            self._active[job_id] = ActiveJob(process, event_queue, cancel_event, monitor)
        process.start()
        monitor.start()
        self._update_job(job_id, status="running", message="Worker started")
        return job_id

    def _monitor(self, job_id: str, process: mp.Process, event_queue):
        terminal = False
        while process.is_alive() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            event_type = event.get("type", "progress")
            sequence = self._append_event(job_id, event)
            event["sequence"] = sequence
            if event_type == "completed":
                self._update_job(job_id, status="completed", stage="completed", progress=1.0, message=event.get("message"))
                terminal = True
            elif event_type == "failed":
                self._update_job(job_id, status="failed", message=event.get("message"), error=event.get("traceback"))
                terminal = True
            elif event_type == "cancelled":
                self._update_job(job_id, status="cancelled", message=event.get("message"))
                terminal = True
            else:
                self._update_job(
                    job_id,
                    status="running",
                    stage=event.get("stage"),
                    progress=event.get("fraction", 0.0),
                    message=event.get("message"),
                )
        process.join(timeout=1)
        if not terminal:
            self._append_event(job_id, {"type": "failed", "message": "Worker exited unexpectedly"})
            self._update_job(job_id, status="failed", message="Worker exited unexpectedly")
        with self._lock:
            self._active.pop(job_id, None)

    def _append_event(self, job_id: str, event: dict) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM events WHERE job_id=?", (job_id,)
            ).fetchone()
            sequence = int(row[0])
            payload = {**event, "job_id": job_id, "sequence": sequence}
            connection.execute(
                "INSERT INTO events VALUES (?,?,?,?)",
                (job_id, sequence, json.dumps(payload), _now()),
            )
        return sequence

    def _update_job(self, job_id: str, **fields):
        if not fields:
            return
        fields["updated_at"] = _now()
        assignments = ",".join(f"{key}=?" for key in fields)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?",
                (*fields.values(), job_id),
            )

    def get(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 30) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def events_after(self, job_id: str, sequence: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM events WHERE job_id=? AND sequence>? ORDER BY sequence",
                (job_id, sequence),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            active = self._active.get(job_id)
        if not active:
            return False
        active.cancel_event.set()
        self._update_job(job_id, status="cancelling", message="Cancellation requested")
        return True

    def output_directory(self, job_id: str) -> Path | None:
        job = self.get(job_id)
        return Path(job["output_dir"]) if job else None

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._active:
                return False
        job = self.get(job_id)
        if not job:
            return False
        target = Path(job["output_dir"]).resolve()
        root = self.root.resolve()
        if target.parent != root or target == root:
            raise RuntimeError("Refusing to remove a directory outside the AlloyModelForge job root")
        with self._connect() as connection:
            connection.execute("DELETE FROM events WHERE job_id=?", (job_id,))
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        if target.exists():
            shutil.rmtree(target)
        return True
