"""Build every beginner Web preset through the running HTTP service.

This is an end-to-end scientific smoke test: it submits real jobs, waits for the
builder processes, checks exported structures, preview atoms, and validation
results, then writes a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(base_url: str, path: str, payload: object | None = None) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc


def wait_for_job(base_url: str, job_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request_json(base_url, f"/api/jobs/{job_id}")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.4)
    raise TimeoutError(f"Job {job_id} exceeded {timeout:.0f} seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output", type=Path, default=Path("validation-output/web-e2e/report.json")
    )
    args = parser.parse_args()

    catalog = request_json(args.base_url, "/api/presets")
    results = []
    failures = []
    for preset in catalog:
        started = time.monotonic()
        try:
            specification = request_json(args.base_url, f"/api/presets/{preset['id']}")
            submitted = request_json(
                args.base_url, "/api/jobs", {"specification": specification}
            )
            job_id = submitted["job_id"]
            job = wait_for_job(args.base_url, job_id, args.timeout)
            if job["status"] != "completed":
                raise RuntimeError(job.get("error") or f"job status is {job['status']}")
            files = request_json(args.base_url, f"/api/jobs/{job_id}/files")
            preview = request_json(args.base_url, f"/api/jobs/{job_id}/preview")
            analysis = request_json(args.base_url, f"/api/jobs/{job_id}/analysis")
            required = {"model.extxyz", "metadata.json", "validation.json", "specification.yaml"}
            missing = sorted(required.difference(files))
            validation = analysis.get("validation") or {}
            checks = {
                "required_files": not missing,
                "positive_atom_count": preview.get("total_atoms", 0) > 0,
                "preview_not_empty": preview.get("displayed_atoms", 0) > 0,
                "validation_accepted": validation.get("status") in {"PASS", "WARNING"}
                and not validation.get("errors"),
                "atom_counts_agree": preview.get("total_atoms") == validation.get("atom_count"),
            }
            if not all(checks.values()):
                raise RuntimeError(f"failed checks={checks}, missing={missing}")
            results.append(
                {
                    "preset_id": preset["id"],
                    "name": preset["name_en"],
                    "job_id": job_id,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "atom_count": preview["total_atoms"],
                    "grain_count": validation.get("grain_count"),
                    "phase_count": validation.get("phase_count"),
                    "validation_status": validation.get("status"),
                    "validation_warnings": validation.get("warnings", []),
                    "files": files,
                    "checks": checks,
                }
            )
            print(f"PASS {preset['id']}: {preview['total_atoms']} atoms")
        except Exception as exc:  # keep running to expose all workflow failures
            failures.append({"preset_id": preset["id"], "error": str(exc)})
            print(f"FAIL {preset['id']}: {exc}")

    report = {
        "base_url": args.base_url,
        "preset_count": len(catalog),
        "passed": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {args.output.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
