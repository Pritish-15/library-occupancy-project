"""Demo video upload + occupancy visualization routes."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import load_config, project_path
from src.vision.detect import process_video

router = APIRouter(prefix="/demo", tags=["demo"])

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_BYTES = 40 * 1024 * 1024


def _jobs_dir() -> Path:
    path = project_path("data", "uploads")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _capacity(library_id: str, zone_id: str) -> tuple[int, str, str]:
    cfg = load_config()
    for lib in cfg["libraries"]:
        if lib["id"] != library_id:
            continue
        for zone in lib["zones"]:
            if zone["id"] == zone_id:
                return int(zone["capacity"]), lib["name"], zone["name"]
    raise HTTPException(404, "Unknown library or zone")


def _run_job(job_id: str, src: Path, library_id: str, zone_id: str) -> None:
    dest_dir = _jobs_dir() / job_id
    try:
        capacity, lib_name, zone_name = _capacity(library_id, zone_id)
        summary = process_video(
            src,
            dest_dir / "annotated.mp4",
            capacity=capacity,
            preview_dir=dest_dir / "frames",
        )
        summary.update(
            {
                "library_id": library_id,
                "zone_id": zone_id,
                "library_name": lib_name,
                "zone_name": zone_name,
            }
        )
        (dest_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with LOCK:
            JOBS[job_id].update({"status": "done", "summary": summary})
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update({"status": "error", "error": str(exc)})


@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    library_id: str = Form(...),
    zone_id: str = Form(...),
):
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported format. Use {', '.join(sorted(ALLOWED))}")
    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(400, "File too large (40 MB max for this demo)")
    if not payload:
        raise HTTPException(400, "Empty file")
    _capacity(library_id, zone_id)
    job_id = uuid.uuid4().hex[:12]
    dest_dir = _jobs_dir() / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = dest_dir / f"source{suffix}"
    src.write_bytes(payload)
    with LOCK:
        JOBS[job_id] = {"status": "running", "summary": None, "error": None}
    thread = threading.Thread(target=_run_job, args=(job_id, src, library_id, zone_id), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        summary_path = _jobs_dir() / job_id / "summary.json"
        if summary_path.exists():
            return {"job_id": job_id, "status": "done", "summary": json.loads(summary_path.read_text(encoding="utf-8"))}
        raise HTTPException(404, "Unknown job")
    return {"job_id": job_id, **job}


@router.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    path = _jobs_dir() / job_id / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(path, media_type="video/mp4", filename="library-detection-demo.mp4")


@router.get("/jobs/{job_id}/frames/{name}")
def job_frame(job_id: str, name: str):
    path = (_jobs_dir() / job_id / "frames" / Path(name).name)
    if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(404, "Frame not found")
    return FileResponse(path)
