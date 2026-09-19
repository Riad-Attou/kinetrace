from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from kinetrace import __version__
from kinetrace.engines import EngineName, engine_catalog, engine_status
from kinetrace.jobs import JobRecord, job_store
from kinetrace.processor import process_video
from kinetrace.settings import settings

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings.ensure_directories()
    application.state.executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="kinetrace-inference"
    )
    yield
    application.state.executor.shutdown(wait=False, cancel_futures=False)


app = FastAPI(
    title="KineTrace API",
    version=__version__,
    description="Local video-to-skeleton processing service",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "models": {
            "pose": settings.pose_model.exists(),
            "hands": settings.hand_model.exists(),
        },
        "engines": engine_catalog(),
    }


@app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    video: Annotated[UploadFile, File(...)],
    engine: Annotated[EngineName, Form()] = "mediapipe",
) -> dict[str, object]:
    readiness = engine_status(engine)
    if not readiness["available"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{readiness['label']} is not ready. {readiness['reason']}",
        )

    filename = Path(video.filename or "video.mp4").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video extension '{extension or 'none'}'.",
        )

    job_id = uuid4().hex
    job_directory = settings.data_dir / "jobs" / job_id
    job_directory.mkdir(parents=True, exist_ok=False)
    source_path = job_directory / f"source{extension}"
    maximum_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with source_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > maximum_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video exceeds the {settings.max_upload_mb} MB local limit.",
                    )
                destination.write(chunk)
    except Exception:
        source_path.unlink(missing_ok=True)
        job_directory.rmdir()
        raise
    finally:
        await video.close()

    record = JobRecord(
        id=job_id,
        filename=filename,
        source_path=source_path,
        directory=job_directory,
        engine=engine,
    )
    job_store.add(record)
    request.app.state.executor.submit(process_video, job_id, job_store)
    return record.public()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    return _record_or_404(job_id).public()


@app.get("/api/jobs/{job_id}/source")
def get_source(job_id: str) -> FileResponse:
    record = _record_or_404(job_id)
    return FileResponse(record.source_path, filename=record.filename)


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    record = _completed_record(job_id)
    if record.result_path is None or not record.result_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Motion result is unavailable for this job",
        )
    return FileResponse(record.result_path, media_type="application/json", filename="motion.json")


@app.get("/api/jobs/{job_id}/bvh")
def get_bvh(job_id: str) -> FileResponse:
    record = _completed_record(job_id)
    if record.bvh_path is None or not record.bvh_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="BVH export is unavailable for this job",
        )
    return FileResponse(record.bvh_path, media_type="text/plain", filename="motion.bvh")


@app.get("/api/jobs/{job_id}/soma")
def get_soma_motion(job_id: str) -> FileResponse:
    record = _completed_record(job_id)
    if record.soma_npz_path is None or not record.soma_npz_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SOMA retarget motion is unavailable for this job",
        )
    return FileResponse(
        record.soma_npz_path,
        media_type="application/octet-stream",
        filename="gemx-soma-motion.npz",
    )


def _record_or_404(job_id: str) -> JobRecord:
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job")
    return record


def _completed_record(job_id: str) -> JobRecord:
    record = _record_or_404(job_id)
    if record.status != "complete":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is not complete")
    return record
