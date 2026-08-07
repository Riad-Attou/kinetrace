from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from kinetrace.engines import EngineName

JobStatus = Literal["queued", "processing", "complete", "failed"]


@dataclass(slots=True)
class JobRecord:
    id: str
    filename: str
    source_path: Path
    directory: Path
    engine: EngineName = "mediapipe"
    status: JobStatus = "queued"
    progress: float = 0.0
    stage: str = "Queued"
    error: str | None = None
    result_path: Path | None = None
    bvh_path: Path | None = None
    created_at: str = ""

    def public(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("source_path")
        payload.pop("directory")
        payload.pop("result_path")
        payload.pop("bvh_path")
        payload["downloads"] = (
            {
                "json": f"/api/jobs/{self.id}/result",
                "bvh": f"/api/jobs/{self.id}/bvh",
                "source": f"/api/jobs/{self.id}/source",
            }
            if self.status == "complete"
            else {"source": f"/api/jobs/{self.id}/source"}
        )
        return payload


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = RLock()

    def add(self, record: JobRecord) -> None:
        with self._lock:
            if not record.created_at:
                record.created_at = datetime.now(UTC).isoformat()
            self._jobs[record.id] = record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for field, value in changes.items():
                setattr(record, field, value)


job_store = JobStore()
