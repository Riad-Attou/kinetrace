from pathlib import Path

from fastapi.testclient import TestClient

from kinetrace.api import app
from kinetrace.jobs import JobRecord, job_store


def test_completed_gemx_job_exposes_soma_retarget_motion(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    result_path = tmp_path / "motion.json"
    bvh_path = tmp_path / "motion.bvh"
    soma_path = tmp_path / "gemx-soma-motion.npz"
    source_path.write_bytes(b"video")
    result_path.write_text("{}", encoding="utf-8")
    bvh_path.write_text("HIERARCHY", encoding="utf-8")
    soma_path.write_bytes(b"soma-npz")

    record = JobRecord(
        id="completed-gemx-with-soma",
        filename="source.mp4",
        source_path=source_path,
        directory=tmp_path,
        engine="gemx",
        status="complete",
        progress=1.0,
        result_path=result_path,
        bvh_path=bvh_path,
        soma_npz_path=soma_path,
    )
    job_store.add(record)

    with TestClient(app) as client:
        job_response = client.get(f"/api/jobs/{record.id}")
        motion_response = client.get(f"/api/jobs/{record.id}/soma")

    assert job_response.status_code == 200
    assert job_response.json()["downloads"]["soma"] == f"/api/jobs/{record.id}/soma"
    assert motion_response.status_code == 200
    assert motion_response.content == b"soma-npz"
    assert "gemx-soma-motion.npz" in motion_response.headers["content-disposition"]


def test_completed_mediapipe_job_has_no_soma_download(tmp_path: Path) -> None:
    record = JobRecord(
        id="completed-mediapipe-without-soma",
        filename="source.mp4",
        source_path=tmp_path / "source.mp4",
        directory=tmp_path,
        engine="mediapipe",
        status="complete",
    )

    assert "soma" not in record.public()["downloads"]


def test_missing_completed_exports_return_not_found(tmp_path: Path) -> None:
    record = JobRecord(
        id="completed-job-with-missing-exports",
        filename="source.mp4",
        source_path=tmp_path / "source.mp4",
        directory=tmp_path,
        status="complete",
    )
    job_store.add(record)

    with TestClient(app) as client:
        result_response = client.get(f"/api/jobs/{record.id}/result")
        bvh_response = client.get(f"/api/jobs/{record.id}/bvh")

    assert result_response.status_code == 404
    assert bvh_response.status_code == 404
