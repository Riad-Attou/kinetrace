from fastapi.testclient import TestClient

from kinetrace.api import app


def test_health_reports_local_engine() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.2.0"
    assert set(payload["models"]) == {"pose", "hands"}


def test_unknown_job_is_not_found() -> None:
    with TestClient(app) as client:
        response = client.get("/api/jobs/does-not-exist")

    assert response.status_code == 404
