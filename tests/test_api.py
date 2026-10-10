"""Offline API tests; no network or model downloads."""

from pathlib import Path

from fastapi.testclient import TestClient

from vsearch.api.app import app
from vsearch.asr.transcript import Segment
from vsearch.index.manifest import Manifest
from vsearch.index.transcripts import TranscriptIndex

client = TestClient(app)


def test_health_and_validation() -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/v1/search", json={"query": ""}).status_code == 422
    assert client.get("/v1/videos/../../etc/passwd/moments").status_code != 200


def test_search_and_html_escaping(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("VSEARCH_DATA_DIR", str(tmp_path))  # type: ignore[attr-defined]
    video_id = "a" * 24
    store = Manifest(tmp_path)
    store.upsert_video(video_id, "demo.mp4", None, video_id, 4, 100, 100, True, "m1")
    store.set_status(video_id, "done")
    TranscriptIndex(tmp_path).replace(video_id, [Segment(1, 3, "<img src=x> laptop")])
    response = client.post("/v1/search", json={"query": "laptop"})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
    html = client.get("/").text
    assert "textContent" in html
    assert "innerHTML" not in html
    assert client.delete(f"/v1/videos/{video_id}").json()["deleted"]
    assert client.post("/v1/search", json={"query": "laptop"}).json()["results"] == []
