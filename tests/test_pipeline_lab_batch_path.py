"""Pipeline Lab batch_path endpoint (no main.py execution)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_LAB_MAX_PARALLEL", "1")
    import pipeline_lab.server.app as lab_app

    monkeypatch.setattr(lab_app, "RUNS_ROOT", tmp_path / "runs")
    lab_app.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    lab_app._runs.clear()
    while not lab_app._job_queue.empty():
        try:
            lab_app._job_queue.get_nowait()
        except Exception:
            break

    def _noop_execute_run(run_id: str) -> None:
        run_dir = lab_app.RUNS_ROOT / run_id
        with lab_app._run_lock:
            st = lab_app._runs.get(run_id)
            if st:
                st.status = "done"
                st.process = None
        (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(lab_app, "_execute_run", _noop_execute_run)

    from fastapi.testclient import TestClient

    return TestClient(lab_app.app)


def test_batch_path_queues_runs_with_shared_inode(client, tmp_path):
    import pipeline_lab.server.app as lab_app

    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"not-a-real-mp4-but-bytes")

    r = client.post(
        "/api/runs/batch_path",
        json={
            "video_path": str(vid),
            "runs": [
                {"recipe_name": "A", "fields": {}},
                {"recipe_name": "B", "fields": {"pose_stride": 2}},
            ],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "queued"
    assert data["run_count"] == 2
    assert len(data["run_ids"]) == 2
    assert data.get("batch_id")

    lab_app._job_queue.join()

    rows = client.get("/api/runs").json()
    by_id = {x["run_id"]: x for x in rows}
    for rid in data["run_ids"]:
        assert rid in by_id
        assert by_id[rid]["batch_id"] == data["batch_id"]

    root = lab_app.RUNS_ROOT
    paths = []
    for rid in data["run_ids"]:
        found = list((root / rid).glob("input_video*"))
        assert len(found) == 1
        paths.append(found[0])
    try:
        assert paths[0].stat().st_ino == paths[1].stat().st_ino
    except AssertionError:
        pytest.skip("hardlink not shared (different FS semantics)")


def test_batch_path_rejects_missing_video(client, tmp_path):
    r = client.post(
        "/api/runs/batch_path",
        json={"video_path": str(tmp_path / "nope.mp4"), "runs": [{"recipe_name": "x", "fields": {}}]},
    )
    assert r.status_code == 400


def test_pipeline_matrix_endpoint(client):
    r = client.get("/api/pipeline_matrix")
    assert r.status_code == 200
    body = r.json()
    assert "recipes" in body
    assert len(body["recipes"]) >= 5
