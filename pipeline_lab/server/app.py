"""
Pipeline Lab API: queue Sway pose runs, stream progress, serve outputs with Range support.

  cd sway_pose_mvp
  pip install -r pipeline_lab/server/requirements.txt
  uvicorn pipeline_lab.server.app:app --reload --host 127.0.0.1 --port 8765

Static UI (after `npm run build` in pipeline_lab/web): set PIPELINE_LAB_WEB_DIST or use ../web/dist.

Model reuse: each queued run starts a new ``main.py`` process, so YOLO/ViTPose are **loaded from disk
into memory every run** (normal for this architecture). ViTPose tries the local Hugging Face cache first
so cached weights do not trigger Hub traffic. For a fully air‑gapped machine after prefetch, run the API
with ``SWAY_OFFLINE=1`` (see ``python -m tools.prefetch_models`` / README).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import sys
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Repo root: sway_pose_mvp/
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "pipeline_lab" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def _merge_lab_request_into_manifest(run_dir: Path) -> None:
    """
    Attach Lab UI field snapshot to run_manifest (main.py does not receive request.json).
    """
    mp = run_dir / "run_manifest.json"
    req = run_dir / "request.json"
    if not mp.is_file() or not req.is_file():
        return
    try:
        with open(req, encoding="utf-8") as f:
            meta = json.load(f)
        fields = meta.get("fields")
        if not isinstance(fields, dict):
            fields = {}
        with open(mp, encoding="utf-8") as f:
            man = json.load(f)
        rcf = man.get("run_context_final")
        if not isinstance(rcf, dict):
            rcf = {}
        rcf["fields"] = fields
        rn = meta.get("recipe_name")
        if isinstance(rn, str) and rn.strip():
            rcf["recipe_name"] = rn.strip()
        man["run_context_final"] = rcf
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return


def _disk_run_status(run_dir: Path) -> str:
    """
    Infer status when the job is not in _runs (API restarted) or for list_runs merge.

    Worker writes params.yaml at the start of _execute_run (before main.py runs), so
    ``params.yaml`` without ``run_manifest.json`` means the pipeline has been dequeued
    and is running or has crashed before writing the manifest — never ``unknown`` for
    a normal in-flight run.
    """
    if (run_dir / "run_manifest.json").is_file():
        return "done"
    if (run_dir / "params.yaml").is_file():
        return "running"
    if (run_dir / "request.json").is_file():
        return "queued"
    return "unknown"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.pipeline_config_schema import (  # noqa: E402
    PIPELINE_PARAM_FIELDS,
    apply_master_locked_post_pose_prune_params,
    apply_master_locked_pre_pose_prune_params,
    apply_master_locked_reid_dedup_params,
    apply_master_locked_smooth_params,
    freeze_lab_subprocess_detection_env,
    freeze_lab_subprocess_hybrid_sam_env,
    freeze_lab_subprocess_phase3_stitch_env,
    freeze_lab_subprocess_pose_env,
    freeze_lab_subprocess_smooth_env,
    schema_payload,
)
from sway.pipeline_matrix_presets import pipeline_matrix_for_api  # noqa: E402


def _default_ui_fields() -> Dict[str, Any]:
    """Schema defaults only (matches web ``defaultsFromSchema``)."""
    out: Dict[str, Any] = {}
    for f in PIPELINE_PARAM_FIELDS:
        if f.get("type") == "info":
            continue
        d = f.get("default")
        if d is None:
            continue
        if f.get("type") == "string" and d == "":
            continue
        out[f["id"]] = d
    return out


def _effective_ui_fields(stored: Any) -> Dict[str, Any]:
    """Merge client overrides on top of schema defaults (drafts often send ``{}`` until Save)."""
    base = _default_ui_fields()
    if not isinstance(stored, dict):
        return dict(base)
    return {**base, **stored}


# Lab UI enum labels -> main.py --pose-model (base|large|huge|rtmpose)
_POSE_MODEL_CLI: Dict[str, str] = {
    "ViTPose-Base": "base",
    "ViTPose-Large": "large",
    "ViTPose-Huge": "huge",
    "RTMPose-L": "rtmpose",
    "Sapiens (ViTPose-Base fallback)": "sapiens",
}


@dataclass
class RunState:
    run_id: str
    status: str = "queued"  # queued | running | done | error | cancelled
    error: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    recipe_name: str = ""
    video_stem: str = ""
    batch_id: str = ""


def _pipeline_subprocess_is_alive(st: Optional[RunState]) -> bool:
    """True only when this server holds a live main.py Popen for the run."""
    if st is None or st.process is None:
        return False
    return st.process.poll() is None


_runs: Dict[str, RunState] = {}
_run_lock = threading.Lock()
_job_queue: "queue.Queue[str]" = queue.Queue()
_max_parallel = int(os.environ.get("PIPELINE_LAB_MAX_PARALLEL", "1"))


def _build_params_yaml(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Map UI field ids to YAML/env-applied keys (includes SWAY_*)."""
    by_id = {f["id"]: f for f in PIPELINE_PARAM_FIELDS}
    out: Dict[str, Any] = {}
    for fid, val in fields.items():
        if val is None or val == "":
            continue
        spec = by_id.get(fid)
        if not spec:
            continue
        binding = spec["binding"]
        if binding == "yaml_pruning_weight":
            key = spec["key"]
            pw = out.setdefault("PRUNING_WEIGHTS", {})
            pw[str(key)] = float(val)
            continue
        if binding != "yaml":
            continue
        key = spec["key"]
        t = spec["type"]
        if t == "bool":
            out[key] = bool(val)
        elif t == "int":
            out[key] = int(val)
        elif t == "float":
            out[key] = float(val)
        else:
            out[key] = val
    apply_master_locked_pre_pose_prune_params(out)
    apply_master_locked_reid_dedup_params(out)
    apply_master_locked_post_pose_prune_params(out)
    apply_master_locked_smooth_params(out)
    return out


def _validate_pipeline_fields(_fields: Dict[str, Any]) -> None:
    """Reserved for server-side validation; tracker backends map to env in _subprocess_env."""


_REID_PRESET_FILES = {
    "osnet_x0_25": REPO_ROOT / "models" / "osnet_x0_25_msmt17.pt",
    "osnet_x1_0": REPO_ROOT / "models" / "osnet_x1_0_msmt17.pt",
}


def _normalize_tracker_technology(raw: Any) -> str:
    """Lab enum + legacy saved runs → ``deep_ocsort`` | ``deep_ocsort_osnet``."""
    legacy = {
        "BoxMOT": "deep_ocsort",
        "BoT-SORT": "deep_ocsort",
        "ByteTrack": "deep_ocsort",
        "OC-SORT": "deep_ocsort",
        "StrongSORT": "deep_ocsort_osnet",
    }
    if raw is None or raw == "":
        return "deep_ocsort"
    if not isinstance(raw, str):
        return "deep_ocsort"
    s = raw.strip()
    if s in legacy:
        return legacy[s]
    if s in ("deep_ocsort", "deep_ocsort_osnet"):
        return s
    return "deep_ocsort"


def _subprocess_env(fields: Dict[str, Any]) -> Dict[str, str]:
    """Extra env vars from schema (binding env) + Lab-only tracker mapping."""
    by_id = {f["id"]: f for f in PIPELINE_PARAM_FIELDS}
    env = os.environ.copy()
    for fid, val in fields.items():
        if val is None or val == "":
            continue
        spec = by_id.get(fid)
        if not spec or spec["binding"] != "env":
            continue
        key = spec["key"]
        t = spec["type"]
        if t == "bool":
            env[key] = "1" if val else "0"
        else:
            env[key] = str(val)
    # Normalize association metric for tracker.py (case-insensitive)
    am = str(env.get("SWAY_BOXMOT_ASSOC_METRIC", "")).strip().lower()
    if am in ("iou", "giou", "diou", "ciou"):
        env["SWAY_BOXMOT_ASSOC_METRIC"] = am
    # tracker_technology → BoxMOT Deep OC-SORT, optional track-time OSNet (see pipeline_config_schema)
    tt = _normalize_tracker_technology(fields.get("tracker_technology"))
    env["SWAY_USE_BOXMOT"] = "1"
    env["SWAY_BOXMOT_TRACKER"] = "deepocsort"
    custom_rw = str(fields.get("sway_boxmot_reid_weights") or "").strip()
    preset = fields.get("sway_boxmot_reid_model")
    if tt == "deep_ocsort_osnet":
        env["SWAY_BOXMOT_REID_ON"] = "1"
        if custom_rw:
            p = Path(custom_rw).expanduser()
            if p.is_file():
                env["SWAY_BOXMOT_REID_WEIGHTS"] = str(p.resolve())
            else:
                env.pop("SWAY_BOXMOT_REID_WEIGHTS", None)
        elif isinstance(preset, str) and preset in _REID_PRESET_FILES:
            pw = _REID_PRESET_FILES[preset]
            if pw.is_file():
                env["SWAY_BOXMOT_REID_WEIGHTS"] = str(pw.resolve())
            else:
                env.pop("SWAY_BOXMOT_REID_WEIGHTS", None)
        else:
            env.pop("SWAY_BOXMOT_REID_WEIGHTS", None)
    else:
        env["SWAY_BOXMOT_REID_ON"] = "0"
        env.pop("SWAY_BOXMOT_REID_WEIGHTS", None)
    # Aligns with global_track_link.py: unset SWAY_GLOBAL_AFLINK = allow neural when weights exist.
    mode = fields.get("sway_global_aflink_mode", "neural_if_available")
    if mode == "force_heuristic":
        env["SWAY_GLOBAL_AFLINK"] = "0"
    else:
        env.pop("SWAY_GLOBAL_AFLINK", None)
    freeze_lab_subprocess_detection_env(env)
    freeze_lab_subprocess_hybrid_sam_env(env)
    freeze_lab_subprocess_phase3_stitch_env(env)
    freeze_lab_subprocess_pose_env(env)
    freeze_lab_subprocess_smooth_env(env)
    return env


def _cli_from_fields(fields: Dict[str, Any]) -> List[str]:
    by_id = {f["id"]: f for f in PIPELINE_PARAM_FIELDS}
    extra: List[str] = []
    for fid, val in fields.items():
        if val is None or val == "":
            continue
        spec = by_id.get(fid)
        if not spec or spec["binding"] != "cli":
            continue
        key = spec["key"]
        if key == "pose_model":
            raw = str(val).strip()
            cli_pose = _POSE_MODEL_CLI.get(raw)
            if cli_pose:
                extra.extend(["--pose-model", cli_pose])
        elif key == "pose_stride":
            extra.extend(["--pose-stride", str(int(val))])
        elif key == "montage" and val:
            extra.append("--montage")
        elif key == "save_phase_previews" and val:
            extra.append("--save-phase-previews")
        elif key == "temporal_pose_refine":
            if val:
                extra.append("--temporal-pose-refine")
            else:
                extra.append("--no-temporal-pose-refine")
        elif key == "temporal_pose_radius":
            extra.extend(["--temporal-pose-radius", str(int(val))])
    return extra


def _worker_loop() -> None:
    while True:
        run_id = _job_queue.get()
        try:
            _execute_run(run_id)
        except Exception as e:  # noqa: BLE001
            with _run_lock:
                st = _runs.get(run_id)
                if st:
                    st.status = "error"
                    st.error = str(e)
        finally:
            _job_queue.task_done()


def _execute_run(run_id: str) -> None:
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        # Run directory removed (e.g. user deleted a queued job before the worker started).
        with _run_lock:
            _runs.pop(run_id, None)
        return
    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    video = run_dir / "input_video"
    # resolve actual extension
    vids = list(run_dir.glob("input_video*"))
    if not vids:
        raise FileNotFoundError("missing input video")
    video_path = vids[0]

    meta_path = run_dir / "request.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    fields = _effective_ui_fields(meta.get("fields"))

    params = _build_params_yaml(fields)
    params_path = run_dir / "params.yaml"
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)

    progress_path = run_dir / "progress.jsonl"
    manifest_path = run_dir / "run_manifest.json"

    cli_extra = _cli_from_fields(fields)
    # Default Lab behavior: previews on unless the client explicitly sets save_phase_previews false
    if fields.get("save_phase_previews") is not False:
        if "--save-phase-previews" not in cli_extra:
            cli_extra.append("--save-phase-previews")
    py_exe = os.environ.get("PIPELINE_LAB_PYTHON", "python3")
    cmd = [
        py_exe,
        "-u",  # unbuffered stdout so Lab console updates during long CPU phases
        str(REPO_ROOT / "main.py"),
        str(video_path),
        "--output-dir",
        str(out_dir),
        "--params",
        str(params_path),
        "--progress-jsonl",
        str(progress_path),
        "--run-manifest",
        str(manifest_path),
    ] + cli_extra

    env = _subprocess_env(fields)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    with _run_lock:
        st = _runs.get(run_id)
        if st is None:
            return
        st.status = "running"

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _run_lock:
        st.process = proc

    log_path = run_dir / "stdout.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
    rc = proc.wait()

    with _run_lock:
        st = _runs.get(run_id)
        if st is None:
            return
        st.process = None
        cancelled = st.stop_requested
        st.stop_requested = False
        if cancelled:
            st.status = "cancelled"
            st.error = None
        elif rc == 0:
            _merge_lab_request_into_manifest(run_dir)
            st.status = "done"
        else:
            st.status = "error"
            st.error = f"exit code {rc}"


def _rehydrate_pending_runs_on_startup() -> int:
    """
    After uvicorn restart, the in-memory queue and _runs are empty but run folders still exist.
    Re-queue any job that never finished (no run_manifest.json) so batches actually drain.
    Stale ``running`` rows (params.yaml but no manifest, no live subprocess) are reset to queued.
    """
    pending: List[Tuple[float, str, Path]] = []
    for p in RUNS_ROOT.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        rid = p.name
        req = p / "request.json"
        if not req.is_file():
            continue
        if (p / "run_manifest.json").is_file():
            continue
        ds = _disk_run_status(p)
        if ds == "unknown":
            continue
        if ds == "running":
            try:
                (p / "params.yaml").unlink(missing_ok=True)
            except OSError:
                pass
            try:
                (p / "progress.jsonl").unlink(missing_ok=True)
            except OSError:
                pass
        try:
            mtime = req.stat().st_mtime
        except OSError:
            continue
        pending.append((mtime, rid, p))
    pending.sort(key=lambda x: (x[0], x[1]))
    requeued = 0
    for mtime, rid, run_dir in pending:
        try:
            with open(run_dir / "request.json", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        recipe_name = str(meta.get("recipe_name") or "")
        video_stem = str(meta.get("video_stem") or "")
        batch_id = str(meta.get("batch_id") or "").strip()
        created = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        with _run_lock:
            if rid in _runs:
                continue
            _runs[rid] = RunState(
                run_id=rid,
                status="queued",
                recipe_name=recipe_name,
                video_stem=video_stem,
                batch_id=batch_id,
                created=created,
            )
        _job_queue.put(rid)
        requeued += 1
    if requeued:
        print(
            f"[pipeline_lab] Rehydrated {requeued} queued run(s) from disk (pending jobs after restart).",
            flush=True,
        )
    return requeued


@asynccontextmanager
async def _lab_lifespan(app: FastAPI) -> AsyncIterator[None]:
    _rehydrate_pending_runs_on_startup()
    yield


for _ in range(max(1, _max_parallel)):
    threading.Thread(target=_worker_loop, daemon=True).start()


app = FastAPI(title="Sway Pipeline Lab", lifespan=_lab_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("PIPELINE_LAB_CORS", "http://127.0.0.1:5173,http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/runs/rehydrate_from_disk")
def rehydrate_from_disk() -> Dict[str, Any]:
    """
    Re-queue unfinished runs after a crash or if the Lab was restarted without draining the queue.
    Safe to call multiple times (skips runs already registered in memory).
    """
    n = _rehydrate_pending_runs_on_startup()
    return {"ok": True, "rehydrated": n}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


@app.get("/api/schema")
def get_schema() -> Dict[str, Any]:
    return schema_payload()


@app.get("/api/pipeline_matrix")
def get_pipeline_matrix() -> Dict[str, Any]:
    """Curated A/B recipes (single-knob deltas) for ``POST /api/runs/batch_path`` and the Lab UI."""
    return pipeline_matrix_for_api()


def _link_or_copy_video(src: Path, dst: Path) -> None:
    """Hardlink when same filesystem (saves space for N-way matrix); else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@app.get("/api/models/status")
def models_status() -> Dict[str, Any]:
    """Which optional weight files exist under models/ (for Lab badges)."""
    md = REPO_ROOT / "models"
    names = (
        "yolo26s.pt",
        "yolo26l.pt",
        "yolo26l_dancetrack.pt",
        "yolo26x.pt",
        "osnet_x0_25_msmt17.pt",
        "osnet_x1_0_msmt17.pt",
        "motionagformer-l-h36m.pth.tr",
        "27_243_45.2.bin",
    )
    out = {n: (md / n).is_file() for n in names}
    mag_root = REPO_ROOT / "vendor" / "MotionAGFormer"
    out["motionagformer_repo"] = (mag_root / "model" / "MotionAGFormer.py").is_file()
    pfv2_root = REPO_ROOT / "vendor" / "PoseFormerV2"
    out["poseformerv2_repo"] = (pfv2_root / "common" / "model_poseformer.py").is_file()
    return out


class CreateRunRequest(BaseModel):
    video_path: str
    recipe_name: str = ""
    fields: Dict[str, Any] = Field(default_factory=dict)


def _enqueue_run(
    run_id: str,
    run_dir: Path,
    recipe_name: str,
    fields: Dict[str, Any],
    video_stem: str,
    meta_extra: Dict[str, Any],
) -> None:
    meta = {
        "recipe_name": recipe_name,
        "fields": _effective_ui_fields(fields),
        "video_stem": video_stem,
        **meta_extra,
    }
    bid = str(meta_extra.get("batch_id") or "").strip()
    if bid:
        meta["batch_id"] = bid
    with open(run_dir / "request.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with _run_lock:
        _runs[run_id] = RunState(
            run_id=run_id,
            recipe_name=recipe_name,
            video_stem=video_stem,
            batch_id=bid,
        )
    _job_queue.put(run_id)


@app.post("/api/runs")
def create_run(req: CreateRunRequest) -> JSONResponse:
    """Queue a run by copying a file that already exists on the server (automation / headless)."""
    try:
        _validate_pipeline_fields(req.fields or {})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    disk_path = req.video_path
    p = Path(disk_path)
    if not p.is_file():
        raise HTTPException(400, "video_path must exist")

    run_id = str(uuid.uuid4())
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".mp4"
    dest = run_dir / f"input_video{ext}"
    shutil.copy2(disk_path, dest)
    _enqueue_run(
        run_id,
        run_dir,
        req.recipe_name,
        req.fields,
        p.stem,
        {"source_path": disk_path},
    )
    return JSONResponse({"run_id": run_id, "status": "queued"})


@app.post("/api/runs/upload")
async def create_run_upload(
    file: UploadFile = File(...),
    recipe_name: str = Form(""),
    fields_json: str = Form("{}"),
) -> JSONResponse:
    """Queue a run from a browser upload (drag-and-drop or file picker)."""
    try:
        fields = json.loads(fields_json or "{}")
        if not isinstance(fields, dict):
            raise ValueError("fields must be a JSON object")
        _validate_pipeline_fields(fields)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(400, f"invalid fields_json: {e}") from e

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    run_id = str(uuid.uuid4())
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    if ext.lower() not in (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    ):
        ext = ".mp4"
    dest = run_dir / f"input_video{ext}"
    dest.write_bytes(raw)

    stem = Path(file.filename or "video").stem
    _enqueue_run(
        run_id,
        run_dir,
        recipe_name,
        fields,
        stem,
        {"upload_filename": file.filename},
    )
    return JSONResponse({"run_id": run_id, "status": "queued"})


class BatchRunSpec(BaseModel):
    recipe_name: str = ""
    fields: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/runs/batch_upload")
async def create_runs_batch_upload(
    file: UploadFile = File(...),
    runs_json: str = Form(...),
) -> JSONResponse:
    """Queue multiple runs that share one uploaded video (same bytes copied per run dir)."""
    try:
        parsed = json.loads(runs_json or "[]")
        if not isinstance(parsed, list):
            raise ValueError("runs_json must be a JSON array")
        specs = [BatchRunSpec.model_validate(x) for x in parsed]
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(400, f"invalid runs_json: {e}") from e
    if not specs:
        raise HTTPException(400, "runs_json must contain at least one run")
    for sp in specs:
        try:
            _validate_pipeline_fields(sp.fields or {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    if ext.lower() not in (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    ):
        ext = ".mp4"
    stem = Path(file.filename or "video").stem
    batch_id = str(uuid.uuid4())
    run_ids: List[str] = []

    for spec in specs:
        run_id = str(uuid.uuid4())
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        dest = run_dir / f"input_video{ext}"
        dest.write_bytes(raw)
        _enqueue_run(
            run_id,
            run_dir,
            spec.recipe_name,
            spec.fields,
            stem,
            {
                "upload_filename": file.filename,
                "batch_id": batch_id,
            },
        )
        run_ids.append(run_id)

    return JSONResponse({"batch_id": batch_id, "run_ids": run_ids, "status": "queued"})


class BatchPathRunSpec(BaseModel):
    recipe_name: str = ""
    fields: Dict[str, Any] = Field(default_factory=dict)


class CreateBatchPathRequest(BaseModel):
    """Queue many runs that share one video copied once into the Lab runs tree (then hardlinked per run)."""

    video_path: str
    runs: List[BatchPathRunSpec]
    source_label: str = ""


@app.post("/api/runs/batch_path")
def create_runs_batch_path(req: CreateBatchPathRequest) -> JSONResponse:
    """
    Automation / CLI: ``video_path`` must exist on the machine running uvicorn.
    Copies the file once under ``runs/_batch_staging/<batch_id>/``, then hardlinks into each run dir
    (fallback: copy) so large matrix batches do not multiply disk usage.
    """
    if not req.runs:
        raise HTTPException(400, "runs must contain at least one spec")
    src = Path(req.video_path)
    if not src.is_file():
        raise HTTPException(400, "video_path must exist")

    for sp in req.runs:
        try:
            _validate_pipeline_fields(sp.fields or {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    batch_id = str(uuid.uuid4())
    ext = src.suffix or ".mp4"
    if ext.lower() not in (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    ):
        ext = ".mp4"

    staging_dir = RUNS_ROOT / "_batch_staging" / batch_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_video = staging_dir / f"source{ext}"
    shutil.copy2(src, staging_video)

    stem = src.stem
    if req.source_label and req.source_label.strip():
        stem = req.source_label.strip()

    run_ids: List[str] = []
    for spec in req.runs:
        run_id = str(uuid.uuid4())
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        dest = run_dir / f"input_video{ext}"
        _link_or_copy_video(staging_video, dest)
        _enqueue_run(
            run_id,
            run_dir,
            spec.recipe_name,
            spec.fields,
            stem,
            {
                "source_path": str(src.resolve()),
                "batch_id": batch_id,
                "batch_kind": "batch_path",
            },
        )
        run_ids.append(run_id)

    if staging_video.is_file():
        staging_video.unlink()
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    return JSONResponse(
        {
            "batch_id": batch_id,
            "run_ids": run_ids,
            "status": "queued",
            "video_stem": stem,
            "run_count": len(run_ids),
        }
    )


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    """Merge in-memory state with runs on disk (survives API restart)."""
    rows: Dict[str, Dict[str, Any]] = {}
    with _run_lock:
        for r in _runs.values():
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "status": r.status,
                "error": r.error,
                "recipe_name": r.recipe_name,
                "video_stem": r.video_stem,
                "created": r.created,
                "batch_id": r.batch_id,
            }
            if r.status == "running":
                row["subprocess_alive"] = _pipeline_subprocess_is_alive(r)
            rows[r.run_id] = row
    for p in RUNS_ROOT.iterdir():
        if not p.is_dir():
            continue
        rid = p.name
        if rid.startswith("_"):
            continue
        if rid in rows:
            continue
        meta = p / "request.json"
        recipe_name = ""
        video_stem = ""
        batch_id_disk = ""
        if meta.is_file():
            try:
                with open(meta, encoding="utf-8") as f:
                    m = json.load(f)
                recipe_name = m.get("recipe_name") or ""
                video_stem = m.get("video_stem") or ""
                batch_id_disk = str(m.get("batch_id") or "").strip()
            except Exception:
                pass
        status = _disk_run_status(p)
        created = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        disk_row: Dict[str, Any] = {
            "run_id": rid,
            "status": status,
            "error": None,
            "recipe_name": recipe_name,
            "video_stem": video_stem,
            "created": created,
            "batch_id": batch_id_disk,
        }
        if status == "running":
            # No in-memory Popen after API restart — Stop will not work; Delete is allowed.
            disk_row["subprocess_alive"] = False
        rows[rid] = disk_row
    return sorted(rows.values(), key=lambda x: (x.get("created") or "", x["run_id"]), reverse=True)


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> JSONResponse:
    """
    Terminate the main.py subprocess for this run (only while the Lab API still holds
    the Popen handle). After an API restart, the orphan process cannot be stopped here.
    """
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run")
    with _run_lock:
        st = _runs.get(run_id)
        if st is None or st.status != "running":
            raise HTTPException(
                409,
                "run is not active in this server process (already finished, queued, or API was restarted). "
                "If status still shows running, the folder is stale — use Delete to remove it, or quit main.py in Activity Monitor.",
            )
        proc = st.process
        if proc is None or proc.poll() is not None:
            raise HTTPException(
                409,
                "no live subprocess for this run (already exited or API state is stale). You can delete the run from the Lab.",
            )
        st.stop_requested = True
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    return JSONResponse({"ok": True, "run_id": run_id, "message": "terminate sent"})


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> Response:
    """Remove run directory from disk and in-memory state. Refuses while the worker is running this job."""
    base = RUNS_ROOT.resolve()
    run_dir = (RUNS_ROOT / run_id).resolve()
    try:
        run_dir.relative_to(base)
    except ValueError:
        raise HTTPException(403, "invalid path") from None
    
    in_memory = False
    with _run_lock:
        st = _runs.get(run_id)
        if st is not None:
            in_memory = True
            if _pipeline_subprocess_is_alive(st):
                raise HTTPException(409, "cannot delete while the pipeline subprocess is still running — stop it first")

    if not run_dir.is_dir() and not in_memory:
        raise HTTPException(404, "unknown run")

    with _run_lock:
        if run_id in _runs:
            del _runs[run_id]

    shutil.rmtree(run_dir, ignore_errors=True)
    return Response(status_code=204)


@app.post("/api/runs/{run_id}/rerun")
def rerun_run(run_id: str) -> JSONResponse:
    """
    Queue a new run that copies the source run's input video and pipeline fields from request.json.
    """
    base = RUNS_ROOT.resolve()
    src = (RUNS_ROOT / run_id).resolve()
    try:
        src.relative_to(base)
    except ValueError:
        raise HTTPException(403, "invalid path") from None
    if not src.is_dir():
        raise HTTPException(404, "unknown run")
    req_path = src / "request.json"
    if not req_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="source run has no request.json (cannot copy configuration)",
        )
    with _run_lock:
        st = _runs.get(run_id)
        if _pipeline_subprocess_is_alive(st):
            raise HTTPException(
                status_code=409,
                detail="wait for this run to finish or stop it before rerunning",
            )
    try:
        meta = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"invalid request.json: {e}") from e

    raw_fields = meta.get("fields")
    fields: Dict[str, Any] = dict(raw_fields) if isinstance(raw_fields, dict) else {}
    try:
        _validate_pipeline_fields(fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    vids = list(src.glob("input_video*"))
    if not vids:
        raise HTTPException(
            status_code=400,
            detail="source run has no input_video file",
        )
    video_src = vids[0]

    new_id = str(uuid.uuid4())
    new_dir = (RUNS_ROOT / new_id).resolve()
    try:
        new_dir.relative_to(base)
    except ValueError:
        raise HTTPException(403, "invalid path") from None
    new_dir.mkdir(parents=True, exist_ok=True)
    ext = video_src.suffix or ".mp4"
    dest = new_dir / f"input_video{ext}"
    shutil.copy2(video_src, dest)

    recipe_name = str(meta.get("recipe_name") or "")
    video_stem = str(meta.get("video_stem") or "").strip() or dest.stem

    _enqueue_run(
        new_id,
        new_dir,
        recipe_name,
        fields,
        video_stem,
        {"rerun_of": run_id},
    )
    return JSONResponse({"run_id": new_id, "status": "queued", "rerun_of": run_id})


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run")
    manifest_path = run_dir / "run_manifest.json"
    with _run_lock:
        st = _runs.get(run_id)
    if st:
        out: Dict[str, Any] = {
            "run_id": st.run_id,
            "status": st.status,
            "error": st.error,
            "recipe_name": st.recipe_name,
            "video_stem": st.video_stem,
            "created": st.created,
        }
        if st.status == "running":
            out["subprocess_alive"] = _pipeline_subprocess_is_alive(st)
    else:
        disk_status = _disk_run_status(run_dir)
        out = {
            "run_id": run_id,
            "status": disk_status,
            "error": None,
            "recipe_name": "",
            "video_stem": "",
            "created": datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        if disk_status == "running":
            out["subprocess_alive"] = False
        meta = run_dir / "request.json"
        if meta.is_file():
            try:
                with open(meta, encoding="utf-8") as f:
                    m = json.load(f)
                out["recipe_name"] = m.get("recipe_name") or ""
                out["video_stem"] = m.get("video_stem") or ""
            except Exception:
                pass
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            out["manifest"] = json.load(f)
    return out


@app.get("/api/runs/{run_id}/config")
def get_run_config(run_id: str) -> Dict[str, Any]:
    """UI fields from request.json plus resolved params.yaml (written when the worker starts)."""
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run")
    req_path = run_dir / "request.json"
    if not req_path.is_file():
        raise HTTPException(404, "request.json not found (run not queued yet)")
    with open(req_path, encoding="utf-8") as f:
        request_meta: Dict[str, Any] = json.load(f)
    merged_fields = _effective_ui_fields(request_meta.get("fields"))
    params_yaml: Optional[Dict[str, Any]] = None
    py = run_dir / "params.yaml"
    if py.is_file():
        try:
            with open(py, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            params_yaml = loaded if isinstance(loaded, dict) else {}
        except Exception:
            params_yaml = None
    return {
        "recipe_name": request_meta.get("recipe_name") or "",
        "video_stem": request_meta.get("video_stem") or "",
        "fields": merged_fields,
        "request_meta": request_meta,
        "params_yaml": params_yaml,
    }


@app.get("/api/runs/{run_id}/progress")
def get_progress(run_id: str) -> List[Dict[str, Any]]:
    p = RUNS_ROOT / run_id / "progress.jsonl"
    if not p.is_file():
        return []
    lines = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return lines


def _tail_log_lines(path: Path, max_lines: int, max_bytes: int = 450_000) -> List[str]:
    """Last max_lines non-empty text lines from a file (read tail up to max_bytes)."""
    if not path.is_file():
        return []
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if len(raw) > max_bytes:
        raw = raw[-max_bytes:]
    text = raw.decode("utf-8", errors="replace")
    parts = text.splitlines()
    out = [ln for ln in parts if ln.strip() != ""]
    if len(out) <= max_lines:
        return out
    return out[-max_lines:]


@app.get("/api/runs/{run_id}/log")
def get_run_log(run_id: str, lines: int = 200) -> Dict[str, Any]:
    """Tail of main.py stdout captured as stdout.log in the run directory."""
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "unknown run")
    n = max(20, min(int(lines), 2500))
    log_path = run_dir / "stdout.log"
    return {"lines": _tail_log_lines(log_path, n), "filename": "stdout.log"}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    """SSE: stream new progress JSON lines (simple poll tail)."""
    progress_path = RUNS_ROOT / run_id / "progress.jsonl"

    async def gen():
        last_size = 0
        rid_dir = RUNS_ROOT / run_id
        while True:
            with _run_lock:
                st = _runs.get(run_id)
            disk_st = _disk_run_status(rid_dir)
            if st:
                status_out = st.status
                terminal = st.status in ("done", "error")
            else:
                status_out = disk_st
                terminal = disk_st == "done"
            if terminal:
                yield f"data: {json.dumps({'status': status_out})}\n\n"
                break

            if not progress_path.is_file():
                yield f"data: {json.dumps({'status': status_out})}\n\n"
                await asyncio.sleep(0.5)
                continue
            sz = progress_path.stat().st_size
            if sz > last_size:
                with open(progress_path, "rb") as f:
                    f.seek(last_size)
                    chunk = f.read().decode("utf-8", errors="replace")
                for line in chunk.splitlines():
                    line = line.strip()
                    if line:
                        yield f"data: {line}\n\n"
                last_size = sz
            yield f"data: {json.dumps({'status': status_out})}\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/pose_3d")
def get_pose_3d(run_id: str) -> Dict[str, Any]:
    """Return data.json['pose_3d'] for the Three.js viewer (404 if missing)."""
    dj = (RUNS_ROOT / run_id / "output" / "data.json").resolve()
    base = (RUNS_ROOT / run_id).resolve()
    try:
        dj.relative_to(base)
    except ValueError:
        raise HTTPException(403, "invalid path") from None
    if not dj.is_file():
        raise HTTPException(status_code=404, detail="data.json not found")
    try:
        data = json.loads(dj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"invalid data.json: {e}") from e
    pose_3d = data.get("pose_3d")
    if not pose_3d:
        raise HTTPException(
            status_code=404,
            detail="pose_3d not in data.json — enable 3D lift (SWAY_3D_LIFT) and ensure MotionAGFormer + weights are installed.",
        )
    return pose_3d


@app.get("/api/runs/{run_id}/file/{path:path}")
def serve_run_file(run_id: str, path: str):
    """Serve files from run directory (output/phase_previews, manifest, etc.)."""
    base = (RUNS_ROOT / run_id).resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(403, "invalid path") from None
    if not target.is_file():
        raise HTTPException(404, "not found")
    media_type = None
    suf = target.suffix.lower()
    if suf == ".mp4":
        media_type = "video/mp4"
    elif suf == ".webm":
        media_type = "video/webm"
    elif suf == ".mov":
        media_type = "video/quicktime"
    elif suf == ".json":
        media_type = "application/json"
    return FileResponse(target, filename=target.name, media_type=media_type)


web_dist = os.environ.get("PIPELINE_LAB_WEB_DIST")
if web_dist and Path(web_dist).is_dir():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="ui")
