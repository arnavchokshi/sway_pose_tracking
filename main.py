"""
Sway Pose Tracking V3.4 — Main Orchestrator

Canonical plan: docs/FINAL_OPTIMIZED_PIPELINE.md

Pipeline order (prints [1/11]…[11/11]) matches doc Phases 1–10 plus export:
  [1/11]–[2/11]  Phase 1 Detection + Phase 2 Tracking (one streaming pass)
  [3/11]  Phase 3 Post-track stitching (dormant, stitch, coalesce, merge)
  [4/11]  Phase 4 Pre-pose pruning
  [5/11]  Phase 5 Pose (ViTPose)
  [6/11]  Phase 6 Association (occlusion re-ID, crossover, acceleration audit)
  [7/11]  Phase 7 Collision cleanup (keypoint dedup, bbox sanitize)
  [8/11]  Phase 8 Post-pose pruning (Tier A/B/C)
  [9/11]  Phase 9 Temporal smoothing (1 Euro)
  [10/11] Phase 10 Spatio-temporal scoring
  [11/11] Export & visualization
Hybrid SAM handoff: docs/HYBRID_SAM_PIPELINE_HANDOFF.txt
"""
# Suppress protobuf/onnx MessageFactory GetPrototype noise (non-fatal)
import sys
import warnings

import _repo_path  # noqa: F401 — adds repo root to sys.path for `sway` package


class _FilterStderr:
    """Filter repetitive AttributeError from protobuf/onnx that doesn't affect execution."""
    def __init__(self, stream):
        self.stream = stream
        self._buf = ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line += "\n"
            if "GetPrototype" in line:
                continue
            self.stream.write(line)
    def flush(self):
        if self._buf:
            if "GetPrototype" not in self._buf:
                self.stream.write(self._buf)
            self._buf = ""
        self.stream.flush()


sys.stderr = _FilterStderr(sys.stderr)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

import json
import logging
import os
import queue
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import torch

from sway.server_runtime_perf import apply_server_runtime_perf

apply_server_runtime_perf()

from sway.bidirectional_track_merge import (
    bidirectional_iou_threshold,
    bidirectional_min_match_frames,
    bidirectional_track_pass_enabled,
    merge_forward_backward_tracks,
    remap_reverse_pass_timeline,
    reverse_video_via_ffmpeg,
)
from sway.experimental_hooks import (
    maybe_gnn_refine_raw_tracks,
    write_hmr_mesh_sidecar_json,
)
from sway.checkpoint_io import (
    CHECKPOINT_BOUNDARIES,
    load_after_phase_3,
    load_after_phase_4,
    load_after_phase_5,
    load_after_phase_8,
    load_phase1_yolo_dets,
    read_manifest,
    save_after_phase_3,
    save_after_phase_4,
    save_after_phase_5,
    save_after_phase_8,
    save_final_marker,
    save_live_phase3_bundle,
    save_phase1_yolo_dets,
    validate_manifest,
)
from sway.mot_format import build_phase3_tracking_data_json
from sway.phase_debug_log import PhaseDebugLogger
from sway.tracker import (
    _use_boxmot,
    apply_post_track_stitching,
    boxmot_tracker_kind_from_env,
    iter_video_frames,
    run_boxmot_tracking_from_yolo_dets,
    run_phase1_yolo_only_boxmot,
    run_tracking_before_post_stitch,
)
from sway.track_stats_export import compute_track_quality_stats, write_track_stats_json
from sway.track_pruning import (
    compute_confirmed_human_set,
    compute_phase7_voting_prune_set,
    EDGE_MARGIN_FRAC,
    EDGE_PRESENCE_FRAC,
    log_pruned_tracks,
    prune_cause_config,
    prune_ultra_low_skeleton_tracks,
    PRUNING_WEIGHTS,
    PRUNE_THRESHOLD,
    raw_tracks_to_per_frame,
    ULTRA_LOW_SKELETON_FRAME_FRAC,
    ULTRA_LOW_SKELETON_MEAN,
)
from sway.interp_utils import blend_pose_keypoints_scores
from sway.pose_crop_temporal import apply_temporal_pose_crop
from sway.pose_estimator import (
    PoseEstimator,
    smart_expand_bbox_xyxy,
    vitpose_debug_enabled,
    vitpose_smart_pad_enabled,
)


def _resolve_sapiens_torchscript_path() -> Optional[str]:
    """If ``SWAY_SAPIENS_TORCHSCRIPT`` points to a file, return its resolved path string."""
    raw = os.environ.get("SWAY_SAPIENS_TORCHSCRIPT", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return str(p.resolve()) if p.is_file() else None


def _model_id_for_sapiens_choice(args: Any) -> str:
    """ViTPose manifest id for pose_model=sapiens (native TorchScript path or HF ViTPose-Base id)."""
    if str(getattr(args, "pose_model", "") or "") != "sapiens":
        return "usyd-community/vitpose-plus-base"
    ck = _resolve_sapiens_torchscript_path()
    if ck:
        return f"sapiens-torchscript:{ck}"
    return "usyd-community/vitpose-plus-base"
from sway.temporal_pose_refine import (
    apply_temporal_keypoint_smoothing,
    temporal_pose_radius,
    want_temporal_pose_refine,
)
from sway.crossover import (
    COLLISION_CENTER_DIST_FRAC,
    COLLISION_KPT_DIST_FRAC,
    DEDUP_ANTIPARTNER_MIN_IOU,
    DEDUP_KPT_TIGHT_FRAC,
    DEDUP_MIN_PAIR_OKS,
    DEDUP_TORSO_MEDIAN_FRAC,
    apply_crossover_refinement,
    apply_acceleration_audit,
    apply_occlusion_reid,
    compute_visibility_scores,
    deduplicate_collocated_poses,
    sanitize_pose_bbox_consistency,
)
from sway.reid_embedder import extract_embeddings
from sway.pipeline_config_schema import (
    apply_master_locked_detection_env,
    apply_master_locked_hybrid_sam_env,
    apply_master_locked_phase3_stitch_env,
    apply_master_locked_pose_env,
    apply_master_locked_post_pose_prune_params,
    apply_master_locked_pre_pose_prune_params,
    apply_master_locked_reid_dedup_params,
    apply_master_locked_smooth_env,
    apply_master_locked_smooth_params,
)
from sway.smoother import PoseSmoother
from sway.scoring import process_all_frames_scoring_vectorized
from sway.pipeline_env_params import apply_sway_params_to_env
from sway.phase4_replay import run_pre_pose_prune_phase
from sway.prune_preview_overlay import (
    PRE_POSE_PREVIEW_RULES,
    COLLISION_PREVIEW_RULES,
    POST_POSE_PREVIEW_RULES,
    build_prune_overlay_index,
    wrap_draw_fn_with_prune_overlays,
)
from sway.visualizer import (
    build_dropped_pose_overlay,
    build_pruned_overlay_for_review,
    render_and_export,
    render_phase_clip,
    snapshot_tid_box_map,
    stitch_montage,
    draw_boxes_only,
    draw_phase1_detection_preview,
    draw_tracks_post_stitch_preview,
    draw_frame_with_boxes,
    draw_frame,
)


def _apply_stack_default_env() -> None:
    """
    Production stack defaults for group-dance runs. Unset is OK: setdefault only fills missing keys.
    Override with env or params YAML (SWAY_* applied after this).
    """
    os.environ.setdefault("SWAY_GROUP_VIDEO", "1")
    os.environ.setdefault("SWAY_GLOBAL_LINK", "1")
    # Depth-based stage polygon is easy to get wrong on group/competition video; opt-in via SWAY_AUTO_STAGE_DEPTH=1.
    os.environ.setdefault("SWAY_AUTO_STAGE_DEPTH", "0")


def _ensure_collision_cleanup_logging() -> None:
    """
    Collision cleanup mutates poses without prune_log for limb keypoint zeros.
    Emit sway.crossover INFO on stderr during the pipeline (root logger defaults to WARNING).
    """
    log = logging.getLogger("sway.crossover")
    if getattr(log, "_sway_pipeline_collision_handler", None) is not None:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("  [collision] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    log._sway_pipeline_collision_handler = h  # type: ignore[attr-defined]


def _pose_gap_interp_runtime(
    gap_interp_mode: Optional[str] = None,
    gsi_lengthscale: Optional[float] = None,
) -> Tuple[str, float]:
    """Read optional pose-stride gap interpolation; default linear (unchanged pipeline)."""
    mode = gap_interp_mode
    if mode is None:
        mode = os.environ.get("SWAY_POSE_GAP_INTERP_MODE", "linear").strip().lower() or "linear"
    gsl = gsi_lengthscale
    if gsl is None:
        pgl = os.environ.get("SWAY_POSE_GSI_LENGTHSCALE", "").strip()
        if pgl:
            gsl = float(pgl)
        else:
            g = os.environ.get("SWAY_GSI_LENGTHSCALE", "").strip()
            gsl = float(g) if g else 0.35
    return str(mode), float(gsl)


def _interpolate_pose_gaps(
    raw_poses_by_frame: list,
    frames_stored: list,
    stride: int,
    *,
    gap_interp_mode: Optional[str] = None,
    gsi_lengthscale: Optional[float] = None,
) -> None:
    """Fill pose gaps for frames skipped by pose_stride. Modifies raw_poses_by_frame in place."""
    mode, gsi_l = _pose_gap_interp_runtime(gap_interp_mode, gsi_lengthscale)
    n = len(raw_poses_by_frame)
    for idx in range(n):
        if idx % stride == 0:
            continue
        prev_idx = (idx // stride) * stride
        next_idx = (idx // stride + 1) * stride
        prev_poses = raw_poses_by_frame[prev_idx]
        next_poses = raw_poses_by_frame[next_idx] if next_idx < n else prev_poses
        track_ids = frames_stored[idx][3] if frames_stored[idx] else []
        if not track_ids:
            continue
        interpolated = {}
        denom = next_idx - prev_idx
        t = (idx - prev_idx) / denom if denom > 0 else 1.0
        for tid in track_ids:
            if tid in prev_poses and tid in next_poses:
                kp_prev = prev_poses[tid]["keypoints"]
                kp_next = next_poses[tid]["keypoints"]
                sc_prev = prev_poses[tid]["scores"]
                sc_next = next_poses[tid]["scores"]
                kp, sc = blend_pose_keypoints_scores(
                    kp_prev, kp_next, sc_prev, sc_next, float(t), mode=mode, gsi_l=gsi_l
                )
                interpolated[int(tid)] = {"keypoints": kp, "scores": sc}
            elif tid in prev_poses:
                interpolated[int(tid)] = dict(prev_poses[tid])
            elif tid in next_poses:
                interpolated[int(tid)] = dict(next_poses[tid])
        raw_poses_by_frame[idx] = interpolated


def _parse_stage_polygon_env() -> Optional[List[Tuple[float, float]]]:
    """Normalized [0,1] vertices from SWAY_STAGE_POLYGON JSON, e.g. [[0.1,0.2],[0.9,0.2],[0.9,0.95],[0.1,0.95]]."""
    raw = os.environ.get("SWAY_STAGE_POLYGON", "").strip()
    if not raw:
        return None
    try:
        pts = json.loads(raw)
        if not isinstance(pts, list) or len(pts) < 3:
            print("  Warning: SWAY_STAGE_POLYGON must be a JSON array of at least 3 [x,y] pairs; ignoring.")
            return None
        return [(float(p[0]), float(p[1])) for p in pts]
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        print("  Warning: SWAY_STAGE_POLYGON invalid JSON; ignoring.")
        return None


def get_device() -> torch.device:
    """Select compute device for ViTPose and shared sway inference.

    Override with ``SWAY_TORCH_DEVICE``: ``cpu``, ``mps``, or ``cuda`` / ``cuda:0``.
    Default: MPS on Apple Silicon when available, else CUDA when available (servers / Lambda),
    else CPU.
    """
    import os

    raw = os.environ.get("SWAY_TORCH_DEVICE", "").strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if raw.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda:0" if raw == "cuda" else raw)
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


_LAB_CTX: Optional[Dict[str, Any]] = None
# Serialize progress.jsonl appends (main thread + pose busy-heartbeat thread).
_LAB_PROGRESS_IO_LOCK = threading.Lock()


def _lab_track_summary_heuristic(raw_tracks: Dict[int, List[Any]], ystride: int) -> Dict[str, Any]:
    """Cheap post-stitch stats for the Lab (no ground truth — not TrackEval IDF1/HOTA)."""
    from sway.track_observation import coerce_observation

    if not raw_tracks:
        return {
            "track_count": 0,
            "median_track_observations": 0.0,
            "mean_track_observations": 0.0,
            "internal_timeline_jumps": 0,
            "note": "IDF1, IDSW, HOTA require MOT ground truth (see python -m tools.benchmark_trackeval).",
        }
    lens: List[int] = []
    jumps = 0
    ys = max(1, int(ystride))
    thresh = max(ys * 3, 5)
    for _tid, entries in raw_tracks.items():
        if not entries:
            continue
        frames = sorted({int(coerce_observation(e).frame_idx) for e in entries})
        lens.append(len(frames))
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] > thresh:
                jumps += 1
    arr = np.asarray(lens, dtype=np.float64)
    return {
        "track_count": int(len(raw_tracks)),
        "median_track_observations": float(np.median(arr)) if arr.size else 0.0,
        "mean_track_observations": float(np.mean(arr)) if arr.size else 0.0,
        "internal_timeline_jumps": int(jumps),
        "note": "IDF1, IDSW, HOTA require MOT ground truth (see python -m tools.benchmark_trackeval).",
    }


def _merge_hybrid_sam_stats(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Sum counters across forward + reverse tracking passes (bidirectional)."""
    if not b:
        return dict(a) if a else {}
    if not a:
        return dict(b)
    out = dict(a)
    for k in (
        "hybrid_sam_frames_total",
        "hybrid_sam_frames_refined",
        "hybrid_sam_predict_calls",
        "hybrid_sam_weak_cue_skips",
    ):
        out[k] = int(a.get(k) or 0) + int(b.get(k) or 0)
    out["hybrid_sam_weak_cues_enabled"] = bool(
        a.get("hybrid_sam_weak_cues_enabled") or b.get("hybrid_sam_weak_cues_enabled")
    )
    return out


def _pose_gap_interp_mode_read() -> str:
    return os.environ.get("SWAY_POSE_GAP_INTERP_MODE", "linear").strip().lower() or "linear"


def _build_pipeline_diagnostics(
    hybrid_sam_stats: Dict[str, Any],
    args: Any,
    params: dict,
) -> Dict[str, Any]:
    """Structured Lab / manifest payload: what ran and how optional features behaved."""
    from sway.global_track_link import _force_heuristic_global_link, resolve_aflink_weights
    from sway.tracker import load_tracking_runtime

    tr = load_tracking_runtime()
    p = resolve_aflink_weights()
    gl = os.environ.get("SWAY_GLOBAL_LINK", "").lower() in ("1", "true", "yes")
    force_h = _force_heuristic_global_link()
    weights_ok = p.is_file()
    if not gl:
        af_eff = "disabled"
    elif weights_ok and not force_h:
        af_eff = "neural"
    else:
        af_eff = "heuristic"
    pgl = os.environ.get("SWAY_POSE_GSI_LENGTHSCALE", "").strip()
    gsi = float(pgl) if pgl else float(os.environ.get("SWAY_GSI_LENGTHSCALE", "0.35") or 0.35)
    vis_gsi_v = os.environ.get("SWAY_VIS_GSI_LENGTHSCALE", "").strip()
    vis_gsi_eff = float(vis_gsi_v) if vis_gsi_v else gsi
    vis_mode = os.environ.get("SWAY_VIS_TEMPORAL_INTERP_MODE", "linear").strip().lower() or "linear"
    h = dict(hybrid_sam_stats) if hybrid_sam_stats else {}
    total = max(1, int(h.get("hybrid_sam_frames_total") or 0))
    refined = int(h.get("hybrid_sam_frames_refined") or 0)
    h["hybrid_sam_refined_frac"] = round(refined / total, 4) if total else 0.0
    weak_sk = int(h.get("hybrid_sam_weak_cue_skips") or 0)
    h["hybrid_sam_weak_cue_skip_note"] = (
        f"{weak_sk} high-IoU frames skipped SAM (weak-cue gate)."
        if weak_sk
        else "No weak-cue SAM skips this run."
    )
    stage_wall: Dict[str, Any] = {"per_stage_elapsed_s": {}, "sum_stages_s": 0.0, "count": 0}
    if _LAB_CTX is not None:
        slog = _LAB_CTX.get("stage_log") or []
        per: Dict[str, float] = {}
        for e in slog:
            sk = e.get("stage_key")
            es = e.get("elapsed_s")
            if sk is not None and es is not None:
                per[str(sk)] = float(es)
        stage_wall = {
            "per_stage_elapsed_s": per,
            "sum_stages_s": round(sum(per.values()), 3) if per else 0.0,
            "count": len(per),
        }
    tp = "botsort"
    if _use_boxmot():
        kind = boxmot_tracker_kind_from_env()
        reid_on = os.environ.get("SWAY_BOXMOT_REID_ON", "").strip().lower() in ("1", "true", "yes")
        if kind == "deepocsort" and reid_on:
            tp = "boxmot:deepocsort+osnet"
        else:
            tp = f"boxmot:{kind}"
    _sapiens_cli = str(getattr(args, "pose_model", "") or "") == "sapiens"
    _sapiens_ck = _resolve_sapiens_torchscript_path() if _sapiens_cli else None
    exp = {
        "gnn_track_refine": os.environ.get("SWAY_GNN_TRACK_REFINE", "").strip().lower()
        in ("1", "true", "yes"),
        "sapiens_pose_cli": _sapiens_cli,
        "sapiens_native_torchscript": bool(_sapiens_ck),
        "hmr_mesh_sidecar": os.environ.get("SWAY_HMR_MESH_SIDECAR", "").strip().lower()
        in ("1", "true", "yes"),
    }
    return {
        "hybrid_sam": h,
        "tracker_path": tp,
        "interpolation": {
            "box_mode": str(tr.get("box_interp_mode", "linear")),
            "box_gsi_lengthscale": float(tr.get("gsi_lengthscale", 0.35)),
            "pose_stride_gap_mode": _pose_gap_interp_mode_read(),
            "pose_stride_cli": int(getattr(args, "pose_stride", 1)),
            "export_video_temporal": vis_mode,
            "export_gsi_lengthscale_effective": vis_gsi_eff,
        },
        "global_stitch": {
            "enabled": gl,
            "aflink_effective": af_eff,
            "aflink_weights_found": weights_ok,
            "aflink_weights_path": str(p.resolve()) if weights_ok else None,
        },
        "bidirectional_track_pass": bidirectional_track_pass_enabled(),
        "temporal_pose_refine": want_temporal_pose_refine(getattr(args, "temporal_pose_refine", False)),
        "temporal_pose_radius": temporal_pose_radius(getattr(args, "temporal_pose_radius", 2)),
        "export_track_stats_json": True,
        "run_quality": {
            **stage_wall,
            "track_stats_json_requested": True,
            "note": "per_stage_elapsed_s sums major Lab progress stages for this video run (not pytest golden_bench).",
        },
        "experimental": exp,
        "notes": {
            "gsi": "GSI modes only change interpolated geometry (boxes / pose gaps / export MP4), not discrete JSON keyframes.",
            "aflink": "Neural AFLink runs only when global link is on, weights exist, and Lab does not force heuristic.",
        },
    }


def _lab_init(
    *,
    progress_jsonl: Optional[str],
    run_manifest_path: Optional[str],
) -> None:
    """Pipeline Lab: optional progress JSONL and run manifest targets."""
    global _LAB_CTX
    if not progress_jsonl and not run_manifest_path:
        _LAB_CTX = None
        return
    import time
    from pathlib import Path as P
    _LAB_CTX = {
        "progress_jsonl": progress_jsonl,
        "run_manifest_path": P(run_manifest_path) if run_manifest_path else None,
        "previews": [],
        "stage_log": [],
        "perf_t0": time.perf_counter(),
        "run_context": {},
    }


def _lab_update_context(**kwargs: Any) -> None:
    """Merge key/value run metadata into progress JSONL lines and final manifest."""
    if _LAB_CTX is None:
        return
    ctx = _LAB_CTX.setdefault("run_context", {})
    for k, v in kwargs.items():
        if v is not None:
            if isinstance(v, float):
                ctx[k] = round(v, 6) if abs(v) < 1e10 else v
            elif isinstance(v, (list, dict)) and len(str(v)) > 2000:
                ctx[k] = f"<omitted len={len(v)}>"
            else:
                ctx[k] = v


def _lab_progress(
    stage: int,
    stage_key: str,
    label: str,
    *,
    status: str = "done",
    preview_relpath: Optional[str] = None,
    elapsed_s: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if _LAB_CTX is None:
        return
    import time
    cumulative = time.perf_counter() - float(_LAB_CTX["perf_t0"])
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "cumulative_elapsed_s": round(cumulative, 3),
        "stage": stage,
        "stage_key": stage_key,
        "label": label,
        "status": status,
    }
    if preview_relpath:
        payload["preview_relpath"] = preview_relpath
    if elapsed_s is not None:
        payload["elapsed_s"] = round(elapsed_s, 3)
    rc = _LAB_CTX.get("run_context") or {}
    if rc:
        payload["meta"] = dict(rc)
    if extra:
        payload["extra"] = extra
    # Heartbeats (status=running) go to progress.jsonl only — avoid bloating manifest stage_log.
    if status != "running":
        _LAB_CTX.setdefault("stage_log", []).append(payload)
    pj = _LAB_CTX.get("progress_jsonl")
    if pj:
        line = json.dumps(payload) + "\n"
        with _LAB_PROGRESS_IO_LOCK:
            with open(pj, "a", encoding="utf-8") as f:
                f.write(line)


_T_pose = TypeVar("_T_pose")


def _call_with_pose_busy_heartbeat(
    *,
    fn: Callable[[], _T_pose],
    t0_phase5: float,
    frame_idx: int,
    total_frames: int,
    phase5_pose_passes: int,
    n_tracks: int,
    pct: int,
) -> _T_pose:
    """
    Run ViTPose in a worker thread while the main thread emits heartbeats every ~hb seconds.

    The previous design ran ``fn`` on the main thread and used a second thread for heartbeats.
    During CPU-bound HuggingFace processor work, the main thread holds the GIL, so the heartbeat
    thread never ran — the Lab UI stayed at ~0% with no new lines for minutes (looks hung).

    Inverting roles fixes that: the main thread ``join(timeout=hb)`` loops and prints between
    waits; ViTPose runs on the worker thread.
    """
    import time as _time

    hb = float(os.environ.get("SWAY_POSE_BUSY_HEARTBEAT_SEC", "5"))
    if hb <= 0:
        return fn()

    slot: list[Any] = []
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            slot.append(fn())
        except BaseException as e:
            err.append(e)

    th = threading.Thread(target=_worker, name="pose_forward", daemon=False)
    th.start()
    while True:
        th.join(timeout=hb)
        if not th.is_alive():
            break
        elapsed = _time.time() - t0_phase5
        print(
            f"  [phase5] ViTPose still running — frame {frame_idx + 1}/{total_frames} "
            f"({pct}%), {elapsed:.0f}s in phase 5 (CPU preprocess or GPU; first batch can take minutes)…",
            flush=True,
        )
        _lab_progress(
            4,
            "pose",
            "Phase 5: Pose estimation",
            status="running",
            extra={
                "frame": int(frame_idx + 1),
                "total_frames": int(total_frames),
                "pct": int(pct),
                "pose_pass": int(phase5_pose_passes),
                "tracks": int(n_tracks),
                "wall_s_in_phase": round(elapsed, 1),
                "vitpose_busy": True,
            },
        )
    if err:
        raise err[0]
    if not slot:
        raise RuntimeError("pose forward worker exited without result")
    return slot[0]


def _lab_register_preview(stage_key: str, rel_path: str) -> None:
    if _LAB_CTX is None:
        return
    _LAB_CTX.setdefault("previews", []).append({"stage_key": stage_key, "relpath": rel_path})


def _lab_make_tracking_progress_callback(video_path_str: str):
    """Throttled Lab heartbeats during Phases 1–2 (YOLO + tracking)."""
    import time as _time

    if _LAB_CTX is None or not _LAB_CTX.get("progress_jsonl"):
        return None
    cap = cv2.VideoCapture(video_path_str)
    total_guess = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if total_guess < 1:
        total_guess = 0
    t_phase0 = _time.perf_counter()
    # Same clock for t_phase0 and "now" — mixing perf_counter with time.time() produced
    # wall_s_in_phase ≈ Unix epoch (billions of seconds).
    state: Dict[str, Any] = {"last_wall": t_phase0, "last_pct": -1}

    def _cb(frame_idx: int, infer_n: int, tracks_n: int) -> None:
        now = _time.perf_counter()
        wall = now - t_phase0
        ex: Dict[str, Any] = {
            "yolo_passes": int(infer_n),
            "tracks": int(tracks_n),
            "wall_s_in_phase": round(wall, 1),
            "frame": int(frame_idx + 1),
        }
        tf = total_guess
        pct: Optional[int] = None
        if tf > 0:
            pct = min(100, max(0, int(100 * (frame_idx + 1) / tf)))
            ex["total_frames"] = int(tf)
            ex["pct"] = int(pct)
        wall_dt = now - float(state["last_wall"])
        last_pct = int(state["last_pct"])
        pct_milestone = pct is not None and (pct >= last_pct + 5 or pct >= 99)
        pass_milestone = infer_n == 1 or (infer_n % 25 == 0)
        time_hb = wall_dt >= 20.0
        if not (pct_milestone or pass_milestone or time_hb):
            return
        if pct is not None and pct_milestone:
            state["last_pct"] = int(pct)
        state["last_wall"] = now
        _lab_progress(
            1,
            "phases_1_2",
            "Phases 1–2: Detection & tracking",
            status="running",
            extra=ex,
        )

    return _cb


def _lab_phase4_tick(t0: float, step: str, pct_approx: int, state: Dict[str, Any]) -> None:
    if _LAB_CTX is None or not _LAB_CTX.get("progress_jsonl"):
        return
    import time as _time

    now = _time.time()
    if now - float(state.get("last_emit", 0)) < 6.0 and pct_approx < 95:
        return
    state["last_emit"] = now
    _lab_progress(
        3,
        "pre_pose_prune",
        "Phase 4: Pre-pose pruning",
        status="running",
        extra={
            "step": step,
            "pct": int(pct_approx),
            "wall_s_in_phase": round(now - t0, 1),
        },
    )


def _prune_events_extra(prune_log_entries: list, start_i: int, max_items: int = 400) -> Dict[str, Any]:
    """Slice of prune_log_entries for progress.jsonl (cap size for browser JSON)."""
    chunk = prune_log_entries[start_i:]
    if not chunk:
        return {}
    if len(chunk) > max_items:
        return {
            "prune_events": chunk[:max_items],
            "prune_events_truncated": True,
            "prune_events_total": len(chunk),
        }
    return {"prune_events": chunk}


def _lab_write_manifest(
    *,
    video_path,
    output_dir,
    args,
    params: dict,
    model_id: str,
    total_elapsed: float,
    final_video_relpath: str,
    view_variants: Optional[Dict[str, str]] = None,
) -> None:
    if _LAB_CTX is None or _LAB_CTX.get("run_manifest_path") is None:
        return
    import subprocess
    from pathlib import Path as P

    git_sha = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(P(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            git_sha = r.stdout.strip()
    except Exception:
        pass

    env_keys = [
        "SWAY_YOLO_WEIGHTS",
        "SWAY_USE_BOXMOT",
        "SWAY_VITPOSE_MODEL",
        "SWAY_TRACKER_YAML",
        "SWAY_GROUP_VIDEO",
        "SWAY_GLOBAL_LINK",
        "SWAY_CHUNK_SIZE",
        "SWAY_DETECT_SIZE",
        "SWAY_YOLO_CONF",
        "SWAY_YOLO_DETECTION_STRIDE",
        "SWAY_YOLO_INFER_BATCH",
        "SWAY_YOLO_HALF",
        "SWAY_YOLO_ENGINE",
        "SWAY_VITPOSE_MAX_PER_FORWARD",
        "SWAY_VITPOSE_FP32",
        "SWAY_SAPIENS_TORCHSCRIPT",
        "SWAY_BIDIRECTIONAL_TRACK_PASS",
        "SWAY_BIDIRECTIONAL_IOU_THRESH",
        "SWAY_BIDIRECTIONAL_MIN_MATCH_FRAMES",
        "SWAY_STAGE_POLYGON",
        "SWAY_AUTO_STAGE_DEPTH",
        "SWAY_TEMPORAL_POSE_REFINE",
        "SWAY_TEMPORAL_POSE_RADIUS",
        "SWAY_3D_LIFT",
        "SWAY_LIFT_BACKEND",
        "SWAY_MOTIONAGFORMER_ROOT",
        "SWAY_MOTIONAGFORMER_WEIGHTS",
        "SWAY_POSEFORMERV2_ROOT",
        "SWAY_POSEFORMERV2_WEIGHTS",
        "SWAY_POSEFORMERV2_NFRAMES",
        "SWAY_POSEFORMERV2_FRAME_KEPT",
        "SWAY_POSEFORMERV2_COEFF_KEPT",
        "SWAY_BOX_INTERP_MODE",
        "SWAY_GSI_LENGTHSCALE",
        "SWAY_POSE_GAP_INTERP_MODE",
        "SWAY_POSE_GSI_LENGTHSCALE",
        "SWAY_HYBRID_SAM_WEAK_CUES",
        "SWAY_HYBRID_WEAK_CONF_DELTA",
        "SWAY_HYBRID_WEAK_HEIGHT_FRAC",
        "SWAY_HYBRID_WEAK_MATCH_IOU",
        "SWAY_VIS_TEMPORAL_INTERP_MODE",
        "SWAY_VIS_GSI_LENGTHSCALE",
        "SWAY_AFLINK_THR_T0",
        "SWAY_AFLINK_THR_T1",
        "SWAY_AFLINK_THR_S",
        "SWAY_AFLINK_THR_P",
        "SWAY_BOXMOT_TRACKER",
        "SWAY_GNN_TRACK_REFINE",
        "SWAY_HMR_MESH_SIDECAR",
    ]
    env_snap = {k: os.environ.get(k) for k in env_keys if os.environ.get(k)}

    manifest: Dict[str, Any] = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "vitpose_model_id": model_id,
        "cli": {
            "pose_model": getattr(args, "pose_model", None),
            "pose_stride": getattr(args, "pose_stride", None),
            "temporal_pose_refine": want_temporal_pose_refine(getattr(args, "temporal_pose_refine", False)),
            "temporal_pose_radius": temporal_pose_radius(getattr(args, "temporal_pose_radius", 2)),
            "montage": getattr(args, "montage", None),
            "save_phase_previews": getattr(args, "save_phase_previews", None),
            "params_file": getattr(args, "params", None),
        },
        "params": params,
        "env": env_snap,
        "previews": list(_LAB_CTX.get("previews", [])),
        "final_video_relpath": final_video_relpath,
        "view_variants": view_variants or {},
        "total_elapsed_s": round(total_elapsed, 3),
        "git_commit": git_sha,
        "pipeline_stages": list(_LAB_CTX.get("stage_log", [])),
        "run_context_final": dict(_LAB_CTX.get("run_context", {})),
    }
    mp = _LAB_CTX["run_manifest_path"]
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Run manifest: {mp}")


def _apply_params_to_env(params: dict) -> None:
    """Allow YAML to set SWAY_* and common offline env vars before tracking."""
    apply_sway_params_to_env(params)


def log_resource_usage(phase_name: str = "") -> None:
    """Log current CPU, RAM, and GPU usage for hardware sizing (e.g. cloud server)."""
    try:
        import psutil
    except ImportError:
        return
    p = psutil.Process()
    # Process memory (RSS) in GB
    rss_gb = p.memory_info().rss / (1024**3)
    # System memory
    vmem = psutil.virtual_memory()
    sys_used_gb = vmem.used / (1024**3)
    sys_total_gb = vmem.total / (1024**3)
    cpu_pct = p.cpu_percent(interval=0.1) if hasattr(p, "cpu_percent") else 0
    try:
        sys_cpu_pct = psutil.cpu_percent(interval=0.1)
    except Exception:
        sys_cpu_pct = 0
    parts = [f"RAM: {sys_used_gb:.1f}/{sys_total_gb:.1f} GB (process: {rss_gb:.2f} GB)", f"CPU: {cpu_pct:.0f}% (system: {sys_cpu_pct:.0f}%)"]
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        parts.append(f"GPU: {a:.2f} GB alloc / {r:.2f} GB reserved (device total: {total:.1f} GB)")
    prefix = "  [Resources" + (f" @ {phase_name}" if phase_name else "") + "] "
    print(prefix + " | ".join(parts))


def main():
    import argparse
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Sway Pose Tracking V3.4 - Pose estimation for dance videos"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input MP4 video (e.g., input/dance.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        choices=["base", "large", "huge", "rtmpose", "sapiens"],
        default="base",
        help=(
            "2D pose: ViTPose+ base/large/huge, rtmpose (RTMPose-L via MMPose), or sapiens. "
            "For sapiens: set SWAY_SAPIENS_TORCHSCRIPT to a COCO-17 .pt2 file for native "
            "Meta Sapiens (TorchScript); otherwise ViTPose-Base is used. "
            "Override ViTPose checkpoint with SWAY_VITPOSE_MODEL (non-sapiens)."
        ),
    )
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=1,
        choices=[1, 2],
        help="Run pose every Nth frame; interpolate others. 2 = ~2x faster.",
    )
    parser.add_argument(
        "--temporal-pose-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After ViTPose: confidence-weighted (x,y) smoothing over ±N frames per track. "
            "Default off (master stack); --temporal-pose-refine to enable. Not the Poseidon video model. "
            "Env SWAY_TEMPORAL_POSE_REFINE=0|1 overrides CLI; production sets 0 unless SWAY_UNLOCK_SMOOTH_TUNING=1."
        ),
    )
    parser.add_argument(
        "--temporal-pose-radius",
        type=int,
        default=2,
        help="Half-window in frames for --temporal-pose-refine (clamped 0–8). Env: SWAY_TEMPORAL_POSE_RADIUS.",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        default=False,
        help="Generate a pipeline montage video showing each phase.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to YAML file with parameter overrides (e.g. SYNC_SCORE_MIN: 0.12)",
    )
    parser.add_argument(
        "--save-phase-previews",
        action="store_true",
        default=False,
        help="Write phase_previews/*.mp4 (Pipeline Lab).",
    )
    parser.add_argument(
        "--save-live-artifacts",
        action="store_true",
        default=False,
        help="Write output/live_artifacts/ (phase3 bundle + phase1 npz) for Lab live parameter preview.",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=str,
        default=None,
        help="Append one JSON line per major pipeline stage.",
    )
    parser.add_argument(
        "--run-manifest",
        type=str,
        default=None,
        help="Write run_manifest.json to this path when the run completes.",
    )
    parser.add_argument(
        "--stop-after-boundary",
        type=str,
        default=None,
        choices=list(CHECKPOINT_BOUNDARIES),
        help="Write checkpoint at this boundary and exit (MASTER_PIPELINE_GUIDELINE §4.4).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Directory with checkpoint_manifest.json from a prior partial run.",
    )
    parser.add_argument(
        "--expect-boundary",
        type=str,
        default=None,
        choices=list(CHECKPOINT_BOUNDARIES),
        help="Require checkpoint manifest boundary_id to match (unless --force-checkpoint-load).",
    )
    parser.add_argument(
        "--force-checkpoint-load",
        action="store_true",
        default=False,
        help="Allow resume when video size or params fingerprint differs from checkpoint.",
    )
    parser.add_argument(
        "--phase-debug-jsonl",
        type=str,
        default=None,
        help="Append phase_summary JSON lines (default: output_dir/phase_debug.jsonl when checkpoint flags used).",
    )
    parser.add_argument(
        "--no-pose-3d-lift",
        action="store_true",
        default=False,
        help="Disable MotionAGFormer 3D lift and pose_3d JSON export. Env SWAY_3D_LIFT=0 also disables.",
    )
    args = parser.parse_args()
    _apply_stack_default_env()

    params = {}
    if args.params:
        import yaml
        with open(args.params) as f:
            params = yaml.safe_load(f) or {}

    apply_master_locked_pre_pose_prune_params(params)
    apply_master_locked_reid_dedup_params(params)
    apply_master_locked_post_pose_prune_params(params)
    apply_master_locked_smooth_params(params)
    _apply_params_to_env(params)
    apply_master_locked_detection_env()
    apply_master_locked_hybrid_sam_env()
    # ByteTrack (Lab "Fast preview" / tracker_technology=bytetrack): skip SAM2 overlap refiner — much faster than Deep OC-SORT + hybrid SAM.
    if boxmot_tracker_kind_from_env() == "bytetrack":
        os.environ["SWAY_HYBRID_SAM_OVERLAP"] = "0"
    apply_master_locked_phase3_stitch_env()
    apply_master_locked_pose_env()
    apply_master_locked_smooth_env()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (Path("input")).mkdir(exist_ok=True)
    (Path("models")).mkdir(exist_ok=True)

    manifest_path = args.run_manifest
    if manifest_path is None and (args.save_phase_previews or args.progress_jsonl):
        manifest_path = str(output_dir / "run_manifest.json")
    _lab_init(progress_jsonl=args.progress_jsonl, run_manifest_path=manifest_path)
    _lab_update_context(
        input_video=str(video_path.resolve()),
        input_bytes=int(video_path.stat().st_size),
    )

    phase_preview_dir = None
    montage_dir = None
    if args.save_phase_previews:
        phase_preview_dir = output_dir / "phase_previews"
        phase_preview_dir.mkdir(parents=True, exist_ok=True)
    if args.montage and not args.save_phase_previews:
        montage_dir = output_dir / "_montage_clips"
        montage_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = phase_preview_dir if phase_preview_dir is not None else montage_dir

    params_path = Path(args.params).resolve() if args.params else None
    ck_subdir = output_dir / "checkpoints"
    _use_ck = bool(args.stop_after_boundary or args.resume_from)
    _pdbg_path = args.phase_debug_jsonl
    if _pdbg_path is None and _use_ck:
        _pdbg_path = str(output_dir / "phase_debug.jsonl")
    phase_dbg = PhaseDebugLogger(Path(_pdbg_path) if _pdbg_path else None)

    resume_dir = Path(args.resume_from).resolve() if args.resume_from else None
    start_marker = 0
    if resume_dir is not None:
        man0 = read_manifest(resume_dir)
        validate_manifest(
            man0,
            video_path=video_path,
            params_path=params_path,
            expect_boundary=args.expect_boundary,
            force=bool(args.force_checkpoint_load),
        )
        b0 = str(man0.get("boundary_id") or "")
        start_marker = {
            "after_phase_1": 2,
            "after_phase_3": 4,
            "after_phase_4": 5,
            "after_phase_5": 6,
            "after_phase_8": 9,
        }.get(b0, -1)
        if start_marker < 0:
            raise SystemExit(f"Unsupported resume checkpoint boundary_id: {b0!r}")

    montage_clips = []
    montage_clip_frames = 0
    montage_start = 0

    vis_skip = float(params.get("POSE_VISIBILITY_THRESHOLD", 0.3))
    pose_crop_smooth_alpha = float(params.get("POSE_CROP_SMOOTH_ALPHA", 0) or 0)
    pose_crop_foot_bias_frac = float(params.get("POSE_CROP_FOOT_BIAS_FRAC", 0) or 0)
    pose_crop_head_bias_frac = float(params.get("POSE_CROP_HEAD_BIAS_FRAC", 0) or 0)
    pose_crop_anti_jitter_px = float(params.get("POSE_CROP_ANTI_JITTER_PX", 0) or 0)

    device = get_device()
    print(f"Using device: {device}")
    _lab_update_context(device=str(device), pose_model_cli=getattr(args, "pose_model", None))
    log_resource_usage("startup")

    total_start = time.time()
    phase3_checkpoint_stop_pending = False

    _stop_b = args.stop_after_boundary
    _track_lab_cb = _lab_make_tracking_progress_callback(str(video_path))
    hybrid_sam_stats: Dict[str, Any] = {}
    phase1_dets_by_frame: Dict[int, Any] = {}
    phase1_pre_classical_by_frame: Dict[int, Any] = {}
    raw_pre: Dict[int, Any] = {}
    raw_tracks: Dict[int, Any] = {}
    total_frames = 0
    output_fps = 30.0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080
    ystride = 1
    dt_track = 0.0
    dt_stitch = 0.0

    def _ck_dir(name: str) -> Path:
        d = ck_subdir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _run_phase3_stitch() -> None:
        nonlocal raw_tracks, dt_stitch
        print("\n[3/11] Phase 3 — Post-track stitching (dormant, fragment stitch, coalesce, merge)…")
        t0 = time.time()
        ph3_stop = threading.Event()
        if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
            _lab_progress(
                2,
                "phase3_post_stitch",
                "Phase 3: Post-track stitching",
                status="running",
                extra={"wall_s_in_phase": 0.0, "step": "stitch_coalesce_merge"},
            )

            def _ph3_pulse() -> None:
                while not ph3_stop.wait(timeout=10.0):
                    _lab_progress(
                        2,
                        "phase3_post_stitch",
                        "Phase 3: Post-track stitching",
                        status="running",
                        extra={
                            "wall_s_in_phase": round(time.time() - t0, 1),
                            "step": "stitch_coalesce_merge",
                        },
                    )

            threading.Thread(target=_ph3_pulse, daemon=True).start()
        try:
            raw_for_stitch = raw_pre
            try:
                from sway.dancer_registry_pipeline import (
                    apply_dancer_registry_crossover_pass,
                    phase13_dancer_registry_enabled,
                )

                if phase13_dancer_registry_enabled():
                    print(
                        "\n  [dancer_registry] Crossover verify pass (zonal HSV, no SAM)…",
                        flush=True,
                    )
                    raw_for_stitch = apply_dancer_registry_crossover_pass(
                        raw_pre,
                        str(video_path),
                        int(total_frames),
                        int(frame_width),
                        int(frame_height),
                        float(native_fps),
                    )
            except Exception as _dr_ex:  # noqa: BLE001
                print(f"  [dancer_registry] Crossover pass failed: {_dr_ex}", flush=True)
                raw_for_stitch = raw_pre
            raw_tracks = apply_post_track_stitching(
                raw_for_stitch,
                total_frames,
                ystride=ystride,
                video_path=str(video_path),
                native_fps=float(native_fps),
            )
        finally:
            ph3_stop.set()
        dt_stitch = time.time() - t0
        raw_tracks = maybe_gnn_refine_raw_tracks(raw_tracks, int(total_frames), int(ystride))
        print(f"  └─ {dt_stitch:.1f}s")
        log_resource_usage("3-post-track-stitch")

    if start_marker in (5, 6, 9):
        pass
    elif start_marker == 4:
        b3 = load_after_phase_3(resume_dir)
        raw_tracks = b3["raw_tracks"]
        total_frames = int(b3["total_frames"])
        output_fps = float(b3["output_fps"])
        native_fps = float(b3["native_fps"])
        frame_width = int(b3["frame_width"])
        frame_height = int(b3["frame_height"])
        ystride = int(b3["ystride"])
        hybrid_sam_stats = dict(b3.get("hybrid_sam_stats") or {})
        phase1_dets_by_frame = {}
        phase1_pre_classical_by_frame = {}
        dt_track = 0.0
        dt_stitch = 0.0
    elif start_marker == 2:
        yolo_map, meta = load_phase1_yolo_dets(resume_dir)
        total_frames = int(meta["total_frames"])
        output_fps = float(meta["output_fps"])
        native_fps = float(meta["native_fps"])
        frame_width = int(meta["frame_width"])
        frame_height = int(meta["frame_height"])
        ystride = int(meta["ystride"])
        if bidirectional_track_pass_enabled():
            print(
                "  Note: SWAY_BIDIRECTIONAL_TRACK_PASS is ignored when resuming from after_phase_1.",
                flush=True,
            )
        print("\n[1/11] Phase 1 — (from checkpoint) skipped; running tracking + hybrid from YOLO dets…")
        _p2_ck_msg = (
            "[2/11] Phase 2 — Tracking (BoxMOT ByteTrack from stored YOLO dets; hybrid SAM off)"
            if boxmot_tracker_kind_from_env() == "bytetrack"
            else "[2/11] Phase 2 — Tracking (BoxMOT + hybrid SAM from stored YOLO boxes)"
        )
        print(_p2_ck_msg)
        t0 = time.time()
        (
            raw_pre,
            total_frames,
            output_fps,
            _frames_list,
            native_fps,
            frame_width,
            frame_height,
            ystride,
            hybrid_sam_stats,
            phase1_dets_by_frame,
            phase1_pre_classical_by_frame,
        ) = run_boxmot_tracking_from_yolo_dets(str(video_path), yolo_map, lab_on_infer=_track_lab_cb)
        _pc = meta.get("phase1_pre_classical")
        if isinstance(_pc, dict) and _pc:
            phase1_pre_classical_by_frame = _pc
        dt_track = time.time() - t0
        print(f"  └─ Phases 1–2 (from checkpoint): {dt_track:.1f}s")
        log_resource_usage("1-2-detection-tracking")
        phase_dbg.log("1-2", dt_track, {"resume": "after_phase_1", "raw_track_count": len(raw_pre)})
        _run_phase3_stitch()
        phase_dbg.log("3", dt_stitch, {"track_count": len(raw_tracks)})
        if _stop_b == "after_phase_3":
            save_after_phase_3(
                _ck_dir("after_phase_3"),
                raw_tracks,
                total_frames=total_frames,
                native_fps=native_fps,
                output_fps=output_fps,
                frame_width=frame_width,
                frame_height=frame_height,
                ystride=ystride,
                hybrid_sam_stats=hybrid_sam_stats,
                video_path=video_path,
                params_path=params_path,
            )
            print(f"Checkpoint written: {_ck_dir('after_phase_3')}", flush=True)
            phase3_checkpoint_stop_pending = True
    else:
        if _stop_b == "after_phase_1":
            if not _use_boxmot():
                raise SystemExit(
                    "--stop-after-boundary after_phase_1 requires the BoxMOT path (default). "
                    "BoT-SORT runs are not split for phase-1-only checkpoints."
                )
            print("\n[1/11] Phase 1 — Detection (YOLO only, no hybrid SAM / no tracker)")
            t0 = time.time()
            (
                total_frames,
                output_fps,
                native_fps,
                frame_width,
                frame_height,
                ystride,
                phase1_dets_by_frame,
                phase1_pre_classical_by_frame,
            ) = run_phase1_yolo_only_boxmot(str(video_path), lab_on_infer=_track_lab_cb)
            dt_track = time.time() - t0
            print(f"  └─ Phase 1 (YOLO only): {dt_track:.1f}s")
            phase_dbg.log("1", dt_track, {"det_frames_with_dets": len(phase1_dets_by_frame)})
            save_phase1_yolo_dets(
                _ck_dir("after_phase_1"),
                phase1_dets_by_frame,
                total_frames=total_frames,
                native_fps=native_fps,
                output_fps=output_fps,
                frame_width=frame_width,
                frame_height=frame_height,
                ystride=ystride,
                video_path=video_path,
                params_path=params_path,
                phase1_pre_classical_by_frame=phase1_pre_classical_by_frame,
            )
            print(f"Checkpoint written: {_ck_dir('after_phase_1')}", flush=True)
            if clip_dir is not None:
                phase1_fd = []
                for i in range(total_frames):
                    pairs = phase1_dets_by_frame.get(i, [])
                    phase1_fd.append(
                        {
                            "frame_idx": i,
                            "phase1_boxes": [p[0] for p in pairs],
                            "phase1_confs": [p[1] for p in pairs],
                            "boxes": [],
                            "track_ids": [],
                            "poses": {},
                        }
                    )
                p0 = clip_dir / "00_phase1_detections.mp4"
                montage_clips.append(
                    render_phase_clip(
                        video_path,
                        phase1_fd,
                        "Phase 1: Detections",
                        draw_phase1_detection_preview,
                        native_fps,
                        output_fps,
                        p0,
                        caption="YOLO boxes + confidence only",
                        full_length=True,
                        show_title_card=False,
                    )
                )
                prv0 = f"phase_previews/{p0.name}"
                _lab_progress(
                    1,
                    "phase1_detections",
                    "Phase 1: Detections (checkpoint — after_phase_1)",
                    elapsed_s=dt_track,
                    preview_relpath=prv0,
                    extra={"checkpoint_boundary": "after_phase_1"},
                )
                _lab_register_preview("phase1_detections", prv0)
                _lab_update_context(
                    checkpoint_boundary="after_phase_1",
                    vitpose_model_id="(not loaded — stopped before pose)",
                )
                _lab_write_manifest(
                    video_path=video_path,
                    output_dir=output_dir,
                    args=args,
                    params=params,
                    model_id="checkpoint_after_phase_1",
                    total_elapsed=time.time() - total_start,
                    final_video_relpath=prv0,
                    view_variants={"full": prv0},
                )
            return

        print("\n[1/11] Phase 1 — Detection (YOLO)")
        _p2_track_msg = (
            "[2/11] Phase 2 — Tracking (BoxMOT ByteTrack; hybrid SAM off — fast preview path)"
            if boxmot_tracker_kind_from_env() == "bytetrack"
            else "[2/11] Phase 2 — Tracking (BoxMOT Deep OC-SORT, optional track-time OSNet; hybrid SAM when enabled)"
        )
        print(_p2_track_msg)
        print("  (Single pass over the video: detection and association per frame.)")
        t0 = time.time()
        (
            raw_pre,
            total_frames,
            output_fps,
            _frames_list,
            native_fps,
            frame_width,
            frame_height,
            ystride,
            hybrid_sam_stats,
            phase1_dets_by_frame,
            phase1_pre_classical_by_frame,
        ) = run_tracking_before_post_stitch(str(video_path), lab_on_infer=_track_lab_cb)
        if bidirectional_track_pass_enabled():
            t_bi = time.time()
            print(
                "  Optional bidirectional pass: SWAY_BIDIRECTIONAL_TRACK_PASS=1 "
                "(reverse video + second track, then merge; ~2× Phase 1–2 time)…",
                flush=True,
            )
            rev_path = output_dir / ".sway_bidirectional_rev_input.mp4"
            try:
                reverse_video_via_ffmpeg(video_path.resolve(), rev_path)
                (
                    raw_rev,
                    tf_rev,
                    _,
                    _,
                    _,
                    _,
                    _,
                    ys_rev,
                    hybrid_rev,
                    _phase1_rev,
                    _phase1_pre_rev,
                ) = run_tracking_before_post_stitch(str(rev_path), lab_on_infer=_track_lab_cb)
                hybrid_sam_stats = _merge_hybrid_sam_stats(hybrid_sam_stats, hybrid_rev)
                if tf_rev != total_frames or int(ys_rev) != int(ystride):
                    print(
                        f"  Warning: reverse tracking shape mismatch "
                        f"(frames {tf_rev} vs {total_frames}, stride {ys_rev} vs {ystride}); skipping merge.",
                        flush=True,
                    )
                else:
                    rev_mapped = remap_reverse_pass_timeline(raw_rev, total_frames)
                    iou_t = bidirectional_iou_threshold()
                    min_m = bidirectional_min_match_frames()
                    n_before = len(raw_pre)
                    raw_pre = merge_forward_backward_tracks(
                        raw_pre,
                        rev_mapped,
                        iou_threshold=iou_t,
                        min_match_frames=min_m,
                    )
                    print(
                        f"  └─ Bidirectional merge: {n_before} → {len(raw_pre)} raw tracks "
                        f"(IoU≥{iou_t}, ≥{min_m} matched frames); +{time.time() - t_bi:.1f}s",
                        flush=True,
                    )
            except Exception as ex:
                print(f"  Warning: bidirectional pass skipped ({ex})", flush=True)
            finally:
                try:
                    rev_path.unlink(missing_ok=True)
                except OSError:
                    pass
        dt_track = time.time() - t0
        print(f"  └─ Phases 1–2: {dt_track:.1f}s")
        log_resource_usage("1-2-detection-tracking")
        phase_dbg.log("1-2", dt_track, {"raw_track_count": len(raw_pre)})
        _run_phase3_stitch()
        phase_dbg.log("3", dt_stitch, {"track_count": len(raw_tracks)})
        if _stop_b == "after_phase_3":
            save_after_phase_3(
                _ck_dir("after_phase_3"),
                raw_tracks,
                total_frames=total_frames,
                native_fps=native_fps,
                output_fps=output_fps,
                frame_width=frame_width,
                frame_height=frame_height,
                ystride=ystride,
                hybrid_sam_stats=hybrid_sam_stats,
                video_path=video_path,
                params_path=params_path,
            )
            print(f"Checkpoint written: {_ck_dir('after_phase_3')}", flush=True)
            phase3_checkpoint_stop_pending = True

    phase3_ck_preview_relpath: Optional[str] = None
    if start_marker not in (5, 6, 9):
        _lab_update_context(
            total_frames=int(total_frames),
            native_fps=round(float(native_fps), 4),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            raw_track_count=int(len(raw_tracks)),
            output_fps=round(float(output_fps), 4),
            track_summary=_lab_track_summary_heuristic(raw_tracks, int(ystride)),
        )

        ts_path = output_dir / "track_stats.json"
        stats = compute_track_quality_stats(raw_tracks, int(total_frames), int(ystride))
        write_track_stats_json(ts_path, stats)
        print(f"  Track stats written: {ts_path}", flush=True)
        _lab_update_context(track_stats_path=str(ts_path.name))

        if clip_dir is not None:
            print(
                "  Phase preview videos: full source length, no title slate "
                "(encoding may take noticeably longer than short samples).",
                flush=True,
            )
            all_ids = set(raw_tracks.keys())
            all_tracking = raw_tracks_to_per_frame(raw_tracks, total_frames, all_ids)
            phase1_fd = []
            for i in range(total_frames):
                pairs = phase1_dets_by_frame.get(i, [])
                phase1_fd.append(
                    {
                        "frame_idx": i,
                        "phase1_boxes": [p[0] for p in pairs],
                        "phase1_confs": [p[1] for p in pairs],
                        "boxes": [],
                        "track_ids": [],
                        "poses": {},
                    }
                )
            p0 = clip_dir / "00_phase1_detections.mp4"
            montage_clips.append(
                render_phase_clip(
                    video_path,
                    phase1_fd,
                    "Phase 1: Detections",
                    draw_phase1_detection_preview,
                    native_fps,
                    output_fps,
                    p0,
                    caption="All detector boxes + confidence (no track IDs)",
                    full_length=True,
                    show_title_card=False,
                )
            )
            prv0 = f"phase_previews/{p0.name}" if args.save_phase_previews else None
            if prv0:
                _lab_register_preview("phase1_detections", prv0)
                _lab_progress(
                    1,
                    "phase1_detections",
                    "Phase 1: Detections (all boxes + conf)",
                    preview_relpath=prv0,
                    extra={"det_frames": sum(1 for i in range(total_frames) if phase1_dets_by_frame.get(i))},
                )
            all_fd = [
                {
                    "frame_idx": i,
                    "boxes": t["boxes"],
                    "track_ids": t["track_ids"],
                    "poses": {},
                    "sam2_input_roi_xyxy": t.get("sam2_input_roi_xyxy"),
                    "is_sam_refined": t.get("is_sam_refined"),
                    "segmentation_masks": t.get("segmentation_masks"),
                }
                for i, t in enumerate(all_tracking)
            ]
            p1 = clip_dir / "01_tracks_post_stitch.mp4"
            montage_clips.append(render_phase_clip(
                video_path, all_fd, "Phases 1–3: Tracks",
                lambda f, d: draw_tracks_post_stitch_preview(
                    f,
                    d["boxes"],
                    d["track_ids"],
                    d.get("sam2_input_roi_xyxy"),
                    is_sam_refined=d.get("is_sam_refined"),
                    segmentation_masks=d.get("segmentation_masks"),
                ),
                native_fps, output_fps, p1,
                caption="After detection, tracking, and post-track stitching",
                full_length=True,
                show_title_card=False,
            ))
            prv = f"phase_previews/{p1.name}" if args.save_phase_previews else None
            _lab_extra_early = {
                "track_ids_sample": sorted(raw_tracks.keys(), key=lambda x: int(x) if isinstance(x, int) else 0)[:80],
                "track_count": len(raw_tracks),
            }
            _lab_progress(1, "phases_1_2", "Phases 1–2: Detection & tracking", elapsed_s=dt_track, extra=_lab_extra_early)
            _lab_progress(
                2,
                "phase3_post_stitch",
                "Phase 3: Post-track stitching",
                elapsed_s=dt_stitch,
                preview_relpath=prv,
                extra={**_lab_extra_early, "phase": "post_track_stitch"},
            )
            if prv:
                _lab_register_preview("tracks_post_stitch", prv)
            if phase3_checkpoint_stop_pending and prv:
                phase3_ck_preview_relpath = prv
        else:
            _lab_progress(1, "phases_1_2", "Phases 1–2: Detection & tracking", elapsed_s=dt_track)
            _lab_progress(2, "phase3_post_stitch", "Phase 3: Post-track stitching", elapsed_s=dt_stitch)
    elif start_marker == 4:
        _lab_update_context(
            total_frames=int(total_frames),
            native_fps=round(float(native_fps), 4),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            raw_track_count=int(len(raw_tracks)),
            output_fps=round(float(output_fps), 4),
            track_summary=_lab_track_summary_heuristic(raw_tracks, int(ystride)),
        )
        ts_path = output_dir / "track_stats.json"
        stats = compute_track_quality_stats(raw_tracks, int(total_frames), int(ystride))
        write_track_stats_json(ts_path, stats)
        print(f"  Track stats written: {ts_path}", flush=True)
        _lab_update_context(track_stats_path=str(ts_path.name))

    if phase3_checkpoint_stop_pending:
        dj = build_phase3_tracking_data_json(
            video_path=str(video_path),
            raw_tracks=raw_tracks,
            total_frames=int(total_frames),
            native_fps=float(native_fps),
            output_fps=float(output_fps),
        )
        data_path = output_dir / "data.json"
        data_path.write_text(json.dumps(dj, separators=(",", ":")), encoding="utf-8")
        print(f"  Tracking-only data.json written: {data_path}", flush=True)
        _lab_update_context(
            checkpoint_boundary="after_phase_3",
            vitpose_model_id="(not loaded — stopped before pose)",
            pipeline_diagnostics=_build_pipeline_diagnostics(hybrid_sam_stats, args, params),
        )
        rel = phase3_ck_preview_relpath or ""
        _lab_write_manifest(
            video_path=video_path,
            output_dir=output_dir,
            args=args,
            params=params,
            model_id="checkpoint_after_phase_3",
            total_elapsed=time.time() - total_start,
            final_video_relpath=rel,
            view_variants={"full": rel} if rel else {},
        )
        return

    if getattr(args, "save_live_artifacts", False) and start_marker < 5:
        save_live_phase3_bundle(
            output_dir,
            raw_tracks=raw_tracks,
            total_frames=total_frames,
            native_fps=native_fps,
            output_fps=output_fps,
            frame_width=frame_width,
            frame_height=frame_height,
            ystride=ystride,
            hybrid_sam_stats=hybrid_sam_stats,
        )
        if phase1_dets_by_frame or phase1_pre_classical_by_frame:
            la = output_dir / "live_artifacts"
            la.mkdir(parents=True, exist_ok=True)
            save_phase1_yolo_dets(
                la,
                phase1_dets_by_frame,
                total_frames=total_frames,
                native_fps=native_fps,
                output_fps=output_fps,
                frame_width=frame_width,
                frame_height=frame_height,
                ystride=ystride,
                video_path=video_path,
                params_path=params_path,
                phase1_pre_classical_by_frame=phase1_pre_classical_by_frame,
                write_manifest_file=False,
            )

    if start_marker < 5:
        # Diagnostic log for debugging (prune_log.json) + progress.jsonl phase_detail
        prune_log_entries = []
        _prune_mark = 0
        _p4_lab_state: Dict[str, Any] = {"last_emit": 0.0}
        t0 = time.time()
        (
            surviving_ids,
            prune_log_entries,
            tracking_results,
            initial_count,
            tracker_ids_before_prune,
        ) = run_pre_pose_prune_phase(
            raw_tracks=raw_tracks,
            total_frames=total_frames,
            frame_width=frame_width,
            frame_height=frame_height,
            video_path=video_path,
            params=params,
            lab_phase4_tick=_lab_phase4_tick,
            lab_state=_p4_lab_state,
            lab_update_context=_lab_update_context,
            quiet=False,
        )
        dt_preprune = time.time() - t0
        print(f"  └─ {dt_preprune:.1f}s")
        log_resource_usage("4-pre-pose-prune")

        _ex4 = {
            **_prune_events_extra(prune_log_entries, _prune_mark),
            "tracks_before_prune": initial_count,
            "tracks_after_pre_pose_prune": len(surviving_ids),
        }
        _prune_mark = len(prune_log_entries)

        prv2: Optional[str] = None
        if clip_dir is not None:
            pre_fd = [
                {"frame_idx": i, "boxes": t["boxes"], "track_ids": t["track_ids"], "poses": {}}
                for i, t in enumerate(tracking_results)
            ]
            p2 = clip_dir / "02_pre_pose_prune.mp4"
            _ov_pre = build_prune_overlay_index(
                prune_log_entries,
                PRE_POSE_PREVIEW_RULES,
                max(0, int(total_frames) - 1),
                raw_tracks=raw_tracks,
            )
            _draw_pre = wrap_draw_fn_with_prune_overlays(
                lambda f, d: draw_boxes_only(f, d["boxes"], d["track_ids"]),
                _ov_pre,
            )
            montage_clips.append(render_phase_clip(
                video_path,
                pre_fd,
                "Phase 4: After pre-pose prune",
                _draw_pre,
                native_fps,
                output_fps,
                p2,
                caption="Surviving boxes + pruned-track highlights (see legend in Lab)",
                full_length=True,
                show_title_card=False,
            ))
            prv2 = f"phase_previews/{p2.name}" if args.save_phase_previews else None
            _lab_progress(3, "pre_pose_prune", "Phase 4: Pre-pose pruning", elapsed_s=dt_preprune, preview_relpath=prv2, extra=_ex4)
            if prv2:
                _lab_register_preview("pre_pose_prune", prv2)
        else:
            _lab_progress(3, "pre_pose_prune", "Phase 4: Pre-pose pruning", elapsed_s=dt_preprune, extra=_ex4)

        phase_dbg.log("4", dt_preprune, {"tracks_after_pre_pose_prune": len(surviving_ids)})
        if _stop_b == "after_phase_4":
            save_after_phase_4(
                _ck_dir("after_phase_4"),
                raw_tracks=raw_tracks,
                surviving_ids=surviving_ids,
                tracking_results=tracking_results,
                total_frames=total_frames,
                frame_width=frame_width,
                frame_height=frame_height,
                native_fps=native_fps,
                output_fps=output_fps,
                prune_log_entries=prune_log_entries,
                tracker_ids_before_prune=tracker_ids_before_prune,
                hybrid_sam_stats=hybrid_sam_stats,
                video_path=video_path,
                params_path=params_path,
            )
            print(f"Checkpoint written: {_ck_dir('after_phase_4')}", flush=True)
            _lab_update_context(
                checkpoint_boundary="after_phase_4",
                vitpose_model_id="(not loaded — stopped before pose)",
                pipeline_diagnostics=_build_pipeline_diagnostics(hybrid_sam_stats, args, params),
            )
            rel4 = prv2 or ""
            _lab_write_manifest(
                video_path=video_path,
                output_dir=output_dir,
                args=args,
                params=params,
                model_id="checkpoint_after_phase_4",
                total_elapsed=time.time() - total_start,
                final_video_relpath=rel4,
                view_variants={"full": rel4} if rel4 else {},
            )
            return
    elif start_marker == 5:
        # Resume from after_phase_4 checkpoint only (not after_phase_5 — that dir has phase5.pkl.gz).
        b4 = load_after_phase_4(resume_dir)
        raw_tracks = b4["raw_tracks"]
        surviving_ids = set(b4["surviving_ids"])
        tracking_results = b4["tracking_results"]
        total_frames = int(b4["total_frames"])
        frame_width = int(b4["frame_width"])
        frame_height = int(b4["frame_height"])
        native_fps = float(b4["native_fps"])
        output_fps = float(b4["output_fps"])
        prune_log_entries = list(b4["prune_log_entries"])
        tracker_ids_before_prune = list(b4["tracker_ids_before_prune"])
        hybrid_sam_stats = dict(b4.get("hybrid_sam_stats") or {})
        initial_count = len(raw_tracks)
        _prune_mark = len(prune_log_entries)
        dt_preprune = 0.0
        _ex4 = {
            **_prune_events_extra(prune_log_entries, 0),
            "tracks_before_prune": initial_count,
            "tracks_after_pre_pose_prune": len(surviving_ids),
        }

    if start_marker < 6:
        # ── Phase 5: Pose estimation with visibility scoring ──────────────────
        use_rtmpose = args.pose_model == "rtmpose"
        use_sapiens_requested = args.pose_model == "sapiens"
        sapiens_ckpt = _resolve_sapiens_torchscript_path() if use_sapiens_requested else None
        use_native_sapiens = bool(sapiens_ckpt)
        env_pose = os.environ.get("SWAY_VITPOSE_MODEL", "").strip()
        if use_rtmpose:
            model_id = "rtmpose-l (MMPose)"
        elif use_sapiens_requested and use_native_sapiens:
            model_id = f"sapiens-torchscript:{sapiens_ckpt}"
        elif use_sapiens_requested:
            model_id = "usyd-community/vitpose-plus-base"
            print(
                "  Note: --pose-model sapiens — set SWAY_SAPIENS_TORCHSCRIPT to a COCO-17 .pt2 file for native "
                "Sapiens; using ViTPose-Base for 2D keypoints this run.",
                flush=True,
            )
        elif env_pose:
            model_id = env_pose
        elif args.pose_model == "huge":
            model_id = "usyd-community/vitpose-plus-huge"
        elif args.pose_model == "large":
            model_id = "usyd-community/vitpose-plus-large"
        else:
            model_id = "usyd-community/vitpose-plus-base"
        _lab_update_context(vitpose_model_id=model_id, pose_stride=int(args.pose_stride))
        stride_note = f" stride={args.pose_stride}" if args.pose_stride > 1 else ""
        if use_rtmpose:
            pose_label = "RTMPose-L"
        elif use_native_sapiens:
            pose_label = "Sapiens (TorchScript)"
        elif use_sapiens_requested:
            pose_label = "Sapiens (ViTPose-Base fallback)"
        else:
            pose_label = f"ViTPose-{args.pose_model.title()}"
        print(f"\n[5/11] Phase 5 — Pose estimation ({pose_label}{stride_note}, visibility-gated)…")
        t0 = time.time()
        if use_rtmpose:
            from sway.rtmpose_estimator import RTMPoseEstimator

            print(f"  Loading RTMPose (MMPose) on {device}…", flush=True)
            pose_estimator = RTMPoseEstimator(device=device)
        elif use_native_sapiens:
            from sway.sapiens_estimator import SapiensTorchscriptPoseEstimator

            print(f"  Loading Sapiens TorchScript (COCO-17) from {sapiens_ckpt}…", flush=True)
            pose_estimator = SapiensTorchscriptPoseEstimator(device=device, checkpoint_path=sapiens_ckpt)
        else:
            print(f"  Loading ViTPose ({model_id}) on {device}…", flush=True)
            pose_estimator = PoseEstimator(device=device, model_name=model_id)
        if use_rtmpose:
            backend = "RTMPose"
        elif use_native_sapiens:
            backend = "Sapiens (TorchScript)"
        elif use_sapiens_requested:
            backend = "ViTPose (Sapiens slot — fallback weights)"
        else:
            backend = "ViTPose"
        print(
            f"  {backend} weights loaded in {time.time() - t0:.1f}s (first forward may add MPS/CUDA compile time)",
            flush=True,
        )
        if vitpose_debug_enabled():
            print(
                "  ViTPose verbose debug: Phase 5 logs embeddings timing, hybrid SAM plain vs masked split, "
                "and each ViTPose processor/model/post step.",
                flush=True,
            )

        raw_poses_by_frame = [{} for _ in range(total_frames)]
        embeddings_by_frame = [{} for _ in range(total_frames)]  # V3.8: Appearance for Re-ID
        frames_stored = [None] * total_frames
        last_pct = -1
        occluded_skips = 0
        # Phase 5 progress: old code only printed every 10% of *video* frames, so the first ~10%
        # of frames could run ViTPose with zero logs (minutes on huge + MPS). See SWAY_POSE_LOG_* env.
        phase5_pose_passes = 0
        last_phase5_wall_log = time.time()
        pose_log_every_sec = float(os.environ.get("SWAY_POSE_LOG_EVERY_SEC", "5"))
        pose_log_every_n = max(1, int(os.environ.get("SWAY_POSE_LOG_EVERY_N_PASSES", "8")))
        pose_slow_sec = float(os.environ.get("SWAY_POSE_SLOW_FORWARD_SEC", "4"))
        phase5_logged_first_forward = False

        frame_q = queue.Queue(maxsize=30)
        def frame_producer():
            for f_idx, frame in iter_video_frames(str(video_path)):
                if f_idx >= total_frames:
                    break
                frame_q.put((f_idx, frame))
            frame_q.put(None)

        # Cache last good pose per track for occluded frames
        last_good_pose: dict = {}

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(frame_producer)

            prev_boxes = {}
            prev_poses = {}
            pose_crop_state: Dict[int, Tuple[float, float, float, float]] = {}

            while True:
                item = frame_q.get()
                if item is None:
                    break
                frame_idx, frame = item

                tracking = (
                    tracking_results[frame_idx]
                    if frame_idx < len(tracking_results)
                    else {
                        "boxes": [],
                        "track_ids": [],
                        "confs": [],
                        "is_sam_refined": [],
                        "segmentation_masks": [],
                        "sam2_input_roi_xyxy": None,
                    }
                )
                boxes = tracking["boxes"]
                track_ids = tracking["track_ids"]
                seg_masks = tracking.get("segmentation_masks") or []
                if len(seg_masks) < len(boxes):
                    seg_masks = list(seg_masks) + [None] * (len(boxes) - len(seg_masks))
                frames_stored[frame_idx] = (frame_idx, None, boxes, track_ids)

                # V3.8: Extract appearance embeddings for Re-ID (red vs blue during occlusion)
                if len(boxes) > 0:
                    if vitpose_debug_enabled():
                        t_emb = time.perf_counter()
                        print(
                            f"  [phase5_dbg] frame {frame_idx + 1}: extract_embeddings start "
                            f"(n_boxes={len(boxes)})",
                            flush=True,
                        )
                    embeddings_by_frame[frame_idx] = extract_embeddings(
                        frame, boxes, track_ids, method="hsv_strip"
                    )
                    if vitpose_debug_enabled():
                        print(
                            f"  [phase5_dbg] frame {frame_idx + 1}: extract_embeddings done "
                            f"{(time.perf_counter() - t_emb) * 1000:.1f}ms",
                            flush=True,
                        )

                run_pose = (frame_idx % args.pose_stride == 0) and len(boxes) > 0
                if run_pose:
                    phase5_pose_passes += 1
                    pct = int(100 * (frame_idx + 1) / total_frames) if total_frames else 0
                    now = time.time()
                    wall_dt = now - last_phase5_wall_log
                    pct_milestone = pct >= last_pct + 5 or pct == 100
                    pass_milestone = phase5_pose_passes == 1 or (phase5_pose_passes % pose_log_every_n == 0)
                    time_heartbeat = wall_dt >= pose_log_every_sec
                    if pct_milestone or pass_milestone or time_heartbeat:
                        elapsed = now - t0
                        print(
                            f"  [phase5] {elapsed:.0f}s | frame {frame_idx + 1}/{total_frames} ({pct}%) "
                            f"| pose_pass {phase5_pose_passes} | tracks {len(boxes)}",
                            flush=True,
                        )
                        _lab_progress(
                            4,
                            "pose",
                            "Phase 5: Pose estimation",
                            status="running",
                            extra={
                                "frame": int(frame_idx + 1),
                                "total_frames": int(total_frames),
                                "pct": int(pct),
                                "pose_pass": int(phase5_pose_passes),
                                "tracks": int(len(boxes)),
                                "wall_s_in_phase": round(elapsed, 1),
                            },
                        )
                        last_phase5_wall_log = now
                        if pct_milestone:
                            last_pct = pct

                    # V3.4: Compute visibility scores to skip occluded tracks
                    vis_scores = compute_visibility_scores(boxes, track_ids)
                    if vitpose_debug_enabled():
                        print(
                            f"  [phase5_dbg] frame {frame_idx + 1}: visibility scores ready "
                            f"(n_tracks={len(boxes)})",
                            flush=True,
                        )

                    boxes_to_estimate = []
                    ids_to_estimate = []
                    paddings = []
                    masks_to_estimate: list = []
                    poses = {}

                    _smart_pose_crop = vitpose_smart_pad_enabled()
                    for i, (tid, box) in enumerate(zip(track_ids, boxes)):
                        m = seg_masks[i] if i < len(seg_masks) else None
                        pose_box = (
                            smart_expand_bbox_xyxy(
                                box,
                                prev_boxes.get(tid),
                                frame_width,
                                frame_height,
                            )
                            if _smart_pose_crop
                            else box
                        )
                        if (
                            pose_crop_smooth_alpha > 0
                            or pose_crop_foot_bias_frac > 0
                            or pose_crop_head_bias_frac > 0
                            or pose_crop_anti_jitter_px > 0
                        ):
                            pose_box = apply_temporal_pose_crop(
                                int(tid),
                                pose_box,
                                frame_w=frame_width,
                                frame_h=frame_height,
                                smooth_alpha=pose_crop_smooth_alpha,
                                foot_bias_frac=pose_crop_foot_bias_frac,
                                head_bias_frac=pose_crop_head_bias_frac,
                                anti_jitter_px=pose_crop_anti_jitter_px,
                                state=pose_crop_state,
                            )
                        # V3.4: Skip pose estimation for heavily occluded tracks
                        vis = vis_scores.get(tid, 1.0)
                        if vis < vis_skip:
                            # If tracker box jumped (ID switch), don't use last_good_pose — run pose
                            use_decayed = False
                            if tid in last_good_pose and tid in prev_boxes:
                                x1, y1, x2, y2 = box
                                px1, py1, px2, py2 = prev_boxes[tid]
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                                jump = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                                thresh = 0.5 * max(x2 - x1, y2 - y1, 1.0)
                                use_decayed = jump <= thresh
                            if use_decayed and tid in last_good_pose:
                                decayed = last_good_pose[tid].copy()
                                decayed["keypoints"] = decayed["keypoints"].copy()
                                decayed["keypoints"][:, 2] *= 0.85
                                decayed["scores"] = decayed["scores"].copy() * 0.85
                                poses[tid] = decayed
                                last_good_pose[tid] = decayed
                                occluded_skips += 1
                            else:
                                # Box jumped or no prev pose: run pose to avoid limbs outside bbox
                                boxes_to_estimate.append(pose_box)
                                ids_to_estimate.append(tid)
                                paddings.append(0.0 if _smart_pose_crop else 0.15)
                                masks_to_estimate.append(m)
                            continue

                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        w, h = x2 - x1, y2 - y1

                        if tid in prev_boxes and tid in prev_poses:
                            px1, py1, px2, py2 = prev_boxes[tid]
                            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                            pw, ph = px2 - px1, py2 - py1

                            dx = cx - pcx
                            dy = cy - pcy
                            dw = abs(w - pw)
                            dh = abs(h - ph)
                            movement = np.sqrt(dx**2 + dy**2)

                            if _smart_pose_crop:
                                pad = 0.0
                            else:
                                pad = 0.15
                                if movement > 15.0 or dw > 15.0 or dh > 15.0:
                                    pad = 0.25
                                elif movement < 6.0 and dw < 6.0 and dh < 6.0:
                                    pad = 0.10
                        else:
                            pad = 0.0 if _smart_pose_crop else 0.15

                        boxes_to_estimate.append(pose_box)
                        ids_to_estimate.append(tid)
                        paddings.append(pad)
                        masks_to_estimate.append(m)

                    if vitpose_debug_enabled():
                        print(
                            f"  [phase5_dbg] frame {frame_idx + 1}: boxes_to_estimate={len(boxes_to_estimate)}",
                            flush=True,
                        )

                    if boxes_to_estimate:
                        # Contiguous RGB avoids PIL/transformers edge cases with BGR ::-1 views.
                        frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
                        seg_arg = masks_to_estimate if any(x is not None for x in masks_to_estimate) else None
                        n_mask = sum(1 for x in (seg_arg or []) if x is not None)
                        if not phase5_logged_first_forward:
                            print(
                                f"  [phase5] first pose forward ({backend}): {len(boxes_to_estimate)} people "
                                f"({n_mask} mask-gated) — first batch often slow (graph compile on MPS/CUDA)…",
                                flush=True,
                            )
                            phase5_logged_first_forward = True
                        if vitpose_debug_enabled():
                            print(
                                f"  [phase5_dbg] frame {frame_idx + 1}: BGR→RGB done "
                                f"h×w={frame_rgb.shape[0]}×{frame_rgb.shape[1]} "
                                f"boxes_to_estimate={len(boxes_to_estimate)} seg_arg="
                                f"{'list' if seg_arg is not None else 'None'} n_sam_masks={n_mask}",
                                flush=True,
                            )
                            print(
                                f"  [phase5_dbg] frame {frame_idx + 1}: calling estimate_poses now…",
                                flush=True,
                            )
                        t_inf = time.perf_counter()
                        estimated = _call_with_pose_busy_heartbeat(
                            fn=lambda: pose_estimator.estimate_poses(
                                frame_rgb,
                                boxes_to_estimate,
                                ids_to_estimate,
                                paddings,
                                segmentation_masks=seg_arg,
                            ),
                            t0_phase5=t0,
                            frame_idx=frame_idx,
                            total_frames=total_frames,
                            phase5_pose_passes=phase5_pose_passes,
                            n_tracks=len(boxes),
                            pct=pct,
                        )
                        inf_dt = time.perf_counter() - t_inf
                        if vitpose_debug_enabled():
                            print(
                                f"  [phase5_dbg] frame {frame_idx + 1}: estimate_poses returned "
                                f"{(inf_dt * 1000):.1f}ms n_tracks_out={len(estimated)}",
                                flush=True,
                            )
                        if inf_dt >= pose_slow_sec:
                            print(
                                f"  [phase5] slow forward {inf_dt:.1f}s @ frame {frame_idx + 1} "
                                f"({len(boxes_to_estimate)} people, {n_mask} mask paths)",
                                flush=True,
                            )
                        for tid, est in estimated.items():
                            poses[tid] = est
                            last_good_pose[tid] = {
                                "keypoints": est["keypoints"].copy(),
                                "scores": est["scores"].copy(),
                            }

                    for tid, est in poses.items():
                        prev_poses[tid] = est
                    for tid, box in zip(track_ids, boxes):
                        prev_boxes[tid] = box
                else:
                    poses = {}

                raw_poses_by_frame[frame_idx] = poses

        if args.pose_stride > 1:
            _interpolate_pose_gaps(raw_poses_by_frame, frames_stored, args.pose_stride)

        if want_temporal_pose_refine(args.temporal_pose_refine):
            tr = temporal_pose_radius(args.temporal_pose_radius)
            print(f"  Temporal keypoint refine (±{tr} frames, confidence-weighted; not Poseidon)…")
            t_tr = time.time()
            apply_temporal_keypoint_smoothing(raw_poses_by_frame, radius=tr)
            print(f"  └─ {time.time() - t_tr:.2f}s")

        if occluded_skips:
            print(f"  Skipped {occluded_skips} occluded track-frames (visibility < {vis_skip})")
        dt_pose = time.time() - t0
        print(f"  └─ {dt_pose:.1f}s")
        log_resource_usage("5-pose")

        # Build all_frame_data_pre for downstream phases
        all_frame_data_pre = []
        for i, (fidx, _frame, boxes, track_ids) in enumerate(frames_stored):
            raw_poses = raw_poses_by_frame[i]
            emb = embeddings_by_frame[i] if i < len(embeddings_by_frame) else {}
            tr = tracking_results[i] if i < len(tracking_results) else {}
            sam_f = list(tr.get("is_sam_refined") or [])
            sam_m = list(tr.get("segmentation_masks") or [])
            nb = len(track_ids)
            sam_f = [bool(sam_f[j]) if j < len(sam_f) else False for j in range(nb)]
            sam_m = [sam_m[j] if j < len(sam_m) else None for j in range(nb)]
            all_frame_data_pre.append({
                "frame_idx": fidx,
                "frame": None,
                "boxes": list(boxes),
                "track_ids": list(track_ids),
                "poses": dict(raw_poses),
                "embeddings": emb,  # V3.8: Per-track appearance for crossover Re-ID
                "is_sam_refined": sam_f,
                "segmentation_masks": sam_m,
                "sam2_input_roi_xyxy": tr.get("sam2_input_roi_xyxy"),
            })

        _ex5 = {
            "vitpose_model_id": model_id,
            "pose_stride": int(args.pose_stride),
            "occluded_track_frames_skipped": int(occluded_skips),
        }

        prv3: Optional[str] = None
        if clip_dir is not None:
            p3 = clip_dir / "03_pose.mp4"
            montage_clips.append(render_phase_clip(
                video_path,
                all_frame_data_pre,
                "Phase 5: Pose estimation",
                lambda f, d: draw_frame_with_boxes(f, d["boxes"], d["track_ids"], d["poses"]),
                native_fps,
                output_fps,
                p3,
                caption="Boxes + skeleton (before re-ID / dedup)",
                full_length=True,
                show_title_card=False,
            ))
            prv3 = f"phase_previews/{p3.name}" if args.save_phase_previews else None
            _lab_progress(4, "pose", "Phase 5: Pose estimation", elapsed_s=dt_pose, preview_relpath=prv3, extra=_ex5)
            if prv3:
                _lab_register_preview("pose", prv3)
        else:
            _lab_progress(4, "pose", "Phase 5: Pose estimation", elapsed_s=dt_pose, extra=_ex5)

        phase_dbg.log("5", dt_pose, {"occluded_track_frames_skipped": int(occluded_skips)})
        if _stop_b == "after_phase_5":
            save_after_phase_5(
                _ck_dir("after_phase_5"),
                all_frame_data_pre=all_frame_data_pre,
                tracking_results=tracking_results,
                raw_poses_by_frame=raw_poses_by_frame,
                embeddings_by_frame=embeddings_by_frame,
                frames_stored=frames_stored,
                raw_tracks=raw_tracks,
                total_frames=total_frames,
                frame_width=frame_width,
                frame_height=frame_height,
                native_fps=native_fps,
                output_fps=output_fps,
                prune_log_entries=prune_log_entries,
                tracker_ids_before_prune=tracker_ids_before_prune,
                surviving_ids=surviving_ids,
                hybrid_sam_stats=hybrid_sam_stats,
                video_path=video_path,
                params_path=params_path,
            )
            print(f"Checkpoint written: {_ck_dir('after_phase_5')}", flush=True)
            _lab_update_context(
                checkpoint_boundary="after_phase_5",
                pipeline_diagnostics=_build_pipeline_diagnostics(hybrid_sam_stats, args, params),
            )
            rel5 = prv3 or ""
            _lab_write_manifest(
                video_path=video_path,
                output_dir=output_dir,
                args=args,
                params=params,
                model_id=model_id,
                total_elapsed=time.time() - total_start,
                final_video_relpath=rel5,
                view_variants={"full": rel5} if rel5 else {},
            )
            return
    elif start_marker == 6:
        # Resume from after_phase_5 (phase5.pkl.gz). Do not use load_after_phase_4 here.
        b5 = load_after_phase_5(resume_dir)
        all_frame_data_pre = b5["all_frame_data_pre"]
        tracking_results = b5["tracking_results"]
        raw_poses_by_frame = b5["raw_poses_by_frame"]
        embeddings_by_frame = b5["embeddings_by_frame"]
        frames_stored = b5["frames_stored"]
        raw_tracks = b5["raw_tracks"]
        total_frames = int(b5["total_frames"])
        frame_width = int(b5["frame_width"])
        frame_height = int(b5["frame_height"])
        native_fps = float(b5["native_fps"])
        output_fps = float(b5["output_fps"])
        prune_log_entries = list(b5["prune_log_entries"])
        tracker_ids_before_prune = list(b5["tracker_ids_before_prune"])
        surviving_ids = set(b5["surviving_ids"])
        hybrid_sam_stats = dict(b5.get("hybrid_sam_stats") or {})
        use_rtmpose = args.pose_model == "rtmpose"
        use_sapiens_requested = args.pose_model == "sapiens"
        env_pose = os.environ.get("SWAY_VITPOSE_MODEL", "").strip()
        if use_rtmpose:
            model_id = "rtmpose-l (MMPose)"
        elif use_sapiens_requested:
            model_id = _model_id_for_sapiens_choice(args)
        elif env_pose:
            model_id = env_pose
        elif args.pose_model == "huge":
            model_id = "usyd-community/vitpose-plus-huge"
        elif args.pose_model == "large":
            model_id = "usyd-community/vitpose-plus-large"
        else:
            model_id = "usyd-community/vitpose-plus-base"
        _lab_update_context(vitpose_model_id=model_id, pose_stride=int(args.pose_stride))
        occluded_skips = 0
        dt_pose = 0.0
        _ex5 = {
            "vitpose_model_id": model_id,
            "pose_stride": int(args.pose_stride),
            "occluded_track_frames_skipped": 0,
        }

    if start_marker < 9:
        # ── Phase 6: Association (occlusion re-ID before collision dedup) ──
        _prune_mark_ph67 = len(prune_log_entries)
        print("\n[6/11] Phase 6 — Association (occlusion re-ID, crossover, acceleration audit)…")
        t0 = time.time()
        _reid_kw = {}
        if "REID_MAX_FRAME_GAP" in params:
            _reid_kw["max_frame_gap"] = params["REID_MAX_FRAME_GAP"]
        if "REID_MIN_OKS" in params:
            _reid_kw["min_oks"] = params["REID_MIN_OKS"]
        _reid_kw["debug"] = bool(os.environ.get("SWAY_REID_DEBUG"))
        _reid_kw["total_frames"] = total_frames
        assoc_stop = threading.Event()
        if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
            _lab_progress(
                5,
                "association",
                "Phase 6: Association",
                status="running",
                extra={"wall_s_in_phase": 0.0, "step": "occlusion_reid_crossover_accel"},
            )

            def _assoc_pulse() -> None:
                while not assoc_stop.wait(timeout=12.0):
                    _lab_progress(
                        5,
                        "association",
                        "Phase 6: Association",
                        status="running",
                        extra={
                            "wall_s_in_phase": round(time.time() - t0, 1),
                            "step": "occlusion_reid_crossover_accel",
                        },
                    )

            threading.Thread(target=_assoc_pulse, daemon=True).start()
        try:
            apply_occlusion_reid(all_frame_data_pre, **_reid_kw)
            apply_crossover_refinement(all_frame_data_pre, frame_width=frame_width, frame_height=frame_height)
            apply_acceleration_audit(all_frame_data_pre)
        finally:
            assoc_stop.set()
        dt_association = time.time() - t0
        print(f"  └─ Phase 6: {dt_association:.1f}s")

        # ── Phase 7: Collision cleanup (keypoint dedup + bbox sanitize) ──────
        print("\n[7/11] Phase 7 — Collision cleanup (keypoint dedup, bbox sanitize)…")
        t0 = time.time()
        _ph7_n = len(all_frame_data_pre)
        _ph7_stride = max(1, _ph7_n // 25) if _ph7_n > 0 else 1
        _ensure_collision_cleanup_logging()
        # V3.8: Removed late_entrant_candidates protection — ghosts (e.g. 61) were exempt from
        # post-pose prunes and dedup. Now all tracks subject to same prune/dedup rules.
        late_entrant_candidates = set()
        # Build track frame count for dedup: when two tracks overlap, keep the longer one
        track_frame_count: dict[int, int] = {}
        for fd in all_frame_data_pre:
            for tid in fd.get("track_ids", []):
                track_frame_count[tid] = track_frame_count.get(tid, 0) + 1
        _dedup_kpt_frac = float(params.get("COLLISION_KPT_DIST_FRAC", COLLISION_KPT_DIST_FRAC))
        _dedup_ctr_frac = float(params.get("COLLISION_CENTER_DIST_FRAC", COLLISION_CENTER_DIST_FRAC))
        _dedup_ap_iou = float(params.get("DEDUP_ANTIPARTNER_MIN_IOU", DEDUP_ANTIPARTNER_MIN_IOU))
        _dedup_kpt_tight = float(params.get("DEDUP_KPT_TIGHT_FRAC", DEDUP_KPT_TIGHT_FRAC))
        _dedup_torso = float(params.get("DEDUP_TORSO_MEDIAN_FRAC", DEDUP_TORSO_MEDIAN_FRAC))
        _dedup_pair_oks = float(params.get("DEDUP_MIN_PAIR_OKS", DEDUP_MIN_PAIR_OKS))
        dedup_count = 0
        sanitize_count = 0
        sanitize_keypoints_zeroed = 0
        phase6_log: list = []
        snap_pre_dedup = [snapshot_tid_box_map(fd) for fd in all_frame_data_pre]
        for _pi, fd in enumerate(all_frame_data_pre):
            if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
                if _pi % _ph7_stride == 0 or _pi == _ph7_n - 1:
                    _lab_progress(
                        6,
                        "collision_cleanup",
                        "Phase 7: Collision cleanup",
                        status="running",
                        extra={
                            "step": "deduplicate_collocated_poses",
                            "frame": int(_pi + 1),
                            "total_frames": int(_ph7_n),
                            "pct": int(100 * (_pi + 1) / _ph7_n) if _ph7_n else 0,
                            "wall_s_in_phase": round(time.time() - t0, 1),
                        },
                    )
            before = len(fd["poses"])
            deduplicate_collocated_poses(
                fd,
                kpt_dist_frac=_dedup_kpt_frac,
                center_dist_frac=_dedup_ctr_frac,
                dedup_antipartner_min_iou=_dedup_ap_iou,
                dedup_kpt_tight_frac=_dedup_kpt_tight,
                dedup_torso_median_frac=_dedup_torso,
                dedup_min_pair_oks=_dedup_pair_oks,
                protected_tids=late_entrant_candidates,
                track_frame_count=track_frame_count,
                phase6_log=phase6_log,
            )
            dedup_count += before - len(fd["poses"])
        snap_post_dedup_pre_sanitize = [snapshot_tid_box_map(fd) for fd in all_frame_data_pre]
        for _si, fd in enumerate(all_frame_data_pre):
            if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
                if _si % _ph7_stride == 0 or _si == _ph7_n - 1:
                    _lab_progress(
                        6,
                        "collision_cleanup",
                        "Phase 7: Collision cleanup",
                        status="running",
                        extra={
                            "step": "sanitize_pose_bbox",
                            "frame": int(_si + 1),
                            "total_frames": int(_ph7_n),
                            "pct": int(100 * (_si + 1) / _ph7_n) if _ph7_n else 0,
                            "wall_s_in_phase": round(time.time() - t0, 1),
                        },
                    )
            n_pose, n_kpt = sanitize_pose_bbox_consistency(fd, phase6_log=phase6_log)
            sanitize_count += n_pose
            sanitize_keypoints_zeroed += n_kpt
        if dedup_count:
            print(f"  Removed {dedup_count} duplicate pose overlays")
        if sanitize_count:
            print(f"  Sanitized {sanitize_count} poses with keypoints outside bbox")
        if sanitize_keypoints_zeroed:
            print(
                f"  Zeroed {sanitize_keypoints_zeroed} limb keypoint confidences "
                f"(outside bbox; not recorded in prune_log_entries)"
            )
        if phase6_log:
            prune_log_entries.extend(phase6_log)
        prune_log_entries.append(
            {
                "rule": "phase6_summary",
                "dedup_removed_poses": int(dedup_count),
                "sanitize_removed_poses": int(sanitize_count),
                "sanitize_keypoints_zeroed": int(sanitize_keypoints_zeroed),
                "per_event_log_count": len(phase6_log),
            }
        )
        dt_collision = time.time() - t0
        print(f"  └─ Phase 7: {dt_collision:.1f}s")
        log_resource_usage("6-association")
        log_resource_usage("7-collision-cleanup")

        _ex67 = {
            **_prune_events_extra(prune_log_entries, _prune_mark_ph67),
            "dedup_removed_pose_rows": int(dedup_count),
            "sanitize_removed_poses": int(sanitize_count),
            "sanitize_keypoints_zeroed": int(sanitize_keypoints_zeroed),
            "phase6_event_log_rows": len(phase6_log),
        }
        _prune_mark = len(prune_log_entries)

        if clip_dir is not None:
            p4 = clip_dir / "04_phases_6_7.mp4"
            _ov_ph67 = build_prune_overlay_index(
                prune_log_entries,
                COLLISION_PREVIEW_RULES,
                max(0, int(total_frames) - 1),
            )
            _draw_ph67 = wrap_draw_fn_with_prune_overlays(
                lambda f, d: draw_frame_with_boxes(f, d["boxes"], d["track_ids"], d["poses"]),
                _ov_ph67,
            )
            montage_clips.append(render_phase_clip(
                video_path,
                all_frame_data_pre,
                "Phases 6–7: Association & collision cleanup",
                _draw_ph67,
                native_fps,
                output_fps,
                p4,
                caption="Poses + dedup/sanitize highlights burned in",
                full_length=True,
                show_title_card=False,
            ))
            prv4 = f"phase_previews/{p4.name}" if args.save_phase_previews else None
            _lab_progress(
                5,
                "association",
                "Phase 6: Association",
                elapsed_s=dt_association,
                extra={
                    "steps": ["occlusion_reid", "crossover_refinement", "acceleration_audit"],
                    "reid_max_frame_gap": _reid_kw.get("max_frame_gap"),
                    "reid_min_oks": _reid_kw.get("min_oks"),
                },
            )
            _lab_progress(
                6,
                "collision_cleanup",
                "Phase 7: Collision cleanup",
                elapsed_s=dt_collision,
                preview_relpath=prv4,
                extra=_ex67,
            )
            if prv4:
                _lab_register_preview("phases_6_7", prv4)
        else:
            _lab_progress(
                5,
                "association",
                "Phase 6: Association",
                elapsed_s=dt_association,
                extra={
                    "steps": ["occlusion_reid", "crossover_refinement", "acceleration_audit"],
                    "reid_max_frame_gap": _reid_kw.get("max_frame_gap"),
                    "reid_min_oks": _reid_kw.get("min_oks"),
                },
            )
            _lab_progress(6, "collision_cleanup", "Phase 7: Collision cleanup", elapsed_s=dt_collision, extra=_ex67)

        # ── Phase 8: Post-pose pruning — Tier C, Tier A whitelist, Tier B vote ─
        _prune_mark_ph8 = len(prune_log_entries)
        print("\n[8/11] Phase 8 — Post-pose pruning (Tier C auto-reject, Tier B weighted rules)…")
        t0 = time.time()

        _edge_m = float(params.get("EDGE_MARGIN_FRAC", 0.15))
        _ch_span = float(params.get("CONFIRMED_HUMAN_MIN_SPAN_FRAC", 0.10))
        confirmed_humans = compute_confirmed_human_set(
            all_frame_data_pre,
            total_frames,
            frame_width=frame_width,
            edge_margin_frac=_edge_m,
            min_span_frac=_ch_span,
        )
        phase7_poses_by_frame = [fd["poses"] for fd in all_frame_data_pre]

        _tier_c_kw = {}
        if "TIER_C_SKELETON_MEAN" in params:
            _tier_c_kw["mean_thresh"] = float(params["TIER_C_SKELETON_MEAN"])
        if "TIER_C_LOW_FRAME_FRAC" in params:
            _tier_c_kw["min_low_frac"] = float(params["TIER_C_LOW_FRAME_FRAC"])
        tier_c_ids = prune_ultra_low_skeleton_tracks(
            surviving_ids,
            phase7_poses_by_frame,
            **_tier_c_kw,
        )
        if tier_c_ids:
            print(f"  Pruned {len(tier_c_ids)} tracks (Tier C: no confident skeleton)")
            _tcm = float(_tier_c_kw.get("mean_thresh", ULTRA_LOW_SKELETON_MEAN))
            _tcl = float(_tier_c_kw.get("min_low_frac", ULTRA_LOW_SKELETON_FRAME_FRAC))
            log_pruned_tracks(
                raw_tracks,
                tier_c_ids,
                "tier_c_auto_reject",
                frame_width,
                frame_height,
                prune_log_entries,
                total_frames=total_frames,
                cause_config=prune_cause_config(
                    "tier_c_auto_reject",
                    "Phase 8 Tier C: mean keypoint confidence below TIER_C_SKELETON_MEAN and/or "
                    "fraction of low-confidence frames above TIER_C_LOW_FRAME_FRAC.",
                    {"TIER_C_SKELETON_MEAN": _tcm, "TIER_C_LOW_FRAME_FRAC": _tcl},
                ),
            )

        surviving_after_tier_c = surviving_ids - tier_c_ids
        _p7_vote_weights = dict(PRUNING_WEIGHTS)
        if isinstance(params.get("PRUNING_WEIGHTS"), dict):
            _p7_vote_weights.update({str(k): float(v) for k, v in params["PRUNING_WEIGHTS"].items()})
        _prune_threshold = float(params.get("PRUNE_THRESHOLD", PRUNE_THRESHOLD))

        _sync_min = float(params.get("SYNC_SCORE_MIN", 0.10))
        _mirror_edge = float(params.get("EDGE_MARGIN_FRAC", 0.15))
        _mirror_pres = float(params.get("EDGE_PRESENCE_FRAC", 0.3))
        _mirror_lb = float(params.get("min_lower_body_conf", 0.3))
        _mean_conf_min = float(params.get("MEAN_CONFIDENCE_MIN", 0.45))
        _jitter_max = float(params.get("JITTER_RATIO_MAX", 0.10))

        ph8_stop = threading.Event()
        if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
            _lab_progress(
                7,
                "post_pose_prune",
                "Phase 8: Post-pose pruning",
                status="running",
                extra={"wall_s_in_phase": 0.0, "step": "tier_b_voting"},
            )

            def _ph8_pulse() -> None:
                while not ph8_stop.wait(timeout=12.0):
                    _lab_progress(
                        7,
                        "post_pose_prune",
                        "Phase 8: Post-pose pruning",
                        status="running",
                        extra={
                            "wall_s_in_phase": round(time.time() - t0, 1),
                            "step": "tier_b_voting",
                        },
                    )

            threading.Thread(target=_ph8_pulse, daemon=True).start()
        try:
            voting_prune_ids, _tier_b_telemetry = compute_phase7_voting_prune_set(
                all_frame_data_pre,
                surviving_after_tier_c,
                raw_tracks,
                phase7_poses_by_frame,
                frame_width,
                frame_height,
                total_frames,
                confirmed_humans,
                _p7_vote_weights,
                _prune_threshold,
                min_sync_score=_sync_min,
                edge_margin_frac=_mirror_edge,
                edge_presence_frac=_mirror_pres,
                min_lower_body_conf=_mirror_lb,
                min_mean_conf=_mean_conf_min,
                max_jitter=_jitter_max,
                phase7_prune_log=prune_log_entries,
                phase8_log_context={
                    "SYNC_SCORE_MIN": _sync_min,
                    "EDGE_MARGIN_FRAC": _mirror_edge,
                    "EDGE_PRESENCE_FRAC": _mirror_pres,
                    "min_lower_body_conf": _mirror_lb,
                    "MEAN_CONFIDENCE_MIN": _mean_conf_min,
                    "JITTER_RATIO_MAX": _jitter_max,
                },
            )
        finally:
            ph8_stop.set()
        if voting_prune_ids:
            print(
                f"  Pruned {len(voting_prune_ids)} tracks (Tier B voting, "
                f"threshold={_prune_threshold})"
            )
            for tid in sorted(voting_prune_ids):
                te = _tier_b_telemetry.get(tid, {})
                rs = te.get("rule_hits") or {}
                hits = [k.replace("prune_", "").replace("_tracks", "") for k, v in rs.items() if v > 0]
                hit_s = ",".join(hits) if hits else "?"
                wsum = te.get("weighted_sum", "?")
                print(f"    [Tier B] track {tid}: weighted_sum={wsum} hits=[{hit_s}]")

        phase7_prune_ids = tier_c_ids | voting_prune_ids
        surviving_after_prune = set()
        for fd in all_frame_data_pre:
            surviving_after_prune.update(t for t in fd["track_ids"] if t not in phase7_prune_ids)
        print(f"  {len(surviving_after_prune)} tracks after post-pose pruning")
        _lab_update_context(
            surviving_after_post_pose_prune=int(len(surviving_after_prune)),
            post_pose_tier_c=int(len(tier_c_ids)),
            post_pose_voting_pruned=int(len(voting_prune_ids)),
        )
        dt_postprune = time.time() - t0
        print(f"  └─ {dt_postprune:.1f}s")
        log_resource_usage("8-post-pose-prune")

        postprune_fd = []
        for fd_pre in all_frame_data_pre:
            filt_poses = {tid: d for tid, d in fd_pre["poses"].items() if tid not in phase7_prune_ids}
            filt_boxes = [b for b, tid in zip(fd_pre["boxes"], fd_pre["track_ids"]) if tid not in phase7_prune_ids and tid in filt_poses]
            filt_tids = [tid for tid in fd_pre["track_ids"] if tid not in phase7_prune_ids and tid in filt_poses]
            postprune_fd.append({"frame_idx": fd_pre["frame_idx"], "boxes": filt_boxes,
                                  "track_ids": filt_tids, "poses": filt_poses})
        _ex8 = {
            **_prune_events_extra(prune_log_entries, _prune_mark_ph8),
            "tier_c_pruned": len(tier_c_ids),
            "tier_b_voting_pruned": len(voting_prune_ids),
            "prune_threshold": _prune_threshold,
            "tracks_after_post_pose_prune": len(surviving_after_prune),
        }
        _prune_mark = len(prune_log_entries)

        prv5: Optional[str] = None
        if clip_dir is not None:
            p5 = clip_dir / "05_post_pose_prune.mp4"
            _fd_by_fi = {int(fd["frame_idx"]): fd for fd in all_frame_data_pre}
            _ov_post = build_prune_overlay_index(
                prune_log_entries,
                POST_POSE_PREVIEW_RULES,
                max(0, int(total_frames) - 1),
                raw_tracks=raw_tracks,
                frame_data_by_idx=_fd_by_fi,
            )
            _draw_post = wrap_draw_fn_with_prune_overlays(
                lambda f, d: draw_boxes_only(f, d["boxes"], d["track_ids"]),
                _ov_post,
            )
            montage_clips.append(render_phase_clip(
                video_path, postprune_fd, "Phase 8: After post-pose prune",
                _draw_post,
                native_fps, output_fps, p5,
                caption="Surviving boxes + Tier B/C prune highlights burned in",
                full_length=True,
                show_title_card=False,
            ))
            prv5 = f"phase_previews/{p5.name}" if args.save_phase_previews else None
            _lab_progress(7, "post_pose_prune", "Phase 8: Post-pose pruning", elapsed_s=dt_postprune, preview_relpath=prv5, extra=_ex8)
            if prv5:
                _lab_register_preview("post_pose_prune", prv5)
        else:
            _lab_progress(7, "post_pose_prune", "Phase 8: Post-pose pruning", elapsed_s=dt_postprune, extra=_ex8)

        phase_dbg.log(
            "8",
            dt_postprune,
            {"tracks_after_post_pose_prune": len(surviving_after_prune)},
        )
        if _stop_b == "after_phase_8":
            save_after_phase_8(
                _ck_dir("after_phase_8"),
                all_frame_data_pre=all_frame_data_pre,
                phase7_prune_ids=phase7_prune_ids,
                raw_tracks=raw_tracks,
                total_frames=total_frames,
                frame_width=frame_width,
                frame_height=frame_height,
                native_fps=native_fps,
                output_fps=output_fps,
                prune_log_entries=prune_log_entries,
                tracker_ids_before_prune=tracker_ids_before_prune,
                surviving_ids=surviving_ids,
                hybrid_sam_stats=hybrid_sam_stats,
                snap_pre_dedup=snap_pre_dedup,
                snap_post_dedup_pre_sanitize=snap_post_dedup_pre_sanitize,
                video_path=video_path,
                params_path=params_path,
            )
            print(f"Checkpoint written: {_ck_dir('after_phase_8')}", flush=True)
            _lab_update_context(
                checkpoint_boundary="after_phase_8",
                pipeline_diagnostics=_build_pipeline_diagnostics(hybrid_sam_stats, args, params),
            )
            rel8 = prv5 or ""
            _lab_write_manifest(
                video_path=video_path,
                output_dir=output_dir,
                args=args,
                params=params,
                model_id=model_id,
                total_elapsed=time.time() - total_start,
                final_video_relpath=rel8,
                view_variants={"full": rel8} if rel8 else {},
            )
            return
    else:
        b8 = load_after_phase_8(resume_dir)
        all_frame_data_pre = b8["all_frame_data_pre"]
        phase7_prune_ids = set(b8["phase7_prune_ids"])
        raw_tracks = b8["raw_tracks"]
        total_frames = int(b8["total_frames"])
        frame_width = int(b8["frame_width"])
        frame_height = int(b8["frame_height"])
        native_fps = float(b8["native_fps"])
        output_fps = float(b8["output_fps"])
        prune_log_entries = list(b8["prune_log_entries"])
        tracker_ids_before_prune = list(b8["tracker_ids_before_prune"])
        surviving_ids = set(b8["surviving_ids"])
        hybrid_sam_stats = dict(b8.get("hybrid_sam_stats") or {})
        snap_pre_dedup = list(b8.get("snap_pre_dedup") or [])
        snap_post_dedup_pre_sanitize = list(b8.get("snap_post_dedup_pre_sanitize") or [])
        surviving_after_prune = set()
        for fd in all_frame_data_pre:
            surviving_after_prune.update(t for t in fd["track_ids"] if t not in phase7_prune_ids)
        tier_c_ids = set()
        voting_prune_ids = set()
        dt_postprune = 0.0
        dt_association = 0.0
        dt_collision = 0.0
        _ex8 = {}
        _ex67 = {}
        use_rtmpose = args.pose_model == "rtmpose"
        use_sapiens_requested = args.pose_model == "sapiens"
        env_pose = os.environ.get("SWAY_VITPOSE_MODEL", "").strip()
        if use_rtmpose:
            model_id = "rtmpose-l (MMPose)"
        elif use_sapiens_requested:
            model_id = _model_id_for_sapiens_choice(args)
        elif env_pose:
            model_id = env_pose
        elif args.pose_model == "huge":
            model_id = "usyd-community/vitpose-plus-huge"
        elif args.pose_model == "large":
            model_id = "usyd-community/vitpose-plus-large"
        else:
            model_id = "usyd-community/vitpose-plus-base"
        _lab_update_context(vitpose_model_id=model_id, pose_stride=int(args.pose_stride))

    # ── Phase 9: Temporal smoothing (1 Euro, conf<0.3 guard) ────────────
    print("\n[9/11] Phase 9 — Temporal smoothing (1 Euro filter)…")
    t0 = time.time()
    sm_cut = float(params.get("SMOOTHER_MIN_CUTOFF", 1.0))
    sm_beta = float(params.get("SMOOTHER_BETA", 0.7))
    smoother = PoseSmoother(min_cutoff=sm_cut, beta=sm_beta)
    all_frame_data = []
    _smooth_n = len(all_frame_data_pre)
    _smooth_stride = max(1, _smooth_n // 30) if _smooth_n > 0 else 1

    for i, fd_pre in enumerate(all_frame_data_pre):
        if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
            if i % _smooth_stride == 0 or i == _smooth_n - 1:
                _lab_progress(
                    8,
                    "smooth",
                    "Phase 9: Temporal smoothing",
                    status="running",
                    extra={
                        "frame": int(i + 1),
                        "total_frames": int(_smooth_n),
                        "pct": int(100 * (i + 1) / _smooth_n) if _smooth_n else 0,
                        "wall_s_in_phase": round(time.time() - t0, 1),
                    },
                )
        fidx = fd_pre["frame_idx"]
        boxes = fd_pre["boxes"]
        track_ids = fd_pre["track_ids"]
        poses_raw = fd_pre["poses"]

        poses_filtered = {tid: data for tid, data in poses_raw.items() if tid not in phase7_prune_ids}
        # Filter out pruned tracks AND tracks that have no pose data (stripped by dedup)
        boxes_filtered = [b for b, tid in zip(boxes, track_ids) if tid not in phase7_prune_ids and tid in poses_filtered]
        track_ids_filtered = [tid for tid in track_ids if tid not in phase7_prune_ids and tid in poses_filtered]
        sam_f = fd_pre.get("is_sam_refined") or []
        sam_m = fd_pre.get("segmentation_masks") or []
        sam_f_f = []
        sam_m_f = []
        for j, (b, tid) in enumerate(zip(boxes, track_ids)):
            if tid not in phase7_prune_ids and tid in poses_filtered:
                sam_f_f.append(bool(sam_f[j]) if j < len(sam_f) else False)
                sam_m_f.append(sam_m[j] if j < len(sam_m) else None)

        frame_time = fidx / output_fps
        smoothed_poses = smoother.smooth_frame(poses_filtered, frame_time)

        all_frame_data.append({
            "frame_idx": fidx,
            "frame": None,
            "boxes": boxes_filtered,
            "track_ids": track_ids_filtered,
            "poses": smoothed_poses,
            "track_angles": {},
            "consensus_angles": {},
            "deviations": {},
            "is_sam_refined": sam_f_f,
            "segmentation_masks": sam_m_f,
        })

    dt_smooth = time.time() - t0
    print(f"  └─ {dt_smooth:.1f}s")
    log_resource_usage("9-smoothing")

    _ex9 = {"smoother_min_cutoff": sm_cut, "smoother_beta": sm_beta}

    if clip_dir is not None:
        p6 = clip_dir / "06_smooth.mp4"
        montage_clips.append(render_phase_clip(
            video_path,
            all_frame_data,
            "Phase 9: Smoothed poses",
            lambda f, d: draw_frame_with_boxes(f, d["boxes"], d["track_ids"], d["poses"]),
            native_fps,
            output_fps,
            p6,
            caption="After temporal smoothing",
            full_length=True,
            show_title_card=False,
        ))
        prv6 = f"phase_previews/{p6.name}" if args.save_phase_previews else None
        _lab_progress(8, "smooth", "Phase 9: Temporal smoothing", elapsed_s=dt_smooth, preview_relpath=prv6, extra=_ex9)
        if prv6:
            _lab_register_preview("smooth", prv6)
    else:
        _lab_progress(8, "smooth", "Phase 9: Temporal smoothing", elapsed_s=dt_smooth, extra=_ex9)

    want_3d_lift = os.environ.get("SWAY_3D_LIFT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ) and not bool(getattr(args, "no_pose_3d_lift", False))
    pose_3d_blob: Optional[Dict[str, Any]] = None
    lift_export_depth_series: Optional[list] = None
    if want_3d_lift:
        try:
            from sway.depth_stage import collect_strided_depth_series, get_depth_array
            from sway.pose_lift_3d import export_3d_for_viewer, lift_poses_to_3d
            from sway.video_camera_probe import probe_intrinsics_from_video

            video_camera = probe_intrinsics_from_video(
                str(video_path), int(frame_width), int(frame_height)
            )
            if video_camera:
                feq = video_camera.get("focal_length_35mm_equiv_mm")
                sk = video_camera.get("source_key", "")
                print(
                    f"  [3D Lift] Camera from video metadata ({sk}): "
                    f"{feq:.1f}mm equiv → fx={video_camera['fx']:.1f}, fy={video_camera['fy']:.1f}",
                    flush=True,
                )
            else:
                fov_guess = os.environ.get("SWAY_PINHOLE_FOV_DEG", "70")
                print(
                    f"  [3D Lift] No ffprobe 35mm-equivalent tags — using default pinhole FOV "
                    f"({fov_guess}°, SWAY_PINHOLE_FOV_DEG). Set SWAY_FX and SWAY_FY for accurate "
                    f"world layout, or fix container metadata.",
                    flush=True,
                )

            depth_dynamic = os.environ.get("SWAY_DEPTH_DYNAMIC", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            stride_s = os.environ.get("SWAY_DEPTH_STRIDE_FRAMES", "").strip()
            depth_stride = int(stride_s) if stride_s else max(1, int(round(float(output_fps))))

            depth_series = None
            if depth_dynamic:
                depth_series = collect_strided_depth_series(
                    iter_video_frames(str(video_path)),
                    depth_stride,
                )
            if not depth_series:
                depth_series = []
                for fi, fr in iter_video_frames(str(video_path)):
                    d = get_depth_array(fr)
                    if d is not None:
                        depth_series.append((int(fi), d))
                    break

            lift_export_depth_series = depth_series if depth_series else None

            lift_poses_to_3d(
                all_frame_data,
                total_frames,
                frame_width,
                frame_height,
                depth_series=lift_export_depth_series,
                video_camera=video_camera,
            )
            tids_final = sorted(
                surviving_after_prune,
                key=lambda x: int(x) if isinstance(x, int) else 0,
            )
            pose_3d_blob = export_3d_for_viewer(
                all_frame_data,
                tids_final,
                len(all_frame_data),
                float(output_fps),
                frame_width,
                frame_height,
                video_camera=video_camera,
            )
        except Exception as ex:
            print(f"  [3D Lift] Skipped: {ex}", flush=True)

    # ── Phase 10: Spatio-temporal scoring (circmean, cDTW, per-joint) ───
    print("\n[10/11] Phase 10 — Spatio-temporal scoring…")
    t0 = time.time()
    score_stop = threading.Event()
    if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
        _lab_progress(
            9,
            "scoring",
            "Phase 10: Scoring",
            status="running",
            extra={"wall_s_in_phase": 0.0, "step": "vectorized_scoring"},
        )

        def _score_pulse() -> None:
            while not score_stop.wait(timeout=10.0):
                _lab_progress(
                    9,
                    "scoring",
                    "Phase 10: Scoring",
                    status="running",
                    extra={
                        "wall_s_in_phase": round(time.time() - t0, 1),
                        "step": "vectorized_scoring",
                    },
                )

        threading.Thread(target=_score_pulse, daemon=True).start()
    try:
        scoring_data = process_all_frames_scoring_vectorized(all_frame_data)
    finally:
        score_stop.set()
    if scoring_data is not None:
        for i, fd in enumerate(all_frame_data):
            fd["track_angles"] = scoring_data["track_angles"][i]
            fd["consensus_angles"] = scoring_data["consensus_angles"][i]
            fd["deviations"] = scoring_data["deviations"][i]
            fd["shape_errors"] = scoring_data.get("shape_errors", [{} for _ in all_frame_data])[i]
            fd["timing_errors"] = scoring_data.get("timing_errors", [{} for _ in all_frame_data])[i]

    dt_score = time.time() - t0
    print(f"  └─ {dt_score:.1f}s")
    log_resource_usage("10-scoring")

    _ex10 = {"scoring_enabled": scoring_data is not None}

    if clip_dir is not None:
        p7 = clip_dir / "07_scoring.mp4"
        montage_clips.append(render_phase_clip(
            video_path,
            all_frame_data,
            "Phase 10: Scoring",
            lambda f, d: draw_frame(
                f,
                d["boxes"],
                d["track_ids"],
                d["poses"],
                deviations=d.get("deviations"),
                shape_errors=d.get("shape_errors"),
                timing_errors=d.get("timing_errors"),
            ),
            native_fps,
            output_fps,
            p7,
            caption="Heatmap skeletons (scored)",
            full_length=True,
            show_title_card=False,
        ))
        prv7 = f"phase_previews/{p7.name}" if args.save_phase_previews else None
        _lab_progress(9, "scoring", "Phase 10: Scoring", elapsed_s=dt_score, preview_relpath=prv7, extra=_ex10)
        if prv7:
            _lab_register_preview("scoring", prv7)
    else:
        _lab_progress(9, "scoring", "Phase 10: Scoring", elapsed_s=dt_score, extra=_ex10)

    # ── Export (after Phase 10) ───────────────────────────────────────────
    print("\n[11/11] Export — rendering and writing outputs…")
    t0 = time.time()
    pruned_overlay = build_pruned_overlay_for_review(
        all_frame_data_pre,
        phase7_prune_ids,
        raw_tracks,
        prune_log_entries,
    )
    dropped_pose_overlay = build_dropped_pose_overlay(
        snap_pre_dedup,
        snap_post_dedup_pre_sanitize,
        all_frame_data_pre,
    )
    _export_lab_cb = None
    if _LAB_CTX is not None and _LAB_CTX.get("progress_jsonl"):
        t_export0 = time.perf_counter()

        def _export_lab(extra: Dict[str, Any]) -> None:
            merged = dict(extra)
            merged["wall_s_in_phase"] = round(time.perf_counter() - t_export0, 1)
            _lab_progress(10, "export", "Export", status="running", extra=merged)

        _export_lab_cb = _export_lab
    view_variants = render_and_export(
        video_path=video_path,
        all_frame_data=all_frame_data,
        processed_fps=output_fps,
        native_fps=native_fps,
        output_dir=output_dir,
        pruned_overlay=pruned_overlay,
        prune_entries=prune_log_entries,
        dropped_pose_overlay=dropped_pose_overlay,
        pose_3d=pose_3d_blob,
        lab_export_progress=_export_lab_cb,
        lift_depth_series=lift_export_depth_series,
    )
    dt_export = time.time() - t0
    print(f"  └─ {dt_export:.1f}s")
    log_resource_usage("11-export")

    final_video_relpath = f"{video_path.stem}_poses.mp4"
    _lab_progress(
        10,
        "export",
        "Export",
        elapsed_s=dt_export,
        preview_relpath=final_video_relpath,
        extra={
            "render_variants": list(view_variants.keys()) if view_variants else [],
            "prune_log_file": "prune_log.json",
            "data_json": "data.json",
        },
    )

    if args.montage and montage_clips:
        print("\n  Stitching montage...")
        stitch_montage(montage_clips, output_dir / "montage.mp4", native_fps)
    if montage_dir is not None:
        import shutil as _shutil
        _shutil.rmtree(montage_dir, ignore_errors=True)

    total_elapsed = time.time() - total_start
    print(f"\nDone. Outputs in {output_dir}")
    print(f"Total pipeline: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log_resource_usage("final")

    _lab_update_context(total_pipeline_wall_s=round(total_elapsed, 3))
    write_hmr_mesh_sidecar_json(output_dir)
    _lab_update_context(
        pipeline_diagnostics=_build_pipeline_diagnostics(hybrid_sam_stats, args, params),
    )
    _lab_progress(
        11,
        "pipeline_total",
        "End-to-end wall time (all stages)",
        elapsed_s=round(total_elapsed, 3),
        extra={
            "output_dir": str(output_dir.resolve()),
            "final_mp4": final_video_relpath,
        },
    )

    _lab_write_manifest(
        video_path=video_path,
        output_dir=output_dir,
        args=args,
        params=params,
        model_id=model_id,
        total_elapsed=total_elapsed,
        final_video_relpath=final_video_relpath,
        view_variants=view_variants,
    )

    save_final_marker(
        output_dir,
        video_path=video_path,
        params_path=params_path,
        extra={"total_elapsed_s": round(total_elapsed, 3)},
    )

    # Write prune diagnostic log for debugging
    prune_log_path = output_dir / "prune_log.json"
    try:
        _sam_roi_frames = raw_tracks_to_per_frame(
            raw_tracks, total_frames, set(raw_tracks.keys())
        )
        hybrid_sam_frame_rois = [
            {"frame_idx": i, "roi_xyxy": t["sam2_input_roi_xyxy"]}
            for i, t in enumerate(_sam_roi_frames)
            if t.get("sam2_input_roi_xyxy") is not None
        ]
        prune_log_data = {
            "video_path": str(video_path),
            "native_fps": round(float(native_fps), 4),
            "total_frames": total_frames,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "tracker": {
                "track_ids_before_prune": tracker_ids_before_prune,
                "count": len(tracker_ids_before_prune),
            },
            "surviving_after_pre_pose": sorted(surviving_ids, key=lambda x: int(x) if isinstance(x, int) else 999),
            "surviving_after_post_pose": sorted(surviving_after_prune, key=lambda x: int(x) if isinstance(x, int) else 999),
            "hybrid_sam_frame_rois": hybrid_sam_frame_rois,
            "prune_entries": prune_log_entries,
        }
        with open(prune_log_path, "w") as f:
            json.dump(prune_log_data, f, indent=2)
        print(f"  Diagnostic log: {prune_log_path}")
    except Exception as e:
        print(f"  Could not write prune log: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
