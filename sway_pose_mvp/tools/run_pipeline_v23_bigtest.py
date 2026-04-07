"""
Run the current SWAY pipeline on BigTest video with per-phase video export.

Produces one annotated video per layer:
  output/bigtest_v23/phase1_detection.mp4
  output/bigtest_v23/phase2_masks.mp4
  output/bigtest_v23/phase3_vitpose.mp4
  output/bigtest_v23/phase3_vitpose_overlay.json  (ViTPose keypoints + viz knobs; for re-rendering MP4)
  output/bigtest_v23/phase3_rtmwx.mp4
  output/bigtest_v23/phase4_tracking_forward.mp4
  output/bigtest_v23/phase4_tracking_bidirectional.mp4
  output/bigtest_v23/phase6_reid_fusion.mp4
  output/bigtest_v23/phase7_darkzone_resolution.mp4
  output/bigtest_v23/phase7_legacy_pose3d_compat.mp4
  output/bigtest_v23/phase8_final_optimized.mp4

Usage:
  source .plan_env && python -m tools.run_pipeline_v23_bigtest
"""
import _repo_path  # noqa: F401

import inspect
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

# Protobuf v5 removed MessageFactory.GetPrototype; several downstream deps still call it.
try:
    from google.protobuf import message_factory as _pb_message_factory

    if not hasattr(_pb_message_factory.MessageFactory, "GetPrototype"):
        def _compat_get_prototype(self, descriptor):
            return _pb_message_factory.GetMessageClass(descriptor)

        _pb_message_factory.MessageFactory.GetPrototype = _compat_get_prototype  # type: ignore[attr-defined]
except Exception:
    pass

# Keep third-party deprecation noise out of runtime logs.
warnings.filterwarnings(
    "ignore",
    message=r"`rcond` parameter will change to the default of machine precision.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated since version 0.26.*",
    category=FutureWarning,
)

def _check_transformers_version():
    try:
        import transformers
        from packaging.version import Version
        v = Version(transformers.__version__)
        if v >= Version("5.0.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} requires PyTorch>=2.4, "
                f"but torch {torch.__version__} is installed. "
                f"Pin transformers==4.57.6: pip install transformers==4.57.6"
            )
    except ImportError:
        pass

_check_transformers_version()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline_current_runner")
logging.getLogger("boxmot.appearance.backends.base_backend").setLevel(logging.CRITICAL)
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.disable("boxmot.appearance.backends.base_backend")
except Exception:
    pass

_default_video = "/Users/arnavchokshi/Desktop/BigTest/BigTest.mov"
if not Path(_default_video).exists():
    _default_video = "/Users/arnavchokshi/Desktop/newTest.mov"
VIDEO_PATH = Path(os.environ.get("SWAY_VIDEO_PATH", _default_video))
_repo_root = Path(__file__).resolve().parent.parent
_default_output_dir = _repo_root / "output" / VIDEO_PATH.stem
_out_override = os.environ.get("SWAY_OUTPUT_DIR", "").strip()
if _out_override:
    _out_path = Path(_out_override).expanduser()
    OUTPUT_DIR = _out_path if _out_path.is_absolute() else (_repo_root / _out_path)
else:
    _suffix = os.environ.get("SWAY_OUTPUT_SUFFIX", "").strip()
    OUTPUT_DIR = _default_output_dir if not _suffix else (_repo_root / "output" / f"{VIDEO_PATH.stem}_{_suffix}")

TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128),
]


def _safe_cmd_output(args: List[str], cwd: Optional[Path] = None) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    return (proc.stdout or "").strip() if proc.returncode == 0 else ""


def _collect_manifest_env() -> Dict[str, str]:
    keys = [
        "SWAY_VIDEO_PATH",
        "SWAY_DETECTOR_PRECISION",
        "SWAY_ENABLE_DEIMV2",
        "SWAY_TRACKER_BACKEND",
        "SWAY_TRACKER_AB",
        "SWAY_TRACKER_AB_BACKENDS",
        "SWAY_TRACKER_AB_MIN_OVERLAP_FRAMES",
        "SWAY_MASK_FRAME_STRIDE",
        "SWAY_MASK_REUSE_IOU",
        "SWAY_SHORT_TRACK_MIN_FRAMES",
        "SWAY_SHORT_TRACK_MIN_CONF",
        "SWAY_FACE_EMBED_STRIDE",
        "SWAY_PART_CACHE_IOU_THRESH",
        "SWAY_PART_CACHE_MAX_STALE",
        "SWAY_PART_CACHE_ENABLED",
        "SWAY_REID_PART_MODEL",
        "SWAY_STOP_AFTER_PHASE3",
        "SWAY_STOP_AFTER_PHASE4",
        "SWAY_SHARED_MODELS_DIR",
        "SWAY_BOXMOT_REID_WEIGHTS",
        "SWAY_TORCH_DEVICE",
        "SWAY_ENROLLMENT_MIN_COMPLETION_RATIO",
        "SWAY_ENROLLMENT_QFLOOR_RETRY_STEPS",
        "SWAY_ENROLLMENT_QFLOOR_STEP",
        "SWAY_ENROLLMENT_QFLOOR_MIN",
        "SWAY_PHASE8_NEIGHBOR_WINDOW",
        "SWAY_PHASE8_NEIGHBOR_MIN_RATIO",
        "SWAY_PHASE8_XTRACK_TARGET_APPLY_RATIO",
        "SWAY_RECORDING_MODE_POLICY",
        "SWAY_RECORDING_MODE",
        "SWAY_FORMATION_JSON",
        "SWAY_MAX_FRAMES",
        "SWAY_FORMATION_START_INDEX",
        "SWAY_FORMATION_START_OFFSET_SEC",
        "SWAY_FORMATION_AUDIO_PATH",
        "SWAY_PHASE35_ENABLED",
        "SWAY_PHASE35_MODE",
        "SWAY_PHASE35_FORCE_FAIL_OPEN",
        "SWAY_PHASE35_TIMEOUT_MS",
        "SWAY_PHASE35_IOU_THRESH",
        "SWAY_PHASE35_LOW_CONF_THRESH",
        "SWAY_PHASE35_MIN_MARGIN",
        "SWAY_PHASE35_TEMPORAL_WEIGHT",
        "SWAY_PHASE35_HAIR_HAND_GUARD",
        "SWAY_PHASE35_HAIR_HAND_CONF_THRESH",
        "SWAY_PHASE35_HAIR_HAND_RT_CONF_THRESH",
        "SWAY_PHASE35_HAIR_HAND_DISAGREE_PX",
        "SWAY_PHASE35_HAIR_HAND_ARM_RATIO_MAX",
        "SWAY_PHASE35_HAIR_HAND_TEMPORAL_JUMP_PX",
        "SWAY_PHASE35_HAIR_HAND_LOCK_FRAMES",
        "SWAY_PHASE35_HAIR_HAND_REPLACE_ELBOW",
        "SWAY_VALIDATION_CLIP_SET",
        "SWAY_VALIDATION_CLIP_ID",
        "SWAY_VALIDATION_CLIP_TIER",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            out[k] = str(v)
    return out


def _build_run_manifest(
    *,
    repo_root: Path,
    device: str,
    video_path: Path,
    output_dir: Path,
    tracker_backend_requested: str,
    tracker_backend_effective: str,
    precision_requested: str,
    precision_effective: str,
    phase_times_ms: Dict[str, float],
    tracker_ab_report: Dict[str, Any],
    summary: Dict[str, Any],
    reid_feature_mode: str = "unknown",
    reid_feature_mode_reason: str = "",
    gate_status: Optional[Dict[str, Any]] = None,
    run_valid_for_ranking: Optional[bool] = None,
) -> Dict[str, Any]:
    gpu_name = ""
    cuda_version = ""
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = str(torch.cuda.get_device_name(0))
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
        except Exception:
            pass

    return {
        "schema_version": "v23_run_manifest_v1",
        "timestamp_epoch_s": int(time.time()),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "git": {
            "sha": _safe_cmd_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "branch": _safe_cmd_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
            "is_dirty": bool(_safe_cmd_output(["git", "status", "--porcelain"], cwd=repo_root)),
        },
        "runtime": {
            "torch_device": device,
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
        },
        "run": {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "phase_times_ms": phase_times_ms,
        },
        "pipeline_diagnostics": {
            "detector_precision_requested": precision_requested,
            "detector_precision_effective": precision_effective,
            "tracker_backend_requested": tracker_backend_requested,
            "tracker_backend_effective": tracker_backend_effective,
            "tracker_ab": tracker_ab_report,
            "reid_feature_mode": reid_feature_mode,
            "reid_feature_mode_reason": reid_feature_mode_reason,
        },
        "env": _collect_manifest_env(),
        "gate_status": gate_status or {},
        "run_valid_for_ranking": bool(run_valid_for_ranking) if run_valid_for_ranking is not None else None,
        "summary_digest": {
            "total_frames": int(summary.get("total_frames", 0) or 0),
            "dancers_enrolled": int(summary.get("dancers_enrolled", 0) or 0),
            "dancers_lifted": int(summary.get("dancers_lifted", 0) or 0),
            "total_ms": float(summary.get("total_ms", 0.0) or 0.0),
        },
    }


_ARTIFACT_PURPOSES: Dict[str, Dict[str, str]] = {
    "phase1_detection.mp4": {"phase": "1", "purpose": "Phase-1 detection visualization."},
    "phase2_masks.mp4": {"phase": "2", "purpose": "Phase-2 SAM masking visualization."},
    "phase3_vitpose.mp4": {"phase": "3", "purpose": "Phase-3 ViTPose overlay visualization."},
    "phase3_rtmwx.mp4": {"phase": "3", "purpose": "Phase-3 RTMW-X overlay visualization."},
    "phase4_tracking_forward.mp4": {"phase": "4", "purpose": "Phase-4 forward tracking visualization."},
    "phase4_tracking_bidirectional.mp4": {"phase": "4", "purpose": "Phase-4 bidirectional tracking visualization."},
    "phase6_reid_fusion.mp4": {"phase": "6", "purpose": "Phase-6 Re-ID fusion assignments visualization."},
    "phase6_reid.mp4": {"phase": "6", "purpose": "Backward-compatible alias of phase6_reid_fusion.mp4."},
    "phase7_darkzone_resolution.mp4": {"phase": "7", "purpose": "Phase-7 dark-zone resolution visualization."},
    "phase7_darkzone.mp4": {"phase": "7", "purpose": "Backward-compatible alias of phase7_darkzone_resolution.mp4."},
    "phase7_legacy_pose3d_compat.mp4": {"phase": "legacy", "purpose": "Legacy pose+3D compatibility visualization."},
    "legacy_pose3d.mp4": {"phase": "legacy", "purpose": "Backward-compatible alias of phase7_legacy_pose3d_compat.mp4."},
    "phase8_final_optimized.mp4": {"phase": "8", "purpose": "Phase-8 final optimized render."},
    "phase8_final.mp4": {"phase": "8", "purpose": "Backward-compatible alias of phase8_final_optimized.mp4."},
    "summary.json": {"phase": "all", "purpose": "High-level pipeline summary and diagnostics."},
    "run_manifest.json": {"phase": "all", "purpose": "Runtime environment, git, and reproducibility manifest."},
    "master_pipeline_report.json": {"phase": "all", "purpose": "Master report that links all major outputs."},
    "artifacts_index.json": {"phase": "all", "purpose": "Index of every output artifact with explicit purpose."},
    "detections_phase1.json": {"phase": "1", "purpose": "Per-frame detector outputs and confidences."},
    "masks_phase2.json": {"phase": "2", "purpose": "Per-frame SAM mask quality diagnostics."},
    "phase3_vitpose_overlay.json": {"phase": "3", "purpose": "ViTPose keypoints for re-rendering phase-3 video."},
    "phase3_5_disambiguation.json": {"phase": "3.5", "purpose": "Phase-3.5 per-frame ambiguity resolution decisions."},
    "phase3_5_metrics.json": {"phase": "3.5", "purpose": "Phase-3.5 aggregate metrics and overlap-bin diagnostics."},
    "baseline_validation_report.json": {"phase": "all", "purpose": "Locked baseline metrics + validation clip metadata for comparisons."},
    "tracklets_forward.json": {"phase": "4", "purpose": "Forward pass tracker boxes and IDs."},
    "tracklets_backward.json": {"phase": "4", "purpose": "Backward pass tracker boxes and IDs."},
    "data.json": {"phase": "4", "purpose": "Tracking export for MOT-style tooling / sweeps."},
    "gallery_identity_bank.json": {"phase": "5", "purpose": "Enrollment identity bank (json metadata)."},
    "gallery_identity_bank.npz": {"phase": "5", "purpose": "Enrollment identity bank (numpy payloads)."},
    "phase6_identity_assignments_reid.json": {"phase": "6", "purpose": "Frame-by-frame Re-ID assignments and confidences."},
    "identity_assignments_phase5.json": {"phase": "6", "purpose": "Backward-compatible alias of phase6 Re-ID assignments."},
    "pose2d_phase6.json": {"phase": "legacy", "purpose": "Per-frame 2D pose keypoints with model confidence."},
    "final_identity_tracks.json": {"phase": "8", "purpose": "Final corrected identity-track assignments per frame."},
    "evaluation_metrics.json": {"phase": "8", "purpose": "Computed quality metrics for run evaluation."},
    "coaching_motion_analysis.json": {"phase": "8", "purpose": "Per-dancer jerk and pairwise synchrony coaching metrics."},
    "switch_event_log.json": {"phase": "8", "purpose": "Identity switch event log and reasons."},
    "enrollment_reject_log.json": {"phase": "5", "purpose": "Enrollment rejection diagnostics."},
    "phase8_reject_log.json": {"phase": "8", "purpose": "Phase-8 correction rejection diagnostics."},
    "tracker_ab_overlap.json": {"phase": "4", "purpose": "Optional tracker A/B overlap benchmark results."},
    "phase4_identity_assignments.json": {"phase": "4", "purpose": "Phase-4-only stop output: provisional identities."},
    "formation_diagnostics.json": {"phase": "4", "purpose": "Formation alignment diagnostics (phase-4 stop mode)."},
    "summary_phase3_only.json": {"phase": "3", "purpose": "Summary for stop-after-phase3 runs."},
    "summary_phase4_only.json": {"phase": "4", "purpose": "Summary for stop-after-phase4 runs."},
}


def _csv_env_list(name: str) -> List[str]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _write_baseline_validation_report(
    *,
    output_dir: Path,
    video_path: Path,
    summary_obj: Dict[str, Any],
    eval_metrics_obj: Optional[Dict[str, Any]],
    phase35_status: Optional[Dict[str, Any]],
) -> None:
    report = {
        "schema_version": "baseline_validation_report_v1",
        "video_path": str(video_path),
        "validation_clip": {
            "clip_id": (os.environ.get("SWAY_VALIDATION_CLIP_ID", "") or "").strip(),
            "tier": (os.environ.get("SWAY_VALIDATION_CLIP_TIER", "") or "").strip(),
            "clip_set": _csv_env_list("SWAY_VALIDATION_CLIP_SET"),
        },
        "baseline_locked_metrics": {
            "phase_times_ms": dict(summary_obj.get("phase_times_ms", {})),
            "total_frames": int(summary_obj.get("total_frames", 0) or 0),
            "dancers_enrolled": int(summary_obj.get("dancers_enrolled", 0) or 0),
            "dancers_lifted": int(summary_obj.get("dancers_lifted", 0) or 0),
            "run_valid_for_ranking": summary_obj.get("run_valid_for_ranking"),
        },
        "evaluation_metrics": eval_metrics_obj if isinstance(eval_metrics_obj, dict) else {},
        "phase35_status": phase35_status if isinstance(phase35_status, dict) else {},
    }
    with open(output_dir / "baseline_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)


def _write_alias_copy(src: Path, alias_name: str) -> None:
    """Keep backward-compatible artifact names while using clearer canonical names."""
    if not src.exists():
        return
    dst = src.parent / alias_name
    try:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
    except Exception as exc:
        logger.warning("Alias copy failed (%s -> %s): %s", src.name, alias_name, exc)


def _artifact_phase_from_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("phase"):
        p = stem.split("_", 1)[0]
        return p.replace("phase", "") if p[5:].isdigit() else p
    if "legacy" in stem:
        return "legacy"
    if name in _ARTIFACT_PURPOSES:
        return _ARTIFACT_PURPOSES[name]["phase"]
    return "aux"


def _write_artifact_indexes(
    *,
    output_dir: Path,
    summary: Dict[str, Any],
    run_manifest: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    artifacts: List[Dict[str, Any]] = []
    for p in sorted(output_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        meta = _ARTIFACT_PURPOSES.get(p.name, {})
        artifacts.append(
            {
                "name": p.name,
                "phase": meta.get("phase", _artifact_phase_from_name(p.name)),
                "purpose": meta.get("purpose", "Pipeline artifact (purpose not yet annotated)."),
                "size_bytes": int(p.stat().st_size),
                "kind": p.suffix.lower().lstrip(".") or "file",
            }
        )

    index_payload = {
        "schema_version": "v23_artifacts_index_v1",
        "output_dir": str(output_dir),
        "artifacts": artifacts,
    }
    with open(output_dir / "artifacts_index.json", "w") as f:
        json.dump(index_payload, f, indent=2)
    logger.info("Wrote artifacts_index.json")

    master_payload: Dict[str, Any] = {
        "schema_version": "v23_master_pipeline_report_v1",
        "video": summary.get("video"),
        "output_dir": str(output_dir),
        "summary": summary,
        "run_manifest": run_manifest,
        "artifacts_index_path": "artifacts_index.json",
    }
    if extra:
        master_payload["extra"] = extra
    with open(output_dir / "master_pipeline_report.json", "w") as f:
        json.dump(master_payload, f, indent=2)
    logger.info("Wrote master_pipeline_report.json")


def _color_for_id(tid: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def _draw_text(frame, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _identity_label(
    dancer_id: int,
    track_id: int,
    *,
    use_names: bool,
    identity_name_by_did: Optional[Dict[int, str]] = None,
) -> str:
    if dancer_id <= 0:
        return f"T{track_id}"
    if use_names and identity_name_by_did:
        nm = str(identity_name_by_did.get(int(dancer_id), "")).strip()
        if nm:
            return nm
    return f"D{dancer_id}"


def _match_dets_to_tracks(dets, tracking_frame) -> Dict[int, int]:
    """Match detections to track IDs by bbox IoU. Returns {det_index: track_id}."""
    result = {}
    if not dets or not tracking_frame.objects:
        return result

    track_items = [(tid, obj) for tid, obj in tracking_frame.objects.items() if obj.bbox is not None]
    used_tracks = set()

    for di, det in enumerate(dets):
        best_tid = -1
        best_iou = 0.0
        db = det.bbox
        for tid, obj in track_items:
            if tid in used_tracks:
                continue
            tb = obj.bbox
            ix1 = max(db[0], tb[0]); iy1 = max(db[1], tb[1])
            ix2 = min(db[2], tb[2]); iy2 = min(db[3], tb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_d = max(0, db[2] - db[0]) * max(0, db[3] - db[1])
            area_t = max(0, tb[2] - tb[0]) * max(0, tb[3] - tb[1])
            union = area_d + area_t - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        if best_tid >= 0 and best_iou > 0.1:
            result[di] = best_tid
            used_tracks.add(best_tid)
    return result


def _draw_phase_banner(frame, text, color=(50, 50, 50)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), color, -1)
    _draw_text(frame, text, (10, 22), (255, 255, 255), 0.6, 1)


def _compute_completion_ratio(selected_count: int, target_count: int, min_ratio: float) -> Tuple[float, bool]:
    ratio = float(selected_count) / float(max(1, target_count))
    return ratio, bool(ratio >= min_ratio)


def _xtrack_tiebreak_should_swap(candidate_score: float, incumbent_score: float, margin: float = 0.05) -> bool:
    return bool(candidate_score > (incumbent_score + margin))


def _phase8_required_vote_ratio(vote_count: int, base_ratio: float) -> float:
    """Allow smaller windows to pass with proportional consensus.
    
    With wider neighbor_window=4, we get up to 8 votes. Relax thresholds
    for small vote counts so corrections can actually be applied.
    """
    ratio = float(base_ratio)
    if vote_count <= 0:
        return ratio
    if vote_count <= 2:
        return min(ratio, 0.50)
    if vote_count <= 4:
        return min(ratio, 0.40)
    return ratio


def _resolve_identity_for_render(
    frame_idx: int,
    track_id: int,
    final_identity_tracks_by_frame: Dict[int, Dict[int, int]],
    reid_assignments_phase4: Dict[int, Dict[int, int]],
    last_valid_did_by_track: Dict[int, int],
    frame_render_assignments: Optional[Dict[int, int]] = None,
) -> Tuple[int, str]:
    frame_render_assignments = frame_render_assignments if frame_render_assignments is not None else {}

    did = int(final_identity_tracks_by_frame.get(frame_idx, {}).get(track_id, -1))
    if did > 0:
        last_valid_did_by_track[track_id] = did
        frame_render_assignments[did] = track_id
        return did, "final_identity_tracks"

    did = int(reid_assignments_phase4.get(frame_idx, {}).get(track_id, -1))
    if did > 0:
        last_valid_did_by_track[track_id] = did
        frame_render_assignments[did] = track_id
        return did, "phase4_assignments"

    did = int(last_valid_did_by_track.get(track_id, -1))
    if did > 0:
        existing_tid = frame_render_assignments.get(did, -1)
        if existing_tid >= 0 and existing_tid != track_id:
            return -1, "unknown_sticky_conflict"
        frame_render_assignments[did] = track_id
        return did, "sticky_last_valid"

    return -1, "unknown"


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe_detections(dets: List["Detection"], iou_thresh: float = 0.8) -> List["Detection"]:
    if not dets:
        return []
    kept: List["Detection"] = []
    for d in sorted(dets, key=lambda x: float(x.confidence), reverse=True):
        if all(_bbox_iou(d.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def _has_overlap(dets: List["Detection"], iou_thresh: float = 0.30) -> bool:
    n = len(dets)
    if n < 2:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_iou(dets[i].bbox, dets[j].bbox) >= iou_thresh:
                return True
    return False


def _get_overlapping_det_indices(dets: List["Detection"], iou_thresh: float = 0.15) -> set:
    """Return set of detection indices that overlap with at least one other detection."""
    overlapping: set = set()
    n = len(dets)
    if n < 2:
        return overlapping
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_iou(dets[i].bbox, dets[j].bbox) >= iou_thresh:
                overlapping.add(i)
                overlapping.add(j)
    return overlapping


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a boolean mask by (dx, dy) pixels using np.roll with zero-fill."""
    shifted = mask.copy()
    if dy != 0:
        shifted = np.roll(shifted, dy, axis=0)
        if dy > 0:
            shifted[:dy, :] = False
        else:
            shifted[dy:, :] = False
    if dx != 0:
        shifted = np.roll(shifted, dx, axis=1)
        if dx > 0:
            shifted[:, :dx] = False
        else:
            shifted[:, dx:] = False
    return shifted


def _bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) center of an [x1, y1, x2, y2] bbox."""
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _avg_best_iou(src: List["Detection"], ref: List["Detection"]) -> float:
    if not src or not ref:
        return 0.0
    vals = []
    for s in src:
        vals.append(max(_bbox_iou(s.bbox, r.bbox) for r in ref))
    return float(np.mean(vals)) if vals else 0.0


def _average_hist_dicts(items: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not items:
        return {}
    keys = sorted(set().union(*[set(x.keys()) for x in items if x]))
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = [np.asarray(d[k], dtype=np.float32) for d in items if d and k in d]
        if not vals:
            continue
        arr = np.stack(vals, axis=0)
        m = np.mean(arr, axis=0).astype(np.float32)
        s = float(np.sum(m))
        if s > 1e-8:
            m /= s
        out[k] = m
    return out


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

class PhaseVideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        self.frame_count = 0
        if not self.writer.isOpened():
            logger.warning("VideoWriter failed to open: %s", path)

    def write(self, frame: np.ndarray):
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        self.writer.release()
        logger.info("Wrote %d frames to %s", self.frame_count, self.path.name)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    if not VIDEO_PATH.exists():
        logger.error("BigTest video not found at %s", VIDEO_PATH)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "pipeline_current_log.txt"
    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Current SWAY Pipeline Runner — BigTest")
    logger.info("Video: %s", VIDEO_PATH)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info("Video metadata: %dx%d @ %.1f fps, %d metadata frames", width, height, fps, total_frames)

    max_frames = None
    max_frames_env = (os.environ.get("SWAY_MAX_FRAMES", "") or "").strip()
    if max_frames_env:
        try:
            max_frames = max(1, int(max_frames_env))
            logger.info("SWAY_MAX_FRAMES requested: %d", max_frames)
        except ValueError:
            logger.warning("Invalid SWAY_MAX_FRAMES value=%r; ignoring", max_frames_env)

    # Some codecs/container combinations report unreliable CAP_PROP_FRAME_COUNT.
    # For full-length runs, prefer a real sequential readable-frame count.
    if max_frames is None:
        cap_count = cv2.VideoCapture(str(VIDEO_PATH))
        readable_frames = 0
        while True:
            ok_count, fr_count = cap_count.read()
            if not ok_count or fr_count is None:
                break
            readable_frames += 1
        cap_count.release()
        if readable_frames > 0 and readable_frames != total_frames:
            logger.info(
                "Adjusted frame count using readable stream scan: metadata=%d readable=%d",
                total_frames,
                readable_frames,
            )
            total_frames = readable_frames
    effective_total_frames = min(total_frames, max_frames) if max_frames is not None else total_frames
    logger.info(
        "Deferring full frame load: enrollment uses lazy frame reader first (effective frames=%d)",
        effective_total_frames,
    )

    lazy_cache: Dict[int, np.ndarray] = {}

    def _lazy_get_frame(frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= effective_total_frames:
            return None
        if frame_idx in lazy_cache:
            return lazy_cache[frame_idx]
        cap_local = cv2.VideoCapture(str(VIDEO_PATH))
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_local, frame_local = cap_local.read()
        cap_local.release()
        if not ret_local or frame_local is None:
            return None
        lazy_cache[frame_idx] = frame_local
        return frame_local

    def _materialize_all_frames() -> List[np.ndarray]:
        logger.info("Loading all frames into memory (post-enrollment)...")
        t_load = time.monotonic()
        cap_local = cv2.VideoCapture(str(VIDEO_PATH))
        out_frames: List[np.ndarray] = []
        target_n = effective_total_frames
        for fidx_load in range(target_n):
            ret_local, frame_local = cap_local.read()
            if not ret_local or frame_local is None:
                break
            # Keep decode sequential: random seek per frame can stall on some codecs.
            frame_out = lazy_cache.get(fidx_load)
            if frame_out is None:
                frame_out = frame_local
                lazy_cache[fidx_load] = frame_local
            out_frames.append(frame_out)
        cap_local.release()
        logger.info("Loaded %d frames in %.1fs", len(out_frames), time.monotonic() - t_load)
        if len(out_frames) != total_frames:
            logger.info(
                "Metadata frame count differs from decoded stream (%d vs %d); using decoded frames.",
                total_frames,
                len(out_frames),
            )
        logger.info("Video: %dx%d @ %.1f fps, %d frames (%.1fs)",
                    width, height, fps, len(out_frames), len(out_frames) / max(1e-6, fps))
        return out_frames

    # --- Import pipeline modules ---
    from sway.detr_detector import Detection
    from sway.enrollment import enroll_dancers, DancerGallery
    from sway.reid_fusion import ReIDFusionEngine
    from sway.reid_mlp_router import ReIDMLPRouter
    from sway.collision_solver import CoalescenceDetector
    from sway.noougat_graph_stitch import NOOUGATGraphStitcher, TrackletNode
    from sway.mamba_ssm_lifter import lift_poses_v23
    from sway.bpbreid_extractor import BPBreIDExtractor
    from sway.color_histogram_reid import ColorHistogramExtractor
    from sway.pose_estimator import PoseEstimator
    from sway.rtmw_384_hybrid_estimator import RTMW384HybridEstimator
    from sway.formation_identity import (
        StartAlignment,
        build_formation_assignments,
        detect_recording_mode,
        estimate_audio_offset,
        estimate_start_offset_spatial,
        expected_positions_at,
        fuse_start_alignment,
        load_formation_timeline,
    )
    from sway.phase3_vitpose_viz import (
        PHASE3_SKEL_BONES,
        apply_frame_confidence_grade,
        draw_vitpose_confidence_overlay,
        frame_mean_keypoint_conf,
        phase3_adaptive_thresh,
    )
    from sway.phase35_ambiguity_resolver import Phase35Config, resolve_phase35

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import torch
    from PIL import Image

    stop_after_phase3 = str(os.environ.get("SWAY_STOP_AFTER_PHASE3", "0")).strip().lower() in {"1", "true", "yes", "on"}
    phase35_enabled = str(os.environ.get("SWAY_PHASE35_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    phase35_mode = (os.environ.get("SWAY_PHASE35_MODE", "off") or "off").strip().lower()
    if phase35_mode not in {"off", "shadow", "active"}:
        logger.warning("Unknown SWAY_PHASE35_MODE=%s; defaulting to off", phase35_mode)
        phase35_mode = "off"
    phase35_force_fail_open = str(os.environ.get("SWAY_PHASE35_FORCE_FAIL_OPEN", "1")).strip().lower() in {"1", "true", "yes", "on"}
    phase35_timeout_ms = float(os.environ.get("SWAY_PHASE35_TIMEOUT_MS", "0") or 0.0)
    phase35_cfg = Phase35Config(
        iou_ambiguous_thresh=float(os.environ.get("SWAY_PHASE35_IOU_THRESH", "0.35") or 0.35),
        low_conf_thresh=float(os.environ.get("SWAY_PHASE35_LOW_CONF_THRESH", "0.45") or 0.45),
        min_margin=float(os.environ.get("SWAY_PHASE35_MIN_MARGIN", "0.03") or 0.03),
        temporal_weight=float(os.environ.get("SWAY_PHASE35_TEMPORAL_WEIGHT", "0.15") or 0.15),
        hair_hand_guard_enabled=str(os.environ.get("SWAY_PHASE35_HAIR_HAND_GUARD", "0")).strip().lower() in {"1", "true", "yes", "on"},
        hair_hand_conf_thresh=float(os.environ.get("SWAY_PHASE35_HAIR_HAND_CONF_THRESH", "0.75") or 0.75),
        hair_hand_rt_conf_thresh=float(os.environ.get("SWAY_PHASE35_HAIR_HAND_RT_CONF_THRESH", "0.35") or 0.35),
        hair_hand_disagree_px=float(os.environ.get("SWAY_PHASE35_HAIR_HAND_DISAGREE_PX", "28.0") or 28.0),
        hair_hand_arm_ratio_max=float(os.environ.get("SWAY_PHASE35_HAIR_HAND_ARM_RATIO_MAX", "1.9") or 1.9),
        hair_hand_temporal_jump_px=float(os.environ.get("SWAY_PHASE35_HAIR_HAND_TEMPORAL_JUMP_PX", "30.0") or 30.0),
        hair_hand_lock_frames=int(os.environ.get("SWAY_PHASE35_HAIR_HAND_LOCK_FRAMES", "4") or 4),
        hair_hand_replace_elbow=str(os.environ.get("SWAY_PHASE35_HAIR_HAND_REPLACE_ELBOW", "1")).strip().lower() in {"1", "true", "yes", "on"},
    )
    phase35_status: Dict[str, Any] = {
        "enabled": bool(phase35_enabled),
        "mode": phase35_mode,
        "applied": False,
        "ran": False,
        "fail_open": False,
        "fallback_reason": "",
        "fallback_count": 0,
        "override_ratio": 0.0,
        "diagnostics_path": "",
        "metrics_path": "",
        "elapsed_ms": 0.0,
    }
    fast_ab_mode = stop_after_phase3 and (
        str(os.environ.get("SWAY_FAST_AB_MODE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )
    recording_mode_policy = (os.environ.get("SWAY_RECORDING_MODE_POLICY", "auto_then_confirm") or "auto_then_confirm").strip()
    recording_mode_override = (os.environ.get("SWAY_RECORDING_MODE", "") or "").strip()
    default_formation_json = _repo_root / "data" / "bigTest_formations.json"
    formation_json_path = Path(os.environ.get("SWAY_FORMATION_JSON", str(default_formation_json))).expanduser()
    formation_timeline = load_formation_timeline(str(formation_json_path))
    identity_name_by_did: Dict[int, str] = {}
    use_name_labels = False
    mode_info: Dict[str, Any] = {
        "policy": recording_mode_policy,
        "override": recording_mode_override,
        "formation_json": str(formation_json_path),
        "formation_available": bool(formation_timeline is not None),
        "detected_mode": "windowed",
        "final_mode": "windowed",
        "mode_confidence": 0.0,
        "mode_reason": "not_evaluated",
        "confirmation_required": False,
    }
    formation_alignment_info: Dict[str, Any] = {}

    models_dir = Path(os.environ.get("SWAY_SHARED_MODELS_DIR", "models")).expanduser()
    if not models_dir.is_absolute():
        models_dir = (_repo_root / models_dir).resolve()
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    logger.info("Compute device: %s", DEVICE)
    if DEVICE == "cuda":
        try:
            logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
            logger.info("Torch CUDA build: %s", getattr(torch.version, "cuda", "unknown"))
        except Exception:
            pass

    reid_part_model = (os.environ.get("SWAY_REID_PART_MODEL", "bpbreid") or "bpbreid").strip().lower()
    reid_feature_mode = "unknown"
    reid_feature_mode_reason = ""

    def _assert_torchreid_preflight() -> None:
        nonlocal reid_feature_mode, reid_feature_mode_reason
        if reid_part_model != "bpbreid":
            return
        try:
            from torchreid.utils import FeatureExtractor  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Preflight gate HARD FAIL: torchreid is not importable. "
                "Install via: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from exc
        import torch
        ckpt_path = models_dir / "bpbreid_r50_market_msmt17.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Preflight gate HARD FAIL: BPBreID checkpoint not found at {ckpt_path}"
            )
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", encoding="latin1")
            sd = ckpt["state_dict"]
            emb_dim = sd["classifier.weight"].shape[1]
            assert emb_dim == 2048, f"BPBreID emb dim={emb_dim}, expected 2048"
        except Exception as exc:
            raise RuntimeError(
                f"Preflight gate HARD FAIL: BPBreID checkpoint validation failed: {exc}"
            ) from exc
        reid_feature_mode = "torchreid"
        reid_feature_mode_reason = "startup preflight: torchreid + BPBreID checkpoint validated (2048-d)"
        logger.info("Preflight gate PASSED: torchreid + BPBreID checkpoint (2048-d)")

    _assert_torchreid_preflight()

    # --- 1. YOLO detector ---
    logger.info("Loading YOLO detector...")
    t0 = time.monotonic()
    from ultralytics import YOLO
    # Current pipeline policy: detector is fixed to yolo26l_dancetrack.pt.
    default_yolo = str(models_dir / "yolo26l_dancetrack.pt")
    if not Path(default_yolo).exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {default_yolo}. No fallback allowed."
        )
    yolo_weights = default_yolo
    yolo_model = YOLO(yolo_weights)
    logger.info("YOLO loaded: %s (%.1fs)", yolo_weights, time.monotonic() - t0)

    class YOLODetectorWrapper:
        def __init__(self, model):
            self.model = model
        def detect(self, frame, frame_idx=0):
            results = self.model(frame, verbose=False, classes=[0], conf=0.3)
            dets = []
            for r in results:
                for box in r.boxes:
                    bbox = box.xyxy[0].cpu().numpy().astype(np.float32)
                    conf = float(box.conf[0])
                    dets.append(Detection(bbox=bbox, confidence=conf, class_id=0))
            return dets

    yolo_detector = YOLODetectorWrapper(yolo_model)

    # --- 1b. Detector policy ---
    # Current pipeline policy: YOLO-only detector path (no hybrid DETR arbitration).
    precision_requested = "yolo26l_dancetrack"
    precision_effective_name = "yolo26l_dancetrack"
    enable_deimv2 = False
    detr_detector = None
    detector = yolo_detector
    logger.info("Detector policy: YOLO-only (%s)", yolo_weights)

    # --- 2. BoxMOT tracker ---
    logger.info("Loading BoxMOT tracker backend...")
    import boxmot as _boxmot
    try:
        from boxmot import DeepOcSort
    except ImportError:
        from boxmot import DeepOCSORT as DeepOcSort
    osnet_weights = models_dir / "osnet_x0_25_msmt17.pt"
    # Keep tracker parameters tunable from env; defaults preserve prior behavior.
    _doc_det_thresh = float(os.environ.get("SWAY_TRACKER_DET_THRESH", "0.30") or 0.30)
    _doc_max_age = int(os.environ.get("SWAY_TRACKER_MAX_AGE", "30") or 30)
    _doc_iou = float(os.environ.get("SWAY_TRACKER_IOU_THRESHOLD", "0.30") or 0.30)
    _doc_delta_t = int(os.environ.get("SWAY_TRACKER_DELTA_T", "3") or 3)
    _doc_inertia = float(os.environ.get("SWAY_TRACKER_INERTIA", "0.20") or 0.20)
    _doc_w_assoc_emb = float(os.environ.get("SWAY_TRACKER_W_ASSOC_EMB", "0.50") or 0.50)

    # BoxMOT expects CUDA device ordinals ("0"), not the generic "cuda" string.
    _boxmot_device = "0" if str(DEVICE).lower().startswith("cuda") else DEVICE

    def _make_deepocsort_tracker():
        """Initialize DeepOcSort across BoxMOT API variants.

        BoxMOT constructor signatures changed across versions (e.g. v10 requires
        reid_weights + half while earlier versions use model_weights + fp16).
        We adapt kwargs dynamically to avoid hard failures in benchmark runs.
        """
        base_values: Dict[str, Any] = {
            "model_weights": osnet_weights,
            "reid_weights": osnet_weights,
            "device": _boxmot_device,
            "fp16": False,
            "half": False,
            "det_thresh": _doc_det_thresh,
            "max_age": _doc_max_age,
            "iou_threshold": _doc_iou,
            "iou_thresh": _doc_iou,
            "delta_t": _doc_delta_t,
            "inertia": _doc_inertia,
            "w_association_emb": _doc_w_assoc_emb,
        }
        sig = inspect.signature(DeepOcSort.__init__)
        accepted = {p.name for p in sig.parameters.values() if p.name != "self"}
        kwargs: Dict[str, Any] = {}
        for key, value in base_values.items():
            if key in accepted:
                kwargs[key] = value
        return DeepOcSort(**kwargs)

    def _build_tracker_kwargs(cls) -> Dict[str, Any]:
        """Filter kwargs to the backend constructor signature."""
        base_values: Dict[str, Any] = {
            "model_weights": osnet_weights,
            "device": _boxmot_device,
            "fp16": False,
            "det_thresh": _doc_det_thresh,
            "iou_threshold": _doc_iou,
            "iou_thresh": _doc_iou,
            "max_age": _doc_max_age,
            "max_obs": _doc_max_age,
            "delta_t": _doc_delta_t,
            "inertia": _doc_inertia,
            "w_association_emb": _doc_w_assoc_emb,
        }
        sig = inspect.signature(cls.__init__)
        accepted = {p.name for p in sig.parameters.values() if p.name != "self"}
        kwargs: Dict[str, Any] = {}
        for key, value in base_values.items():
            if key in accepted:
                kwargs[key] = value
        if "reid_weights" in accepted:
            kwargs.setdefault("reid_weights", osnet_weights)
        if "device" in accepted:
            kwargs.setdefault("device", _boxmot_device)
        if "half" in accepted:
            kwargs.setdefault("half", False)
        return kwargs

    def _make_tracker_for_backend(backend_name: str, *, allow_fallback: bool = True):
        name = (backend_name or "deepocsort").strip().lower()
        if name in {"deepocsort", "deep_ocsort", "ocsort_deep"}:
            return _make_deepocsort_tracker(), "deepocsort", "ok", None

        class_candidates = {
            "ocsort": "OCSORT",
            "botsort": "BoTSORT",
            "strongsort": "StrongSORT",
        }
        class_name = class_candidates.get(name)
        cls = getattr(_boxmot, class_name, None) if class_name else None
        if cls is None:
            msg = f"Tracker backend '{name}' unavailable in this boxmot build"
            if allow_fallback:
                logger.warning("%s; using deepocsort", msg)
                return _make_deepocsort_tracker(), "deepocsort_fallback", "fallback", msg
            logger.warning("%s; no fallback permitted for this run", msg)
            return None, name, "unavailable", msg

        candidate_kwargs = [_build_tracker_kwargs(cls), {}]
        errors: List[str] = []
        for kwargs in candidate_kwargs:
            try:
                return cls(**kwargs), name, "ok", None
            except Exception as exc:
                errors.append(str(exc))
                continue

        err_msg = f"Tracker backend '{name}' failed to initialize ({'; '.join(errors[-2:])})"
        if allow_fallback:
            logger.warning("%s; using deepocsort", err_msg)
            return _make_deepocsort_tracker(), "deepocsort_fallback", "fallback", err_msg
        logger.warning("%s; no fallback permitted for this run", err_msg)
        return None, name, "init_failed", err_msg

    tracker_backend_requested = os.environ.get("SWAY_TRACKER_BACKEND", "deepocsort")
    boxmot_tracker, tracker_backend_effective, tracker_backend_status, tracker_backend_error = _make_tracker_for_backend(
        tracker_backend_requested,
        allow_fallback=True,
    )
    logger.info(
        "BoxMOT tracker loaded: requested=%s effective=%s status=%s",
        tracker_backend_requested,
        tracker_backend_effective,
        tracker_backend_status,
    )
    if tracker_backend_error:
        logger.warning("Tracker backend init note: %s", tracker_backend_error)

    if fast_ab_mode:
        logger.info("Fast A/B mode enabled for phase3-only run: skipping heavy BPBreID/ArcFace/VitPose model loads")

        class _DummyPartResult:
            def __init__(self):
                self.global_emb = None
                self.part_embs: Dict[str, np.ndarray] = {}

        class _DummyPartExtractor:
            def extract(self, crop, keypoints=None, mask=None):
                return _DummyPartResult()

        class _DummyPoseEstimator:
            def estimate_poses(self, frame, bboxes, track_ids):
                return {}

        bpbreid_part_extractor = _DummyPartExtractor()
        pose_estimator = _DummyPoseEstimator()
        reid_feature_mode = "fast_ab_dummy"
        reid_feature_mode_reason = "SWAY_FAST_AB_MODE enabled"

        def extract_bpbreid_embedding(crop_bgr: np.ndarray) -> np.ndarray:
            return np.zeros((2048,), dtype=np.float32)

        def extract_face_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
            return None
        def extract_face_embeddings_batch(crops_bgr: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
            return [None for _ in crops_bgr]
    else:
        # --- 3. BPBreID (torchreid strict mode) ---
        logger.info("Loading BPBreID part extractor (strict, no fallback)...")
        bpbreid_part_extractor = BPBreIDExtractor(device=DEVICE)
        reid_feature_mode = bpbreid_part_extractor.reid_feature_mode
        reid_feature_mode_reason = bpbreid_part_extractor.reid_feature_mode_reason
        logger.info("ReID feature mode: %s (%s)", reid_feature_mode, reid_feature_mode_reason or "n/a")
        if reid_part_model == "bpbreid" and reid_feature_mode != "torchreid":
            raise RuntimeError(
                "ReID feature-path gate failed: expected torchreid mode for bpbreid, "
                f"got {reid_feature_mode}"
            )

        def extract_bpbreid_embedding(crop_bgr: np.ndarray) -> np.ndarray:
            """Extract 2048-d L2-normalized embedding via strict BPBreID extractor."""
            result = bpbreid_part_extractor.extract(crop_bgr)
            emb = result.global_emb
            return emb / (np.linalg.norm(emb) + 1e-8)

        # --- 4. ArcFace (insightface) ---
        logger.info("Loading ArcFace face recognition...")
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("ArcFace (buffalo_l) loaded")

        def extract_face_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
            faces = face_app.get(crop_bgr)
            if not faces:
                return None
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            if best.det_score < 0.5:
                return None
            emb = best.embedding.astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)
        def extract_face_embeddings_batch(crops_bgr: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
            out: List[Optional[np.ndarray]] = []
            for c in crops_bgr:
                try:
                    out.append(extract_face_embedding(c))
                except Exception:
                    out.append(None)
            return out

        # --- 5. VitPose (repo PoseEstimator wrapper) ---
        logger.info("Loading VitPose pose estimator...")
        pose_estimator = PoseEstimator(device=DEVICE, model_name="usyd-community/vitpose-plus-large")
        logger.info("VitPose-plus-large loaded")

    logger.info("All models loaded successfully (strict, no fallbacks)")

    # =====================================================================
    # Enrollment is intentionally removed from startup. It only occurs at
    # mid-pipeline Phase 5 (after tracking + dual-pose evidence).
    color_hist_extractor = ColorHistogramExtractor()
    enrollment_reject_log: List[Dict[str, Any]] = []
    galleries: List[DancerGallery] = []
    target_people = 0
    target_cap = max(1, int(os.environ.get("SWAY_ENROLLMENT_MAX_IDS", "12") or 12))
    selected_quality_floor = 0.0
    enrollment_completion_ratio = 0.0
    min_enrollment_ratio = max(
        0.0,
        min(1.0, float(os.environ.get("SWAY_ENROLLMENT_MIN_COMPLETION_RATIO", "0.0") or 0.0)),
    )
    phase0_time = 0.0

    # Full video is needed for downstream phases.
    frames = _materialize_all_frames()
    if not frames:
        raise RuntimeError("No readable frames available.")

    # =====================================================================
    # PHASE 1: Detection (every frame)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: YOLO26L Detection")
    logger.info("=" * 70)
    t_phase = time.monotonic()

    all_detections: List[List[Detection]] = []
    detection_provenance: List[str] = []
    det_agreement_scores: List[float] = []
    writer1 = PhaseVideoWriter(OUTPUT_DIR / "phase1_detection.mp4", fps, width, height)
    phase1_heartbeat_every = max(1, int(os.environ.get("SWAY_PHASE1_HEARTBEAT_EVERY", "30") or 30))
    logger.info("Phase 1 heartbeat: every %d frames (set SWAY_PHASE1_HEARTBEAT_EVERY to override)", phase1_heartbeat_every)
    phase1_det_total = 0

    phase3_viz_min_point_conf = max(
        0.0,
        min(1.0, float(os.environ.get("SWAY_PHASE3_VIZ_MIN_POINT_CONF", "0.03") or 0.03)),
    )
    phase3_viz_min_bone_conf = max(
        0.0,
        min(1.0, float(os.environ.get("SWAY_PHASE3_VIZ_MIN_BONE_CONF", "0.05") or 0.05)),
    )
    phase3_viz_adapt_quantile = max(
        0.0,
        min(1.0, float(os.environ.get("SWAY_PHASE3_VIZ_ADAPT_QUANTILE", "0.70") or 0.70)),
    )

    def _crop_mask_to_box(
        full_mask: Optional[np.ndarray],
        box_xyxy: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> Optional[np.ndarray]:
        """Convert full-frame SAM mask into bbox-aligned mask expected by pose backends."""
        if full_mask is None or full_mask.size == 0:
            return None
        x1, y1, x2, y2 = (int(round(box_xyxy[0])), int(round(box_xyxy[1])), int(round(box_xyxy[2])), int(round(box_xyxy[3])))
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(x1 + 1, min(frame_w, x2))
        y2 = max(y1 + 1, min(frame_h, y2))
        if y2 <= y1 or x2 <= x1:
            return None
        m = np.asarray(full_mask).astype(bool)
        if m.shape[0] != frame_h or m.shape[1] != frame_w:
            m = cv2.resize(m.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        cropped = m[y1:y2, x1:x2]
        return cropped if cropped.size > 0 and cropped.any() else None

    for fidx, frame in enumerate(frames):
        dets = _dedupe_detections(detector.detect(frame, frame_idx=fidx))
        source = "yolo26l_dancetrack"

        all_detections.append(dets)
        detection_provenance.append(source)
        phase1_det_total += len(dets)

        phase1_iter = fidx + 1
        if phase1_iter % phase1_heartbeat_every == 0 or phase1_iter == len(frames):
            elapsed = time.monotonic() - t_phase
            eta_s = ((elapsed / max(1, phase1_iter)) * (len(frames) - phase1_iter)) if phase1_iter < len(frames) else 0.0
            det_avg = float(phase1_det_total) / float(max(1, phase1_iter))
            logger.info(
                "  Phase 1 progress %d/%d (%.1f%%) | frame=%d dets=%d avg_dets=%.2f src=%s | elapsed=%.1fs eta=%.1fs",
                phase1_iter,
                len(frames),
                100.0 * phase1_iter / max(1, len(frames)),
                fidx,
                len(dets),
                det_avg,
                source,
                elapsed,
                eta_s,
            )

        out = frame.copy()
        _draw_phase_banner(out, f"Phase 1: Detection | Frame {fidx} | {len(dets)} persons | src={source}")
        for det in dets:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            _draw_text(out, f"{det.confidence:.2f}", (x1, y1 - 8), (0, 255, 0))
        writer1.write(out)

    writer1.release()

    # GAP 1-A: serialize detections with provenance
    det_json = []
    for fidx_d, (dets_d, prov_d) in enumerate(zip(all_detections, detection_provenance)):
        frame_dets = []
        for d in dets_d:
            frame_dets.append({
                "bbox": [float(v) for v in d.bbox],
                "confidence": float(d.confidence),
                "source": prov_d,
            })
        det_json.append({"frame": fidx_d, "detections": frame_dets})
    with open(OUTPUT_DIR / "detections_phase1.json", "w") as f:
        json.dump(det_json, f)
    logger.info("Wrote detections_phase1.json (%d frames)", len(det_json))

    phase1_time = (time.monotonic() - t_phase) * 1000
    logger.info("Phase 1 complete: %d frames, %.1fms (%.1f fps)",
                len(frames), phase1_time, len(frames) / (phase1_time / 1000))
    if detection_provenance:
        logger.info(
            "Phase 1 provenance: detector=yolo26l_dancetrack frames=%d",
            len(detection_provenance),
        )

    det_boxes_by_frame: Dict[int, Sequence[Tuple[float, float, float, float]]] = {}
    for fidx_det, dets in enumerate(all_detections):
        det_boxes_by_frame[fidx_det] = [
            (float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3]))
            for d in dets
        ]
    mode_decision = detect_recording_mode(
        formation_timeline,
        det_boxes_by_frame,
        width=width,
        height=height,
        policy=recording_mode_policy,
        override_mode=recording_mode_override,
    )
    mode_info.update(
        {
            "detected_mode": mode_decision.detected_mode,
            "final_mode": mode_decision.final_mode,
            "mode_confidence": float(mode_decision.confidence),
            "mode_reason": mode_decision.reason,
            "confirmation_required": bool(mode_decision.confirmation_required),
        }
    )
    logger.info(
        "Recording mode decision: detected=%s final=%s conf=%.3f reason=%s confirm=%s",
        mode_decision.detected_mode,
        mode_decision.final_mode,
        float(mode_decision.confidence),
        mode_decision.reason,
        bool(mode_decision.confirmation_required),
    )

    # =====================================================================
    # PHASE 2: Pixel-Perfect Person Masking (SAM2.1 default)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Pixel-Perfect Person Masking (SAM)")
    logger.info("=" * 70)
    t_phase2_mask = time.monotonic()

    all_masks: Dict[int, Dict[int, np.ndarray]] = {}
    mask_quality: Dict[int, Dict[int, float]] = {}
    sam_model_loaded = False
    disable_sam_masking = str(os.environ.get("SWAY_DISABLE_SAM_MASKING", "0")).strip().lower() in {"1", "true", "yes", "on"}

    _sam_env = os.environ.get("SWAY_SAM2_WEIGHTS", "").strip()
    if _sam_env:
        _p = Path(_sam_env).expanduser()
        sam_weights = _p if _p.is_absolute() else (models_dir / _sam_env)
    else:
        sam_weights = models_dir / "sam2.1_hiera_large.pt"
        if not sam_weights.exists():
            sam_weights = models_dir / "sam2.1_l.pt"
        if not sam_weights.exists():
            sam_weights = models_dir / "sam2.1_b.pt"

    sam_uly_model = None
    if disable_sam_masking:
        logger.info("SWAY_DISABLE_SAM_MASKING enabled: skipping SAM loading and mask generation")
    else:
        try:
            from ultralytics import SAM as UlySAM
            if sam_weights.exists():
                uly_weight = str(sam_weights)
            else:
                # Ultralytics hub name (e.g. sam2.1_l.pt) or explicit SWAY_SAM2_WEIGHTS when not on disk.
                uly_weight = _sam_env if _sam_env else "sam2.1_l.pt"
            sam_uly_model = UlySAM(uly_weight)
            sam_model_loaded = True
            logger.info("SAM2.1 loaded via ultralytics: %s", uly_weight)
        except Exception as exc:
            logger.warning("Ultralytics SAM load failed: %s; masking disabled", exc)

    # Current pipeline policy: Phase 2 uses only SAM2.1 direct path (SAM3 disabled).
    use_sam3_tracker = False
    sam3_tracker = None
    if sam_model_loaded and sam_uly_model is not None:
        logger.info("SAM3 disabled by policy; using direct SAM2.1 path only")

    phase2_reused_masks = 0
    phase2_propagated_masks = 0
    phase2_bbox_fill_masks = 0
    phase2_sam_masks = 0
    if sam_model_loaded and sam_uly_model is not None:
        phase2_heartbeat_every = max(1, int(os.environ.get("SWAY_PHASE2_HEARTBEAT_EVERY", "30") or 30))
        mask_frame_stride = max(1, int(os.environ.get("SWAY_MASK_FRAME_STRIDE", "1") or 1))
        mask_reuse_iou = float(os.environ.get("SWAY_MASK_REUSE_IOU", "0.70") or 0.70)
        selective_masking = str(os.environ.get("SWAY_SELECTIVE_MASKING", "0")).strip().lower() not in {"0", "false", "no", "off"}
        mask_propagation_iou = 0.30
        logger.info("Phase 2 heartbeat: every %d frames (set SWAY_PHASE2_HEARTBEAT_EVERY to override)", phase2_heartbeat_every)
        logger.info("Phase 2 runtime knobs: mask_stride=%d reuse_iou=%.2f selective_masking=%s",
                     mask_frame_stride, mask_reuse_iou, selective_masking)
        phase2_mask_total = 0
        phase2_frames_with_masks = 0
        prev_dets_for_reuse: List[np.ndarray] = []
        prev_masks_for_reuse: Dict[int, np.ndarray] = {}
        prev_quality_for_reuse: Dict[int, float] = {}
        for fidx in range(len(frames)):
            dets = all_detections[fidx]
            frame = frames[fidx]
            frame_masks: Dict[int, np.ndarray] = {}
            frame_quality: Dict[int, float] = {}
            bboxes_list = [det.bbox.tolist() for det in dets]
            overlap_now = _has_overlap(dets, iou_thresh=0.35)
            run_full_sam = (fidx == 0) or (fidx % mask_frame_stride == 0) or overlap_now

            # --- Stride frame: reuse or propagate masks from previous frame ---
            if not run_full_sam and prev_dets_for_reuse and prev_masks_for_reuse:
                for di, det in enumerate(dets):
                    best_prev_idx = -1
                    best_iou = 0.0
                    for pidx, pb in enumerate(prev_dets_for_reuse):
                        iou = _bbox_iou(det.bbox, pb)
                        if iou > best_iou:
                            best_iou = iou
                            best_prev_idx = pidx

                    if best_prev_idx >= 0 and best_iou >= mask_reuse_iou and best_prev_idx in prev_masks_for_reuse:
                        frame_masks[di] = prev_masks_for_reuse[best_prev_idx]
                        frame_quality[di] = float(prev_quality_for_reuse.get(best_prev_idx, 0.0))
                        phase2_reused_masks += 1
                    elif best_prev_idx >= 0 and best_iou >= mask_propagation_iou and best_prev_idx in prev_masks_for_reuse:
                        prev_bbox = prev_dets_for_reuse[best_prev_idx]
                        cur_cx, cur_cy = _bbox_center(det.bbox)
                        prev_cx, prev_cy = _bbox_center(prev_bbox)
                        dx = int(round(cur_cx - prev_cx))
                        dy = int(round(cur_cy - prev_cy))
                        propagated = _shift_mask(prev_masks_for_reuse[best_prev_idx], dx, dy)
                        if propagated.any():
                            frame_masks[di] = propagated
                            frame_quality[di] = float(prev_quality_for_reuse.get(best_prev_idx, 0.0)) * 0.9
                            phase2_propagated_masks += 1

            if dets:
                if sam3_tracker is not None and fidx > 0:
                    try:
                        tf = sam3_tracker.track_frame(frame, fidx, new_detections=dets)
                        for di, det in enumerate(dets):
                            for tid, obj in tf.objects.items():
                                if obj.mask is not None and obj.bbox is not None:
                                    if _bbox_iou(det.bbox, obj.bbox) > 0.5:
                                        frame_masks[di] = obj.mask
                                        mask_pixels = float(obj.mask.sum())
                                        bbox_area = max(1.0, float((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])))
                                        frame_quality[di] = min(1.0, mask_pixels / bbox_area)
                                        break
                    except Exception:
                        pass

                missing_dets = [i for i in range(len(dets)) if i not in frame_masks]
                if missing_dets and run_full_sam:
                    overlapping_indices = _get_overlapping_det_indices(dets, iou_thresh=0.15) if selective_masking else set(range(len(dets)))

                    sam_dets = [i for i in missing_dets if i in overlapping_indices]
                    bbox_fill_dets = [i for i in missing_dets if i not in overlapping_indices] if selective_masking else []

                    # Bbox-fill for non-overlapping detections (fast path)
                    for di in bbox_fill_dets:
                        bbox = dets[di].bbox
                        fh, fw = frame.shape[:2]
                        x1 = max(0, int(bbox[0]))
                        y1 = max(0, int(bbox[1]))
                        x2 = min(fw, int(bbox[2]))
                        y2 = min(fh, int(bbox[3]))
                        if x2 > x1 and y2 > y1:
                            bbox_mask = np.zeros((fh, fw), dtype=bool)
                            bbox_mask[y1:y2, x1:x2] = True
                            frame_masks[di] = bbox_mask
                            frame_quality[di] = 0.75
                            phase2_bbox_fill_masks += 1

                    # SAM only for overlapping detections (or all if selective disabled)
                    if sam_dets:
                        try:
                            sam_bboxes = [dets[i].bbox.tolist() for i in sam_dets]
                            results = sam_uly_model(frame, bboxes=sam_bboxes, verbose=False)
                            if results and len(results) > 0 and results[0].masks is not None:
                                masks_data = results[0].masks.data.cpu().numpy()
                                for ri, di in enumerate(sam_dets):
                                    if ri < masks_data.shape[0]:
                                        mask = masks_data[ri].astype(bool)
                                        if mask.any():
                                            frame_masks[di] = mask
                                            mask_pixels = float(mask.sum())
                                            bbox_area = max(1.0, float((dets[di].bbox[2] - dets[di].bbox[0]) * (dets[di].bbox[3] - dets[di].bbox[1])))
                                            frame_quality[di] = min(1.0, mask_pixels / bbox_area)
                                            phase2_sam_masks += 1
                        except Exception as exc:
                            if fidx == 0:
                                logger.warning("SAM inference failed on frame %d: %s", fidx, exc)
                    elif not selective_masking and missing_dets:
                        try:
                            results = sam_uly_model(frame, bboxes=bboxes_list, verbose=False)
                            if results and len(results) > 0 and results[0].masks is not None:
                                masks_data = results[0].masks.data.cpu().numpy()
                                for di in missing_dets:
                                    if di < masks_data.shape[0]:
                                        mask = masks_data[di].astype(bool)
                                        if mask.any():
                                            frame_masks[di] = mask
                                            mask_pixels = float(mask.sum())
                                            bbox_area = max(1.0, float((dets[di].bbox[2] - dets[di].bbox[0]) * (dets[di].bbox[3] - dets[di].bbox[1])))
                                            frame_quality[di] = min(1.0, mask_pixels / bbox_area)
                                            phase2_sam_masks += 1
                        except Exception as exc:
                            if fidx == 0:
                                logger.warning("SAM inference failed on frame %d: %s", fidx, exc)

            if frame_masks:
                all_masks[fidx] = frame_masks
                mask_quality[fidx] = frame_quality
                phase2_frames_with_masks += 1
                phase2_mask_total += len(frame_masks)
                prev_masks_for_reuse = {k: v for k, v in frame_masks.items()}
                prev_quality_for_reuse = {k: float(v) for k, v in frame_quality.items()}
            else:
                prev_masks_for_reuse = {}
                prev_quality_for_reuse = {}
            prev_dets_for_reuse = [np.array(det.bbox, dtype=np.float32) for det in dets]

            phase2_iter = fidx + 1
            if phase2_iter % phase2_heartbeat_every == 0 or phase2_iter == len(frames):
                elapsed = time.monotonic() - t_phase2_mask
                eta_s = ((elapsed / max(1, phase2_iter)) * (len(frames) - phase2_iter)) if phase2_iter < len(frames) else 0.0
                avg_masks = float(phase2_mask_total) / float(max(1, phase2_iter))
                logger.info(
                    "  Phase 2 progress %d/%d (%.1f%%) | frame=%d dets=%d masks=%d avg_masks=%.2f frames_with_masks=%d "
                    "| tracker=%s sam=%d bbox_fill=%d reused=%d propagated=%d | elapsed=%.1fs eta=%.1fs",
                    phase2_iter,
                    len(frames),
                    100.0 * phase2_iter / max(1, len(frames)),
                    fidx,
                    len(dets),
                    len(frame_masks),
                    avg_masks,
                    phase2_frames_with_masks,
                    "active" if sam3_tracker is not None else "direct",
                    phase2_sam_masks,
                    phase2_bbox_fill_masks,
                    phase2_reused_masks,
                    phase2_propagated_masks,
                    elapsed,
                    eta_s,
                )
        logger.info("Phase 2 mask breakdown: sam=%d bbox_fill=%d reused=%d propagated=%d total=%d",
                     phase2_sam_masks, phase2_bbox_fill_masks, phase2_reused_masks, phase2_propagated_masks,
                     phase2_sam_masks + phase2_bbox_fill_masks + phase2_reused_masks + phase2_propagated_masks)
    else:
        logger.info("SAM masking skipped (no model); downstream uses bbox fallback")

    # Write mask phase video
    writer2m = PhaseVideoWriter(OUTPUT_DIR / "phase2_masks.mp4", fps, width, height)
    for fidx, frame in enumerate(frames):
        out = frame.copy()
        fm = all_masks.get(fidx, {})
        _draw_phase_banner(out, f"Phase 2: Masks | Frame {fidx} | {len(fm)} masks")
        dets = all_detections[fidx]
        for di, det in enumerate(dets):
            mask = fm.get(di)
            if mask is not None:
                color = _color_for_id(di)
                overlay = out.copy()
                overlay[mask] = (
                    np.array(color, dtype=np.uint8) * 0.4 + out[mask] * 0.6
                ).astype(np.uint8)
                out = overlay
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        writer2m.write(out)
    writer2m.release()

    # Serialize mask quality
    mask_json = []
    for fidx in range(len(frames)):
        fq = mask_quality.get(fidx, {})
        mask_json.append({"frame": fidx, "mask_count": len(fq), "qualities": {str(k): v for k, v in fq.items()}})
    with open(OUTPUT_DIR / "masks_phase2.json", "w") as f:
        json.dump(mask_json, f)

    phase2_mask_time = (time.monotonic() - t_phase2_mask) * 1000
    logger.info("Phase 2 (Masking) complete: %.1fms, %d frames with masks", phase2_mask_time, len(all_masks))

    # =====================================================================
    # PHASE 3: Dual pose pre-track (ViTPose + RTMW-X)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Dual Pose Pre-Track (ViTPose + RTMW-X)")
    logger.info("=" * 70)
    t_phase3_dualpose = time.monotonic()
    phase35_disambiguation_time = 0.0
    phase35_diag: Dict[str, Any] = {}
    pretrack_pose_by_frame_det: Dict[int, Dict[int, Dict[str, Any]]] = {}
    rtmw_joint_counts_full: List[int] = []
    writer3_vit = PhaseVideoWriter(OUTPUT_DIR / "phase3_vitpose.mp4", fps, width, height)
    writer3_rtmw = PhaseVideoWriter(OUTPUT_DIR / "phase3_rtmwx.mp4", fps, width, height)
    try:
        rtmw_x_pretrack = RTMW384HybridEstimator(device=DEVICE)
    except Exception as exc:
        logger.warning("RTMW-X pre-track init failed: %s", exc)
        rtmw_x_pretrack = None

    for fidx, frame in enumerate(frames):
        dets = all_detections[fidx]
        boxes = [tuple(float(v) for v in d.bbox.tolist()) for d in dets]
        pose_ids = [int(i + 1) for i in range(len(dets))]
        _fm = all_masks.get(fidx, {})
        seg_masks = [
            _crop_mask_to_box(_fm.get(i), boxes[i], width, height) if i < len(boxes) else None
            for i in range(len(dets))
        ]
        if dets:
            try:
                vit_out = pose_estimator.estimate_poses(
                    frame,
                    boxes,
                    pose_ids,
                    segmentation_masks=seg_masks,
                )
            except Exception:
                vit_out = {}
            if rtmw_x_pretrack is not None:
                try:
                    rtmw_out = rtmw_x_pretrack.estimate_poses(
                        frame=frame,
                        boxes=boxes,
                        track_ids=pose_ids,
                        segmentation_masks=seg_masks,
                    )
                except Exception:
                    rtmw_out = {}
            else:
                rtmw_out = {}
        else:
            vit_out = {}
            rtmw_out = {}

        out_rtmw = frame.copy()

        frame_map: Dict[int, Dict[str, Any]] = {}
        vk_for_grade: List[Optional[np.ndarray]] = []
        for di, pid in enumerate(pose_ids):
            v = vit_out.get(int(pid))
            r = rtmw_out.get(int(pid))
            v_kp = np.asarray(v.get("keypoints"), dtype=np.float32) if v and v.get("keypoints") is not None else None
            r_kp = np.asarray(r.get("keypoints"), dtype=np.float32) if r and r.get("keypoints") is not None else None
            r_kp_full = (
                np.asarray(r.get("keypoints_full"), dtype=np.float32)
                if r and r.get("keypoints_full") is not None
                else r_kp
            )
            r_joint_count = int(r_kp_full.shape[0]) if isinstance(r_kp_full, np.ndarray) else 0
            if r_joint_count > 0:
                rtmw_joint_counts_full.append(r_joint_count)
            v_mean = float(np.mean(v_kp[:, 2])) if v_kp is not None and v_kp.shape[0] > 0 else 0.0
            r_mean = float(np.mean(r_kp[:, 2])) if r_kp is not None and r_kp.shape[0] > 0 else 0.0
            frame_map[int(di)] = {
                "vitpose_keypoints": v_kp,
                "rtmwx_keypoints": r_kp,
                "rtmwx_keypoints_full": r_kp_full,
                "rtmwx_joint_count_full": r_joint_count,
                "vitpose_mean_conf": v_mean,
                "rtmwx_mean_conf": r_mean,
            }
            if v_kp is not None and v_kp.shape[0] > 0:
                vk_for_grade.append(v_kp)
            else:
                vk_for_grade.append(None)

        out_vit = frame.copy()
        apply_frame_confidence_grade(out_vit, frame_mean_keypoint_conf(vk_for_grade))
        _draw_phase_banner(out_vit, f"Phase 3: ViTPose | conf-graded | Frame {fidx} | dets={len(dets)}")
        _draw_phase_banner(out_rtmw, f"Phase 3: RTMW-X | Frame {fidx} | dets={len(dets)}")

        for di, _pid in enumerate(pose_ids):
            dual = frame_map[int(di)]
            v_kp = dual["vitpose_keypoints"]
            r_kp_full = dual["rtmwx_keypoints_full"]
            x1, y1, x2, y2 = dets[di].bbox.astype(int)
            cv2.rectangle(out_rtmw, (x1, y1), (x2, y2), (80, 220, 80), 2)

            if v_kp is not None:
                draw_vitpose_confidence_overlay(
                    out_vit,
                    np.asarray(v_kp, dtype=np.float32),
                    bbox_xyxy=(x1, y1, x2, y2),
                    min_point_conf=phase3_viz_min_point_conf,
                    min_bone_conf=phase3_viz_min_bone_conf,
                    adapt_quantile=phase3_viz_adapt_quantile,
                    bones=PHASE3_SKEL_BONES,
                )

            if isinstance(r_kp_full, np.ndarray) and r_kp_full.size > 0:
                _r_scores = r_kp_full[:, 2] if r_kp_full.shape[1] >= 3 else np.zeros((r_kp_full.shape[0],), dtype=np.float32)
                r_point_thr = phase3_adaptive_thresh(
                    _r_scores,
                    default_thr=0.20,
                    floor_thr=phase3_viz_min_point_conf,
                    adapt_quantile=phase3_viz_adapt_quantile,
                )
                r_drawn = 0
                for j in range(r_kp_full.shape[0]):
                    if r_kp_full[j, 2] > r_point_thr:
                        cv2.circle(out_rtmw, (int(r_kp_full[j, 0]), int(r_kp_full[j, 1])), 2, (0, 220, 255), -1)
                        r_drawn += 1
                if r_drawn == 0:
                    order = np.argsort(-r_kp_full[:, 2])
                    for j in order[: min(20, len(order))]:
                        if r_kp_full[j, 2] <= 0.0:
                            continue
                        cv2.circle(out_rtmw, (int(r_kp_full[j, 0]), int(r_kp_full[j, 1])), 2, (0, 220, 255), -1)
                _draw_text(
                    out_rtmw,
                    f"J={r_kp_full.shape[0]}",
                    (x1, max(16, y1 - 8)),
                    (0, 220, 255),
                    0.5,
                )
        pretrack_pose_by_frame_det[int(fidx)] = frame_map
        writer3_vit.write(out_vit)
        writer3_rtmw.write(out_rtmw)
        if fidx % 60 == 0:
            logger.info(
                "  Phase 3 pre-track frame %d/%d | dets=%d",
                fidx,
                len(frames),
                len(dets),
            )
    writer3_vit.release()
    writer3_rtmw.release()
    try:
        overlay_payload: Dict[str, Any] = {
            "metadata": {
                "video_path": str(VIDEO_PATH.resolve()),
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "num_frames": int(len(all_detections)),
                "phase3_viz": {
                    "min_point_conf": float(phase3_viz_min_point_conf),
                    "min_bone_conf": float(phase3_viz_min_bone_conf),
                    "adapt_quantile": float(phase3_viz_adapt_quantile),
                },
            },
            "frames": [],
        }
        for fidx_w in range(len(all_detections)):
            dets_w = all_detections[fidx_w]
            fm_w = pretrack_pose_by_frame_det.get(int(fidx_w), {})
            det_rows: List[Dict[str, Any]] = []
            for di_w in range(len(dets_w)):
                dual_w = fm_w.get(int(di_w), {})
                vk_w = dual_w.get("vitpose_keypoints")
                rk_full_w = dual_w.get("rtmwx_keypoints_full")
                row: Dict[str, Any] = {
                    "bbox": [float(b) for b in dets_w[di_w].bbox],
                    "detector_confidence": float(dets_w[di_w].confidence),
                    "vitpose_mean_conf": float(dual_w.get("vitpose_mean_conf", 0.0) or 0.0),
                    "rtmwx_mean_conf": float(dual_w.get("rtmwx_mean_conf", 0.0) or 0.0),
                }
                if vk_w is not None:
                    row["vitpose_keypoints"] = np.asarray(vk_w, dtype=np.float32).tolist()
                if rk_full_w is not None:
                    row["rtmwx_keypoints_full"] = np.asarray(rk_full_w, dtype=np.float32).tolist()
                det_rows.append(row)
            overlay_payload["frames"].append({"frame_idx": int(fidx_w), "detections": det_rows})
        _ov_path = OUTPUT_DIR / "phase3_vitpose_overlay.json"
        with open(_ov_path, "w") as f:
            json.dump(overlay_payload, f)
        logger.info("Wrote phase3_vitpose_overlay.json (%d frames)", len(overlay_payload["frames"]))
    except Exception as exc:
        logger.warning("phase3_vitpose_overlay.json write failed: %s", exc)
    if rtmw_x_pretrack is not None:
        max_rtmw_joints = max(rtmw_joint_counts_full) if rtmw_joint_counts_full else 0
        logger.info("Phase 3 RTMW-X joint cardinality (max observed): %d", max_rtmw_joints)
        if max_rtmw_joints < 100:
            raise RuntimeError(
                f"RTMW-X full-joint requirement failed: expected >=100 joints, observed {max_rtmw_joints}."
            )
    phase3_dualpose_time = (time.monotonic() - t_phase3_dualpose) * 1000
    logger.info("Phase 3 (Dual Pose Pre-Track) complete: %.1fms", phase3_dualpose_time)

    # =====================================================================
    # PHASE 3.5: Ambiguity resolver (feature-flagged sidecar)
    # =====================================================================
    if phase35_enabled and phase35_mode in {"shadow", "active"}:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3.5: Ambiguity Resolver (%s mode)", phase35_mode)
        logger.info("=" * 70)
        phase35_status["ran"] = True
        t_phase35 = time.monotonic()
        try:
            all_detection_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
            for fidx in range(len(all_detections)):
                all_detection_boxes[int(fidx)] = [
                    tuple(float(v) for v in d.bbox.tolist()) for d in all_detections[fidx]
                ]
            resolved_map, phase35_diag = resolve_phase35(
                pretrack_pose_by_frame_det=pretrack_pose_by_frame_det,
                all_detection_boxes=all_detection_boxes,
                cfg=phase35_cfg,
            )
            phase35_disambiguation_time = (time.monotonic() - t_phase35) * 1000
            phase35_status["elapsed_ms"] = float(phase35_disambiguation_time)
            if phase35_timeout_ms > 0 and phase35_disambiguation_time > phase35_timeout_ms:
                raise TimeoutError(
                    f"phase3.5 timed out budget={phase35_timeout_ms:.1f}ms "
                    f"actual={phase35_disambiguation_time:.1f}ms"
                )
            if not isinstance(resolved_map, dict):
                raise RuntimeError("phase3.5 returned invalid map payload")
            det_total = int(phase35_diag.get("detections_total", 0) or 0)
            overrides_total = int(phase35_diag.get("overrides_total", 0) or 0)
            phase35_status["override_ratio"] = float(overrides_total) / float(max(1, det_total))
            if phase35_mode == "active":
                pretrack_pose_by_frame_det = resolved_map
                phase35_status["applied"] = True
            disamb_path = OUTPUT_DIR / "phase3_5_disambiguation.json"
            metrics_path = OUTPUT_DIR / "phase3_5_metrics.json"
            with open(disamb_path, "w") as f:
                json.dump(
                    {
                        "schema_version": "phase3_5_disambiguation_v1",
                        "mode": phase35_mode,
                        "applied": bool(phase35_status["applied"]),
                        "fallback": bool(phase35_status["fail_open"]),
                        "diagnostics": phase35_diag,
                    },
                    f,
                )
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "schema_version": "phase3_5_metrics_v1",
                        "mode": phase35_mode,
                        "elapsed_ms": float(phase35_disambiguation_time),
                        "override_ratio": float(phase35_status["override_ratio"]),
                        "detections_total": int(phase35_diag.get("detections_total", 0) or 0),
                        "detections_ambiguous": int(phase35_diag.get("detections_ambiguous", 0) or 0),
                        "overrides_total": int(phase35_diag.get("overrides_total", 0) or 0),
                        "bins": dict(phase35_diag.get("bins", {})),
                        "margin_stats": dict(phase35_diag.get("margin_stats", {})),
                    },
                    f,
                    indent=2,
                )
            phase35_status["diagnostics_path"] = "phase3_5_disambiguation.json"
            phase35_status["metrics_path"] = "phase3_5_metrics.json"
            logger.info(
                "Phase 3.5 complete: %.1fms | ambiguous=%d overrides=%d mode=%s applied=%s",
                phase35_disambiguation_time,
                int(phase35_diag.get("detections_ambiguous", 0) or 0),
                int(phase35_diag.get("overrides_total", 0) or 0),
                phase35_mode,
                phase35_status["applied"],
            )
        except Exception as exc:
            phase35_status["fallback_count"] = int(phase35_status.get("fallback_count", 0) or 0) + 1
            if not phase35_force_fail_open:
                raise
            phase35_status["fail_open"] = True
            phase35_status["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Phase 3.5 fail-open fallback to baseline: %s", exc)
    elif phase35_enabled and phase35_mode == "off":
        logger.info("Phase 3.5 configured but mode=off, skipping.")

    # =====================================================================
    # PHASE 4: Tracking (Forward + Backward)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Bidirectional Tracking")
    logger.info("=" * 70)
    t_phase = time.monotonic()

    all_track_results: List[np.ndarray] = []
    frame_contaminated_tids: Dict[int, set] = {}
    writer2 = PhaseVideoWriter(OUTPUT_DIR / "phase4_tracking_forward.mp4", fps, width, height)

    for fidx, frame in enumerate(frames):
        dets = all_detections[fidx]

        if dets:
            det_array = np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, 0] for d in dets],
                dtype=np.float32,
            )
        else:
            det_array = np.empty((0, 6), dtype=np.float32)

        tracks = boxmot_tracker.update(det_array, frame)
        all_track_results.append(tracks)
        contaminated = set()
        for i in range(len(tracks)):
            bi = np.array([tracks[i][0], tracks[i][1], tracks[i][2], tracks[i][3]], dtype=np.float32)
            for j in range(i + 1, len(tracks)):
                bj = np.array([tracks[j][0], tracks[j][1], tracks[j][2], tracks[j][3]], dtype=np.float32)
                if _bbox_iou(bi, bj) > 0.35:
                    contaminated.add(int(tracks[i][4]))
                    contaminated.add(int(tracks[j][4]))
        frame_contaminated_tids[fidx] = contaminated

        if fidx % 30 == 0:
            logger.info(
                "  Frame %d: %d detections -> %d tracks (contaminated=%d)",
                fidx,
                len(dets),
                len(tracks),
                len(contaminated),
            )

        out = frame.copy()
        _draw_phase_banner(out, f"Phase 4: BoxMOT Tracking | Frame {fidx} | {len(tracks)} tracks")

        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            color = _color_for_id(tid)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            _draw_text(out, f"T{tid}", (x1, y1 - 8), color)

        writer2.write(out)

    writer2.release()
    phase3_track_time = (time.monotonic() - t_phase) * 1000
    logger.info("Phase 4 (Forward Tracking) complete: %.1fms", phase3_track_time)

    tracker_ab_report: Dict[str, Any] = {"enabled": False}
    enable_tracker_ab = str(os.environ.get("SWAY_TRACKER_AB", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if enable_tracker_ab:
        tracker_ab_backends = [
            x.strip().lower()
            for x in os.environ.get("SWAY_TRACKER_AB_BACKENDS", "deepocsort,strongsort,botsort,ocsort").split(",")
            if x.strip()
        ]
        tracker_ab_min_overlap_frames = max(1, int(os.environ.get("SWAY_TRACKER_AB_MIN_OVERLAP_FRAMES", "8") or 8))

        overlap_slice_frames: List[int] = []
        overlap_slice_mode = "contamination_runs"
        run_start = None
        for fidx_ab in range(len(frames)):
            contam_ab = frame_contaminated_tids.get(fidx_ab, set())
            if len(contam_ab) >= 2 and run_start is None:
                run_start = fidx_ab
            elif len(contam_ab) < 2 and run_start is not None:
                if (fidx_ab - run_start) >= tracker_ab_min_overlap_frames:
                    overlap_slice_frames.extend(list(range(run_start, fidx_ab)))
                run_start = None
        if run_start is not None and (len(frames) - run_start) >= tracker_ab_min_overlap_frames:
            overlap_slice_frames.extend(list(range(run_start, len(frames))))
        overlap_slice_frames = sorted(set(overlap_slice_frames))
        if not overlap_slice_frames:
            # Fallback for clips with sparse/no overlap contamination:
            # select a dense contiguous window by track activity.
            fallback_window = min(
                len(frames),
                max(
                    tracker_ab_min_overlap_frames,
                    int(os.environ.get("SWAY_TRACKER_AB_FALLBACK_WINDOW", "96") or 96),
                ),
            )
            if fallback_window > 0:
                if len(frames) <= fallback_window:
                    best_start = 0
                else:
                    best_start = 0
                    best_score = sum(len(all_track_results[i]) for i in range(fallback_window))
                    score = best_score
                    for start in range(1, len(frames) - fallback_window + 1):
                        score += len(all_track_results[start + fallback_window - 1]) - len(all_track_results[start - 1])
                        if score > best_score:
                            best_score = score
                            best_start = start
                overlap_slice_frames = list(range(best_start, best_start + fallback_window))
                overlap_slice_mode = "fallback_dense_window"

        tracker_ab_report = {
            "enabled": True,
            "requested_backends": tracker_ab_backends,
            "min_overlap_frames": tracker_ab_min_overlap_frames,
            "overlap_slice_mode": overlap_slice_mode,
            "overlap_slice_frame_count": len(overlap_slice_frames),
            "results": [],
        }
        if overlap_slice_frames:
            logger.info(
                "Tracker A/B enabled: evaluating %d backends on %d overlap frames",
                len(tracker_ab_backends),
                len(overlap_slice_frames),
            )
            for backend_name in tracker_ab_backends:
                no_fallback_backends = {"strongsort", "botsort"}
                allow_fallback = backend_name not in no_fallback_backends
                trk, effective_name, backend_status, backend_error = _make_tracker_for_backend(
                    backend_name,
                    allow_fallback=allow_fallback,
                )
                if trk is None:
                    tracker_ab_report["results"].append(
                        {
                            "backend_requested": backend_name,
                            "backend_effective": effective_name,
                            "status": backend_status,
                            "init_error": backend_error,
                            "overlap_frames_evaluated": len(overlap_slice_frames),
                        }
                    )
                    continue
                unique_ids = set()
                tracks_total = 0
                contam_total = 0
                for fidx_ab in overlap_slice_frames:
                    dets_ab = all_detections[fidx_ab]
                    if dets_ab:
                        det_array_ab = np.array(
                            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, 0] for d in dets_ab],
                            dtype=np.float32,
                        )
                    else:
                        det_array_ab = np.empty((0, 6), dtype=np.float32)
                    tr = trk.update(det_array_ab, frames[fidx_ab])
                    tracks_total += int(len(tr))
                    for i in range(len(tr)):
                        unique_ids.add(int(tr[i][4]))
                        bi = np.array([tr[i][0], tr[i][1], tr[i][2], tr[i][3]], dtype=np.float32)
                        for j in range(i + 1, len(tr)):
                            bj = np.array([tr[j][0], tr[j][1], tr[j][2], tr[j][3]], dtype=np.float32)
                            if _bbox_iou(bi, bj) > 0.35:
                                contam_total += 1

                tracker_ab_report["results"].append(
                    {
                        "backend_requested": backend_name,
                        "backend_effective": effective_name,
                        "status": backend_status,
                        "init_error": backend_error,
                        "overlap_frames_evaluated": len(overlap_slice_frames),
                        "total_tracks_emitted": int(tracks_total),
                        "mean_tracks_per_frame": float(tracks_total) / float(max(1, len(overlap_slice_frames))),
                        "unique_track_ids": int(len(unique_ids)),
                        "overlap_conflict_pairs": int(contam_total),
                    }
                )
        else:
            logger.info("Tracker A/B enabled but no overlap slices met minimum duration")

    # GAP 3-A: Backward tracking pass
    logger.info("Running backward tracking pass...")
    t_bwd = time.monotonic()
    backward_tracker = _make_deepocsort_tracker()
    all_backward_tracks: List[np.ndarray] = []
    n_frames = len(frames)
    for rev_idx in range(n_frames):
        orig_idx = n_frames - 1 - rev_idx
        dets = all_detections[orig_idx]
        if dets:
            det_array = np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, 0] for d in dets],
                dtype=np.float32,
            )
        else:
            det_array = np.empty((0, 6), dtype=np.float32)
        bwd_tracks = backward_tracker.update(det_array, frames[orig_idx])
        all_backward_tracks.append(bwd_tracks)
    # Reverse so index matches original timeline
    all_backward_tracks = all_backward_tracks[::-1]
    bwd_time = (time.monotonic() - t_bwd) * 1000
    logger.info("Backward tracking complete: %.1fms, %d frames", bwd_time, len(all_backward_tracks))

    # Serialize tracklets
    def _serialize_tracks(tracks_list, path):
        data = []
        for fidx, tr in enumerate(tracks_list):
            frame_tracks = []
            for t in tr:
                frame_tracks.append({
                    "bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                    "track_id": int(t[4]),
                })
            data.append({"frame": fidx, "tracks": frame_tracks})
        with open(path, "w") as f:
            json.dump(data, f)

    _serialize_tracks(all_track_results, OUTPUT_DIR / "tracklets_forward.json")
    _serialize_tracks(all_backward_tracks, OUTPUT_DIR / "tracklets_backward.json")
    logger.info("Wrote tracklets_forward.json and tracklets_backward.json")

    def _build_phase3_raw_tracks(tracks_list) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float], float]]]:
        raw_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], float]]] = {}
        for fidx_rt, tr_rt in enumerate(tracks_list):
            for t_rt in tr_rt:
                tid_rt = int(t_rt[4])
                x1_rt, y1_rt, x2_rt, y2_rt = float(t_rt[0]), float(t_rt[1]), float(t_rt[2]), float(t_rt[3])
                conf_rt = float(t_rt[5]) if len(t_rt) > 5 else 1.0
                raw_tracks.setdefault(tid_rt, []).append((fidx_rt, (x1_rt, y1_rt, x2_rt, y2_rt), conf_rt))
        return raw_tracks

    def _write_phase3_data_json(path: Path, tracks_list) -> None:
        from sway.mot_format import build_phase3_tracking_data_json

        raw_tracks = _build_phase3_raw_tracks(tracks_list)
        data_json = build_phase3_tracking_data_json(
            video_path=str(VIDEO_PATH),
            raw_tracks=raw_tracks,
            total_frames=len(frames),
            native_fps=float(fps),
            output_fps=float(fps),
        )
        # Enrich metadata for downstream TrackEval wrappers.
        data_json.setdefault("metadata", {})
        data_json["metadata"]["frame_width"] = int(width)
        data_json["metadata"]["frame_height"] = int(height)
        with open(path, "w") as f:
            json.dump(data_json, f, indent=2)
        logger.info("Wrote data.json (phase3 tracking export)")

    # Bidirectional tracking video
    writer3b = PhaseVideoWriter(OUTPUT_DIR / "phase4_tracking_bidirectional.mp4", fps, width, height)
    for fidx, frame in enumerate(frames):
        out = frame.copy()
        fwd_tr = all_track_results[fidx]
        bwd_tr = all_backward_tracks[fidx]
        _draw_phase_banner(out, f"Phase 4: Bidir Tracking | Frame {fidx} | fwd={len(fwd_tr)} bwd={len(bwd_tr)}")
        for t in fwd_tr:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            cv2.rectangle(out, (x1, y1), (x2, y2), _color_for_id(tid), 2)
            _draw_text(out, f"F{tid}", (x1, y1 - 8), _color_for_id(tid))
        for t in bwd_tr:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
        writer3b.write(out)
    writer3b.release()

    # --- Solution C: Early structural/reflection pruning (pre-ReID) ---
    # Remove noisy short-lived tracks and reflection-like extras before Re-ID to reduce
    # fragmentation and prevent ghost IDs from entering identity association.
    # Preserves tracks involved in dark-zone stitches.
    short_track_min_frames = int(os.environ.get("SWAY_SHORT_TRACK_MIN_FRAMES", "8") or 8)
    short_track_min_conf = float(os.environ.get("SWAY_SHORT_TRACK_MIN_CONF", "0.30") or 0.30)
    reflection_edge_margin = float(os.environ.get("SWAY_REFLECTION_EDGE_MARGIN_FRAC", "0.16") or 0.16)
    reflection_edge_presence = float(os.environ.get("SWAY_REFLECTION_EDGE_PRESENCE_FRAC", "0.60") or 0.60)
    reflection_min_sign_conflict = float(os.environ.get("SWAY_REFLECTION_MIN_SIGN_CONFLICT_FRAC", "0.55") or 0.55)
    reflection_max_height_frac = float(os.environ.get("SWAY_REFLECTION_MAX_HEIGHT_FRAC", "0.90") or 0.90)
    formation_overcap_buffer = int(os.environ.get("SWAY_FORMATION_OVERCAP_BUFFER", "1") or 1)
    formation_prune_max_per_iter = int(os.environ.get("SWAY_FORMATION_PRUNE_MAX_PER_ITER", "2") or 2)
    _track_lengths: Dict[int, int] = {}
    _track_mean_conf: Dict[int, float] = {}
    _track_det_confs: Dict[int, List[float]] = {}
    _track_centers_x: Dict[int, List[float]] = {}
    _track_heights: Dict[int, List[float]] = {}
    _track_vx_sign_conflict: Dict[int, float] = {}
    _track_edge_presence: Dict[int, float] = {}
    for fidx_st, tr_st in enumerate(all_track_results):
        for t_st in tr_st:
            tid_st = int(t_st[4])
            _track_lengths[tid_st] = _track_lengths.get(tid_st, 0) + 1
            x1_st = float(t_st[0])
            x2_st = float(t_st[2])
            y1_st = float(t_st[1])
            y2_st = float(t_st[3])
            cx_st = 0.5 * (x1_st + x2_st)
            h_st = max(0.0, y2_st - y1_st)
            _track_centers_x.setdefault(tid_st, []).append(cx_st)
            if h_st > 0.0:
                _track_heights.setdefault(tid_st, []).append(h_st)
            conf_st = float(t_st[5]) if len(t_st) > 5 else 0.5
            _track_det_confs.setdefault(tid_st, []).append(conf_st)
    for tid_st, confs_st in _track_det_confs.items():
        _track_mean_conf[tid_st] = float(np.mean(confs_st))

    # Pre-compute global directional trend and track-level reflection stats.
    group_vx_samples: List[float] = []
    for tr_st in all_track_results:
        if len(tr_st) <= 0:
            continue
        cxs = [0.5 * (float(t[0]) + float(t[2])) for t in tr_st]
        if cxs:
            group_vx_samples.append(float(np.median(cxs)))
    group_dx = [group_vx_samples[i] - group_vx_samples[i - 1] for i in range(1, len(group_vx_samples))]
    group_mean_vx = float(np.median(group_dx)) if group_dx else 0.0
    edge_left = width * reflection_edge_margin
    edge_right = width * (1.0 - reflection_edge_margin)
    median_track_height = float(np.median([h for hs in _track_heights.values() for h in hs])) if _track_heights else 0.0

    for tid_st, cxs in _track_centers_x.items():
        if len(cxs) < 2:
            _track_vx_sign_conflict[tid_st] = 0.0
            _track_edge_presence[tid_st] = 0.0
            continue
        deltas = [cxs[i] - cxs[i - 1] for i in range(1, len(cxs))]
        mean_vx = float(np.mean(deltas)) if deltas else 0.0
        conflict = 0.0
        if abs(group_mean_vx) >= 0.8 and abs(mean_vx) >= 0.8:
            conflict = 1.0 if (group_mean_vx * mean_vx) < 0.0 else 0.0
        edge_hits = sum(1 for cx in cxs if cx <= edge_left or cx >= edge_right)
        _track_edge_presence[tid_st] = float(edge_hits) / float(max(1, len(cxs)))
        _track_vx_sign_conflict[tid_st] = conflict

    pruned_short_tids: set = set()
    pruned_reflection_tids: set = set()
    for tid_st, length_st in _track_lengths.items():
        if length_st >= short_track_min_frames:
            continue
        if _track_mean_conf.get(tid_st, 0.0) >= 0.70:
            continue
        pruned_short_tids.add(tid_st)

    # Aggressive pre-ID reflection prune (hard gate). This runs before any assignment.
    for tid_st in _track_lengths.keys():
        if tid_st in pruned_short_tids:
            continue
        edge_presence = float(_track_edge_presence.get(tid_st, 0.0))
        sign_conflict = float(_track_vx_sign_conflict.get(tid_st, 0.0))
        med_h = float(np.median(_track_heights.get(tid_st, [0.0])))
        rel_h = (med_h / median_track_height) if median_track_height > 1e-6 else 1.0
        if (
            edge_presence >= reflection_edge_presence
            and sign_conflict >= reflection_min_sign_conflict
            and rel_h <= reflection_max_height_frac
        ):
            pruned_reflection_tids.add(tid_st)

    # Formation-count-aware extra prune: when tracks exceed performer cap, peel off the
    # strongest reflection-like candidates first to keep ID stage from over-counting.
    expected_performers = len(formation_timeline.performers) if formation_timeline is not None else 0
    formation_overcap_tids: set = set()
    if (
        mode_info.get("final_mode") == "formation"
        and expected_performers > 0
    ):
        active_tids = set(_track_lengths.keys()) - pruned_short_tids - pruned_reflection_tids
        allowed_tracks = max(1, expected_performers + max(0, formation_overcap_buffer))
        over_count = max(0, len(active_tids) - allowed_tracks)
        if over_count > 0:
            candidates: List[Tuple[float, int]] = []
            for tid_st in active_tids:
                edge_presence = float(_track_edge_presence.get(tid_st, 0.0))
                sign_conflict = float(_track_vx_sign_conflict.get(tid_st, 0.0))
                mean_conf = float(_track_mean_conf.get(tid_st, 0.0))
                med_h = float(np.median(_track_heights.get(tid_st, [0.0])))
                rel_h = (med_h / median_track_height) if median_track_height > 1e-6 else 1.0
                # Higher score => more likely reflection/ghost.
                score = (
                    0.52 * edge_presence
                    + 0.28 * sign_conflict
                    + 0.12 * max(0.0, min(1.0, 1.1 - rel_h))
                    + 0.08 * max(0.0, min(1.0, 0.9 - mean_conf))
                )
                if score >= 0.50:
                    candidates.append((score, tid_st))
            candidates.sort(reverse=True)
            max_take = max(1, min(over_count, max(1, formation_prune_max_per_iter)))
            for _, tid_st in candidates[:max_take]:
                formation_overcap_tids.add(tid_st)

    pruned_early_tids = set(pruned_short_tids) | set(pruned_reflection_tids) | set(formation_overcap_tids)

    if pruned_short_tids:
        logger.info(
            "Solution C: pruning %d short tracks (<%d frames, low conf): %s",
            len(pruned_short_tids), short_track_min_frames,
            sorted(pruned_short_tids)[:20],
        )
    if pruned_reflection_tids:
        logger.info(
            "Solution C+: pruning %d reflection-like tracks (edge/inverted/size gate): %s",
            len(pruned_reflection_tids),
            sorted(pruned_reflection_tids)[:20],
        )
    if formation_overcap_tids:
        logger.info(
            "Solution C++: formation over-cap prune %d tracks (expected=%d buffer=%d): %s",
            len(formation_overcap_tids),
            int(expected_performers),
            int(formation_overcap_buffer),
            sorted(formation_overcap_tids)[:20],
        )
    if pruned_early_tids:
        for fidx_st in range(len(all_track_results)):
            all_track_results[fidx_st] = np.array(
                [t for t in all_track_results[fidx_st] if int(t[4]) not in pruned_early_tids],
                dtype=all_track_results[fidx_st].dtype,
            ).reshape(-1, all_track_results[fidx_st].shape[1]) if len(all_track_results[fidx_st]) > 0 else all_track_results[fidx_st]

    phase3_prune_diag = {
        "short_track_pruned": int(len(pruned_short_tids)),
        "reflection_pruned": int(len(pruned_reflection_tids)),
        "formation_overcap_pruned": int(len(formation_overcap_tids)),
        "total_pre_reid_pruned": int(len(pruned_early_tids)),
        "group_mean_vx": float(group_mean_vx),
        "expected_performers": int(expected_performers),
        "formation_overcap_buffer": int(formation_overcap_buffer),
        "reflection_edge_margin_frac": float(reflection_edge_margin),
        "reflection_edge_presence_frac": float(reflection_edge_presence),
        "reflection_min_sign_conflict_frac": float(reflection_min_sign_conflict),
        "reflection_max_height_frac": float(reflection_max_height_frac),
    }

    # =====================================================================
    # PHASE 5: Enrollment after tracking (SAM2.1 + ViTPose + RTMW-X evidence)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Enrollment from tracking + SAM2.1 + dual-pose evidence")
    logger.info("=" * 70)
    t_phase0_deferred = time.monotonic()

    # Select the richest tracked frame after Phase 2/3 processing.
    _best_enroll_frame_idx = -1
    _best_enroll_count = -1
    _best_enroll_tracks: np.ndarray = np.empty((0, 6), dtype=np.float32)
    for _fidx, _tracks in enumerate(all_track_results):
        _c = int(len(_tracks))
        if _c > _best_enroll_count:
            _best_enroll_count = _c
            _best_enroll_frame_idx = int(_fidx)
            _best_enroll_tracks = _tracks

    deferred_enrollment_dets: List[Detection] = []
    deferred_masks_by_idx: Dict[int, np.ndarray] = {}
    if _best_enroll_frame_idx >= 0 and len(_best_enroll_tracks) > 0:
        _eframe = frames[_best_enroll_frame_idx]
        for _ti, _t in enumerate(_best_enroll_tracks):
            _bbox = np.array([_t[0], _t[1], _t[2], _t[3]], dtype=np.float32)
            _conf = float(_t[5]) if len(_t) > 5 else 0.5
            deferred_enrollment_dets.append(
                Detection(bbox=_bbox, confidence=_conf, class_id=0)
            )

        if sam_uly_model is not None and deferred_enrollment_dets:
            try:
                _sam_boxes = [d.bbox.tolist() for d in deferred_enrollment_dets]
                _sam_res = sam_uly_model(_eframe, bboxes=_sam_boxes, verbose=False)
                if _sam_res and len(_sam_res) > 0 and _sam_res[0].masks is not None:
                    _masks_data = _sam_res[0].masks.data.cpu().numpy()
                    for _mi in range(min(len(deferred_enrollment_dets), _masks_data.shape[0])):
                        _m = _masks_data[_mi].astype(bool)
                        if _m.any():
                            deferred_masks_by_idx[_mi] = _m
            except Exception as _sam_exc:
                logger.warning("Deferred enrollment SAM2.1 mask extraction failed: %s", _sam_exc)

        # Use precomputed Phase-3 dual-pose evidence (balanced policy:
        # SAM2.1 quality + either ViTPose OR RTMW-X high quality).
        _joint_keep_idx: List[int] = []
        _vit_thr = float(os.environ.get("SWAY_ENROLL_VITPOSE_MIN_CONF", "0.35") or 0.35)
        _rtmw_thr = float(os.environ.get("SWAY_ENROLL_RTMWX_MIN_CONF", "0.35") or 0.35)
        _sam_q_thr = float(os.environ.get("SWAY_ENROLL_SAM_MASK_MIN_QUALITY", "0.20") or 0.20)
        _dual_map = pretrack_pose_by_frame_det.get(int(_best_enroll_frame_idx), {})
        _preferred_idx: List[int] = []
        for _i, _det in enumerate(deferred_enrollment_dets):
            _best_di = -1
            _best_iou = 0.0
            for _di0, _det0 in enumerate(all_detections[_best_enroll_frame_idx]):
                _iou = _bbox_iou(_det.bbox, _det0.bbox)
                if _iou > _best_iou:
                    _best_iou = _iou
                    _best_di = _di0
            _dual = _dual_map.get(int(_best_di), {})
            _v_mean = float(_dual.get("vitpose_mean_conf", 0.0) or 0.0)
            _r_mean = float(_dual.get("rtmwx_mean_conf", 0.0) or 0.0)
            _sam_q = float(mask_quality.get(int(_best_enroll_frame_idx), {}).get(int(_best_di), 0.0))
            _sam_ok = (_best_di in deferred_masks_by_idx and _sam_q >= _sam_q_thr)
            _pose_ok = (_v_mean >= _vit_thr) or (_r_mean >= _rtmw_thr)
            if _sam_ok and _pose_ok:
                _joint_keep_idx.append(_i)
                if (_v_mean >= _vit_thr) and (_r_mean >= _rtmw_thr):
                    _preferred_idx.append(_i)
        if _preferred_idx:
            _joint_keep_idx = _preferred_idx
        if not _joint_keep_idx:
            _joint_keep_idx = list(range(len(deferred_enrollment_dets)))

        if _joint_keep_idx:
            deferred_enrollment_dets = [deferred_enrollment_dets[i] for i in _joint_keep_idx]
            deferred_masks_by_idx = {
                new_i: deferred_masks_by_idx[old_i]
                for new_i, old_i in enumerate(_joint_keep_idx)
                if old_i in deferred_masks_by_idx
            }

        class _DeferredFaceModel:
            def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
                return extract_face_embedding(crop_bgr)

        _enroll_kp_by_idx: Dict[int, np.ndarray] = {}
        for _di_new, _det_new in enumerate(deferred_enrollment_dets):
            _best_di = -1
            _best_iou = 0.0
            for _di0, _det0 in enumerate(all_detections[_best_enroll_frame_idx]):
                _iou = _bbox_iou(_det_new.bbox, _det0.bbox)
                if _iou > _best_iou:
                    _best_iou = _iou
                    _best_di = _di0
            _dual = _dual_map.get(int(_best_di), {})
            _vk = _dual.get("vitpose_keypoints")
            _rk = _dual.get("rtmwx_keypoints")
            if isinstance(_vk, np.ndarray) and isinstance(_rk, np.ndarray):
                _v_mean = float(np.mean(_vk[:, 2])) if _vk.shape[0] > 0 else 0.0
                _r_mean = float(np.mean(_rk[:, 2])) if _rk.shape[0] > 0 else 0.0
                _enroll_kp_by_idx[_di_new] = _vk if _v_mean >= _r_mean else _rk
            elif isinstance(_vk, np.ndarray):
                _enroll_kp_by_idx[_di_new] = _vk
            elif isinstance(_rk, np.ndarray):
                _enroll_kp_by_idx[_di_new] = _rk

        _enroll_models: Dict[str, Any] = {
            "part_reid": bpbreid_part_extractor,
            "color_hist": color_hist_extractor,
            "face_reid": _DeferredFaceModel(),
            "keypoints": _enroll_kp_by_idx,
        }
        galleries = enroll_dancers(
            _eframe,
            deferred_enrollment_dets,
            sam2_masks=deferred_masks_by_idx or None,
            models=_enroll_models,
            frame_idx=int(_best_enroll_frame_idx),
        )
    else:
        galleries = []

    from sway.enrollment import save_gallery

    save_gallery(galleries, OUTPUT_DIR / "gallery_identity_bank.json")
    fusion_engine = ReIDFusionEngine(gallery=galleries, mlp_router=ReIDMLPRouter())
    phase0_time = (time.monotonic() - t_phase0_deferred) * 1000
    target_people = len(galleries)
    target_cap = max(target_cap, len(galleries))
    enrollment_completion_ratio = 1.0 if target_people > 0 else 0.0
    selected_quality_floor = 0.0
    logger.info(
        "Phase 5 enrollment complete: %.1fms | frame=%d | enrolled=%d",
        phase0_time,
        _best_enroll_frame_idx,
        len(galleries),
    )

    stop_after_phase3 = str(os.environ.get("SWAY_STOP_AFTER_PHASE3", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if stop_after_phase3:
        _write_phase3_data_json(OUTPUT_DIR / "data.json", all_track_results)
        if tracker_ab_report.get("enabled"):
            with open(OUTPUT_DIR / "tracker_ab_overlap.json", "w") as f:
                json.dump(tracker_ab_report, f, indent=2)
            logger.info("Wrote tracker_ab_overlap.json (%d backend results)", len(tracker_ab_report.get("results", [])))
        quick_summary = {
            "video": str(VIDEO_PATH),
            "total_frames": len(frames),
            "phase_times_ms": {
                "phase1_detection": phase1_time,
                "phase2_masking": phase2_mask_time,
                "phase3_dual_pose": phase3_dualpose_time,
                "phase3_5_disambiguation": phase35_disambiguation_time,
                "phase4_tracking_fwd": phase3_track_time,
                "phase4_tracking_bwd": bwd_time,
            },
            "tracker_backend_requested": tracker_backend_requested,
            "tracker_backend_effective": tracker_backend_effective,
            "tracker_backend_status": tracker_backend_status,
            "tracker_backend_error": tracker_backend_error,
            "tracker_ab_report": tracker_ab_report,
            "pre_reid_pruning": phase3_prune_diag,
            "phase35_status": phase35_status,
            "note": "Stopped after tracking stage due to SWAY_STOP_AFTER_PHASE3=1",
        }
        with open(OUTPUT_DIR / "summary_phase3_only.json", "w") as f:
            json.dump(quick_summary, f, indent=2)
        logger.info("Wrote summary_phase3_only.json")
        _write_baseline_validation_report(
            output_dir=OUTPUT_DIR,
            video_path=VIDEO_PATH,
            summary_obj=quick_summary,
            eval_metrics_obj=None,
            phase35_status=phase35_status,
        )
        phase3_manifest = _build_run_manifest(
            repo_root=Path(__file__).resolve().parent.parent,
            device=DEVICE,
            video_path=VIDEO_PATH,
            output_dir=OUTPUT_DIR,
            tracker_backend_requested=tracker_backend_requested,
            tracker_backend_effective=tracker_backend_effective,
            precision_requested=precision_requested,
            precision_effective=precision_effective_name,
            phase_times_ms=quick_summary["phase_times_ms"],
            tracker_ab_report=tracker_ab_report,
            summary=quick_summary,
            reid_feature_mode=reid_feature_mode,
            reid_feature_mode_reason=reid_feature_mode_reason,
        )
        with open(OUTPUT_DIR / "run_manifest.json", "w") as f:
            json.dump(phase3_manifest, f, indent=2)
        logger.info("Wrote run_manifest.json")
        _write_artifact_indexes(
            output_dir=OUTPUT_DIR,
            summary=quick_summary,
            run_manifest=phase3_manifest,
            extra={"stop_mode": "phase3"},
        )
        logger.info("Stopping after phase 3 as requested.")
        return

    # =====================================================================
    # PHASE 7: Hierarchical Graph Stitching + Dark-Zone Resolution
    # =====================================================================
    # (moved from old Phase 4 position — now runs before Re-ID)

    # =====================================================================
    # PHASE 6: Re-ID Fusion (with per-frame feature extraction + Hungarian)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Omni-Fusion Multi-Signal Re-ID")
    logger.info("=" * 70)
    t_phase = time.monotonic()

    from scipy.optimize import linear_sum_assignment as hungarian

    reid_assignments: Dict[int, Dict[int, int]] = {}
    reid_confidences: Dict[int, Dict[int, float]] = {}
    track_last_dancer: Dict[int, int] = {}
    track_last_bbox: Dict[int, np.ndarray] = {}
    frame_track_embeddings: Dict[int, Dict[int, np.ndarray]] = {}
    frame_track_centers: Dict[int, Dict[int, Tuple[float, float]]] = {}
    track_switches = 0
    track_assign_events = 0
    duplicate_dancer_frames = 0
    dancer_conflict_alarms = 0

    # --- Phase 0 observability: per-switch reason taxonomy ---
    switch_event_log: List[Dict] = []
    SWITCH_REASON_OVERLAP_PRESSURE = "overlap_pressure"
    SWITCH_REASON_EMBEDDING_CHALLENGER = "embedding_challenger_margin"
    SWITCH_REASON_LOCK_OVERRIDE = "lock_override"
    SWITCH_REASON_CONTAMINATION_CONFLICT = "contamination_conflict"
    SWITCH_REASON_COLD_START = "cold_start"
    SWITCH_REASON_UNKNOWN = "unknown"
    SWITCH_REASON_COOLDOWN_BLOCKED = "switch_cooldown_blocked"
    # Per-track confirmed identity state for hysteresis
    track_confirm_count: Dict[int, int] = {}
    track_locked: Dict[int, bool] = {}
    # Switch-lock cooldown: prevent cascading identity switches within K frames
    switch_cooldown_frames = max(1, int(os.environ.get("SWAY_REID_SWITCH_COOLDOWN_FRAMES", "30") or 30))
    track_last_switch_frame: Dict[int, int] = {}
    # GAP 5-A: visibility-aware dynamic Re-ID router
    reid_router = ReIDMLPRouter()
    writer5_reid = PhaseVideoWriter(OUTPUT_DIR / "phase6_reid_fusion.mp4", fps, width, height)
    phase5_heartbeat_every = max(1, int(os.environ.get("SWAY_PHASE5_HEARTBEAT_EVERY", "30") or 30))
    logger.info("Phase 6 heartbeat: every %d frames (set SWAY_PHASE5_HEARTBEAT_EVERY to override)", phase5_heartbeat_every)
    phase5_total_active_tracks = 0
    phase5_total_matches = 0
    face_embed_stride = max(1, int(os.environ.get("SWAY_FACE_EMBED_STRIDE", "2") or 2))
    part_cache_ttl = max(1, int(os.environ.get("SWAY_PART_CACHE_TTL", "3") or 3))
    part_cache_iou = float(os.environ.get("SWAY_PART_CACHE_IOU", "0.92") or 0.92)
    part_embed_cache: Dict[int, Dict[str, Any]] = {}
    part_cache_hits = 0
    part_cache_misses = 0

    # FIX 3.6: Temporal embedding smoothing — sliding window history per track
    TEMPORAL_WINDOW = int(os.environ.get("SWAY_REID_TEMPORAL_WINDOW", "5") or 5)
    track_emb_history: Dict[int, List[np.ndarray]] = {}

    for fidx, frame in enumerate(frames):
        tracks = all_track_results[fidx]
        active_tids = []
        track_embeddings: List[np.ndarray] = []
        track_colors: List[Dict] = []
        track_faces: List[Optional[np.ndarray]] = []
        track_bboxes: List[np.ndarray] = []
        track_contaminated: List[bool] = []
        track_part_results: List[Optional["PartEmbeddings"]] = []

        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emb = extract_bpbreid_embedding(crop)
            # FIX 3.6: Smooth embedding using temporal sliding window
            hist = track_emb_history.setdefault(tid, [])
            hist.append(emb)
            if len(hist) > TEMPORAL_WINDOW:
                hist.pop(0)
            if len(hist) > 1:
                smoothed = np.mean(np.stack(hist, axis=0), axis=0).astype(np.float32)
                smoothed /= (np.linalg.norm(smoothed) + 1e-8)
                emb = smoothed
            track_embeddings.append(emb)

            ch = None
            try:
                ch = color_hist_extractor.extract(crop, None, None)
            except Exception:
                pass
            track_colors.append(ch)

            face_emb = extract_face_embedding(crop) if (fidx % face_embed_stride == 0) else None
            track_faces.append(face_emb)

            # FIX 4.1: Extract part embeddings ONCE per track per frame (not per gallery comparison).
            cached_part = None
            prev_cached = part_embed_cache.get(tid)
            if prev_cached is not None:
                age = int(fidx - int(prev_cached.get("frame", -9999)))
                prev_bb = prev_cached.get("bbox")
                if (
                    age <= part_cache_ttl
                    and prev_bb is not None
                    and _bbox_iou(np.array([x1, y1, x2, y2], dtype=np.float32), np.array(prev_bb, dtype=np.float32)) >= part_cache_iou
                ):
                    cached_part = prev_cached.get("part")
                    part_cache_hits += 1
            if cached_part is None:
                try:
                    cached_part = bpbreid_part_extractor.extract(crop, keypoints=None, mask=None)
                except Exception:
                    cached_part = None
                part_cache_misses += 1
                if cached_part is not None:
                    part_embed_cache[tid] = {
                        "frame": int(fidx),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "part": cached_part,
                    }
            track_part_results.append(cached_part)

            active_tids.append(tid)
            track_bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            track_contaminated.append(tid in frame_contaminated_tids.get(fidx, set()))

        frame_assignments: Dict[int, int] = {}

        if active_tids and galleries:
            n_t = len(active_tids)
            n_g = len(galleries)
            score_matrix = np.zeros((n_t, n_g), dtype=np.float64)

            for ti in range(n_t):
                t_emb = track_embeddings[ti]
                t_ch = track_colors[ti]
                t_face = track_faces[ti]

                # GAP 5-A: compute visibility-aware dynamic weights
                signal_avail = {
                    "global": True,
                    "color": t_ch is not None,
                    "face": t_face is not None,
                    "spatial": True,
                    "part": True,
                    "motion": True,
                }
                signal_conf = {
                    "global": 0.8,
                    "color": 0.6 if t_ch else 0.0,
                    "face": 0.9 if t_face is not None else 0.0,
                    "spatial": 0.5,
                    "part": 0.6,
                    "motion": 0.4,
                }
                dyn_weights = reid_router.compute_weights(signal_avail, signal_conf)

                # Cold-start helper: nearest enrolled dancer by spatial prior for early frames.
                bb_ti = track_bboxes[ti]
                pos_ti = ((bb_ti[0] + bb_ti[2]) / 2 / width, (bb_ti[1] + bb_ti[3]) / 2 / height)
                nearest_spatial_did = -1
                nearest_spatial_dist = 999.0
                for g_sp in galleries:
                    d_sp = np.sqrt(
                        (pos_ti[0] - g_sp.spatial_position[0]) ** 2
                        + (pos_ti[1] - g_sp.spatial_position[1]) ** 2
                    )
                    if d_sp < nearest_spatial_dist:
                        nearest_spatial_dist = float(d_sp)
                        nearest_spatial_did = int(g_sp.dancer_id)

                for gi in range(n_g):
                    g = galleries[gi]
                    signal_scores: Dict[str, float] = {}

                    if g.global_embedding is not None:
                        sim = float(np.dot(t_emb, g.global_embedding))
                        signal_scores["global"] = max(0, sim)

                    if t_ch and g.color_histograms:
                        try:
                            cd = color_hist_extractor.compare(g.color_histograms, t_ch)
                            signal_scores["color"] = max(0, 1.0 - cd)
                        except Exception:
                            pass

                    if t_face is not None and g.face_embedding is not None:
                        sim = float(np.dot(t_face, g.face_embedding))
                        signal_scores["face"] = max(0, sim)

                    bb = track_bboxes[ti]
                    pos = ((bb[0]+bb[2])/2/width, (bb[1]+bb[3])/2/height)
                    sp_dist = np.sqrt((pos[0]-g.spatial_position[0])**2 + (pos[1]-g.spatial_position[1])**2)
                    signal_scores["spatial"] = max(0, 1.0 - sp_dist * 2.0)

                    # GAP 5-B: part-level embedding similarity (FIX 4.1: use cached per-track extraction)
                    if g.part_embeddings and len(g.part_embeddings) > 1:
                        part_result = track_part_results[ti]
                        if part_result is not None:
                            try:
                                part_sims = []
                                for pname, pvec in part_result.part_embs.items():
                                    if pname in g.part_embeddings:
                                        part_sims.append(float(np.dot(pvec, g.part_embeddings[pname])))
                                if part_sims:
                                    signal_scores["part"] = max(0, float(np.mean(part_sims)))
                            except Exception:
                                pass

                    # GAP 5-C: motion consistency
                    prev_bb = track_last_bbox.get(active_tids[ti])
                    if prev_bb is not None:
                        vel = (
                            (bb[0]+bb[2])/2 - (prev_bb[0]+prev_bb[2])/2,
                            (bb[1]+bb[3])/2 - (prev_bb[1]+prev_bb[3])/2,
                        )
                        vel_mag = np.sqrt(vel[0]**2 + vel[1]**2) / max(width, 1)
                        motion_score = max(0, 1.0 - vel_mag * 5.0)
                        signal_scores["motion"] = motion_score

                    # Strategy G fusion: BPBreID(0.4) + Face(0.3) + Color(0.3)
                    # When face unavailable: BPBreID(0.5) + Color(0.5)
                    _g_global = signal_scores.get("global", 0.0)
                    _g_color = signal_scores.get("color", 0.0)
                    _g_face = signal_scores.get("face", 0.0)
                    _has_face_sig = "face" in signal_scores
                    _has_color_sig = "color" in signal_scores
                    if _has_face_sig and _has_color_sig:
                        score = 0.40 * _g_global + 0.30 * _g_face + 0.30 * _g_color
                    elif _has_color_sig:
                        score = 0.50 * _g_global + 0.50 * _g_color
                    elif _has_face_sig:
                        score = 0.50 * _g_global + 0.50 * _g_face
                    else:
                        score = _g_global

                    prev_d = track_last_dancer.get(active_tids[ti], -1)
                    cold_start_window = int(os.environ.get("SWAY_REID_COLD_START_FRAMES", "2") or 2)
                    cold_start_bonus = float(os.environ.get("SWAY_REID_COLD_START_BONUS", "0.12") or 0.12)
                    if (
                        prev_d <= 0
                        and fidx < max(cold_start_window, 0)
                        and g.dancer_id == nearest_spatial_did
                    ):
                        # Encourage early-frame stability when temporal prior is unavailable.
                        score += cold_start_bonus * max(0.0, 1.0 - nearest_spatial_dist * 2.5)
                    if prev_d == g.dancer_id:
                        score += 0.20
                    score_matrix[ti, gi] = min(1.0, max(0.0, score))

            cost_matrix = 1.0 - score_matrix
            row_ind, col_ind = hungarian(cost_matrix)

            # --- Solution B: Hysteresis/ID-lock state machine ---
            lock_confirm_frames = int(os.environ.get("SWAY_REID_LOCK_CONFIRM_FRAMES", "5") or 5)
            lock_switch_penalty = float(os.environ.get("SWAY_REID_LOCK_SWITCH_PENALTY", "0.18") or 0.18)
            unlock_contradiction_margin = float(os.environ.get("SWAY_REID_UNLOCK_MARGIN", "0.25") or 0.25)

            used_dancers = set()
            for r, c in zip(row_ind, col_ind):
                tid = active_tids[r]
                did = galleries[c].dancer_id
                best_score = float(score_matrix[r, c])
                switch_margin = float(os.environ.get("SWAY_REID_SWITCH_MARGIN", "0.12") or 0.12)
                prev_d = track_last_dancer.get(tid, -1)
                is_locked = track_locked.get(tid, False)

                if prev_d > 0 and prev_d != did:
                    last_sw_frame = track_last_switch_frame.get(tid, -9999)
                    frames_since_switch = fidx - last_sw_frame
                    if frames_since_switch < switch_cooldown_frames:
                        did = prev_d
                        best_score = float(score_matrix[r, next((gi for gi, g in enumerate(galleries) if g.dancer_id == prev_d), c)])
                    else:
                        prev_col = next((gi for gi, g in enumerate(galleries) if g.dancer_id == prev_d), None)
                        if prev_col is not None:
                            prev_score = float(score_matrix[r, prev_col])

                            effective_margin = switch_margin
                            if is_locked:
                                effective_margin += lock_switch_penalty

                            if prev_d not in used_dancers and prev_score + effective_margin >= best_score and prev_score > 0.20:
                                did = prev_d
                                best_score = prev_score
                            elif is_locked and (best_score - prev_score) < unlock_contradiction_margin:
                                did = prev_d
                                best_score = prev_score

                if track_contaminated[r]:
                    if prev_d > 0:
                        prev_col = next((gi for gi, g in enumerate(galleries) if g.dancer_id == prev_d), None)
                        if prev_col is not None:
                            prev_score = float(score_matrix[r, prev_col])
                            contamination_margin = 0.15 + (0.10 if is_locked else 0.0)
                            if prev_score >= best_score - contamination_margin:
                                did = prev_d
                                best_score = prev_score
                if best_score > 0.25:
                    if did in used_dancers:
                        dancer_conflict_alarms += 1
                        continue
                    frame_assignments[tid] = did
                    used_dancers.add(did)

        reid_assignments[fidx] = frame_assignments
        # GAP 5-D: store per-assignment confidence
        frame_conf: Dict[int, float] = {}
        for tid_a, did_a in frame_assignments.items():
            ti_idx = active_tids.index(tid_a) if tid_a in active_tids else -1
            gi_idx = next((gi for gi, g in enumerate(galleries) if g.dancer_id == did_a), -1)
            if ti_idx >= 0 and gi_idx >= 0:
                frame_conf[tid_a] = float(score_matrix[ti_idx, gi_idx])
        reid_confidences[fidx] = frame_conf
        phase5_total_active_tracks += len(active_tids)
        phase5_total_matches += len(frame_assignments)

        frame_track_embeddings[fidx] = {}
        frame_track_centers[fidx] = {}
        for tid, bb in zip(active_tids, track_bboxes):
            cxy = (float((bb[0] + bb[2]) * 0.5), float((bb[1] + bb[3]) * 0.5))
            frame_track_centers[fidx][tid] = cxy
        for tid, emb in zip(active_tids, track_embeddings):
            frame_track_embeddings[fidx][tid] = emb
        seen_dancers = set()
        has_duplicate = False
        overlap_count_this_frame = sum(1 for c in track_contaminated if c)
        for tid, did in frame_assignments.items():
            prev = track_last_dancer.get(tid, -1)
            if prev > 0 and prev != did:
                track_switches += 1
                reason = SWITCH_REASON_UNKNOWN
                if tid in frame_contaminated_tids.get(fidx, set()):
                    reason = SWITCH_REASON_CONTAMINATION_CONFLICT
                elif overlap_count_this_frame >= 2:
                    reason = SWITCH_REASON_OVERLAP_PRESSURE
                elif prev <= 0:
                    reason = SWITCH_REASON_COLD_START
                else:
                    ti_idx_sw = active_tids.index(tid) if tid in active_tids else -1
                    if ti_idx_sw >= 0:
                        prev_col_sw = next((gi for gi, g in enumerate(galleries) if g.dancer_id == prev), None)
                        new_col_sw = next((gi for gi, g in enumerate(galleries) if g.dancer_id == did), None)
                        if prev_col_sw is not None and new_col_sw is not None:
                            margin = float(score_matrix[ti_idx_sw, new_col_sw]) - float(score_matrix[ti_idx_sw, prev_col_sw])
                            reason = SWITCH_REASON_EMBEDDING_CHALLENGER
                switch_event_log.append({
                    "frame": fidx,
                    "track_id": int(tid),
                    "from_dancer": int(prev),
                    "to_dancer": int(did),
                    "reason": reason,
                    "confidence": float(frame_conf.get(tid, 0.0)),
                    "overlap_tracks_in_frame": overlap_count_this_frame,
                })
                # Update hysteresis counters on switch
                track_confirm_count[tid] = 0
                track_locked[tid] = False
                track_last_switch_frame[tid] = fidx
            else:
                cc = track_confirm_count.get(tid, 0) + 1
                track_confirm_count[tid] = cc
                if cc >= 5:
                    track_locked[tid] = True
            track_assign_events += 1
            track_last_dancer[tid] = did
            if did in seen_dancers:
                has_duplicate = True
            seen_dancers.add(did)
        if has_duplicate:
            duplicate_dancer_frames += 1
        for tid, bb in zip(active_tids, track_bboxes):
            track_last_bbox[tid] = bb.copy()

        phase5_iter = fidx + 1
        if phase5_iter % phase5_heartbeat_every == 0 or phase5_iter == len(frames):
            elapsed = time.monotonic() - t_phase
            eta_s = ((elapsed / max(1, phase5_iter)) * (len(frames) - phase5_iter)) if phase5_iter < len(frames) else 0.0
            match_rate = (
                100.0 * float(phase5_total_matches) / float(max(1, phase5_total_active_tracks))
                if phase5_total_active_tracks > 0
                else 0.0
            )
            frame_mean_conf = float(np.mean(list(frame_conf.values()))) if frame_conf else 0.0
            logger.info(
                "  Phase 6 progress %d/%d (%.1f%%) | frame=%d active=%d matched=%d frame_mean_conf=%.3f | cumulative_match_rate=%.2f%% switches=%d dup_frames=%d conflicts=%d | elapsed=%.1fs eta=%.1fs",
                phase5_iter,
                len(frames),
                100.0 * phase5_iter / max(1, len(frames)),
                fidx,
                len(active_tids),
                len(frame_assignments),
                frame_mean_conf,
                match_rate,
                track_switches,
                duplicate_dancer_frames,
                dancer_conflict_alarms,
                elapsed,
                eta_s,
            )

        out = frame.copy()
        _draw_phase_banner(out, f"Phase 6: Re-ID Fusion | Frame {fidx} | {len(frame_assignments)} matched")

        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            dancer_id = frame_assignments.get(tid, -1)
            color = _color_for_id(dancer_id) if dancer_id >= 0 else (128, 128, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if dancer_id >= 0:
                ident_label = _identity_label(
                    int(dancer_id),
                    int(tid),
                    use_names=bool(use_name_labels),
                    identity_name_by_did=identity_name_by_did,
                )
                label = f"T{tid}->{ident_label}"
            else:
                label = f"T{tid}->?"
            _draw_text(out, label, (x1, y1 - 8), color)

        writer5_reid.write(out)

    writer5_reid.release()

    # GAP 5-D: serialize identity assignments with confidence
    id_assign_json = []
    for fidx_a in range(len(frames)):
        fa = reid_assignments.get(fidx_a, {})
        fc = reid_confidences.get(fidx_a, {})
        entries = []
        for tid_a, did_a in fa.items():
            entries.append({"track_id": int(tid_a), "dancer_id": int(did_a), "confidence": fc.get(tid_a, 0.0)})
        id_assign_json.append({"frame": fidx_a, "assignments": entries})
    with open(OUTPUT_DIR / "phase6_identity_assignments_reid.json", "w") as f:
        json.dump(id_assign_json, f)
    logger.info("Wrote phase6_identity_assignments_reid.json")
    _write_alias_copy(OUTPUT_DIR / "phase6_identity_assignments_reid.json", "identity_assignments_phase5.json")
    phase5_reid_time = (time.monotonic() - t_phase) * 1000
    logger.info("Phase 6 complete: %.1fms", phase5_reid_time)
    logger.info(
        "Phase 6 runtime knobs: face_stride=%d part_cache_hits=%d part_cache_misses=%d",
        face_embed_stride,
        part_cache_hits,
        part_cache_misses,
    )
    if track_assign_events > 0:
        logger.info(
            "Phase 6 diagnostics: track_switches=%d over %d assignments (%.2f%%), duplicate_dancer_frames=%d, dancer_conflict_alarms=%d",
            track_switches,
            track_assign_events,
            100.0 * float(track_switches) / float(track_assign_events),
            duplicate_dancer_frames,
            dancer_conflict_alarms,
        )

    # =====================================================================
    # PHASE 7: Dark-Zone Resolution (coalescence -> NOOUGAT graph stitch)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: Global Dark-Zone Resolution")
    logger.info("=" * 70)
    t_phase = time.monotonic()

    # More sensitive thresholds so dark-zone events activate in dense dance clips.
    coalescence_detector = CoalescenceDetector(iou_thresh=0.25, consecutive_frames=2)
    graph_stitcher = NOOUGATGraphStitcher()
    event_entry_data: Dict[Tuple[int, ...], Tuple[int, Dict[int, TrackletNode]]] = {}
    stitched_alias: Dict[int, int] = {}
    stitch_pairs_applied = 0

    for fidx in range(len(frames)):
        tracks = all_track_results[fidx]
        bboxes = {}
        for t in tracks:
            tid = int(t[4])
            bboxes[tid] = np.array([t[0], t[1], t[2], t[3]], dtype=np.float32)
        new_events = coalescence_detector.check(bboxes, fidx)
        for ev in new_events:
            key = tuple(sorted(int(tid) for tid in ev.track_ids))
            entry_nodes: Dict[int, TrackletNode] = {}
            prev_centers = frame_track_centers.get(max(0, fidx - 1), {})
            curr_centers = frame_track_centers.get(fidx, {})
            curr_embs = frame_track_embeddings.get(fidx, {})
            for tid in ev.track_ids:
                if tid not in curr_centers or tid not in curr_embs:
                    continue
                cxy = curr_centers[tid]
                pxy = prev_centers.get(tid, cxy)
                vel = (cxy[0] - pxy[0], cxy[1] - pxy[1])
                entry_nodes[int(tid)] = TrackletNode(
                    track_id=int(tid),
                    start_frame=int(fidx),
                    end_frame=int(fidx),
                    embedding=curr_embs[tid].astype(np.float32),
                    spatial_trajectory=[cxy],
                    velocity=(float(vel[0]), float(vel[1])),
                    is_entry=True,
                )
            if entry_nodes:
                event_entry_data[key] = (int(fidx), entry_nodes)

        exit_events = coalescence_detector.check_exits(bboxes, fidx)
        for ev in exit_events:
            key = tuple(sorted(int(tid) for tid in ev.track_ids))
            if key not in event_entry_data:
                continue
            entry_frame, entry_nodes = event_entry_data[key]
            exit_nodes: List[TrackletNode] = []
            curr_centers = frame_track_centers.get(fidx, {})
            curr_embs = frame_track_embeddings.get(fidx, {})
            for tid in ev.track_ids:
                if tid not in curr_centers or tid not in curr_embs:
                    continue
                cxy = curr_centers[tid]
                exit_nodes.append(
                    TrackletNode(
                        track_id=int(tid),
                        start_frame=int(fidx),
                        end_frame=int(fidx),
                        embedding=curr_embs[tid].astype(np.float32),
                        spatial_trajectory=[cxy],
                        velocity=(0.0, 0.0),
                        is_entry=False,
                    )
                )
            if not exit_nodes:
                continue
            zone = graph_stitcher.create_dark_zone(
                entry_frame=entry_frame,
                exit_frame=int(fidx),
                entry_tracklets=list(entry_nodes.values()),
                exit_tracklets=exit_nodes,
            )
            assigns = graph_stitcher.resolve_dark_zone(zone)
            # GAP 4-C: gallery hard exclusion — reject stitches between tracks
            # that are confidently assigned to different dancers
            filtered_assigns = []
            for entry_tid, exit_tid in assigns:
                entry_did = track_last_dancer.get(entry_tid, -1)
                exit_did = track_last_dancer.get(exit_tid, -1)
                if entry_did > 0 and exit_did > 0 and entry_did != exit_did:
                    entry_conf = reid_confidences.get(
                        max(frame_track_centers.keys()) if frame_track_centers else 0, {}
                    ).get(entry_tid, 0.0)
                    exit_conf = reid_confidences.get(fidx, {}).get(exit_tid, 0.0)
                    if entry_conf > 0.5 and exit_conf > 0.5:
                        logger.debug(
                            "Hard exclusion: rejecting stitch T%d(D%d)->T%d(D%d)",
                            entry_tid, entry_did, exit_tid, exit_did,
                        )
                        continue
                filtered_assigns.append((entry_tid, exit_tid))
            stitch_pairs_applied += len(filtered_assigns)
            for entry_tid, exit_tid in filtered_assigns:
                stitched_alias[int(exit_tid)] = int(entry_tid)

    # FIX 3.4: Match backward tracklets to forward tracklets by bbox IoU
    # instead of using orphan tid+100000 offsets that create isolated nodes.
    bwd_to_fwd_map: Dict[Tuple[int, int], int] = {}  # (fidx, bwd_tid) -> fwd_tid
    for fidx_b in range(len(frames)):
        bwd_tr = all_backward_tracks[fidx_b]
        fwd_tr = all_track_results[fidx_b]
        if len(bwd_tr) == 0 or len(fwd_tr) == 0:
            continue
        used_fwd = set()
        for bt in bwd_tr:
            bwd_bb = np.array([bt[0], bt[1], bt[2], bt[3]], dtype=np.float32)
            bwd_tid = int(bt[4])
            best_fwd_tid = -1
            best_iou = 0.3
            for ft in fwd_tr:
                fwd_bb = np.array([ft[0], ft[1], ft[2], ft[3]], dtype=np.float32)
                fwd_tid = int(ft[4])
                if fwd_tid in used_fwd:
                    continue
                iou = _bbox_iou(bwd_bb, fwd_bb)
                if iou > best_iou:
                    best_iou = iou
                    best_fwd_tid = fwd_tid
            if best_fwd_tid >= 0:
                bwd_to_fwd_map[(fidx_b, bwd_tid)] = best_fwd_tid
                used_fwd.add(best_fwd_tid)

    bwd_tracklet_spans: Dict[int, List[int]] = {}
    for (fidx_b, bwd_tid), fwd_tid in bwd_to_fwd_map.items():
        bwd_tracklet_spans.setdefault(fwd_tid, []).append(fidx_b)

    for fwd_tid, frame_list in bwd_tracklet_spans.items():
        fwd_frames = [fi for fi in range(len(frames))
                      if any(int(t[4]) == fwd_tid for t in all_track_results[fi])]
        if not fwd_frames:
            continue
        bwd_only = sorted(set(frame_list) - set(fwd_frames))
        if not bwd_only:
            continue
        first_bwd = bwd_only[0]
        last_bwd = bwd_only[-1]
        emb = frame_track_embeddings.get(min(fwd_frames, key=lambda f: abs(f - first_bwd)), {}).get(fwd_tid)
        if emb is None:
            continue
        entry_node = TrackletNode(
            track_id=fwd_tid,
            start_frame=first_bwd,
            end_frame=last_bwd,
            embedding=emb.astype(np.float32),
            spatial_trajectory=[frame_track_centers.get(first_bwd, {}).get(fwd_tid, (0.0, 0.0))],
            velocity=(0.0, 0.0),
            is_entry=True,
        )
        exit_node = TrackletNode(
            track_id=fwd_tid,
            start_frame=last_bwd,
            end_frame=last_bwd,
            embedding=emb.astype(np.float32),
            spatial_trajectory=[frame_track_centers.get(last_bwd, {}).get(fwd_tid, (0.0, 0.0))],
            velocity=(0.0, 0.0),
            is_entry=False,
        )
        graph_stitcher.create_dark_zone(
            entry_frame=first_bwd, exit_frame=last_bwd,
            entry_tracklets=[entry_node], exit_tracklets=[exit_node],
        )
    logger.info("Backward tracklet integration: %d bwd-fwd matches, %d gap zones created",
                len(bwd_to_fwd_map), len(bwd_tracklet_spans))

    stitch_result = graph_stitcher.stitch_all()
    stitch_pairs_applied += len(stitch_result.stitches)

    # GAP 4-A: hierarchical merge — group resolved zones and merge clusters
    if hasattr(graph_stitcher, 'hierarchical_merge'):
        try:
            hier_result = graph_stitcher.hierarchical_merge()
            if hier_result:
                for src_tid, dst_tid in hier_result:
                    stitched_alias[int(dst_tid)] = int(src_tid)
                    stitch_pairs_applied += 1
                logger.info("Hierarchical merge: %d additional links", len(hier_result))
        except Exception as exc:
            logger.debug("Hierarchical merge not available: %s", exc)

    logger.info("Dark zones resolved: %d, stitches applied: %d",
                stitch_result.dark_zones_resolved, stitch_pairs_applied)
    logger.info("Dark-zone alias links applied: %d", len(stitched_alias))

    # Build phase4 assignments after dark-zone aliasing.
    # Windowed mode keeps ReID-driven assignment; formation mode swaps to
    # timeline-constrained assignment (hard-capped performer identities).
    reid_assignments_phase4: Dict[int, Dict[int, int]] = {}
    if mode_info.get("final_mode") == "formation" and formation_timeline is not None:
        start_index_override = os.environ.get("SWAY_FORMATION_START_INDEX", "").strip()
        idx_override = int(start_index_override) if start_index_override else None
        spatial_align = estimate_start_offset_spatial(
            formation_timeline,
            all_track_results,
            width=width,
            height=height,
            fps=float(fps),
            start_index_override=idx_override,
        )
        fixed_offset_env = os.environ.get("SWAY_FORMATION_START_OFFSET_SEC", "").strip()
        if fixed_offset_env:
            try:
                spatial_align = StartAlignment(
                    start_offset_sec=float(fixed_offset_env),
                    start_formation_index=int(spatial_align.start_formation_index),
                    spatial_confidence=1.0,
                    audio_confidence=0.0,
                    selected_flip_x=bool(spatial_align.selected_flip_x),
                    reason="start_offset_override",
                )
            except ValueError:
                pass
        audio_offset, audio_conf, audio_reason = estimate_audio_offset(VIDEO_PATH, formation_timeline)
        fused_align = fuse_start_alignment(spatial_align, audio_offset, audio_conf)
        reid_assignments_phase4, identity_name_by_did, formation_diag = build_formation_assignments(
            formation_timeline,
            all_track_results,
            width=width,
            height=height,
            fps=float(fps),
            start_offset_sec=float(fused_align.start_offset_sec),
            flip_x=bool(fused_align.selected_flip_x),
            max_match_dist=float(os.environ.get("SWAY_FORMATION_MAX_MATCH_DIST", "0.22") or 0.22),
        )
        use_name_labels = True
        formation_alignment_info = {
            "start_formation_index": int(fused_align.start_formation_index),
            "start_offset_sec": float(fused_align.start_offset_sec),
            "spatial_confidence": float(fused_align.spatial_confidence),
            "audio_confidence": float(fused_align.audio_confidence),
            "audio_reason": audio_reason,
            "flip_x": bool(fused_align.selected_flip_x),
            "assignment_stats": formation_diag,
        }
        logger.info(
            "Formation alignment: idx=%d offset=%.3fs spatial_conf=%.3f audio_conf=%.3f flip_x=%s",
            int(fused_align.start_formation_index),
            float(fused_align.start_offset_sec),
            float(fused_align.spatial_confidence),
            float(fused_align.audio_confidence),
            bool(fused_align.selected_flip_x),
        )
    else:
        for fidx in range(len(frames)):
            out_map: Dict[int, int] = {}
            fmap = reid_assignments.get(fidx, {})
            frame_conf = reid_confidences.get(fidx, {})
            best_by_dancer: Dict[int, Tuple[int, float]] = {}
            for tid, did in fmap.items():
                raw_tid = int(tid)
                conf = float(frame_conf.get(tid, 0.0))
                prev = best_by_dancer.get(did)
                if prev is None or conf > prev[1]:
                    best_by_dancer[did] = (raw_tid, conf)

            for did, (raw_tid, _conf) in best_by_dancer.items():
                out_map[int(raw_tid)] = int(did)
            reid_assignments_phase4[fidx] = out_map

    writer4 = PhaseVideoWriter(OUTPUT_DIR / "phase7_darkzone_resolution.mp4", fps, width, height)
    for fidx, frame in enumerate(frames):
        tracks = all_track_results[fidx]
        out = frame.copy()
        _draw_phase_banner(out, f"Phase 7: Dark-Zone | Frame {fidx} | {stitch_result.dark_zones_resolved} zones resolved")

        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            dancer_id = reid_assignments_phase4.get(fidx, {}).get(tid, -1)
            color = _color_for_id(dancer_id) if dancer_id >= 0 else (128, 128, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = _identity_label(
                int(dancer_id),
                int(tid),
                use_names=bool(use_name_labels),
                identity_name_by_did=identity_name_by_did,
            )
            _draw_text(out, label, (x1, y1 - 8), color)

        writer4.write(out)

    writer4.release()
    phase4_time = (time.monotonic() - t_phase) * 1000
    logger.info("Phase 7 complete: %.1fms", phase4_time)
    stop_after_phase4 = str(os.environ.get("SWAY_STOP_AFTER_PHASE4", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if stop_after_phase4:
        phase4_identity_json = []
        phase4_total_tracks = 0
        phase4_total_assigned = 0
        phase4_unknown = 0
        phase4_switches = 0
        phase4_overcap = 0
        phase4_track_last: Dict[int, int] = {}
        max_allowed_did = len(identity_name_by_did) if identity_name_by_did else max(1, len(galleries))
        for fidx_s in range(len(frames)):
            fr_map = reid_assignments_phase4.get(fidx_s, {})
            phase4_identity_json.append(
                {"frame": int(fidx_s), "assignments": {str(int(k)): int(v) for k, v in fr_map.items()}}
            )
            phase4_total_assigned += int(len(fr_map))
            for tid_s, did_s in fr_map.items():
                tid_i = int(tid_s)
                did_i = int(did_s)
                prev_did = phase4_track_last.get(tid_i, -1)
                if prev_did > 0 and prev_did != did_i:
                    phase4_switches += 1
                phase4_track_last[tid_i] = did_i
                if did_i > max_allowed_did:
                    phase4_overcap += 1
            tracks_s = all_track_results[fidx_s]
            phase4_total_tracks += int(len(tracks_s))
            phase4_unknown += max(0, int(len(tracks_s) - len(fr_map)))

        with open(OUTPUT_DIR / "phase4_identity_assignments.json", "w") as f:
            json.dump(phase4_identity_json, f, indent=2)
        phase4_report = {
            "video": str(VIDEO_PATH),
            "mode": mode_info,
            "formation_alignment": formation_alignment_info,
            "phase35_status": phase35_status,
            "phase4_metrics": {
                "frames": int(len(frames)),
                "total_tracks": int(phase4_total_tracks),
                "total_assigned": int(phase4_total_assigned),
                "unknown_tracks": int(phase4_unknown),
                "unknown_rate": float(phase4_unknown) / float(max(1, phase4_total_tracks)),
                "track_switches": int(phase4_switches),
                "track_switch_rate": float(phase4_switches) / float(max(1, phase4_total_assigned)),
                "overcap_assignments": int(phase4_overcap),
                "max_allowed_did": int(max_allowed_did),
            },
            "notes": "Stopped after phase 4 due to SWAY_STOP_AFTER_PHASE4=1",
        }
        with open(OUTPUT_DIR / "formation_diagnostics.json", "w") as f:
            json.dump(phase4_report, f, indent=2)
        _write_phase3_data_json(OUTPUT_DIR / "data.json", all_track_results)
        with open(OUTPUT_DIR / "summary_phase4_only.json", "w") as f:
            json.dump(phase4_report, f, indent=2)
        _write_baseline_validation_report(
            output_dir=OUTPUT_DIR,
            video_path=VIDEO_PATH,
            summary_obj=phase4_report,
            eval_metrics_obj=None,
            phase35_status=phase35_status,
        )
        phase4_manifest = _build_run_manifest(
            repo_root=Path(__file__).resolve().parent.parent,
            device=DEVICE,
            video_path=VIDEO_PATH,
            output_dir=OUTPUT_DIR,
            tracker_backend_requested=tracker_backend_requested,
            tracker_backend_effective=tracker_backend_effective,
            precision_requested=precision_requested,
            precision_effective=precision_effective_name,
            phase_times_ms={
                "phase1_detection": phase1_time,
                "phase2_masking": phase2_mask_time,
                "phase3_dual_pose": phase3_dualpose_time,
                "phase3_5_disambiguation": phase35_disambiguation_time,
                "phase4_tracking_fwd": phase3_track_time,
                "phase4_tracking_bwd": bwd_time,
                "phase6_reid": phase5_reid_time,
                "phase7_darkzone": phase4_time,
            },
            tracker_ab_report=tracker_ab_report,
            summary=phase4_report,
            reid_feature_mode=reid_feature_mode,
            reid_feature_mode_reason=reid_feature_mode_reason,
        )
        with open(OUTPUT_DIR / "run_manifest.json", "w") as f:
            json.dump(phase4_manifest, f, indent=2)
        logger.info("Wrote run_manifest.json")
        _write_artifact_indexes(
            output_dir=OUTPUT_DIR,
            summary=phase4_report,
            run_manifest=phase4_manifest,
            extra={"stop_mode": "phase4"},
        )
        logger.info("Wrote phase4_identity_assignments.json")
        logger.info("Wrote formation_diagnostics.json")
        logger.info("Wrote summary_phase4_only.json")
        logger.info("Stopping after phase 4 due to SWAY_STOP_AFTER_PHASE4=1")
        return

    # =====================================================================
    # LEGACY POSE/3D BLOCK (kept for compatibility artifacts)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("LEGACY: Pose Estimation + Physics-Aware 3D Lifting")
    logger.info("=" * 70)
    t_phase = time.monotonic()

    logger.info("Running legacy VitPose/3D block on tracked persons...")
    kp2d_per_track: Dict[int, List[np.ndarray]] = {}
    conf_per_track_raw: Dict[int, List[np.ndarray]] = {}
    pose_cache: Dict[int, Dict[int, np.ndarray]] = {}
    pose_samples = 0
    pose_bad_geom = 0
    pose_low_conf = 0
    legacy_pose_conf_raw_means: List[float] = []
    legacy_pose_conf_post_gate_means: List[float] = []
    legacy_pose_total_joints = 0
    legacy_pose_zeroed_joints = 0

    for fidx, frame in enumerate(frames):
        tracks = all_track_results[fidx]
        if fidx % 60 == 0:
            logger.info("  VitPose frame %d/%d (%d tracks)", fidx, len(frames), len(tracks))

        boxes: List[Tuple[float, float, float, float]] = []
        track_ids: List[int] = []
        dancer_ids: List[int] = []
        det_indices_for_boxes: List[Optional[int]] = []
        segmentation_masks_for_boxes: List[Optional[np.ndarray]] = []
        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 12 or y2 - y1 < 24:
                continue
            did = reid_assignments_phase4.get(fidx, {}).get(tid, -1)
            if did <= 0:
                continue
            box_xyxy = (float(x1), float(y1), float(x2), float(y2))
            det_idx_for_mask = None
            if fidx in all_masks:
                for di_m, det_m in enumerate(all_detections[fidx]):
                    if _bbox_iou(det_m.bbox, np.array([x1, y1, x2, y2], dtype=np.float32)) > 0.5:
                        det_idx_for_mask = di_m
                        break
            person_mask = all_masks.get(fidx, {}).get(det_idx_for_mask) if det_idx_for_mask is not None else None
            boxes.append(box_xyxy)
            track_ids.append(int(tid))
            dancer_ids.append(int(did))
            det_indices_for_boxes.append(det_idx_for_mask)
            segmentation_masks_for_boxes.append(person_mask)

        if boxes:
            try:
                pose_results = pose_estimator.estimate_poses(
                    frame,
                    boxes,
                    dancer_ids,
                    segmentation_masks=segmentation_masks_for_boxes,
                )
            except Exception as exc:
                logger.warning("PoseEstimator failed on frame %d: %s", fidx, exc)
                pose_results = {}
            det_idx_by_did = {int(did_v): det_i for did_v, det_i in zip(dancer_ids, det_indices_for_boxes)}
            for did, bx in zip(dancer_ids, boxes):
                pd = pose_results.get(int(did))
                if not pd:
                    continue
                kp = np.asarray(pd["keypoints"], dtype=np.float32)
                if kp is None or kp.shape[0] < 17:
                    continue
                kp = kp.copy()
                raw_scores = kp[:, 2].astype(np.float32).copy()
                legacy_pose_conf_raw_means.append(float(np.mean(raw_scores)))
                x1, y1, x2, y2 = bx
                # GAP 6-A: geometric validity vs SAM / bbox (do NOT overwrite ViTPose scores in exports)
                det_idx_for_mask = det_idx_by_did.get(int(did))
                person_mask = all_masks.get(fidx, {}).get(det_idx_for_mask) if det_idx_for_mask is not None else None

                in_bbox = (
                    (kp[:, 0] >= x1) & (kp[:, 0] <= x2) &
                    (kp[:, 1] >= y1) & (kp[:, 1] <= y2)
                )
                if person_mask is not None:
                    in_mask = np.zeros(kp.shape[0], dtype=bool)
                    for j_idx in range(kp.shape[0]):
                        jx, jy = int(round(kp[j_idx, 0])), int(round(kp[j_idx, 1]))
                        if 0 <= jy < person_mask.shape[0] and 0 <= jx < person_mask.shape[1]:
                            in_mask[j_idx] = bool(person_mask[jy, jx])
                    geom_valid = in_mask
                else:
                    geom_valid = in_bbox
                invalid = ~geom_valid
                legacy_pose_total_joints += int(kp.shape[0])
                legacy_pose_zeroed_joints += int(np.count_nonzero(invalid))
                conf_for_lift = raw_scores * (~invalid).astype(np.float32)
                legacy_pose_conf_post_gate_means.append(float(np.mean(conf_for_lift)))
                # Preserve ViTPose per-keypoint confidence for pose2d_phase6.json / coaching analytics.
                kp[:, 2] = raw_scores
                visible_model = raw_scores > 0.3
                vis_count = int(np.count_nonzero(visible_model))
                in_count = int(np.count_nonzero(geom_valid & visible_model))
                pose_samples += 1
                if vis_count > 0:
                    if in_count / max(1, vis_count) < 0.6:
                        pose_bad_geom += 1
                    if float(np.mean(raw_scores[visible_model])) < 0.35:
                        pose_low_conf += 1
                pose_cache.setdefault(fidx, {})[int(did)] = kp
                kp2d_per_track.setdefault(int(did), []).append(kp[:, :2])
                conf_per_track_raw.setdefault(int(did), []).append(conf_for_lift)

    # --- Temporal pose keypoint smoothing (fix jerk regression) ---
    _tpr_enabled = os.environ.get("SWAY_TEMPORAL_POSE_REFINE", "1").strip().lower() not in ("0", "false", "no", "off")
    _tpr_radius = max(0, min(8, int(os.environ.get("SWAY_TEMPORAL_POSE_RADIUS", "2") or 2)))
    if _tpr_enabled and _tpr_radius > 0 and pose_cache:
        logger.info("Temporal keypoint smoothing: radius=%d", _tpr_radius)
        for did_s in set(did_k for fc_s in pose_cache.values() for did_k in fc_s):
            frames_with_did = sorted(f for f in pose_cache if did_s in pose_cache[f])
            if len(frames_with_did) < 2:
                continue
            kps_seq = [pose_cache[f][did_s] for f in frames_with_did]
            for i_s, f_s in enumerate(frames_with_did):
                lo = max(0, i_s - _tpr_radius)
                hi = min(len(frames_with_did), i_s + _tpr_radius + 1)
                kp_orig = kps_seq[i_s]
                weight_sum = np.zeros(kp_orig.shape[0], dtype=np.float32)
                xy_acc = np.zeros((kp_orig.shape[0], 2), dtype=np.float32)
                for j_s in range(lo, hi):
                    kp_n = kps_seq[j_s]
                    dist_w = 1.0 / (1.0 + abs(j_s - i_s))
                    w = kp_n[:, 2] * dist_w
                    weight_sum += w
                    xy_acc[:, 0] += w * kp_n[:, 0]
                    xy_acc[:, 1] += w * kp_n[:, 1]
                valid = weight_sum > 1e-6
                kp_orig[valid, 0] = xy_acc[valid, 0] / weight_sum[valid]
                kp_orig[valid, 1] = xy_acc[valid, 1] / weight_sum[valid]
                pose_cache[f_s][did_s] = kp_orig
        logger.info("Temporal keypoint smoothing applied")

    legacy_pose_conf_raw_mean = float(np.mean(legacy_pose_conf_raw_means)) if legacy_pose_conf_raw_means else 0.0
    legacy_pose_conf_post_gate_mean = (
        float(np.mean(legacy_pose_conf_post_gate_means)) if legacy_pose_conf_post_gate_means else 0.0
    )
    legacy_pose_zeroed_joint_fraction = (
        float(legacy_pose_zeroed_joints) / float(max(1, legacy_pose_total_joints))
    )
    logger.info(
        "Legacy pose confidence diagnostics: raw_mean=%.4f post_gate_mean=%.4f drop=%.4f zeroed_joint_frac=%.3f",
        legacy_pose_conf_raw_mean,
        legacy_pose_conf_post_gate_mean,
        max(0.0, legacy_pose_conf_raw_mean - legacy_pose_conf_post_gate_mean),
        legacy_pose_zeroed_joint_fraction,
    )

    # --- Solution D: per-joint confidence_state enum ---
    # Thresholds for confidence state classification
    CONF_HIGH_THRESH = float(os.environ.get("SWAY_POSE_CONF_HIGH", "0.5") or 0.5)
    CONF_MED_THRESH = float(os.environ.get("SWAY_POSE_CONF_MED", "0.3") or 0.3)
    CONF_LOW_THRESH = float(os.environ.get("SWAY_POSE_CONF_LOW", "0.1") or 0.1)

    def _joint_confidence_state(conf_val: float) -> str:
        if conf_val >= CONF_HIGH_THRESH:
            return "high"
        if conf_val >= CONF_MED_THRESH:
            return "medium"
        if conf_val >= CONF_LOW_THRESH:
            return "low"
        return "missing"

    pose2d_json = []
    for fidx_p in range(len(frames)):
        fc = pose_cache.get(fidx_p, {})
        frame_data = []
        for did_p, kp_p in fc.items():
            n_joints = min(17, kp_p.shape[0])
            confs = [float(kp_p[j, 2]) for j in range(n_joints)]
            visibility_flags = [bool(confs[j] > 0.3) for j in range(n_joints)]
            confidence_states = [_joint_confidence_state(confs[j]) for j in range(n_joints)]
            # Geometric validity vs tracker box + SAM instance (for separating "model sure" vs "inside silhouette")
            bx_geom: Optional[Tuple[float, float, float, float]] = None
            for t_row in all_track_results[fidx_p]:
                x1g, y1g, x2g, y2g, tgid = (
                    float(t_row[0]),
                    float(t_row[1]),
                    float(t_row[2]),
                    float(t_row[3]),
                    int(t_row[4]),
                )
                did_g = int(reid_assignments_phase4.get(fidx_p, {}).get(tgid, -1))
                if did_g == int(did_p):
                    bx_geom = (x1g, y1g, x2g, y2g)
                    break
            geom_in_roi: List[bool] = []
            if bx_geom is not None:
                gx1, gy1, gx2, gy2 = bx_geom
                det_idx_g = None
                if fidx_p in all_masks:
                    for di_g, det_g in enumerate(all_detections[fidx_p]):
                        if _bbox_iou(det_g.bbox, np.array([gx1, gy1, gx2, gy2], dtype=np.float32)) > 0.5:
                            det_idx_g = di_g
                            break
                pm = all_masks.get(fidx_p, {}).get(det_idx_g) if det_idx_g is not None else None
                for j in range(n_joints):
                    jx, jy = int(round(kp_p[j, 0])), int(round(kp_p[j, 1]))
                    in_b = bool(gx1 <= kp_p[j, 0] <= gx2 and gy1 <= kp_p[j, 1] <= gy2)
                    if pm is not None and 0 <= jy < pm.shape[0] and 0 <= jx < pm.shape[1]:
                        geom_in_roi.append(bool(pm[jy, jx]))
                    else:
                        geom_in_roi.append(in_b)
            else:
                geom_in_roi = [True] * n_joints
            frame_data.append({
                "dancer_id": int(did_p),
                "keypoints": [[float(kp_p[j, 0]), float(kp_p[j, 1]), confs[j]] for j in range(n_joints)],
                "visibility": visibility_flags,
                "confidence_state": confidence_states,
                "geom_in_instance_or_bbox": geom_in_roi,
            })
        pose2d_json.append({"frame": fidx_p, "poses": frame_data})
    with open(OUTPUT_DIR / "pose2d_phase6.json", "w") as f:
        json.dump(pose2d_json, f)
    logger.info("Wrote pose2d_phase6.json")

    kp2d_arrays = {}
    conf_arrays = {}
    for tid, positions in kp2d_per_track.items():
        if len(positions) >= 5:
            kp2d_arrays[tid] = np.stack(positions, axis=0)
            confs = conf_per_track_raw.get(tid, [])
            if confs:
                conf_arrays[tid] = np.stack(confs, axis=0)
            else:
                conf_arrays[tid] = np.full((len(positions), 17), 0.8, dtype=np.float32)

    # Forward 3D lifting
    scene = lift_poses_v23(kp2d_arrays, conf_arrays, width, height)
    logger.info("3D scene (forward): %d dancers lifted, %d total frames", len(scene.poses), scene.frame_count)

    # GAP 7-A: bidirectional 3D lifting — run backward and blend
    kp2d_reversed = {}
    conf_reversed = {}
    for tid, arr in kp2d_arrays.items():
        kp2d_reversed[tid] = arr[::-1].copy()
        conf_reversed[tid] = conf_arrays[tid][::-1].copy() if tid in conf_arrays else np.full_like(arr[:, :, 0], 0.8)
    scene_bwd = lift_poses_v23(kp2d_reversed, conf_reversed, width, height)
    for tid in scene.poses:
        if tid in scene_bwd.poses:
            fwd_kp = scene.poses[tid].keypoints_3d
            bwd_kp = scene_bwd.poses[tid].keypoints_3d[::-1]
            if fwd_kp.shape == bwd_kp.shape:
                fwd_conf = conf_arrays.get(tid, np.full((fwd_kp.shape[0], 17), 0.8))
                bwd_conf = conf_reversed.get(tid, np.full((bwd_kp.shape[0], 17), 0.8))[::-1]
                if fwd_conf.shape[0] == fwd_kp.shape[0] and bwd_conf.shape[0] == bwd_kp.shape[0]:
                    fwd_w = np.mean(fwd_conf, axis=1, keepdims=True)[..., np.newaxis]
                    bwd_w = np.mean(bwd_conf, axis=1, keepdims=True)[..., np.newaxis]
                    total_w = fwd_w + bwd_w + 1e-8
                    blended = (fwd_kp * fwd_w + bwd_kp * bwd_w) / total_w
                    scene.poses[tid].keypoints_3d = blended
    logger.info("Bidirectional 3D blend complete for %d dancers", len(scene.poses))

    for tid, pose in scene.poses.items():
        logger.info("  Dancer %d: %d frames, physics=%s, perspective=%s",
                     tid, pose.keypoints_3d.shape[0], pose.physics_refined, pose.perspective_corrected)
    if pose_samples > 0:
        logger.info(
            "Pose diagnostics: samples=%d, bad_geometry=%d (%.2f%%), low_conf=%d (%.2f%%)",
            pose_samples,
            pose_bad_geom,
            100.0 * float(pose_bad_geom) / float(pose_samples),
            pose_low_conf,
            100.0 * float(pose_low_conf) / float(pose_samples),
        )

    writer5 = PhaseVideoWriter(OUTPUT_DIR / "phase7_legacy_pose3d_compat.mp4", fps, width, height)

    SKELETON_BONES = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4),
    ]

    for fidx, frame in enumerate(frames):
        tracks = all_track_results[fidx]
        out = frame.copy()
        frame_poses = pose_cache.get(fidx, {})
        _draw_phase_banner(out, f"Legacy Pose/3D Compat | Frame {fidx} | {len(frame_poses)} skeletons")

        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            dancer_id = reid_assignments_phase4.get(fidx, {}).get(tid, -1)
            color = _color_for_id(dancer_id) if dancer_id >= 0 else (200, 200, 200)

            kp = frame_poses.get(dancer_id)
            if kp is not None:
                for j1, j2 in SKELETON_BONES:
                    if j1 < len(kp) and j2 < len(kp) and kp[j1, 2] > 0.3 and kp[j2, 2] > 0.3:
                        pt1 = (int(kp[j1, 0]), int(kp[j1, 1]))
                        pt2 = (int(kp[j2, 0]), int(kp[j2, 1]))
                        cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)

                for k in range(min(17, len(kp))):
                    if kp[k, 2] > 0.3:
                        pt = (int(kp[k, 0]), int(kp[k, 1]))
                        cv2.circle(out, pt, 4, color, -1)

            label = _identity_label(
                int(dancer_id),
                int(tid),
                use_names=bool(use_name_labels),
                identity_name_by_did=identity_name_by_did,
            )
            _draw_text(out, label, (x1, y1 - 8), color, 0.5)

        writer5.write(out)

    writer5.release()
    phase67_time = (time.monotonic() - t_phase) * 1000
    logger.info("Legacy pose/3D block complete: %.1fms", phase67_time)

    # =====================================================================
    # PHASE 8: Global Joint Optimization (Final Consistency Pass)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8: Global Joint Optimization")
    logger.info("=" * 70)
    t_phase8 = time.monotonic()

    phase8_corrections = 0
    phase8_eligible_same_tid = 0
    phase8_eligible_xtrack = 0
    phase8_applied_xtrack = 0
    phase8_conf_thresh = float(os.environ.get("SWAY_PHASE8_CONF_THRESH", "0.90") or 0.90)
    phase8_kin_conf_thresh = float(os.environ.get("SWAY_PHASE8_KIN_CONF_THRESH", "0.70") or 0.70)
    phase8_xtrack_conf_thresh = float(os.environ.get("SWAY_PHASE8_XTRACK_CONF_THRESH", "0.88") or 0.88)
    phase8_xtrack_gate_px = float(os.environ.get("SWAY_PHASE8_XTRACK_GATE_PX", str(width * 0.12)) or (width * 0.12))
    phase8_neighbor_window = max(1, int(os.environ.get("SWAY_PHASE8_NEIGHBOR_WINDOW", "4") or 4))
    phase8_neighbor_min_ratio = max(
        0.3, min(1.0, float(os.environ.get("SWAY_PHASE8_NEIGHBOR_MIN_RATIO", "0.50") or 0.50))
    )
    phase8_target_apply_ratio = max(
        0.0, min(1.0, float(os.environ.get("SWAY_PHASE8_XTRACK_TARGET_APPLY_RATIO", "0.00") or 0.00))
    )

    bbox_by_frame_tid: Dict[int, Dict[int, np.ndarray]] = {}
    for fidx_b, tracks_b in enumerate(all_track_results):
        frame_map: Dict[int, np.ndarray] = {}
        for t_b in tracks_b:
            frame_map[int(t_b[4])] = np.array([t_b[0], t_b[1], t_b[2], t_b[3]], dtype=np.float32)
        bbox_by_frame_tid[fidx_b] = frame_map

    def _nearest_assigned_did(
        frame_idx: int, center_xy: Tuple[float, float], assign_map: Dict[int, int]
    ) -> Tuple[int, float]:
        best_tid = -1
        best_dist = 1e12
        fmap = bbox_by_frame_tid.get(frame_idx, {})
        cx, cy = center_xy
        for tid_n, did_n in assign_map.items():
            bb_n = fmap.get(tid_n)
            if bb_n is None:
                continue
            ncx = float((bb_n[0] + bb_n[2]) * 0.5)
            ncy = float((bb_n[1] + bb_n[3]) * 0.5)
            dist = float(np.hypot(ncx - cx, ncy - cy))
            if dist < best_dist:
                best_dist = dist
                best_tid = int(tid_n)
        if best_tid < 0:
            return -1, 1e12
        return int(assign_map.get(best_tid, -1)), best_dist

    def _window_neighbor_vote(
        frame_idx: int, center_xy: Tuple[float, float]
    ) -> Tuple[int, float, float, int]:
        votes: List[int] = []
        max_dist = 0.0
        max_off = max(1, phase8_neighbor_window)
        for off in range(-max_off, max_off + 1):
            if off == 0:
                continue
            neigh_idx = frame_idx + off
            if neigh_idx < 0 or neigh_idx >= len(frames):
                continue
            neigh_assign = reid_assignments_phase4.get(neigh_idx, {})
            did_v, dist_v = _nearest_assigned_did(neigh_idx, center_xy, neigh_assign)
            if did_v > 0 and dist_v <= phase8_xtrack_gate_px:
                votes.append(int(did_v))
                max_dist = max(max_dist, float(dist_v))
        if not votes:
            return -1, 0.0, max_dist, 0
        from collections import Counter as _VoteCounter

        ctr = _VoteCounter(votes)
        cand_did, cnt = ctr.most_common(1)[0]
        ratio = float(cnt) / float(len(votes))
        return int(cand_did), ratio, max_dist, int(len(votes))

    # --- Phase 0 observability: phase-8 reject reason telemetry ---
    phase8_reject_log: List[Dict] = []
    P8_REJECT_CONF_TOO_HIGH = "conf_above_threshold"
    P8_REJECT_NEIGHBORS_DISAGREE = "prev_next_disagree"
    P8_REJECT_DUPLICATE_WOULD_RESULT = "duplicate_dancer_block"
    P8_REJECT_GATE_TOO_FAR = "gate_distance_exceeded"
    P8_REJECT_NO_PREV_DID = "no_prev_identity"
    P8_REJECT_KIN_CONF_OK = "kinematic_conf_ok"

    for fidx_opt in range(1, len(frames) - 1):
        prev_assign = reid_assignments_phase4.get(fidx_opt - 1, {})
        curr_assign = reid_assignments_phase4.get(fidx_opt, {})
        next_assign = reid_assignments_phase4.get(fidx_opt + 1, {})

        for tid_opt, did_opt in list(curr_assign.items()):
            prev_did = prev_assign.get(tid_opt, -1)
            next_did = next_assign.get(tid_opt, -1)
            if prev_did > 0 and next_did > 0 and prev_did == next_did and did_opt != prev_did:
                phase8_eligible_same_tid += 1
                curr_conf = reid_confidences.get(fidx_opt, {}).get(tid_opt, 0.0)
                if curr_conf < phase8_conf_thresh:
                    curr_assign[tid_opt] = prev_did
                    phase8_corrections += 1
                else:
                    phase8_reject_log.append({
                        "frame": fidx_opt, "track_id": int(tid_opt), "type": "same_tid",
                        "reason": P8_REJECT_CONF_TOO_HIGH,
                        "conf": round(curr_conf, 4), "threshold": phase8_conf_thresh,
                        "prev_did": int(prev_did), "curr_did": int(did_opt),
                    })

        for tid_opt in list(curr_assign.keys()):
            prev_bb = bbox_by_frame_tid.get(fidx_opt - 1, {}).get(tid_opt)
            curr_bb = bbox_by_frame_tid.get(fidx_opt, {}).get(tid_opt)
            if prev_bb is None or curr_bb is None:
                continue
            displacement = np.sqrt(
                ((curr_bb[0] + curr_bb[2]) / 2 - (prev_bb[0] + prev_bb[2]) / 2) ** 2
                + ((curr_bb[1] + curr_bb[3]) / 2 - (prev_bb[1] + prev_bb[3]) / 2) ** 2
            )
            if displacement > width * 0.3:
                curr_conf = reid_confidences.get(fidx_opt, {}).get(tid_opt, 0.0)
                if curr_conf < phase8_kin_conf_thresh:
                    prev_did = prev_assign.get(tid_opt, -1)
                    if prev_did > 0:
                        curr_assign[tid_opt] = prev_did
                        phase8_corrections += 1
                    else:
                        phase8_reject_log.append({
                            "frame": fidx_opt, "track_id": int(tid_opt), "type": "kinematic",
                            "reason": P8_REJECT_NO_PREV_DID,
                            "displacement_px": round(float(displacement), 1),
                        })
                else:
                    phase8_reject_log.append({
                        "frame": fidx_opt, "track_id": int(tid_opt), "type": "kinematic",
                        "reason": P8_REJECT_KIN_CONF_OK,
                        "conf": round(curr_conf, 4), "threshold": phase8_kin_conf_thresh,
                        "displacement_px": round(float(displacement), 1),
                    })

        for tid_opt, did_opt in list(curr_assign.items()):
            curr_bb = bbox_by_frame_tid.get(fidx_opt, {}).get(tid_opt)
            if curr_bb is None:
                continue
            curr_conf = reid_confidences.get(fidx_opt, {}).get(tid_opt, 0.0)
            if curr_conf >= phase8_xtrack_conf_thresh:
                continue
            center = (float((curr_bb[0] + curr_bb[2]) * 0.5), float((curr_bb[1] + curr_bb[3]) * 0.5))
            cand_did, vote_ratio, vote_max_dist, vote_count = _window_neighbor_vote(fidx_opt, center)
            required_vote_ratio = _phase8_required_vote_ratio(vote_count=vote_count, base_ratio=phase8_neighbor_min_ratio)
            if cand_did <= 0 or vote_ratio < required_vote_ratio:
                phase8_reject_log.append(
                    {
                        "frame": fidx_opt,
                        "track_id": int(tid_opt),
                        "type": "xtrack",
                        "reason": P8_REJECT_NEIGHBORS_DISAGREE,
                        "candidate_did": int(cand_did),
                        "vote_ratio": round(float(vote_ratio), 4),
                        "vote_count": int(vote_count),
                        "min_ratio": round(float(required_vote_ratio), 4),
                    }
                )
                continue
            if cand_did == did_opt:
                continue
            if vote_max_dist > phase8_xtrack_gate_px:
                phase8_reject_log.append({
                    "frame": fidx_opt, "track_id": int(tid_opt), "type": "xtrack",
                    "reason": P8_REJECT_GATE_TOO_FAR,
                    "max_dist": round(vote_max_dist, 1),
                    "gate_px": round(phase8_xtrack_gate_px, 1),
                    "candidate_did": int(cand_did),
                    "vote_ratio": round(float(vote_ratio), 4),
                })
                continue
            phase8_eligible_xtrack += 1
            incumbent_tid = next(
                (int(tid_other) for tid_other, did_other in curr_assign.items() if tid_other != tid_opt and did_other == cand_did),
                -1,
            )
            if incumbent_tid >= 0:
                incumbent_conf = float(reid_confidences.get(fidx_opt, {}).get(incumbent_tid, 0.0))
                candidate_score = (
                    0.60 * float(vote_ratio)
                    + 0.20 * (1.0 - float(curr_conf))
                    + 0.20 * max(0.0, 1.0 - float(vote_max_dist) / max(1.0, phase8_xtrack_gate_px))
                )
                incumbent_score = 0.70 * float(incumbent_conf) + 0.30
                score_delta = candidate_score - incumbent_score
                if did_opt > 0 and _xtrack_tiebreak_should_swap(candidate_score, incumbent_score, margin=0.05) and not any(
                    tid_other not in (tid_opt, incumbent_tid) and did_other == did_opt
                    for tid_other, did_other in curr_assign.items()
                ):
                    curr_assign[incumbent_tid] = int(did_opt)
                    curr_assign[tid_opt] = int(cand_did)
                    phase8_corrections += 1
                    phase8_applied_xtrack += 1
                else:
                    phase8_reject_log.append(
                        {
                            "frame": fidx_opt,
                            "track_id": int(tid_opt),
                            "type": "xtrack",
                            "reason": P8_REJECT_DUPLICATE_WOULD_RESULT,
                            "candidate_did": int(cand_did),
                            "incumbent_tid": int(incumbent_tid),
                            "vote_ratio": round(float(vote_ratio), 4),
                            "candidate_score": round(float(candidate_score), 5),
                            "incumbent_score": round(float(incumbent_score), 5),
                            "score_delta": round(float(score_delta), 5),
                        }
                    )
                    continue
            else:
                curr_assign[tid_opt] = int(cand_did)
                phase8_corrections += 1
                phase8_applied_xtrack += 1

    # Build final identity snapshot (source-of-truth for Phase 8 rendering)
    final_identity_tracks_by_frame: Dict[int, Dict[int, int]] = {}
    final_tracks_json = []
    for fidx_f in range(len(frames)):
        fa = reid_assignments_phase4.get(fidx_f, {})
        final_identity_tracks_by_frame[fidx_f] = {int(k): int(v) for k, v in fa.items()}
        final_tracks_json.append(
            {"frame": int(fidx_f), "assignments": {str(k): int(v) for k, v in fa.items()}}
        )

    phase8_render_unknown_tracks = 0
    phase8_render_final_tracks_hits = 0
    phase8_render_phase4_hits = 0
    phase8_render_sticky_hits = 0
    phase8_render_last_pose_hits = 0
    last_valid_did_by_track: Dict[int, int] = {}
    last_pose_by_did: Dict[int, np.ndarray] = {}

    # Write phase8 video
    writer8 = PhaseVideoWriter(OUTPUT_DIR / "phase8_final_optimized.mp4", fps, width, height)
    for fidx, frame in enumerate(frames):
        tracks = all_track_results[fidx]
        out = frame.copy()
        frame_render_assignments: Dict[int, int] = {}
        for did_kp, kp_v in pose_cache.get(fidx, {}).items():
            if int(did_kp) > 0 and kp_v is not None:
                last_pose_by_did[int(did_kp)] = kp_v
        _draw_phase_banner(out, f"Phase 8: Optimized | Frame {fidx} | {phase8_corrections} corrections")

        # Two-pass render: first resolve authoritative IDs (no duplicates possible),
        # then fill in sticky only where no conflict.
        track_render_info: List[Tuple[int, int, int, int, int, int, str]] = []
        used_dids_this_frame: set = set()

        # Pass 1: authoritative assignments (final + phase4)
        for t in tracks:
            x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            did = int(final_identity_tracks_by_frame.get(fidx, {}).get(tid, -1))
            if did > 0:
                last_valid_did_by_track[tid] = did
                used_dids_this_frame.add(did)
                track_render_info.append((x1, y1, x2, y2, tid, did, "final_identity_tracks"))
                phase8_render_final_tracks_hits += 1
                continue
            did = int(reid_assignments_phase4.get(fidx, {}).get(tid, -1))
            if did > 0:
                last_valid_did_by_track[tid] = did
                used_dids_this_frame.add(did)
                track_render_info.append((x1, y1, x2, y2, tid, did, "phase4_assignments"))
                phase8_render_phase4_hits += 1
                continue
            track_render_info.append((x1, y1, x2, y2, tid, -1, "pending_sticky"))

        # Pass 2: sticky fallback only where dancer_id not already used
        for idx_r in range(len(track_render_info)):
            x1, y1, x2, y2, tid, did, source = track_render_info[idx_r]
            if source != "pending_sticky":
                continue
            sticky_did = int(last_valid_did_by_track.get(tid, -1))
            if sticky_did > 0 and sticky_did not in used_dids_this_frame:
                used_dids_this_frame.add(sticky_did)
                track_render_info[idx_r] = (x1, y1, x2, y2, tid, sticky_did, "sticky_last_valid")
                phase8_render_sticky_hits += 1
            else:
                track_render_info[idx_r] = (x1, y1, x2, y2, tid, -1, "unknown")
                phase8_render_unknown_tracks += 1

        # Draw all tracks
        for x1, y1, x2, y2, tid, dancer_id, source in track_render_info:
            color = _color_for_id(dancer_id) if dancer_id >= 0 else (128, 128, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = _identity_label(
                int(dancer_id),
                int(tid),
                use_names=bool(use_name_labels),
                identity_name_by_did=identity_name_by_did,
            )
            _draw_text(out, label, (x1, y1 - 8), color)
            kp = pose_cache.get(fidx, {}).get(dancer_id)
            if kp is None and dancer_id > 0:
                kp = last_pose_by_did.get(dancer_id)
                if kp is not None:
                    phase8_render_last_pose_hits += 1
            if kp is not None:
                for j1, j2 in SKELETON_BONES:
                    if j1 < len(kp) and j2 < len(kp):
                        c1, c2 = float(kp[j1, 2]), float(kp[j2, 2])
                        if c1 < CONF_LOW_THRESH or c2 < CONF_LOW_THRESH:
                            continue
                        pt1 = (int(kp[j1, 0]), int(kp[j1, 1]))
                        pt2 = (int(kp[j2, 0]), int(kp[j2, 1]))
                        if c1 < CONF_MED_THRESH or c2 < CONF_MED_THRESH:
                            cv2.line(out, pt1, pt2, (140, 140, 140), 1, cv2.LINE_AA)
                        elif c1 < CONF_HIGH_THRESH or c2 < CONF_HIGH_THRESH:
                            cv2.line(out, pt1, pt2, (180, 180, 180), 2, cv2.LINE_AA)
                        else:
                            cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)
        writer8.write(out)
    writer8.release()

    phase8_time = (time.monotonic() - t_phase8) * 1000
    logger.info(
        "Phase 8 complete: %.1fms, %d corrections applied (eligible same-tid=%d, eligible xtrack=%d, applied xtrack=%d)",
        phase8_time,
        phase8_corrections,
        phase8_eligible_same_tid,
        phase8_eligible_xtrack,
        phase8_applied_xtrack,
    )
    phase8_xtrack_apply_ratio = float(phase8_applied_xtrack) / float(max(1, phase8_eligible_xtrack))
    logger.info(
        "Phase 8 xtrack apply ratio: %.4f (target >= %.4f)",
        phase8_xtrack_apply_ratio,
        phase8_target_apply_ratio,
    )
    if phase8_reject_log:
        from collections import Counter as _Counter
        reject_reasons = _Counter(r["reason"] for r in phase8_reject_log)
        logger.info("Phase 8 reject reason breakdown: %s", dict(reject_reasons))
    logger.info(
        "Phase 8 render identity source hits: final=%d phase4=%d sticky=%d unknown=%d last_pose_cache=%d",
        phase8_render_final_tracks_hits,
        phase8_render_phase4_hits,
        phase8_render_sticky_hits,
        phase8_render_unknown_tracks,
        phase8_render_last_pose_hits,
    )

    # Serialize final outputs
    with open(OUTPUT_DIR / "final_identity_tracks.json", "w") as f:
        json.dump(final_tracks_json, f)

    final_duplicate_frames = 0
    for fr in final_tracks_json:
        dids = list(fr["assignments"].values())
        if len(dids) != len(set(dids)):
            final_duplicate_frames += 1
    logger.info("Final identity duplicates: %d/%d frames", final_duplicate_frames, len(final_tracks_json))

    # GAP C-A: confidence threading summary
    phase3_vit_conf_means: List[float] = []
    phase3_rtmw_conf_means: List[float] = []
    phase3_rtmw_full_conf_means: List[float] = []
    for _frame_map in pretrack_pose_by_frame_det.values():
        for _pose_evidence in _frame_map.values():
            _vk = _pose_evidence.get("vitpose_keypoints")
            _rk = _pose_evidence.get("rtmwx_keypoints")
            _rkf = _pose_evidence.get("rtmwx_keypoints_full")
            if isinstance(_vk, np.ndarray) and _vk.size > 0:
                phase3_vit_conf_means.append(float(np.mean(_vk[:, 2])))
            if isinstance(_rk, np.ndarray) and _rk.size > 0:
                phase3_rtmw_conf_means.append(float(np.mean(_rk[:, 2])))
            if isinstance(_rkf, np.ndarray) and _rkf.size > 0:
                phase3_rtmw_full_conf_means.append(float(np.mean(_rkf[:, 2])))

    phase3_vit_conf_mean = float(np.mean(phase3_vit_conf_means)) if phase3_vit_conf_means else 0.0
    phase3_rtmw_conf_mean = float(np.mean(phase3_rtmw_conf_means)) if phase3_rtmw_conf_means else 0.0
    phase3_rtmw_full_conf_mean = float(np.mean(phase3_rtmw_full_conf_means)) if phase3_rtmw_full_conf_means else 0.0
    pose_mean_source = "phase3_vitpose_pretrack" if phase3_vit_conf_means else "legacy_pose_cache_post_gate"
    pose_mean_selected = phase3_vit_conf_mean if phase3_vit_conf_means else legacy_pose_conf_post_gate_mean

    conf_flow = {
        "detection_mean_conf": float(np.mean([d.confidence for dets in all_detections for d in dets])) if any(all_detections) else 0.0,
        "reid_mean_conf": float(np.mean([c for fc in reid_confidences.values() for c in fc.values()])) if reid_confidences else 0.0,
        "pose_mean_conf": pose_mean_selected,
        "pose_mean_conf_source": pose_mean_source,
        "phase3_vitpose_mean_conf": phase3_vit_conf_mean,
        "phase3_rtmwx_mean_conf_17j": phase3_rtmw_conf_mean,
        "phase3_rtmwx_mean_conf_full": phase3_rtmw_full_conf_mean,
        "legacy_pose_mean_conf_raw": legacy_pose_conf_raw_mean,
        "legacy_pose_mean_conf_post_gate": legacy_pose_conf_post_gate_mean,
        "legacy_pose_zeroed_joint_fraction": legacy_pose_zeroed_joint_fraction,
    }
    logger.info(
        "Confidence flow: det=%.3f reid=%.3f pose=%.3f (source=%s | phase3_vit=%.3f phase3_rtmw17=%.3f phase3_rtmwfull=%.3f legacy_post=%.3f legacy_raw=%.3f)",
        conf_flow["detection_mean_conf"],
        conf_flow["reid_mean_conf"],
        conf_flow["pose_mean_conf"],
        conf_flow["pose_mean_conf_source"],
        conf_flow["phase3_vitpose_mean_conf"],
        conf_flow["phase3_rtmwx_mean_conf_17j"],
        conf_flow["phase3_rtmwx_mean_conf_full"],
        conf_flow["legacy_pose_mean_conf_post_gate"],
        conf_flow["legacy_pose_mean_conf_raw"],
    )

    from sway.pose_coaching_metrics import build_coaching_motion_analysis

    coaching_motion_analysis: Dict[str, Any] = {}
    try:
        coaching_motion_analysis = build_coaching_motion_analysis(pose_cache, fps=float(fps))
        with open(OUTPUT_DIR / "coaching_motion_analysis.json", "w") as f:
            json.dump(coaching_motion_analysis, f, indent=2)
        logger.info("Wrote coaching_motion_analysis.json")
    except Exception as exc:
        logger.warning("coaching_motion_analysis export failed: %s", exc)

    # GAP E-A: evaluation metrics (proxy metrics without GT)
    eval_metrics = {
        "track_fragmentation": len(set(int(t[4]) for tr in all_track_results for t in tr)),
        "mean_track_length": float(np.mean([len([1 for tr in all_track_results if any(int(t[4]) == tid for t in tr)])
                                             for tid in set(int(t[4]) for tr in all_track_results for t in tr)])) if all_track_results else 0.0,
        "id_switches": int(track_switches),
        "id_switch_rate": float(track_switches) / max(track_assign_events, 1),
        "duplicate_id_frames": int(final_duplicate_frames),
        "dark_zones_resolved": int(stitch_result.dark_zones_resolved),
        "phase8_corrections": int(phase8_corrections),
        "phase8_xtrack_apply_ratio": float(phase8_xtrack_apply_ratio),
        "confidence_flow": conf_flow,
        "pose_bad_geometry_rate": float(pose_bad_geom) / max(pose_samples, 1),
        "pose_low_conf_rate": float(pose_low_conf) / max(pose_samples, 1),
        "phase8_render_identity_sources": {
            "final_identity_tracks_hits": int(phase8_render_final_tracks_hits),
            "phase4_hits": int(phase8_render_phase4_hits),
            "sticky_hits": int(phase8_render_sticky_hits),
            "unknown_tracks": int(phase8_render_unknown_tracks),
            "last_pose_cache_hits": int(phase8_render_last_pose_hits),
        },
        "recording_mode": mode_info,
        "formation_alignment": formation_alignment_info,
        "coaching_motion": {
            "artifact": "coaching_motion_analysis.json",
            "mean_angle_jerk_proxy": float(coaching_motion_analysis.get("mean_temporal_jerk_angle_proxy", 0.0) or 0.0),
            "phrase_count": len(coaching_motion_analysis.get("phrases", [])),
            "pairwise_sync_pairs": len(coaching_motion_analysis.get("pairwise_motion_sync", [])),
        },
    }

    # Temporal jerk score (3D pose smoothness)
    jerk_scores = []
    jerk_by_dancer: Dict[str, float] = {}
    for tid_j, pose_j in scene.poses.items():
        kp3d = pose_j.keypoints_3d
        if kp3d.shape[0] >= 4:
            vel = np.diff(kp3d, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            jerk_mag = np.mean(np.linalg.norm(jerk.reshape(-1, 3), axis=1))
            jmf = float(jerk_mag)
            jerk_scores.append(jmf)
            jerk_by_dancer[str(int(tid_j))] = jmf
    eval_metrics["mean_temporal_jerk"] = float(np.mean(jerk_scores)) if jerk_scores else 0.0
    eval_metrics["temporal_jerk_3d_by_dancer"] = jerk_by_dancer

    # --- Phase 0 observability: overlap-window evaluation slices ---
    overlap_windows: List[Dict] = []
    in_overlap = False
    ow_start = 0
    ow_switches = 0
    for fidx_ow in range(len(frames)):
        contam = frame_contaminated_tids.get(fidx_ow, set())
        if len(contam) >= 2:
            if not in_overlap:
                in_overlap = True
                ow_start = fidx_ow
                ow_switches = 0
            ow_switches += sum(1 for ev in switch_event_log if ev["frame"] == fidx_ow)
        else:
            if in_overlap:
                overlap_windows.append({
                    "start_frame": ow_start,
                    "end_frame": fidx_ow - 1,
                    "duration_frames": fidx_ow - ow_start,
                    "switches_in_window": ow_switches,
                })
                in_overlap = False
    if in_overlap:
        overlap_windows.append({
            "start_frame": ow_start,
            "end_frame": len(frames) - 1,
            "duration_frames": len(frames) - ow_start,
            "switches_in_window": ow_switches,
        })

    # --- Phase 0 observability: short-track distribution ---
    from collections import defaultdict as _defaultdict, Counter as _Counter2
    track_frame_counts: Dict[int, int] = _defaultdict(int)
    for tr_list in all_track_results:
        for t_st in tr_list:
            track_frame_counts[int(t_st[4])] += 1
    short_track_counts = {
        "under_5": sum(1 for v in track_frame_counts.values() if v < 5),
        "under_10": sum(1 for v in track_frame_counts.values() if v < 10),
        "under_20": sum(1 for v in track_frame_counts.values() if v < 20),
        "total_unique_tracks": len(track_frame_counts),
    }

    # --- Phase 0 observability: enrollment proximity pairs ---
    enroll_proximity_pairs: List[Dict] = []
    import math as _math
    for i_ep in range(len(galleries)):
        for j_ep in range(i_ep + 1, len(galleries)):
            ga, gb = galleries[i_ep], galleries[j_ep]
            dist_ep = _math.dist(ga.spatial_position, gb.spatial_position)
            if dist_ep < 0.10:
                enroll_proximity_pairs.append({
                    "dancer_a": int(ga.dancer_id),
                    "dancer_b": int(gb.dancer_id),
                    "spatial_distance": round(dist_ep, 5),
                })
    if enroll_proximity_pairs:
        logger.info("Enrollment near-duplicate pairs (dist < 0.10): %s", enroll_proximity_pairs)

    # --- Phase 0 observability: per-joint confidence distribution ---
    joint_conf_all = []
    for fc_jc in pose_cache.values():
        for kp_jc in fc_jc.values():
            for j_jc in range(min(17, kp_jc.shape[0])):
                joint_conf_all.append(float(kp_jc[j_jc, 2]))
    joint_conf_stats = {}
    if joint_conf_all:
        jca = sorted(joint_conf_all)
        joint_conf_stats = {
            "total_keypoints": len(jca),
            "below_0.2": sum(1 for c in jca if c < 0.2),
            "below_0.2_pct": round(sum(1 for c in jca if c < 0.2) / len(jca) * 100, 2),
            "below_0.3": sum(1 for c in jca if c < 0.3),
            "q10": round(jca[int(0.1 * len(jca))], 4),
            "q25": round(jca[int(0.25 * len(jca))], 4),
            "median": round(jca[int(0.5 * len(jca))], 4),
        }

    eval_metrics["short_track_distribution"] = short_track_counts
    eval_metrics["overlap_windows"] = overlap_windows
    eval_metrics["enrollment_proximity_pairs"] = enroll_proximity_pairs
    eval_metrics["joint_confidence_distribution"] = joint_conf_stats

    with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("Wrote evaluation_metrics.json")

    # --- Phase 0 observability: serialize event logs ---
    with open(OUTPUT_DIR / "switch_event_log.json", "w") as f:
        json.dump(switch_event_log, f, indent=2)
    logger.info("Wrote switch_event_log.json (%d events)", len(switch_event_log))

    with open(OUTPUT_DIR / "enrollment_reject_log.json", "w") as f:
        json.dump(enrollment_reject_log, f, indent=2)
    logger.info("Wrote enrollment_reject_log.json (%d events)", len(enrollment_reject_log))

    with open(OUTPUT_DIR / "phase8_reject_log.json", "w") as f:
        json.dump(phase8_reject_log, f, indent=2)
    logger.info("Wrote phase8_reject_log.json (%d events)", len(phase8_reject_log))
    if tracker_ab_report.get("enabled"):
        with open(OUTPUT_DIR / "tracker_ab_overlap.json", "w") as f:
            json.dump(tracker_ab_report, f, indent=2)
        logger.info("Wrote tracker_ab_overlap.json (%d backend results)", len(tracker_ab_report.get("results", [])))

    # =====================================================================
    # Summary
    # =====================================================================
    total_time = (
        phase0_time
        + phase1_time
        + phase2_mask_time
        + phase3_dualpose_time
        + phase35_disambiguation_time
        + phase3_track_time
        + bwd_time
        + phase5_reid_time
        + phase4_time
        + phase67_time
        + phase8_time
    )
    preflight_gate_passed = bool(reid_part_model != "bpbreid" or reid_feature_mode == "torchreid")
    enrollment_gate_passed = bool(enrollment_completion_ratio >= min_enrollment_ratio)
    reid_feature_path_gate_passed = bool(reid_part_model != "bpbreid" or reid_feature_mode == "torchreid")
    render_source_gate_passed = True
    correction_calibration_gate_passed = bool(phase8_xtrack_apply_ratio >= phase8_target_apply_ratio)
    gate_status = {
        "preflight_gate": {
            "passed": bool(preflight_gate_passed),
            "required_mode": "torchreid" if reid_part_model == "bpbreid" else "n/a",
            "actual_mode": reid_feature_mode,
        },
        "enrollment_gate": {
            "passed": bool(enrollment_gate_passed),
            "completion_ratio": float(enrollment_completion_ratio),
            "min_required_ratio": float(min_enrollment_ratio),
        },
        "reid_feature_path_gate": {
            "passed": bool(reid_feature_path_gate_passed),
            "mode": reid_feature_mode,
            "reason": reid_feature_mode_reason,
        },
        "render_source_gate": {
            "passed": bool(render_source_gate_passed),
            "identity_order": (
                "final_identity_tracks->formation_phase4->sticky_last_valid"
                if mode_info.get("final_mode") == "formation"
                else "final_identity_tracks->reid_assignments_phase4->sticky_last_valid"
            ),
            "unknown_tracks": int(phase8_render_unknown_tracks),
        },
        "correction_calibration_gate": {
            "passed": bool(correction_calibration_gate_passed),
            "xtrack_apply_ratio": float(phase8_xtrack_apply_ratio),
            "target_apply_ratio": float(phase8_target_apply_ratio),
        },
    }
    run_valid_for_ranking = bool(
        preflight_gate_passed
        and enrollment_gate_passed
        and reid_feature_path_gate_passed
        and render_source_gate_passed
        and correction_calibration_gate_passed
    )

    # Backward-compatible aliases for renamed artifacts.
    _write_alias_copy(OUTPUT_DIR / "phase6_reid_fusion.mp4", "phase6_reid.mp4")
    _write_alias_copy(OUTPUT_DIR / "phase7_darkzone_resolution.mp4", "phase7_darkzone.mp4")
    _write_alias_copy(OUTPUT_DIR / "phase7_legacy_pose3d_compat.mp4", "legacy_pose3d.mp4")
    _write_alias_copy(OUTPUT_DIR / "phase8_final_optimized.mp4", "phase8_final.mp4")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE V23 COMPLETE")
    logger.info("=" * 70)
    logger.info("  Video: %s (%d frames, %.1fs)", VIDEO_PATH.name, len(frames), len(frames) / fps)
    logger.info("  Dancers enrolled: %d", len(galleries))
    logger.info("  Phase 1 (Detection):   %8.1fms  (%.1f fps)", phase1_time, len(frames) / (phase1_time / 1000))
    logger.info("  Phase 2 (Masking):     %8.1fms", phase2_mask_time)
    logger.info("  Phase 3 (Dual Pose):   %8.1fms", phase3_dualpose_time)
    logger.info("  Phase 4 (Tracking):    %8.1fms (fwd) + %.1fms (bwd)", phase3_track_time, bwd_time)
    logger.info("  Phase 5 (Enrollment):  %8.1fms", phase0_time)
    logger.info("  Phase 6 (Re-ID):       %8.1fms", phase5_reid_time)
    logger.info("  Phase 7 (Dark-Zone):   %8.1fms", phase4_time)
    logger.info("  Legacy Pose/3D block:  %8.1fms", phase67_time)
    logger.info("  Phase 8 (Optimize):    %8.1fms", phase8_time)
    logger.info("  TOTAL:                 %8.1fms", total_time)
    logger.info("")
    logger.info("  Output videos:")
    _canonical_videos = [
        "phase1_detection.mp4",
        "phase2_masks.mp4",
        "phase3_vitpose.mp4",
        "phase3_rtmwx.mp4",
        "phase4_tracking_forward.mp4",
        "phase4_tracking_bidirectional.mp4",
        "phase6_reid_fusion.mp4",
        "phase7_darkzone_resolution.mp4",
        "phase7_legacy_pose3d_compat.mp4",
        "phase8_final_optimized.mp4",
    ]
    for name in _canonical_videos:
        f = OUTPUT_DIR / name
        if not f.exists():
            continue
        size_mb = f.stat().st_size / 1024 / 1024
        logger.info("    %s (%.1f MB)", f.name, size_mb)
    logger.info("  Log: %s", log_path)

    summary = {
        "video": str(VIDEO_PATH),
        "total_frames": len(frames),
        "dancers_enrolled": len(galleries),
        "phase_times_ms": {
            "phase1_detection": phase1_time,
            "phase2_masking": phase2_mask_time,
            "phase3_dual_pose": phase3_dualpose_time,
            "phase3_5_disambiguation": phase35_disambiguation_time,
            "phase4_tracking_fwd": phase3_track_time,
            "phase4_tracking_bwd": bwd_time,
            "phase5_enrollment": phase0_time,
            "phase6_reid": phase5_reid_time,
            "phase7_darkzone": phase4_time,
            "legacy_pose_3d_block": phase67_time,
            "phase8_optimization": phase8_time,
        },
        "total_ms": total_time,
        "dancers_lifted": len(scene.poses),
        "phase1_provenance": {
            "requested_precision": precision_requested,
            "effective_precision": precision_effective_name,
            "deimv2_enabled": bool(enable_deimv2),
            "yolo_like": sum(1 for s in detection_provenance if "yolo" in s or s == "unknown"),
            "detr_like": len(detection_provenance) - sum(1 for s in detection_provenance if "yolo" in s or s == "unknown"),
            "mean_agreement": float(np.mean(det_agreement_scores)) if det_agreement_scores else 0.0,
        },
        "phase2_runtime_knobs": {
            "mask_frame_stride": int(os.environ.get("SWAY_MASK_FRAME_STRIDE", "1") or 1),
            "mask_reuse_iou": float(os.environ.get("SWAY_MASK_REUSE_IOU", "0.70") or 0.70),
            "reused_masks": int(phase2_reused_masks),
        },
        "phase3_tracker": {
            "requested_backend": tracker_backend_requested,
            "effective_backend": tracker_backend_effective,
            "status": tracker_backend_status,
            "error": tracker_backend_error,
            "ab_report": tracker_ab_report,
        },
        "phase35_diagnostics": phase35_status,
        "phase5_reid_diagnostics": {
            "reid_feature_mode": reid_feature_mode,
            "reid_feature_mode_reason": reid_feature_mode_reason,
            "track_switches": int(track_switches),
            "track_assign_events": int(track_assign_events),
            "track_switch_rate": float(track_switches) / float(track_assign_events) if track_assign_events else 0.0,
            "duplicate_dancer_frames": int(duplicate_dancer_frames),
            "dancer_conflict_alarms": int(dancer_conflict_alarms),
            "face_embed_stride": int(face_embed_stride),
            "part_cache_hits": int(part_cache_hits),
            "part_cache_misses": int(part_cache_misses),
            "name_labels_enabled": bool(use_name_labels),
        },
        "phase4_diagnostics": {
            "dark_zones_resolved": int(stitch_result.dark_zones_resolved),
            "stitches": int(stitch_pairs_applied),
            "alias_links_applied": int(len(stitched_alias)),
        },
        "phase67_diagnostics": {
            "pose_samples": int(pose_samples),
            "pose_bad_geometry": int(pose_bad_geom),
            "pose_bad_geometry_rate": float(pose_bad_geom) / float(pose_samples) if pose_samples else 0.0,
            "pose_low_conf": int(pose_low_conf),
            "pose_low_conf_rate": float(pose_low_conf) / float(pose_samples) if pose_samples else 0.0,
        },
        "phase8_diagnostics": {
            "corrections": int(phase8_corrections),
            "eligible_same_tid": int(phase8_eligible_same_tid),
            "eligible_xtrack": int(phase8_eligible_xtrack),
            "applied_xtrack": int(phase8_applied_xtrack),
            "xtrack_apply_ratio": float(phase8_xtrack_apply_ratio),
            "target_xtrack_apply_ratio": float(phase8_target_apply_ratio),
            "render_unknown_tracks": int(phase8_render_unknown_tracks),
            "render_sticky_hits": int(phase8_render_sticky_hits),
            "render_final_tracks_hits": int(phase8_render_final_tracks_hits),
            "reject_reason_counts": dict(_Counter2(r["reason"] for r in phase8_reject_log)) if phase8_reject_log else {},
        },
        "observability": {
            "switch_events_total": len(switch_event_log),
            "switch_reason_counts": dict(_Counter2(ev["reason"] for ev in switch_event_log)) if switch_event_log else {},
            "overlap_windows_count": len(overlap_windows),
            "short_track_distribution": short_track_counts,
            "pre_reid_pruning": phase3_prune_diag,
            "enrollment_proximity_pairs": enroll_proximity_pairs,
            "joint_confidence_distribution": joint_conf_stats,
        },
        "confidence_flow": conf_flow,
        "evaluation_metrics": eval_metrics,
        "recording_mode": mode_info,
        "formation_alignment": formation_alignment_info,
        "identity_name_map": identity_name_by_did,
        "gate_status": gate_status,
        "run_valid_for_ranking": bool(run_valid_for_ranking),
        "masks_generated": len(all_masks),
        "backward_tracks_frames": len(all_backward_tracks),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_baseline_validation_report(
        output_dir=OUTPUT_DIR,
        video_path=VIDEO_PATH,
        summary_obj=summary,
        eval_metrics_obj=eval_metrics,
        phase35_status=phase35_status,
    )
    _write_phase3_data_json(OUTPUT_DIR / "data.json", all_track_results)
    run_manifest = _build_run_manifest(
        repo_root=Path(__file__).resolve().parent.parent,
        device=DEVICE,
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        tracker_backend_requested=tracker_backend_requested,
        tracker_backend_effective=tracker_backend_effective,
        precision_requested=precision_requested,
        precision_effective=precision_effective_name,
        phase_times_ms=summary["phase_times_ms"],
        tracker_ab_report=tracker_ab_report,
        summary=summary,
        reid_feature_mode=reid_feature_mode,
        reid_feature_mode_reason=reid_feature_mode_reason,
        gate_status=gate_status,
        run_valid_for_ranking=run_valid_for_ranking,
    )
    with open(OUTPUT_DIR / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info("Wrote run_manifest.json")
    _write_artifact_indexes(
        output_dir=OUTPUT_DIR,
        summary=summary,
        run_manifest=run_manifest,
        extra={
            "evaluation_metrics_path": "evaluation_metrics.json",
            "coaching_motion_analysis_path": "coaching_motion_analysis.json",
        },
    )

    logger.info("\nDone. Open %s to view results.", OUTPUT_DIR)


if __name__ == "__main__":
    main()
