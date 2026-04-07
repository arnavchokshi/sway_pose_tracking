"""
Versioned pipeline checkpoint save/load for resume and experiment branching.

Boundary IDs match docs/MASTER_PIPELINE_GUIDELINE.md §4.4.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 1

CHECKPOINT_BOUNDARIES = (
    "after_phase_1",
    "after_phase_2",
    "after_phase_3",
    "after_phase_4",
    "after_phase_5",
    "after_phase_8",
    "final",
)


def _sha256_file(path: Path, max_bytes: int = 256 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n += len(chunk)
            if n > max_bytes:
                h.update(chunk[: len(chunk) - (n - max_bytes)])
                break
            h.update(chunk)
    return h.hexdigest()


def video_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    out: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }
    try:
        out["sha256_prefix"] = _sha256_file(path, max_bytes=8 * 1024 * 1024)
    except OSError:
        out["sha256_prefix"] = ""
    return out


def params_fingerprint(params_path: Optional[Path]) -> str:
    if params_path is None or not params_path.is_file():
        return ""
    try:
        raw = params_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()[:16]
    except OSError:
        return ""


def yolo_weights_fingerprint() -> str:
    try:
        from sway.tracker import resolve_yolo_inference_weights

        p = resolve_yolo_inference_weights()
        return str(p)
    except Exception:
        return ""


def write_manifest(
    checkpoint_dir: Path,
    *,
    boundary_id: str,
    video_fp: Dict[str, Any],
    params_fp: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "boundary_id": boundary_id,
        "created_unix": time.time(),
        "video": video_fp,
        "params_yaml_sha256_16": params_fp,
        "yolo_weights_resolved": yolo_weights_fingerprint(),
        "serialization": "sway_checkpoint_v1",
    }
    if extra:
        man.update(extra)
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )


def read_manifest(checkpoint_dir: Path) -> Dict[str, Any]:
    p = checkpoint_dir / "checkpoint_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing checkpoint_manifest.json in {checkpoint_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def validate_manifest(
    manifest: Dict[str, Any],
    *,
    video_path: Path,
    params_path: Optional[Path],
    expect_boundary: Optional[str],
    force: bool,
) -> None:
    if force:
        return
    if expect_boundary and manifest.get("boundary_id") != expect_boundary:
        raise ValueError(
            f"Checkpoint boundary {manifest.get('boundary_id')!r} != expected {expect_boundary!r}"
        )
    vf = manifest.get("video") or {}
    if int(vf.get("size_bytes", -1)) != int(video_path.stat().st_size):
        raise ValueError("Checkpoint video size mismatch vs input video")
    if params_path and params_path.is_file():
        want = params_fingerprint(params_path)
        got = manifest.get("params_yaml_sha256_16") or ""
        if want and got and want != got:
            raise ValueError("params.yaml fingerprint mismatch vs checkpoint (use --force-checkpoint-load)")


def save_phase1_yolo_dets(
    checkpoint_dir: Path,
    yolo_dets_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    *,
    total_frames: int,
    native_fps: float,
    output_fps: float,
    frame_width: int,
    frame_height: int,
    ystride: int,
    video_path: Path,
    params_path: Optional[Path],
    phase1_pre_classical_by_frame: Optional[
        Dict[int, List[Tuple[Tuple[float, float, float, float], float]]]
    ] = None,
    write_manifest_file: bool = True,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if write_manifest_file:
        write_manifest(
            checkpoint_dir,
            boundary_id="after_phase_1",
            video_fp=video_fingerprint(video_path),
            params_fp=params_fingerprint(params_path),
        )
    keys = sorted(yolo_dets_by_frame.keys())
    max_f = max(keys) if keys else -1
    n_frames = max(total_frames, max_f + 1)
    counts = np.zeros(n_frames, dtype=np.int32)
    flat_boxes: List[float] = []
    flat_confs: List[float] = []
    for fi in keys:
        pairs = yolo_dets_by_frame[fi]
        counts[fi] = len(pairs)
        for (x1, y1, x2, y2), cf in pairs:
            flat_boxes.extend([x1, y1, x2, y2])
            flat_confs.append(float(cf))
    payload: Dict[str, Any] = {
        "counts": counts,
        "boxes": np.array(flat_boxes, dtype=np.float32).reshape(-1, 4),
        "confs": np.array(flat_confs, dtype=np.float32),
        "total_frames": np.int32(total_frames),
        "native_fps": np.float32(native_fps),
        "output_fps": np.float32(output_fps),
        "frame_width": np.int32(frame_width),
        "frame_height": np.int32(frame_height),
        "ystride": np.int32(ystride),
    }
    if phase1_pre_classical_by_frame is not None:
        counts_pc = np.zeros(n_frames, dtype=np.int32)
        flat_b_pc: List[float] = []
        flat_c_pc: List[float] = []
        for fi in keys:
            pairs_pc = phase1_pre_classical_by_frame.get(fi, [])
            counts_pc[fi] = len(pairs_pc)
            for (x1, y1, x2, y2), cf in pairs_pc:
                flat_b_pc.extend([x1, y1, x2, y2])
                flat_c_pc.append(float(cf))
        payload["counts_pc"] = counts_pc
        payload["boxes_pc"] = np.array(flat_b_pc, dtype=np.float32).reshape(-1, 4) if flat_b_pc else np.zeros((0, 4), dtype=np.float32)
        payload["confs_pc"] = np.array(flat_c_pc, dtype=np.float32)
    np.savez_compressed(checkpoint_dir / "phase1_yolo.npz", **payload)


def load_live_phase3_bundle(output_dir: Path) -> Dict[str, Any]:
    """Load bundle written by save_live_phase3_bundle (live_artifacts/phase3_bundle.pkl.gz)."""
    p = output_dir / "live_artifacts"
    return load_pickle_state(p, "phase3_bundle")


def load_phase1_yolo_npz_file(npz_path: Path) -> Tuple[
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[str, Any],
]:
    """Load ``phase1_yolo.npz`` without requiring ``checkpoint_manifest.json`` (e.g. live_artifacts cache)."""
    z = np.load(npz_path, allow_pickle=False)
    return _decode_phase1_npz(z, manifest=None)


def _decode_phase1_npz(
    z: Any,
    *,
    manifest: Optional[Dict[str, Any]],
) -> Tuple[
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[str, Any],
]:
    counts = z["counts"]
    boxes = z["boxes"]
    confs = z["confs"]
    meta = {
        "total_frames": int(z["total_frames"]),
        "native_fps": float(z["native_fps"]),
        "output_fps": float(z["output_fps"]),
        "frame_width": int(z["frame_width"]),
        "frame_height": int(z["frame_height"]),
        "ystride": int(z["ystride"]),
    }
    if manifest is not None:
        meta["manifest"] = manifest
    out: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    cursor = 0
    for fi in range(len(counts)):
        n = int(counts[fi])
        if n == 0:
            continue
        pairs = []
        for _ in range(n):
            row = boxes[cursor]
            pairs.append(
                (
                    (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                    float(confs[cursor]),
                )
            )
            cursor += 1
        out[fi] = pairs
    if cursor != len(boxes):
        raise ValueError("phase1_yolo.npz corrupt: counts vs boxes mismatch")
    pre_out: Optional[Dict[int, List[Tuple[Tuple[float, float, float, float], float]]]] = None
    if "counts_pc" in z.files:
        counts_pc = z["counts_pc"]
        boxes_pc = z["boxes_pc"]
        confs_pc = z["confs_pc"]
        pre_out = {}
        cur_pc = 0
        for fi in range(len(counts_pc)):
            n = int(counts_pc[fi])
            if n == 0:
                continue
            pairs_pc = []
            for _ in range(n):
                row = boxes_pc[cur_pc]
                pairs_pc.append(
                    (
                        (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                        float(confs_pc[cur_pc]),
                    )
                )
                cur_pc += 1
            pre_out[fi] = pairs_pc
        if cur_pc != len(boxes_pc):
            raise ValueError("phase1_yolo.npz corrupt: counts_pc vs boxes_pc mismatch")
    meta["phase1_pre_classical"] = pre_out
    return out, meta


def load_phase1_yolo_dets(
    checkpoint_dir: Path,
) -> Tuple[
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[str, Any],
]:
    manifest = read_manifest(checkpoint_dir)
    z = np.load(checkpoint_dir / "phase1_yolo.npz", allow_pickle=False)
    out, meta = _decode_phase1_npz(z, manifest=manifest)
    return out, meta


def _gzip_pickle(obj: Any, path: Path) -> None:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(gzip.compress(raw, compresslevel=6))


def _gzip_unpickle(path: Path) -> Any:
    raw = gzip.decompress(path.read_bytes())
    return pickle.loads(raw)


def save_pickle_state(checkpoint_dir: Path, name: str, obj: Any) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _gzip_pickle(obj, checkpoint_dir / f"{name}.pkl.gz")


def load_pickle_state(checkpoint_dir: Path, name: str) -> Any:
    return _gzip_unpickle(checkpoint_dir / f"{name}.pkl.gz")


def save_after_phase_2(
    checkpoint_dir: Path,
    raw_pre: Any,
    *,
    total_frames: int,
    native_fps: float,
    output_fps: float,
    frame_width: int,
    frame_height: int,
    ystride: int,
    hybrid_sam_stats: Dict[str, Any],
    video_path: Path,
    params_path: Optional[Path],
    phase1_dets_by_frame: Optional[Dict[int, Any]] = None,
    phase1_pre_classical_by_frame: Optional[Dict[int, Any]] = None,
) -> None:
    """Checkpoint after Phase 1–2 (detection + tracking) before post-track stitch."""
    write_manifest(
        checkpoint_dir,
        boundary_id="after_phase_2",
        video_fp=video_fingerprint(video_path),
        params_fp=params_fingerprint(params_path),
        extra={"note": "raw_pre is pre-stitch tracks; optional phase1 dets for preview on resume"},
    )
    save_pickle_state(
        checkpoint_dir,
        "phase2_pre_stitch",
        {
            "raw_pre": raw_pre,
            "total_frames": total_frames,
            "native_fps": native_fps,
            "output_fps": output_fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "ystride": ystride,
            "hybrid_sam_stats": hybrid_sam_stats,
            "phase1_dets_by_frame": dict(phase1_dets_by_frame or {}),
            "phase1_pre_classical_by_frame": dict(phase1_pre_classical_by_frame or {}),
        },
    )


def load_after_phase_2(checkpoint_dir: Path) -> Dict[str, Any]:
    read_manifest(checkpoint_dir)
    return load_pickle_state(checkpoint_dir, "phase2_pre_stitch")


def save_after_phase_3(
    checkpoint_dir: Path,
    raw_tracks: Any,
    *,
    total_frames: int,
    native_fps: float,
    output_fps: float,
    frame_width: int,
    frame_height: int,
    ystride: int,
    hybrid_sam_stats: Dict[str, Any],
    video_path: Path,
    params_path: Optional[Path],
) -> None:
    write_manifest(
        checkpoint_dir,
        boundary_id="after_phase_3",
        video_fp=video_fingerprint(video_path),
        params_fp=params_fingerprint(params_path),
        extra={"note": "raw_tracks gzip_pickle includes TrackObservation masks"},
    )
    save_pickle_state(
        checkpoint_dir,
        "raw_tracks",
        {
            "raw_tracks": raw_tracks,
            "total_frames": total_frames,
            "native_fps": native_fps,
            "output_fps": output_fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "ystride": ystride,
            "hybrid_sam_stats": hybrid_sam_stats,
        },
    )


def load_after_phase_3(checkpoint_dir: Path) -> Dict[str, Any]:
    read_manifest(checkpoint_dir)
    return load_pickle_state(checkpoint_dir, "raw_tracks")


def save_after_phase_4(
    checkpoint_dir: Path,
    *,
    raw_tracks: Any,
    surviving_ids: Set[int],
    tracking_results: List[Dict[str, Any]],
    total_frames: int,
    frame_width: int,
    frame_height: int,
    native_fps: float,
    output_fps: float,
    prune_log_entries: List[Any],
    tracker_ids_before_prune: List[int],
    hybrid_sam_stats: Dict[str, Any],
    video_path: Path,
    params_path: Optional[Path],
) -> None:
    write_manifest(
        checkpoint_dir,
        boundary_id="after_phase_4",
        video_fp=video_fingerprint(video_path),
        params_fp=params_fingerprint(params_path),
    )
    save_pickle_state(
        checkpoint_dir,
        "phase4",
        {
            "raw_tracks": raw_tracks,
            "surviving_ids": surviving_ids,
            "tracking_results": tracking_results,
            "total_frames": total_frames,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "native_fps": native_fps,
            "output_fps": output_fps,
            "prune_log_entries": prune_log_entries,
            "tracker_ids_before_prune": tracker_ids_before_prune,
            "hybrid_sam_stats": hybrid_sam_stats,
        },
    )


def load_after_phase_4(checkpoint_dir: Path) -> Dict[str, Any]:
    read_manifest(checkpoint_dir)
    return load_pickle_state(checkpoint_dir, "phase4")


def save_after_phase_5(
    checkpoint_dir: Path,
    *,
    all_frame_data_pre: List[Dict[str, Any]],
    tracking_results: List[Dict[str, Any]],
    raw_poses_by_frame: List[Dict[str, Any]],
    embeddings_by_frame: List[Dict[str, Any]],
    frames_stored: List[Any],
    raw_tracks: Any,
    total_frames: int,
    frame_width: int,
    frame_height: int,
    native_fps: float,
    output_fps: float,
    prune_log_entries: List[Any],
    tracker_ids_before_prune: List[int],
    surviving_ids: Set[int],
    hybrid_sam_stats: Dict[str, Any],
    video_path: Path,
    params_path: Optional[Path],
) -> None:
    write_manifest(
        checkpoint_dir,
        boundary_id="after_phase_5",
        video_fp=video_fingerprint(video_path),
        params_fp=params_fingerprint(params_path),
    )
    save_pickle_state(
        checkpoint_dir,
        "phase5",
        {
            "all_frame_data_pre": all_frame_data_pre,
            "tracking_results": tracking_results,
            "raw_poses_by_frame": raw_poses_by_frame,
            "embeddings_by_frame": embeddings_by_frame,
            "frames_stored": frames_stored,
            "raw_tracks": raw_tracks,
            "total_frames": total_frames,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "native_fps": native_fps,
            "output_fps": output_fps,
            "prune_log_entries": prune_log_entries,
            "tracker_ids_before_prune": tracker_ids_before_prune,
            "surviving_ids": surviving_ids,
            "hybrid_sam_stats": hybrid_sam_stats,
        },
    )


def load_after_phase_5(checkpoint_dir: Path) -> Dict[str, Any]:
    read_manifest(checkpoint_dir)
    return load_pickle_state(checkpoint_dir, "phase5")


def save_after_phase_8(
    checkpoint_dir: Path,
    *,
    all_frame_data_pre: List[Dict[str, Any]],
    phase7_prune_ids: Set[int],
    raw_tracks: Any,
    total_frames: int,
    frame_width: int,
    frame_height: int,
    native_fps: float,
    output_fps: float,
    prune_log_entries: List[Any],
    tracker_ids_before_prune: List[int],
    surviving_ids: Set[int],
    hybrid_sam_stats: Dict[str, Any],
    snap_pre_dedup: List[Any],
    snap_post_dedup_pre_sanitize: List[Any],
    video_path: Path,
    params_path: Optional[Path],
) -> None:
    write_manifest(
        checkpoint_dir,
        boundary_id="after_phase_8",
        video_fp=video_fingerprint(video_path),
        params_fp=params_fingerprint(params_path),
    )
    save_pickle_state(
        checkpoint_dir,
        "phase8",
        {
            "all_frame_data_pre": all_frame_data_pre,
            "phase7_prune_ids": phase7_prune_ids,
            "raw_tracks": raw_tracks,
            "total_frames": total_frames,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "native_fps": native_fps,
            "output_fps": output_fps,
            "prune_log_entries": prune_log_entries,
            "tracker_ids_before_prune": tracker_ids_before_prune,
            "surviving_ids": surviving_ids,
            "hybrid_sam_stats": hybrid_sam_stats,
            "snap_pre_dedup": snap_pre_dedup,
            "snap_post_dedup_pre_sanitize": snap_post_dedup_pre_sanitize,
        },
    )


def load_after_phase_8(checkpoint_dir: Path) -> Dict[str, Any]:
    read_manifest(checkpoint_dir)
    return load_pickle_state(checkpoint_dir, "phase8")


def save_live_phase3_bundle(
    output_dir: Path,
    *,
    raw_tracks: Any,
    total_frames: int,
    native_fps: float,
    output_fps: float,
    frame_width: int,
    frame_height: int,
    ystride: int,
    hybrid_sam_stats: Dict[str, Any],
) -> None:
    """Persist post–phase-3 state for Pipeline Lab live pruning preview (full runs, not resume checkpoints)."""
    d = output_dir / "live_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    save_pickle_state(
        d,
        "phase3_bundle",
        {
            "raw_tracks": raw_tracks,
            "total_frames": total_frames,
            "native_fps": native_fps,
            "output_fps": output_fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "ystride": ystride,
            "hybrid_sam_stats": hybrid_sam_stats,
        },
    )


def save_final_marker(
    output_dir: Path,
    *,
    video_path: Path,
    params_path: Optional[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    d = output_dir / "checkpoint_final"
    d.mkdir(parents=True, exist_ok=True)
    write_manifest(
        d,
        boundary_id="final",
        video_fp=video_fingerprint(video_path),
        params_fp=params_fingerprint(params_path),
        extra=extra or {},
    )
    (d / "COMPLETE.txt").write_text("Pipeline completed phases 9–11; see run_manifest.json.\n", encoding="utf-8")
