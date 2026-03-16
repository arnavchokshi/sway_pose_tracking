"""
Sway Pose Tracking V3.4 — Main Orchestrator

Production-ready, M2-optimized pipeline:
  1. Streaming ingestion & detection (YOLO every frame)
  2. High-tenacity tracking & box stitch (track_buffer=90, relative radius)
  3. Resolution-aware box pruning (duration, kinetic, spatial outlier, traversal, bbox size, mirrors)
  4. High-fidelity pose (ViTPose, fp16 MPS) with visibility scoring
  5. Keypoint collision dedup (remove duplicate pose overlays)
  6. Skeleton re-association (OKS stitch, hybrid CVM crossover)
  7. Post-pose sync pruning + smart mirror pruning
  8. Temporal smoothing (1 Euro, conf<0.3 guard)
  9. Spatio-temporal scoring & export
"""
# Suppress protobuf/onnx MessageFactory GetPrototype noise (non-fatal)
import sys
import warnings


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

import sys
import warnings
import queue
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from tracker import run_tracking, iter_video_frames
from track_pruning import (
    prune_tracks,
    prune_geometric_mirrors,
    prune_spatial_outliers,
    prune_short_tracks,
    prune_bbox_size_outliers,
    prune_bad_aspect_ratio,
    prune_smart_mirrors,
    prune_low_sync_tracks,
    prune_low_confidence_tracks,
    prune_jittery_tracks,
    prune_completeness_audit,
    prune_by_stage_polygon,
    log_pruned_tracks,
    raw_tracks_to_per_frame,
)
from pose_estimator import PoseEstimator
from crossover import (
    apply_crossover_refinement,
    apply_occlusion_reid,
    compute_visibility_scores,
    deduplicate_collocated_poses,
)
from smoother import PoseSmoother
from scoring import process_all_frames_scoring_vectorized
from visualizer import (
    render_and_export,
    render_phase_clip,
    stitch_montage,
    draw_boxes_only,
    draw_frame_with_boxes,
    draw_frame,
)


def _interpolate_pose_gaps(
    raw_poses_by_frame: list,
    frames_stored: list,
    stride: int,
) -> None:
    """Fill pose gaps for frames skipped by pose_stride. Modifies raw_poses_by_frame in place."""
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
                kp = (1 - t) * kp_prev + t * kp_next
                sc = (1 - t) * sc_prev + t * sc_next
                interpolated[int(tid)] = {"keypoints": np.asarray(kp, dtype=np.float32), "scores": np.asarray(sc, dtype=np.float32)}
            elif tid in prev_poses:
                interpolated[int(tid)] = dict(prev_poses[tid])
            elif tid in next_poses:
                interpolated[int(tid)] = dict(next_poses[tid])
        raw_poses_by_frame[idx] = interpolated


def get_device() -> torch.device:
    """Select compute device: MPS (Apple Silicon) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        choices=["base", "large"],
        default="base",
        help="ViTPose model: base (faster) or large (more accurate)",
    )
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=1,
        choices=[1, 2],
        help="Run pose every Nth frame; interpolate others. 2 = ~2x faster.",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        default=False,
        help="Generate a pipeline montage video showing each phase.",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (Path("input")).mkdir(exist_ok=True)
    (Path("models")).mkdir(exist_ok=True)

    montage_clips = []
    montage_dir = output_dir / "_montage_clips"
    if args.montage:
        montage_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    log_resource_usage("startup")

    total_start = time.time()

    # ── Phase 1 & 2: Detection + Tracking ────────────────────────────────
    print("\n[1/9] Running detection & tracking (YOLO + BoT-SORT, streaming)...")
    t0 = time.time()
    raw_tracks, total_frames, output_fps, _frames_list, native_fps, frame_width, frame_height = run_tracking(
        str(video_path)
    )
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("1-detection+tracking")

    if args.montage:
        montage_clip_frames = int(native_fps * 6)
        montage_start = max(0, total_frames // 2 - (montage_clip_frames * 4) // 2)
        all_ids = set(raw_tracks.keys())
        all_tracking = raw_tracks_to_per_frame(raw_tracks, total_frames, all_ids)
        all_fd = [{"frame_idx": i, "boxes": t["boxes"], "track_ids": t["track_ids"],
                    "poses": {}} for i, t in enumerate(all_tracking)]
        montage_clips.append(render_phase_clip(
            video_path, all_fd, "Stage 1: Detection",
            lambda f, d: draw_boxes_only(f, d["boxes"], d["track_ids"]),
            native_fps, output_fps, montage_dir / "01_detection.mp4",
            clip_duration=6.0,
            start_frame=montage_start,
            caption="All raw YOLO bounding boxes (noise and all)",
        ))

    # ── Phase 3: Pre-pose pruning (V3.4 enhanced) ───────────────────────
    print("\n[2/9] Track pruning (duration, kinetic, spatial, traversal, bbox size, mirrors)...")
    t0 = time.time()
    surviving_ids = prune_tracks(raw_tracks, total_frames)
    initial_count = len(raw_tracks)

    # Log tracks that failed duration+kinetic filter
    duration_kinetic_pruned = set(raw_tracks.keys()) - surviving_ids
    if duration_kinetic_pruned:
        print(f"  Pruned {len(duration_kinetic_pruned)} tracks (duration/kinetic filter)")
        log_pruned_tracks(raw_tracks, duration_kinetic_pruned, "duration/kinetic", frame_width, frame_height)

    stage_pruned_ids = prune_by_stage_polygon(raw_tracks, surviving_ids, frame_width, frame_height)
    surviving_ids = surviving_ids - stage_pruned_ids
    if stage_pruned_ids:
        print(f"  Pruned {len(stage_pruned_ids)} tracks outside stage polygon")
        log_pruned_tracks(raw_tracks, stage_pruned_ids, "stage_polygon", frame_width, frame_height)

    spatial_pruned_ids = prune_spatial_outliers(
        raw_tracks, surviving_ids, frame_width, frame_height
    )
    surviving_ids = surviving_ids - spatial_pruned_ids
    if spatial_pruned_ids:
        print(f"  Pruned {len(spatial_pruned_ids)} spatial outliers (far from group)")
        log_pruned_tracks(raw_tracks, spatial_pruned_ids, "spatial_outlier", frame_width, frame_height)

    short_pruned_ids = prune_short_tracks(
        raw_tracks, surviving_ids, total_frames
    )
    surviving_ids = surviving_ids - short_pruned_ids
    if short_pruned_ids:
        print(f"  Pruned {len(short_pruned_ids)} short tracks (<20% of video)")
        log_pruned_tracks(raw_tracks, short_pruned_ids, "short_track", frame_width, frame_height)

    bbox_pruned_ids = prune_bbox_size_outliers(raw_tracks, surviving_ids)
    surviving_ids = surviving_ids - bbox_pruned_ids
    if bbox_pruned_ids:
        print(f"  Pruned {len(bbox_pruned_ids)} bbox size outliers")
        log_pruned_tracks(raw_tracks, bbox_pruned_ids, "bbox_size", frame_width, frame_height)

    aspect_pruned_ids = prune_bad_aspect_ratio(raw_tracks, surviving_ids)
    surviving_ids = surviving_ids - aspect_pruned_ids
    if aspect_pruned_ids:
        print(f"  Pruned {len(aspect_pruned_ids)} non-person aspect ratios (wider than tall)")
        log_pruned_tracks(raw_tracks, aspect_pruned_ids, "aspect_ratio", frame_width, frame_height)

    geometric_mirror_ids = prune_geometric_mirrors(
        raw_tracks, surviving_ids, frame_width, frame_height
    )
    surviving_ids = surviving_ids - geometric_mirror_ids
    if geometric_mirror_ids:
        print(f"  Pruned {len(geometric_mirror_ids)} geometric mirrors (edge + inverted velocity)")
        log_pruned_tracks(raw_tracks, geometric_mirror_ids, "geometric_mirror", frame_width, frame_height)

    tracking_results = raw_tracks_to_per_frame(raw_tracks, total_frames, surviving_ids)
    print(f"  Kept {len(surviving_ids)} of {initial_count} tracks after pre-pose pruning")
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("2-pruning")

    # ── Phase 4: Pose estimation with visibility scoring ─────────────────
    model_id = "usyd-community/vitpose-plus-large" if args.pose_model == "large" else "usyd-community/vitpose-plus-base"
    stride_note = f" stride={args.pose_stride}" if args.pose_stride > 1 else ""
    print(f"\n[3/9] Running pose estimation (ViTPose-{args.pose_model.title()}{stride_note}, visibility-gated)...")
    t0 = time.time()
    pose_estimator = PoseEstimator(device=device, model_name=model_id)

    raw_poses_by_frame = [{}] * total_frames
    frames_stored = [None] * total_frames
    last_pct = -1
    occluded_skips = 0

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

        while True:
            item = frame_q.get()
            if item is None:
                break
            frame_idx, frame = item

            tracking = (
                tracking_results[frame_idx]
                if frame_idx < len(tracking_results)
                else {"boxes": [], "track_ids": [], "confs": []}
            )
            boxes = tracking["boxes"]
            track_ids = tracking["track_ids"]
            frames_stored[frame_idx] = (frame_idx, None, boxes, track_ids)

            run_pose = (frame_idx % args.pose_stride == 0) and len(boxes) > 0
            if run_pose:
                pct = int(100 * (frame_idx + 1) / total_frames) if total_frames else 0
                if pct >= last_pct + 10 or pct == 100:
                    print(f"  Frame {frame_idx + 1}/{total_frames} ({pct}%)")
                    last_pct = pct

                # V3.4: Compute visibility scores to skip occluded tracks
                vis_scores = compute_visibility_scores(boxes, track_ids)

                boxes_to_estimate = []
                ids_to_estimate = []
                paddings = []
                poses = {}

                for tid, box in zip(track_ids, boxes):
                    # V3.4: Skip pose estimation for heavily occluded tracks
                    vis = vis_scores.get(tid, 1.0)
                    if vis < 0.3:
                        if tid in last_good_pose:
                            decayed = last_good_pose[tid].copy()
                            decayed["keypoints"] = decayed["keypoints"].copy()
                            decayed["keypoints"][:, 2] *= 0.85
                            decayed["scores"] = decayed["scores"].copy() * 0.85
                            poses[tid] = decayed
                            last_good_pose[tid] = decayed
                        occluded_skips += 1
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

                        pad = 0.15
                        if movement > 15.0 or dw > 15.0 or dh > 15.0:
                            pad = 0.25
                        elif movement < 6.0 and dw < 6.0 and dh < 6.0:
                            pad = 0.10
                    else:
                        pad = 0.15

                    boxes_to_estimate.append(box)
                    ids_to_estimate.append(tid)
                    paddings.append(pad)

                if boxes_to_estimate:
                    frame_rgb = frame[:, :, ::-1]
                    estimated = pose_estimator.estimate_poses(frame_rgb, boxes_to_estimate, ids_to_estimate, paddings)
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

    if occluded_skips:
        print(f"  Skipped {occluded_skips} occluded track-frames (visibility < 0.3)")
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("3-pose")

    # Build all_frame_data_pre for downstream phases
    all_frame_data_pre = []
    for i, (fidx, _frame, boxes, track_ids) in enumerate(frames_stored):
        raw_poses = raw_poses_by_frame[i]
        all_frame_data_pre.append({
            "frame_idx": fidx,
            "frame": None,
            "boxes": list(boxes),
            "track_ids": list(track_ids),
            "poses": dict(raw_poses),
        })

    # ── Phase 5: Keypoint collision dedup ─────────────────────────────────
    print("\n[4/9] Keypoint collision dedup...")
    t0 = time.time()
    dedup_count = 0
    for fd in all_frame_data_pre:
        before = len(fd["poses"])
        deduplicate_collocated_poses(fd)
        dedup_count += before - len(fd["poses"])
    if dedup_count:
        print(f"  Removed {dedup_count} duplicate pose overlays")
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("4-dedup")

    # ── Phase 6: Occlusion re-ID + crossover refinement ──────────────────
    print("\n[5/9] Occlusion re-ID + crossover refinement (hybrid CVM)...")
    t0 = time.time()
    apply_occlusion_reid(all_frame_data_pre)
    apply_crossover_refinement(all_frame_data_pre, frame_width=frame_width, frame_height=frame_height)
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("5-crossover")

    # ── Phase 7: Post-pose pruning (sync score + smart mirror + completeness) ─
    print("\n[6/9] Post-pose pruning (sync score + smart mirror + completeness)...")
    t0 = time.time()

    sync_prune_ids = prune_low_sync_tracks(all_frame_data_pre, surviving_ids)
    if sync_prune_ids:
        print(f"  Pruned {len(sync_prune_ids)} tracks with low sync score (non-dancers)")
        log_pruned_tracks(raw_tracks, sync_prune_ids, "low_sync", frame_width, frame_height)

    mirror_prune_ids = prune_smart_mirrors(
        raw_tracks, surviving_ids, [fd["poses"] for fd in all_frame_data_pre], frame_width
    )
    if mirror_prune_ids:
        print(f"  Pruned {len(mirror_prune_ids)} mirror tracks")
        log_pruned_tracks(raw_tracks, mirror_prune_ids, "smart_mirror", frame_width, frame_height)

    completeness_prune_ids = prune_completeness_audit(
        raw_tracks, surviving_ids, raw_poses_by_frame, frame_width, frame_height
    )
    if completeness_prune_ids:
        print(f"  Pruned {len(completeness_prune_ids)} tracks (seated/partial-body observers)")
        log_pruned_tracks(raw_tracks, completeness_prune_ids, "completeness", frame_width, frame_height)

    low_conf_prune_ids = prune_low_confidence_tracks(
        surviving_ids, [fd["poses"] for fd in all_frame_data_pre]
    )
    if low_conf_prune_ids:
        print(f"  Pruned {len(low_conf_prune_ids)} tracks with low mean keypoint confidence (non-person)")
        log_pruned_tracks(raw_tracks, low_conf_prune_ids, "low_confidence", frame_width, frame_height)

    jitter_prune_ids = prune_jittery_tracks(
        raw_tracks, surviving_ids, [fd["poses"] for fd in all_frame_data_pre]
    )
    if jitter_prune_ids:
        print(f"  Pruned {len(jitter_prune_ids)} tracks with excessive keypoint jitter (non-person)")
        log_pruned_tracks(raw_tracks, jitter_prune_ids, "jitter", frame_width, frame_height)

    phase7_prune_ids = sync_prune_ids | mirror_prune_ids | completeness_prune_ids | low_conf_prune_ids | jitter_prune_ids
    surviving_after_prune = set()
    for fd in all_frame_data_pre:
        surviving_after_prune.update(t for t in fd["track_ids"] if t not in phase7_prune_ids)
    print(f"  {len(surviving_after_prune)} tracks after post-pose pruning")
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("6-post-prune")

    if args.montage:
        postprune_fd = []
        for fd_pre in all_frame_data_pre:
            filt_poses = {tid: d for tid, d in fd_pre["poses"].items() if tid not in phase7_prune_ids}
            filt_boxes = [b for b, tid in zip(fd_pre["boxes"], fd_pre["track_ids"]) if tid not in phase7_prune_ids and tid in filt_poses]
            filt_tids = [tid for tid in fd_pre["track_ids"] if tid not in phase7_prune_ids and tid in filt_poses]
            postprune_fd.append({"frame_idx": fd_pre["frame_idx"], "boxes": filt_boxes,
                                  "track_ids": filt_tids, "poses": filt_poses})
        montage_clips.append(render_phase_clip(
            video_path, postprune_fd, "Stage 2: Pruning",
            lambda f, d: draw_boxes_only(f, d["boxes"], d["track_ids"]),
            native_fps, output_fps, montage_dir / "02_pruning.mp4",
            clip_duration=6.0,
            start_frame=montage_start,
            caption="Only surviving dancer boxes after all pruning",
        ))

    # ── Phase 8: Temporal smoothing (1 Euro, conf<0.3 guard) ─────────────
    print("\n[7/9] Temporal smoothing (1 Euro filter)...")
    t0 = time.time()
    smoother = PoseSmoother(min_cutoff=1.0, beta=0.7)
    all_frame_data = []

    for i, fd_pre in enumerate(all_frame_data_pre):
        fidx = fd_pre["frame_idx"]
        boxes = fd_pre["boxes"]
        track_ids = fd_pre["track_ids"]
        poses_raw = fd_pre["poses"]

        poses_filtered = {tid: data for tid, data in poses_raw.items() if tid not in phase7_prune_ids}
        # Filter out pruned tracks AND tracks that have no pose data (stripped by dedup)
        boxes_filtered = [b for b, tid in zip(boxes, track_ids) if tid not in phase7_prune_ids and tid in poses_filtered]
        track_ids_filtered = [tid for tid in track_ids if tid not in phase7_prune_ids and tid in poses_filtered]

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
        })

    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("7-smoothing")

    # ── Phase 9: Spatio-temporal scoring (circmean, cDTW, per-joint) ─────
    print("\n[8/9] Spatio-temporal scoring...")
    t0 = time.time()
    scoring_data = process_all_frames_scoring_vectorized(all_frame_data)
    if scoring_data is not None:
        for i, fd in enumerate(all_frame_data):
            fd["track_angles"] = scoring_data["track_angles"][i]
            fd["consensus_angles"] = scoring_data["consensus_angles"][i]
            fd["deviations"] = scoring_data["deviations"][i]
            fd["shape_errors"] = scoring_data.get("shape_errors", [{} for _ in all_frame_data])[i]
            fd["timing_errors"] = scoring_data.get("timing_errors", [{} for _ in all_frame_data])[i]

    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("8-scoring")

    if args.montage:
        montage_clips.append(render_phase_clip(
            video_path, all_frame_data, "Stage 3: Pose Estimation",
            lambda f, d: draw_frame_with_boxes(f, d["boxes"], d["track_ids"], d["poses"]),
            native_fps, output_fps, montage_dir / "03_pose.mp4",
            clip_duration=6.0,
            start_frame=montage_start + montage_clip_frames * 2,
            caption="Boxes + skeleton overlays on dancers",
        ))

    # ── Phase 10: Export & visualization ──────────────────────────────────
    print("\n[9/9] Rendering and exporting...")
    t0 = time.time()
    render_and_export(
        video_path=video_path,
        all_frame_data=all_frame_data,
        processed_fps=output_fps,
        native_fps=native_fps,
        output_dir=output_dir,
    )
    print(f"  └─ {time.time() - t0:.1f}s")
    log_resource_usage("9-export")

    if args.montage:
        montage_clips.append(render_phase_clip(
            video_path, all_frame_data, "Stage 4: Scoring",
            lambda f, d: draw_frame(f, d["boxes"], d["track_ids"], d["poses"],
                                     deviations=d.get("deviations"),
                                     shape_errors=d.get("shape_errors"),
                                     timing_errors=d.get("timing_errors")),
            native_fps, output_fps, montage_dir / "04_scoring.mp4",
            clip_duration=6.0,
            start_frame=montage_start + montage_clip_frames * 3,
            caption="Colored heatmap skeletons (final output)",
        ))
        print("\n  Stitching montage...")
        stitch_montage(montage_clips, output_dir / "montage.mp4", native_fps)
        import shutil as _shutil
        _shutil.rmtree(montage_dir, ignore_errors=True)

    total_elapsed = time.time() - total_start
    print(f"\nDone. Outputs in {output_dir}")
    print(f"Total pipeline: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log_resource_usage("final")


if __name__ == "__main__":
    main()
