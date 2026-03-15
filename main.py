"""
Sway Pose Tracking V3.3 — Main Orchestrator

Production-ready, M2-optimized pipeline:
  1. Streaming ingestion & detection (YOLO every frame, no stride)
  2. High-tenacity tracking & box stitch (track_buffer=90)
  3. Resolution-aware box pruning (20% duration, 3% kinetic, geometric mirrors)
  4. High-fidelity pose (ViTPose-Large, fp16 MPS)
  5. Skeleton re-association (OKS stitch, crossover)
  6. Smart mirror pruning (edge + inverted velocity + low lower-body conf)
  7. Temporal smoothing (1 Euro, conf<0.3 guard)
  8. Spatio-temporal scoring & export
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

import argparse
import time
from pathlib import Path

import cv2
import torch

from tracker import run_tracking, iter_video_frames
from track_pruning import (
    prune_tracks,
    prune_geometric_mirrors,
    prune_smart_mirrors,
    raw_tracks_to_per_frame,
)
from pose_estimator import PoseEstimator
from crossover import apply_crossover_refinement, apply_occlusion_reid
from smoother import PoseSmoother
from scoring import process_all_frames_scoring_vectorized
from visualizer import render_and_export


def get_device() -> torch.device:
    """Select compute device: MPS (Apple Silicon) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Sway Pose Tracking V3.3 - Pose estimation for dance videos"
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
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (Path("input")).mkdir(exist_ok=True)
    (Path("models")).mkdir(exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    total_start = time.time()

    # Phase 1 & 2: Streaming ingestion, detection, tracking (native FPS)
    print("\n[1/8] Running detection & tracking (YOLO11l + BoT-SORT @ native FPS, streaming)...")
    t0 = time.time()
    raw_tracks, total_frames, output_fps, _frames_list, native_fps, frame_width, frame_height = run_tracking(
        str(video_path)
    )
    print(f"  └─ {time.time() - t0:.1f}s")

    # Phase 3: Resolution-aware box pruning
    print("\n[2/8] Track pruning (20% duration, 3% kinetic)...")
    t0 = time.time()
    surviving_ids = prune_tracks(raw_tracks, total_frames)
    geometric_mirror_ids = prune_geometric_mirrors(
        raw_tracks, surviving_ids, frame_width, frame_height
    )
    surviving_ids = surviving_ids - geometric_mirror_ids
    if geometric_mirror_ids:
        print(f"  Pruned {len(geometric_mirror_ids)} geometric mirrors (edge + inverted velocity)")
    tracking_results = raw_tracks_to_per_frame(raw_tracks, total_frames, surviving_ids)
    print(f"  Kept {len(surviving_ids)} of {len(raw_tracks)} tracks after pruning")
    print(f"  └─ {time.time() - t0:.1f}s")

    # Phase 4: Pose estimation (stream frames from video)
    print("\n[3/8] Running pose estimation (ViTPose-Large)...")
    t0 = time.time()
    pose_estimator = PoseEstimator(device=device)

    raw_poses_by_frame = []
    frames_stored = []
    last_pct = -1
    for frame_idx, frame in iter_video_frames(str(video_path)):
        if frame_idx >= total_frames:
            break
        # Progress feedback every 10%
        pct = int(100 * (frame_idx + 1) / total_frames) if total_frames else 0
        if pct >= last_pct + 10 or pct == 100:
            print(f"  Frame {frame_idx + 1}/{total_frames} ({pct}%)")
            last_pct = pct

        frame_rgb = frame[:, :, ::-1]
        tracking = (
            tracking_results[frame_idx]
            if frame_idx < len(tracking_results)
            else {"boxes": [], "track_ids": [], "confs": []}
        )
        boxes = tracking["boxes"]
        track_ids = tracking["track_ids"]

        if len(boxes) > 0:
            poses = pose_estimator.estimate_poses(frame_rgb, boxes, track_ids)
        else:
            poses = {}

        raw_poses_by_frame.append(poses)
        frames_stored.append((frame_idx, None, boxes, track_ids))

    print(f"  └─ {time.time() - t0:.1f}s")

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

    # Phase 5: Occlusion re-ID + crossover refinement
    print("\n[4/8] Occlusion re-ID + crossover refinement...")
    t0 = time.time()
    apply_occlusion_reid(all_frame_data_pre)
    apply_crossover_refinement(all_frame_data_pre)
    print(f"  └─ {time.time() - t0:.1f}s")

    # Phase 6: Smart mirror pruning
    print("\n[5/8] Smart mirror pruning...")
    t0 = time.time()
    mirror_prune_ids = prune_smart_mirrors(
        raw_tracks, surviving_ids, [fd["poses"] for fd in all_frame_data_pre], frame_width
    )
    phase6_prune_ids = mirror_prune_ids
    surviving_after_mirror = set()
    for fd in all_frame_data_pre:
        surviving_after_mirror.update(t for t in fd["track_ids"] if t not in phase6_prune_ids)
    if mirror_prune_ids:
        print(f"  Pruned {len(mirror_prune_ids)} mirror tracks")
    print(f"  {len(surviving_after_mirror)} tracks after mirror pruning")
    print(f"  └─ {time.time() - t0:.1f}s")

    # Phase 7: Temporal smoothing (1 Euro, conf<0.3 guard)
    print("\n[6/8] Temporal smoothing (1 Euro filter)...")
    t0 = time.time()
    smoother = PoseSmoother(min_cutoff=1.0, beta=0.7)
    all_frame_data = []

    for i, fd_pre in enumerate(all_frame_data_pre):
        fidx = fd_pre["frame_idx"]
        boxes = fd_pre["boxes"]
        track_ids = fd_pre["track_ids"]
        poses_raw = fd_pre["poses"]

        poses_filtered = {tid: data for tid, data in poses_raw.items() if tid not in phase6_prune_ids}
        boxes_filtered = [b for b, tid in zip(boxes, track_ids) if tid not in phase6_prune_ids]
        track_ids_filtered = [tid for tid in track_ids if tid not in phase6_prune_ids]

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

    # Phase 8: Spatio-temporal scoring (circmean, cDTW, per-joint)
    print("\n[7/8] Spatio-temporal scoring...")
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

    # Phase 9: Export & visualization
    print("\n[8/8] Rendering and exporting...")
    t0 = time.time()
    render_and_export(
        video_path=video_path,
        all_frame_data=all_frame_data,
        processed_fps=output_fps,
        native_fps=native_fps,
        output_dir=output_dir,
    )
    print(f"  └─ {time.time() - t0:.1f}s")
    total_elapsed = time.time() - total_start
    print(f"\nDone. Outputs in {output_dir}")
    print(f"Total pipeline: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
