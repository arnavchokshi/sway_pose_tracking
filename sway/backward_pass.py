"""
Backward-Pass Gap Filling System (PLAN_16)

After the forward tracking sweep, runs a second pass on the reversed video.
Tracks that start cleanly in reverse often correspond to forward tracks that
ended with identity loss. The stitch layer merges forward and reverse tracks.

Env:
  SWAY_BACKWARD_PASS_ENABLED          – 0|1 (default 1)
  SWAY_BACKWARD_COI_ENABLED           – 0|1 (default 1, LOCKED) — run COI in reverse pass too
  SWAY_BACKWARD_STITCH_MIN_SIMILARITY – min fusion score for stitching (default 0.60)
  SWAY_BACKWARD_STITCH_MAX_GAP        – max frame gap for stitching (default 300)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sway.cross_object_interaction import CrossObjectInteraction
from sway.reid_fusion import ReIDQuery

logger = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


@dataclass
class ForwardTrack:
    """Summary of a forward-pass track."""
    track_id: int
    dancer_id: int
    start_frame: int
    end_frame: int
    embeddings: Optional[object] = None  # frozen gallery at track end
    is_dormant: bool = False


@dataclass
class ReverseTrack:
    """Summary of a reverse-pass track."""
    reverse_id: int
    start_frame_reversed: int   # frame index in reversed timeline
    end_frame_reversed: int
    start_frame_original: int   # mapped back to original timeline
    end_frame_original: int
    embeddings: Optional[object] = None


@dataclass
class MergedTrack:
    """A forward track stitched with a reverse track."""
    dancer_id: int
    forward_track_id: int
    reverse_track_id: Optional[int]
    start_frame: int
    end_frame: int
    gap_filled_frames: List[int] = field(default_factory=list)
    stitch_confidence: float = 0.0


def is_backward_pass_enabled() -> bool:
    return _env_bool("SWAY_BACKWARD_PASS_ENABLED", True)


def is_backward_coi_enabled() -> bool:
    """Whether to run Cross-Object Interaction during backward pass (LOCKED ON)."""
    return _env_bool("SWAY_BACKWARD_COI_ENABLED", True)


def read_frames_reversed(video_path: str) -> List[np.ndarray]:
    """Read all frames from video and return in reversed order.

    For efficiency, decodes once forward and reverses in memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames.reverse()

    logger.info(
        "Loaded %d frames for backward pass (%d→%d)",
        len(frames), len(frames) - 1, 0,
    )
    return frames


def run_backward_pass(
    video_path: str,
    forward_tracks: List[ForwardTrack],
    tracker_factory=None,
    detector=None,
    device: str = "cuda",
) -> List[ReverseTrack]:
    """Run the tracking pipeline on the reversed video.

    Args:
        video_path: path to the original video.
        forward_tracks: summary of forward-pass tracks.
        tracker_factory: callable to create a fresh tracker.
        detector: detection backend.
        device: torch device.

    Returns:
        List of ReverseTrack from the backward pass.
    """
    frames = read_frames_reversed(video_path)
    if not frames:
        return []

    total_frames = len(frames)
    reverse_tracks: List[ReverseTrack] = []
    progress_every = max(1, _env_int("SWAY_BACKWARD_PROGRESS_EVERY", 25))
    det_stride = max(1, _env_int("SWAY_BACKWARD_DET_STRIDE", 30))
    t0 = time.time()

    # Create a fresh tracker for the reverse pass
    if tracker_factory is None:
        return []
    tracker = tracker_factory(device=device)
    coi = CrossObjectInteraction() if is_backward_coi_enabled() else None

    def _detect_frame(frame_in: np.ndarray, frame_idx: int) -> list:
        if detector is None:
            return []
        try:
            out = detector.detect(frame_in, frame_idx=frame_idx)
        except TypeError:
            out = detector.detect(frame_in)
        if isinstance(out, tuple) and len(out) == 2:
            out = out[0]
        return out or []

    for ridx, frame in enumerate(frames):
        if ridx == 0 or ridx % progress_every == 0:
            elapsed = max(1e-6, time.time() - t0)
            fps_eff = (ridx + 1) / elapsed
            eta_s = max(0.0, (total_frames - (ridx + 1)) / max(fps_eff, 1e-6))
            pct = (100.0 * (ridx + 1)) / max(1, total_frames)
            print(
                f"  [backward-progress] frame {ridx + 1}/{total_frames} "
                f"({pct:.1f}%) | {fps_eff:.2f} fps | ETA ~{eta_s/60.0:.1f} min",
                flush=True,
            )
        # Map reversed index to original frame index
        original_idx = total_frames - 1 - ridx

        dets = _detect_frame(frame, ridx) if (ridx == 0 or ridx % det_stride == 0) else []
        results = tracker.process_frame(frame, ridx, detections=dets if dets else None)

        if coi is not None and results:
            masks_dict = {r.track_id: r.mask for r in results if r.mask is not None}
            logits_dict = tracker.get_logit_scores() if hasattr(tracker, "get_logit_scores") else {}
            for tid, logit in logits_dict.items():
                coi.update_logits(tid, logit)
            actions = coi.check_collisions(masks_dict, logits_dict, ridx)
            for action in actions:
                if action.mode == "delete" and hasattr(tracker, "remove_memory_entries"):
                    tracker.remove_memory_entries(action.track_id, action.start_frame, ridx)
                elif action.mode == "freeze" and hasattr(tracker, "freeze_memory"):
                    tracker.freeze_memory(action.track_id)

        # Collect reverse track info on the last frame
        for r in results:
            found = False
            for rt in reverse_tracks:
                if rt.reverse_id == r.track_id:
                    rt.end_frame_reversed = ridx
                    rt.start_frame_original = original_idx
                    found = True
                    break
            if not found:
                reverse_tracks.append(ReverseTrack(
                    reverse_id=r.track_id,
                    start_frame_reversed=ridx,
                    end_frame_reversed=ridx,
                    start_frame_original=original_idx,
                    end_frame_original=original_idx,
                    embeddings=ReIDQuery(
                        track_id=int(r.track_id),
                        spatial_position=(
                            float((r.bbox_xyxy[0] + r.bbox_xyxy[2]) * 0.5),
                            float((r.bbox_xyxy[1] + r.bbox_xyxy[3]) * 0.5),
                        ),
                    ),
                ))

    # Fix original frame mapping
    for rt in reverse_tracks:
        rt.end_frame_original = total_frames - 1 - rt.start_frame_reversed
        rt.start_frame_original = total_frames - 1 - rt.end_frame_reversed

    logger.info("Backward pass: %d reverse tracks found", len(reverse_tracks))
    return reverse_tracks


def stitch_forward_reverse(
    forward_tracks: List[ForwardTrack],
    reverse_tracks: List[ReverseTrack],
    fusion_engine=None,
    min_similarity: Optional[float] = None,
    max_gap: Optional[int] = None,
) -> List[MergedTrack]:
    """Stitch forward and reverse tracks to fill gaps.

    Args:
        forward_tracks: forward-pass track summaries.
        reverse_tracks: reverse-pass track summaries.
        fusion_engine: ReIDFusionEngine for similarity computation.
        min_similarity: minimum fusion score for stitching.
        max_gap: maximum frame gap for stitching.

    Returns:
        List of MergedTrack results.
    """
    if min_similarity is None:
        min_similarity = _env_float("SWAY_BACKWARD_STITCH_MIN_SIMILARITY", 0.60)
    if max_gap is None:
        max_gap = _env_int("SWAY_BACKWARD_STITCH_MAX_GAP", 300)

    merged: List[MergedTrack] = []
    used_reverse: set = set()

    dormant_tracks = [ft for ft in forward_tracks if ft.is_dormant]

    for ft in dormant_tracks:
        best_rt = None
        best_sim = -1.0

        for rt in reverse_tracks:
            if rt.reverse_id in used_reverse:
                continue

            # Check temporal overlap/gap
            gap = rt.start_frame_original - ft.end_frame
            if gap < 0 or gap > max_gap:
                continue

            # Compute similarity (placeholder — real implementation uses fusion_engine)
            sim = 0.7  # default similarity when fusion_engine not available
            if fusion_engine is not None and ft.embeddings and rt.embeddings:
                _, sim = fusion_engine.match(rt.embeddings)

            if sim > best_sim:
                best_sim = sim
                best_rt = rt

        if best_rt is not None and best_sim >= min_similarity:
            gap_frames = list(range(ft.end_frame + 1, best_rt.start_frame_original))
            merged.append(MergedTrack(
                dancer_id=ft.dancer_id,
                forward_track_id=ft.track_id,
                reverse_track_id=best_rt.reverse_id,
                start_frame=ft.start_frame,
                end_frame=best_rt.end_frame_original,
                gap_filled_frames=gap_frames,
                stitch_confidence=best_sim,
            ))
            used_reverse.add(best_rt.reverse_id)
            logger.info(
                "Stitched: forward track %d → reverse track %d (sim=%.3f, gap=%d frames)",
                ft.track_id, best_rt.reverse_id, best_sim, len(gap_frames),
            )

    # Forward tracks that didn't stitch
    stitched_fwd = {m.forward_track_id for m in merged}
    for ft in forward_tracks:
        if ft.track_id not in stitched_fwd:
            merged.append(MergedTrack(
                dancer_id=ft.dancer_id,
                forward_track_id=ft.track_id,
                reverse_track_id=None,
                start_frame=ft.start_frame,
                end_frame=ft.end_frame,
            ))

    logger.info(
        "Stitch complete: %d merged tracks (%d with gap fill)",
        len(merged), sum(1 for m in merged if m.reverse_track_id is not None),
    )
    return merged
