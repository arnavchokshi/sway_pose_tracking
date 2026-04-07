"""
Re-run Phase 4 (pre-pose pruning) from raw tracks + params.

Used by main.py and Pipeline Lab live preview so pruning sliders reflect the same logic as a full run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sway.track_pruning import (
    ASPECT_RATIO_MAX,
    AUDIENCE_REGION_WINDOW_FRAMES,
    AUDIENCE_REGION_X_MIN_FRAC,
    AUDIENCE_REGION_Y_MIN_FRAC,
    BBOX_SIZE_MAX_FRAC,
    BBOX_SIZE_MIN_FRAC,
    EDGE_ENTRANT_MARGIN_FRAC,
    EDGE_MARGIN_FRAC,
    EDGE_PRESENCE_FRAC,
    KINETIC_STD_FRAC,
    LATE_ENTRANT_MAX_SPAN_FRAC,
    LATE_ENTRANT_START_FRAC,
    log_pruned_tracks,
    prune_audience_region,
    prune_bad_aspect_ratio,
    prune_bbox_size_outliers,
    prune_by_stage_polygon,
    prune_geometric_mirrors,
    prune_late_entrant_short_span,
    prune_short_tracks,
    prune_spatial_outliers,
    prune_tracks,
    prune_cause_config,
    raw_tracks_to_per_frame,
    SHORT_TRACK_MIN_FRAC,
    SPATIAL_OUTLIER_STD_FACTOR,
)
from sway.tracker import iter_video_frames


def _parse_stage_polygon_env() -> Optional[List[Tuple[float, float]]]:
    raw = os.environ.get("SWAY_STAGE_POLYGON", "").strip()
    if not raw:
        return None
    try:
        pts = json.loads(raw)
        if not isinstance(pts, list) or len(pts) < 3:
            return None
        return [(float(p[0]), float(p[1])) for p in pts]
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        return None


def run_pre_pose_prune_phase(
    *,
    raw_tracks: Dict[int, List[Any]],
    total_frames: int,
    frame_width: int,
    frame_height: int,
    video_path: Path,
    params: Dict[str, Any],
    lab_phase4_tick: Optional[Callable[[float, str, int, Dict[str, Any]], None]] = None,
    lab_state: Optional[Dict[str, Any]] = None,
    lab_update_context: Optional[Callable[..., None]] = None,
    quiet: bool = False,
) -> Tuple[Set[int], List[Any], List[Dict[str, Any]], int, List[int]]:
    """
    Returns:
        surviving_ids, prune_log_entries, tracking_results, initial_count, tracker_ids_before_prune
    """
    prune_log_entries: List[Any] = []
    tracker_ids_before_prune = sorted(raw_tracks.keys(), key=lambda x: int(x) if isinstance(x, int) else 999)

    if not quiet:
        print("\n[4/11] Phase 4 — Pre-pose pruning (duration, kinetic, spatial, stage, mirrors, …)…")
    import time

    t0 = time.time()
    st = lab_state or {"last_emit": 0.0}

    def _tick(name: str, pct: int) -> None:
        if lab_phase4_tick is not None:
            lab_phase4_tick(t0, name, pct, st)

    _tick("duration_kinetic_scan", 5)
    _prune_kw: Dict[str, Any] = {}
    if "min_duration_ratio" in params:
        _prune_kw["min_duration_ratio"] = params["min_duration_ratio"]
    if "KINETIC_STD_FRAC" in params:
        _prune_kw["kinetic_std_frac"] = params["KINETIC_STD_FRAC"]
    surviving_ids = prune_tracks(raw_tracks, total_frames, **_prune_kw)
    initial_count = len(raw_tracks)
    _tick("after_duration_kinetic", 18)

    duration_kinetic_pruned = set(raw_tracks.keys()) - surviving_ids
    if duration_kinetic_pruned:
        if not quiet:
            print(f"  Pruned {len(duration_kinetic_pruned)} tracks (duration/kinetic filter)")
        _mdr = float(_prune_kw.get("min_duration_ratio", 0.20))
        _ksf = float(_prune_kw.get("kinetic_std_frac", KINETIC_STD_FRAC))
        log_pruned_tracks(
            raw_tracks,
            duration_kinetic_pruned,
            "duration/kinetic",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "duration/kinetic",
                "Phase 4: track failed duration (min fraction of possible lifespan + floor) "
                "or normalized kinetic (bbox center std vs KINETIC_STD_FRAC×median dancer bbox height).",
                {"min_duration_ratio": _mdr, "KINETIC_STD_FRAC": _ksf},
            ),
        )

    stage_polygon = _parse_stage_polygon_env()
    if stage_polygon is None and os.environ.get("SWAY_AUTO_STAGE_DEPTH", "0") == "1":
        from sway.depth_stage import estimate_stage_polygon

        first_frame = None
        for _, fr in iter_video_frames(str(video_path)):
            first_frame = fr
            break
        if first_frame is not None:
            stage_polygon = estimate_stage_polygon(first_frame)
            if stage_polygon and not quiet:
                print(f"  Auto stage polygon: {len(stage_polygon)} vertices (Depth Anything V2)")
            elif not stage_polygon and not quiet:
                print("  Auto stage polygon: skipped (depth unavailable or heuristic failed)")
    elif stage_polygon and not quiet:
        print(f"  Stage polygon: {len(stage_polygon)} vertices (SWAY_STAGE_POLYGON)")
    _tick("stage_spatial_short_audience", 42)

    stage_pruned_ids = prune_by_stage_polygon(
        raw_tracks, surviving_ids, frame_width, frame_height, polygon_normalized=stage_polygon
    )
    surviving_ids = surviving_ids - stage_pruned_ids
    if stage_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(stage_pruned_ids)} tracks outside stage polygon")
        _poly_src = (
            "SWAY_STAGE_POLYGON"
            if os.environ.get("SWAY_STAGE_POLYGON")
            else (
                "auto_depth"
                if stage_polygon and os.environ.get("SWAY_AUTO_STAGE_DEPTH", "0") == "1"
                else ("custom" if stage_polygon else "none")
            )
        )
        log_pruned_tracks(
            raw_tracks,
            stage_pruned_ids,
            "stage_polygon",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "stage_polygon",
                "Phase 4: median foot position (bottom-center of bbox) lies outside the stage polygon (normalized coords).",
                {"stage_polygon_source": _poly_src},
            ),
        )

    _spatial_kw: Dict[str, Any] = {}
    if "SPATIAL_OUTLIER_STD_FACTOR" in params:
        _spatial_kw["outlier_std_factor"] = params["SPATIAL_OUTLIER_STD_FACTOR"]
    spatial_pruned_ids = prune_spatial_outliers(
        raw_tracks, surviving_ids, frame_width, frame_height, **_spatial_kw
    )
    surviving_ids = surviving_ids - spatial_pruned_ids
    if spatial_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(spatial_pruned_ids)} spatial outliers (far from group)")
        _osf = float(_spatial_kw.get("outlier_std_factor", SPATIAL_OUTLIER_STD_FACTOR))
        log_pruned_tracks(
            raw_tracks,
            spatial_pruned_ids,
            "spatial_outlier",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "spatial_outlier",
                "Phase 4: median position farther than SPATIAL_OUTLIER_STD_FACTOR×σ from group centroid (with spread floor).",
                {"SPATIAL_OUTLIER_STD_FACTOR": _osf},
            ),
        )

    _short_kw: Dict[str, Any] = {}
    if "SHORT_TRACK_MIN_FRAC" in params:
        _short_kw["min_frac"] = params["SHORT_TRACK_MIN_FRAC"]
    short_pruned_ids = prune_short_tracks(
        raw_tracks,
        surviving_ids,
        total_frames,
        frame_width=frame_width,
        frame_height=frame_height,
        **_short_kw,
    )
    surviving_ids = surviving_ids - short_pruned_ids
    if short_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(short_pruned_ids)} short tracks (<20% of video)")
        _stf = float(_short_kw.get("min_frac", SHORT_TRACK_MIN_FRAC))
        log_pruned_tracks(
            raw_tracks,
            short_pruned_ids,
            "short_track",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "short_track",
                "Phase 4: detection count below SHORT_TRACK_MIN_FRAC×video length (with edge-entrant exemption rules).",
                {"SHORT_TRACK_MIN_FRAC": _stf, "EDGE_ENTRANT_MARGIN_FRAC": EDGE_ENTRANT_MARGIN_FRAC},
            ),
        )

    _audience_kw: Dict[str, Any] = {}
    if "AUDIENCE_REGION_X_MIN_FRAC" in params:
        _audience_kw["x_min_frac"] = params["AUDIENCE_REGION_X_MIN_FRAC"]
    if "AUDIENCE_REGION_Y_MIN_FRAC" in params:
        _audience_kw["y_min_frac"] = params["AUDIENCE_REGION_Y_MIN_FRAC"]
    audience_pruned_ids = prune_audience_region(
        raw_tracks, surviving_ids, frame_width, frame_height, **_audience_kw
    )
    surviving_ids = surviving_ids - audience_pruned_ids
    if audience_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(audience_pruned_ids)} tracks in audience region (bottom-right)")
        _ax = float(_audience_kw.get("x_min_frac", AUDIENCE_REGION_X_MIN_FRAC))
        _ay = float(_audience_kw.get("y_min_frac", AUDIENCE_REGION_Y_MIN_FRAC))
        log_pruned_tracks(
            raw_tracks,
            audience_pruned_ids,
            "audience_region",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "audience_region",
                "Phase 4: median position in early window falls in audience rectangle (x≥AUDIENCE_REGION_X_MIN_FRAC, y≥AUDIENCE_REGION_Y_MIN_FRAC).",
                {
                    "AUDIENCE_REGION_X_MIN_FRAC": _ax,
                    "AUDIENCE_REGION_Y_MIN_FRAC": _ay,
                    "AUDIENCE_REGION_WINDOW_FRAMES": int(
                        _audience_kw.get("window_frames", AUDIENCE_REGION_WINDOW_FRAMES)
                    ),
                },
            ),
        )

    late_span_pruned_ids = prune_late_entrant_short_span(
        raw_tracks,
        surviving_ids,
        total_frames,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    surviving_ids = surviving_ids - late_span_pruned_ids
    if late_span_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(late_span_pruned_ids)} late-entrant short-span tracks")
        log_pruned_tracks(
            raw_tracks,
            late_span_pruned_ids,
            "late_entrant_short_span",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "late_entrant_short_span",
                "Phase 4: first detection after LATE_ENTRANT_START_FRAC of video and span < LATE_ENTRANT_MAX_SPAN_FRAC×length (edge entrant exempt).",
                {
                    "LATE_ENTRANT_START_FRAC": LATE_ENTRANT_START_FRAC,
                    "LATE_ENTRANT_MAX_SPAN_FRAC": LATE_ENTRANT_MAX_SPAN_FRAC,
                    "EDGE_ENTRANT_MARGIN_FRAC": EDGE_ENTRANT_MARGIN_FRAC,
                },
            ),
        )

    bbox_pruned_ids = prune_bbox_size_outliers(raw_tracks, surviving_ids, frame_height=frame_height)
    surviving_ids = surviving_ids - bbox_pruned_ids
    if bbox_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(bbox_pruned_ids)} bbox size outliers")
        log_pruned_tracks(
            raw_tracks,
            bbox_pruned_ids,
            "bbox_size",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "bbox_size",
                "Phase 4: median bbox height outside BBOX_SIZE_MIN_FRAC–BBOX_SIZE_MAX_FRAC of group median (foreground relaxed max).",
                {"BBOX_SIZE_MIN_FRAC": BBOX_SIZE_MIN_FRAC, "BBOX_SIZE_MAX_FRAC": BBOX_SIZE_MAX_FRAC},
            ),
        )

    aspect_pruned_ids = prune_bad_aspect_ratio(raw_tracks, surviving_ids)
    surviving_ids = surviving_ids - aspect_pruned_ids
    if aspect_pruned_ids:
        if not quiet:
            print(f"  Pruned {len(aspect_pruned_ids)} non-person aspect ratios (wider than tall)")
        log_pruned_tracks(
            raw_tracks,
            aspect_pruned_ids,
            "aspect_ratio",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "aspect_ratio",
                "Phase 4: median bbox width/height exceeds ASPECT_RATIO_MAX (non-person objects).",
                {"ASPECT_RATIO_MAX": ASPECT_RATIO_MAX},
            ),
        )

    _gem = float(params.get("EDGE_MARGIN_FRAC", EDGE_MARGIN_FRAC))
    _gep = float(params.get("EDGE_PRESENCE_FRAC", EDGE_PRESENCE_FRAC))
    geometric_mirror_ids = prune_geometric_mirrors(
        raw_tracks, surviving_ids, frame_width, frame_height
    )
    surviving_ids = surviving_ids - geometric_mirror_ids
    if geometric_mirror_ids:
        if not quiet:
            print(f"  Pruned {len(geometric_mirror_ids)} geometric mirrors (edge + inverted velocity)")
        log_pruned_tracks(
            raw_tracks,
            geometric_mirror_ids,
            "geometric_mirror",
            frame_width,
            frame_height,
            prune_log_entries,
            total_frames=total_frames,
            cause_config=prune_cause_config(
                "geometric_mirror",
                "Phase 4: edge-persistent track with inverted horizontal velocity (mirror reflection heuristic).",
                {"EDGE_MARGIN_FRAC": _gem, "EDGE_PRESENCE_FRAC": _gep},
            ),
        )

    _tick("assemble_tracking_results", 88)
    tracking_results = raw_tracks_to_per_frame(raw_tracks, total_frames, surviving_ids)
    if not quiet:
        print(f"  Kept {len(surviving_ids)} of {initial_count} tracks after pre-pose pruning")
    if lab_update_context is not None:
        lab_update_context(
            surviving_after_pre_pose_prune=int(len(surviving_ids)),
            pre_pose_prune_dropped=int(initial_count - len(surviving_ids)),
        )

    return surviving_ids, prune_log_entries, tracking_results, initial_count, tracker_ids_before_prune
