"""
Burn prune / collision / post-pose annotations into phase preview MP4s (Pipeline Lab).

Matches Watch panel rules in pipeline_lab/web/src/lib/watchPrune.ts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .track_observation import coerce_observation

# --- Rule sets (mirror watchPrune.ts) ---

PRE_POSE_PREVIEW_RULES = frozenset(
    {
        "duration/kinetic",
        "stage_polygon",
        "spatial_outlier",
        "short_track",
        "audience_region",
        "late_entrant_short_span",
        "bbox_size",
        "aspect_ratio",
        "geometric_mirror",
    }
)

COLLISION_PREVIEW_RULES = frozenset(
    {
        "deduplicate_collocated_poses",
        "sanitize_pose_bbox_consistency",
    }
)

POST_POSE_PREVIEW_RULES = frozenset(
    {
        "tier_c_auto_reject",
        "phase7_voting",
    }
)

# BGR (OpenCV). Align loosely with Watch legend colors.
_RULE_COLOR_BGR: Dict[str, Tuple[int, int, int]] = {
    "duration/kinetic": (180, 105, 255),
    "stage_polygon": (200, 130, 255),
    "spatial_outlier": (147, 20, 255),
    "short_track": (255, 200, 100),
    "audience_region": (255, 160, 120),
    "late_entrant_short_span": (255, 180, 200),
    "bbox_size": (200, 180, 255),
    "aspect_ratio": (255, 100, 180),
    "geometric_mirror": (255, 255, 100),
    "deduplicate_collocated_poses": (40, 146, 251),  # orange
    "sanitize_pose_bbox_consistency": (252, 132, 192),  # purple
    "tier_c_auto_reject": (113, 113, 248),
    "phase7_voting": (36, 36, 220),
}

_KEPT_DEDUP_BGR = (80, 220, 74)  # green, dashed = kept ID in dedup pair

# Per-frame bbox comes from prune_log (exact), not from tracker history.
_LOG_BBOX_RULES = frozenset(
    {
        "deduplicate_collocated_poses",
        "sanitize_pose_bbox_consistency",
    }
)

_RULE_SHORT: Dict[str, str] = {
    "duration/kinetic": "DUR",
    "stage_polygon": "STG",
    "spatial_outlier": "SPC",
    "short_track": "SHORT",
    "audience_region": "AUD",
    "late_entrant_short_span": "LATE",
    "bbox_size": "BOX",
    "aspect_ratio": "ASP",
    "geometric_mirror": "MIR",
    "deduplicate_collocated_poses": "DEDUP",
    "sanitize_pose_bbox_consistency": "SAN",
    "tier_c_auto_reject": "TIER_C",
    "phase7_voting": "TIER_B",
}


def _finite_bbox_xyxy(e: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    b = e.get("bbox_xyxy")
    if isinstance(b, list) and len(b) >= 4:
        q = tuple(float(x) for x in b[:4])
        if all(np.isfinite(q)):
            return q
    m = e.get("bbox_xyxy_median")
    if isinstance(m, list) and len(m) >= 4:
        q = tuple(float(x) for x in m[:4])
        if all(np.isfinite(q)):
            return q
    return None


def _frames_for_entry(e: Dict[str, Any], max_frame_idx: int) -> List[int]:
    rule = str(e.get("rule") or "")
    if rule == "phase6_summary":
        return []
    if isinstance(e.get("frame_idx"), (int, float)):
        fi = int(e["frame_idx"])
        if 0 <= fi <= max_frame_idx:
            return [fi]
        return []
    fr = e.get("frame_range")
    if not isinstance(fr, list) or len(fr) < 2:
        fr = e.get("frame_span")
    if isinstance(fr, list) and len(fr) >= 2:
        a = int(max(0, min(int(fr[0]), max_frame_idx)))
        b = int(max(0, min(int(fr[1]), max_frame_idx)))
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return []


def _overlay_label_with_cause(entry: Dict[str, Any], base: str, summary_max: int = 44) -> str:
    """Append truncated cause_config.summary for burned-in video text."""
    cc = entry.get("cause_config")
    if not isinstance(cc, dict):
        return base
    summ = cc.get("summary")
    if not isinstance(summ, str) or not summ.strip():
        return base
    extra = " ".join(summ.split())
    if len(extra) > summary_max:
        extra = extra[: summary_max - 1] + "…"
    return f"{base} | {extra}"


def _entry_overlay_specs(e: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each spec: bbox (x1,y1,x2,y2), color BGR, label str, dashed bool."""
    rule = str(e.get("rule") or "")
    out: List[Dict[str, Any]] = []

    if rule == "deduplicate_collocated_poses":
        sup = _finite_bbox_xyxy(e)
        tid = e.get("track_id")
        if sup is not None:
            lab0 = f"DEDUP rm {tid}" if tid is not None else "DEDUP rm"
            lab = _overlay_label_with_cause(e, lab0)
            out.append({"bbox": sup, "color": _RULE_COLOR_BGR[rule], "label": lab, "dashed": False})
        ob = e.get("other_bbox_xyxy")
        if isinstance(ob, list) and len(ob) >= 4:
            bb = tuple(float(x) for x in ob[:4])
            if all(np.isfinite(bb)):
                kt = e.get("kept_track_id")
                lab0 = f"KEPT {kt}" if kt is not None else "KEPT"
                lab = _overlay_label_with_cause(e, lab0)
                out.append({"bbox": bb, "color": _KEPT_DEDUP_BGR, "label": lab, "dashed": True})
        return out

    bb = _finite_bbox_xyxy(e)
    if bb is None:
        return []

    short = _RULE_SHORT.get(rule, rule.replace("/", "_")[:8].upper())
    tid = e.get("track_id")
    lab0 = f"{short} {tid}" if tid is not None else short
    lab = _overlay_label_with_cause(e, lab0)
    color = _RULE_COLOR_BGR.get(rule, (80, 80, 255))
    out.append({"bbox": bb, "color": color, "label": lab, "dashed": False})
    return out


def _make_track_prune_spec(
    rule: str,
    tid: Any,
    bbox: Tuple[float, float, float, float],
    prune_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    short = _RULE_SHORT.get(rule, str(rule).replace("/", "_")[:8].upper())
    base = f"{short} {tid}" if tid is not None else short
    lab = _overlay_label_with_cause(prune_entry or {}, base)
    color = _RULE_COLOR_BGR.get(rule, (80, 80, 255))
    return {"bbox": bbox, "color": color, "label": lab, "dashed": False}


def _bbox_from_raw_tracks(
    raw_tracks: Dict[int, Any],
    tid: int,
    frame_idx: int,
) -> Optional[Tuple[float, float, float, float]]:
    entries = raw_tracks.get(int(tid))
    if not entries:
        return None
    for ent in entries:
        obs = coerce_observation(ent)
        if int(obs.frame_idx) != int(frame_idx):
            continue
        bb = obs.bbox
        q = tuple(float(x) for x in bb[:4])
        if all(np.isfinite(q)):
            return q
    return None


def _bbox_from_frame_data(
    fd: Optional[Dict[str, Any]],
    tid: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not fd:
        return None
    tids = fd.get("track_ids") or []
    boxes = fd.get("boxes") or []
    key = int(tid)
    for i, t in enumerate(tids):
        if int(t) != key:
            continue
        if i >= len(boxes):
            break
        b = boxes[i]
        q = tuple(float(x) for x in b[:4])
        if all(np.isfinite(q)):
            return q
    return None


def build_prune_overlay_index(
    prune_entries: List[Dict[str, Any]],
    rules,
    max_frame_idx: int,
    *,
    raw_tracks: Optional[Dict[int, Any]] = None,
    frame_data_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Map source frame index -> overlay specs.

    * Dedup / sanitize: use bbox stored in the log at ``frame_idx`` (already exact).
    * Track-level prunes (pre/post pose, etc.): use the **tracker box for that frame**
      from ``raw_tracks`` and/or ``frame_data_by_idx`` so boxes move frame-to-frame.
      If no observation exists on a frame, that frame is skipped (no median fallback).
    """
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    mf = max(0, int(max_frame_idx))

    for e in prune_entries:
        rule = str(e.get("rule") or "")
        if rule not in rules:
            continue

        if rule in _LOG_BBOX_RULES:
            specs = _entry_overlay_specs(e)
            if not specs:
                continue
            for fi in _frames_for_entry(e, mf):
                by_frame.setdefault(fi, []).extend(specs)
            continue

        tid = e.get("track_id")
        if tid is None:
            continue
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            continue

        frames = _frames_for_entry(e, mf)
        if not frames:
            continue

        for fi in frames:
            bb = None
            if raw_tracks is not None:
                bb = _bbox_from_raw_tracks(raw_tracks, tid_int, fi)
            if bb is None and frame_data_by_idx is not None:
                bb = _bbox_from_frame_data(frame_data_by_idx.get(fi), tid_int)
            if bb is None:
                continue
            by_frame.setdefault(fi, []).append(_make_track_prune_spec(rule, tid_int, bb, prune_entry=e))

    return by_frame


def _dashed_rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int,
    dash: int = 10,
    gap: int = 6,
) -> None:
    def _seg_h(y: int, xa: int, xb: int) -> None:
        if xa > xb:
            xa, xb = xb, xa
        x = xa
        while x < xb:
            xe = min(xb, x + dash)
            cv2.line(img, (x, y), (xe, y), color, thickness, cv2.LINE_AA)
            x = xe + gap

    def _seg_v(x: int, ya: int, yb: int) -> None:
        if ya > yb:
            ya, yb = yb, ya
        y = ya
        while y < yb:
            ye = min(yb, y + dash)
            cv2.line(img, (x, y), (x, ye), color, thickness, cv2.LINE_AA)
            y = ye + gap

    _seg_h(y1, x1, x2)
    _seg_h(y2, x1, x2)
    _seg_v(x1, y1, y2)
    _seg_v(x2, y1, y2)


def draw_prune_preview_overlays(frame_bgr: np.ndarray, specs: List[Dict[str, Any]]) -> None:
    """Draw in-place on an already-rendered BGR frame."""
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(w, h) / 900.0)
    thick = max(1, int(round(scale * 2)))

    for sp in specs:
        bbox = sp["bbox"]
        x1 = int(round(max(0, min(bbox[0], w - 1))))
        y1 = int(round(max(0, min(bbox[1], h - 1))))
        x2 = int(round(max(0, min(bbox[2], w - 1))))
        y2 = int(round(max(0, min(bbox[3], h - 1))))
        if x2 <= x1 or y2 <= y1:
            continue
        color = tuple(int(c) for c in sp["color"])
        if sp.get("dashed"):
            _dashed_rect(frame_bgr, x1, y1, x2, y2, color, max(2, thick))
        else:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, max(2, thick), cv2.LINE_AA)

        label = str(sp.get("label", ""))
        if not label:
            continue
        ty = max(18, y1 - 4)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx1, ty1 = x1, ty - th - 6
        tx2, ty2 = min(w - 2, tx1 + tw + 8), ty + 4
        ty1 = max(0, ty1)
        cv2.rectangle(frame_bgr, (tx1, ty1), (tx2, ty2), (22, 22, 22), -1, cv2.LINE_AA)
        cv2.rectangle(frame_bgr, (tx1, ty1), (tx2, ty2), color, 1, cv2.LINE_AA)
        cv2.putText(
            frame_bgr,
            label,
            (tx1 + 4, ty2 - 5),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )


def wrap_draw_fn_with_prune_overlays(
    draw_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    overlay_by_frame: Dict[int, List[Dict[str, Any]]],
) -> Callable[[np.ndarray, Dict[str, Any]], np.ndarray]:
    """After base draw, burn prune specs for ``interp['frame_idx']``."""

    def wrapped(frame: np.ndarray, interp: Dict[str, Any]) -> np.ndarray:
        out = draw_fn(frame, interp)
        fi = interp.get("frame_idx")
        if isinstance(fi, (int, float)):
            specs = overlay_by_frame.get(int(fi))
            if specs:
                draw_prune_preview_overlays(out, specs)
        return out

    return wrapped
