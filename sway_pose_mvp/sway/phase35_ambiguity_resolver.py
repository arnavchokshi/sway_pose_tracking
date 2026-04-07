from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Phase35Config:
    iou_ambiguous_thresh: float = 0.35
    low_conf_thresh: float = 0.45
    min_margin: float = 0.03
    temporal_weight: float = 0.15
    hair_hand_guard_enabled: bool = False
    hair_hand_conf_thresh: float = 0.75
    hair_hand_rt_conf_thresh: float = 0.35
    hair_hand_disagree_px: float = 28.0
    hair_hand_arm_ratio_max: float = 1.9
    hair_hand_temporal_jump_px: float = 30.0
    hair_hand_lock_frames: int = 4
    hair_hand_replace_elbow: bool = True


def _bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(1e-6, a_area + b_area - inter)
    return float(inter / union)


def _mean_conf(kp: Optional[np.ndarray]) -> float:
    if not isinstance(kp, np.ndarray) or kp.size == 0 or kp.shape[1] < 3:
        return 0.0
    return float(np.mean(kp[:, 2]))


def _temporal_consistency(cur: Optional[np.ndarray], prev: Optional[np.ndarray]) -> float:
    if not isinstance(cur, np.ndarray) or not isinstance(prev, np.ndarray):
        return 0.0
    if cur.size == 0 or prev.size == 0:
        return 0.0
    n = min(cur.shape[0], prev.shape[0])
    if n <= 0:
        return 0.0
    cur_xy = cur[:n, :2]
    prev_xy = prev[:n, :2]
    d = np.linalg.norm(cur_xy - prev_xy, axis=1)
    # Normalize to [0,1]-like score where lower motion gives higher consistency.
    return float(np.exp(-np.mean(d) / 35.0))


def _copy_dual(dual: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in dual.items():
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(v, dtype=np.float32).copy()
        else:
            out[k] = v
    return out


def _overlap_level(overlap_count: int) -> str:
    if overlap_count <= 0:
        return "easy"
    if overlap_count == 1:
        return "medium"
    return "hard"


def _joint_conf(kp: Optional[np.ndarray], j: int) -> float:
    if not isinstance(kp, np.ndarray) or kp.size == 0:
        return 0.0
    if kp.shape[0] <= j or kp.shape[1] < 3:
        return 0.0
    return float(kp[j, 2])


def _joint_xy(kp: Optional[np.ndarray], j: int) -> Optional[np.ndarray]:
    if not isinstance(kp, np.ndarray) or kp.size == 0:
        return None
    if kp.shape[0] <= j or kp.shape[1] < 2:
        return None
    return np.asarray(kp[j, :2], dtype=np.float32)


def _dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


def _is_hair_hand_suspicious(
    *,
    vit: np.ndarray,
    rtm: Optional[np.ndarray],
    prev_selected: Optional[np.ndarray],
    side: str,
    cfg: Phase35Config,
) -> bool:
    if side == "left":
        s_idx, e_idx, w_idx = 5, 7, 9
    else:
        s_idx, e_idx, w_idx = 6, 8, 10

    v_wc = _joint_conf(vit, w_idx)
    if v_wc < cfg.hair_hand_conf_thresh:
        return False

    v_s = _joint_xy(vit, s_idx)
    v_e = _joint_xy(vit, e_idx)
    v_w = _joint_xy(vit, w_idx)
    if v_s is None or v_e is None or v_w is None:
        return False

    upper = _dist(v_s, v_e)
    lower = _dist(v_e, v_w)
    arm_ratio_bad = (upper > 1e-3 and (lower / upper) > cfg.hair_hand_arm_ratio_max)

    disagree_bad = False
    if isinstance(rtm, np.ndarray):
        r_w = _joint_xy(rtm, w_idx)
        r_wc = _joint_conf(rtm, w_idx)
        if r_w is not None and r_wc >= cfg.hair_hand_rt_conf_thresh:
            disagree_bad = _dist(v_w, r_w) >= cfg.hair_hand_disagree_px

    jump_bad = False
    if isinstance(prev_selected, np.ndarray):
        p_w = _joint_xy(prev_selected, w_idx)
        if p_w is not None:
            jump_bad = _dist(v_w, p_w) >= cfg.hair_hand_temporal_jump_px

    return bool(arm_ratio_bad or disagree_bad or jump_bad)


def _apply_hair_hand_replacements(
    *,
    out_dual: Dict[str, Any],
    vit: np.ndarray,
    rtm: Optional[np.ndarray],
    prev_selected: Optional[np.ndarray],
    cfg: Phase35Config,
    lock_remaining: Dict[str, int],
) -> Tuple[int, Dict[str, int], List[str]]:
    """
    Replace suspicious ViTPose wrist (and optional elbow) with RTMW joints.
    Returns (num_joints_replaced, updated_lock_remaining, reasons).
    """
    replaced = 0
    reasons: List[str] = []
    if not isinstance(rtm, np.ndarray):
        return replaced, lock_remaining, reasons

    out_kp = np.asarray(out_dual.get("vitpose_keypoints"), dtype=np.float32).copy()
    if out_kp.shape[0] < 11 or rtm.shape[0] < 11:
        return replaced, lock_remaining, reasons

    for side in ("left", "right"):
        if side == "left":
            e_idx, w_idx = 7, 9
        else:
            e_idx, w_idx = 8, 10
        locked = int(lock_remaining.get(side, 0))
        suspicious = _is_hair_hand_suspicious(
            vit=vit,
            rtm=rtm,
            prev_selected=prev_selected,
            side=side,
            cfg=cfg,
        )
        should_replace = bool(locked > 0 or suspicious)
        if not should_replace:
            lock_remaining[side] = max(0, locked - 1)
            continue

        # RTMW must have at least minimal confidence on the wrist.
        if _joint_conf(rtm, w_idx) < cfg.hair_hand_rt_conf_thresh:
            lock_remaining[side] = max(0, locked - 1)
            continue

        out_kp[w_idx, : min(out_kp.shape[1], rtm.shape[1])] = rtm[w_idx, : min(out_kp.shape[1], rtm.shape[1])]
        replaced += 1
        reasons.append(f"{side}_wrist")
        if cfg.hair_hand_replace_elbow and _joint_conf(rtm, e_idx) >= cfg.hair_hand_rt_conf_thresh:
            out_kp[e_idx, : min(out_kp.shape[1], rtm.shape[1])] = rtm[e_idx, : min(out_kp.shape[1], rtm.shape[1])]
            replaced += 1
            reasons.append(f"{side}_elbow")
        # refresh short lock to avoid flicker across adjacent frames
        lock_remaining[side] = int(cfg.hair_hand_lock_frames)

    if replaced > 0:
        out_dual["vitpose_keypoints"] = out_kp
        out_dual["vitpose_mean_conf"] = float(_mean_conf(out_kp))
    return replaced, lock_remaining, reasons


def resolve_phase35(
    *,
    pretrack_pose_by_frame_det: Dict[int, Dict[int, Dict[str, Any]]],
    all_detection_boxes: Dict[int, List[Tuple[float, float, float, float]]],
    cfg: Optional[Phase35Config] = None,
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], Dict[str, Any]]:
    """
    Ambiguity resolver sidecar for Phase 3.5.

    It keeps baseline behavior by default and only switches to RTMW candidate when
    overlap/low-confidence ambiguity is detected with a sufficient scoring margin.
    """
    cfg = cfg or Phase35Config()
    resolved: Dict[int, Dict[int, Dict[str, Any]]] = {}
    diagnostics: Dict[str, Any] = {
        "schema_version": "phase35_diagnostics_v1",
        "frames_total": int(len(pretrack_pose_by_frame_det)),
        "frames_with_ambiguity": 0,
        "detections_total": 0,
        "detections_ambiguous": 0,
        "overrides_total": 0,
        "bins": {"easy": {"detections": 0, "overrides": 0}, "medium": {"detections": 0, "overrides": 0}, "hard": {"detections": 0, "overrides": 0}},
        "margin_stats": {"mean": 0.0, "p50": 0.0, "p90": 0.0},
        "per_frame": [],
        "hair_hand_guard": {
            "enabled": bool(cfg.hair_hand_guard_enabled),
            "detections_evaluated": 0,
            "detections_with_replacements": 0,
            "joints_replaced_total": 0,
            "replacement_reasons": {},
        },
    }
    all_margins: List[float] = []
    prev_selected_by_det: Dict[int, np.ndarray] = {}
    lock_state_by_det: Dict[int, Dict[str, int]] = {}

    for fidx in sorted(pretrack_pose_by_frame_det.keys()):
        frame_map = pretrack_pose_by_frame_det.get(fidx, {})
        boxes = all_detection_boxes.get(fidx, [])
        frame_out: Dict[int, Dict[str, Any]] = {}
        frame_ambiguous = 0
        frame_overrides = 0
        frame_rows: List[Dict[str, Any]] = []

        for di in sorted(frame_map.keys()):
            dual = frame_map[di]
            out_dual = _copy_dual(dual)
            diagnostics["detections_total"] += 1

            overlap_count = 0
            if di < len(boxes):
                b0 = boxes[di]
                for dj, bj in enumerate(boxes):
                    if dj == di:
                        continue
                    if _bbox_iou_xyxy(b0, bj) >= cfg.iou_ambiguous_thresh:
                        overlap_count += 1
            level = _overlap_level(overlap_count)
            diagnostics["bins"][level]["detections"] += 1

            vit = dual.get("vitpose_keypoints")
            rtm = dual.get("rtmwx_keypoints")
            v_mean = float(dual.get("vitpose_mean_conf", _mean_conf(vit)) or 0.0)
            r_mean = float(dual.get("rtmwx_mean_conf", _mean_conf(rtm)) or 0.0)
            ambiguous = (overlap_count > 0) or (v_mean < cfg.low_conf_thresh)

            selected = "vitpose"
            margin = 0.0
            if ambiguous and isinstance(vit, np.ndarray) and isinstance(rtm, np.ndarray):
                frame_ambiguous += 1
                diagnostics["detections_ambiguous"] += 1
                prev_kp = prev_selected_by_det.get(di)
                score_v = v_mean + cfg.temporal_weight * _temporal_consistency(vit, prev_kp)
                score_r = r_mean + cfg.temporal_weight * _temporal_consistency(rtm, prev_kp)
                margin = float(score_r - score_v)
                if margin >= cfg.min_margin:
                    selected = "rtmwx"
                    out_dual["phase35_selected"] = "rtmwx"
                    out_dual["phase35_score_margin"] = margin
                    out_dual["vitpose_keypoints"] = np.asarray(rtm, dtype=np.float32).copy()
                    out_dual["vitpose_mean_conf"] = float(r_mean)
                    frame_overrides += 1
                    diagnostics["overrides_total"] += 1
                    diagnostics["bins"][level]["overrides"] += 1
                else:
                    out_dual["phase35_selected"] = "vitpose"
                    out_dual["phase35_score_margin"] = margin
                all_margins.append(margin)
            else:
                out_dual["phase35_selected"] = "vitpose"
                out_dual["phase35_score_margin"] = 0.0

            hair_replaced = 0
            hair_reasons: List[str] = []
            if cfg.hair_hand_guard_enabled and isinstance(vit, np.ndarray):
                diagnostics["hair_hand_guard"]["detections_evaluated"] += 1
                prev_kp = prev_selected_by_det.get(di)
                lock_state = dict(lock_state_by_det.get(di, {"left": 0, "right": 0}))
                hair_replaced, lock_state, hair_reasons = _apply_hair_hand_replacements(
                    out_dual=out_dual,
                    vit=vit,
                    rtm=rtm if isinstance(rtm, np.ndarray) else None,
                    prev_selected=prev_kp,
                    cfg=cfg,
                    lock_remaining=lock_state,
                )
                lock_state_by_det[di] = lock_state
                if hair_replaced > 0:
                    diagnostics["hair_hand_guard"]["detections_with_replacements"] += 1
                    diagnostics["hair_hand_guard"]["joints_replaced_total"] += int(hair_replaced)
                    rr = diagnostics["hair_hand_guard"]["replacement_reasons"]
                    for reason in hair_reasons:
                        rr[reason] = int(rr.get(reason, 0) or 0) + 1
                    out_dual["phase35_hair_hand_guard_applied"] = True
                    out_dual["phase35_hair_hand_guard_replaced_joints"] = list(hair_reasons)
                else:
                    out_dual["phase35_hair_hand_guard_applied"] = False

            selected_kp = out_dual.get("vitpose_keypoints")
            if isinstance(selected_kp, np.ndarray):
                prev_selected_by_det[di] = np.asarray(selected_kp, dtype=np.float32).copy()

            frame_rows.append(
                {
                    "det_idx": int(di),
                    "selected": selected,
                    "ambiguous": bool(ambiguous),
                    "overlap_count": int(overlap_count),
                    "vitpose_mean_conf": float(v_mean),
                    "rtmwx_mean_conf": float(r_mean),
                    "margin": float(margin),
                    "level": level,
                    "hair_hand_replaced_joints": int(hair_replaced),
                }
            )
            frame_out[di] = out_dual

        if frame_ambiguous > 0:
            diagnostics["frames_with_ambiguity"] += 1
        diagnostics["per_frame"].append(
            {
                "frame_idx": int(fidx),
                "detections": int(len(frame_map)),
                "ambiguous": int(frame_ambiguous),
                "overrides": int(frame_overrides),
                "rows": frame_rows,
            }
        )
        resolved[fidx] = frame_out

    if all_margins:
        arr = np.asarray(all_margins, dtype=np.float32)
        diagnostics["margin_stats"] = {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    return resolved, diagnostics
