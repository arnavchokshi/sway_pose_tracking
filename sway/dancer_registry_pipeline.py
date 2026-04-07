"""
Optional Phases 1–3 enhancement: Dancer Registry + post-crossover verify (no SAM at overlap).

Enabled when ``SWAY_PHASE13_MODE=dancer_registry`` (Pipeline Lab preset / env).

1. **Warm-up profiles (~10s):** zonal HSV histograms (torso / legs split) + bbox aspect ratio,
   updated only when the track is spatially isolated (no other box center within 1.5× body scale).
2. **Crossover monitor:** while Deep OC-SORT runs as usual, a video pass detects pairs that overlapped
   then separated cleanly; compares current appearance to locked profiles and may **swap** track IDs
   retroactively over the crossover window.
3. **Dormant relink:** after motion-based dormant merge (``apply_dormant_merges``), tries appearance
   matches for gaps that motion did not merge.

Requires one sequential decode of the source video after tracking (no SAM, histograms only).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sway.track_observation import TrackObservation, coerce_observation

TrackEntry = Any  # TrackObservation or legacy 3-tuple


def phase13_dancer_registry_enabled() -> bool:
    v = os.environ.get("SWAY_PHASE13_MODE", "").strip().lower()
    return v in ("dancer_registry", "registry", "1", "true", "yes")


def _box_w(box: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(box[2] - box[0]))


def _box_h(box: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(box[3] - box[1]))


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return float(inter / u) if u > 0 else 0.0


def _edge_distance(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
    dy = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return float(np.hypot(dx, dy))


def _is_isolated(
    self_box: Tuple[float, float, float, float],
    others: List[Tuple[float, float, float, float]],
    *,
    isolation_mult: float,
) -> bool:
    sc = _center(self_box)
    scale = max(_box_w(self_box), _box_h(self_box))
    thr = isolation_mult * scale
    for ob in others:
        oc = _center(ob)
        if float(np.hypot(sc[0] - oc[0], sc[1] - oc[1])) < thr:
            return False
    return True


def _extract_zonal_features(
    frame_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
    *,
    fw: int,
    fh: int,
) -> Optional[Tuple[np.ndarray, float]]:
    """Returns (32-dim normalized HSV-H histogram concat top|bottom, aspect w/h)."""
    x1 = int(max(0, min(fw - 1, box[0])))
    y1 = int(max(0, min(fh - 1, box[1])))
    x2 = int(max(0, min(fw, box[2])))
    y2 = int(max(0, min(fh, box[3])))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ww = hsv.shape[:2]
    mid = max(1, hh // 2)
    top = hsv[:mid, :]
    bot = hsv[mid:, :]
    ht = cv2.calcHist([top], [0], None, [16], [0, 180])
    hb = cv2.calcHist([bot], [0], None, [16], [0, 180])
    cv2.normalize(ht, ht, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1.0, 0.0, cv2.NORM_L1)
    feat = np.concatenate([ht.flatten(), hb.flatten()]).astype(np.float64)
    s = float(feat.sum())
    if s > 1e-9:
        feat /= s
    aspect = float(ww) / float(max(hh, 1))
    return feat, aspect


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))))


def _aspect_sim(a: float, b: float) -> float:
    da = abs(np.log((a + 1e-6) / (b + 1e-6)))
    return float(np.exp(-da))


def _profile_score(feat: np.ndarray, asp: float, prof: Optional[Tuple[np.ndarray, float]]) -> float:
    if prof is None:
        return 0.0
    pf, pa = prof
    return _bhattacharyya(feat, pf) * 0.85 + _aspect_sim(asp, pa) * 0.15


def _build_frame_map(
    raw_tracks: Dict[int, List[TrackEntry]],
) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    out: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    for tid, entries in raw_tracks.items():
        for ent in entries:
            obs = coerce_observation(ent)
            fi = int(obs.frame_idx)
            out.setdefault(fi, {})[int(tid)] = tuple(float(x) for x in obs.bbox[:4])
    return out


def _entry_frame(ent: TrackEntry) -> int:
    return int(coerce_observation(ent).frame_idx)


def _swap_track_interval(
    raw_tracks: Dict[int, List[TrackEntry]],
    id_a: int,
    id_b: int,
    f0: int,
    f1: int,
) -> None:
    """Move observations in [f0,f1] between id_a and id_b."""
    if id_a == id_b or id_a not in raw_tracks or id_b not in raw_tracks:
        return
    la = raw_tracks[id_a]
    lb = raw_tracks[id_b]
    from_a = [e for e in la if f0 <= _entry_frame(e) <= f1]
    from_b = [e for e in lb if f0 <= _entry_frame(e) <= f1]
    if not from_a and not from_b:
        return
    raw_tracks[id_a] = [e for e in la if e not in from_a]
    raw_tracks[id_b] = [e for e in lb if e not in from_b]
    raw_tracks[id_a].extend(from_b)
    raw_tracks[id_b].extend(from_a)
    raw_tracks[id_a].sort(key=_entry_frame)
    raw_tracks[id_b].sort(key=_entry_frame)


def apply_dancer_registry_crossover_pass(
    raw_tracks: Dict[int, List[TrackEntry]],
    video_path: str,
    total_frames: int,
    frame_width: int,
    frame_height: int,
    native_fps: float,
) -> Dict[int, List[TrackEntry]]:
    """
    Single video scan: build/update appearance profiles, detect overlap→separation events,
    optionally swap IDs. Mutates and returns ``raw_tracks``.
    """
    if total_frames <= 0 or not raw_tracks:
        return raw_tracks

    warmup_sec = float(os.environ.get("SWAY_REGISTRY_WARMUP_SEC", "10").strip() or "10")
    isolation_mult = float(os.environ.get("SWAY_REGISTRY_ISOLATION_MULT", "1.5").strip() or "1.5")
    touch_iou = float(os.environ.get("SWAY_REGISTRY_TOUCH_IOU", "0.12").strip() or "0.12")
    clear_iou = float(os.environ.get("SWAY_REGISTRY_CLEAR_IOU", "0.02").strip() or "0.02")
    sep_mult = float(os.environ.get("SWAY_REGISTRY_SEPARATION_MULT", "1.5").strip() or "1.5")
    swap_margin = float(os.environ.get("SWAY_REGISTRY_SWAP_MARGIN", "0.06").strip() or "0.06")

    warmup_end = int(min(total_frames - 1, max(1, round(warmup_sec * max(native_fps, 1.0)))))

    frame_map = _build_frame_map(raw_tracks)
    # Locked profile: tid -> (feat32, aspect) after warmup; during warmup we use running accumulators
    locked: Dict[int, Tuple[np.ndarray, float]] = {}
    accum_n: Dict[int, int] = {}
    accum_feat: Dict[int, np.ndarray] = {}
    accum_asp: Dict[int, float] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [dancer_registry] Video open failed; skipping registry pass.", flush=True)
        return raw_tracks

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame_width)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame_height)

    # pair -> { touching, start_frame, last_overlap_f }
    pair_touch: Dict[Tuple[int, int], Dict[str, Any]] = {}

    fidx = 0
    swaps = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if fidx >= total_frames:
            break

        boxes_by_tid = frame_map.get(fidx, {})
        tids = sorted(boxes_by_tid.keys())

        # --- profile updates (isolated only) ---
        for tid in tids:
            box = boxes_by_tid[tid]
            others = [boxes_by_tid[j] for j in tids if j != tid]
            if not _is_isolated(box, others, isolation_mult=isolation_mult):
                continue
            zf = _extract_zonal_features(frame_bgr, box, fw=fw, fh=fh)
            if zf is None:
                continue
            feat, asp = zf
            if fidx <= warmup_end:
                n = accum_n.get(tid, 0) + 1
                accum_n[tid] = n
                if tid not in accum_feat:
                    accum_feat[tid] = feat.copy()
                    accum_asp[tid] = asp
                else:
                    accum_feat[tid] += feat
                    accum_asp[tid] += asp
                inv = 1.0 / float(n)
                lf = accum_feat[tid] * inv
                s = float(lf.sum())
                if s > 1e-9:
                    lf /= s
                locked[tid] = (lf.astype(np.float64), accum_asp[tid] * inv)
            else:
                if tid not in locked:
                    locked[tid] = (feat.copy(), asp)
                else:
                    pf, pa = locked[tid]
                    # EMA when isolated after warm-up (soft update)
                    alpha = 0.08
                    nf = (1 - alpha) * pf + alpha * feat
                    ns = float(nf.sum())
                    if ns > 1e-9:
                        nf /= ns
                    na = (1 - alpha) * pa + alpha * asp
                    locked[tid] = (nf.astype(np.float64), float(na))

        # --- pair crossover ---
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_i, tid_j = int(tids[i]), int(tids[j])
                ba = boxes_by_tid[tid_i]
                bb = boxes_by_tid[tid_j]
                key = (tid_i, tid_j) if tid_i < tid_j else (tid_j, tid_i)
                iou = _iou(ba, bb)
                st = pair_touch.get(key)
                touching = st is not None and bool(st.get("touching"))

                if iou >= touch_iou:
                    if not touching:
                        pair_touch[key] = {"touching": True, "start_f": fidx, "last_o": fidx}
                    else:
                        st["last_o"] = fidx
                elif touching and iou <= clear_iou:
                    wavg = 0.5 * (_box_w(ba) + _box_w(bb))
                    if _edge_distance(ba, bb) >= sep_mult * max(8.0, wavg):
                        f0 = int(st.get("start_f", fidx))
                        a, b = key[0], key[1]
                        za = _extract_zonal_features(frame_bgr, ba, fw=fw, fh=fh)
                        zb = _extract_zonal_features(frame_bgr, bb, fw=fw, fh=fh)
                        pa_l = locked.get(a)
                        pb_l = locked.get(b)
                        if za and zb and pa_l and pb_l:
                            fa, asa = za
                            fb, asb = zb
                            s_same = _profile_score(fa, asa, pa_l) + _profile_score(fb, asb, pb_l)
                            s_swap = _profile_score(fa, asa, pb_l) + _profile_score(fb, asb, pa_l)
                            if s_swap > s_same + swap_margin:
                                _swap_track_interval(raw_tracks, a, b, f0, fidx)
                                frame_map = _build_frame_map(raw_tracks)
                                swaps += 1
                                if a in locked and b in locked:
                                    locked[a], locked[b] = locked[b], locked[a]
                                if a in accum_feat and b in accum_feat:
                                    accum_feat[a], accum_feat[b] = (
                                        accum_feat[b].copy(),
                                        accum_feat[a].copy(),
                                    )
                                    accum_n[a], accum_n[b] = accum_n[b], accum_n[a]
                                    accum_asp[a], accum_asp[b] = accum_asp[b], accum_asp[a]
                        pair_touch[key] = {"touching": False, "start_f": -1, "last_o": -1}

        fidx += 1

    cap.release()
    if swaps:
        print(f"  [dancer_registry] Crossover verify: {swaps} ID swap(s) applied.", flush=True)
    return raw_tracks


def apply_dancer_registry_appearance_dormant(
    raw_tracks: Dict[int, List[TrackEntry]],
    video_path: str,
    total_frames: int,
    _native_fps: float,
) -> Dict[int, List[TrackEntry]]:
    """
    After motion dormant merges: relink segments using zonal appearance when motion match failed.
    One extra video scan to sample first isolated frame of candidate new segments.
    """
    if total_frames <= 0 or not raw_tracks:
        return raw_tracks

    track_buffer = 90
    max_gap = int(os.environ.get("SWAY_DORMANT_MAX_GAP", "150") or "150")
    match_thr = float(os.environ.get("SWAY_REGISTRY_DORMANT_MATCH", "0.82").strip() or "0.82")
    isolation_mult = float(os.environ.get("SWAY_REGISTRY_ISOLATION_MULT", "1.5").strip() or "1.5")

    # Motion dormant already ran inside apply_post_track_stitching; re-run would be wrong.
    # This pass only merges pairs that satisfy gap + appearance, mimicking dormant_tracks loop.

    def seg_meta(tid: int):
        ent = sorted(raw_tracks.get(tid, []), key=_entry_frame)
        if not ent:
            return None
        return {"tid": tid, "start": _entry_frame(ent[0]), "end": _entry_frame(ent[-1]), "entries": ent}

    frame_map = _build_frame_map(raw_tracks)
    seg_end: Dict[int, int] = {}
    seg_start: Dict[int, int] = {}
    for tid, ents in raw_tracks.items():
        if not ents:
            continue
        s = sorted(ents, key=_entry_frame)
        seg_start[int(tid)] = _entry_frame(s[0])
        seg_end[int(tid)] = _entry_frame(s[-1])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return raw_tracks
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    first_feat: Dict[int, Tuple[np.ndarray, float]] = {}
    last_feat: Dict[int, Tuple[np.ndarray, float]] = {}

    fidx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or fidx >= total_frames:
            break
        boxes_by_tid = frame_map.get(fidx, {})
        for tid, box in boxes_by_tid.items():
            tid = int(tid)
            others = [boxes_by_tid[j] for j in boxes_by_tid if j != tid]
            if not _is_isolated(box, others, isolation_mult=isolation_mult):
                continue
            zf = _extract_zonal_features(frame_bgr, box, fw=fw, fh=fh)
            if not zf:
                continue
            feat, asp = zf[0].copy(), zf[1]
            st = seg_start.get(tid, 0)
            en = seg_end.get(tid, total_frames)
            if tid not in first_feat and fidx >= st:
                first_feat[tid] = (feat.copy(), asp)
            if st <= fidx <= en:
                last_feat[tid] = (feat.copy(), asp)
        fidx += 1
    cap.release()

    changed = True
    merges = 0
    while changed:
        changed = False
        metas = [seg_meta(t) for t in list(raw_tracks.keys())]
        metas = [m for m in metas if m is not None]
        metas.sort(key=lambda m: m["start"])
        for A in metas:
            if A["tid"] not in raw_tracks:
                continue
            ent_a = sorted(raw_tracks[A["tid"]], key=_entry_frame)
            if len(ent_a) < 1:
                continue
            end_a = _entry_frame(ent_a[-1])
            fa_end = last_feat.get(A["tid"])
            if fa_end is None:
                fa_end = first_feat.get(A["tid"])
            if fa_end is None:
                continue
            for B in metas:
                if B["tid"] == A["tid"] or B["tid"] not in raw_tracks or A["tid"] not in raw_tracks:
                    continue
                if B["start"] <= end_a:
                    continue
                gap = B["start"] - end_a - 1
                if gap <= track_buffer or gap > max_gap:
                    continue
                fb0 = first_feat.get(B["tid"])
                if fb0 is None:
                    continue
                sc = _profile_score(fb0[0], fb0[1], fa_end)
                if sc >= match_thr:
                    merged_list = ent_a + sorted(raw_tracks[B["tid"]], key=_entry_frame)
                    raw_tracks[A["tid"]] = merged_list
                    del raw_tracks[B["tid"]]
                    first_feat.pop(B["tid"], None)
                    last_feat.pop(B["tid"], None)
                    frame_map = _build_frame_map(raw_tracks)
                    merges += 1
                    changed = True
                    break
            if changed:
                break

    if merges:
        print(f"  [dancer_registry] Appearance dormant: {merges} segment merge(s).", flush=True)
    return raw_tracks
