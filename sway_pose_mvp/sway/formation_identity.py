from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FormationPerformer:
    performer_id: str
    name: str


@dataclass
class FormationTimeline:
    width: float
    depth: float
    formations: List[Dict[str, Tuple[float, float]]]
    formation_names: List[str]
    formation_durations: List[float]
    formation_start_times: List[float]
    performers: List[FormationPerformer]
    music_url: str


@dataclass
class ModeDecision:
    detected_mode: str
    final_mode: str
    confidence: float
    reason: str
    confirmation_required: bool


@dataclass
class StartAlignment:
    start_offset_sec: float
    start_formation_index: int
    spatial_confidence: float
    audio_confidence: float
    selected_flip_x: bool
    reason: str


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_xy(x: float, y: float, width: float, depth: float, flip_x: bool) -> Tuple[float, float]:
    w = max(1e-6, float(width))
    d = max(1e-6, float(depth))
    nx = max(0.0, min(1.0, x / w))
    ny = max(0.0, min(1.0, y / d))
    if flip_x:
        nx = 1.0 - nx
    return nx, ny


def load_formation_timeline(path: str) -> Optional[FormationTimeline]:
    p = Path(path)
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    width = _safe_float(raw.get("width"), 48.0)
    depth = _safe_float(raw.get("depth"), 27.0)
    formation_rows = raw.get("formations") or []
    names = [str(x) for x in (raw.get("formationNames") or [])]
    durations = [_safe_float(x, 5.0) for x in (raw.get("formationDurations") or [])]
    if not durations:
        durations = [5.0 for _ in formation_rows]
    if len(durations) < len(formation_rows):
        durations.extend([durations[-1] if durations else 5.0] * (len(formation_rows) - len(durations)))
    elif len(durations) > len(formation_rows):
        durations = durations[: len(formation_rows)]

    performer_name_by_id: Dict[str, str] = {}
    for row in formation_rows:
        for node in row:
            user = node.get("user") if isinstance(node, dict) else None
            if not isinstance(user, dict):
                continue
            oid = str(user.get("$oid") or "").strip()
            if not oid:
                continue
            nm = str(node.get("name") or node.get("label") or "").strip()
            if nm:
                performer_name_by_id[oid] = nm
            elif oid not in performer_name_by_id:
                performer_name_by_id[oid] = oid

    performers = [
        FormationPerformer(performer_id=pid, name=performer_name_by_id[pid])
        for pid in sorted(performer_name_by_id.keys())
    ]
    perf_set = set(performer_name_by_id.keys())
    parsed_formations: List[Dict[str, Tuple[float, float]]] = []
    for row in formation_rows:
        fmap: Dict[str, Tuple[float, float]] = {}
        for node in row:
            user = node.get("user") if isinstance(node, dict) else None
            if not isinstance(user, dict):
                continue
            oid = str(user.get("$oid") or "").strip()
            if not oid or oid not in perf_set:
                continue
            x = _safe_float(node.get("x"), 0.0)
            y = _safe_float(node.get("y"), 0.0)
            fmap[oid] = (x, y)
        parsed_formations.append(fmap)

    start_times: List[float] = []
    t = 0.0
    for dur in durations:
        start_times.append(t)
        t += max(0.0, float(dur))

    music_url = str(raw.get("musicUrl") or "").strip()
    return FormationTimeline(
        width=width,
        depth=depth,
        formations=parsed_formations,
        formation_names=names,
        formation_durations=durations,
        formation_start_times=start_times,
        performers=performers,
        music_url=music_url,
    )


def _estimate_motion_score(
    detections_by_frame: Dict[int, Sequence[Tuple[float, float, float, float]]],
    width: int,
    height: int,
    max_frames: int = 180,
) -> float:
    frames = sorted(int(k) for k in detections_by_frame.keys())[:max_frames]
    if len(frames) < 2:
        return 0.0
    disp_values: List[float] = []
    for f0, f1 in zip(frames[:-1], frames[1:]):
        b0 = detections_by_frame.get(f0) or []
        b1 = detections_by_frame.get(f1) or []
        if not b0 or not b1:
            continue
        c0 = np.array([[(a + c) * 0.5 / max(width, 1), (b + d) * 0.5 / max(height, 1)] for a, b, c, d in b0], dtype=np.float32)
        c1 = np.array([[(a + c) * 0.5 / max(width, 1), (b + d) * 0.5 / max(height, 1)] for a, b, c, d in b1], dtype=np.float32)
        for p in c0:
            dists = np.linalg.norm(c1 - p[None, :], axis=1)
            if dists.size:
                disp_values.append(float(np.min(dists)))
    if not disp_values:
        return 0.0
    return float(np.median(np.asarray(disp_values, dtype=np.float32)))


def detect_recording_mode(
    timeline: Optional[FormationTimeline],
    detections_by_frame: Dict[int, Sequence[Tuple[float, float, float, float]]],
    width: int,
    height: int,
    policy: str = "auto_then_confirm",
    override_mode: str = "",
) -> ModeDecision:
    override = (override_mode or "").strip().lower()
    if override in {"formation", "windowed"}:
        return ModeDecision(
            detected_mode=override,
            final_mode=override,
            confidence=1.0,
            reason="explicit_override",
            confirmation_required=False,
        )
    if timeline is None:
        return ModeDecision(
            detected_mode="windowed",
            final_mode="windowed",
            confidence=0.95,
            reason="formation_data_missing",
            confirmation_required=False,
        )

    motion = _estimate_motion_score(detections_by_frame, width, height)
    if motion >= 0.03:
        detected, conf, reason = "formation", min(0.98, 0.55 + motion * 5.0), "early_motion_signature"
    else:
        detected, conf, reason = "windowed", min(0.98, 0.65 + (0.03 - motion) * 5.0), "low_motion_signature"

    pol = (policy or "auto_then_confirm").strip().lower()
    if pol not in {"auto_then_confirm", "fully_auto", "explicit_user_choice"}:
        pol = "auto_then_confirm"
    if pol == "explicit_user_choice":
        final_mode = "windowed"
        conf = 0.5
    else:
        final_mode = detected
    return ModeDecision(
        detected_mode=detected,
        final_mode=final_mode,
        confidence=float(max(0.0, min(1.0, conf))),
        reason=reason,
        confirmation_required=(pol == "auto_then_confirm"),
    )


def expected_positions_at(timeline: FormationTimeline, absolute_sec: float) -> Dict[str, Tuple[float, float]]:
    if not timeline.formations:
        return {}
    t = max(0.0, float(absolute_sec))
    idx = len(timeline.formations) - 1
    for i, st in enumerate(timeline.formation_start_times):
        dur = timeline.formation_durations[i] if i < len(timeline.formation_durations) else 0.0
        if st <= t < st + max(1e-6, float(dur)):
            idx = i
            break
    return timeline.formations[idx]


def _frame_observed_centers(
    frame_tracks: Sequence[Tuple[float, ...]],
    width: int,
    height: int,
) -> Tuple[List[int], np.ndarray]:
    tids: List[int] = []
    obs: List[Tuple[float, float]] = []
    for tr in frame_tracks:
        if len(tr) < 5:
            continue
        x1, y1, x2, y2, tid = tr[0], tr[1], tr[2], tr[3], tr[4]
        if x2 - x1 < 8 or y2 - y1 < 16:
            continue
        tids.append(int(tid))
        obs.append(((float(x1) + float(x2)) * 0.5 / max(width, 1), (float(y1) + float(y2)) * 0.5 / max(height, 1)))
    if not obs:
        return tids, np.zeros((0, 2), dtype=np.float32)
    return tids, np.asarray(obs, dtype=np.float32)


def _match_cost(
    observed_xy: np.ndarray,
    expected_xy: np.ndarray,
    max_dist: float,
) -> Tuple[float, float]:
    if observed_xy.size == 0 or expected_xy.size == 0:
        return 1.0, 0.0
    cost = np.zeros((observed_xy.shape[0], expected_xy.shape[0]), dtype=np.float32)
    for i in range(observed_xy.shape[0]):
        d = np.linalg.norm(expected_xy - observed_xy[i : i + 1], axis=1)
        cost[i] = d
    rows, cols = linear_sum_assignment(cost)
    dsel = cost[rows, cols]
    in_gate = dsel <= max_dist
    cov = float(np.mean(in_gate)) if dsel.size else 0.0
    if not np.any(in_gate):
        return 1.0, cov
    return float(np.mean(dsel[in_gate])), cov


def estimate_start_offset_spatial(
    timeline: FormationTimeline,
    all_track_results: Sequence[Sequence[Tuple[float, float, float, float, int]]],
    width: int,
    height: int,
    fps: float,
    start_index_override: Optional[int] = None,
    max_samples: int = 120,
    frame_step: int = 5,
    max_dist: float = 0.22,
) -> StartAlignment:
    if not timeline.formations:
        return StartAlignment(0.0, 0, 0.0, 0.0, False, "empty_formation_timeline")
    if start_index_override is not None:
        idx = int(max(0, min(len(timeline.formations) - 1, int(start_index_override))))
        return StartAlignment(
            start_offset_sec=float(timeline.formation_start_times[idx]),
            start_formation_index=idx,
            spatial_confidence=1.0,
            audio_confidence=0.0,
            selected_flip_x=False,
            reason="start_index_override",
        )

    frame_ids = list(range(0, min(len(all_track_results), max_samples), max(1, frame_step)))
    if not frame_ids:
        return StartAlignment(0.0, 0, 0.0, 0.0, False, "no_frames_available")

    best = None
    second = None
    for idx in range(len(timeline.formations)):
        start_sec = float(timeline.formation_start_times[idx])
        for flip_x in (False, True):
            errs: List[float] = []
            covs: List[float] = []
            for fidx in frame_ids:
                tids, obs = _frame_observed_centers(all_track_results[fidx], width, height)
                if not tids:
                    continue
                exp_map = expected_positions_at(timeline, start_sec + (float(fidx) / max(1e-6, fps)))
                exp_xy: List[Tuple[float, float]] = []
                for performer in timeline.performers:
                    if performer.performer_id in exp_map:
                        ex, ey = exp_map[performer.performer_id]
                        exp_xy.append(_normalize_xy(ex, ey, timeline.width, timeline.depth, flip_x))
                if not exp_xy:
                    continue
                err, cov = _match_cost(obs, np.asarray(exp_xy, dtype=np.float32), max_dist=max_dist)
                errs.append(err)
                covs.append(cov)
            if not errs:
                continue
            mean_err = float(np.mean(errs))
            mean_cov = float(np.mean(covs)) if covs else 0.0
            # Lower is better.
            score = mean_err + (1.0 - mean_cov) * 0.4
            rec = (score, idx, flip_x, mean_err, mean_cov)
            if best is None or score < best[0]:
                second = best
                best = rec
            elif second is None or score < second[0]:
                second = rec

    if best is None:
        return StartAlignment(0.0, 0, 0.0, 0.0, False, "spatial_alignment_failed")

    margin = 0.05 if second is None else max(0.0, float(second[0] - best[0]))
    conf = max(0.0, min(1.0, 0.4 + margin * 8.0 + (1.0 - best[3]) * 0.3 + best[4] * 0.3))
    idx = int(best[1])
    return StartAlignment(
        start_offset_sec=float(timeline.formation_start_times[idx]),
        start_formation_index=idx,
        spatial_confidence=float(conf),
        audio_confidence=0.0,
        selected_flip_x=bool(best[2]),
        reason="spatial_alignment",
    )


def _load_mono_audio(path: Path, sr: int = 22050, max_seconds: float = 120.0) -> Optional[np.ndarray]:
    try:
        import librosa  # type: ignore
    except Exception:
        return None
    if not path.is_file():
        return None
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_seconds)
    except Exception:
        return None
    if y is None or len(y) < 512:
        return None
    return y.astype(np.float32)


def _video_audio_wav(video_path: Path, sr: int = 22050) -> Optional[Path]:
    ffmpeg_bin = os.environ.get("SWAY_FFMPEG_BIN", "ffmpeg")
    tmp = Path(tempfile.gettempdir()) / f"sway_video_audio_{os.getpid()}.wav"
    cmd = (
        f"{ffmpeg_bin} -y -i \"{video_path}\" -vn -ac 1 -ar {int(sr)} "
        f"-loglevel error \"{tmp}\""
    )
    rc = os.system(cmd)
    if rc != 0 or not tmp.is_file():
        return None
    return tmp


def estimate_audio_offset(
    video_path: Path,
    timeline: FormationTimeline,
    sr: int = 22050,
) -> Tuple[Optional[float], float, str]:
    ref_override = os.environ.get("SWAY_FORMATION_AUDIO_PATH", "").strip()
    ref_path: Optional[Path] = None
    cleanup_ref = False
    if ref_override:
        p = Path(ref_override).expanduser()
        if p.is_file():
            ref_path = p
    if ref_path is None and timeline.music_url:
        # Optional remote fetch; errors are tolerated.
        try:
            fd, tmp_name = tempfile.mkstemp(prefix="sway_ref_audio_", suffix=".m4a")
            os.close(fd)
            urllib.request.urlretrieve(timeline.music_url, tmp_name)
            ref_path = Path(tmp_name)
            cleanup_ref = True
        except Exception:
            ref_path = None
    vid_wav = _video_audio_wav(video_path, sr=sr)
    if vid_wav is None:
        return None, 0.0, "video_audio_extract_failed"
    vid_y = _load_mono_audio(vid_wav, sr=sr)
    ref_y = _load_mono_audio(ref_path, sr=sr) if ref_path is not None else None
    try:
        vid_wav.unlink(missing_ok=True)
    except OSError:
        pass
    if cleanup_ref and ref_path is not None:
        try:
            ref_path.unlink(missing_ok=True)
        except OSError:
            pass
    if vid_y is None or ref_y is None:
        return None, 0.0, "audio_unavailable"

    # Simple normalized cross-correlation.
    v = vid_y - float(np.mean(vid_y))
    r = ref_y - float(np.mean(ref_y))
    if np.std(v) < 1e-8 or np.std(r) < 1e-8:
        return None, 0.0, "audio_low_variance"
    corr = np.correlate(r, v, mode="valid")
    if corr.size == 0:
        return None, 0.0, "audio_no_overlap"
    best_idx = int(np.argmax(corr))
    max_corr = float(corr[best_idx])
    norm = float(np.linalg.norm(v) * np.linalg.norm(r[: len(v)]) + 1e-8)
    conf = max(0.0, min(1.0, max_corr / norm))
    return float(best_idx / float(sr)), conf, "audio_correlation"


def fuse_start_alignment(
    spatial: StartAlignment,
    audio_offset_sec: Optional[float],
    audio_confidence: float,
) -> StartAlignment:
    if audio_offset_sec is None or audio_confidence <= 0.0:
        return spatial
    ws = max(0.05, float(spatial.spatial_confidence))
    wa = max(0.0, float(audio_confidence))
    fused = (spatial.start_offset_sec * ws + float(audio_offset_sec) * wa) / (ws + wa)
    return StartAlignment(
        start_offset_sec=float(fused),
        start_formation_index=spatial.start_formation_index,
        spatial_confidence=float(spatial.spatial_confidence),
        audio_confidence=float(audio_confidence),
        selected_flip_x=bool(spatial.selected_flip_x),
        reason="spatial_audio_fused",
    )


def build_formation_assignments(
    timeline: FormationTimeline,
    all_track_results: Sequence[Sequence[Tuple[float, float, float, float, int]]],
    width: int,
    height: int,
    fps: float,
    start_offset_sec: float,
    flip_x: bool,
    max_match_dist: float = 0.22,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, str], Dict[str, float]]:
    performer_ids = [p.performer_id for p in timeline.performers]
    did_for_perf = {pid: idx + 1 for idx, pid in enumerate(performer_ids)}
    label_map = {did_for_perf[p.performer_id]: (p.name or p.performer_id) for p in timeline.performers}

    assignments_by_frame: Dict[int, Dict[int, int]] = {}
    track_to_perf: Dict[int, str] = {}
    perf_to_track: Dict[str, int] = {}
    reentry_reused = 0
    assigned_pairs = 0

    for fidx, tracks in enumerate(all_track_results):
        frame_map: Dict[int, int] = {}
        tids, obs = _frame_observed_centers(tracks, width, height)
        exp_map = expected_positions_at(timeline, float(start_offset_sec) + float(fidx) / max(1e-6, fps))
        exp_ids = [pid for pid in performer_ids if pid in exp_map]
        exp_xy = np.asarray(
            [_normalize_xy(exp_map[pid][0], exp_map[pid][1], timeline.width, timeline.depth, flip_x) for pid in exp_ids],
            dtype=np.float32,
        )

        if obs.size > 0 and exp_xy.size > 0:
            cost = np.zeros((obs.shape[0], exp_xy.shape[0]), dtype=np.float32)
            for i in range(obs.shape[0]):
                d = np.linalg.norm(exp_xy - obs[i : i + 1], axis=1)
                cost[i] = d
            # Track continuity bonus (as lower cost) for currently linked pairs.
            for i, tid in enumerate(tids):
                prev_pid = track_to_perf.get(int(tid))
                if prev_pid is None:
                    continue
                if prev_pid in exp_ids:
                    j = exp_ids.index(prev_pid)
                    cost[i, j] = max(0.0, cost[i, j] - 0.10)
            rows, cols = linear_sum_assignment(cost)
            used_perf: set = set()
            for r, c in zip(rows, cols):
                dist = float(cost[r, c])
                if dist > max_match_dist:
                    continue
                tid = int(tids[r])
                pid = exp_ids[c]
                if pid in used_perf:
                    continue
                used_perf.add(pid)
                prev_pid = track_to_perf.get(tid)
                if prev_pid is not None and prev_pid == pid:
                    reentry_reused += 1
                track_to_perf[tid] = pid
                perf_to_track[pid] = tid
                frame_map[tid] = did_for_perf[pid]
                assigned_pairs += 1
        assignments_by_frame[int(fidx)] = frame_map

    diag = {
        "assigned_pairs": float(assigned_pairs),
        "reentry_reused": float(reentry_reused),
        "performer_cap": float(len(performer_ids)),
        "start_offset_sec": float(start_offset_sec),
    }
    return assignments_by_frame, label_map, diag
