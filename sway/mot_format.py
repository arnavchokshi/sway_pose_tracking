"""
MOTChallenge-format I/O for TrackEval: export pipeline JSON to tracker .txt rows.

MOT row (comma-separated): frame, id, x, y, w, h, conf, -1, -1, -1
(frame is 1-based in MOTChallenge; TrackEval uses same convention.)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sway.track_observation import coerce_observation

# raw_tracks: track_id -> list of (frame_idx_0based, box_xyxy, conf)
RawTrackEntry = Tuple[int, Tuple[float, float, float, float], float]


def raw_tracks_to_mot_lines(
    raw_tracks: Dict[int, List[RawTrackEntry]],
    *,
    as_mot_gt: bool = False,
) -> List[str]:
    """MOT lines from tracker output (same frame indexing as OpenCV frame_idx: 0-based → MOT 1-based).

    Use ``as_mot_gt=True`` when feeding TrackEval's GT reader (mark/conf/class/visibility must be 1).
    """
    lines: List[str] = []
    for tid, entries in raw_tracks.items():
        for f0, box, conf in entries:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            cf = float(conf) if conf is not None else 1.0
            lines.append(
                xyxy_to_mot_line(int(f0) + 1, int(tid), x1, y1, x2, y2, cf, is_gt=as_mot_gt)
            )
    lines.sort(key=lambda ln: (int(float(ln.split(",")[0])), int(float(ln.split(",")[1]))))
    return lines


def xyxy_to_mot_line(
    frame_1based: int,
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    conf: float = 1.0,
    *,
    is_gt: bool = False,
) -> str:
    """
    TrackEval MOTChallenge expects >=8 columns for GT (class at index 7 = pedestrian 1).
    Tracker rows: conf at 6, class 1 at 7 when 8+ columns.
    """
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if is_gt:
        return f"{frame_1based},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1"
    return f"{frame_1based},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},1,1"


def build_phase3_tracking_data_json(
    *,
    video_path: str,
    raw_tracks: Dict[int, List[RawTrackEntry]],
    total_frames: int,
    native_fps: float,
    output_fps: float,
) -> Dict[str, Any]:
    """
    Minimal ``data.json`` for runs that stop at ``after_phase_3`` (no pose / full export).

    Matches the shape consumed by :func:`data_json_to_mot_lines` so TrackEval / ``auto_sweep``
    can score without running Phase 4+.
    """
    per_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float], float]]] = defaultdict(
        list
    )
    for tid, entries in raw_tracks.items():
        for entry in entries:
            if entry is None:
                continue
            obs = coerce_observation(entry)
            f0, box, conf = int(obs.frame_idx), obs.bbox, float(obs.conf)
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            per_frame[f0].append((int(tid), (x1, y1, x2, y2), float(conf)))

    frames: List[Dict[str, Any]] = []
    for fi in range(int(total_frames)):
        rows = per_frame.get(fi, [])
        rows.sort(key=lambda x: x[0])
        tracks: Dict[str, Any] = {}
        for tid, box, conf in rows:
            tracks[str(tid)] = {
                "box": [box[0], box[1], box[2], box[3]],
                "confidence": conf,
            }
        frames.append({"frame_idx": fi, "tracks": tracks})

    return {
        "metadata": {
            "video_path": video_path,
            "fps": float(output_fps),
            "native_fps": float(native_fps),
            "num_frames": int(total_frames),
            "export_kind": "tracking_only_after_phase_3",
        },
        "track_summaries": {},
        "frames": frames,
    }


def data_json_to_mot_lines(data: Dict[str, Any]) -> List[str]:
    """Build MOT lines from sway data.json (frames[].tracks[].box)."""
    lines: List[str] = []
    frames = data.get("frames") or []
    for fr in frames:
        f0 = int(fr.get("frame_idx", 0))
        f1 = f0 + 1  # MOT 1-based
        tracks = fr.get("tracks") or {}
        for tid_str, tdata in tracks.items():
            try:
                tid = int(tid_str)
            except (TypeError, ValueError):
                continue
            box = tdata.get("box")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            conf = float(tdata.get("confidence", tdata.get("conf", 1.0)))
            lines.append(xyxy_to_mot_line(f1, tid, x1, y1, x2, y2, conf, is_gt=False))
    return lines


def write_mot_file(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_mot_lines_from_file(path: Path) -> List[str]:
    text = path.read_text().strip()
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def mot_lines_to_seq_info(lines: List[str]) -> Tuple[int, int]:
    """Return (max_frame_1based, max_id) from MOT lines."""
    max_f, max_id = 0, 0
    for ln in lines:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            max_f = max(max_f, int(float(parts[0])))
            max_id = max(max_id, int(float(parts[1])))
        except ValueError:
            continue
    return max_f, max_id
