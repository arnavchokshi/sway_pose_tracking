"""
Extract approximate pinhole intrinsics from container metadata (ffprobe).

iPhones and some cameras embed a 35mm-equivalent focal length in QuickTime / MP4
tags. We map that to pixel focal lengths using full-frame horizontal sensor width
(36mm), matching common photography conventions.

Requires ``ffprobe`` on PATH (same as the rest of the pipeline). When no usable
tags exist, returns None and callers fall back to SWAY_PINHOLE_FOV_DEG.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from typing import Any, Dict, Iterator, Optional, Tuple

# Full-frame still format horizontal extent (mm); used with 35mm-equivalent focal length.
_FULL_FRAME_WIDTH_MM = 36.0

_TAG_PRIORITY_EXACT = (
    "com.apple.quicktime.camera.focal_length.35mm_equivalent",
    "camera.focal_length.35mm_equivalent",
)

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def video_intrinsics_probe_enabled() -> bool:
    """When False, skip ffprobe (always use FOV default or SWAY_FX/SWAY_FY)."""
    return _env_flag("SWAY_VIDEO_INTRINSICS_PROBE", True)


def _normalize_key(k: str) -> str:
    return k.strip().lower().replace(" ", "_")


def _first_number(s: str) -> Optional[float]:
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _iter_video_tag_pairs(ffprobe: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    fmt = ffprobe.get("format") or {}
    for k, v in (fmt.get("tags") or {}).items():
        if v is not None and str(v).strip():
            yield str(k), str(v)
    for st in ffprobe.get("streams") or []:
        if st.get("codec_type") != "video":
            continue
        for k, v in (st.get("tags") or {}).items():
            if v is not None and str(v).strip():
                yield str(k), str(v)


def _collect_tags_flat(ffprobe: Dict[str, Any]) -> Dict[str, str]:
    """Last writer wins (video stream tags typically override format tags)."""
    out: Dict[str, str] = {}
    for k, v in _iter_video_tag_pairs(ffprobe):
        out[k] = v
    return out


def _find_35mm_equivalent_mm(tags: Dict[str, str]) -> Tuple[Optional[float], Optional[str]]:
    norm_map = {_normalize_key(k): (k, v) for k, v in tags.items()}
    for exact in _TAG_PRIORITY_EXACT:
        e = exact.lower()
        if e in norm_map:
            raw = norm_map[e][1]
            n = _first_number(raw)
            if n is not None and n > 0.0:
                return n, norm_map[e][0]
    for nk, (orig_k, raw) in norm_map.items():
        if "35mm" in nk and "focal" in nk and "equivalent" in nk:
            n = _first_number(raw)
            if n is not None and n > 0.0:
                return n, orig_k
        if "focal" in nk and "35" in nk and ("equiv" in nk or "equivalent" in nk):
            n = _first_number(raw)
            if n is not None and n > 0.0:
                return n, orig_k
        if nk in ("focallengthin35mmformat", "focal_length_in_35mm_format"):
            n = _first_number(raw)
            if n is not None and n > 0.0:
                return n, orig_k
    return None, None


def fx_fy_from_35mm_equivalent(f_eq_mm: float, width: int, height: int) -> Tuple[float, float, float]:
    """Return (fx, fy, fov_deg) using horizontal full-frame FOV; square pixels (fy = fx * h/w)."""
    w, h = float(width), float(height)
    if f_eq_mm <= 0.0 or w <= 0.0 or h <= 0.0:
        raise ValueError("invalid dimensions or focal length")
    # Horizontal FOV of a full-frame 36×24mm sensor with focal length f_eq.
    hfov_rad = 2.0 * math.atan(_FULL_FRAME_WIDTH_MM / (2.0 * f_eq_mm))
    tan_half = math.tan(0.5 * hfov_rad)
    if tan_half <= 1e-9:
        raise ValueError("degenerate FOV")
    fx = 0.5 * w / tan_half
    fy = fx * (h / w)
    fov_deg = math.degrees(hfov_rad)
    return float(fx), float(fy), float(fov_deg)


def ffprobe_json(video_path: str, timeout_s: float = 15.0) -> Optional[Dict[str, Any]]:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [
                exe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        return json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def probe_intrinsics_from_video(
    video_path: str,
    frame_width: int,
    frame_height: int,
) -> Optional[Dict[str, Any]]:
    """
    If ffprobe finds a 35mm-equivalent focal length, return pinhole parameters.

    Returns dict with keys: fx, fy, cx, cy, fov_deg, source_key, method.
    """
    if not video_intrinsics_probe_enabled():
        return None
    data = ffprobe_json(video_path)
    if not data:
        return None
    tags = _collect_tags_flat(data)
    f_eq, source_key = _find_35mm_equivalent_mm(tags)
    if f_eq is None:
        return None
    try:
        fx, fy, fov_deg = fx_fy_from_35mm_equivalent(f_eq, frame_width, frame_height)
    except ValueError:
        return None
    cx = frame_width * 0.5
    cy = frame_height * 0.5
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fov_deg": fov_deg,
        "source_key": source_key,
        "focal_length_35mm_equiv_mm": float(f_eq),
        "method": "ffprobe_35mm_equiv_full_frame_horizontal",
    }
