#!/usr/bin/env python3
"""
End-to-end pipeline validation: run ``main.py``, assert all 11 phases complete, check outputs.

**Depth (default on):**
  - Parses **per-phase wall times** from logs (Phases 1–2 combined, then 3…11).
  - **Deep JSON checks** on ``data.json`` (frame_idx coherence, finite keypoints, track ID charset).
  - **Prune / export coherence** (``prune_log`` vs ``data.json``).
  - **Failure patterns**: Traceback, CUDA OOM hints, etc.
  - **Optimization hints** if one phase dominates total wall time.
  - Optional **JSON report** per run (``--write-reports``).

**Single run (default):** synthetic MP4 + fast **smoke** preset (unless ``--full``).

**Matrix:** ``--matrix-file`` / ``--builtin-matrix`` / ``--extended-matrix`` (many built-in presets).

Examples::

  python -m tools.validate_pipeline_e2e
  python -m tools.validate_pipeline_e2e clip.mp4 --full --write-reports
  python -m tools.validate_pipeline_e2e --extended-matrix --timeout 7200
  python -m tools.validate_pipeline_e2e --matrix-file benchmarks/pipeline_validate_matrix.example.yaml --no-deep

Exit code: 0 = all runs passed, 1 = failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.technology_contracts import validate_run_against_contracts  # noqa: E402


def _extract_stop_boundary(extra_args: List[str]) -> str:
    for i, a in enumerate(extra_args):
        if a == "--stop-after-boundary" and i + 1 < len(extra_args):
            return str(extra_args[i + 1]).strip()
    return "final"

PHASE_MARKERS: Tuple[int, ...] = tuple(range(1, 12))

# Preset: fast smoke for CI / short synthetic clip
_SMOKE_ENV = {
    "SWAY_UNLOCK_HYBRID_SAM_TUNING": "1",
    "SWAY_HYBRID_SAM_OVERLAP": "0",
    "SWAY_UNLOCK_POSE_TUNING": "1",
    "SWAY_3D_LIFT": "0",
}
_SMOKE_ARGS = [
    "--pose-model",
    "base",
    "--pose-stride",
    "2",
    "--no-temporal-pose-refine",
    "--no-pose-3d-lift",
]

BUILTIN_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "smoke_quick",
        "description": "Fast smoke (SAM off, no 3D lift, pose stride 2)",
        "env": dict(_SMOKE_ENV),
        "args": list(_SMOKE_ARGS),
    },
    {
        "name": "smoke_yolo_stride2",
        "description": "Smoke + YOLO detection stride 2",
        "env": {**_SMOKE_ENV, "SWAY_YOLO_DETECTION_STRIDE": "2"},
        "args": list(_SMOKE_ARGS),
    },
    {
        "name": "smoke_pose_stride1",
        "description": "Smoke but pose every frame (slower; stronger pose coverage)",
        "env": dict(_SMOKE_ENV),
        "args": [
            "--pose-model",
            "base",
            "--pose-stride",
            "1",
            "--no-temporal-pose-refine",
            "--no-pose-3d-lift",
        ],
    },
    {
        "name": "smoke_track_stats_json",
        "description": "Smoke (track_stats.json always written after Phase 3)",
        "env": {**_SMOKE_ENV},
        "args": list(_SMOKE_ARGS),
    },
    {
        "name": "smoke_yolo_infer_batch_2",
        "description": "Smoke + YOLO predict batch size 2 (BoxMOT path)",
        "env": {**_SMOKE_ENV, "SWAY_UNLOCK_DETECTION_TUNING": "1", "SWAY_YOLO_INFER_BATCH": "2"},
        "args": list(_SMOKE_ARGS),
    },
]

EXTENDED_BUILTIN_PRESETS: List[Dict[str, Any]] = BUILTIN_PRESETS + [
    {
        "name": "smoke_vitpose_max_forward",
        "description": "Smoke + ViTPose chunked forwards (crowded-frame path)",
        "env": {**_SMOKE_ENV, "SWAY_UNLOCK_POSE_TUNING": "1", "SWAY_VITPOSE_MAX_PER_FORWARD": "4"},
        "args": list(_SMOKE_ARGS),
    },
    {
        "name": "smoke_group_video_off",
        "description": "Smoke + SWAY_GROUP_VIDEO=0 (no auto 960 detect bump)",
        "env": {**_SMOKE_ENV, "SWAY_UNLOCK_DETECTION_TUNING": "1", "SWAY_GROUP_VIDEO": "0"},
        "args": list(_SMOKE_ARGS),
    },
    {
        "name": "smoke_global_link_off",
        "description": "Smoke + SWAY_GLOBAL_LINK=0",
        "env": {**_SMOKE_ENV, "SWAY_UNLOCK_PHASE3_STITCH_TUNING": "1", "SWAY_GLOBAL_LINK": "0"},
        "args": list(_SMOKE_ARGS),
    },
]


@dataclass
class RunResult:
    name: str
    ok: bool
    duration_s: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    phase_timings_s: Dict[str, float] = field(default_factory=dict)
    deep_notes: List[str] = field(default_factory=list)
    stdout_tail: str = ""
    report: Dict[str, Any] = field(default_factory=dict)


def write_synthetic_video(path: Path, n_frames: int = 64, fps: float = 16.0) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(
            "OpenCV VideoWriter could not open mp4v — install OpenCV or pass --video."
        )
    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (40, 40, 50)
        x = 80 + (i * 6) % max(1, w - 200)
        cv2.rectangle(img, (x, 120), (x + 140, 420), (180, 180, 220), -1)
        cv2.circle(img, (x + 70, 160), 35, (200, 200, 200), -1)
        vw.write(img)
    vw.release()
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError(f"Synthetic video invalid: {path}")


def validate_phase_markers(combined_log: str) -> List[str]:
    errs: List[str] = []
    positions: List[Tuple[int, int]] = []
    for n in PHASE_MARKERS:
        token = f"[{n}/11]"
        if token not in combined_log:
            errs.append(f"Missing phase marker {token}")
            continue
        positions.append((n, combined_log.find(token)))
    if errs:
        return errs
    idxs = [p for _, p in positions]
    if idxs != sorted(idxs):
        errs.append("Phase markers appear out of order in log")
    return errs


def _log_section(log: str, start_token: str, end_token: Optional[str]) -> str:
    i0 = log.find(start_token)
    if i0 < 0:
        return ""
    if end_token:
        i1 = log.find(end_token, i0 + len(start_token))
        return log[i0 : i1 if i1 > i0 else len(log)]
    return log[i0:]


def parse_phase_timings_s(log: str) -> Dict[str, float]:
    """
    Extract wall seconds per phase from main.py log.
    Phases 1–2 share one line: ``Phases 1–2: X.Xs`` (en-dash or hyphen).
    """
    out: Dict[str, float] = {}
    sec_12 = _log_section(log, "[1/11]", "[3/11]")
    if sec_12:
        m = re.search(r"Phases\s+1[\u2013\-]2[^:\n]*:\s*([\d.]+)\s*s", sec_12)
        if m:
            out["1-2"] = float(m.group(1))
        else:
            ms = list(re.finditer(r"└─[^\n]*?([\d.]+)\s*s", sec_12))
            if ms:
                out["1-2"] = float(ms[-1].group(1))
    for n in range(3, 12):
        start = f"[{n}/11]"
        end = f"[{n + 1}/11]" if n < 11 else None
        chunk = _log_section(log, start, end)
        if not chunk:
            continue
        ms = list(re.finditer(r"└─[^\n]*?([\d.]+)\s*s", chunk))
        if ms:
            out[str(n)] = float(ms[-1].group(1))
    return out


def validate_log_hard_failures(combined: str) -> List[str]:
    errs: List[str] = []
    if "Traceback (most recent call last)" in combined:
        errs.append("Python Traceback present in log")
    if re.search(r"\bCUDA out of memory\b", combined, re.I):
        errs.append("CUDA OOM message in log")
    if re.search(r"RuntimeError:\s*CUDA", combined):
        errs.append("CUDA RuntimeError in log")
    return errs


def scan_log_for_issues(combined: str) -> List[str]:
    hints: List[str] = []
    if "CUDA not available" in combined and "YOLO" in combined:
        hints.append("YOLO+track on CPU — long runs on real clips.")
    if re.search(r"Could not write prune log", combined):
        hints.append("prune log write reported an error")
    if combined.count("Warning:") > 8:
        hints.append(f"Many 'Warning:' lines ({combined.count('Warning:')}) — review log")
    if "MPS backend" in combined and "falling back" in combined.lower():
        hints.append("Possible MPS fallback mentioned")
    return hints


def optimization_hints(phase_timings: Dict[str, float], wall_s: float) -> List[str]:
    if wall_s <= 0 or not phase_timings:
        return []
    hints: List[str] = []
    total_ph = sum(phase_timings.values())
    if total_ph <= 0:
        return hints
    for k, v in sorted(phase_timings.items(), key=lambda x: -x[1]):
        frac = v / wall_s
        if frac >= 0.55 and v >= 5.0:
            hints.append(f"Phase {k} used {frac:.0%} of total wall ({v:.1f}s) — main optimization target")
            break
    if phase_timings.get("1-2", 0) / wall_s >= 0.4 and wall_s > 30:
        hints.append(
            "Phases 1–2 dominate wall time — consider detection stride or GPU; "
            "YOLO infer batch is locked to 1 unless SWAY_UNLOCK_DETECTION_TUNING=1"
        )
    if phase_timings.get("5", 0) / wall_s >= 0.35 and wall_s > 30:
        hints.append("Phase 5 (pose) is heavy — try --pose-stride 2 or SWAY_VITPOSE_MAX_PER_FORWARD")
    return hints


def _iter_keypoint_rows(tracks: Dict[str, Any]) -> Any:
    for _tid, tdata in tracks.items():
        kp = tdata.get("keypoints")
        if kp is None:
            continue
        if isinstance(kp, list):
            for row in kp:
                yield row


def validate_data_json_deep(
    data: dict,
    *,
    max_keypoint_rows: int = 50_000,
) -> Tuple[List[str], List[str]]:
    """Structural + numeric finiteness checks on exported JSON."""
    errs: List[str] = []
    warns: List[str] = []

    meta = data.get("metadata") or {}
    for key in ("fps", "native_fps"):
        v = meta.get(key)
        if v is not None:
            try:
                fv = float(v)
                if fv <= 0 or not math.isfinite(fv):
                    errs.append(f"metadata.{key} invalid: {v}")
            except (TypeError, ValueError):
                errs.append(f"metadata.{key} not numeric: {v}")

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        return errs, warns

    names = meta.get("keypoint_names")
    if names is not None and not isinstance(names, list):
        warns.append("metadata.keypoint_names is not a list")

    seen_any_track = False
    rows_checked = 0
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            errs.append(f"frames[{i}] is not an object")
            continue
        fi = fr.get("frame_idx")
        if fi is not None and int(fi) != i:
            warns.append(f"frames[{i}].frame_idx={fi} != index {i}")
        tracks = fr.get("tracks")
        if not isinstance(tracks, dict):
            errs.append(f"frames[{i}].tracks is not an object")
            continue
        if tracks:
            seen_any_track = True
        for tid_str, tdata in tracks.items():
            if not isinstance(tdata, dict):
                errs.append(f"frames[{i}].tracks[{tid_str}] not an object")
                continue
            for row in _iter_keypoint_rows({tid_str: tdata}):
                if rows_checked >= max_keypoint_rows:
                    break
                rows_checked += 1
                if not isinstance(row, (list, tuple)):
                    continue
                for j, v in enumerate(row[:4]):
                    try:
                        x = float(v)
                        if not math.isfinite(x):
                            errs.append(f"non-finite keypoint at frame {i} track {tid_str} comp {j}")
                    except (TypeError, ValueError):
                        warns.append(f"non-numeric keypoint at frame {i} track {tid_str}")
        if rows_checked >= max_keypoint_rows:
            warns.append(f"keypoint scan capped at {max_keypoint_rows} rows (video long)")
            break

    if not seen_any_track:
        warns.append("no tracks in any frame (empty scene or all pruned — may be OK for synthetic)")

    summaries = data.get("track_summaries")
    if summaries is not None and not isinstance(summaries, dict):
        warns.append("track_summaries is not an object")

    return errs, warns


def validate_prune_vs_data(prune: dict, n_frames: int) -> List[str]:
    warns: List[str] = []
    tf = prune.get("total_frames")
    if tf is not None and int(tf) != int(n_frames):
        warns.append(f"prune_log.total_frames={tf} vs data frames len={n_frames}")
    surv = prune.get("surviving_after_post_pose")
    if isinstance(surv, list) and len(surv) > n_frames:
        warns.append("surviving_after_post_pose list longer than frame count (unexpected)")
    return warns


def validate_track_stats_if_present(output_dir: Path) -> Tuple[List[str], List[str]]:
    p = output_dir / "track_stats.json"
    if not p.is_file():
        return [], []
    errs: List[str] = []
    warns: List[str] = []
    try:
        with open(p, encoding="utf-8") as f:
            ts = json.load(f)
    except json.JSONDecodeError as e:
        return [f"track_stats.json invalid: {e}"], []
    if ts.get("schema_version") != 1:
        warns.append(f"track_stats schema_version={ts.get('schema_version')} (expected 1)")
    if ts.get("num_tracks", 0) < 0:
        errs.append("track_stats num_tracks negative")
    return errs, warns


def validate_outputs(
    output_dir: Path,
    video_stem: str,
    *,
    deep: bool,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Shallow + optional deep checks. Returns errors, warnings, report_fragment."""
    errors: List[str] = []
    warnings: List[str] = []
    fragment: Dict[str, Any] = {}

    data_path = output_dir / "data.json"
    prune_path = output_dir / "prune_log.json"
    mp4_path = output_dir / f"{video_stem}_poses.mp4"

    if not data_path.is_file():
        errors.append(f"Missing {data_path}")
        return errors, warnings, fragment
    if not prune_path.is_file():
        errors.append(f"Missing {prune_path}")
        return errors, warnings, fragment
    if not mp4_path.is_file() or mp4_path.stat().st_size < 100:
        errors.append(f"Missing or tiny output video {mp4_path}")

    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"data.json invalid JSON: {e}")
        return errors, warnings, fragment

    meta = data.get("metadata") or {}
    frames = data.get("frames")
    if not isinstance(frames, list):
        errors.append("data.json: 'frames' is not a list")
        return errors, warnings, fragment

    nf_meta = meta.get("num_frames")
    if nf_meta is not None and int(nf_meta) != len(frames):
        errors.append(f"metadata.num_frames ({nf_meta}) != len(frames) ({len(frames)})")

    try:
        with open(prune_path, encoding="utf-8") as f:
            prune = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"prune_log.json invalid JSON: {e}")
        return errors, warnings, fragment

    warnings.extend(validate_prune_vs_data(prune, len(frames)))

    if deep:
        d_err, d_warn = validate_data_json_deep(data)
        errors.extend(d_err)
        warnings.extend(d_warn)
        fragment["data_json_deep"] = {
            "frames": len(frames),
            "errors": len(d_err),
            "warnings": len(d_warn),
        }

    ts_err, ts_warn = validate_track_stats_if_present(output_dir)
    errors.extend(ts_err)
    warnings.extend(ts_warn)

    return errors, warnings, fragment


def load_matrix_yaml(path: Path) -> List[Dict[str, Any]]:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    presets = raw.get("presets")
    if not isinstance(presets, list) or not presets:
        raise ValueError(f"{path}: expected non-empty 'presets' list")
    out = []
    for p in presets:
        if not isinstance(p, dict) or "name" not in p:
            raise ValueError(f"{path}: each preset needs a 'name'")
        out.append(
            {
                "name": str(p["name"]),
                "description": str(p.get("description", "")),
                "env": dict(p.get("env") or {}),
                "args": list(p.get("args") or []),
            }
        )
    return out


def run_one(
    *,
    video: Path,
    output_dir: Path,
    extra_env: Dict[str, str],
    extra_args: List[str],
    name: str,
    timeout_sec: Optional[int],
    deep: bool,
    write_report: bool,
    strict_guardrails: bool = False,
) -> RunResult:
    import time

    t0 = time.perf_counter()
    env = os.environ.copy()
    for k, v in extra_env.items():
        env[str(k)] = str(v)

    cmd = [
        sys.executable,
        str(MAIN_PY),
        str(video),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    errs: List[str] = []
    warns: List[str] = []
    hints: List[str] = []
    deep_notes: List[str] = []

    errs.extend(validate_log_hard_failures(combined))

    if proc.returncode != 0:
        errs.append(f"main.py exit code {proc.returncode}")

    errs.extend(validate_phase_markers(combined))

    if strict_guardrails:
        boundary = _extract_stop_boundary(extra_args)
        env_map = {str(k): str(v) for k, v in extra_env.items()}
        for v in validate_run_against_contracts(combined, env_map, output_dir, boundary):
            errs.append(f"[guardrails] {v.contract_name}/{v.clause}: {v.detail}")

    phase_timings = parse_phase_timings_s(combined)
    hints.extend(optimization_hints(phase_timings, elapsed))
    hints.extend(scan_log_for_issues(combined))

    frag: Dict[str, Any] = {
        "preset": name,
        "wall_seconds": round(elapsed, 3),
        "exit_code": proc.returncode,
        "phase_timings_s": phase_timings,
    }

    if proc.returncode == 0:
        o_err, o_warn, out_frag = validate_outputs(output_dir, video.stem, deep=deep)
        errs.extend(o_err)
        warns.extend(o_warn)
        frag.update(out_frag)
        if deep and phase_timings:
            deep_notes.append(
                "phase timings (s): " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(phase_timings.items(), key=lambda x: (len(x[0]), x[0])))
            )

    ok = len(errs) == 0
    tail = combined[-6000:] if len(combined) > 6000 else combined

    report = {
        "ok": ok,
        "errors": errs,
        "warnings": warns,
        "hints": hints,
        "deep_notes": deep_notes,
        **frag,
        "command": cmd,
    }

    if write_report and output_dir.is_dir():
        rp = output_dir / "pipeline_validate_report.json"
        try:
            with open(rp, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except OSError:
            pass

    return RunResult(
        name=name,
        ok=ok,
        duration_s=elapsed,
        errors=errs,
        warnings=warns,
        hints=hints,
        phase_timings_s=phase_timings,
        deep_notes=deep_notes,
        stdout_tail=tail,
        report=report,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deep end-to-end validation for sway_pose_mvp main.py.",
    )
    ap.add_argument("video", type=str, nargs="?", default="", help="Input MP4 (default: synthetic)")
    ap.add_argument("--output-dir", type=str, default="", help="Base output dir (per-preset subdirs)")
    ap.add_argument("--full", action="store_true", help="No smoke env/args (full default pipeline)")
    ap.add_argument("--matrix-file", type=str, default="", help="YAML presets list")
    ap.add_argument("--builtin-matrix", action="store_true", help="Run BUILTIN_PRESETS (5 configs)")
    ap.add_argument(
        "--extended-matrix",
        action="store_true",
        help="Run EXTENDED_BUILTIN_PRESETS (8 configs: batching, vitpose chunk, group/global toggles)",
    )
    ap.add_argument("--no-deep", action="store_true", help="Skip deep data.json / phase-hint analysis")
    ap.add_argument("--write-reports", action="store_true", help="Write pipeline_validate_report.json per output dir")
    ap.add_argument("--timeout", type=int, default=0, help="Subprocess timeout seconds (0 = none)")
    ap.add_argument("--keep-temp", action="store_true", help="Keep synthetic video / temp dirs")
    ap.add_argument(
        "--strict-guardrails",
        action="store_true",
        help="After each run, enforce sway/technology_contracts (phase fallthrough, branch rules, etc.).",
    )
    args = ap.parse_args()

    if not MAIN_PY.is_file():
        print(f"main.py not found at {MAIN_PY}", file=sys.stderr)
        return 1

    deep = not args.no_deep
    timeout = int(args.timeout) if int(args.timeout) > 0 else None

    synthetic_path: Optional[Path] = None
    if args.video.strip():
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.is_file():
            print(f"Video not found: {video_path}", file=sys.stderr)
            return 1
    else:
        td = Path(tempfile.mkdtemp(prefix="sway_validate_"))
        synthetic_path = td / "synthetic_validate.mp4"
        print(f"Generating synthetic video: {synthetic_path}", flush=True)
        try:
            write_synthetic_video(synthetic_path)
        except Exception as e:
            print(f"Synthetic video failed: {e}", file=sys.stderr)
            return 1
        video_path = synthetic_path
        if args.keep_temp:
            print(f"Keeping synthetic (--keep-temp): {synthetic_path}", flush=True)

    presets: List[Dict[str, Any]] = []
    if args.matrix_file:
        mf = Path(args.matrix_file).expanduser()
        if not mf.is_file():
            print(f"Matrix file not found: {mf}", file=sys.stderr)
            return 1
        try:
            presets = load_matrix_yaml(mf.resolve())
        except Exception as e:
            print(f"Matrix YAML error: {e}", file=sys.stderr)
            return 1
    elif args.extended_matrix:
        presets = list(EXTENDED_BUILTIN_PRESETS)
    elif args.builtin_matrix:
        presets = list(BUILTIN_PRESETS)
    else:
        if args.full:
            presets = [
                {
                    "name": "full_default",
                    "description": "CLI defaults only",
                    "env": {},
                    "args": [],
                }
            ]
        else:
            presets = [
                {
                    "name": "single_smoke",
                    "description": "Single fast smoke",
                    "env": dict(BUILTIN_PRESETS[0]["env"]),
                    "args": list(BUILTIN_PRESETS[0]["args"]),
                }
            ]

    results: List[RunResult] = []
    for preset in presets:
        name = preset["name"]
        p_env = {str(k): str(v) for k, v in (preset.get("env") or {}).items()}
        p_args = list(preset.get("args") or [])
        if args.output_dir.strip():
            out = Path(args.output_dir).expanduser() / f"_validate_{name}"
        else:
            out = Path(tempfile.mkdtemp(prefix=f"sway_out_{name}_"))
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Preset: {name} ===", flush=True)
        if preset.get("description"):
            print(f"    {preset['description']}", flush=True)
        print(f"    output: {out}", flush=True)

        res = run_one(
            video=video_path,
            output_dir=out,
            extra_env=p_env,
            extra_args=p_args,
            name=name,
            timeout_sec=timeout,
            deep=deep,
            write_report=args.write_reports,
            strict_guardrails=bool(args.strict_guardrails),
        )
        results.append(res)
        print(f"    {'OK' if res.ok else 'FAIL'} in {res.duration_s:.1f}s", flush=True)
        if res.phase_timings_s:
            pt = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(res.phase_timings_s.items(), key=lambda x: (len(x[0]), x[0])))
            print(f"    phase timings: {pt}", flush=True)
        for n in res.deep_notes:
            print(f"    [deep] {n}", flush=True)
        for h in res.hints:
            print(f"    [hint] {h}", flush=True)
        for w in res.warnings:
            print(f"    [warn] {w}", flush=True)
        for e in res.errors:
            print(f"    [error] {e}", file=sys.stderr)
        if not res.ok:
            print("    --- log tail ---", file=sys.stderr)
            print(res.stdout_tail, file=sys.stderr)
        if args.write_reports:
            print(f"    report: {out / 'pipeline_validate_report.json'}", flush=True)
        if args.keep_temp:
            print(f"    --keep-temp: {out}", flush=True)

    if synthetic_path is not None and not args.keep_temp:
        try:
            synthetic_path.unlink(missing_ok=True)
            synthetic_path.parent.rmdir()
        except OSError:
            pass

    all_reports = [r.report for r in results]
    if args.output_dir.strip() and args.write_reports:
        summary_path = Path(args.output_dir).expanduser() / "pipeline_validate_summary.json"
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"presets": all_reports}, f, indent=2)
            print(f"\nSummary: {summary_path}", flush=True)
        except OSError:
            pass

    failed = [r for r in results if not r.ok]
    print("\n=== Summary ===", flush=True)
    for r in results:
        print(f"  {r.name}: {'PASS' if r.ok else 'FAIL'} ({r.duration_s:.1f}s)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
