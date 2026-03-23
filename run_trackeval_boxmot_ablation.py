#!/usr/bin/env python3
"""
TrackEval ablation: BoT-SORT (SWAY_USE_BOXMOT=0) vs BoxMOT (default).

Uses tracking-only MOT export (fast — no pose / render).

Modes
-----
1) Human MOT ground truth (recommended for “net positive” claims):
   --gt-mot-dir DIR  with files named {video_stem}_gt.txt (MOTChallenge rows).

2) Baseline agreement (no labels): BoT-SORT trajectories are treated as reference
   “GT” and BoxMOT is evaluated against them. High scores mean the two trackers
   segment the video similarly — not that either matches human truth.

Examples
--------
  python run_trackeval_boxmot_ablation.py \\
    --videos ~/Desktop/IMG_0256.mov \\
    --out-dir output/trackeval_ablation \\
    --baseline-as-reference

  python run_trackeval_boxmot_ablation.py \\
    --videos clip1.mov clip2.mov \\
    --out-dir output/te \\
    --gt-mot-dir benchmarks/mot_gt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


def _video_dims(video: Path) -> Tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()
    return w, h


def _run_tracking_export_mot(
    video: Path,
    out_mot: Path,
    use_boxmot: bool,
    cwd: Path,
    *,
    extra_gtref_mot: Optional[Path] = None,
) -> Tuple[int, int, int]:
    """Subprocess: clean env for boxmot flag; returns (width, height, total_frames).

    If ``extra_gtref_mot`` is set (BoT-SORT run only), also write MOTChallenge-style GT rows
    (mark 1,1,1) for TrackEval's GT loader when using BoT-SORT as pseudo ground truth.
    """
    env = os.environ.copy()
    if use_boxmot:
        env.pop("SWAY_USE_BOXMOT", None)
    else:
        env["SWAY_USE_BOXMOT"] = "0"

    gtref = str(extra_gtref_mot) if extra_gtref_mot else ""

    code = f"""import json, sys, cv2
from pathlib import Path
sys.path.insert(0, {str(cwd)!r})
from sway.tracker import run_tracking
from sway.mot_format import raw_tracks_to_mot_lines, write_mot_file

v = Path({str(video)!r})
cap = cv2.VideoCapture(str(v))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
cap.release()

raw_tracks, total_frames, *_ = run_tracking(str(v))
lines = raw_tracks_to_mot_lines(raw_tracks, as_mot_gt=False)
out = Path({str(out_mot)!r})
write_mot_file(lines, out)
gtref_path = {gtref!r}
if gtref_path:
    write_mot_file(raw_tracks_to_mot_lines(raw_tracks, as_mot_gt=True), Path(gtref_path))
print(json.dumps({{"width": w, "height": h, "total_frames": total_frames, "mot_rows": len(lines)}}))
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr or "")
        raise RuntimeError(f"tracking export failed for {video} (boxmot={use_boxmot}): {r.stderr}")
    meta = json.loads(r.stdout.strip().splitlines()[-1])
    return int(meta["width"]), int(meta["height"]), int(meta["total_frames"])


def _trackeval(
    cwd: Path,
    gt_mot: Path,
    pred_mot: Path,
    seq: str,
    w: int,
    h: int,
) -> Dict[str, Any]:
    sys.path.insert(0, str(cwd))
    from sway.mot_format import load_mot_lines_from_file
    from sway.trackeval_runner import run_trackeval_single_sequence

    gt_lines = load_mot_lines_from_file(gt_mot)
    pr_lines = load_mot_lines_from_file(pred_mot)
    return run_trackeval_single_sequence(gt_lines, pr_lines, seq, w, h)


def _pick_metrics(flat: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "HOTA_HOTA" in flat:
        out["HOTA_HOTA"] = flat["HOTA_HOTA"]
    else:
        for k, v in flat.items():
            if k.startswith("HOTA_HOTA"):
                out["HOTA_HOTA"] = v
                break
    for k in (
        "CLEAR_MOTA",
        "CLEAR_IDF1",
        "Identity_IDF1",
        "Identity_IDSW",
        "CLEAR_IDSW",
        "CLEAR_FP",
        "CLEAR_FN",
    ):
        if k in flat:
            out[k] = flat[k]
    return out


def _mot_pred_lines_to_gtref(lines: List[str]) -> List[str]:
    """TrackEval GT reader expects mark/class/vis = 1,1,1 (not detector conf)."""
    out: List[str] = []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) >= 9:
            parts[6] = "1"
            parts[7] = "1"
            parts[8] = "1"
            out.append(",".join(parts))
        else:
            out.append(ln)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="TrackEval BoT-SORT vs BoxMOT ablation (tracking MOT)")
    ap.add_argument("--videos", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--gt-mot-dir",
        type=Path,
        default=None,
        help="Directory with {stem}_gt.txt MOT ground truth per video",
    )
    ap.add_argument(
        "--baseline-as-reference",
        action="store_true",
        help="Use BoT-SORT MOT as pseudo-GT vs BoxMOT (no human labels)",
    )
    ap.add_argument(
        "--trackeval-only",
        action="store_true",
        help="Skip tracking; use existing *_botsort.mot.txt + *_boxmot.mot.txt under --out-dir "
        "(writes .gtref from botsort pred lines, then runs baseline TrackEval).",
    )
    args = ap.parse_args()

    if not args.gt_mot_dir and not args.baseline_as_reference:
        print(
            "Error: specify --gt-mot-dir (human MOT) and/or --baseline-as-reference.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.trackeval_only and not args.videos:
        print("Error: --trackeval-only requires --videos (for stems and frame size).", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = SCRIPT_DIR

    summary: Dict[str, Any] = {"clips": [], "note": None}
    if args.baseline_as_reference and not args.gt_mot_dir:
        summary["note"] = (
            "baseline_as_reference: metrics measure agreement of BoxMOT with BoT-SORT "
            "trajectories, not accuracy vs human labels."
        )

    for video in args.videos:
        video = video.expanduser().resolve()
        if not video.is_file():
            print(f"Skip missing: {video}", file=sys.stderr)
            continue
        stem = video.stem
        clip_dir = out_dir / stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        mot_bs = clip_dir / f"{stem}_botsort.mot.txt"
        mot_bs_gtref = clip_dir / f"{stem}_botsort.gtref.mot.txt"
        mot_bm = clip_dir / f"{stem}_boxmot.mot.txt"
        gtref_arg = mot_bs_gtref if args.baseline_as_reference else None

        print(f"\n=== {video.name} ===")

        if args.trackeval_only:
            w, h = _video_dims(video)
            if not mot_bs.is_file() or not mot_bm.is_file():
                print(f"  Skip: missing MOT under {clip_dir}", file=sys.stderr)
                continue
            sys.path.insert(0, str(cwd))
            from sway.mot_format import load_mot_lines_from_file, write_mot_file

            if args.baseline_as_reference:
                bs_lines = load_mot_lines_from_file(mot_bs)
                write_mot_file(_mot_pred_lines_to_gtref(bs_lines), mot_bs_gtref)
        else:
            print("  Running BoT-SORT (export MOT)...")
            w, h, _nf = _run_tracking_export_mot(
                video, mot_bs, use_boxmot=False, cwd=cwd, extra_gtref_mot=gtref_arg
            )
            print("  Running BoxMOT (export MOT)...")
            w2, h2, _ = _run_tracking_export_mot(video, mot_bm, use_boxmot=True, cwd=cwd)
            w, h = w2 or w, h2 or h

        clip_result: Dict[str, Any] = {
            "video": str(video),
            "stem": stem,
            "im_width": w,
            "im_height": h,
            "mot_files": {"botsort": str(mot_bs), "boxmot": str(mot_bm)},
        }
        if args.baseline_as_reference:
            clip_result["mot_files"]["botsort_gtref"] = str(mot_bs_gtref)

        if args.gt_mot_dir:
            gt_path = (args.gt_mot_dir / f"{stem}_gt.txt").resolve()
            if not gt_path.is_file():
                clip_result["human_gt_error"] = f"missing {gt_path}"
            else:
                print("  TrackEval vs human GT (BoT-SORT)...")
                m0 = _trackeval(cwd, gt_path, mot_bs, stem + "_gt_vs_botsort", w, h)
                print("  TrackEval vs human GT (BoxMOT)...")
                m1 = _trackeval(cwd, gt_path, mot_bm, stem + "_gt_vs_boxmot", w, h)
                clip_result["vs_human_gt"] = {
                    "botsort": _pick_metrics(m0),
                    "boxmot": _pick_metrics(m1),
                }
                flat0 = {k: v for k, v in m0.items() if k != "sequence"}
                flat1 = {k: v for k, v in m1.items() if k != "sequence"}
                clip_result["vs_human_gt_full"] = {"botsort": flat0, "boxmot": flat1}

        if args.baseline_as_reference:
            print("  TrackEval pseudo-GT = BoT-SORT (.gtref), pred = BoxMOT...")
            mb = _trackeval(cwd, mot_bs_gtref, mot_bm, stem + "_bs_ref", w, h)
            clip_result["boxmot_vs_botsort_reference"] = _pick_metrics(mb)
            clip_result["boxmot_vs_botsort_reference_full"] = {
                k: v for k, v in mb.items() if k != "sequence"
            }

        summary["clips"].append(clip_result)

    out_json = out_dir / "trackeval_ablation_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
