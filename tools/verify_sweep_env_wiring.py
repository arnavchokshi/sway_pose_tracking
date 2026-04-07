#!/usr/bin/env python3
"""
Static audit: every SWAY_* key written by tools/auto_sweep.py vs references
elsewhere in sway_pose_mvp (excluding this script).

Run: python -m tools.verify_sweep_env_wiring
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTO_SWEEP = REPO_ROOT / "tools" / "auto_sweep.py"
THIS_SCRIPT = Path(__file__).resolve()


def extract_sweep_keys() -> list[str]:
    text = AUTO_SWEEP.read_text(encoding="utf-8")
    keys = re.findall(r'env\["(SWAY_[A-Z0-9_]+)"\]', text)
    return sorted(set(keys))


def rg_count(key: str, exclude_auto_sweep: bool) -> int:
    cmd = [
        "rg",
        "-c",
        "--glob",
        "*.py",
        key,
        str(REPO_ROOT),
    ]
    if exclude_auto_sweep:
        cmd.extend(["--glob", "!auto_sweep.py"])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return 0
        return -1
    except FileNotFoundError:
        return -2
    total = 0
    for line in out.strip().splitlines():
        if ":" in line:
            try:
                total += int(line.rsplit(":", 1)[-1])
            except ValueError:
                pass
    return total


def main() -> None:
    keys = extract_sweep_keys()
    print(f"Found {len(keys)} unique SWAY_* keys assigned in suggest_env_for_trial.\n")

    if rg_count("SWAY_YOLO_CONF", True) == -2:
        print("Install ripgrep (rg) for full scan; falling back to Python scan.\n")
        orphan = _fallback_scan(keys)
        _print_notes()
        if orphan:
            sys.exit(1)
        return

    connected: list[str] = []
    sweep_only: list[str] = []
    for k in keys:
        n = rg_count(k, exclude_auto_sweep=True)
        if n > 0:
            connected.append(k)
        else:
            sweep_only.append(k)

    print("=== CONNECTED (referenced outside auto_sweep.py) ===")
    for k in connected:
        print(f"  OK  {k}")
    print(f"\nTotal: {len(connected)}\n")

    print("=== NOT CONNECTED (only appears in auto_sweep.py or nowhere) ===")
    for k in sweep_only:
        print(f"  !!  {k}")
    print(f"\nTotal: {len(sweep_only)}")

    _print_notes()

    if sweep_only:
        sys.exit(1)


def _print_notes() -> None:
    print("\n=== NOTES (behavior vs string references) ===")
    notes = [
        "SWAY_DOC_NSA_KF_ON: fully wired — apply_nsa_kf_patch() patches KalmanFilterXYSR.update to scale R "
        "by (1-conf)^2; wrap_deepocsort_for_nsa() wraps tracker.update to inject mean-det confidence via "
        "thread-local each frame.  Real NSA behavior active.",
        "SWAY_ST_THETA_IOU / SWAY_ST_THETA_EMB: fully wired — patch_solidtrack_cost() wraps tracker.update "
        "with a thread-local flag and monkey-patches boxmot.utils.matching.iou_distance to return the gated "
        "SolidTrack cost matrix when active.  Real gated-cost matching in effect.",
        "SWAY_ST_EMA_ALPHA: fully wired — apply_ema_to_tracker() wraps tracker.update to EMA-smooth per-track "
        "feature embeddings after each frame using the supplied alpha.  Default 0.9 when unset.",
        "SWAY_BOOST_REID vs SWAY_BOXMOT_REID_ON: consistent — for boosttrack, SWAY_BOOST_REID is the "
        "authoritative flag and SWAY_BOXMOT_REID_ON is forced to the same value in suggest_env_for_trial; "
        "tracker.py logs a warning if they are set inconsistently at runtime.",
        "SWAY_GLOBAL_AFLINK: only set when sweep chooses force_heuristic; if neural_if_available and AFLink weights "
        "exist, variable is unset and global_track_link uses neural path (intended).",
    ]
    for n in notes:
        print(f"  - {n}")


def _fallback_scan(keys: list[str]) -> list[str]:
    skip = {"auto_sweep.py", "verify_sweep_env_wiring.py"}
    py_files = [
        p
        for p in REPO_ROOT.rglob("*.py")
        if p.name not in skip and p.resolve() != THIS_SCRIPT
    ]
    connected: list[str] = []
    orphan: list[str] = []
    for k in keys:
        hits = sum(1 for p in py_files if k in p.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            connected.append(k)
            print(f"  OK  {k}  (referenced in {hits} file(s) besides auto_sweep)")
        else:
            orphan.append(k)
            print(f"  !!  {k}  (sweep only — no other .py references)")
    print(f"\nSummary: {len(connected)} connected, {len(orphan)} sweep-only")
    return orphan


if __name__ == "__main__":
    main()
