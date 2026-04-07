#!/usr/bin/env python3
"""
Queue a multi-stage checkpoint **fan-out tree** on Pipeline Lab: every run that
finishes a stage becomes a parent for **all** variants in the next stage (full
Cartesian product). Uses ``POST /api/runs/batch_path`` with per-run ``checkpoint``
(resume_from, expect_boundary, stop_after_boundary).

Requires uvicorn for the Lab API. Outputs are under ``pipeline_lab/runs/<uuid>/output/``
so they appear in the Lab and Watch UI.

  cd sway_pose_mvp
  python -m tools.pipeline_tree_queue --tree pipeline_lab/tree_presets/bigtest_checkpoint_tree.yaml

**Multiple videos (same tree, back-to-back):** pass ``--video`` more than once, or set
``video_paths`` in the YAML (list of absolute paths). Each video runs the full stage
sequence before the next; all runs share one ``batch_id`` for the Lab.

Open the Lab **home** URL (same host as the API when using the built UI). The web app
discovers active batches automatically; ``?batch=<id>`` is optional.

Options:
  --video PATH           Repeat for each input clip (overrides YAML video_path / video_paths).
  --dry-run              Print stage sizes only (no HTTP).
  --abort-on-error       Stop the tree if any stage run fails, is cancelled, or vanishes from the API.
  --continue-on-error    Accepted for compatibility; default is already to continue with successful parents.
  --missing-grace-polls  Consecutive list polls a run_id may be absent before it is treated as removed (default: 6).
  --strict-status        Only trust API status=done (skip run_manifest.json reconciliation).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tree_checkpoint_queue_lib import (  # noqa: E402
    TreeQueueError,
    load_tree_yaml,
    resolve_video_paths_from_doc,
    run_checkpoint_tree_session,
    validate_stages,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Queue checkpoint fan-out tree on Pipeline Lab (CLI)")
    ap.add_argument(
        "--tree",
        type=Path,
        required=True,
        help="YAML tree definition (see pipeline_lab/tree_presets/)",
    )
    ap.add_argument("--lab-url", type=str, default="http://localhost:8765", help="Lab API base URL")
    ap.add_argument(
        "--ui-origin",
        type=str,
        default=None,
        help="Origin for printed Lab links (defaults to --lab-url). Set e.g. http://localhost:5173 for Vite dev.",
    )
    ap.add_argument("--poll", type=float, default=10.0, help="Seconds between status polls when waiting")
    ap.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="PATH",
        help="Input video (repeat for multiple clips; same tree runs on each, sequentially). Overrides YAML paths.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned fan-out only (no HTTP)")
    ap.add_argument(
        "--abort-on-error",
        action="store_true",
        help="Stop the tree when any run in a stage fails, is cancelled, or disappears from the API "
        "(default: keep only successful parents and queue the next stage).",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--missing-grace-polls",
        type=int,
        default=6,
        metavar="N",
        help="Treat a run as removed after N consecutive polls where it is missing from GET /api/runs (default: 6)",
    )
    ap.add_argument(
        "--strict-status",
        action="store_true",
        help="Do not promote runs from error/cancelled to OK when run_manifest.json on disk "
        "looks successful (default: reconcile once per stage for fewer false failures).",
    )
    args = ap.parse_args()
    continue_on_error = not args.abort_on_error
    if args.continue_on_error:
        continue_on_error = True

    tree_path = args.tree.expanduser()
    if not tree_path.is_file():
        raise SystemExit(f"Tree file not found: {tree_path}")

    try:
        doc = load_tree_yaml(tree_path)
        stages = validate_stages(doc.get("stages"))
    except TreeQueueError as e:
        raise SystemExit(str(e)) from e

    if args.video:
        path_strs = [str(p) for p in args.video]
    else:
        path_strs = resolve_video_paths_from_doc(doc)
    if not path_strs:
        raise SystemExit(
            "No videos: set video_path or video_paths in the YAML, or pass --video one or more times."
        )
    video_paths = [Path(p).expanduser() for p in path_strs]

    video_display_labels: List[str] | None = None
    if len(video_paths) == 1:
        sl = str(doc.get("source_label") or "").strip()
        if sl:
            video_display_labels = [sl]

    base = args.lab_url.rstrip("/")
    ui = (args.ui_origin or args.lab_url).rstrip("/")
    batch_id = str(uuid.uuid4())

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"Planned fan-out (batch_id={batch_id}), {len(video_paths)} video(s):")
    per_video_total = 0
    parents_n = 1
    for i, st in enumerate(stages):
        v = len(st["variants"])
        n_this = parents_n * v
        per_video_total += n_this
        log(f"  stage {i + 1} {st['name']!r}: queue {n_this} run(s) per video  (leaves after stage: {n_this})")
        parents_n = n_this
    log(f"  Total main.py invocations per video: {per_video_total}")
    log(f"  Grand total (all videos): {per_video_total * len(video_paths)}")

    if args.dry_run:
        log("\nDry run — no requests sent.")
        for i, vp in enumerate(video_paths):
            log(f"  video {i + 1}: {vp}")
        log(f"After a real run, open Lab: {ui}/  (optional: {ui}/?batch={batch_id})")
        return

    for vp in video_paths:
        if not vp.is_file():
            raise SystemExit(f"video_path must exist on this machine (Lab resolves it): {vp}")

    try:
        run_checkpoint_tree_session(
            base=base,
            stages=stages,
            video_paths=video_paths,
            batch_id=batch_id,
            poll_s=args.poll,
            continue_on_error=continue_on_error,
            missing_grace_polls=args.missing_grace_polls,
            strict_status=args.strict_status,
            log=log,
            video_display_labels=video_display_labels,
        )
    except TreeQueueError as e:
        raise SystemExit(str(e)) from e

    log(f"\nLab home: {ui}/")
    log(f"Optional deep link: {ui}/?batch={batch_id}")
    log(f"API: GET {base}/api/batches  |  filter runs: batch_id == {batch_id!r}")
    log(f"Watch: {ui}/watch/<run_id>")


if __name__ == "__main__":
    main()
