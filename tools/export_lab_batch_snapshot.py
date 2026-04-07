#!/usr/bin/env python3
"""
Fetch Pipeline Lab run rows (optionally filtered by batch_id) and write a YAML snapshot
for archival / handoff. Does not copy run output folders (those stay under pipeline_lab/runs/).

  cd sway_pose_mvp
  python3 -m tools.export_lab_batch_snapshot --batch-id 86581e7d-b116-42bc-be60-fedc65a88598

Re-run when the tree finishes; use --label complete for the filename suffix.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install PyYAML") from e


def _get_json(url: str, timeout_s: float = 120.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Lab batch run list to YAML snapshot")
    ap.add_argument("--lab-url", default="http://localhost:8765", help="Lab API base URL")
    ap.add_argument("--batch-id", default="", help="Only runs with this batch_id (substring match ok)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "pipeline_lab" / "batch_exports",
        help="Output directory",
    )
    ap.add_argument(
        "--label",
        default="",
        help="Suffix for filename e.g. partial | complete (default: timestamp only)",
    )
    args = ap.parse_args()

    base = args.lab_url.rstrip("/")
    rows = _get_json(f"{base}/api/runs")
    if not isinstance(rows, list):
        raise SystemExit(f"Unexpected /api/runs response: {type(rows)}")

    bid = str(args.batch_id or "").strip()
    if bid:
        rows = [r for r in rows if isinstance(r, dict) and bid in str(r.get("batch_id") or "")]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.label}" if str(args.label or "").strip() else ""
    safe_batch = bid.replace("/", "_")[:8] if bid else "all"
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lab_runs_{safe_batch}{suffix}_{ts}.yaml"

    from collections import Counter

    st = Counter(str(r.get("status") or "?") for r in rows)

    slim: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: (str(x.get("created") or ""), str(x.get("run_id") or ""))):
        if not isinstance(r, dict):
            continue
        slim.append(
            {
                "run_id": r.get("run_id"),
                "status": r.get("status"),
                "recipe_name": r.get("recipe_name"),
                "video_stem": r.get("video_stem"),
                "batch_id": r.get("batch_id"),
                "created": r.get("created"),
                "error": r.get("error"),
            }
        )

    doc: Dict[str, Any] = {
        "exported_at_utc": ts,
        "lab_url": base,
        "filter_batch_id_substring": bid or None,
        "run_count": len(slim),
        "status_counts": dict(st),
        "tree_yaml": "pipeline_lab/tree_presets/bigtest_checkpoint_tree.yaml",
        "runs": slim,
    }

    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {out_path} ({len(slim)} runs)")


if __name__ == "__main__":
    main()
