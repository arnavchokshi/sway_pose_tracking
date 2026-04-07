"""Append-only structured phase summaries for operator debugging (see MASTER_PIPELINE_GUIDELINE §4.5)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class PhaseDebugLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None

    def log(self, phase: str, wall_s: float, debug: Optional[Dict[str, Any]] = None) -> None:
        if self.path is None:
            return
        rec = {
            "kind": "phase_summary",
            "phase": phase,
            "wall_s": round(float(wall_s), 4),
            "ts_unix": time.time(),
            "debug": debug or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def maybe_write_debug_file(output_dir: Path, name: str, text: str) -> None:
    if os.environ.get("SWAY_PHASE_DEBUG_FILES", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    d = output_dir / "debug"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")
