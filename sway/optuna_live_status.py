"""
Atomic JSON snapshots of an Optuna study for live monitoring (UI, scp, rsync).

Written after each trial by ``tools.auto_sweep`` when ``--status-json`` is enabled
(default path under ``output/sweeps/optuna/``).

CLI: ``python -m tools.export_optuna_study_status`` (refresh from storage without running sweep).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_study_status_payload(study: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flat dict safe for JSON (trial states, params, user_attrs, best)."""
    trials_out: List[Dict[str, Any]] = []
    for t in study.get_trials(deepcopy=False):
        trials_out.append(
            {
                "number": int(t.number),
                "state": t.state.name if hasattr(t.state, "name") else str(t.state),
                "value": t.value,
                "params": dict(t.params) if t.params else {},
                "user_attrs": dict(t.user_attrs) if t.user_attrs else {},
            }
        )

    best: Optional[Dict[str, Any]] = None
    try:
        bt = study.best_trial
        best = {
            "number": int(bt.number),
            "value": bt.value,
            "params": dict(bt.params) if bt.params else {},
            "user_attrs": dict(bt.user_attrs) if bt.user_attrs else {},
        }
    except ValueError:
        pass

    direction = getattr(study.direction, "name", None) or str(study.direction)

    payload: Dict[str, Any] = {
        "schema": "sway_optuna_sweep_status_v1",
        "updated_unix": time.time(),
        "study_name": study.study_name,
        "direction": direction,
        "n_trials_total": len(trials_out),
        "n_complete": sum(1 for x in trials_out if x["state"] == "COMPLETE"),
        "n_pruned": sum(1 for x in trials_out if x["state"] == "PRUNED"),
        "n_other": sum(1 for x in trials_out if x["state"] not in ("COMPLETE", "PRUNED")),
        "best": best,
        "trials": trials_out,
    }
    if extra:
        payload["meta"] = extra
    return payload


def write_live_sweep_status(
    study: Any,
    path: Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``build_study_status_payload`` to ``path`` atomically (replace)."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_study_status_payload(study, extra=extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
