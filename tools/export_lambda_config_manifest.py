#!/usr/bin/env python3
"""
Export a machine-readable manifest of sweep/runtime configuration space.

Usage:
  python -m tools.export_lambda_config_manifest
  python -m tools.export_lambda_config_manifest --output docs/LAMBDA_CONFIGURATION_MANIFEST.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTO_SWEEP = REPO_ROOT / "tools" / "auto_sweep.py"


def _extract_sway_env_keys(source: str) -> List[str]:
    keys = re.findall(r'env\["(SWAY_[A-Z0-9_]+)"\]\s*=', source)
    return sorted(set(keys))


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _node_src(source: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(source, node)
    return (seg or "").strip()


def _parse_step_kw(call: ast.Call) -> Optional[float]:
    for kw in call.keywords:
        if kw.arg == "step":
            v = _safe_literal(kw.value)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _count_choices(low: float, high: float, step: Optional[float]) -> Optional[int]:
    if step is None or step <= 0:
        return None
    n = int(round((high - low) / step)) + 1
    return max(n, 0)


@dataclass
class SuggestParam:
    name: str
    kind: str
    domain: Dict[str, Any]


def _extract_suggest_params(source: str) -> List[SuggestParam]:
    tree = ast.parse(source)
    out: List[SuggestParam] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "trial"
                and fn.attr.startswith("suggest_")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                pname = node.args[0].value
                skind = fn.attr.replace("suggest_", "")
                domain: Dict[str, Any] = {}

                if skind == "categorical" and len(node.args) >= 2:
                    lit = _safe_literal(node.args[1])
                    if isinstance(lit, (list, tuple)):
                        domain["values"] = list(lit)
                        domain["choice_count"] = len(lit)
                    else:
                        domain["values_expr"] = _node_src(source, node.args[1])
                        domain["choice_count"] = None

                elif skind in ("int", "float") and len(node.args) >= 3:
                    low = _safe_literal(node.args[1])
                    high = _safe_literal(node.args[2])
                    step = _parse_step_kw(node)
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                        if skind == "int" and step is None:
                            step = 1.0
                        domain["low"] = low
                        domain["high"] = high
                        domain["step"] = step
                        domain["choice_count"] = _count_choices(float(low), float(high), step)
                    else:
                        domain["low_expr"] = _node_src(source, node.args[1])
                        domain["high_expr"] = _node_src(source, node.args[2])
                        domain["step"] = step
                        domain["choice_count"] = None
                else:
                    domain["raw_call"] = _node_src(source, node)

                out.append(SuggestParam(name=pname, kind=skind, domain=domain))

            self.generic_visit(node)

    Visitor().visit(tree)

    # Deduplicate while preserving first appearance
    seen = set()
    uniq: List[SuggestParam] = []
    for p in out:
        if p.name in seen:
            continue
        seen.add(p.name)
        uniq.append(p)
    return uniq


def _python_references_for_key(key: str) -> int:
    total = 0
    for p in REPO_ROOT.rglob("*.py"):
        if p == AUTO_SWEEP:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        total += txt.count(key)
    return total


def _load_runtime_catalog() -> Dict[str, Any]:
    # Best-effort import for dynamic option lists.
    try:
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tools import auto_sweep as mod  # type: ignore

        return {
            "yolo_weight_choices": mod._available_yolo_weights(),
            "tracker_types": mod._available_tracker_types(),
            "reid_weight_choices": mod._available_reid_weights(),
        }
    except Exception as e:
        return {"import_error": str(e)}


def build_manifest() -> Dict[str, Any]:
    src = AUTO_SWEEP.read_text(encoding="utf-8")
    sway_keys = _extract_sway_env_keys(src)
    suggest_params = _extract_suggest_params(src)

    discrete_sizes: List[int] = []
    for p in suggest_params:
        n = p.domain.get("choice_count")
        if isinstance(n, int) and n > 0:
            discrete_sizes.append(n)
    discrete_upper_bound: Optional[int]
    if discrete_sizes:
        prod = 1
        for n in discrete_sizes:
            prod *= n
        discrete_upper_bound = prod
    else:
        discrete_upper_bound = None

    env_key_status = []
    for k in sway_keys:
        refs = _python_references_for_key(k)
        env_key_status.append(
            {
                "key": k,
                "referenced_outside_auto_sweep": refs > 0,
                "reference_count_outside_auto_sweep": refs,
            }
        )

    return {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [
            str(AUTO_SWEEP.relative_to(REPO_ROOT)),
            "main.py",
            "sway/sentinel_sbm.py",
            "sway/umot_backtrack.py",
            "sway/mote_disocclusion.py",
        ],
        "runtime_logging_contract": {
            "feature_log_prefix": "[feature]",
            "fields": ["requested", "runtime", "wiring"],
            "fail_fast_env": "SWAY_FAIL_ON_UNWIRED_EXTRAS",
        },
        "advanced_modules": [
            {
                "name": "MOTE",
                "toggle_env": "SWAY_MOTE_DISOCCLUSION",
                "runtime_wired": True,
                "status": "active_when_requested",
            },
            {
                "name": "SentinelSBM",
                "toggle_env": "SWAY_SENTINEL_SBM",
                "runtime_wired": True,
                "status": "active_when_requested",
            },
            {
                "name": "UMOTBacktracker",
                "toggle_env": "SWAY_UMOT_BACKTRACK",
                "runtime_wired": True,
                "status": "active_when_requested",
            },
            {
                "name": "BackwardPass",
                "toggle_env": "SWAY_BACKWARD_PASS_ENABLED",
                "runtime_wired": True,
                "status": "active_when_requested",
            },
        ],
        "sweep_env_keys": env_key_status,
        "trial_parameter_space": [
            {
                "name": p.name,
                "kind": p.kind,
                "domain": p.domain,
            }
            for p in suggest_params
        ],
        "search_space_summary": {
            "trial_param_count": len(suggest_params),
            "sway_env_key_count": len(sway_keys),
            "discrete_upper_bound_if_all_independent": discrete_upper_bound,
            "note": (
                "Upper bound ignores conditional branches and dynamic choice lists; "
                "true reachable space is branch-constrained."
            ),
        },
        "runtime_catalog": _load_runtime_catalog(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lambda configuration manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "LAMBDA_CONFIGURATION_MANIFEST.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {args.output}")


if __name__ == "__main__":
    main()

