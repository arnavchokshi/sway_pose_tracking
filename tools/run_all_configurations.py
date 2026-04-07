#!/usr/bin/env python3
"""
Build and optionally execute a broad "all coded options" run matrix.

Design goal:
- Cover every coded SWAY_* configuration option at least once.
- Avoid impossible Cartesian explosion by default (one-option-at-a-time matrix).
- Support optional exhaustive Cartesian only for fully discrete subsets.

Examples:
  # Generate plan only
  python -m tools.run_all_configurations --video /abs/path/video.mp4

  # Generate + execute
  python -m tools.run_all_configurations --video /abs/path/video.mp4 --execute
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.technology_contracts import (  # noqa: E402
    contracts_requiring_full_pipeline,
    format_violations,
    validate_run_against_contracts,
)

AUTO_SWEEP = REPO_ROOT / "tools" / "auto_sweep.py"
FUTURE_PIPELINE_DOC = REPO_ROOT / "docs" / "Future_Plans" / "FUTURE_PIPELINE.md"
CONFIG_CATALOG_DOC = REPO_ROOT / "docs" / "CONFIGURATION_CATALOG.md"
MASTER_PIPELINE_DOC = REPO_ROOT / "docs" / "MASTER_PIPELINE_GUIDELINE.md"
TECHNICAL_PIPELINE_DOC = REPO_ROOT / "docs" / "TECHNICAL_PIPELINE_PAPER.md"
MANIFEST_JSON = REPO_ROOT / "docs" / "LAMBDA_CONFIGURATION_MANIFEST.json"
DEFAULT_PLAN_JSON = REPO_ROOT / "docs" / "LAMBDA_ALL_CONFIG_RUN_PLAN.json"
DEFAULT_RESULTS_JSONL = REPO_ROOT / "output" / "all_config_runs.jsonl"
DEFAULT_FAILURES_JSONL = REPO_ROOT / "output" / "all_config_failures.jsonl"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "output" / "all_config_summary.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "all_config_runs"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_all_sway_keys_from_repo() -> List[str]:
    keys = set()
    env_patterns = [
        re.compile(r'os\.environ\.get\(\s*"(SWAY_[A-Z0-9_]*[A-Z0-9])"'),
        re.compile(r'os\.getenv\(\s*"(SWAY_[A-Z0-9_]*[A-Z0-9])"'),
        re.compile(r'os\.environ\[\s*"(SWAY_[A-Z0-9_]*[A-Z0-9])"\s*\]'),
        re.compile(r'env\[\s*"(SWAY_[A-Z0-9_]*[A-Z0-9])"\s*\]'),
    ]
    ignored_roots = {"tests", "docs", "vendor", "__pycache__"}
    for py in REPO_ROOT.rglob("*.py"):
        if any(part in ignored_roots for part in py.parts):
            continue
        txt = _read_text(py)
        for pat in env_patterns:
            keys.update(pat.findall(txt))
    return sorted(keys)


def _extract_sway_keys_from_future_doc() -> List[str]:
    if not FUTURE_PIPELINE_DOC.is_file():
        return []
    return _extract_sway_keys_from_doc(FUTURE_PIPELINE_DOC)


def _extract_sway_keys_from_catalog_doc() -> List[str]:
    if not CONFIG_CATALOG_DOC.is_file():
        return []
    return _extract_sway_keys_from_doc(CONFIG_CATALOG_DOC)


def _extract_sway_keys_from_master_doc() -> List[str]:
    if not MASTER_PIPELINE_DOC.is_file():
        return []
    return _extract_sway_keys_from_doc(MASTER_PIPELINE_DOC)


def _extract_sway_keys_from_technical_doc() -> List[str]:
    if not TECHNICAL_PIPELINE_DOC.is_file():
        return []
    return _extract_sway_keys_from_doc(TECHNICAL_PIPELINE_DOC)


def _extract_sway_keys_from_doc(path: Path) -> List[str]:
    txt = _read_text(path)
    # Prefer concrete backticked keys; fallback scans for explicit uppercase tokens.
    backticked = set(re.findall(r"`(SWAY_[A-Z0-9_]+)`", txt))
    # Avoid partial captures from mixed-case words like SWAY_RTMPose_CONFIG.
    fallback = set(re.findall(r"(SWAY_[A-Z0-9_]*[A-Z0-9])(?![A-Za-z0-9_])", txt))
    keys = backticked | fallback
    short_allow = {"SWAY_FX", "SWAY_FY", "SWAY_OFFLINE"}
    out = []
    for k in sorted(keys):
        if k.endswith("_"):
            continue
        if "*" in k:
            continue
        if k.count("_") >= 2 or k in short_allow:
            out.append(k)
    return out


@dataclass
class Domain:
    kind: str  # categorical | int | float | unknown
    values: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None


def _find_trial_call(node: ast.AST) -> Optional[ast.Call]:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            fn = child.func
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "trial"
                and fn.attr.startswith("suggest_")
            ):
                return child
    return None


def _step_kw(call: ast.Call) -> Optional[float]:
    for kw in call.keywords:
        if kw.arg == "step":
            v = _safe_literal(kw.value)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _parse_domain_from_trial_call(call: ast.Call) -> Domain:
    fn = call.func
    assert isinstance(fn, ast.Attribute)
    suggest = fn.attr.replace("suggest_", "")
    if suggest == "categorical" and len(call.args) >= 2:
        values = _safe_literal(call.args[1])
        if isinstance(values, (list, tuple)):
            return Domain(kind="categorical", values=list(values))
    if suggest in ("int", "float") and len(call.args) >= 3:
        lo = _safe_literal(call.args[1])
        hi = _safe_literal(call.args[2])
        step = _step_kw(call)
        if suggest == "int" and step is None:
            step = 1.0
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return Domain(kind=suggest, low=float(lo), high=float(hi), step=step)
    return Domain(kind="unknown")


def _extract_domains_from_auto_sweep() -> Dict[str, Domain]:
    """
    Parse env["SWAY_*"] assignment expressions in auto_sweep and recover trial domains.
    """
    txt = _read_text(AUTO_SWEEP)
    tree = ast.parse(txt)
    domains: Dict[str, Domain] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "env":
                continue
            key_node = target.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
            else:
                continue
            if not key.startswith("SWAY_"):
                continue
            trial_call = _find_trial_call(node.value)
            if trial_call is None:
                continue
            domains[key] = _parse_domain_from_trial_call(trial_call)
    return domains


def _extract_default_values_from_repo() -> Dict[str, str]:
    """
    Parse simple os.environ.get("SWAY_X", "default") patterns across Python files.
    """
    defaults: Dict[str, str] = {}
    pattern = re.compile(r'os\.environ\.get\(\s*"(SWAY_[A-Z0-9_]+)"\s*,\s*(".*?"|\'.*?\'|[0-9.]+)\s*\)')
    for py in REPO_ROOT.rglob("*.py"):
        txt = _read_text(py)
        for m in pattern.finditer(txt):
            key = m.group(1)
            raw = m.group(2).strip()
            if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
                raw = raw[1:-1]
            defaults.setdefault(key, raw)
    return defaults


def _bool_like(key: str, default: Optional[str], domain: Optional[Domain]) -> bool:
    if domain and domain.kind == "categorical" and domain.values:
        vals = {str(v).lower() for v in domain.values}
        if vals.issubset({"0", "1", "true", "false", "yes", "no", "on", "off"}):
            return True
    if default is not None and str(default).lower() in {"0", "1", "true", "false", "yes", "no", "on", "off"}:
        return True
    return key.endswith("_ON") or key.endswith("_ENABLED")


def _numeric_values(lo: float, hi: float, step: Optional[float]) -> List[str]:
    if step and step > 0:
        n = int(round((hi - lo) / step)) + 1
        if n <= 12:
            vals = [lo + i * step for i in range(max(n, 0))]
            return [f"{v:g}" for v in vals]
    mid = (lo + hi) / 2.0
    return [f"{lo:g}", f"{mid:g}", f"{hi:g}"]


def _int_values(lo: float, hi: float, step: Optional[float]) -> List[str]:
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if step and step > 0:
        step_i = max(1, int(round(step)))
        vals = list(range(lo_i, hi_i + 1, step_i))
        if vals and vals[-1] != hi_i:
            vals.append(hi_i)
        return [str(v) for v in vals]
    mid_i = int(round((lo_i + hi_i) / 2.0))
    return [str(v) for v in sorted({lo_i, mid_i, hi_i})]


def _values_for_key(key: str, domain: Optional[Domain], default: Optional[str]) -> List[str]:
    # Prefer explicit domain from auto_sweep.
    if domain:
        if domain.kind == "categorical" and domain.values:
            return [str(v) for v in domain.values]
        if domain.kind == "int" and domain.low is not None and domain.high is not None:
            vals = _int_values(domain.low, domain.high, domain.step)
            return list(dict.fromkeys(vals))
        if domain.kind == "float" and domain.low is not None and domain.high is not None:
            vals = _numeric_values(domain.low, domain.high, domain.step)
            return list(dict.fromkeys(vals))

    # Known categorical surfaces that may not have explicit Optuna domains yet.
    categorical_fallbacks: Dict[str, List[str]] = {
        "SWAY_TRACKER_ENGINE": ["solidtrack", "sam2mot", "sam2_memosort_hybrid", "memosort", "matr"],
        "SWAY_LIFT_BACKEND": ["motionagformer", "motionbert", "poseformerv2"],
        "SWAY_DETECTOR_PRIMARY": ["yolo26l_dancetrack", "rt_detr_l", "rt_detr_x", "co_detr", "co_dino"],
        "SWAY_REID_FINETUNE_BASE_MODEL": ["bpbreid", "osnet"],
    }
    if key in categorical_fallbacks:
        vals = list(categorical_fallbacks[key])
        if default is not None:
            d = str(default).strip()
            if d:
                vals = [d] + [v for v in vals if v != d]
        return vals

    # Fallback heuristics.
    if _bool_like(key, default, domain):
        if default is None:
            return ["0", "1"]
        d = str(default).lower()
        if d in {"1", "true", "yes", "on"}:
            return [str(default), "0"]
        return [str(default), "1"]
    if default is not None:
        d = str(default).strip()
        # Numeric fallback: sweep around defaults when no explicit Optuna domain exists.
        # This gives every numeric key at least a low/mid/high probe in all-config sweeps.
        try:
            if re.fullmatch(r"[-+]?\d+", d):
                v = int(d)
                if v == 0:
                    return ["0", "1", "2"]
                lo = int(round(v * 0.7))
                hi = int(round(v * 1.3))
                vals = sorted({lo, v, hi})
                return [str(x) for x in vals]
            if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", d):
                v = float(d)
                if abs(v) < 1e-12:
                    return ["0", "0.1", "0.25"]
                lo = v * 0.7
                hi = v * 1.3
                vals = [lo, v, hi]
                return [f"{x:g}" for x in vals]
        except Exception:
            pass
    if default is not None:
        return [str(default)]
    return ["1"]


def _load_manifest_keys() -> List[str]:
    if MANIFEST_JSON.is_file():
        try:
            data = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
            rows = data.get("sweep_env_keys") or []
            keys = [r["key"] for r in rows if isinstance(r, dict) and isinstance(r.get("key"), str)]
            return sorted(set(keys))
        except Exception:
            pass
    return []


def build_run_plan(max_cases: int = 0, include_cartesian_discrete: bool = False) -> Dict[str, Any]:
    repo_keys = _extract_all_sway_keys_from_repo()
    manifest_keys = _load_manifest_keys()
    future_doc_keys = _extract_sway_keys_from_future_doc()
    master_doc_keys = _extract_sway_keys_from_master_doc()
    technical_doc_keys = _extract_sway_keys_from_technical_doc()
    catalog_doc_keys = _extract_sway_keys_from_catalog_doc()
    domains = _extract_domains_from_auto_sweep()
    defaults = _extract_default_values_from_repo()

    all_keys = sorted(
        set(repo_keys)
        | set(manifest_keys)
        | set(future_doc_keys)
        | set(master_doc_keys)
        | set(technical_doc_keys)
        | set(catalog_doc_keys)
    )
    doc_only_keys = sorted(set(future_doc_keys) - (set(repo_keys) | set(manifest_keys)))

    baseline_env: Dict[str, str] = {}
    for k in all_keys:
        if k in defaults:
            baseline_env[k] = str(defaults[k])

    cases: List[Dict[str, Any]] = [
        {"name": "baseline_defaults", "params": dict(baseline_env), "kind": "baseline"}
    ]
    seen_param_signatures: Set[Tuple[Tuple[str, str], ...]] = {
        tuple(sorted((k, str(v)) for k, v in baseline_env.items()))
    }

    # One-option-at-a-time matrix: run each coded option value at least once.
    for k in all_keys:
        vals = _values_for_key(k, domains.get(k), defaults.get(k))
        for v in vals:
            p = dict(baseline_env)
            p[k] = str(v)
            sig = tuple(sorted((kk, str(vv)) for kk, vv in p.items()))
            if sig in seen_param_signatures:
                continue
            seen_param_signatures.add(sig)
            cases.append(
                {
                    "name": f"{k}={v}",
                    "params": p,
                    "kind": "one_at_a_time",
                    "focus_key": k,
                }
            )

    # Optional discrete Cartesian subset only (bounded to avoid runaway).
    if include_cartesian_discrete:
        discrete_keys = []
        discrete_domains: Dict[str, List[str]] = {}
        for k, d in domains.items():
            if d.kind == "categorical" and d.values:
                vals = [str(v) for v in d.values]
                if 2 <= len(vals) <= 6:
                    discrete_keys.append(k)
                    discrete_domains[k] = vals
        # Keep only first 8 keys to avoid absurd expansion.
        discrete_keys = discrete_keys[:8]
        if discrete_keys:
            for combo in itertools.product(*(discrete_domains[k] for k in discrete_keys)):
                p = dict(baseline_env)
                name_parts = []
                for k, v in zip(discrete_keys, combo):
                    p[k] = v
                    name_parts.append(f"{k}={v}")
                sig = tuple(sorted((kk, str(vv)) for kk, vv in p.items()))
                if sig in seen_param_signatures:
                    continue
                seen_param_signatures.add(sig)
                cases.append(
                    {
                        "name": "cartesian::" + ",".join(name_parts),
                        "params": p,
                        "kind": "cartesian_discrete",
                    }
                )

    # Coverage audit on complete generated matrix.
    covered_keys_full = set()
    for c in cases:
        covered_keys_full.update(c.get("params", {}).keys())
    missing_covered_keys_full = sorted(set(all_keys) - covered_keys_full)

    selected_cases = cases[:max_cases] if max_cases > 0 else cases
    covered_keys_selected = set()
    for c in selected_cases:
        covered_keys_selected.update(c.get("params", {}).keys())
    missing_covered_keys_selected = sorted(set(all_keys) - covered_keys_selected)

    return {
        "plan_version": 1,
        "future_pipeline_doc_key_count": len(future_doc_keys),
        "future_pipeline_doc_only_keys": doc_only_keys,
        "master_pipeline_doc_key_count": len(master_doc_keys),
        "technical_pipeline_doc_key_count": len(technical_doc_keys),
        "catalog_doc_key_count": len(catalog_doc_keys),
        "strategy": {
            "default": "one_option_at_a_time",
            "includes_cartesian_discrete": include_cartesian_discrete,
            "note": "Default strategy covers every coded option without full cross-product explosion.",
        },
        "sway_key_count": len(all_keys),
        "case_count": len(selected_cases),
        "coverage": {
            "covered_key_count_full": len(covered_keys_full),
            "missing_keys_full": missing_covered_keys_full,
            "is_full_coverage": not missing_covered_keys_full,
            "covered_key_count_selected": len(covered_keys_selected),
            "missing_keys_selected": missing_covered_keys_selected,
            "is_selected_coverage": not missing_covered_keys_selected,
        },
        "cases": selected_cases,
    }


def _parse_feature_line(line: str) -> Optional[Dict[str, str]]:
    m = re.search(
        r"\[feature\]\s*([^:]+):\s*requested=([a-zA-Z]+),\s*runtime=([a-zA-Z]+),\s*wiring=([a-zA-Z]+)",
        line,
    )
    if not m:
        return None
    return {
        "name": m.group(1).strip(),
        "requested": m.group(2).strip().lower(),
        "runtime": m.group(3).strip().lower(),
        "wiring": m.group(4).strip().lower(),
    }


def _boolish_on(v: str) -> bool:
    return v.strip().lower() in {"on", "1", "true", "yes"}


def _resolve_execute_boundary(
    strict_mode: str,
    stop_after_boundary: str,
    params: Dict[str, Any],
) -> str:
    """
    Map CLI --strict-mode to the subprocess --stop-after-boundary value.

    - off: use stop_after_boundary as given.
    - quick: same, but auto-escalate to final when a triggered contract
      requires the full pipeline.
    - full: always final (strict-full profile).
    """
    if strict_mode == "off":
        return stop_after_boundary
    if strict_mode == "full":
        return "final"
    env_map = {str(k): str(v) for k, v in (params or {}).items()}
    if contracts_requiring_full_pipeline(env_map):
        return "final"
    return stop_after_boundary


def _run_case(
    case: Dict[str, Any],
    video: Path,
    output_root: Path,
    stop_after_boundary: str,
    index: int,
    total: int,
    log_root: Path,
    fail_on_unwired_extras: bool,
    strict_mode: str = "off",
) -> Dict[str, Any]:
    run_name = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", case["name"])[:120]
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{index:04d}_{run_name}.log"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(case["params"], tf, sort_keys=True)
        params_path = Path(tf.name)

    effective_stop = _resolve_execute_boundary(strict_mode, stop_after_boundary, case.get("params") or {})
    cmd = [
        sys.executable,
        "main.py",
        str(video),
        "--output-dir",
        str(run_dir),
        "--params",
        str(params_path),
        "--stop-after-boundary",
        effective_stop,
    ]
    env = os.environ.copy()
    if fail_on_unwired_extras:
        env["SWAY_FAIL_ON_UNWIRED_EXTRAS"] = "1"

    t0_wall = time.time()
    feature_failures: List[str] = []
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# case={case['name']}\n")
        logf.write(f"# index={index}/{total}\n")
        logf.write(f"# strict_mode={strict_mode}\n")
        logf.write(f"# effective_stop_after_boundary={effective_stop}\n")
        logf.write(f"# cmd={' '.join(cmd)}\n")
        logf.write(f"# started_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0_wall))}\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(f"[{index}/{total}] {case['name']} | {line}", flush=True)
            logf.write(raw)

            feat = _parse_feature_line(line)
            if feat and _boolish_on(feat["requested"]):
                runtime_ok = _boolish_on(feat["runtime"])
                wiring_ok = feat["wiring"] == "wired"
                if not (runtime_ok and wiring_ok):
                    feature_failures.append(
                        f"{feat['name']}: requested={feat['requested']}, runtime={feat['runtime']}, wiring={feat['wiring']}"
                    )

        rc = proc.wait()

        elapsed_s = round(time.time() - t0_wall, 3)
        logf.write(f"\n# exit_code={rc}\n")
        logf.write(f"# elapsed_s={elapsed_s}\n")
        if feature_failures:
            logf.write("# feature_failures:\n")
            for ff in feature_failures:
                logf.write(f"#   - {ff}\n")
        logf.flush()

    elapsed_s = round(time.time() - t0_wall, 3)

    contract_violations: List[Dict[str, str]] = []
    if strict_mode != "off":
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except OSError:
            log_text = ""
        env_map = {str(k): str(v) for k, v in (case.get("params") or {}).items()}
        violations = validate_run_against_contracts(
            log_text, env_map, run_dir, effective_stop or ""
        )
        contract_violations = [
            {
                "contract_name": v.contract_name,
                "clause": v.clause,
                "detail": v.detail,
                "severity": v.severity,
            }
            for v in violations
        ]
        if violations:
            print(format_violations(violations), flush=True)

    return {
        "name": case["name"],
        "kind": case.get("kind"),
        "return_code": rc,
        "success": rc == 0 and not feature_failures and not contract_violations,
        "feature_failures": feature_failures,
        "contract_violations": contract_violations,
        "strict_mode": strict_mode,
        "effective_stop_after_boundary": effective_stop,
        "duration_s": elapsed_s,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "params_path": str(params_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all coded SWAY configuration options.")
    parser.add_argument(
        "--video",
        type=Path,
        required=False,
        help="Absolute or repo-relative input video path (required only with --execute).",
    )
    parser.add_argument(
        "--stop-after-boundary",
        default="after_phase_3",
        help="Pipeline boundary for faster runs (default: after_phase_3).",
    )
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN_JSON, help="Write generated plan JSON here.")
    parser.add_argument(
        "--results-out",
        type=Path,
        default=DEFAULT_RESULTS_JSONL,
        help="Write execution results JSONL here (when --execute).",
    )
    parser.add_argument(
        "--failures-out",
        type=Path,
        default=DEFAULT_FAILURES_JSONL,
        help="Write failed case rows JSONL here (when --execute).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Write aggregate execution summary JSON here (when --execute).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for pipeline runs.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Directory for per-case live logs (default: <output-root>/logs).",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Cap number of cases (0 means all).")
    parser.add_argument(
        "--start-case-index",
        type=int,
        default=1,
        help="1-based case index to start execution from (default: 1).",
    )
    parser.add_argument(
        "--end-case-index",
        type=int,
        default=0,
        help="1-based case index to end execution at, inclusive (0 means through final case).",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append to existing results/failures files instead of truncating.",
    )
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="Skip cases already present in --results-out by case name.",
    )
    parser.add_argument(
        "--include-cartesian-discrete",
        action="store_true",
        help="Also add bounded Cartesian runs for discrete domains.",
    )
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail plan generation if any discovered SWAY_* key is not covered by generated cases.",
    )
    parser.add_argument(
        "--fail-on-unwired-extras",
        action="store_true",
        help="Set SWAY_FAIL_ON_UNWIRED_EXTRAS=1 for each case to fail fast on requested-but-unwired modules.",
    )
    parser.add_argument(
        "--strict-mode",
        choices=("off", "quick", "full"),
        default="off",
        help=(
            "Technology guardrails after each run (log + outputs vs sway/technology_contracts). "
            "'quick' = validate at resolved boundary; auto-escalate to final when a contract "
            "requires full pipeline. 'full' = always run to final + full contract checks. "
            "Only applies with --execute."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Execute plan; otherwise write plan only.")
    args = parser.parse_args()

    plan = build_run_plan(
        max_cases=args.max_cases,
        include_cartesian_discrete=args.include_cartesian_discrete,
    )
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run plan: {args.plan_out} ({plan['case_count']} cases)")
    coverage = plan.get("coverage") or {}
    missing_keys_full = coverage.get("missing_keys_full") or []
    missing_keys_selected = coverage.get("missing_keys_selected") or []
    if missing_keys_full:
        print(f"[coverage] missing SWAY keys in full generated matrix: {', '.join(missing_keys_full)}")
    else:
        print("[coverage] full SWAY key coverage confirmed in generated matrix.")
    if missing_keys_selected:
        print(f"[coverage] selected run (after --max-cases) omits keys: {', '.join(missing_keys_selected)}")
    else:
        print("[coverage] selected run set covers all SWAY keys.")
    if args.strict_coverage and missing_keys_full:
        raise SystemExit("Strict coverage requested but full generated matrix has missing keys.")
    if args.strict_coverage and args.max_cases > 0 and missing_keys_selected:
        print("[coverage] strict mode validated full matrix; selected subset is intentionally partial due --max-cases.")

    if not args.execute:
        return

    if args.video is None:
        raise SystemExit("--video is required when --execute is set.")
    video = args.video
    if not video.is_absolute():
        video = (REPO_ROOT / video).resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.failures_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.log_root or (args.output_root / "logs")
    log_root.mkdir(parents=True, exist_ok=True)

    total_cases = len(plan["cases"])
    start_idx = max(1, int(args.start_case_index))
    end_idx = total_cases if int(args.end_case_index) <= 0 else min(total_cases, int(args.end_case_index))
    if start_idx > end_idx:
        raise SystemExit(f"Invalid case window: start={start_idx} end={end_idx} total={total_cases}")
    selected_cases = plan["cases"][start_idx - 1 : end_idx]
    if args.skip_existing_results and args.results_out.exists():
        existing_names: Set[str] = set()
        for line in args.results_out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            name = row.get("name")
            if isinstance(name, str):
                existing_names.add(name)
        before = len(selected_cases)
        selected_cases = [c for c in selected_cases if c.get("name") not in existing_names]
        skipped = before - len(selected_cases)
        print(f"Skip-existing enabled: skipped {skipped} already-recorded cases.")
    print(f"Executing {total_cases} cases with live logs at: {log_root}")
    print(f"Video: {video}")
    if args.strict_mode != "off":
        print(f"Strict technology guardrails: --strict-mode={args.strict_mode}", flush=True)
    if start_idx != 1 or end_idx != total_cases:
        print(f"Case window: {start_idx}..{end_idx} (executing {len(selected_cases)} cases)")

    success_count = 0
    failed_count = 0

    results_mode = "a" if args.append_results else "w"
    failures_mode = "a" if args.append_results else "w"
    with (
        args.results_out.open(results_mode, encoding="utf-8") as rf,
        args.failures_out.open(failures_mode, encoding="utf-8") as ff,
    ):
        for i, case in enumerate(selected_cases, start=start_idx):
            print(f"\n=== [{i}/{total_cases}] {case['name']} ===")
            result = _run_case(
                case=case,
                video=video,
                output_root=args.output_root,
                stop_after_boundary=args.stop_after_boundary,
                index=i,
                total=total_cases,
                log_root=log_root,
                fail_on_unwired_extras=args.fail_on_unwired_extras,
                strict_mode=args.strict_mode,
            )
            rf.write(json.dumps(result) + "\n")
            rf.flush()
            if result.get("success"):
                success_count += 1
            else:
                failed_count += 1
                ff.write(json.dumps(result) + "\n")
                ff.flush()

    results_lines = 0
    failures_lines = 0
    if args.results_out.exists():
        results_lines = sum(1 for line in args.results_out.read_text(encoding="utf-8").splitlines() if line.strip())
    if args.failures_out.exists():
        failures_lines = sum(1 for line in args.failures_out.read_text(encoding="utf-8").splitlines() if line.strip())
    summary = {
        "case_count_total_plan": total_cases,
        "executed_window": {"start_case_index": start_idx, "end_case_index": end_idx, "executed_count": len(selected_cases)},
        "success_count_this_invocation": success_count,
        "failed_count_this_invocation": failed_count,
        "results_line_count_total_file": results_lines,
        "failures_line_count_total_file": failures_lines,
        "video": str(video),
        "plan_out": str(args.plan_out),
        "results_out": str(args.results_out),
        "failures_out": str(args.failures_out),
        "log_root": str(log_root),
        "stop_after_boundary": args.stop_after_boundary,
        "strict_mode": args.strict_mode,
        "strict_coverage": bool(args.strict_coverage),
        "coverage": plan.get("coverage"),
    }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results: {args.results_out}")
    print(f"Wrote failures: {args.failures_out}")
    print(f"Wrote summary: {args.summary_out}")
    print(f"Execution complete: success={success_count} failed={failed_count}")


if __name__ == "__main__":
    main()

