"""
Strict per-technology behavioral contracts derived from CONFIGURATION_CATALOG.md.

Each contract declares what MUST be true when a technology is enabled:
  - required feature activation signals
  - allowed pipeline branch
  - required output artifacts
  - forbidden runtime conditions

The validate_run_against_contracts() function checks a completed run's log
and output directory against all applicable contracts and returns hard failures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TechnologyContract:
    """Behavioral contract for one technology or technology group."""

    name: str
    phase: str
    trigger_env_keys: Tuple[str, ...]
    trigger_values: Dict[str, str] = field(default_factory=dict)

    required_feature_signals: Tuple[str, ...] = ()
    allowed_branch: str = "any"  # "future", "boxmot", "any"
    forbidden_branch: str = ""   # "boxmot", "future", ""

    required_artifacts: Tuple[str, ...] = ()
    min_tracks_after_phase3: int = -1  # -1 = no check
    max_warning_storm_count: int = 200
    forbidden_log_patterns: Tuple[str, ...] = ()

    must_reach_phase: int = 0  # 0 = no requirement beyond what boundary says
    full_pipeline_required: bool = False


# ---------------------------------------------------------------------------
# Contract registry
# ---------------------------------------------------------------------------

CONTRACTS: List[TechnologyContract] = [
    # ── Phase 1: Detectors ──────────────────────────────────────────────
    TechnologyContract(
        name="DETR detector family",
        phase="1",
        trigger_env_keys=("SWAY_DETECTOR_PRIMARY",),
        trigger_values={"SWAY_DETECTOR_PRIMARY": "rt_detr_l|rt_detr_x|co_detr|co_dino"},
        allowed_branch="future",
        forbidden_branch="boxmot",
    ),
    TechnologyContract(
        name="Hybrid detector",
        phase="1",
        trigger_env_keys=("SWAY_DETECTOR_HYBRID",),
        trigger_values={"SWAY_DETECTOR_HYBRID": "1|true|yes|on"},
    ),

    # ── Phase 2: Trackers ───────────────────────────────────────────────
    TechnologyContract(
        name="SAM2MOT tracker",
        phase="2",
        trigger_env_keys=("SWAY_TRACKER_ENGINE",),
        trigger_values={"SWAY_TRACKER_ENGINE": "sam2mot"},
        allowed_branch="future",
        forbidden_branch="boxmot",
    ),
    TechnologyContract(
        name="SAM2+MeMoSORT hybrid tracker",
        phase="2",
        trigger_env_keys=("SWAY_TRACKER_ENGINE",),
        trigger_values={"SWAY_TRACKER_ENGINE": "sam2_memosort_hybrid"},
        allowed_branch="future",
        forbidden_branch="boxmot",
    ),
    TechnologyContract(
        name="MATR tracker",
        phase="2",
        trigger_env_keys=("SWAY_TRACKER_ENGINE",),
        trigger_values={"SWAY_TRACKER_ENGINE": "matr"},
        allowed_branch="future",
        forbidden_branch="boxmot",
    ),
    TechnologyContract(
        name="BoxMOT Deep OC-SORT",
        phase="2",
        trigger_env_keys=("SWAY_BOXMOT_TRACKER",),
        trigger_values={"SWAY_BOXMOT_TRACKER": "deepocsort"},
        allowed_branch="boxmot",
        forbidden_branch="future",
    ),
    TechnologyContract(
        name="BoxMOT ByteTrack",
        phase="2",
        trigger_env_keys=("SWAY_BOXMOT_TRACKER",),
        trigger_values={"SWAY_BOXMOT_TRACKER": "bytetrack"},
        allowed_branch="boxmot",
        forbidden_branch="future",
    ),
    TechnologyContract(
        name="Enrollment gallery",
        phase="2",
        trigger_env_keys=("SWAY_ENROLLMENT_ENABLED",),
        trigger_values={"SWAY_ENROLLMENT_ENABLED": "1|true|yes|on"},
        required_artifacts=("gallery.json",),
    ),
    TechnologyContract(
        name="COI module",
        phase="2",
        trigger_env_keys=("SWAY_COI_ENABLED",),
        trigger_values={"SWAY_COI_ENABLED": "1|true|yes|on"},
    ),

    # ── Phase 2X: Optional modules ─────────────────────────────────────
    TechnologyContract(
        name="Backward pass",
        phase="X",
        trigger_env_keys=("SWAY_BACKWARD_PASS_ENABLED",),
        trigger_values={"SWAY_BACKWARD_PASS_ENABLED": "1|true|yes|on"},
        required_feature_signals=("BackwardPass",),
    ),
    TechnologyContract(
        name="MOTE disocclusion",
        phase="X",
        trigger_env_keys=("SWAY_MOTE_DISOCCLUSION",),
        trigger_values={"SWAY_MOTE_DISOCCLUSION": "1|true|yes|on"},
        required_feature_signals=("MOTE",),
    ),
    TechnologyContract(
        name="SentinelSBM",
        phase="X",
        trigger_env_keys=("SWAY_SENTINEL_SBM",),
        trigger_values={"SWAY_SENTINEL_SBM": "1|true|yes|on"},
        required_feature_signals=("SentinelSBM",),
    ),
    TechnologyContract(
        name="UMOT backtracking",
        phase="X",
        trigger_env_keys=("SWAY_UMOT_BACKTRACK",),
        trigger_values={"SWAY_UMOT_BACKTRACK": "1|true|yes|on"},
        required_feature_signals=("UMOTBacktracker",),
    ),

    # ── Phase 3: Post-track stitching ───────────────────────────────────
    TechnologyContract(
        name="Global linker + AFLink",
        phase="3",
        trigger_env_keys=("SWAY_GLOBAL_LINK",),
        trigger_values={"SWAY_GLOBAL_LINK": "1|true|yes|on"},
    ),
    TechnologyContract(
        name="GNN track refine",
        phase="3",
        trigger_env_keys=("SWAY_GNN_TRACK_REFINE",),
        trigger_values={"SWAY_GNN_TRACK_REFINE": "1|true|yes|on"},
    ),

    # Phase 4–11 contracts are validated only on full pipeline runs
    # (see full_pipeline_required + stop_boundary gating below).
    TechnologyContract(
        name="Mask-guided pose",
        phase="5",
        trigger_env_keys=("SWAY_POSE_MASK_GUIDED",),
        trigger_values={"SWAY_POSE_MASK_GUIDED": "1|true|yes|on"},
        full_pipeline_required=True,
    ),
    TechnologyContract(
        name="Temporal pose refine",
        phase="5",
        trigger_env_keys=("SWAY_TEMPORAL_POSE_REFINE",),
        trigger_values={"SWAY_TEMPORAL_POSE_REFINE": "1|true|yes|on"},
        full_pipeline_required=True,
    ),
    TechnologyContract(
        name="3D Lift",
        phase="5b",
        trigger_env_keys=("SWAY_3D_LIFT",),
        trigger_values={"SWAY_3D_LIFT": "1|true|yes|on"},
        full_pipeline_required=True,
    ),
    TechnologyContract(
        name="HMR mesh sidecar",
        phase="11",
        trigger_env_keys=("SWAY_HMR_MESH_SIDECAR",),
        trigger_values={"SWAY_HMR_MESH_SIDECAR": "1|true|yes|on"},
        full_pipeline_required=True,
    ),
]

CONTRACT_BY_NAME: Dict[str, TechnologyContract] = {c.name: c for c in CONTRACTS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_matches_trigger(env: Dict[str, str], contract: TechnologyContract) -> bool:
    """Return True if the run's env activates this contract."""
    if not contract.trigger_env_keys:
        return False
    if not contract.trigger_values:
        return any(str(env.get(k, "")).strip() for k in contract.trigger_env_keys)
    for key in contract.trigger_env_keys:
        val = str(env.get(key, "")).strip().lower()
        if not val:
            continue
        allowed = contract.trigger_values.get(key)
        if allowed is None:
            continue
        opts = [x.strip() for x in allowed.split("|") if x.strip()]
        if val in opts:
            return True
    return False


_PHASE_MARKER_RE = re.compile(r"\[(\d+)/11\]")
_FEATURE_RE = re.compile(
    r"\[feature\]\s*([^:]+):\s*requested=([a-zA-Z]+),\s*runtime=([a-zA-Z]+),\s*wiring=([a-zA-Z]+)"
)
_TRACKER_WARNING_RE = re.compile(r"\[tracker warning\]")
_BACKWARD_SUMMARY_RE = re.compile(r"\[backward\]\s*([\d.]+)s\s*—\s*(\d+)\s*merged tracks\s*\((\d+)\s*with gap fill\)")


def _detect_branch(log: str) -> str:
    """Infer which tracker branch was used: 'future', 'boxmot', or 'unknown'."""
    if "future pipeline:" in log.lower():
        return "future"
    if "BoxMOT" in log:
        return "boxmot"
    return "unknown"


def _count_phase_starts(log: str) -> Dict[int, int]:
    """Count how many times each [N/11] marker appears."""
    counts: Dict[int, int] = {}
    for m in _PHASE_MARKER_RE.finditer(log):
        n = int(m.group(1))
        counts[n] = counts.get(n, 0) + 1
    return counts


def _parse_features(log: str) -> Dict[str, Dict[str, str]]:
    """Parse all [feature] lines into {name: {requested, runtime, wiring}}."""
    features: Dict[str, Dict[str, str]] = {}
    for m in _FEATURE_RE.finditer(log):
        features[m.group(1).strip()] = {
            "requested": m.group(2).strip().lower(),
            "runtime": m.group(3).strip().lower(),
            "wiring": m.group(4).strip().lower(),
        }
    return features


def _count_tracker_warnings(log: str) -> int:
    return len(_TRACKER_WARNING_RE.findall(log))


def _extract_kept_tracks(log: str) -> Optional[Tuple[int, int]]:
    """Parse 'Kept N of M tracks after pre-pose pruning'. Returns (kept, total) or None."""
    m = re.search(r"Kept\s+(\d+)\s+of\s+(\d+)\s+tracks\s+after\s+pre-pose\s+pruning", log)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _extract_raw_track_count(log: str) -> Optional[int]:
    m = re.search(r"\((\d+)\s+raw\s+tracks\)", log)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

@dataclass
class ContractViolation:
    contract_name: str
    clause: str
    detail: str
    severity: str = "FATAL"


def validate_global_pipeline_invariants(log: str) -> List[ContractViolation]:
    """Orchestration + stability checks that apply to every run (no env needed)."""
    violations: List[ContractViolation] = []
    phase_counts = _count_phase_starts(log)
    warning_count = _count_tracker_warnings(log)
    kept = _extract_kept_tracks(log)
    raw_tracks = _extract_raw_track_count(log)

    for phase_num, count in phase_counts.items():
        if count > 1:
            violations.append(
                ContractViolation(
                    contract_name="GLOBAL",
                    clause="phase_no_restart",
                    detail=f"Phase [{phase_num}/11] appeared {count} times in one run (illegal restart/fallthrough)",
                )
            )

    positions = [int(m.group(1)) for m in _PHASE_MARKER_RE.finditer(log)]
    prev = 0
    for p in positions:
        if p < prev:
            violations.append(
                ContractViolation(
                    contract_name="GLOBAL",
                    clause="phase_ordering",
                    detail=f"Phase [{p}/11] appeared after [{prev}/11] (out of order)",
                )
            )
            break
        prev = p

    if warning_count > 200:
        violations.append(
            ContractViolation(
                contract_name="GLOBAL",
                clause="tracker_warning_storm",
                detail=f"Tracker warnings: {warning_count} (threshold: 200). Likely numerical instability.",
            )
        )

    if kept is not None and raw_tracks is not None:
        if raw_tracks > 0 and kept[1] > 0 and kept[0] == 0:
            violations.append(
                ContractViolation(
                    contract_name="GLOBAL",
                    clause="zero_survivor_collapse",
                    detail=(
                        f"Detected {raw_tracks} raw tracks, {kept[1]} entered pruning, but 0 survived. "
                        "Likely pipeline or config defect."
                    ),
                )
            )

    return violations


def validate_run_against_contracts(
    log: str,
    env: Dict[str, str],
    output_dir: Optional[Path] = None,
    stop_boundary: str = "",
) -> List[ContractViolation]:
    """
    Validate a completed pipeline run against all applicable technology contracts.
    Returns a list of violations (empty = all contracts satisfied).
    """
    violations: List[ContractViolation] = []
    violations.extend(validate_global_pipeline_invariants(log))

    branch = _detect_branch(log)
    features = _parse_features(log)

    early_stop = bool(stop_boundary and stop_boundary not in ("final", "", "none"))

    # ── Per-technology contract checks ──────────────────────────────────

    for contract in CONTRACTS:
        if contract.full_pipeline_required and early_stop:
            continue
        if not _env_matches_trigger(env, contract):
            continue

        # Branch exclusivity
        if contract.forbidden_branch and branch == contract.forbidden_branch:
            violations.append(ContractViolation(
                contract_name=contract.name,
                clause="forbidden_branch",
                detail=f"Technology '{contract.name}' forbids branch '{contract.forbidden_branch}' "
                        f"but run used branch '{branch}'",
            ))

        if contract.allowed_branch not in ("any", "") and branch not in (contract.allowed_branch, "unknown"):
            violations.append(ContractViolation(
                contract_name=contract.name,
                clause="wrong_branch",
                detail=f"Technology '{contract.name}' requires branch '{contract.allowed_branch}' "
                        f"but run used branch '{branch}'",
            ))

        # Feature activation signals
        for sig_name in contract.required_feature_signals:
            feat = features.get(sig_name)
            if feat is None:
                violations.append(ContractViolation(
                    contract_name=contract.name,
                    clause="missing_feature_signal",
                    detail=f"Expected [feature] signal for '{sig_name}' but none found in log",
                ))
            elif feat["requested"] in ("on", "1", "true", "yes"):
                if feat["runtime"] not in ("on", "active", "1", "true", "yes"):
                    violations.append(ContractViolation(
                        contract_name=contract.name,
                        clause="feature_not_active",
                        detail=f"Feature '{sig_name}' requested=on but runtime={feat['runtime']}",
                    ))
                if feat["wiring"] != "wired":
                    violations.append(ContractViolation(
                        contract_name=contract.name,
                        clause="feature_not_wired",
                        detail=f"Feature '{sig_name}' wiring={feat['wiring']} (expected wired)",
                    ))

        # Required artifacts
        if output_dir is not None:
            for artifact in contract.required_artifacts:
                p = output_dir / artifact
                if not p.exists():
                    violations.append(ContractViolation(
                        contract_name=contract.name,
                        clause="missing_artifact",
                        detail=f"Required artifact '{artifact}' not found at {p}",
                    ))

        # Forbidden log patterns
        for pattern in contract.forbidden_log_patterns:
            if re.search(pattern, log):
                violations.append(ContractViolation(
                    contract_name=contract.name,
                    clause="forbidden_log_pattern",
                    detail=f"Forbidden pattern '{pattern}' found in log for '{contract.name}'",
                ))

    return violations


def contracts_requiring_full_pipeline(env: Dict[str, str]) -> List[TechnologyContract]:
    """Return contracts triggered by env that require full pipeline (not just phase 3)."""
    return [c for c in CONTRACTS if _env_matches_trigger(env, c) and c.full_pipeline_required]


def format_violations(violations: List[ContractViolation]) -> str:
    if not violations:
        return "All technology contracts satisfied."
    lines = [f"TECHNOLOGY CONTRACT VIOLATIONS ({len(violations)}):"]
    for v in violations:
        lines.append(f"  [{v.severity}] {v.contract_name} / {v.clause}: {v.detail}")
    return "\n".join(lines)
