"""
Shared checkpoint fan-out tree orchestration for Pipeline Lab.

Used by ``pipeline_tree_queue`` CLI and the Lab API (multi-video uploads).
Queues stages over HTTP to ``POST /api/runs/batch_path`` and waits between stages.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required (pip install PyYAML)") from e


class TreeQueueError(Exception):
    """Recoverable / user-facing tree queue failure."""


def load_tree_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TreeQueueError("Tree YAML must be a mapping at the root")
    return raw


def resolve_video_paths_from_doc(doc: Dict[str, Any]) -> List[str]:
    """Paths from ``video_paths`` (list) or legacy ``video_path`` (string)."""
    vp_list = doc.get("video_paths")
    if isinstance(vp_list, list) and vp_list:
        out = [str(x).strip() for x in vp_list if str(x).strip()]
        if out:
            return out
    single = str(doc.get("video_path") or "").strip()
    if single:
        return [single]
    return []


def validate_stages(stages: Any) -> List[Dict[str, Any]]:
    if not isinstance(stages, list) or not stages:
        raise TreeQueueError("Tree YAML: 'stages' must be a non-empty list")
    out: List[Dict[str, Any]] = []
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            raise TreeQueueError(f"stages[{i}] must be a mapping")
        name = str(st.get("name") or f"stage_{i}")
        variants = st.get("variants")
        if not isinstance(variants, list) or not variants:
            raise TreeQueueError(f"Stage {name!r}: need at least one variant")
        for j, v in enumerate(variants):
            if not isinstance(v, dict) or not str(v.get("recipe_name") or "").strip():
                raise TreeQueueError(f"Stage {name!r} variant[{j}]: need recipe_name")
            fld = v.get("fields")
            if fld is not None and not isinstance(fld, dict):
                raise TreeQueueError(f"Stage {name!r} variant[{j}]: fields must be a mapping")
            v.setdefault("fields", {})
        if i == 0:
            if not str(st.get("stop_after_boundary") or "").strip():
                raise TreeQueueError(f"Stage {name!r}: first stage needs stop_after_boundary")
        else:
            if not str(st.get("expect_boundary") or "").strip():
                raise TreeQueueError(f"Stage {name!r}: expect_boundary required when resuming")
            if not str(st.get("resume_checkpoint") or "").strip():
                raise TreeQueueError(f"Stage {name!r}: resume_checkpoint (subdir name) required")
        out.append({**st, "name": name})
    return out


def build_specs_stage0(stage: Dict[str, Any]) -> List[Dict[str, Any]]:
    sab = str(stage["stop_after_boundary"])
    specs: List[Dict[str, Any]] = []
    for v in stage["variants"]:
        specs.append(
            {
                "recipe_name": str(v["recipe_name"]),
                "fields": dict(v.get("fields") or {}),
                "checkpoint": {"stop_after_boundary": sab},
            }
        )
    return specs


def build_specs_resume(stage: Dict[str, Any], parents: List[str]) -> List[Dict[str, Any]]:
    expect = str(stage["expect_boundary"])
    resume_ck = str(stage["resume_checkpoint"])
    sab = str(stage.get("stop_after_boundary") or "").strip()
    force = bool(stage.get("force_checkpoint_load", True))
    specs: List[Dict[str, Any]] = []
    for parent in parents:
        for v in stage["variants"]:
            ck: Dict[str, Any] = {
                "resume_from": f"../{parent}/output/checkpoints/{resume_ck}",
                "expect_boundary": expect,
            }
            if sab:
                ck["stop_after_boundary"] = sab
            if force:
                ck["force_checkpoint_load"] = True
            specs.append(
                {
                    "recipe_name": str(v["recipe_name"]),
                    "fields": dict(v.get("fields") or {}),
                    "checkpoint": ck,
                }
            )
    return specs


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    timeout_s: float = 600.0,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
        data = raw
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise TreeQueueError(f"HTTP {e.code} {url}: {detail[:2000]}") from e
    except urllib.error.URLError as e:
        raise TreeQueueError(f"Request failed {url}: {e}") from e
    if not body.strip():
        return None
    return json.loads(body)


def wait_stage(
    base: str,
    run_ids: List[str],
    *,
    poll_s: float,
    log: Callable[[str], None],
    missing_grace_polls: int,
) -> Tuple[List[str], List[str]]:
    pending = set(run_ids)
    ok: List[str] = []
    bad: List[str] = []
    absent_streak: Dict[str, int] = {str(rid): 0 for rid in run_ids}
    grace = max(1, int(missing_grace_polls))

    while pending:
        rows = http_json("GET", f"{base}/api/runs", None, timeout_s=120.0)
        if not isinstance(rows, list):
            time.sleep(poll_s)
            continue
        by_id = {str(r.get("run_id")): r for r in rows if isinstance(r, dict) and r.get("run_id")}
        for rid in list(pending):
            row = by_id.get(rid)
            if not row:
                absent_streak[rid] = absent_streak.get(rid, 0) + 1
                if absent_streak[rid] >= grace:
                    pending.discard(rid)
                    bad.append(rid)
                    log(
                        f"  … run {rid[:8]}… missing from GET /api/runs after {grace} poll(s) "
                        f"(treated as removed)"
                    )
                continue
            absent_streak[rid] = 0
            st = str(row.get("status") or "")
            if st in ("done", "error", "cancelled"):
                pending.discard(rid)
                if st == "done":
                    ok.append(rid)
                else:
                    bad.append(rid)
        if pending:
            log(f"  … {len(pending)} run(s) still in progress (queued/running)")
            time.sleep(poll_s)
    return ok, bad


def reconcile_ok_via_manifest(
    base: str,
    bad_ids: List[str],
) -> Tuple[List[str], List[str]]:
    rec_ok: List[str] = []
    still_bad: List[str] = []
    for rid in bad_ids:
        try:
            detail = http_json("GET", f"{base}/api/runs/{rid}", None, timeout_s=120.0)
        except TreeQueueError:
            still_bad.append(rid)
            continue
        if not isinstance(detail, dict):
            still_bad.append(rid)
            continue
        if str(detail.get("status") or "") == "done":
            rec_ok.append(rid)
            continue
        if _manifest_indicates_success(detail.get("manifest")):
            rec_ok.append(rid)
            continue
        still_bad.append(rid)
    return rec_ok, still_bad


def _manifest_indicates_success(manifest: Any) -> bool:
    if not isinstance(manifest, dict):
        return False
    if manifest.get("total_elapsed_s") is None:
        return False
    return bool(manifest.get("pipeline_stages") or manifest.get("final_video_relpath"))


def run_checkpoint_tree_session(
    *,
    base: str,
    stages: List[Dict[str, Any]],
    video_paths: List[Path],
    batch_id: str,
    poll_s: float = 10.0,
    continue_on_error: bool = True,
    missing_grace_polls: int = 6,
    strict_status: bool = False,
    log: Optional[Callable[[str], None]] = None,
    # Optional display label per video (Lab video_stem); defaults to each file's stem.
    video_display_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    For each video (sequential): run all stages, then the next video.
    One shared ``batch_id`` for the Lab UI.
    """
    _log = log or (lambda _m: None)
    base = base.rstrip("/")
    if not video_paths:
        raise TreeQueueError("At least one video path is required")

    health = http_json("GET", f"{base}/api/health", None, timeout_s=15.0)
    if not isinstance(health, dict) or health.get("ok") != "true":
        raise TreeQueueError(f"Lab health check failed at {base}/api/health: {health!r}")

    all_stage_run_ids: List[str] = []

    for vi, vp in enumerate(video_paths):
        vp = vp.expanduser()
        if not vp.is_file():
            raise TreeQueueError(f"video_path must exist: {vp}")
        if (
            video_display_labels
            and vi < len(video_display_labels)
            and str(video_display_labels[vi] or "").strip()
        ):
            source_label = str(video_display_labels[vi]).strip()
        else:
            source_label = vp.stem
        _log(f"\n=== Video {vi + 1}/{len(video_paths)}: {source_label!r} ({vp.name}) ===")
        parents: List[str] = []

        for si, stage in enumerate(stages):
            if si == 0:
                specs = build_specs_stage0(stage)
            else:
                if not parents:
                    raise TreeQueueError(
                        f"No parent runs for video {source_label!r} after stage {si} "
                        f"(previous stage produced no successful parents)."
                    )
                specs = build_specs_resume(stage, parents)

            body: Dict[str, Any] = {
                "video_path": str(vp.resolve()),
                "runs": specs,
                "batch_id": batch_id,
                "source_label": source_label,
            }

            _log(
                f"\nPOST {base}/api/runs/batch_path — video {vi + 1}/{len(video_paths)} "
                f"stage {si + 1}/{len(stages)} {stage['name']!r} ({len(specs)} run(s))"
            )
            resp = http_json("POST", f"{base}/api/runs/batch_path", body, timeout_s=900.0)
            if not isinstance(resp, dict):
                raise TreeQueueError(f"Unexpected batch_path response: {resp!r}")
            run_ids = resp.get("run_ids")
            if not isinstance(run_ids, list) or len(run_ids) != len(specs):
                raise TreeQueueError("batch_path returned wrong number of run_ids")
            run_ids = [str(x) for x in run_ids]
            all_stage_run_ids.extend(run_ids)

            _log("Waiting for this stage to finish…")
            ok, bad = wait_stage(
                base,
                run_ids,
                poll_s=poll_s,
                log=_log,
                missing_grace_polls=missing_grace_polls,
            )
            if bad and not strict_status:
                rec_ok, bad = reconcile_ok_via_manifest(base, bad)
                if rec_ok:
                    _log(
                        f"  … reconciled {len(rec_ok)} run(s) via run_manifest.json "
                        f"(API status was error/cancelled but disk shows a completed pipeline)"
                    )
                    ok.extend(rec_ok)
            if bad:
                _log(
                    f"  Stage ended with {len(bad)} non-success run(s): "
                    f"{[b[:8] for b in bad[:12]]}{'…' if len(bad) > 12 else ''}"
                )
                if not continue_on_error:
                    raise TreeQueueError(
                        "Aborting tree (--abort-on-error): omit it to continue with successful parents only."
                    )
            parents = ok
            if not parents:
                n_st = len(stages)
                hint = (
                    f"No successful runs after stage {si + 1}/{n_st} {stage['name']!r} for video {source_label!r}. "
                    f"Open the ?batch= URL, inspect errors, then retry with:\n"
                    f"  python3 -m tools.retry_failed_batch_runs --batch-id {batch_id!r} "
                    "--recipe-prefix '' --disk-incomplete"
                )
                if si + 1 < n_st:
                    raise TreeQueueError(f"No successful runs to fan out from; stopping.\n{hint}")
                raise TreeQueueError(f"Final stage produced zero successful leaf runs.\n{hint}")

    _log("\n=== Tree complete (all videos) ===")
    return {
        "batch_id": batch_id,
        "run_ids": all_stage_run_ids,
        "video_count": len(video_paths),
    }
