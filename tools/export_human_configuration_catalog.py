#!/usr/bin/env python3
"""
Export concise human-readable catalog of technologies + configuration options.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "docs" / "LAMBDA_CONFIGURATION_MANIFEST.json"
OUT_MD = REPO_ROOT / "docs" / "CONFIGURATION_CATALOG.md"
FUTURE_PIPELINE_DOC = REPO_ROOT / "docs" / "Future_Plans" / "FUTURE_PIPELINE.md"


def _read_manifest() -> Dict:
    if not MANIFEST.is_file():
        raise RuntimeError(f"Missing manifest: {MANIFEST}. Run tools.export_lambda_config_manifest first.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _key_group(key: str) -> str:
    # Group by first two SWAY segments to keep doc readable.
    parts = key.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return key


def _find_key_locations(key: str, max_hits: int = 3) -> List[str]:
    hits: List[str] = []
    for py in sorted(REPO_ROOT.rglob("*.py")):
        txt = py.read_text(encoding="utf-8", errors="ignore")
        if key in txt:
            rel = py.relative_to(REPO_ROOT).as_posix()
            hits.append(rel)
            if len(hits) >= max_hits:
                break
    return hits


def _future_rows() -> List[Dict]:
    rows = []
    mod_path = REPO_ROOT / "sway" / "future_modules_registry.py"
    if not mod_path.is_file():
        return rows
    txt = mod_path.read_text(encoding="utf-8", errors="ignore")
    # Lightweight parse for title/status lines in dict rows.
    # Kept resilient; this is for documentation only.
    chunks = txt.split("{")
    for c in chunks:
        if '"id":' not in c or '"status":' not in c:
            continue
        id_m = re.search(r'"id":\s*"([^"]+)"', c)
        title_m = re.search(r'"title":\s*"([^"]+)"', c)
        status_m = re.search(r'"status":\s*"([^"]+)"', c)
        part_m = re.search(r'"part":\s*"([^"]+)"', c)
        if not (id_m and title_m and status_m):
            continue
        rows.append(
            {
                "id": id_m.group(1),
                "title": title_m.group(1),
                "status": status_m.group(1),
                "part": part_m.group(1) if part_m else "?",
            }
        )
    return rows


def _future_doc_sway_keys() -> List[str]:
    if not FUTURE_PIPELINE_DOC.is_file():
        return []
    txt = FUTURE_PIPELINE_DOC.read_text(encoding="utf-8", errors="ignore")
    backticked = set(re.findall(r"`(SWAY_[A-Z0-9_]+)`", txt))
    keys = backticked or set(re.findall(r"SWAY_[A-Z0-9_]*[A-Z0-9]", txt))
    return sorted(k for k in keys if not k.endswith("_") and k.count("_") >= 2)


def _coded_repo_sway_keys() -> List[str]:
    keys = set()
    for py in REPO_ROOT.rglob("*.py"):
        txt = py.read_text(encoding="utf-8", errors="ignore")
        keys.update(re.findall(r"SWAY_[A-Z0-9_]+", txt))
    return sorted(keys)


def main() -> None:
    m = _read_manifest()
    keys = [r["key"] for r in m.get("sweep_env_keys", []) if isinstance(r, dict) and isinstance(r.get("key"), str)]
    keys = sorted(set(keys))
    group_map = defaultdict(list)
    for k in keys:
        group_map[_key_group(k)].append(k)

    adv_rows = m.get("advanced_modules", [])
    future = _future_rows()
    future_doc_keys = _future_doc_sway_keys()
    coded_repo_keys = _coded_repo_sway_keys()
    coded_key_set = set(coded_repo_keys)
    doc_only = [k for k in future_doc_keys if k not in coded_key_set]

    lines: List[str] = []
    lines.append("# Configuration & Technology Catalog")
    lines.append("")
    lines.append("Concise operational catalog for Lambda validation runs.")
    lines.append("")
    lines.append("## What This Covers")
    lines.append("")
    lines.append(f"- Machine-source key inventory: `{MANIFEST.relative_to(REPO_ROOT).as_posix()}`")
    lines.append(f"- SWAY keys in sweep/runtime inventory: **{len(keys)}**")
    lines.append(f"- Trial parameter count: **{m.get('search_space_summary', {}).get('trial_param_count', 'n/a')}**")
    lines.append("- Runtime feature truth logging in `main.py` with `[feature] requested/runtime/wiring`.")
    lines.append("")
    lines.append("## Technology Modules (Runtime Truth)")
    lines.append("")
    lines.append("| Module | Toggle | Runtime wired? | Status |")
    lines.append("|---|---|---:|---|")
    for row in adv_rows:
        lines.append(
            f"| {row.get('name','')} | `{row.get('toggle_env','')}` | "
            f"{'yes' if row.get('runtime_wired') else 'no'} | {row.get('status','')} |"
        )
    lines.append("")
    lines.append("## Future Pipeline Registry Status")
    lines.append("")
    if future:
        lines.append("| Part | Module | Status |")
        lines.append("|---|---|---|")
        for r in future:
            lines.append(f"| {r['part']} | {r['title']} (`{r['id']}`) | {r['status']} |")
    else:
        lines.append("- Future registry not found.")
    lines.append("")
    lines.append("## Future Pipeline Key Coverage")
    lines.append("")
    lines.append(f"- Keys mentioned in `FUTURE_PIPELINE.md`: **{len(future_doc_keys)}**")
    lines.append(f"- Keys currently coded in Python repo: **{len(coded_repo_keys)}**")
    lines.append(f"- Keys currently in sweep/runtime manifest: **{len(keys)}**")
    lines.append(f"- Future-doc keys not currently in coded manifest: **{len(doc_only)}**")
    if doc_only:
        lines.append("")
        lines.append("Doc-only keys (planned / not currently coded in sweep/runtime):")
        lines.append("")
        for k in doc_only:
            lines.append(f"- `{k}`")
    lines.append("")
    lines.append("## Configuration Keys By Group")
    lines.append("")
    lines.append("Each key lists up to 3 primary code locations where it is referenced.")
    lines.append("")
    for g in sorted(group_map):
        lines.append(f"### {g}")
        lines.append("")
        for k in group_map[g]:
            locs = _find_key_locations(k, max_hits=3)
            loc_text = ", ".join(f"`{p}`" for p in locs) if locs else "_no python refs found_"
            lines.append(f"- `{k}` — {loc_text}")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()

