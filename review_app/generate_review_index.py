#!/usr/bin/env python3
"""
Build output/<batch>/review/index.html with an embedded manifest so it works
offline via file:// (no fetch() to local JSON).

Usage:
  python generate_review_index.py /path/to/output_batch
"""

from __future__ import annotations

import json
from html import escape as html_escape
import re
import sys
from pathlib import Path

_MANIFEST_SCRIPT_RE = re.compile(
    r'(<script\s+type="application/json"\s+id="sway-manifest-inline"\s*>)'
    r'(.*?)'
    r'(</script>)',
    re.DOTALL | re.IGNORECASE,
)


def _build_samples(output_root: Path) -> list:
    manifest = _load_batch_manifest(output_root) or {}
    samples_meta = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(samples_meta, dict):
        samples_meta = {}
    samples = []
    for d in sorted(output_root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "review":
            continue
        poses = sorted(d.glob("*_poses.mp4"))
        if not poses:
            continue
        pv = poses[0]
        dj = d / "data.json"
        pl = d / "prune_log.json"
        meta = samples_meta.get(d.name)
        inp = None
        if isinstance(meta, dict):
            ip = meta.get("input_path")
            if isinstance(ip, str) and ip.strip():
                inp = ip.strip()
        rec = {
            "id": d.name,
            "label": d.name,
            "rendered_video": f"../{d.name}/{pv.name}",
            "data_json": f"../{d.name}/data.json" if dj.exists() else None,
            "prune_log": f"../{d.name}/prune_log.json" if pl.exists() else None,
        }
        if inp:
            rec["input_path"] = inp
        samples.append(rec)
    return samples


def _load_batch_manifest(output_root: Path) -> dict | None:
    p = output_root / "batch_manifest.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: generate_review_index.py <output_root>", file=sys.stderr)
        sys.exit(1)
    output_root = Path(sys.argv[1]).expanduser().resolve()
    if not output_root.is_dir():
        print(f"Not a directory: {output_root}", file=sys.stderr)
        sys.exit(1)

    samples = _build_samples(output_root)
    payload = {
        "schema_version": "1.0",
        "output_root_name": output_root.name,
        "output_root": str(output_root),
        "batch_manifest": _load_batch_manifest(output_root),
        "samples": samples,
    }

    template_path = Path(__file__).resolve().parent / "template.html"
    if not template_path.is_file():
        print(f"Missing template: {template_path}", file=sys.stderr)
        sys.exit(1)
    template = template_path.read_text(encoding="utf-8")
    raw = json.dumps(payload, ensure_ascii=False)
    safe = raw.replace("</", "<\\/")
    if _MANIFEST_SCRIPT_RE.search(template):
        html = _MANIFEST_SCRIPT_RE.sub(r"\1" + safe + r"\3", template, count=1)
    else:
        token = "__SWAY_MANIFEST_JSON__"
        if token not in template:
            print(
                "template.html must contain sway-manifest-inline script or "
                f"{token} placeholder",
                file=sys.stderr,
            )
            sys.exit(1)
        html = template.replace(token, safe)

    out_dir = output_root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sidebar + header: visible even if JS fails; file:// often blocks video — user should use serve_review.py
    n = len(samples)
    hdr = f"{html_escape(output_root.name)} · {n} video{'s' if n != 1 else ''}"
    html = html.replace(
        '<span class="sub" id="hdr-meta"></span>',
        f'<span class="sub" id="hdr-meta">{hdr}</span>',
        1,
    )
    html = html.replace(
        '<span class="sub" id="progress-label">0 / 0</span>',
        f'<span class="sub" id="progress-label">0 / {n}</span>',
        1,
    )
    if n:
        static_lis = "".join(
            f'<li style="padding:0.4rem 0.55rem;font-size:0.82rem;border-bottom:1px solid var(--border);color:var(--muted);">'
            f"{html_escape(s['id'])}</li>"
            for s in samples
        )
    else:
        static_lis = (
            '<li style="padding:0.5rem;color:var(--muted);font-size:0.82rem;">'
            "No finished runs yet (no *_poses.mp4 in subfolders).</li>"
        )
    html = html.replace(
        '<ul id="sample-list"></ul>',
        f'<ul id="sample-list">{static_lis}</ul>',
        1,
    )

    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(samples)} sample(s))")
    print(
        "Tip: if the page looks blank, open via local server (not file://):\n"
        f"  python {Path(__file__).resolve().parent / 'serve_review.py'} {output_root}"
    )


if __name__ == "__main__":
    main()
