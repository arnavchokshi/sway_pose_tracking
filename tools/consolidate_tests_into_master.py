#!/usr/bin/env python3
"""
Merge ``tests/test_*.py`` into ``tests/test_MASTER_suite.py`` (run from repo root).

Use when you have checked out or recreated per-module test files and want to rebuild
the single master module. The master file itself is excluded from the merge.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
OUT = TESTS / "test_MASTER_suite.py"

HEADER = '''#!/usr/bin/env python3
"""
MASTER consolidated test suite — all tests from the former `tests/test_*.py` modules.

Run the full suite::

    pytest tests/test_MASTER_suite.py -v
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

'''


def _clean_chunk(name: str, text: str) -> str:
    text = re.sub(r"^#!/usr/bin/env python3\n", "", text)
    # Only one __future__ block allowed per module (already in HEADER).
    text = re.sub(r"^from __future__ import annotations\n", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"\n\nif __name__ == [\"']__main__[\"']:\n(?:    .+\n)+",
        "\n",
        text,
    )
    # Drop local repo-root bootstrapping; suite header defines REPO_ROOT / ROOT / sys.path.
    text = re.sub(
        r"^REPO_ROOT = Path\(__file__\)\.resolve\(\)\.parent\.parent\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^ROOT = Path\(__file__\)\.resolve\(\)\.parent\.parent\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^if str\(REPO_ROOT\) not in sys\.path:\n    sys\.path\.insert\(0, str\(REPO_ROOT\)\)\n+",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^if str\(ROOT\) not in sys\.path:\n    sys\.path\.insert\(0, str\(ROOT\)\)\n+",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^sys\.path\.insert\(0, str\(ROOT\)\)\n+",
        "",
        text,
        flags=re.MULTILINE,
    )
    return f"\n\n# ---------- SOURCE: {name} ----------\n\n{text.strip()}\n"


def main() -> int:
    sources = sorted(TESTS.glob("test_*.py"))
    sources = [p for p in sources if p.name != "test_MASTER_suite.py"]
    if not sources:
        print("No test_*.py sources found.", file=sys.stderr)
        return 1
    parts = [HEADER]
    for p in sources:
        parts.append(_clean_chunk(p.name, p.read_text(encoding="utf-8")))
    OUT.write_text("".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(sources)} modules merged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
