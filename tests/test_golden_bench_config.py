"""golden_bench --config YAML loading (no pytest subprocess)."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_golden_bench_example_yaml_loads():
    p = REPO_ROOT / "benchmarks" / "golden_bench.example.yaml"
    assert p.is_file()
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "tests" in cfg
    assert isinstance(cfg["tests"], list)
    assert len(cfg["tests"]) >= 1
    for rel in cfg["tests"]:
        assert (REPO_ROOT / rel).is_file(), rel
