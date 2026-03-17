# Benchmark: IMG_0256 Ground Truth

## Ground truth (from user)

| Check | Expected |
|-------|----------|
| Tracks at start | 8 (7 center + 1 back right) |
| Tracks at end | 9 (1 late entrant joins) |
| Total unique tracks | 9 |
| Late entrant | 1 (enters ~11 seconds) |

**Must prune:**
1. Person bottom-right at start — only head visible (audience)
2. Mirror reflections
3. Window reflections

## Files

- `IMG_0256_ground_truth.yaml` — Ground truth spec (edit to adjust expected values or regions)
- `../benchmark.py` — Verification script

## Usage

```bash
# Run pipeline and verify
python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml

# Verify existing output only (no pipeline run)
python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml --json output/data.json --no-run
```

Exit code: 0 = pass, 1 = fail

## Parameter tuning loop

To sweep parameters and re-run:

```bash
# Example: try different SYNC_SCORE_MIN (edit track_pruning.py or use env)
for val in 0.08 0.10 0.12; do
  # Set param (you’d need to expose it via env or CLI)
  python main.py /path/to/IMG_0256.mov --output-dir output
  python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml --json output/data.json --no-run
  [ $? -eq 0 ] && echo "PASS at val=$val" && break
done
```

Use `run_sweep.py` for automated tuning: `python run_sweep.py --config benchmarks/sweep_config.yaml [--exhaustive] [--adaptive]`. Log: `output/sweep_log.jsonl` with `effective_params`, `failure_reasons`, `suggested_next_params`.
