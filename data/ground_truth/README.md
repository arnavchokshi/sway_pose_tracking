# Phase 1–3 sweep benchmark data (local only)

Large **videos** and your **MOT** labels live here so paths stay **relative to the repo** (Lambda, CI, teammates clone + `rsync` this folder once).

## Layout

```text
data/ground_truth/
  README.md
  sweep_sequences.example.yaml
  sweep_sequences.yaml          # copy from example; gitignored
  bigtest/BigTest.mov
  bigtest/gt/gt.txt
  mirrortest/IMG_2946.MP4
  mirrortest/gt/gt.txt
  gymtest/IMG_8309.mov
  gymtest/gt/gt.txt
```

## Setup

1. Copy `sweep_sequences.example.yaml` → `sweep_sequences.yaml`.
2. Put videos and `gt/gt.txt` in place (paths must match YAML).
3. Run: `python -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml`

Sequence ids: **bigtest**, **mirrortest**, **gymtest** (lowercase, no spaces).

See `docs/GT_DRIVEN_SWEEP_AND_TUNING_PLAYBOOK.md`.
