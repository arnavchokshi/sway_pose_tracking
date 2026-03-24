# CLI tools

Run from the `sway_pose_mvp/` directory:

```bash
python -m tools.prefetch_models
python -m tools.batch_run_for_review --input-dir input --output-root output/review_batch
python -m tools.benchmark --ground-truth benchmarks/IMG_0256_ground_truth.yaml
```

Other modules: `tools.benchmark_trackeval`, `tools.run_sweep`, `tools.run_baseline_test`, `tools.run_trackeval_boxmot_ablation`, `tools.export_models`.

The main pose pipeline stays at the repo root: `python main.py …`.
