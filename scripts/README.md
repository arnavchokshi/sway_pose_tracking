# Scripts layout

| Area | Path | Notes |
|------|------|--------|
| Public YOLO training | `phase2_public_training/` | DanceTrack / CrowdHuman, Lambda notes |
| Pose research | `phase5_pose_research/` | Experiments around temporal pose |
| One-off renders / compares | `render_hybrid_tracking_video.py`, `boxmot_vs_sam2_side_by_side.py`, `smoke_*.py`, `smoke_pipeline.sh` | Run from `sway_pose_mvp/`; may need env vars from main README |

**CLI tools** (run from repo root, e.g. `python -m tools.prefetch_models`): `tools.benchmark`, `tools.benchmark_trackeval`, `tools.run_sweep`, `tools.run_baseline_test`, `tools.run_trackeval_boxmot_ablation`, `tools.batch_run_for_review`, `tools.prefetch_models`, `tools.export_models`, `tools.pipeline_tree_queue` (multi-stage checkpoint **fan-out** on Pipeline Lab — see `pipeline_lab/README.md`), `tools.retry_failed_batch_runs` (re-queue failed runs for a batch without redoing successes). Primary pipeline entrypoint remains `main.py` at repo root.
