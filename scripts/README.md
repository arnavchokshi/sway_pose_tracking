# Scripts layout

| Area | Path | Notes |
|------|------|--------|
| Public YOLO training | `phase2_public_training/` | DanceTrack / CrowdHuman, Lambda notes |
| Pose research | `phase5_pose_research/` | Experiments around temporal pose |
| One-off renders / compares | `render_hybrid_tracking_video.py`, `boxmot_vs_sam2_side_by_side.py`, `smoke_*.py`, `smoke_pipeline.sh` | Run from `sway_pose_mvp/`; may need env vars from main README |

Top-level **entrypoints** (same directory as `main.py`): `benchmark.py`, `benchmark_trackeval.py`, `run_sweep.py`, `run_baseline_test.py`, `run_trackeval_boxmot_ablation.py`, `batch_run_for_review.py`, `prefetch_models.py`.
