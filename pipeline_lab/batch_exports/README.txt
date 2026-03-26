Pipeline Lab batch snapshots (metadata only).

- lab_runs_*_.yaml — run_id, status, recipe_name, video_stem, batch_id, errors (from GET /api/runs).
- tree_queue_*_.txt — copy of pipeline_lab/tree_presets/last_tree_queue.log at export time.

Re-export when a tree finishes:

  cd sway_pose_mvp
  python3 -m tools.export_lab_batch_snapshot --batch-id '<uuid>' --label complete

Run outputs and checkpoints remain under pipeline_lab/runs/<run_id>/ (gitignored).
