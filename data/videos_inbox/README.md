# Batch video inbox

Drop source videos here (`.mp4`, `.mov`, `.m4v`, `.webm`).

From `sway_pose_mvp/`:

```bash
python batch_run_for_review.py --input-dir data/videos_inbox --output-root output/flight_batch --pose-model base --skip-existing
```

If the machine sleeps or the process stops, run the same command again: clips that already have a finished `*_poses.mp4` are skipped; the rest continue from scratch for that file (no partial final MP4 is left on disk).

After the batch finishes, serve the review UI (recommended — opening the HTML file directly often shows a blank video in Chrome):

```bash
cd sway_pose_mvp
python review_app/serve_review.py output/flight_batch
```

Then open [http://127.0.0.1:8899/review/index.html](http://127.0.0.1:8899/review/index.html).

While using that server, the review page can **Remove from batch** a clip: it deletes the source file from this folder (per `batch_manifest.json`), removes that clip’s output subfolder, and refreshes the list. Export JSONL first if you care about saved labels for that clip.

Refresh the video list while the batch is still running (then reload the page):

```bash
cd sway_pose_mvp
python review_app/generate_review_index.py output/flight_batch
```

From repo root (`sway_test`):

```bash
python sway_pose_mvp/review_app/generate_review_index.py sway_pose_mvp/output/flight_batch
```

If the player is black but the timeline moves: OpenCV’s default MP4 often won’t decode in Chrome/Safari. Re-encode to H.264 (needs ffmpeg: `brew install ffmpeg`):

```bash
cd sway_pose_mvp
python review_app/reencode_poses_for_web.py output/flight_batch
```

Then hard-refresh the review page.

Human review workflow, offline checklist, and JSONL export: see `REVIEW_FEEDBACK.md` in `sway_pose_mvp/`.
