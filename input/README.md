# Input videos (offline batch)

Drop **all** videos you want to process here (`.mp4`, `.mov`, `.m4v`, `.webm`). They are gitignored so large files stay local.

Then from `sway_pose_mvp/`:

```bash
python batch_run_for_review.py --input-dir input --output-root output/review_batch --skip-existing
```

Or process a single file:

```bash
python main.py input/your_video.mp4 --output-dir output/run_name
```
