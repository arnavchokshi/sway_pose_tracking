# Input videos (short clips / smoke tests)

Drop **small** videos you want to try here (`.mp4`, `.mov`, `.m4v`, `.webm`). Blobs are gitignored.

For **many** large flight clips, use **`data/videos_inbox/`** instead (see `data/videos_inbox/README.md`).

Then from `sway_pose_mvp/`:

```bash
python batch_run_for_review.py --input-dir input --output-root output/review_batch --skip-existing
```

Or process a single file:

```bash
python main.py input/your_video.mp4 --output-dir output/run_name
```
