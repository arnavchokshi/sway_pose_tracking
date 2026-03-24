# Local data (not committed)

| Path | Purpose |
|------|--------|
| **`videos_inbox/`** | Large batch queue: drop `.mov` / `.mp4` here, then run `python -m tools.batch_run_for_review` with `--input-dir data/videos_inbox`. See `videos_inbox/README.md`. |

Training caches and downloads stay alongside the repo as today (`datasets/`, `*_hf/`, Hugging Face cache) — see the root `.gitignore`.
