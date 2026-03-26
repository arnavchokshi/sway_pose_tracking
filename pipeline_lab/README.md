# Pipeline Lab

Interactive pathway UI and FastAPI backend to run the Sway pose pipeline with **per-phase parameters**, **parallel recipe lanes**, **phase preview MP4s**, and **side-by-side compare**.

## Security

The API is intended for **local trust only**. It runs `main.py` as the same OS user, accepts **absolute file paths**, and has no authentication. Do not expose it on a public network.

## Setup

From `sway_pose_mvp/` (after the usual `pip install -r requirements.txt`):

```bash
pip install -r pipeline_lab/server/requirements.txt
cd pipeline_lab/web && npm install && npm run build && cd ../..
```

## Canonical URLs (local)

There are only two ports to remember:

| Role | URL | When |
|------|-----|------|
| **Backend (API)** | `http://localhost:8765` | Always — uvicorn listens here. |
| **Frontend (browser)** | `http://localhost:8765` | Built UI + API: set `PIPELINE_LAB_WEB_DIST` and open this tab only. |
| **Frontend (browser)** | `http://localhost:5173` | Dev + HMR: Vite proxies `/api` to 8765, so you still use **one** tab in the browser. |

Source of truth for these strings: `pipeline_lab/web/src/siteUrls.ts`.

## Run the API

**Single-tab “production” local (UI + API on the same port):**

```bash
cd sway_pose_mvp
export PIPELINE_LAB_WEB_DIST="$(pwd)/pipeline_lab/web/dist"
uvicorn pipeline_lab.server.app:app --host localhost --port 8765
```

Open **http://localhost:8765/** only.

**Dev with hot reload (two processes, one browser tab):**

```bash
# terminal 1 — API
uvicorn pipeline_lab.server.app:app --reload --host localhost --port 8765

# terminal 2 — Vite (fixed port 5173)
cd pipeline_lab/web && npm run dev
```

Open **http://localhost:5173/** only; the dev server proxies `/api` to port 8765.

## Environment

| Variable | Meaning |
|----------|---------|
| `PIPELINE_LAB_MAX_PARALLEL` | Worker threads pulling from the job queue (default `1`; raise only if you have enough GPU/RAM for concurrent runs). |
| `PIPELINE_LAB_PYTHON` | Python executable for `main.py` (default `python3`). |
| `PIPELINE_LAB_WEB_DIST` | Path to `web/dist` to serve the SPA from the API root. |
| `PIPELINE_LAB_CORS` | Comma-separated origins for CORS (default: `http://localhost:5173` and `http://localhost:8765` only). |

## CLI flags (orchestrator)

`main.py` additions used by the lab:

- `--save-phase-previews` — writes `output/phase_previews/*.mp4` at each major stage.
- `--progress-jsonl PATH` — appends one JSON object per stage (for UI polling).
- `--run-manifest PATH` — writes resolved config and output paths when the run finishes.

## Schema

Tunable parameters are listed in `sway/pipeline_config_schema.py` and exposed at `GET /api/schema`. Tracker overrides use environment variables (set per subprocess by the server from `binding: env` fields); YAML-backed keys go through `--params`.

## API: video input

- **`POST /api/runs/upload`** — multipart form: `file` (video), optional `recipe_name`, `fields_json` (stringified JSON object). Used by the web UI (drag-and-drop / file picker).
- **`POST /api/runs`** — JSON body `{ "video_path": "/abs/path", "recipe_name", "fields" }` for scripts or when the file already exists on the server.
- **`POST /api/runs/batch_path`** — JSON body `{ "video_path", "runs": [ { "recipe_name", "fields", "checkpoint"? } ], "source_label"?, "batch_id"? }`. Optional per-run **`checkpoint`** matches single-run `POST /api/runs` (e.g. `stop_after_boundary`, `resume_from`, `expect_boundary`, `force_checkpoint_load`). Optional top-level **`batch_id`** tags every run in the request so the Lab can load them together via **`/?batch=<id>`** (progress + logs).

## Checkpoint fan-out tree (CLI)

To run a **full tree** (every branch continues to the next stage — no manual “winner”) overnight on the Lab queue:

```bash
cd sway_pose_mvp
python -m tools.pipeline_tree_queue --tree pipeline_lab/tree_presets/bigtest_checkpoint_tree.yaml
```

Edit `video_path` in the YAML first. The tool prints a **batch id** and a Lab URL. Open the Lab home at `http://localhost:8765/` (or set `--ui-origin` if the UI runs on another origin, e.g. Vite). Optional deep link: `http://localhost:8765/?batch=…`.

Dry-run plan only: add `--dry-run`.

**Retry only failed jobs** (same `batch_id`, re-reads each run’s `request.json` + checkpoint; does not touch successful runs):

```bash
python3 -m tools.retry_failed_batch_runs --batch-id '<uuid-from-tree-output>'
# optional: remove old error rows from the list after capturing specs
python3 -m tools.retry_failed_batch_runs --batch-id '<uuid>' --delete-failed
# optional: block until the new jobs finish
python3 -m tools.retry_failed_batch_runs --batch-id '<uuid>' --wait
```

If you see **“No matching failed runs”** but the jobs really died: after a **uvicorn restart**, `/api/runs` often no longer reports `status=error` for those folders. Scan disk for the same batch (checkpoint present, **no** `run_manifest.json`) instead:

```bash
python3 -m tools.retry_failed_batch_runs --batch-id '<uuid>' --disk-incomplete --delete-failed
```

Debug: `--list-batch` prints every run id the API lists for that `batch_id`.

Run from `sway_pose_mvp/` on the **same machine as uvicorn** (needs `pipeline_lab/runs/` and `source_path` on disk). Restart uvicorn after a `main.py` fix, then retry.

## Runs layout

Each job is stored under `pipeline_lab/runs/<uuid>/`:

- `input_video*` — copy of the source file  
- `params.yaml` — merged YAML overrides  
- `progress.jsonl` — stage events  
- `run_manifest.json` — full manifest after success  
- `output/` — normal `main.py` outputs plus `phase_previews/`
