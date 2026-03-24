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

## Run the API

```bash
cd sway_pose_mvp
export PIPELINE_LAB_WEB_DIST="$(pwd)/pipeline_lab/web/dist"
uvicorn pipeline_lab.server.app:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765/ for the built UI, or for dev with hot reload:

```bash
# terminal 1
uvicorn pipeline_lab.server.app:app --reload --host 127.0.0.1 --port 8765

# terminal 2
cd pipeline_lab/web && npm run dev
```

The Vite dev server proxies `/api` to port 8765.

## Environment

| Variable | Meaning |
|----------|---------|
| `PIPELINE_LAB_MAX_PARALLEL` | Worker threads pulling from the job queue (default `1`; raise only if you have enough GPU/RAM for concurrent runs). |
| `PIPELINE_LAB_PYTHON` | Python executable for `main.py` (default `python3`). |
| `PIPELINE_LAB_WEB_DIST` | Path to `web/dist` to serve the SPA from the API root. |
| `PIPELINE_LAB_CORS` | Comma-separated origins for CORS (default includes Vite dev URL). |

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

## Runs layout

Each job is stored under `pipeline_lab/runs/<uuid>/`:

- `input_video*` — copy of the source file  
- `params.yaml` — merged YAML overrides  
- `progress.jsonl` — stage events  
- `run_manifest.json` — full manifest after success  
- `output/` — normal `main.py` outputs plus `phase_previews/`
