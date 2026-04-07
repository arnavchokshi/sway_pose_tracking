# Separate Plan Workspaces

This setup gives each new plan a clean code workspace while reusing already installed models.

## Why this helps

- Each plan gets its own branch and folder (fully isolated from other plans).
- Code can be started "from scratch" per plan without mixing partial experiments.
- Existing model weights in `models/` are reused (no redownload/reinstall).
- Optional shared venv is available, but not required.

## One-command workflow

From the main `sway_pose_mvp/` repo:

```bash
bash scripts/new_plan_workspace.sh --plan PLAN_22
```

This creates:

- Branch: `plan/plan-22`
- Workspace: `<repo>/plan_workspaces/plan-22`
- Shared models: `<repo>/models` (reused by symlink)
- Workspace env file: `.plan_env` (shared model/cache path exports)
- Optional: shared venv if created with `--with-shared-venv`

## Daily use inside a plan workspace

```bash
cd plan_workspaces/plan-22
source .plan_env
```

Then run your normal commands (`python main.py ...`, `python -m tools...`, tests, etc.).

If you want a shared venv linked into each plan workspace:

```bash
bash scripts/new_plan_workspace.sh --plan PLAN_23 --with-shared-venv
```

## Add your human-readable plan

Keep plan docs in:

`docs/Future_Plans/plans/`

Example:

`docs/Future_Plans/plans/PLAN_22_my_new_idea.md`

An agent can use that doc to generate code in the plan workspace branch, completely separate from other plan branches.

## Important notes

- `models/` in each plan workspace is a symlink to the shared `models/` in the main repo.
- `.plan_env` includes explicit shared paths such as `SWAY_SHARED_MODELS_DIR`, and when present:
  - `SWAY_YOLO_WEIGHTS`
  - `SWAY_AFLINK_WEIGHTS`
  - `SWAY_GNN_WEIGHTS`
- If a plan requires new weights, download once and all plan workspaces can use them.
- If you need strict offline runs, keep using `SWAY_OFFLINE=1` as documented in `README.md`.
