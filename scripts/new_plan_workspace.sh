#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create an isolated plan workspace with shared models (no venv required).

Usage:
  bash scripts/new_plan_workspace.sh --plan PLAN_22
  bash scripts/new_plan_workspace.sh --plan "sam2 rewrite" --base main
  bash scripts/new_plan_workspace.sh --plan PLAN_22 --branch plan/22 --workspace /abs/path/to/ws
  bash scripts/new_plan_workspace.sh --plan PLAN_22 --with-shared-venv

Options:
  --plan <name>         Required plan name/id (used for branch + folder names).
  --base <ref>          Base ref/branch for new plan branch (default: main).
  --branch <name>       Branch name (default: plan/<slugified-plan>).
  --workspace <path>    Worktree path (default: <repo>/plan_workspaces/<slug>).
  --with-shared-venv    Optional: create/reuse shared venv and link worktree .venv.
  --skip-install        With --with-shared-venv, skip pip install for shared venv.
  -h, --help            Show help.

What it does:
  1) Creates an isolated git worktree + branch for the plan.
  2) Symlinks worktree models -> <repo>/models (reuses downloaded weights).
  3) Writes worktree .plan_env with explicit shared model/cache paths.
  4) Optional: creates shared venv and links worktree .venv when requested.
EOF
}

slugify() {
  local raw="$1"
  printf '%s' "$raw" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

assert_safe_to_replace_dir_with_symlink() {
  local dir_path="$1"
  local expected_a="$2"
  local expected_b="$3"

  if [[ ! -d "$dir_path" ]]; then
    return 0
  fi

  local count
  count="$(ls -A "$dir_path" | wc -l | tr -d ' ')"
  if [[ "$count" == "0" ]]; then
    return 0
  fi

  if [[ -f "$dir_path/$expected_a" && -f "$dir_path/$expected_b" && "$count" == "2" ]]; then
    return 0
  fi

  echo "Refusing to replace non-empty directory: $dir_path" >&2
  echo "Directory must be empty or contain only $expected_a and $expected_b." >&2
  exit 1
}

PLAN_NAME=""
BASE_REF="main"
BRANCH_NAME=""
WORKSPACE_PATH=""
SKIP_INSTALL=0
WITH_SHARED_VENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      PLAN_NAME="${2:-}"
      shift 2
      ;;
    --base)
      BASE_REF="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH_NAME="${2:-}"
      shift 2
      ;;
    --workspace)
      WORKSPACE_PATH="${2:-}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --with-shared-venv)
      WITH_SHARED_VENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PLAN_NAME" ]]; then
  echo "--plan is required." >&2
  usage
  exit 1
fi

require_cmd git
require_cmd sed
require_cmd tr
require_cmd ls
require_cmd ln

REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_COMMON_DIR="$(git rev-parse --git-common-dir)"
COMMON_ROOT="$(cd "${GIT_COMMON_DIR}/.." && pwd)"
PLAN_SLUG="$(slugify "$PLAN_NAME")"

if [[ -z "$PLAN_SLUG" ]]; then
  echo "Plan slug is empty after normalization: $PLAN_NAME" >&2
  exit 1
fi

if [[ -z "$BRANCH_NAME" ]]; then
  BRANCH_NAME="plan/${PLAN_SLUG}"
fi

if [[ -z "$WORKSPACE_PATH" ]]; then
  WORKSPACE_PATH="${COMMON_ROOT}/plan_workspaces/${PLAN_SLUG}"
fi

SHARED_VENV="${COMMON_ROOT}/.venv_shared"
SHARED_MODELS="${COMMON_ROOT}/models"
SHARED_HF="${COMMON_ROOT}/.cache/huggingface"
SHARED_TORCH="${COMMON_ROOT}/.cache/torch"

mkdir -p "${COMMON_ROOT}/plan_workspaces"
mkdir -p "$SHARED_HF" "$SHARED_TORCH"

if [[ ! -d "$SHARED_MODELS" ]]; then
  echo "Shared models directory not found: $SHARED_MODELS" >&2
  echo "Create it first (or run from the main sway_pose_mvp repo)." >&2
  exit 1
fi

if [[ -e "$WORKSPACE_PATH" ]]; then
  echo "Workspace path already exists: $WORKSPACE_PATH" >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
  echo "Branch already exists: ${BRANCH_NAME}" >&2
  echo "Choose --branch or delete the old branch/worktree first." >&2
  exit 1
fi

if [[ "$WITH_SHARED_VENV" -eq 1 ]]; then
  require_cmd python3
  echo "Creating shared venv: $SHARED_VENV"
  if [[ ! -d "$SHARED_VENV" ]]; then
    python3 -m venv "$SHARED_VENV"
    if [[ "$SKIP_INSTALL" -eq 0 ]]; then
      "$SHARED_VENV/bin/pip" install --upgrade pip
      "$SHARED_VENV/bin/pip" install -r "${COMMON_ROOT}/requirements.txt"
    fi
  fi
fi

echo "Creating worktree:"
echo "  branch:    $BRANCH_NAME"
echo "  base:      $BASE_REF"
echo "  workspace: $WORKSPACE_PATH"
git -C "$COMMON_ROOT" worktree add -b "$BRANCH_NAME" "$WORKSPACE_PATH" "$BASE_REF"

assert_safe_to_replace_dir_with_symlink "${WORKSPACE_PATH}/models" "README.md" ".gitkeep"
if [[ -L "${WORKSPACE_PATH}/models" ]]; then
  rm -f "${WORKSPACE_PATH}/models"
elif [[ -d "${WORKSPACE_PATH}/models" ]]; then
  rm -rf "${WORKSPACE_PATH}/models"
fi
ln -s "$SHARED_MODELS" "${WORKSPACE_PATH}/models"

if [[ "$WITH_SHARED_VENV" -eq 1 ]]; then
  if [[ -L "${WORKSPACE_PATH}/.venv" ]]; then
    rm -f "${WORKSPACE_PATH}/.venv"
  elif [[ -d "${WORKSPACE_PATH}/.venv" ]]; then
    rm -rf "${WORKSPACE_PATH}/.venv"
  fi
  ln -s "$SHARED_VENV" "${WORKSPACE_PATH}/.venv"
fi

cat > "${WORKSPACE_PATH}/.plan_env" <<EOF
export SWAY_SHARED_MODELS_DIR="${SHARED_MODELS}"
export HF_HOME="${SHARED_HF}"
export HUGGINGFACE_HUB_CACHE="${SHARED_HF}/hub"
export TRANSFORMERS_CACHE="${SHARED_HF}/hub"
export TORCH_HOME="${SHARED_TORCH}"
EOF

if [[ -f "${SHARED_MODELS}/yolo26l.pt" ]]; then
  echo "export SWAY_YOLO_WEIGHTS=\"${SHARED_MODELS}/yolo26l.pt\"" >> "${WORKSPACE_PATH}/.plan_env"
fi
if [[ -f "${SHARED_MODELS}/AFLink_epoch20.pth" ]]; then
  echo "export SWAY_AFLINK_WEIGHTS=\"${SHARED_MODELS}/AFLink_epoch20.pth\"" >> "${WORKSPACE_PATH}/.plan_env"
fi
if [[ -f "${SHARED_MODELS}/gnn_track_refine.pt" ]]; then
  echo "export SWAY_GNN_WEIGHTS=\"${SHARED_MODELS}/gnn_track_refine.pt\"" >> "${WORKSPACE_PATH}/.plan_env"
fi

echo
echo "Plan workspace ready."
echo "Next:"
echo "  cd \"$WORKSPACE_PATH\""
echo "  source .plan_env"
if [[ "$WITH_SHARED_VENV" -eq 1 ]]; then
  echo "  source .venv/bin/activate"
fi
echo "  python -m tools.prefetch_models   # optional, only if adding new weights"
echo
echo "Tip: keep your human-readable plan in docs/Future_Plans/plans/ and build code independently in this worktree."
