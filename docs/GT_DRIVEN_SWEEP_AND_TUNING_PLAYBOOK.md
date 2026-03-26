# Ground-truth–driven Phase 1–3 sweep playbook

*Detection, tracking, stitching — CVAT/MOT validation, Optuna TPE, and reusing winners*

**Single canonical doc** for the same process: bounded search on **Phases 1–3**, scoring vs. labels, logging, promotion, and follow-on tuning.

**Status:** Implementation-ready for Optuna-driven studies; `tools/run_sweep.py` defaults to **`--stop-after-boundary after_phase_3`**.  
**Scope:** Phases 1–3 only until winners are locked; Phases 4–11 are a **second** sweep with different metrics / stop boundaries.  
**Ground truth:** Typically **three** CVAT-annotated videos — boxes + **consistent track IDs** — exported in **MOT** format.

**Also see:** [`MASTER_PIPELINE_GUIDELINE.md`](./MASTER_PIPELINE_GUIDELINE.md), [`PIPELINE_FINDINGS_AND_BEST_CONFIGS.md`](./PIPELINE_FINDINGS_AND_BEST_CONFIGS.md).

**Repo alignment:** TrackEval: `sway/trackeval_runner.py`, `tools/benchmark_trackeval.py`. MOT preds from `data.json`: `sway/mot_format.py` (`data_json_to_mot_lines`). File export: `python -m tools.convert_data_json_to_mot`.

---

## 1. Scope and hard requirements

### 1.1 What this playbook optimizes

**Phases 1–3** — detection, streaming tracking, post-track stitching (**MASTER §5**). You tune **who is in the frame and which track ID they carry**, not pose, pruning, or export.

- **Default:** every worker uses **`--stop-after-boundary after_phase_3`**.
- **`tools/run_sweep.py`** passes that flag **by default**. For **legacy** full-pipeline sweeps (e.g. `benchmarks/sweep_config.yaml` Re-ID / Phase 6+ YAML keys):  
  **`python -m tools.run_sweep --no-stop-after-boundary`**

Stopping at phase 3 cuts roughly **70–80%** of wall time vs. full runs on typical hardware (pose is the heavy step).

### 1.2 Fix — `run_sweep` / subprocess shape

```python
cmd = [
    sys.executable, "main.py", video_path,
    "--output-dir", run_dir,
    "--params", params_path,
    "--stop-after-boundary", "after_phase_3",
]
```

### 1.3 CVAT → MOT (no custom conversion for export)

CVAT: **Actions → Export task dataset → MOT 1.1** → `.zip` with `gt/gt.txt`:

```text
frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility
1,1,1363,569,103,241,1,1,0.86014
```

**Multi-sequence** TrackEval-style layout (optional):

```text
benchmarks/swaydance/
  seqmaps/swaydance-all.txt    # name \n video_a \n video_b \n video_c
  video_a/gt/gt.txt
  video_a/seqinfo.ini
  video_b/...
```

`seqinfo.ini` per sequence (adjust `seqLen`, fps, size):

```ini
[Sequence]
name=video_a
imDir=img1
frameRate=30
seqLen=1800
imWidth=1920
imHeight=1080
imExt=.jpg
```

**Single-sequence** runs: flat `gt.txt` + `trackeval:` in YAML (below) is enough.

This repo’s GT line shape for tooling is also documented in `sway/mot_format.py` (1-based frame, x,y,w,h, etc.).

### 1.4 GNN guard

If **`sway_gnn_track_refine`** is on and **`models/gnn_track_refine.pt`** (or `SWAY_GNN_WEIGHTS`) is missing, the run uses **random** GNN weights → **invalid** sweep data. **Exclude** GNN from the job space or **gate** at generation time (**MASTER §7.2**).

---

## 2. Goals

1. **Search:** Discretized grid or **Optuna TPE** over Phase 1–3 knobs (detector, tracker, hybrid SAM, stitch / phase-13 modes, optional GNN only with weights).
2. **Score:** **TrackEval** on MOT (and optionally YAML expectation checks via `tools/benchmark.py` / `run_sweep`).
3. **Persist:** JSONL / SQLite (Optuna) + `git_sha`, params, paths, metrics.
4. **Promote:** Presets, matrix rows, `PIPELINE_FINDINGS_AND_BEST_CONFIGS.md`, optional `sweep_winners.yaml`.

### 2.1 All benchmark videos every trial (scores are never “one clip only”)

Your **three** labeled videos are **different regimes** (e.g. crowd size, mirrors, gym floor). Automation must assume **every trial is incomplete** until it has been scored on **all** of them.

- **Per trial:** run the **same** params on **each** video → one composite score **per** video → one **global** objective for Optuna (recommended: **harmonic mean** of those composites so a killer score on clip A cannot hide a failure on B or C).
- **What Optuna optimizes** is that **global** number, not any single sequence. **Following runs** (continued studies, new `study_name`, or narrowed search) should keep the **same multi-video aggregate** unless you deliberately run an ablation on one clip.
- **Promotion guardrails** (e.g. reject configs with any per-video composite below a floor) reinforce “good on **all**,” not just on average.

---

## 3. Validation wiring

### 3.1 Per-sequence YAML + TrackEval

```yaml
trackeval:
  gt_mot_file: benchmarks/my_seq_gt.txt
  sequence_name: my_seq
  im_width: 1920
  im_height: 1080
```

```bash
python -m tools.benchmark_trackeval --ground-truth benchmarks/my_seq_ground_truth.yaml --json output/run_id/data.json
```

`pip install trackeval`. Match **frame index and resolution** to the exact file passed to `main.py`.

### 3.2 YAML expectations (smoke / regression)

Scalar checks (counts, late entrants) as in `benchmarks/IMG_0256_ground_truth.yaml`. `run_sweep` can log both TrackEval and benchmark pass/fail in `sweep_log.jsonl`.

---

## 4. “Reasonable exhaustive” search discipline

| Idea | Practice |
|------|-----------|
| **Discrete grid** | Step continuous knobs (e.g. `sway_yolo_conf`: 0.22, 0.26, 0.30). |
| **Stop boundary** | **`after_phase_3`** for MOT/CVAT sweeps unless you explicitly run a longer experiment. |
| **Curated axes** | Start from `sway/pipeline_matrix_presets.py` / Lab matrix (**MASTER §3.4**), then multi-knob combos (see **PIPELINE FINDINGS**). |
| **Master locks** | Default production locks in `pipeline_config_schema.py`. Only use **`SWAY_UNLOCK_*`** in a **labeled** sensitivity study, not the primary automated study. |
| **Incompatible pairs** | e.g. bidirectional + resume from `after_phase_1` (**MASTER §4.4**) — encode exclusions in the job generator. |

Parallel GPUs shrink wall-clock; the job list must still be **finite**.

---

## 5. Orchestration

### 5.1 One run = one output directory

```text
output/sweeps/<sweep_id>/<run_name>/
  data.json
```

```bash
python main.py /path/to/video.mp4 \
  --output-dir output/sweeps/20260326_grid_a/run_042 \
  --params /path/to/run_042_params.yaml \
  --stop-after-boundary after_phase_3
```

Use **`--phase-debug-jsonl`** for per-phase timings/counters (**MASTER §4.5**).

**GPU / OOM:** After **every** job (before the next), call **`gc.collect()`**; if CUDA is used, **`torch.cuda.synchronize()`** and **`torch.cuda.empty_cache()`**. Subprocess-only `main.py` usually frees child VRAM on exit; the parent still needs this if it imports PyTorch or uses persistent workers.

**Server perf env (Lambda / multi-vCPU NVIDIA):** export **`SWAY_SERVER_PERF=1`** so `main.py` enables cuDNN autotune, TF32, and bounded CPU thread pools (`sway/server_runtime_perf.py`). **`tools/auto_sweep`** and **`tools/run_sweep`** merge the same overlay into each child process **only if** the parent shell already has `SWAY_SERVER_PERF=1`.

Before an overnight sweep, run a quick smoke (subprocess env + optional ≤60s `main.py` on a synthetic clip):

```bash
export SWAY_SERVER_PERF=1
python -m tools.smoke_server_perf_env
python -m tools.smoke_server_perf_env --pipeline --timeout 60
```

**One-shot on the box:** `bash scripts/lambda_preflight.sh` (installs requirements, checks models + `sweep_sequences.yaml` paths, runs the env smoke). Add **`PIPELINE=1`** for `main.py` smoke; override timeout with **`PIPELINE_TIMEOUT=90`**.

### 5.2 Batch sources

| Mechanism | Role |
|-----------|------|
| **`tools/run_sweep.py`** | YAML `param_sets`, adaptive hints, `sweep_log.jsonl`. |
| **Lab `POST /api/runs/batch_path`** | Many runs, server path (**MASTER §3.4**). |
| **`tools.pipeline_matrix_runs`** | CLI matrix vs. Lab API. |
| **Checkpoint trees** | `pipeline_lab/tree_presets/` — watch bidirectional + resume rules (**MASTER §4.4**). |
| **Optuna driver** | `tools/auto_sweep.py` (implement per §7–9). |

### 5.3 After each `data.json`

1. TrackEval: `trackeval_from_ground_truth_yaml` or `benchmark_trackeval`.  
2. Optional: `tools/benchmark.py` checks.  
3. **Multi-video:** for each trial, require scores from **every** benchmark video, then aggregate with **harmonic mean** of per-video composite scores (so one strong clip cannot mask a weak one; arithmetic mean alone is too forgiving).

---

## 6. Metrics, ranking, and metric → parameter map

### 6.1 Reading TrackEval (calibrate keys once)

| Metric | Meaning | Heuristic “good” |
|--------|---------|-------------------|
| **HOTA** | Detection + association | > ~0.65 dance (context-dependent) |
| **DetA** | Detection branch | YOLO conf / NMS |
| **AssA** | Association branch | Tracker + stitch |
| **IDSW** | ID switch count | Lower; ~&lt; 5% of GT dets (rule of thumb) |
| **Frag** | Fragmentation | Lower |
| **CLR_Re** | Recall | Missing people if low |
| **CLR_Pr** | Precision | Phantoms if low |

Flattened names in code often look like `HOTA_*`, `CLEAR_*`, `Identity_*` — inspect one `benchmark_trackeval` JSON dump before coding `rank_score`.

### 6.2 Symptom → knob (automation / critique)

- **Low DetA + low CLR_Re:** YOLO missing people → **lower** `sway_yolo_conf`, check **`sway_yolo_weights`**; do **not** tune tracker first.  
- **Low DetA + low CLR_Pr:** clutter → **raise** `sway_yolo_conf`, **raise** `sway_pretrack_nms_iou` (~0.55–0.60).  
- **Good DetA, poor AssA:** IDs wrong → **`sway_boxmot_max_age`**, **`sway_boxmot_match_thresh`**, **`sway_stitch_max_frame_gap`**, AFLink mode.  
- **High IDSW, good DetA:** crossovers → **lower** `sway_boxmot_match_thresh`, **raise** `sway_boxmot_max_age`.  
- **High Frag, good DetA:** broken tracks → **raise** `sway_stitch_max_frame_gap`; **neural_if_available** only if **`models/AFLink_epoch20.pth`** (or `SWAY_AFLINK_WEIGHTS`) exists.  
- **HOTA swings across videos:** overfit → harmonic mean aggregate; weight worst clip.

### 6.3 Composite score (template)

Do **not** sort by one metric. Example (fix keys after calibration):

```python
def rank_score(metrics):
    hota   = metrics["HOTA"]   # e.g. HOTA_HOTA from flattening
    idsw   = metrics["IDSW"]
    frag   = metrics["Frag"]
    recall = metrics["CLR_Re"]
    total_gt_dets = metrics["CLR_TP"] + metrics["CLR_FN"]
    idsw_penalty = min(idsw / max(total_gt_dets * 0.05, 1), 1.0)
    frag_penalty = min(frag / max(total_gt_dets * 0.10, 1), 1.0)
    return (
        0.45 * hota
        + 0.25 * recall
        + 0.20 * (1.0 - idsw_penalty)
        + 0.10 * (1.0 - frag_penalty)
    )
```

```python
import statistics
aggregate = statistics.harmonic_mean([score_a, score_b, score_c])
```

### 6.4 Human review (optional for the loop; recommended before shipping)

The **closed loop is fully automatic:** trials run, metrics score, Optuna (or your ranker) picks a numeric winner. **You do not need a human in that loop** for it to work.

**Spot-checking** top‑k in Lab and writing up **PIPELINE_FINDINGS** is **optional** during long compute — it is reasonable to let automation run for **many hours** first and only look at previews **after** you have a stable leaderboard. Use that pass when you are ready to **trust a config for production**: metrics can disagree with mirrors, crowd edges, or “looks wrong” cases.

**Auto-promotion** (e.g. `sweep_winners.yaml` from fixed rules) can run without human review; treat that file as **machine-nominated** until someone signs off.

---

## 7. Optuna TPE (primary search strategy)

~**10 knobs** × 3 values → **59k** full grid — infeasible. **Random** search does not learn. Use **[Optuna](https://optuna.org/)** + **TPE**: `pip install optuna` (optional `optuna-dashboard`).

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    study_name="swaydance_phase13_v1",
    storage="sqlite:///sweep.db",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
)
study.optimize(objective, n_trials=40, timeout=None)
```

**Objective sketch:** `trial.suggest_categorical("sway_phase13_mode", [...])` first, then **branch** (see **§9**): only suggest SAM/registry knobs that apply to that mode. Always run `main.py` via **CLI** + `--params` YAML (not Lab enqueue) when **`sway_handshake`** trials must vary **`SWAY_HYBRID_SAM_IOU_TRIGGER`** — the Lab API forces **0.10** for handshake (`pipeline_lab/server/app.py`); subprocess **`main.py`** respects **`SWAY_*`** from YAML after params apply. Enforce: **neural AFLink only if weights exist**; **GNN off or gated**; **`SWAY_YOLO_DETECTION_STRIDE=1`** fixed (do not ask Optuna). For each video: TrackEval → `trial.report(score, step=)` → harmonic mean aggregate. Set `trial.set_user_attr` for per-video breakdown.

**Tracker:** every trial must set **Deep OC-SORT without track-time OSNet** (same as Lab `deep_ocsort`): e.g. `SWAY_USE_BOXMOT=1`, `SWAY_BOXMOT_TRACKER=deepocsort`, `SWAY_BOXMOT_REID_ON=0` — align with `pipeline_lab/server/app.py` mapping so runs match production BoxMOT path.

**Params YAML:** `main.py` promotes **`SWAY_*`** from `--params` YAML.

**Budget:** effective dimension count varies by branch (~8–12); **40+** trials minimum; after **40+**, run **`optuna.importance.get_param_importances(study)`** and narrow space.

**Do not vary in the default production study:** `SWAY_UNLOCK_DETECTION_TUNING`, `SWAY_UNLOCK_PHASE3_STITCH_TUNING`, etc.; **`SWAY_HYBRID_SAM_OVERLAP`** stays master-locked; Phase 4+ irrelevant at `after_phase_3`.

```bash
# After copying data/ground_truth/sweep_sequences.example.yaml → sweep_sequences.yaml
# and placing videos + MOT files (paths relative to repo root — works on Lambda after rsync):
python -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml
# Runs until stopped: Ctrl+C / SIGTERM, or `touch output/sweeps/optuna/STOP` (current trial finishes first).
# Optional hard cap: --n-trials 60   Best so far: --show-best
# Default DB: output/sweeps/optuna/sweep.db
optuna-dashboard sqlite:///$(pwd)/output/sweeps/optuna/sweep.db   # optional
```

**MedianPruner:** `auto_sweep.py` calls `trial.report(score, step)` **after each** benchmark video (`sequence_order` in config). If the first clip scores below the running median, Optuna **prunes** the trial and skips the remaining videos — important for paid GPU time.

---

## 8. Implementation layout and artifacts

```text
data/ground_truth/         # portable videos + gt/gt.txt (see README there)
  sweep_sequences.example.yaml   # committed; copy → sweep_sequences.yaml (gitignored)
tools/auto_sweep.py        # Optuna TPE + MedianPruner (per-video report)
tools/convert_data_json_to_mot.py
output/sweeps/optuna/sweep.db    # default SQLite storage (under gitignored output/)
output/sweeps/optuna/sweep_log.jsonl
output/sweeps/optuna/STOP        # optional: create while running to stop after current trial (--stop-file to override path)
```

**MOT file:** `python -m tools.convert_data_json_to_mot data.json pred.txt`  
**Scoring:** prefer `benchmark_trackeval` / `trackeval_runner` — avoid duplicating TrackEval CLIs.

**Between Optuna trials** in a long-lived process: same GPU cleanup as **§5.1**.

---

## 9. Optuna search space (Phase 1–3) — decision tree

**Choreography rule:** **`SWAY_YOLO_DETECTION_STRIDE` is locked to `1` forever for this automation.** Stride **2** skips detector frames; fast dance breaks Kalman / association and burns compute on bad trials. Do not pass `2` to Optuna.

**Tracker rule:** **`deep_ocsort`** only (no ByteTrack, no `deep_ocsort_osnet` in this study). Emit the same BoxMOT env the Lab uses for that card.

**Handshake + SAM IoU:** Pipeline Lab **hard-sets** `SWAY_HYBRID_SAM_IOU_TRIGGER=0.10` when enqueueing **`sway_handshake`**. The sweep driver must call **`python main.py … --params trial.yaml`** so each trial’s YAML can set **`SWAY_PHASE13_MODE=sway_handshake`** **and** **`SWAY_HYBRID_SAM_IOU_TRIGGER`** to the Optuna-chosen value (**§9.4**). Do not route handshake IoU sweeps through **`POST /api/runs`** unless the server is changed to stop overriding IoU.

---

### 9.1 Root — always tuned (every trial)

| YAML / env key | Categorical values |
|----------------|-------------------|
| `SWAY_YOLO_WEIGHTS` | `yolo26l`, `yolo26l_dancetrack`, `yolo26l_dancetrack_crowdhuman` *(drop any choice whose `.pt` is missing)* |
| `SWAY_YOLO_CONF` | `0.15`, `0.18`, `0.22`, `0.26`, `0.30` |
| `SWAY_PRETRACK_NMS_IOU` | `0.40`, `0.45`, `0.50`, `0.55`, `0.60` |
| `SWAY_BOXMOT_MAX_AGE` | `90`, `120`, `150`, `180` |
| `SWAY_BOXMOT_MATCH_THRESH` | `0.20`, `0.25`, `0.30`, `0.35` |
| `SWAY_STITCH_MAX_FRAME_GAP` | `45`, `60`, `75`, `90`, `120` |
| `sway_global_aflink_mode` (logical) | `neural_if_available`, `force_heuristic` *(if neural chosen and `models/AFLink_epoch20.pth` missing → heuristic)* |
| `SWAY_PHASE13_MODE` | `standard`, `dancer_registry`, `sway_handshake` *(branch selector)* |
| `SWAY_YOLO_DETECTION_STRIDE` | **`1` only** (fixed; still write it every trial) |

**AFLink mode → env (no `SWAY_GLOBAL_AFLINK_MODE` key in `main.py`):** `force_heuristic` ⇒ set **`SWAY_GLOBAL_AFLINK=0`**; `neural_if_available` ⇒ **omit** `SWAY_GLOBAL_AFLINK` (neural when `models/AFLink_epoch20.pth` exists) — same as `pipeline_lab/server/app.py`.

Plus **BoxMOT / deep OC-SORT** env (fixed across trials): `SWAY_USE_BOXMOT=1`, `SWAY_BOXMOT_TRACKER=deepocsort`, `SWAY_BOXMOT_REID_ON=0` (and no OSNet weights).

---

### 9.2 Branch A — `standard`

Tune **hybrid SAM overlap trigger** (standard overlap refiner):

| Key | Values |
|-----|--------|
| `SWAY_HYBRID_SAM_IOU_TRIGGER` | `0.35`, `0.40`, `0.42`, `0.45`, `0.50`, `0.55`, `0.60` |

Do **not** sample registry-only env vars in this branch.

---

### 9.3 Branch B — `dancer_registry`

Tune **registry / dormant** knobs (read in `sway/dancer_registry_pipeline.py` / `sway/tracker.py`):

| Key | Values |
|-----|--------|
| `SWAY_REGISTRY_TOUCH_IOU` | `0.05`, `0.08`, `0.10`, `0.12`, `0.15`, `0.18`, `0.20` |
| `SWAY_REGISTRY_SWAP_MARGIN` | `0.02`, `0.04`, `0.06`, `0.08`, `0.10`, `0.12`, `0.15` |
| `SWAY_DORMANT_MAX_GAP` | `90`, `120`, `150`, `180`, `200` |

Do **not** sample `SWAY_HYBRID_SAM_IOU_TRIGGER` in this branch (per product choice: optimize color/registry path, not SAM overlap trigger).

---

### 9.4 Branch C — `sway_handshake`

Tune **handshake SAM2 overlap trigger** — **this is the main handshake knob**; Optuna must be allowed to move it:

| Key | Values |
|-----|--------|
| `SWAY_HYBRID_SAM_IOU_TRIGGER` | `0.05`, `0.10`, `0.15`, `0.20`, `0.25` |

Emit via **`--params` YAML** together with `SWAY_PHASE13_MODE=sway_handshake` so the value is **not** overwritten by the Lab UI lock.

Do **not** sample registry env vars in this branch.

---

### 9.5 Single checklist — everything this automation may set per trial

Copy for implementers; **exactly one branch column** applies after `SWAY_PHASE13_MODE` is chosen:

| Variable | Root (always) | + `standard` | + `dancer_registry` | + `sway_handshake` |
|----------|---------------|--------------|---------------------|---------------------|
| `SWAY_YOLO_WEIGHTS` | ✓ | | | |
| `SWAY_YOLO_CONF` | ✓ | | | |
| `SWAY_PRETRACK_NMS_IOU` | ✓ | | | |
| `SWAY_BOXMOT_MAX_AGE` | ✓ | | | |
| `SWAY_BOXMOT_MATCH_THRESH` | ✓ | | | |
| `SWAY_STITCH_MAX_FRAME_GAP` | ✓ | | | |
| `sway_global_aflink_mode` → `SWAY_GLOBAL_AFLINK` | ✓ | | | |
| `SWAY_YOLO_DETECTION_STRIDE` | **1** (fixed) | | | |
| `SWAY_PHASE13_MODE` | ✓ | | | |
| BoxMOT deep OC-SORT env | fixed | | | |
| `SWAY_HYBRID_SAM_IOU_TRIGGER` | — | ✓ | — | ✓ |
| `SWAY_REGISTRY_TOUCH_IOU` | — | — | ✓ | — |
| `SWAY_REGISTRY_SWAP_MARGIN` | — | — | ✓ | — |
| `SWAY_DORMANT_MAX_GAP` | — | — | ✓ | — |

**Explicitly out of this automation (v1):** `sway_bidirectional_track_pass`, `sway_gnn_track_refine`, `tracker_technology` ≠ deep_ocsort, `sway_yolo_detection_stride` ≠ 1, `SWAY_UNLOCK_*`, and Lab-only overrides that fight the trial YAML.

---

### 9.6 Second study (later)

**Bidirectional** pass (~2× Phase 1–2 cost) stays **out of v1**; rerun a focused study after a winner is found.

---

## 10. Anti-patterns

1. Single-video optimization only.  
2. Deleting **`sweep.db`** — use new **`study_name`** instead.  
3. **`SWAY_UNLOCK_*`** in the default “production” study.  
4. Comparing trials across different **`git_sha`** / code versions.  
5. Ranking by **IDSW** alone — use **§6.3** composite.  
6. Ignoring **wall time** per video.  
7. **GNN** trials without checkpoint.  
8. More trials without **importance analysis** to shrink the space.  
9. **`SWAY_YOLO_DETECTION_STRIDE=2`** (or any Optuna cell for stride ≠ **1**) — wastes trials on choreography.  
10. **Handshake IoU sweeps through Lab enqueue** — Lab forces **0.10**; use **`main.py --params`** (§9, §9.4).

---

## 11. Log schema (recommended)

Minimal fields: `sweep_id`, `run_id`, `timestamp`, `git_sha`, `stop_after_boundary`, `config`, per-video `trackeval` + `composite_score` + `wall_s`, `aggregate_score` (harmonic mean), `diagnosis`, `promoted`.

Full example:

```json
{
  "run_id": "a3f7b2c1",
  "sweep_id": "20260326_phase_a",
  "timestamp": "2026-03-26T14:22:00Z",
  "git_sha": "abc123def456",
  "stop_after_boundary": "after_phase_3",
  "config": { "sway_yolo_conf": 0.22 },
  "videos": {
    "video_a": { "path": "...", "wall_s": 142.3, "composite_score": 0.734 }
  },
  "aggregate_score": 0.698,
  "diagnosis": { "primary_issue": "high_frag", "suggested_action": "increase sway_stitch_max_frame_gap" },
  "promoted": false
}
```

**Promotion to `sweep_winners.yaml`:** all videos evaluated; aggregate **≥ best + 0.01** (tunable); no per-video composite **&lt; 0.50**; **`git_sha`** policy; **COMPLETE** trial, not pruned. Copy winners into **PIPELINE_FINDINGS**.

---

## 12. First-run checklist

1. Lay out GT; verify frames, resolution, 1-based alignment.  
2. One manual `main.py` run with **`after_phase_3`**; inspect `data.json`.  
3. `convert_data_json_to_mot` + spot-check vs. `gt.txt`.  
4. One **`benchmark_trackeval`** end-to-end; note metric keys for §6.3.  
5. `pip install optuna`.  
6. Wire video paths + GT in driver.  
7. Run study; watch `sweep_log.jsonl` / `sweep.db`.  
8. If scores are nonsense, check alignment before more trials.  
9. After ~40 trials, importances → focused study.  
10. Document in **PIPELINE_FINDINGS**.

---

## 13. After Phase 1–3 is locked

Move stop to **`after_phase_5`** (or beyond), add **pose** metrics (OKS or proxies), extend metric→action map for prune/pose sweeps.

---

## 14. Quick commands

```bash
cd sway_pose_mvp/

python -m tools.run_sweep --config benchmarks/sweep_config.yaml
python -m tools.run_sweep --no-stop-after-boundary   # legacy Phase 6+ sweep_config
python -m tools.run_sweep --exhaustive

python -m tools.convert_data_json_to_mot output/.../data.json output/.../pred.txt
python -m tools.benchmark_trackeval --ground-truth benchmarks/my_gt.yaml --json output/.../data.json
python -m tools.benchmark --ground-truth benchmarks/my_gt.yaml --json output/.../data.json --trackeval

python -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml
python -m tools.auto_sweep --timeout-per-video 900
python -m tools.auto_sweep --n-trials 40   # optional fixed cap instead of run-until-stopped
```

---

## 15. Related files

| File | Role |
|------|------|
| `docs/MASTER_PIPELINE_GUIDELINE.md` | Phases, checkpoints, locks |
| `docs/PIPELINE_FINDINGS_AND_BEST_CONFIGS.md` | Winners, full recipes |
| `sway/mot_format.py` | MOT I/O |
| `tools/benchmark_trackeval.py` | TrackEval CLI |
| `tools/run_sweep.py` | Sequential sweep + JSONL |
| `tools/convert_data_json_to_mot.py` | `data.json` → `pred.txt` |
| `tools/auto_sweep.py` | Optuna Phase 1–3 sweep |
| `data/ground_truth/` | Portable benchmark media + `sweep_sequences*.yaml` |
| `benchmarks/sweep_config.yaml` | Legacy full-pipeline param_sets |
| `sway/pipeline_config_schema.py` | Tunable fields, locks |

---

*Living document. Update §6.2 when new failure modes appear in sweep logs.*
