# Pipeline findings and best configurations

Living document: we add dated entries here whenever we compare recipes or knobs and conclude what works best for a given failure mode (occlusion, crowd, mirrors, etc.). Each entry states the **winning configuration in full** so Lab batches, trees, and CLI runs can copy it without guesswork.

**Related:** master-locked stack (detection letterbox, hybrid SAM ROI, stitch thresholds, pose locks, etc.) stays in [`MASTER_PIPELINE_GUIDELINE.md`](./MASTER_PIPELINE_GUIDELINE.md)—this file only records **tunable** choices and our **qualitative verdicts**.

---

## How to add a new finding

1. Give a short **title** and **date**.
2. **Context:** clip(s), phase you judged (e.g. end of Phase 3), and what looked wrong with the baseline.
3. **Winner:** name the recipe or variant.
4. **Full configuration:** complete Lab `fields` table (or equivalent `SWAY_*` YAML) for the winner—not only the diff from baseline.
5. Optional: link to preview paths, Compare/Tinder notes, or benchmark YAMLs under `benchmarks/`.

---

## 2026-03 — BigTest: OpenFloor “recovery bias” beats stock OpenFloor (Phase 1–3)

### Context

- **Video:** `BigTest.mov` (and consistent qualitative pattern on other clips in prior Compare/Tinder sessions).
- **Judgment point:** **End of Phase 3** (detection + tracking + post-track stitch)—preview `01_tracks_post_stitch.mp4`.
- **Detector weights for this test:** `yolo26l_dancetrack_crowdhuman` (both arms of the comparison).
- **Problem with stock “OpenFloor + Competition + SharpHipHop + Standard”:** After a dancer passed behind another, YOLO could still see them briefly; when they **reappeared elsewhere**, the **ID box often did not follow**—it looked **stuck** while other stacks kept “chasing” and re-attached the same ID when the dancer came back.
- **Technical alignment (summary):** Stock OpenFloor uses **stricter** box association (`sway_boxmot_match_thresh` **0.35**), **motion-only** Deep OC-SORT (no track-time OSNet), and a **higher** hybrid SAM IoU trigger (**0.50**) so overlap refinement runs **less** often—good for separation, weaker for **re-association after cover**.

### Verdict

**The refined (“recovery”) variant produced clearly better Phase 3 behavior** on BigTest for ID continuity through behind-body occlusion and re-entry: side-by-side export `output/bigtest_openfloor_phase3_side_by_side.mp4` (left = stock OpenFloor tracking knobs, right = recovery).

We adopt the **recovery** settings below as the **recommended** evolution of the OpenFloor competition stack when **occlusion / reappear elsewhere** matters more than absolute minimum merge risk.

### Saved as configuration (repo)

The winner is checked in as a first-class preset—not only this doc:

| Where | What |
|-------|------|
| **Lab UI — Phases 1–3** | Preset **“Open Floor (recovery bias)”** (`p13_open_floor_recovery`) in `pipeline_lab/web/src/configPresets.ts`. Pick it in the phase-group cards, then match Phases 4–6 / 7–9 to **Competition** + **Sharp Hip-Hop** if you want the full named stack. |
| **`GET /api/pipeline_matrix`** | Row id **`preset_open_competition_recovery`**, label **`M23_preset_open_floor_competition_recovery`** (`sway/pipeline_matrix_presets.py`, version **13**). |
| **`python -m tools.queue_preset_tests`** | Recipe name **`OpenFloor + Competition + SharpHipHop + Standard (recovery bias)`** (`tools/queue_preset_tests.py`). |
| **CLI Phases 1–3 only** | `benchmarks/bigtest_openfloor_phase3_recovery_env.yaml` |

### Stock OpenFloor (reference — not preferred for this failure mode)

For comparison only; matches `tools/queue_preset_tests.py` → `OpenFloor + Competition + SharpHipHop + Standard` tracking-related knobs, with weights swapped to CrowdHuman for the A/B run:

| Field / env | Value |
|-------------|--------|
| `sway_yolo_weights` | `yolo26l_dancetrack_crowdhuman` |
| `sway_yolo_conf` | `0.30` |
| `sway_pretrack_nms_iou` | `0.80` |
| `sway_hybrid_sam_iou_trigger` | **`0.50`** |
| `sway_boxmot_max_age` | **`120`** |
| `sway_boxmot_match_thresh` | **`0.35`** |
| `tracker_technology` | `deep_ocsort` |
| `sway_phase13_mode` | `standard` |

CLI-only mirror (Phases 1–3 quick run): `benchmarks/bigtest_openfloor_phase3_orig_env.yaml`.

---

### Winner: OpenFloor + Competition + SharpHipHop + Standard — **recovery bias** (best results in this study)

**Name (working):** `OpenFloor + Competition + SharpHipHop + Standard (recovery bias)`  
**Intent:** Same preset philosophy as OpenFloor (open stage, competition-grade cleanup intent, sharp hip-hop phase weights) but **easier frame-to-frame box association**, **more hybrid SAM** on overlap, and **slightly longer** track memory so IDs survive short gaps and re-glue when the dancer reappears.

#### Full Lab `fields` (entire recipe)

Use this dict as the **complete** recipe when enqueueing from the Lab or batch tools (schema field IDs). It is the **stock OpenFloor preset** from `queue_preset_tests.py` with **only** the three tracking/overlap knobs changed and **CrowdHuman** detector weights.

```yaml
# OpenFloor + Competition + SharpHipHop + Standard (recovery bias) — FULL recipe
sway_phase13_mode: standard
sway_yolo_weights: yolo26l_dancetrack_crowdhuman
tracker_technology: deep_ocsort
sway_yolo_conf: 0.30
sway_pretrack_nms_iou: 0.80
sway_hybrid_sam_iou_trigger: 0.42
sway_boxmot_max_age: 165
sway_boxmot_match_thresh: 0.29
pose_model: ViTPose-Large
sway_pose_3d_lift: true
temporal_pose_refine: true
pose_visibility_threshold: 0.22
dedup_min_pair_oks: 0.75
dedup_antipartner_min_iou: 0.15
prune_threshold: 0.60
sync_score_min: 0.12
smoother_beta: 0.55
```

#### Delta vs stock OpenFloor (what changed)

| Field | Stock OpenFloor | **Recovery (winner)** |
|-------|-----------------|------------------------|
| `sway_yolo_weights` | `yolo26l_dancetrack` (in matrix preset) | **`yolo26l_dancetrack_crowdhuman`** (required for this study; keep if CH weights help your data) |
| `sway_hybrid_sam_iou_trigger` | `0.50` | **`0.42`** |
| `sway_boxmot_max_age` | `120` | **`165`** |
| `sway_boxmot_match_thresh` | `0.35` | **`0.29`** |

All other fields in the table above match the original OpenFloor + Competition + SharpHipHop + Standard preset.

#### CLI / partial run (`main.py` — `SWAY_*` only)

For **Phase 1–3 only** with `--params`, the repo keeps a copy that matches the **tracking + detector** slice:

- **File:** `benchmarks/bigtest_openfloor_phase3_recovery_env.yaml`

Contents:

```yaml
SWAY_YOLO_WEIGHTS: yolo26l_dancetrack_crowdhuman
SWAY_YOLO_CONF: "0.30"
SWAY_PRETRACK_NMS_IOU: "0.80"
SWAY_HYBRID_SAM_IOU_TRIGGER: "0.42"
SWAY_BOXMOT_MAX_AGE: "165"
SWAY_BOXMOT_MATCH_THRESH: "0.29"
SWAY_BOXMOT_TRACKER: deepocsort
SWAY_BOXMOT_REID_ON: "0"
SWAY_PHASE13_MODE: standard
SWAY_USE_BOXMOT: "1"
```

Example:

```bash
python main.py /path/to/BigTest.mov \
  --output-dir output/run_recovery_p3 \
  --save-phase-previews \
  --stop-after-boundary after_phase_3 \
  --params benchmarks/bigtest_openfloor_phase3_recovery_env.yaml
```

**Note:** `apply_sway_params_to_env` only promotes keys starting with `SWAY_`. Full-pipeline Lab runs should use the **Lab `fields` YAML block** earlier in this section, not the slim env file alone.

#### Tradeoffs (expect these)

- **More hybrid SAM** (lower IoU trigger): **slower** Phase 1–2; on BigTest, SAM refined **many more** frames than stock (qualitative log: ~151 vs ~25 refined frames in one run).
- **Looser match threshold:** slightly **higher** risk of **merging** two people into one ID in extreme overlap; monitor on mirror / identical-outfit scenes.
- **Longer `max_age`:** wrong IDs can **persist** slightly longer if a mistake already happened—pair with qualitative review.

#### Artifacts from the comparison run

| Artifact | Path (under `sway_pose_mvp/`) |
|----------|----------------------------------|
| Phase 3 preview, stock knobs | `output/bigtest_openfloor_orig_p3/phase_previews/01_tracks_post_stitch.mp4` |
| Phase 3 preview, recovery | `output/bigtest_openfloor_recovery_p3/phase_previews/01_tracks_post_stitch.mp4` |
| Side-by-side (left orig, right recovery) | `output/bigtest_openfloor_phase3_side_by_side.mp4` |

---

## Future entries

_Add new sections below (newest first or oldest first—stay consistent)._

<!-- Template:
### YYYY-MM — Short title

**Verdict:** …

**Full configuration:** …
-->
