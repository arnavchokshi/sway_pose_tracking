# Sweep v3 — Full Results Analysis

**This file is the single consolidated record** for sweep_v3: results, best configs, tuning takeaways, and **where every artifact lives on disk** (repo root = `sway_pose_mvp/`).

**Generated:** 2026-03-30  
**GPU Instance (terminated / optional):** `gpu_1x_a10` · `ubuntu@150.136.111.175` · `us-east-1`  
**Study name:** `sweep_v3`  

---

## Single archive — paths on this machine

Everything needed for **future sweeps, retuning, and UI** is under the repo; nothing else is required from the cloud instance.

| What | Path (relative to `sway_pose_mvp/`) |
|------|-------------------------------------|
| Optuna DB (all trials, resumable) | `output/sweeps/optuna/sweep.db` |
| JSON snapshot (347 trials, scores, params, timings) | `output/sweeps/optuna/sweep_status.json` |
| Per-trial event log (sweep_v3) | `output/sweeps/optuna/sweep_log.jsonl` |
| Runner stdout | `output/sweeps/optuna/sweep_runner.log` |
| Earlier study (comparison) | `output/sweeps/optuna/sweep_log_phase13_v2_final_*.jsonl`, `sweep_phase13_v2_final_*.db` |
| Top-20 trial dirs (phase previews + outputs) | `output/sweeps/optuna/trial_XXXXX_<sequence>/` (100 dirs) |
| Sweep video list + weights (5-seq harmonic mean) | `data/ground_truth/sweep_sequences.yaml` |
| All five GT clips + labels | `data/ground_truth/{bigtest,gymtest,mirrortest,aditest,easytest}/` |
| Human-readable analysis (this doc) | `docs/SWEEP_RESULTS_ANALYSIS.md` |
| Playbook for future sweeps | `docs/GT_DRIVEN_SWEEP_AND_TUNING_PLAYBOOK.md` |

**Models** (repro / new runs): `models/` — same weights as Lambda (`yolo26l_dancetrack.pt`, `sam2.1_b.pt`, `osnet_*.pt`, etc.).  

**Code** used on the sweep box matches this repo (`sway/tracker.py`, `tools/auto_sweep.py`, compat modules under `sway/`).  

**Lambda pull (UI):** `POST /api/optuna-sweep/pull-lambda` uses `scp`; on macOS the server must not capture `scp` stdout (fixed in `pipeline_lab/server/app.py`).

---

## Summary

| Metric | Value |
|--------|-------|
| Total trials run | 347 complete, 0 pruned, 11 zero-value (crashed/bad config) |
| Total GPU time | ~20.0 hours |
| Best aggregate score (harmonic mean) | **0.6597** (Trial #173) |
| Sweep direction | MAXIMIZE |
| Sequences evaluated | aditest, bigtest, easytest, gymtest, mirrortest |
| Avg trial duration | ~207 s (~3.5 min per trial) |

---

## Best Trial — #173

**Aggregate harmonic mean: 0.6597**

### Scores per sequence
| Sequence | Score | Duration (s) |
|----------|-------|-------------|
| aditest | 0.8854 | 23.5 |
| easytest | 0.8950 | 26.3 |
| mirrortest | 0.8256 | 52.0 |
| gymtest | 0.6513 | 42.0 |
| bigtest | **0.4876** ← hardest | 72.0 |
| **Total trial** | | **215.9 s** |

### Best-trial parameters (drop-in config)
```yaml
SWAY_YOLO_WEIGHTS: yolo26l_dancetrack
SWAY_YOLO_CONF: 0.16
SWAY_DETECT_SIZE: "800"
SWAY_PRETRACK_NMS_IOU: 0.65

SWAY_BOXMOT_TRACKER: solidtrack
SWAY_BOXMOT_MAX_AGE: 135
SWAY_BOXMOT_MATCH_THRESH: 0.26
SWAY_BOXMOT_REID_WEIGHTS: osnet_x0_25_market1501.pt
b_st_TIOU: 0.3
b_st_TEMB: 0.375
b_st_EMA: 0.91

SWAY_PHASE13_MODE: sway_handshake
b_hs_SAM_IOU: 0.19
b_hs_WEAK: "1"
b_hs_VSTRIDE: "1"
b_hs_ISO: 1.6

SWAY_STITCH_MAX_FRAME_GAP: 110
SWAY_STITCH_RADIUS_BBOX_FRAC: 0.55
SWAY_SHORT_GAP_FRAMES: 20
SWAY_DORMANT_MAX_GAP: 120
SWAY_COALESCENCE_IOU_THRESH: 0.85
SWAY_COALESCENCE_CONSECUTIVE_FRAMES: 8
SWAY_BOX_INTERP_MODE: linear
sway_global_aflink_mode: neural_if_available

SWAY_HYBRID_SAM_MASK_THRESH: 0.5
SWAY_HYBRID_SAM_BBOX_PAD: 4
SWAY_HYBRID_SAM_ROI_PAD_FRAC: 0.25
```

---

## Top 20 Trials

| Rank | # | Score | Tracker | Mode | YOLO conf | NMS IOU | Max Age | Detect Size | bigtest | gymtest | mirror | adi | easy | Dur(s) |
|------|---|-------|---------|------|-----------|---------|---------|-------------|---------|---------|--------|-----|------|--------|
| 1 | 173 | 0.6597 | solidtrack | sway_handshake | 0.16 | 0.65 | 135 | 800 | 0.488 | 0.651 | 0.826 | 0.885 | 0.895 | 216 |
| 2 | 310 | 0.6592 | botsort | standard | 0.18 | 0.61 | 150 | 800 | 0.490 | 0.640 | 0.825 | 0.885 | 0.895 | 202 |
| 3 | 331 | 0.6587 | botsort | standard | 0.21 | 0.59 | 165 | 640 | 0.490 | 0.641 | 0.824 | 0.883 | 0.895 | 216 |
| 4 | 172 | 0.6584 | solidtrack | sway_handshake | 0.17 | 0.63 | 150 | 800 | 0.486 | 0.651 | 0.826 | 0.885 | 0.895 | 223 |
| 5 | 186 | 0.6578 | solidtrack | sway_handshake | 0.12 | 0.51 | 120 | 800 | 0.495 | 0.616 | 0.825 | 0.887 | 0.896 | 220 |
| 6 | 330 | 0.6566 | botsort | standard | 0.17 | 0.65 | 195 | 640 | 0.485 | 0.642 | 0.825 | 0.885 | 0.895 | 211 |
| 7 | 257 | 0.6562 | solidtrack | sway_handshake | 0.10 | 0.63 | 90 | 800 | 0.480 | 0.654 | 0.828 | 0.887 | 0.898 | 220 |
| 8 | 246 | 0.6560 | solidtrack | sway_handshake | 0.10 | 0.64 | 75 | 800 | 0.480 | 0.655 | 0.828 | 0.887 | 0.898 | 219 |
| 9 | 250 | 0.6558 | solidtrack | sway_handshake | 0.10 | 0.64 | 45 | 800 | 0.479 | 0.655 | 0.828 | 0.887 | 0.898 | 229 |
| 10 | 338 | 0.6556 | botsort | standard | 0.17 | 0.59 | 195 | 640 | 0.488 | 0.627 | 0.826 | 0.885 | 0.895 | 213 |
| 11 | 334 | 0.6545 | botsort | standard | 0.16 | 0.61 | 180 | 640 | 0.483 | 0.638 | 0.826 | 0.885 | 0.895 | 217 |
| 12 | 340 | 0.6542 | botsort | standard | 0.18 | 0.65 | 210 | 640 | 0.484 | 0.632 | 0.825 | 0.884 | 0.895 | 215 |
| 13 | 214 | 0.6535 | solidtrack | sway_handshake | 0.19 | 0.63 | 120 | 960 | 0.478 | 0.652 | 0.823 | 0.883 | 0.895 | 251 |
| 14 | 166 | 0.6534 | solidtrack | sway_handshake | 0.17 | 0.64 | 150 | 800 | 0.477 | 0.651 | 0.826 | 0.885 | 0.895 | 203 |
| 15 | 154 | 0.6534 | solidtrack | sway_handshake | 0.11 | 0.64 | 105 | 800 | 0.476 | 0.654 | 0.828 | 0.887 | 0.897 | 209 |
| 16 | 281 | 0.6532 | solidtrack | dancer_registry | 0.10 | 0.65 | 180 | 960 | 0.475 | 0.659 | 0.825 | 0.887 | 0.898 | 260 |
| 17 | 280 | 0.6523 | solidtrack | dancer_registry | 0.12 | 0.64 | 210 | 960 | 0.474 | 0.654 | 0.827 | 0.887 | 0.896 | 276 |
| 18 | 167 | 0.6522 | solidtrack | sway_handshake | 0.13 | 0.63 | 150 | 800 | 0.475 | 0.653 | 0.827 | 0.887 | 0.896 | 206 |
| 19 | 194 | 0.6514 | solidtrack | sway_handshake | 0.20 | 0.64 | 165 | 640 | 0.477 | 0.650 | 0.812 | 0.883 | 0.895 | 214 |
| 20 | 265 | 0.6511 | solidtrack | standard | 0.11 | 0.57 | 45 | 960 | 0.476 | 0.642 | 0.826 | 0.887 | 0.897 | 211 |

---

## What Worked Well

### 1. YOLO Weights: `yolo26l_dancetrack` — universally dominant
All 50 top trials used `yolo26l_dancetrack`. No other weight variant appeared in the top 50. This is the clear winner for detection.

### 2. Tracker: SolidTrack wins on quality, BotSORT wins on speed
- **SolidTrack** (74%): Higher scores on gymtest/mirrortest. Better on harder sequences. Longer tracks, better re-ID.
- **BotSORT** (26%): Slightly better bigtest, faster (~200s vs 220s avg). Simpler config, fewer knobs.
- DeepOCSORt and ByteTrack never reached top 50 — SolidTrack + BotSORT are the only viable trackers.

### 3. Phase 1-3 Mode: `sway_handshake` dominates
- 29/50 top trials used `sway_handshake` (SAM-assisted handshake linking)
- `standard` had 18/50 — competitive especially with BotSORT
- `dancer_registry` scored well on gymtest (0.659 max) but slower and less consistent

### 4. Detection size 800 — best all-round
- 800px: 26/50 top trials — best balance
- 640px: 12/50 — faster, modest score drop (~0.5% aggregate)  
- 960px: 12/50 — no consistent gain over 800, adds ~25% latency

### 5. YOLO conf 0.10–0.21 (sweet spot 0.16–0.17)
Low confidence keeps all detections, post-track NMS does the heavy lifting. Going above 0.25 hurts recall on partially occluded dancers.

### 6. High pre-track NMS IOU (0.59–0.65)
Looser pre-track NMS (keeping more overlapping boxes) consistently outperforms aggressive suppression. Best range: **0.60–0.65**.

### 7. Neural AfLink — always on
`neural_if_available` was universal in the top 50. Never force-heuristic.

### 8. Linear interpolation — strongly preferred
46/50 top trials used `linear` box interpolation. GSI was tried but offered no consistent improvement.

### 9. SAM hybrid: `b_hs_WEAK=1`, `b_hs_VSTRIDE=1`, `b_hs_SAM_IOU ≈ 0.19–0.20`
Weak-cue mode ON, stride 1 (every frame), SAM IoU trigger ~0.19 — tight but not too tight.

### 10. ReID: `osnet_x0_25_market1501` is best bang-for-buck
29/37 SolidTrack top trials used this model. Heavier models (osnet_x1_0_msmt17) added cost with marginal gain.

---

## What Did NOT Work Well

### 1. bigtest remains the hard constraint
**Max score: 0.495** (vs 0.899 for aditest). bigtest is the performance ceiling. Heavy occlusion + group splits + entrance/exit cycling are the main failure modes. No configuration broke through 0.50.

### 2. DeepOCSORt / ByteTrack
Neither tracker appeared in top 50 trials. ByteTrack especially struggled with identity preservation across occlusions.

### 3. Very low max_age (< 45 frames)
Trials with `SWAY_BOXMOT_MAX_AGE < 45` dropped significantly on mirrortest and bigtest. Dancers disappear behind mirrors or other people for extended periods.

### 4. High YOLO conf (> 0.25)
Conf ≥ 0.25 caused recall drops on partially visible dancers. A few early trials confirmed this.

### 5. GSI interpolation
Rarely outperformed linear. Adds complexity with minimal gain.

### 6. `dancer_registry` mode
Competitive scores but 25–35% slower than `sway_handshake`. Not worth the latency unless gymtest specifically needs it.

### 7. Detection size 960
No significant score gain over 800, adds ~25% GPU time per trial. Not recommended for production.

### 8. Zero-value trials (11 total)
Early trials (#12, #21, #46, #61, #65, etc.) had complete score collapse — all sequences zero. Likely a bad param combination (very low conf + very high NMS IOU killing all detections). These were early exploration trials and the optimizer quickly learned to avoid this region.

---

## Score Distribution By Sequence

| Sequence | Min | Max | Mean | Median | Ceiling Assessment |
|----------|-----|-----|------|--------|--------------------|
| aditest | 0.000 | **0.899** | 0.846 | 0.883 | Near-saturated. Top trials cluster at 0.883–0.899. |
| easytest | 0.000 | **0.898** | 0.850 | 0.894 | Near-saturated. Consistent across all top configs. |
| mirrortest | 0.000 | **0.833** | 0.744 | 0.816 | Good, moderate variance. More room to improve. |
| gymtest | 0.000 | **0.659** | 0.578 | 0.605 | Significant variance. dancer_registry helps here. |
| bigtest | 0.000 | **0.495** | 0.416 | 0.433 | **Bottleneck.** Hard ceiling around 0.50. Needs architecture work. |

---

## Timing Data

| Metric | Value |
|--------|-------|
| Fastest trial | 36 s (likely early crash/short run) |
| Slowest trial | 1146 s (early outlier, large param combo) |
| Mean trial duration | 207 s |
| Median trial duration | 211 s |
| **Total GPU compute** | **~20 hours** |
| aditest mean | 22 s |
| easytest mean | 26 s |
| gymtest mean | 39 s |
| mirrortest mean | 51 s |
| bigtest mean | 69 s ← longest (most frames + density) |

---

## Parameter Sweet Spots (from top-50 distribution)

| Parameter | Best Range | Notes |
|-----------|------------|-------|
| `SWAY_YOLO_CONF` | 0.10 – 0.21 | Sweet spot 0.16–0.17 |
| `SWAY_PRETRACK_NMS_IOU` | 0.59 – 0.65 | High is better (looser pre-NMS) |
| `SWAY_DETECT_SIZE` | 800 | 640 if speed matters |
| `SWAY_BOXMOT_MAX_AGE` | 90 – 180 | 120–150 is the core range |
| `SWAY_BOXMOT_MATCH_THRESH` | 0.15 – 0.30 | 0.22–0.27 is safest |
| `SWAY_STITCH_MAX_FRAME_GAP` | 80 – 130 | Longer helps occlusion recovery |
| `SWAY_STITCH_RADIUS_BBOX_FRAC` | 0.55 – 0.90 | |
| `SWAY_SHORT_GAP_FRAMES` | 15 – 25 | |
| `SWAY_DORMANT_MAX_GAP` | 90 – 150 | |
| `SWAY_COALESCENCE_IOU_THRESH` | 0.75 – 0.90 | High = less aggressive merging |
| `SWAY_COALESCENCE_CONSECUTIVE_FRAMES` | 8 – 16 | |
| `b_st_EMA` | 0.85 – 0.93 | High EMA stability wins |
| `b_st_TIOU` | 0.30 – 0.40 | |
| `b_hs_SAM_IOU` | 0.15 – 0.22 | Trigger threshold for SAM |
| `b_hs_ISO` | 1.4 – 1.8 | |
| `SWAY_HYBRID_SAM_MASK_THRESH` | 0.40 – 0.55 | |
| `SWAY_HYBRID_SAM_ROI_PAD_FRAC` | 0.15 – 0.25 | |

---

## Implications for Future Pipeline Development

### Immediate wins
1. **Lock in Trial #173 params** as the production baseline. It consistently wins on the overall harmonic mean.
2. **BotSORT variant** (Trial #310, score 0.6592) is nearly as good but ~15s faster per trial — good for rapid iteration.
3. Pre-NMS IOU should be bumped from 0.5 (old default) to **0.65** — the sweep is definitive here.

### bigtest is the next frontier
The hard ceiling at ~0.50 on bigtest suggests the limiting factors are:
- Track identity loss during group occlusion (not recoverable by tracking params alone)
- Entrance/exit cycling where dancers leave and re-enter frame
- This likely requires **better post-tracking re-ID** (appearance model improvements) or **longer-horizon temporal linking**

### Diminishing returns above trial ~200
The top-20 scores span only 0.6511–0.6597 — a range of 0.0086. The optimizer converged well; further Optuna tuning with the same param space will not move the needle. Next step is **architecture changes** (new backbone, better re-ID model, improved SAM integration).

### SolidTrack + sway_handshake is the production combo
This pairing won 29/50 top slots. BotSORT + standard is a simpler, faster backup that's nearly as good.

---

## Local Data Inventory

The **canonical path list** is the table **Single archive — paths on this machine** at the top of this file. Tree view of the sweep directory:

```
output/sweeps/optuna/
├── sweep.db                          # Optuna SQLite (all studies in this file)
├── sweep_status.json                 # JSON snapshot — sweep_v3 trials + user_attrs
├── sweep_log.jsonl                   # Per-trial JSONL event log (sweep_v3)
├── sweep_log_phase13_v2_final_...    # Previous study log (phase13_v2_final)
├── sweep_phase13_v2_final_....db     # Previous study SQLite DB
├── sweep_runner.log                  # Runner stdout/stderr
├── trial_XXXXX_<sequence>/           # Top 20 trial × 5 sequences ≈ 100 dirs
│   ├── phase_previews/               # Per-phase MP4s for visual QA
│   └── output/                       # data.json, tracking outputs
```

Ground truth for all five sequences: `data/ground_truth/` (see table at top).

`sweep.db` can be re-loaded for analysis or follow-on studies. `sweep_status.json` is what the Lab UI reads for the sweep page.

---

## Previous Study Reference (phase13_v2_final)

An earlier study (`sweep_log_phase13_v2_final_20260327_045558.jsonl`, `sweep_phase13_v2_final_20260327_045558.db`) was also pulled. This was the run immediately before sweep_v3. Both DBs are preserved locally for longitudinal comparison.
