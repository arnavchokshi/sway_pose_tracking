# Human review → model iteration

What to capture when reviewing pipeline outputs, **how to run offline (e.g. on a plane)**, and how we use the export when you’re back online.

## Offline / “plane” checklist

1. **Copy the whole batch folder** (e.g. `output/flight_batch/`) onto your machine — you need per-clip `*_poses.mp4`, `data.json`, and `prune_log.json` (and `review/index.html` from the generator).
2. **Video playback:** Some browsers block local video on `file://`. From `sway_pose_mvp` run:
   ```bash
   python review_app/serve_review.py output/flight_batch
   ```
   then open `http://127.0.0.1:8899/review/index.html` (works fully offline after the first load if assets are local).
3. **Review in the page** — overlays explain what happened:
   - **Green** (in the MP4) = final kept dancers  
   - **Red** (canvas) = track **pruning** (`prune_log` / table)  
   - **Amber** = **collision dedup** (duplicate pose on same person, other ID kept)  
   - **Violet** = **bbox sanitize** (pose dropped vs box)  
   Re-run the pipeline on a clip if `data.json` is missing `pruned_overlay` / `dropped_pose_overlay` (older exports).
4. **Remove a clip from the batch (optional):** With **`serve_review.py`** (http:// only), use **Remove from batch** on a sample. That deletes the **source video** from the input folder (using `batch_manifest.json`), removes that clip’s **output** subfolder, updates the manifest, and regenerates `review/index.html`. Download JSONL first if you need labels for that clip.
5. **Before you close the tab or clear browser data:** Sidebar → **Download JSONL**. That file is the only portable copy of your typed review (it also lives in `localStorage` until cleared).
6. **When you’re back online**, share **`human_review_<folder>.jsonl`** plus the **output folder** (or a zip) for any clips you flagged — enough to tie text to `data.json` frames and video.

## Workflow (batch)

1. **Batch-run** all prepared videos into one folder:
   ```bash
   cd sway_pose_mvp
   python batch_run_for_review.py --input-dir /path/to/your/videos --output-root output/flight_batch --pose-model base --skip-existing
   ```
2. Open the review site (prefer **`serve_review.py`** as above, or generated `review/index.html`).
3. **Review** each clip; the page **autosaves** in the browser.
4. **Download JSONL** and keep it next to the output folder (or zip the whole tree).

## What the UI is optimized to collect

| Area | Why it matters |
|------|----------------|
| **Rating + pass/fail** | Prioritize clips; optional ship / no-ship. |
| **Time ranges + per-row notes** | One window per issue; notes carry **symptoms**, **stage** (detection / tracking / pose / prune / scoring), and **track IDs** — maps to `data.json` / clip mining. |
| **Truth (count + notes)** | Separate wrong **N** boxes from wrong **IDs** without a long form. |
| **For model iteration** | **What worked** (don’t regress), **wrong track/prune/dedup**, **hypothesis & next step** — exported explicitly for whoever fixes the model. |

## Why JSONL + this schema?

JSONL (one JSON object per line) is easy to **append**, **diff**, **grep**, and **feed to tools/LLMs** without loading a huge array into memory.

**Schema version** is `2.2`. No separate **Issues** grid or **reviewer confidence** dropdown — put stage + symptoms in each **Time & issues** row’s note (and optional **For model iteration** text). `reviewer_confidence`, `model_feedback.suspected_subsystems`, and `what_went_wrong` may still appear in exports from older JSONL imports.

## Export file format (per line)

Each line is one object:

```json
{
  "schema_version": "2.2",
  "sample_id": "IMG_2946",
  "output_root_name": "flight_batch",
  "artifacts": {
    "rendered_video": "../IMG_2946/IMG_2946_poses.mp4",
    "data_json": "../IMG_2946/data.json",
    "prune_log": "../IMG_2946/prune_log.json"
  },
  "batch_context": { "...": "from batch_manifest.json if present" },
  "review": {
    "overall_quality": "good",
    "binary_pass": true,
    "reviewer_confidence": null,
    "failure_modes": [],
    "primary_failure_mode": null,
    "problem_segments_sec": [
      { "start_sec": 12.0, "end_sec": 18.0, "note": "elbows unstable" }
    ],
    "affected_tracks_note": "ID 58 prune reason looks wrong",
    "ground_truth": {
      "expected_dancer_count": "5",
      "additional_ground_truth": "5 in frame most of the clip; one partially covered from 0:20",
      "ground_truth_confidence": "",
      "observed_vs_expected": "",
      "scene_and_camera_notes": "",
      "identity_or_costume_notes": "",
      "timing_or_choreography_notes": ""
    },
    "ground_truth_expectation": "5 in frame most of the clip; one partially covered from 0:20",
    "model_feedback": {
      "suspected_subsystems": [],
      "suspected_subsystem": "",
      "failure_scope": "",
      "what_went_wrong": "",
      "what_went_well": "Chorus sync overlay readable",
      "additional_comments": ""
    },
    "notes_for_model": "Suspect short_track false positive; try relaxing min span for edge entrants",
    "manual_status": "",
    "updated_at": "2026-03-19T12:00:00.000Z"
  }
}
```

`ground_truth_expectation` duplicates `ground_truth.additional_ground_truth` for older tooling.

**Every sample appears once** in the download (including rows with empty reviews) so line count matches video count; filter downstream on `review.overall_quality`, free-text fields, or time segments.

## What’s inside `data.json` (for debugging)

- **`frames`**: per-frame poses, boxes, scores (final export).  
- **`metadata.prune_entries`**: pruned tracks and rules (if embedded).  
- **`metadata.review_overlay_legend`**: color key for review overlays.  
- **`pruned_overlay`**: per-frame pruned boxes (red overlay).  
- **`dropped_pose_overlay`**: per-frame **dedup** / **sanitize** drops (amber/violet).  

## After the flight

Share:

- `human_review_<name>.jsonl`
- The **`output_root` folder** (or zip), especially `*/data.json`, `*_poses.mp4`, and `prune_log.json` for samples you flagged.

That is enough to **cluster failure types**, **tie issues to time ranges**, and **prioritize** the next training or heuristics change.
