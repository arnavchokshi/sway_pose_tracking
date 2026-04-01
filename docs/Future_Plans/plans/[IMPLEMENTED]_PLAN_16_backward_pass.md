# Build Backward-Pass Gap Filling System

> **Implementation Phase:** I5 (Backward Pass) · **Not lean core — implement after I1–I4 pass gates**
> **Sweep Gate:** Gate 5 — bigtest HOTA > 0.62, identity switches drop by ≥30%
> Toggle: `SWAY_BACKWARD_PASS_ENABLED=0|1`. ~2x processing time. A/B test ON vs OFF before committing.

**Objective:** After the forward tracking sweep, run a second tracking pass on the reversed video sequence. Tracks that start cleanly in reverse often correspond to forward tracks that ended with identity loss. A dancer who disappears at frame 500 in the forward pass often has a clean track starting at frame 500 in the reverse pass. The existing stitch layer merges forward and reverse tracks, directly addressing the re-entry failure mode that dominates bigtest (dancers disappearing for >120 frames).

## Inputs & Dependencies

* **Upstream Data:** (a) The complete forward-pass tracking results: per-frame track IDs, masks, states, embeddings, (b) the list of DORMANT/LOST tracks with their frozen galleries, (c) the original video file (for reverse-frame reading).
* **Prior Steps:** PLAN_04 (SAM2 tracker for reverse pass), PLAN_05 (COI for reverse pass), PLAN_13 (re-ID fusion for forward↔reverse stitching), PLAN_01 (state machine for reverse tracks).

## Step-by-Step Implementation

1. Create `sway/backward_pass.py`.
2. Implement video reversal: use OpenCV `VideoCapture` to read frames in reverse order. Do NOT physically reverse the video file — read frame indices in descending order: `for i in range(total_frames - 1, -1, -1): cap.set(cv2.CAP_PROP_POS_FRAMES, i)`. Alternatively, preload all frames into memory if the video fits (for a 2-min 1080p video at 30fps: ~3600 frames × 6MB = ~21GB — may not fit). Preferred approach: decode the video once in forward order, save frame indices to a memory-mapped file or frame cache, then read in reverse.
3. Implement `run_backward_pass(video_path, forward_results, sam2_tracker, fusion_engine) -> List[ReverseTrack]`:
   a. Initialize SAM2 tracker (PLAN_04) fresh for the reverse pass — do NOT reuse forward-pass memory.
   b. Initialize COI (PLAN_05) for the reverse pass.
   c. Run the detection + SAM2 tracking pipeline on reversed frames, identical to the forward pass.
   d. The reverse pass produces its own set of tracks with its own IDs. These are independent of forward-pass IDs.
4. Implement `stitch_forward_reverse(forward_tracks, reverse_tracks, fusion_engine) -> List[MergedTrack]`:
   a. For each reverse track, find its temporal overlap with dormant/lost forward tracks.
   b. Compute re-ID similarity (via fusion engine) between the reverse track's embeddings and each dormant forward track's frozen gallery.
   c. If similarity > `SWAY_BACKWARD_STITCH_MIN_SIMILARITY` (default 0.60, sweep 0.40–0.85) AND temporal gap < `SWAY_BACKWARD_STITCH_MAX_GAP` (default 300, sweep 120–600): merge.
   d. Merging: the forward track provides identity (dancer_id from enrollment), the reverse track fills the gap frames with pose/mask data.
   e. For unmatched reverse tracks: if they correspond to a new person not in the enrollment gallery, assign a new ID.
5. Feed merged results to the collision solver (PLAN_15): when resolving group splits, the DP solver receives both forward-pass and reverse-pass observations at the cluster boundary, doubling the identity evidence.
6. Add env vars: `SWAY_BACKWARD_PASS_ENABLED` (default `1`, option to disable), `SWAY_BACKWARD_STITCH_MIN_SIMILARITY` (default 0.60), `SWAY_BACKWARD_STITCH_MAX_GAP` (default 300).
7. In `main.py`, add the backward pass as a post-processing step after the forward pass completes. Guard with `SWAY_BACKWARD_PASS_ENABLED`.

## Technical Considerations & Performance

* **Architecture Notes:** The backward pass approximately doubles processing time. For a 2-min video at 3.5 min/trial: expect ~7 min with backward pass. This is acceptable within the 30-min offline budget. The backward pass is fully independent of the forward pass — it could run in a separate process, but sequential is simpler.
* **Edge Cases & I/O:** Reading video frames in reverse is slow with standard VideoCapture seek. Preferred: decode the entire video once into a frame buffer (numpy memmap or HDF5), then index both forward and backward. This adds ~5 seconds of setup time but makes the reverse pass as fast as the forward pass. If memory is tight, use frame caching with LRU eviction.

## Validation & Testing

* **Verification:** Run on bigtest with backward pass enabled. Log: (a) how many forward dormant tracks get stitched to reverse tracks, (b) identity correctness of the stitched tracks (compare vs GT), (c) gap frames filled.
* **Metrics:** bigtest HOTA should improve by ≥ 5 points (from ~0.55 with other improvements to ~0.60+ with backward pass). ≥ 50% of dormant forward tracks should find a matching reverse track.

## Integration & Next Steps

* **Outputs:** `run_backward_pass()` and `stitch_forward_reverse()` functions. Merged tracks replace the forward-only tracks for downstream processing (pose, scoring, critique).
* **Open Questions/Risks:** Reverse-pass tracking may produce different track segmentation than forward (different clusters, different occlusion patterns). The stitch logic must handle partial temporal overlap, not just gap filling.
