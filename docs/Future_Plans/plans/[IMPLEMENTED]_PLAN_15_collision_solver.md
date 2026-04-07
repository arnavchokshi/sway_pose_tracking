# Build Group-Split Hungarian and DP Collision Solver

> **Implementation Phase:** I2 (Re-ID Upgrade — Hungarian solver) · **Lean Core** · DP solver is an optimization (add after Hungarian validates)
> **Sweep Gate:** Gate 2 — bigtest HOTA > 0.58
> **Swappable component:** `SWAY_COLLISION_SOLVER` — choices: `greedy` (current baseline), `hungarian` (lean core), `dp` (optimization for ≤5 tracks)

**Objective:** Replace greedy sequential matching during group coalescence events with a global optimization approach. When N tracks merge into a cluster and N detections re-emerge, solve the identity assignment as a single Hungarian operation over the full N×N embedding distance matrix. For complex collisions (3+ tracks, long duration), use dynamic programming to evaluate trajectory permutations. This prevents the cascade error where greedy matching gets the first assignment right but depletes the embedding pool, making subsequent assignments wrong.

## Inputs & Dependencies

* **Upstream Data:** (a) Frozen identity embeddings for all N tracks at cluster entry (from enrollment gallery + last clean frame before merge), (b) fresh embeddings for N detections at cluster exit, (c) multi-signal re-ID distances from the fusion engine (PLAN_13).
* **Prior Steps:** PLAN_13 (re-ID fusion provides match distances), PLAN_07 (enrollment gallery provides frozen embeddings), PLAN_01 (state machine detects coalescence via DORMANT transition).

## Step-by-Step Implementation

1. Create `sway/collision_solver.py`.
2. Implement **coalescence event detection** class `CoalescenceDetector`:
   a. Reuse existing coalescence logic from `tracker.py`: when pairwise box/mask IoU exceeds `SWAY_COALESCENCE_IOU_THRESH` (default 0.85) for `SWAY_COALESCENCE_CONSECUTIVE_FRAMES` (default 8) consecutive frames, declare a coalescence event.
   b. Record the entry frame and the set of involved track IDs.
   c. Freeze all identity embeddings for involved tracks at the entry frame: `frozen_embeddings[track_id] = copy(gallery[track_id])`.
   d. During coalescence: do NOT update any involved gallery (alpha = 0 via pose-gated EMA, PLAN_14).
3. Implement **coalescence exit detection**: when the IoU between previously coalesced tracks drops below `SWAY_COALESCENCE_IOU_THRESH * 0.7` (hysteresis), AND the detector finds N distinct detections near the cluster region, declare exit.
4. Implement `solve_hungarian(frozen_embs: List, exit_embs: List, fusion_engine: ReIDFusionEngine) -> List[Tuple[int, int]]`:
   a. Build the N×N cost matrix: `cost[i][j] = 1 - fusion_engine.match_single(frozen_embs[i], exit_embs[j])`.
   b. Run `scipy.optimize.linear_sum_assignment(cost_matrix)`.
   c. Return list of `(frozen_track_id, exit_detection_idx)` pairs.
5. Implement `solve_dp(frozen_embs, exit_embs, fusion_engine, max_perms=120) -> List[Tuple[int, int]]`:
   a. If N ≤ 5 (N! ≤ 120): enumerate all N! permutations.
   b. For each permutation, compute total cost = sum of per-pair distances.
   c. Return the permutation with minimum total cost.
   d. If N > 5: fall back to Hungarian (polynomial time, near-optimal).
6. Implement dispatch logic in `solve_collision(event, exit_detections, fusion_engine) -> List[Tuple[int, int]]`:
   a. N = number of tracks in the coalescence event.
   b. If N < `SWAY_COLLISION_MIN_TRACKS` (default 3): use greedy matching (current behavior, for 1:1 or 2:1 simple cases).
   c. If N ≤ 5: use `solve_dp()`.
   d. If N > 5: use `solve_hungarian()`.
7. After solving: assign exit detections to the corresponding frozen track IDs. Resume tracking with the correct identities. Update galleries with clean post-collision embeddings.
8. Integration with backward pass (PLAN_16): if backward pass data is available, include reverse-pass embeddings as additional evidence in the cost matrix. Double the columns: each frozen track is matched against both forward-exit and reverse-exit observations.

## Technical Considerations & Performance

* **Architecture Notes:** Hungarian is O(N³), DP is O(N! × N). For N ≤ 5 (the typical case — 5 dancers), both are instantaneous (<1ms). The expensive part is computing the N×N distance matrix (6 signal comparisons × N² pairs), but N ≤ 10 makes this negligible.
* **Edge Cases & I/O:** Partial coalescence: not all N tracks may re-emerge simultaneously. If only K < N detections appear at exit: solve the K×N assignment (rectangular matrix). The unmatched frozen tracks remain DORMANT. Handle via `scipy.optimize.linear_sum_assignment` which supports rectangular matrices.

## Validation & Testing

* **Verification:** On bigtest group splits: log the cost matrix, the optimal assignment, and the greedy assignment. Compare: how often does global optimization produce a different (and correct) assignment vs greedy? Count ID switches before/after.
* **Metrics:** ≥ 90% correct identity assignment after group splits (vs ~70% for greedy). Zero cascade errors (second assignment wrong due to first assignment depleting the pool).

## Integration & Next Steps

* **Outputs:** `CoalescenceDetector` + `solve_collision()` function. Consumed by `main.py` tracking loop (called on every coalescence exit event), PLAN_16 (backward pass feeds evidence to solver).
* **Open Questions/Risks:** Detecting the exact "exit frame" is tricky — dancers may separate gradually. Use the hysteresis threshold (0.7× entry threshold) and wait for N distinct detections.
