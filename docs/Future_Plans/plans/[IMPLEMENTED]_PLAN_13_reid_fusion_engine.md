# Build Multi-Signal Re-ID Fusion Engine

> **Implementation Phase:** I2 (Re-ID Upgrade) · **Lean Core** · **Sweep Gate:** Gate 2 — Re-ID accuracy after occlusion > 90%, bigtest HOTA > 0.58
> **Swappable components:** Each signal is independently togglable via `SWAY_REID_*_ENABLED` flags. Signal weights are the primary sweep target (sweep phase S2). Signals are registered via `sway/reid_factory.py` (see FUTURE_PIPELINE.md §12.1).

**Objective:** Implement the confidence-gated weighted ensemble that fuses six independent re-ID signals (part-based appearance, KPR keypoint-prompted, skeleton gait, face, color histogram, spatial formation prior) into a single re-ID score for each candidate identity match. Each signal contributes only when it passes a quality gate. Weights are auto-normalized after gating removes unavailable signals. This fusion is what transforms re-ID from "works 80% of the time" (single OSNet) to "works 97%+ of the time" (every available identity cue exploited).

## Inputs & Dependencies

* **Upstream Data:** For each re-ID query (a re-emerging dancer): the dancer's current feature set (part embeddings, KPR embedding, gait embedding, face embedding, color histograms, spatial position). For the gallery: each enrolled dancer's stored feature set.
* **Prior Steps:** PLAN_07 (enrollment gallery), PLAN_08 (BPBreID), PLAN_09 (KPR), PLAN_10 (MoCos gait), PLAN_11 (ArcFace face), PLAN_12 (color histograms). Each signal module provides an `extract()` and `compare()` interface.

## Step-by-Step Implementation

1. Create `sway/reid_fusion.py`.
2. Define dataclass `ReIDQuery`:
   - `track_id: int`
   - `part_embeddings: Optional[PartEmbeddings]`
   - `kpr_embedding: Optional[np.ndarray]`
   - `gait_embedding: Optional[np.ndarray]`
   - `face_embedding: Optional[np.ndarray]`
   - `color_histograms: Optional[Dict[str, np.ndarray]]`
   - `spatial_position: Tuple[float, float]`
   - `is_multi_person_overlap: bool` (True if query crop contains multiple people)
   - `skeleton_window_length: int` (frames of skeleton data available)
3. Implement class `ReIDFusionEngine`:
   - `__init__(self, weights: Dict[str, float], gallery: List[DancerGallery])`:
     Weights from env: `SWAY_REID_W_PART` (0.30), `W_KPR` (0.15), `W_SKELETON` (0.20), `W_FACE` (0.20), `W_COLOR` (0.10), `W_SPATIAL` (0.05).
   - `match(self, query: ReIDQuery) -> Tuple[int, float]`:
     a. For each enrolled dancer in the gallery, compute per-signal distances:
        - `part_dist`: BPBreID `compare()` — gated: only if `query.part_embeddings` has ≥ `SWAY_REID_PART_MIN_VISIBLE` visible parts (default 3).
        - `kpr_dist`: KPR `compare()` — gated: only if `query.is_multi_person_overlap` is True AND `query.kpr_embedding` is not None.
        - `gait_dist`: MoCos `compare()` — gated: only if `query.skeleton_window_length >= SWAY_REID_SKEL_MIN_WINDOW` (default 30).
        - `face_dist`: ArcFace `compare()` — gated: only if `query.face_embedding` is not None.
        - `color_dist`: color histogram `compare()` — always available (no gate).
        - `spatial_dist`: Euclidean distance between query spatial position and gallery dancer's last known position, normalized by frame diagonal. Apply exponential decay: multiply by `exp(-SWAY_REID_SPATIAL_DECAY * frames_since_last_seen)`. Always available.
     b. Determine active signals: those that passed their quality gates.
     c. Collect active weights. Re-normalize so active weights sum to 1.0.
     d. Compute `fused_score = sum(normalized_weight_i * (1 - dist_i))` for each gallery dancer.
     e. Return `(best_dancer_id, best_score)`.
   - `match_batch(self, queries: List[ReIDQuery]) -> List[Tuple[int, float]]`: Match multiple queries (used for group-split re-ID, PLAN_15).
4. Implement the spatial formation prior inline (no separate module needed):
   - Maintain `last_known_positions: Dict[int, Tuple[float, float, int]]` — (x, y, frame) per dancer_id.
   - Distance = Euclidean between query position and last known position, decayed by time since last seen.
5. Add confidence threshold: if `best_score < SWAY_REID_MIN_CONFIDENCE` (default 0.50), return `(UNKNOWN, best_score)` — do not assign an identity. Flag for human review.
6. Wire into `main.py` re-ID call site (currently in `crossover.py`): when a track transitions from DORMANT → ACTIVE/PARTIAL, call `fusion.match()` with the new track's features against the gallery.

## Technical Considerations & Performance

* **Architecture Notes:** The fusion is pure arithmetic — no neural network in the fusion step itself. Cost: 6 distance computations × N gallery dancers per re-ID event. For N=10 dancers: negligible. The expensive part is feature extraction (handled by the individual signal modules).
* **Edge Cases & I/O:** When ALL optional signals are unavailable (no face, no gait, no KPR, no part embeddings — only color + spatial): the fusion degrades to color + spatial only. This should be rare (part-based re-ID requires only a crop). If it happens, flag the re-ID as low confidence.

## Validation & Testing

* **Verification:** On bigtest: for each re-emergence event, log all 6 signal distances, the active signals, the fused score, and the matched identity. Compare ground truth. Compute re-ID accuracy breakdown: which signals contribute most in which scenarios.
* **Metrics:** Fused re-ID accuracy ≥ 95% on bigtest (target 97%+). Compare vs individual signals: fusion must outperform the best individual signal by ≥ 5%.

## Integration & Next Steps

* **Outputs:** `ReIDFusionEngine` class with `match()` and `match_batch()`. Consumed by: PLAN_15 (Hungarian solver calls `match_batch()`), PLAN_16 (backward pass uses fusion for forward↔reverse stitching), `main.py` re-ID events.
* **Open Questions/Risks:** The signal weights are the highest-impact sweep target. Run a dedicated Optuna sweep (sweep phase S2) to optimize the 6 weights + EMA params.

## Swapability & Experimentation

**Signal registration pattern:** The fusion engine dynamically discovers available signals via a registry. Each signal module (PLAN_08–12) registers itself, and the fusion engine only uses signals whose models are installed and whose `SWAY_REID_*_ENABLED` flag is ON. This means:

- **Lean core (I2):** Only `part` (BPBreID), `kpr`, `color`, and `spatial` are enabled. `skeleton` and `face` are disabled (their plans are experiment add-ons).
- **After I6 experiments:** Enable `skeleton` and/or `face` via env vars. No code changes needed — just flip `SWAY_REID_SKEL_MODEL=mocos` + `SWAY_REID_FACE_MODEL=arcface`.

```python
# sway/reid_factory.py
def create_reid_ensemble(env) -> ReIDFusionEngine:
    signals = []
    # Always available (lean core)
    signals.append(create_part_reid(env.get("SWAY_REID_PART_MODEL", "bpbreid")))
    if env.get("SWAY_REID_KPR_ENABLED", "1") == "1":
        signals.append(KPRSignal(env))
    signals.append(ColorHistogramSignal(env))
    signals.append(SpatialPriorSignal(env))
    # Experiment add-ons (default OFF)
    if env.get("SWAY_REID_SKEL_MODEL"):
        signals.append(create_skeleton_reid(env["SWAY_REID_SKEL_MODEL"]))
    if env.get("SWAY_REID_FACE_MODEL"):
        signals.append(create_face_reid(env["SWAY_REID_FACE_MODEL"]))
    return ReIDFusionEngine(signals, weights_from_env(env))
```

**Manual A/B recipes for signal combinations:**

```bash
# Test lean core Re-ID (part + KPR + color + spatial only)
python -m tools.pipeline_matrix_runs --recipe reid_lean_core ...

# Test with face added
python -m tools.pipeline_matrix_runs --recipe reid_lean_core_face ...

# Test with skeleton gait added
python -m tools.pipeline_matrix_runs --recipe reid_lean_core_skeleton ...

# Test all 6 signals
python -m tools.pipeline_matrix_runs --recipe reid_full_ensemble ...
```

**Sweep readiness:** The 6 signal weights (`SWAY_REID_W_*`) form the core of sweep phase S2. The `suggest_reid_params()` function in `auto_sweep.py` suggests all 6 weights + `SWAY_REID_EMA_ALPHA_HIGH` + `SWAY_REID_EMA_ISOLATION_DIST` = 8-dimensional search. With only lean-core signals enabled, the skeleton and face weights are automatically zeroed by the quality gate — no special handling needed.
