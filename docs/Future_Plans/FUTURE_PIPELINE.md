# Sway Future Pipeline — The Unified Ideal Architecture

**Status:** Living design document — the target architecture for maximum-accuracy single-camera dance critique.

**Generated from:** Section 17 of `TECHNICAL_PIPELINE_PAPER.md`, sweep_v3 results (347 trials, 2026-03-30), `MASTER_PIPELINE_GUIDELINE.md` current-state analysis, and continued SOTA research survey (March 2026).

**North star:** Correct identity preservation through every occlusion event, and honest per-keypoint confidence so the critique layer never gives feedback on joints it cannot actually see.

---

## 1. What the Sweep Proved — and Where It Hit a Wall

### What works (locked production baseline — Trial #173)

| Decision | Evidence |
|----------|----------|
| **YOLO26l-DanceTrack weights** | All top-50 trials used them. No other weight variant appeared. |
| **SolidTrack + sway_handshake** | 29/50 top trials. Best identity continuity through partial occlusion. |
| **Detection size 800px** | 26/50 top trials. 960px adds 25% latency with no consistent gain. |
| **YOLO conf 0.16–0.17** | Low conf keeps all detections; pre-track NMS does the filtering. Above 0.25 loses partially occluded dancers. |
| **Pre-track NMS IoU 0.60–0.65** | Looser NMS (keep more overlapping boxes) consistently wins. |
| **Neural AFLink always on** | Universal in top 50. Heuristic linker never competitive. |
| **Linear interpolation** | 46/50 top trials. GSI offered no improvement. |
| **SAM hybrid: weak-cue ON, stride 1, SAM IoU ~0.19** | Tight trigger, every frame. |
| **osnet_x0_25_market1501** | Best bang-for-buck Re-ID model. Heavier variants add cost with marginal gain. |
| **EMA 0.85–0.93** | High stability wins. |

### The wall: bigtest caps at ~0.50

| Sequence | Ceiling | Assessment |
|----------|---------|------------|
| aditest | 0.899 | Saturated |
| easytest | 0.898 | Saturated |
| mirrortest | 0.833 | Moderate room |
| gymtest | 0.659 | Significant room |
| **bigtest** | **0.495** | **Hard ceiling — architecture-limited** |

The top-20 scores span only 0.0086. The optimizer converged; further Optuna tuning with the same parameter space cannot move the needle. The remaining gap is architectural.

### Root causes (why bigtest fails)

1. **Track identity loss during group occlusion.** When all five dancers cluster, bounding boxes collapse into overlapping rectangles. No amount of max_age tuning helps when the detections themselves are ambiguous.
2. **Entrance/exit cycling.** Dancers who leave and re-enter the frame accumulate wrong ID assignments. Neural AFLink operates within typical gap distributions; exits lasting >120 frames exceed its reliable range.
3. **Group-scale re-ID collapse.** OSNet was trained on pedestrians in normal clothing. Dancers in matching costumes under stage lighting push it past its domain. A crop-level global embedding cannot distinguish people who look nearly identical.
4. **Memory corruption during crossovers.** The tracker's appearance memory continues updating during occlusion, absorbing the wrong person's pixels. When the cluster breaks, the stored embedding no longer represents the original person.

---

## 2. Reframing the Problem for Critique

Everything built so far optimizes for HOTA — a general tracking metric that penalizes any gap in continuity. But the actual goal is **individual dancer critiques**: personalized feedback on form, timing, and synchronization. This changes the problem:

- **We do not need to track during full occlusion.** If a dancer's limbs are not visible, there is no pose to critique. A gap in tracking is acceptable; a gap in identity is not.
- **We absolutely must re-identify correctly on re-emergence.** A wrong re-ID contaminates dancer A's critique with dancer B's frames. This is far worse than a missing interval.
- **Partial visibility is usable.** If upper body is visible, that is valid critique data for arm angles, shoulder alignment, and head position. We track and analyze visible joints, and explicitly flag invisible ones.
- **Per-keypoint confidence is mandatory.** The output must tell the user: "we are confident about these 12 joints but NOT about these 5" — never hallucinate feedback on joints we cannot see.

This reframing relaxes one constraint (continuous tracking through full occlusion) while dramatically tightening two others (re-ID accuracy after occlusion, and honest per-joint confidence reporting).

---

## 3. Quantitative Targets

| Metric | Current (Trial #173) | Target | Notes |
|--------|---------------------|--------|-------|
| **Aggregate HOTA (5-seq harmonic mean)** | 0.6597 | ≥ 0.78 | Closing ~60% of the gap to SOTA (0.755–0.758 SAM2MOT) |
| **bigtest HOTA** | 0.495 | ≥ 0.65 | The critical sequence — dense occlusion + group splits |
| **Re-ID accuracy after occlusion** | ~80% (estimated) | ≥ 97% | Measured as: % of re-emergence events where the correct ID is assigned |
| **Identity switches per sequence** | Unmeasured | ≤ 2 per 1000 frames | |
| **Per-keypoint confidence accuracy** | Not implemented | ≥ 95% | When we say "confident" about a joint, the localization error is ≤ 5px |
| **Limb visibility false-positive rate** | Not implemented | ≤ 3% | We never claim a joint is visible when it is actually occluded |
| **Offline processing budget** | ~3.5 min/trial | ≤ 30 min for 2-min video | Accuracy over speed — this runs after practice, not during |

---

## 4. Design Constraints (Unchanged)

1. **Single monocular camera.** No multi-camera setups. Every component is designed for one viewpoint.
2. **Offline processing.** Accuracy over latency. A 2-minute video may take up to 30 minutes. Critiques are consumed after practice.
3. **No tracking through full occlusion.** If all pixels are hidden, the track goes dormant. No hallucinated trajectory, no interpolated position, no estimated pose. The system explicitly says "I cannot see dancer N from frames X–Y."
4. **No comparative judgments.** The system reports "Dancer 3's right arm was 15 degrees short" — not "Dancer 3 was worse than Dancer 5." Value judgments are for the choreographer.
5. **Honest uncertainty.** If we are not confident about a joint, we say so. Never generate critique on data we do not trust.

---

## 5. The Unified Ideal Pipeline — Layer by Layer

### Layer 0: Enrollment — Bootstrapping Perfect Identity

Before any tracking begins, the system builds a **closed-set identity gallery** for every dancer.

**Why this is the single highest-value change:** It converts re-ID from open-set ("who are you among all possible people?") to closed-set ("which of these N enrolled dancers are you?"). Closed-set re-ID is dramatically easier. Every downstream component benefits because comparisons are against known, high-quality references instead of noisy passively-accumulated embeddings.

**How enrollment works:**

1. During the first 5–10 seconds (or a user-selected "clear" frame where all dancers are separated), run high-quality feature extraction.
2. For each dancer, collect:
   - Multiple appearance embeddings from the best available angles
   - Part-based embeddings (head, torso, upper arms, lower arms, upper legs, lower legs) — not just a global crop
   - A face embedding if the dancer faces the camera at any point (ArcFace / AdaFace)
   - A skeleton gait signature from the opening movement (30–60 frames of 3D pose → MoCos embedding)
   - Color histogram of costume regions (shirt, pants/skirt, shoes) extracted via SAM2 mask — this is the simplest and most robust discriminator when costumes differ even slightly
   - Starting formation position (spatial prior)
3. Each dancer gets a permanent ID and optionally a user-assigned name.
4. All subsequent re-ID is matching against this gallery.

**User-assisted enrollment (Pipeline Lab UI):** Show the first clear frame with all dancers separated. The user clicks each dancer and assigns a name/number. This 10-second manual step eliminates hours of ID confusion downstream.

**Logical hybrid:** The enrollment gallery is not static. As the pipeline processes frames where a dancer is ACTIVE and fully visible (high-confidence state), the gallery is updated via EMA — but only when the per-frame embedding passes a quality gate (see Pose-Gated EMA in Layer 3).

---

### Layer 1: Detection — Co-DETR / Co-DINO with YOLO Fallback

**Current:** YOLO26l-DanceTrack. Fast, single-shot, but relies on NMS to separate overlapping detections — and NMS is where we lose people in dense clusters.

**Ideal:** Replace primary detection with **Co-DETR** or **Co-DINO** (Collaborative Detection Transformers).

**Why transformers beat YOLO here:** DETR-family detectors eliminate NMS entirely. They use learned "object queries" with Hungarian matching — each person gets exactly one predicted box with no duplicate-suppression ambiguity. Co-DETR achieves ~66 AP on COCO vs YOLO's ~55 AP at comparable sizes. SAM2MOT uses Co-DINO as its detection backbone for exactly this reason.

**Hybrid approach — YOLO as fast scout, Co-DETR as precision detector:**
- Run YOLO on every frame as a cheap first pass. When YOLO detects ≤ N non-overlapping people (low-density frames), accept its detections directly — it is fast and accurate in easy cases.
- When YOLO detects overlapping boxes (IoU > 0.3 between any pair) or the SAM2 mask propagation signals drift, invoke Co-DETR on that frame for precision re-detection.
- This hybrid avoids paying the 3–5x Co-DETR cost on every frame while getting its precision exactly when it matters (during occlusion).

**Also investigate:** RT-DETR (real-time DETR variant — faster than Co-DETR, still NMS-free), and fine-tuning YOLO26xl on DanceTrack + CrowdHuman jointly for a stronger single-shot option.

**Detection confidence output:** Every detection includes a confidence score. Detections below 0.5 confidence are flagged as "uncertain" — downstream layers treat them more cautiously (lower weight in EMA updates, wider matching threshold).

---

### Layer 2: Primary Tracking — SAM2 Mask Propagation with Cross-Object Interaction

**Current:** Detect boxes → refine overlapping boxes with SAM → feed refined boxes to SolidTrack → Kalman + appearance association. The tracker works with rectangles. When rectangles overlap, information degrades.

**Ideal:** Promote SAM2 from "helper called on overlap" to **the primary tracking engine**, incorporating the Cross-Object Interaction module from SAM2MOT.

**How tracking-by-segmentation works:**
1. On the first frame, detections (from Layer 1) prompt SAM2 to produce pixel-level segmentation masks for each dancer.
2. On subsequent frames, SAM2 **propagates** each mask forward using its temporal memory — no detector needed for existing tracks.
3. The detector is re-invoked periodically (every K frames, or when SAM2 confidence drops) to discover new entrants or recover from drift.

**Why masks beat boxes:** SAM2 masks are pixel-precise. When two dancers cross, dancer A's mask includes only dancer A's pixels. There is no "overlapping rectangle" ambiguity. SAM2's temporal memory propagates identity through appearance changes far more robustly than a Kalman filter because it attends to actual visual texture, not just box position.

**Critical addition — Cross-Object Interaction (from SAM2MOT):**

This is the key innovation that prevents memory corruption during crossovers:
- Calculate **Mask IoU** between all active tracks every frame.
- When Mask IoU between two tracks exceeds a threshold (severe collision), examine the **variance of logit scores** for each track across the past N frames.
- The track with the abrupt logit score drop is the one being occluded — its memory is **quarantined**. The corrupted memory entries (frames where the wrong person's pixels leaked in) are deleted in real-time.
- When the collision ends (Mask IoU drops), the quarantined track resumes with clean pre-collision memory.

**Why this directly solves bigtest's core failure:** The reason IDs swap during crossovers is that the tracker's memory bank continues updating during the collision, absorbing the wrong person's pixels. By the time the cluster breaks, the stored embedding no longer represents the original person. Cross-Object Interaction prevents this by quarantining memory the moment contamination is detected.

**MeMoSORT integration for motion prediction:**

MeMoSORT (67.9% HOTA on DanceTrack, +1.4% over previous best) is a TBD (tracking-by-detection) tracker, making it a drop-in alongside SAM2:
- Its **memory-augmented Kalman filter** corrects motion estimation errors by comparing predicted positions against a short-term memory of actual positions, then adjusting the process noise matrix. This helps during rapid direction changes (common in choreography).
- Its **Motion-adaptive IoU** dynamically expands the matching space when objects are moving fast and contracts it when stationary. For dancers who suddenly accelerate (leaps, turns), this prevents track loss due to rigid IoU thresholds.

**Hybrid: SAM2 handles mask-level identity; MeMoSORT handles motion prediction.** SAM2 tells us "which pixels belong to this person"; MeMoSORT tells us "where this person is likely to be next frame." The two signals are complementary — SAM2 struggles when masks become very small (distant/tiny dancers), while MeMoSORT struggles when boxes overlap (close dancers). Together they cover each other's blind spots.

---

### Layer 3: Re-ID — Multi-Signal Identity Fusion with Logical Hybrid Strategies

This is the most critical layer. When a dancer re-emerges after occlusion, we must assign the correct ID with near-perfect accuracy. No single model is sufficient. The ideal design fuses multiple independent identity signals with several logical strategies that maximize the value of each technology.

#### Signal 1: Part-Based Appearance Re-ID (PAFormer / BPBreID)

**Replace** OSNet's single global embedding with a **part-based** model.

- Computes separate embeddings for head, torso, upper arms, lower arms, upper legs, lower legs.
- When matching a partially visible person, only visible parts are compared. If upper body is all we see, compare head-to-head and torso-to-torso against the gallery, ignore legs entirely.
- **BPBreID** trains with adversarial occlusion (GiLt) — randomly masks body parts during training, forcing each part embedding to be discriminative on its own.
- Even in matching costumes, dancers differ subtly: skin tone, hair, accessories, body proportions. Part-based re-ID latches onto discriminative parts independently.

#### Signal 2: KPR — Keypoint-Prompted Re-ID for Group Occlusion

**KPR** directly addresses **Multi-Person Ambiguity (MPA)** — the situation where multiple people are visible in the same bounding box, making standard re-ID impossible.

- Uses pose keypoints as explicit prompts to the re-ID model, specifying which person inside an occluded box you are trying to identify.
- For bigtest's group occlusion frames where 3–4 dancers overlap in a tight cluster, this is the correct tool — instead of extracting a useless crop with multiple people, KPR says "identify the person whose skeleton is at these keypoint locations."

**Logical hybrid with SAM2:** Use SAM2's per-person mask to crop only the target person's pixels, then feed the mask-isolated crop to KPR. The pose keypoints tell the model which person to attend to; the mask removes the other people's pixels from the input entirely. This is double assurance — spatial (mask) and structural (keypoints) isolation.

#### Signal 3: Skeleton-Based Gait/Motion Re-ID (MoCos)

**Costume-invariant identity.** Every person has a distinctive biomechanical signature — joint flexibility, stride length, shoulder width ratio, center-of-gravity shift pattern. Even dancers performing identical choreography differ in these individual mechanics.

- **MoCos** processes a 30–60 frame window of 3D skeleton poses as a spatiotemporal graph, producing a gait identity embedding.
- This signal is orthogonal to appearance — it captures **who you are** (biomechanics), not **what you look like** (costume).
- We already compute 3D skeletons. This data is available for free.
- **Limitation:** Requires a temporal window. For very short re-appearances (<30 frames), other signals carry the load.

#### Signal 4: Face Recognition (Opportunistic)

- ArcFace / AdaFace on detected face regions within each person crop.
- Available on maybe 30–50% of frames (faces are often small, blurred, or in profile during dance).
- When available, it is the single most discriminative signal — 99%+ accuracy on clean frontal faces.
- Naturally complements skeleton re-ID: faces are most visible in frontal views, gait signatures are strongest in profile views.

#### Signal 5: Color-Based Identity (Simplest, Most Robust)

- Extract costume color histograms per body region (shirt, pants, shoes) using SAM2 masks to isolate each person's pixels.
- Even "matching" costumes often have subtle color differences under different stage lighting angles.
- This is the fastest signal to compute and provides a useful coarse filter before expensive embedding comparisons.

#### Signal 6: Spatial Formation Prior

- Maintain a running model of each dancer's typical position relative to the group centroid.
- Dance troupes practice formations. If dancer A was in position 3 for the past 200 frames and all other positions are accounted for, the unidentified person in position 3 after a group break is almost certainly dancer A.
- Not absolute (formations change), but a strong soft signal that costs nothing to compute.

#### Fusion: Confidence-Gated Weighted Ensemble

```
final_score = w_part  * part_score     (if visible parts >= 3)
            + w_kpr   * kpr_score      (if in multi-person overlap)
            + w_skel  * skeleton_score  (if temporal window >= 30 frames)
            + w_face  * face_score      (if face detected with sufficient resolution)
            + w_color * color_score     (always available from mask)
            + w_spat  * spatial_score   (always available from formation model)
```

Each signal contributes only when reliable. If no face is detected, `w_face` drops to zero and remaining signals absorb its weight. If skeleton window is too short, `w_skel` is suppressed. The weights can be learned via a small MLP trained on enrollment-vs-reappearance pairs from our own ground truth data.

#### Logical Strategy: Pose-Gated EMA

The enrollment gallery is updated over time, but **only when updates are trustworthy:**

- Run a lightweight pose estimator (ViTPose or RTMPose) and compute a per-frame **distinctiveness score** for each dancer.
- When arms are extended and the dancer is spatially isolated: the frame's embedding is clean. Update EMA aggressively (high alpha).
- When the dancer is crouched in a cluster with overlapping people: the embedding is contaminated. **Freeze EMA entirely** — do not update the gallery with this frame.
- This is a ~2-parameter addition (distinctiveness threshold, EMA alpha range) that prevents gallery pollution — the primary cause of cascading ID errors.

#### Logical Strategy: Contrastive Fine-Tuning on Own Clips

OSNet was trained on Market-1501 (pedestrians in normal clothing, well-lit corridors). Our dancers are in performance costumes under stage lighting. Fine-tuning with even 500 same-ID / different-ID pairs extracted from our own ground truth clips would close the domain gap.

- Run this as a **one-time preprocessing step** using enrollment frames + manually labeled pairs from bigtest.
- Plug the fine-tuned weights into the existing Re-ID slot.
- Expected: large improvement for free — the model learns what "same dancer" looks like in our specific visual domain.

#### Logical Strategy: Group-Split Combinatorial Re-ID

When the coalescence module detects that N tracks have merged into a single detection cluster:

1. **Freeze** a snapshot of all N identity embeddings (from enrollment gallery + last clean frame before merge).
2. While merged, do not update any of the N identities.
3. When N detections re-emerge from the cluster region, solve the assignment as a **single Hungarian operation** over the full N×N embedding distance matrix.
4. This prevents the cascade error where greedy sequential matching gets the first assignment correct but depletes the embedding pool in a way that makes subsequent assignments wrong.

---

### Layer 4: Occlusion-Aware State Machine

Every track exists in one of four states:

```
┌─────────┐    full visibility    ┌──────────┐
│  ACTIVE  │ <──────────────────> │ PARTIAL  │
│(critique │    partial vis.      │(critique  │
│  ON,     │                      │ limited,  │
│  gallery │                      │ visible   │
│  updates)│                      │ joints    │
└────┬─────┘                      │ only)     │
     │  full occlusion            └─────┬────┘
     ▼                                  ▼
┌─────────┐                      ┌──────────┐
│ DORMANT │ <──────────────────> │  LOST    │
│(critique│   timeout expired    │(archived) │
│  OFF,   │                      │           │
│  re-ID  │                      └──────────┘
│  gallery│
│  frozen)│
└─────────┘
```

**ACTIVE:** Full body visible. All keypoints estimated with confidence. Critique data generated. Gallery updated via pose-gated EMA.

**PARTIAL:** Some body parts visible (e.g., upper body during crossover). Keypoints estimated **only for visible joints.** Each joint gets a visibility flag. Critique generated for visible joints only. The system explicitly reports: "Dancer 3 — frames 450–480: upper body feedback available, lower body occluded (no data)." Gallery updated only from visible, high-confidence parts.

**DORMANT:** Person fully occluded. No critique data. Track paused. Gallery **frozen** — no updates. The Cross-Object Interaction module (Layer 2) quarantines memory to prevent contamination. The system waits to re-ID when the person reappears.

**LOST:** After extended timeout (much longer than current max_age), the dormant track moves to a long-term archive. Re-ID is still possible from the archive but at lower priority.

**Key principle:** We never generate critique data for frames we cannot see. We never lose the identity gallery. We never corrupt the gallery with wrong-person pixels.

---

### Layer 5: Backward-Pass Gap Filling

After the forward tracking sweep completes, run a **second pass on the reversed video sequence.** This is unconventional in online tracking but architecturally simple as a post-process.

**Why this works:** Tracks that start cleanly in reverse often correspond to forward tracks that ended with identity loss. A dancer who disappears at frame 500 in the forward pass (identity lost during occlusion) often has a clean track starting at frame 500 in the reverse pass (because in reverse, they "emerge" from the occlusion cleanly).

**How to merge:** The stitch layer (which already exists in the current pipeline) merges forward and reverse tracks. For each reverse track, compute re-ID similarity against all dormant forward tracks. If a confident match is found, fuse them — the forward track provides identity, the reverse track fills the gap.

**Cost:** ~2x trial time. **Expected gain:** Directly addresses the re-entry failure mode that dominates bigtest. Dancers who disappear for >120 frames get a second chance at correct re-ID from the reverse direction.

**Logical hybrid:** Run SAM2MOT's Cross-Object Interaction in the reverse pass too. During the reverse pass, occlusion events happen in reverse order — a cluster "forming" in forward is a cluster "breaking" in reverse. The reverse pass sees the break-apart as a clean emergence, which is SAM2MOT's strongest scenario. Use the clean reverse-pass IDs to validate or correct the forward-pass IDs.

---

### Layer 6: Advanced Trackers for Specific Failure Modes (Experiment Add-Ons — NOT Part of Lean Core)

**Important:** These modules are **experiment add-ons** to be tested ONE AT A TIME after the lean core (Layers 0–5 without backward pass) is validated at Implementation Gate 2 (see Section 11). Each targets a specific remaining failure mode. Do not implement multiple simultaneously until individual impact is measured. See Section 12.3 for the recipe-based workflow to A/B test each module.

#### MOTE — Disocclusion Matrix for Crossover Prediction

MOTE uses optical flow and softmax splatting to generate a **Disocclusion Matrix** — it mathematically predicts where a hidden person will reappear based on the motion field. Instead of waiting for re-emergence and then re-IDing, MOTE tells the system "person A is about to appear at position (x, y) in 3 frames."

**Where this fits:** When the state machine transitions a track from DORMANT back to PARTIAL/ACTIVE, MOTE provides a spatial prediction for the re-emergence location. If the predicted location matches the observed re-emergence, the re-ID confidence is boosted. If it does not match, the system knows something unusual happened (formation change, unexpected movement) and increases the weight on appearance/skeleton re-ID signals.

**Expected impact:** 25% reduction in identity switches over bounding-box-only methods during crossovers.

#### UMOT — Long-Term Dormancy with Historical Backtracking

Standard trackers die if occlusion lasts longer than 1–2 seconds. UMOT introduces a **Historical Backtracking Module** combined with a memory track query that explicitly separates short-term motion prediction from long-term trajectory recovery.

**Where this fits:** For bigtest's entrance/exit cycling (dancers disappearing for >120 frames), UMOT's design allows a track to stay completely dormant and un-updated until the exact moment the cluster breaks apart. Instead of the current max_age timeout guessing, UMOT's backtracking module actively searches the historical trajectory bank when a new detection appears.

#### Sentinel — Survival Boosting for Track Preservation

Sentinel's **Survival Boosting Mechanism (SBM)** preserves tracks at risk of disappearance by exploiting weak detection signals to bridge long occlusions.

**Where this fits:** The current pipeline loses tracks when YOLO confidence drops below threshold during partial occlusion. Sentinel instead maintains a "survival score" for each track — if the track was strong for the last 100 frames and suddenly weakens, SBM gives it a longer grace period and tries to find it using lower-confidence detections that the standard pipeline would discard. This directly targets the bigtest entrance/exit cycling problem.

#### MATR — Maximum Association Accuracy (Nuclear Option)

MATR achieves 71.3 HOTA on DanceTrack (AssA 61.6, IDF1 75.3) by explicitly predicting object movements across frames to update track queries in advance. This is a significant architecture change — it replaces the detect-then-associate paradigm with a predict-then-verify paradigm.

**Assessment:** MATR's DanceTrack numbers are the best in the literature for association metrics specifically. If Layers 0–5 prove insufficient for bigtest, MATR's motion prediction module could be integrated as the association engine, replacing the current SolidTrack/BotSORT association step entirely.

---

### Layer 7: Dynamic Programming for Collision Resolution

When a dense collision occurs (3+ tracks merge into an overlapping cluster), abandon greedy frame-by-frame matching entirely. Instead:

1. Identify the **entry frame** (Frame A: when the cluster forms) and **exit frame** (Frame B: when the cluster breaks apart).
2. At Frame A, snapshot all N track identities with frozen embeddings.
3. At Frame B, observe N detections re-emerging.
4. Treat A and B as two sets of nodes in a bipartite graph.
5. Use **dynamic programming** to calculate minimum-cost paths through the "dark zone" of the occlusion, evaluating every possible trajectory permutation simultaneously.
6. Solve the globally optimal assignment via Hungarian matching over the full N×N distance matrix (multi-signal: appearance + skeleton + color + spatial).

**Why this beats greedy matching:** Greedy sequential matching gets the first assignment right but often makes the second wrong because it has already "spent" the best embedding. Global optimization considers all permutations simultaneously, finding the assignment that minimizes total error across all N people.

**Logical combination with backward pass:** Run the DP solver using both forward-pass and reverse-pass observations at Frame A and Frame B. The reverse pass provides additional identity evidence (it saw the cluster break apart as a clean emergence). The DP solver ingests both sources for maximum confidence.

---

### Layer 8: Pose Estimation — Visibility-Masked, Whole-Body

**Current:** ViTPose+ Large, 17 COCO keypoints, applied uniformly to every crop.

**Ideal:** Upgrade to **RTMW** (Real-Time Multi-person Whole-body) or **ViTPose-Huge** with visibility masking.

**RTMW** outputs body (17), hand (21 per hand), foot, and facial keypoints in one forward pass. For dance critique, hand and foot positioning is often the difference between good and great. "Extend your fingers more during the reach" is impossible with body-only keypoints.

**Visibility masking:** When a dancer is in the PARTIAL state, the pose model receives the SAM2 segmentation mask as an auxiliary input. The mask tells the model "these pixels belong to this person; ignore everything else." Without this, ViTPose detects keypoints on both the target and the occluder, producing chimeric skeletons (left arm from person A, right arm from person B). With the mask, the occluder's signal is suppressed.

**Per-keypoint confidence output:** Every keypoint gets a confidence classification:

| Level | Meaning | Critique action |
|-------|---------|-----------------|
| **HIGH** | Joint clearly visible, localization error ≤ 5px | Full critique feedback |
| **MEDIUM** | Joint partially visible or at mask edge, error 5–15px | Feedback with caveat ("approximate") |
| **LOW** | Joint barely visible, high uncertainty | No critique — report as "uncertain" to user |
| **NOT_VISIBLE** | Joint occluded or outside frame | No critique — report as "not visible" to user |

The confidence level is computed from: (a) the pose model's heatmap peak value, (b) whether the joint falls inside the SAM2 mask, and (c) temporal consistency (is this joint's position consistent with the previous 5 frames?). A joint that has a high heatmap score but falls outside the mask is downgraded to LOW (likely detecting the occluder's joint, not the target's).

---

### Layer 9: 3D Lifting — Multi-Person Joint Estimation

**Current:** MotionAGFormer lifts each person independently with no knowledge of other dancers.

**Ideal:** Use **MotionBERT** (ICCV 2023) with multi-person extension.

MotionBERT's Dual-stream Spatio-temporal Transformer (DSTformer) is pretrained on motion recovery from noisy 2D inputs. It learns kinematic priors about human motion that transfer to 3D lifting, achieving lower error than MotionAGFormer on standard benchmarks.

**Multi-person joint estimation:** For critique, we need not just each dancer's 3D pose in isolation but their **relative 3D positions.** "Is dancer A 30cm too far to the right?" requires a shared coordinate frame. The ideal design:
1. Runs depth estimation once for the full scene (not per-person).
2. Estimates a floor plane from the depth map.
3. Places all 3D skeletons on that floor plane using depth-sorted positions.
4. Produces a physically consistent multi-person 3D scene.

---

### Layer 10: Critique-Specific Scoring

**Current:** Angle deviation + timing errors vs group consensus.

**Ideal:** Five-dimension biomechanical analysis:

| Dimension | What it measures | Method |
|-----------|-----------------|--------|
| **Formation accuracy** | Is each dancer in the right position? | 3D multi-person scene → position error vs formation template |
| **Timing precision** | Is the dancer hitting the beat? | Audio beat grid extraction → movement peak alignment |
| **Extension and line** | Are limbs fully extended? | Joint angle at key moments vs biomechanical maximum or reference |
| **Smoothness** | Is movement fluid or jerky? | Jerk (3rd derivative of position) per joint — lower is smoother |
| **Synchronization** | How closely does each dancer match the group? | 3D spatial deviation + timing offset + amplitude matching |

Each dimension produces a time-series score per dancer. Critique output example:

> Dancer 3, measures 12–16 (0:42–0:55): right arm extension is 15° short of the group at the peak of each cycle. You are consistently 2 frames early on the downbeat. Upper body form is strong — shoulder alignment within 3° of target. **Lower body not visible during frames 670–695 — no feedback for that interval.**

The final sentence is the confidence reporting in action: we tell the user what we cannot see rather than guessing.

---

## 6. Defaults, Sweep Parameters, and Configuration Spec

This section defines, for every layer, what is **locked** (production default — not sweepable), what is **sweepable** (with ranges and recommended defaults for Optuna), and what is an **option** (discrete architectural choice to A/B test).

The pattern follows the existing codebase convention: locked values go in `MASTER_LOCKED_*` dicts, sweep params get `SWAY_*` env vars, and the `SWAY_UNLOCK_*` gates allow overrides for experimentation.

---

### Layer 0: Enrollment

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_ENROLLMENT_ENABLED` | bool | `1` | **No — locked ON** | | Core architecture; disabling reverts to open-set re-ID |
| `SWAY_ENROLLMENT_AUTO_FRAME` | int | `0` (first frame) | Option | `0` or user-selected frame | `0` = auto-select first frame with N separated detections; user override via Lab UI |
| `SWAY_ENROLLMENT_MIN_SEPARATION_PX` | int | `80` | Sweep | `40–150` | Min pixel distance between detected people to consider them "separated" for gallery capture |
| `SWAY_ENROLLMENT_GALLERY_SIGNALS` | str | `part,color,spatial` | Option | Subset of `{part,face,skeleton,color,spatial}` | Which signals to collect at enrollment. Default excludes face/skeleton (need temporal window) |
| `SWAY_ENROLLMENT_COLOR_BINS` | int | `32` | Sweep | `16–64` | Histogram bin count per channel for costume color matching |
| `SWAY_ENROLLMENT_PART_MODEL` | enum | `bpbreid` | Option | `bpbreid`, `paformer` | Which part-based re-ID model for enrollment embeddings |

**Locked decisions:**
- Enrollment is always ON. Without it, the system reverts to sweep_v3's open-set re-ID (~80% accuracy). This is the single highest-value architectural choice and is not negotiable.
- Gallery is always frozen during DORMANT state (never updated with occluded frames).

---

### Layer 1: Detection

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_DETECTOR_PRIMARY` | enum | `yolo26l_dancetrack` | **Option** | `yolo26l_dancetrack`, `co_detr`, `co_dino`, `rt_detr`, `yolo26xl_dancetrack_crowdhuman` | Primary detector backbone. Default stays YOLO for speed; Co-DETR/RT-DETR are A/B test options |
| `SWAY_DETECTOR_HYBRID` | bool | `1` | **Option** | `0` or `1` | When ON + primary=YOLO: invoke precision detector on overlap frames. When OFF: single detector only |
| `SWAY_DETECTOR_PRECISION` | enum | `co_dino` | Option | `co_detr`, `co_dino`, `rt_detr` | Which precision detector for hybrid mode (used on overlap/drift frames) |
| `SWAY_HYBRID_OVERLAP_IOU_TRIGGER` | float | `0.30` | **Sweep** | `0.15–0.50` | Min IoU between any two YOLO boxes to trigger precision detector on that frame |
| `SWAY_YOLO_CONF` | float | `0.16` | **Sweep** | `0.08–0.25` | Carried from sweep_v3. Sweet spot 0.14–0.18. |
| `SWAY_PRETRACK_NMS_IOU` | float | `0.65` | **Sweep** | `0.55–0.75` | Carried from sweep_v3. Looser = better in dense scenes. Only applies when primary=YOLO |
| `SWAY_DETECT_SIZE` | int | `800` | **Sweep** | `640`, `800`, `960` | Carried from sweep_v3. 800 is the proven sweet spot |
| `SWAY_YOLO_WEIGHTS` | enum | `yolo26l_dancetrack` | **Option** | `yolo26l_dancetrack`, `yolo26l_dancetrack_crowdhuman`, `yolo26xl_dancetrack_crowdhuman` | Model variant for YOLO path. Future: fine-tuned YOLO26xl on DanceTrack+CrowdHuman |
| `SWAY_DETECTION_UNCERTAIN_CONF` | float | `0.50` | Sweep | `0.35–0.65` | Detections below this are flagged "uncertain" — downstream uses lower EMA weight |

**Locked decisions:**
- `SWAY_GROUP_VIDEO=1` — always on (crowd mode).
- `SWAY_CHUNK_SIZE=300`, `SWAY_YOLO_INFER_BATCH=1`, `SWAY_YOLO_HALF=0` — carried from current master lock.
- When `SWAY_DETECTOR_PRIMARY=co_detr|co_dino|rt_detr`: NMS params are ignored (DETR is NMS-free).

**Sweep strategy:** First sweep YOLO params (conf, NMS, detect_size) to re-establish baseline on new architecture. Then A/B test `SWAY_DETECTOR_HYBRID=1` vs `0` to measure precision detector impact. Then sweep `SWAY_HYBRID_OVERLAP_IOU_TRIGGER` to find optimal trigger threshold.

---

### Layer 2: Tracking (SAM2 + MeMoSORT)

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_TRACKER_ENGINE` | enum | `sam2mot` | **Option** | `sam2mot`, `solidtrack`, `memosort`, `sam2_memosort_hybrid` | Primary tracking paradigm. `sam2mot` = mask-based; `solidtrack` = current best box-based; `sam2_memosort_hybrid` = recommended ideal |
| `SWAY_SAM2_MODEL` | enum | `sam2.1_b` | Option | `sam2.1_b`, `sam2.1_l`, `sam2.1_h` | SAM2 checkpoint size. Larger = better masks, slower |
| `SWAY_SAM2_REINVOKE_STRIDE` | int | `30` | **Sweep** | `10–60` | How often (frames) to re-invoke detector for new person discovery during SAM2 propagation |
| `SWAY_SAM2_CONFIDENCE_REINVOKE` | float | `0.40` | **Sweep** | `0.20–0.60` | If SAM2 mask confidence drops below this, trigger detector re-invocation immediately |
| `SWAY_COI_MASK_IOU_THRESH` | float | `0.25` | **Sweep** | `0.10–0.50` | Cross-Object Interaction: Mask IoU threshold to detect collision between two tracks |
| `SWAY_COI_LOGIT_VARIANCE_WINDOW` | int | `10` | Sweep | `5–20` | Number of past frames to examine for logit score variance when detecting occlusion victim |
| `SWAY_COI_QUARANTINE_MODE` | enum | `delete` | Option | `delete`, `freeze` | On collision: `delete` corrupted memory entries (SAM2MOT style) or `freeze` them (simpler, less aggressive) |
| `SWAY_MEMOSORT_MEMORY_LENGTH` | int | `30` | Sweep | `10–60` | MeMoSORT: how many past positions the memory-augmented Kalman filter remembers |
| `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA` | float | `0.50` | **Sweep** | `0.20–0.80` | MeMoSORT: how aggressively IoU matching space expands with velocity |
| `SWAY_TRACK_MAX_AGE` | int | `200` | **Sweep** | `90–400` | Frames before DORMANT → LOST. Longer than sweep_v3 (135) because enrollment gallery allows safe long dormancy |
| `SWAY_TRACK_PARTIAL_MASK_FRAC` | float | `0.30` | Sweep | `0.10–0.50` | If mask area is < this fraction of enrollment mask area → transition to PARTIAL state |
| `SWAY_TRACK_DORMANT_MASK_FRAC` | float | `0.05` | Sweep | `0.01–0.15` | If mask area is < this fraction → transition to DORMANT state |

**Locked decisions:**
- Cross-Object Interaction is always ON when `SWAY_TRACKER_ENGINE` involves SAM2. This is the core defense against memory corruption.
- SAM2 mask propagation uses temporal memory (not single-frame inference). This is inherent to the SAM2MOT architecture.
- State machine (ACTIVE/PARTIAL/DORMANT/LOST) is always active regardless of tracker engine.

**Sweep strategy:** Start with `sam2_memosort_hybrid`. Sweep `SWAY_SAM2_REINVOKE_STRIDE` and `SWAY_COI_MASK_IOU_THRESH` as the two highest-impact tracking params. Then sweep MeMoSORT adaptive IoU. Compare against `solidtrack` (current best) as baseline.

**Carried from sweep_v3 (apply when `SWAY_TRACKER_ENGINE=solidtrack`):**

| Parameter | Default | Sweep Range |
|-----------|---------|-------------|
| `SWAY_BOXMOT_MATCH_THRESH` | `0.26` | `0.15–0.35` |
| `b_st_EMA` | `0.91` | `0.85–0.95` |
| `b_st_TIOU` | `0.30` | `0.25–0.45` |
| `b_st_TEMB` | `0.375` | `0.30–0.50` |

---

### Layer 3: Re-ID — Multi-Signal Fusion

#### Signal weights (the primary sweep target)

| Parameter | Type | Default | Sweep? | Range | Notes |
|-----------|------|---------|--------|-------|-------|
| `SWAY_REID_W_PART` | float | `0.30` | **Sweep** | `0.10–0.50` | Weight for part-based appearance re-ID |
| `SWAY_REID_W_KPR` | float | `0.15` | **Sweep** | `0.05–0.30` | Weight for keypoint-prompted re-ID (active only during multi-person overlap) |
| `SWAY_REID_W_SKELETON` | float | `0.20` | **Sweep** | `0.05–0.35` | Weight for skeleton gait re-ID (active only when window >= `SWAY_REID_SKEL_MIN_WINDOW`) |
| `SWAY_REID_W_FACE` | float | `0.20` | **Sweep** | `0.05–0.40` | Weight for face re-ID (active only when face detected) |
| `SWAY_REID_W_COLOR` | float | `0.10` | **Sweep** | `0.05–0.20` | Weight for color histogram matching |
| `SWAY_REID_W_SPATIAL` | float | `0.05` | **Sweep** | `0.02–0.15` | Weight for spatial formation prior |

Weights are auto-normalized after confidence gating removes unavailable signals. The sweep explores relative importance; absolute values are rescaled at runtime.

#### Signal configs

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_REID_PART_MODEL` | enum | `bpbreid` | **Option** | `bpbreid`, `paformer`, `osnet_x0_25` | Part-based model. `osnet_x0_25` = current (global embedding, no parts — fallback baseline) |
| `SWAY_REID_PART_MIN_VISIBLE` | int | `3` | Sweep | `2–5` | Min visible body parts to activate part-based signal |
| `SWAY_REID_KPR_ENABLED` | bool | `1` | Option | `0` or `1` | KPR adds cost; disable if no multi-person overlap in the clip |
| `SWAY_REID_SKEL_MODEL` | enum | `mocos` | Option | `mocos`, `gaitgl` | Skeleton gait model |
| `SWAY_REID_SKEL_MIN_WINDOW` | int | `30` | Sweep | `15–60` | Frames of skeleton data needed before gait signal activates |
| `SWAY_REID_FACE_MODEL` | enum | `arcface` | Option | `arcface`, `adaface` | Face recognition backbone |
| `SWAY_REID_FACE_MIN_SIZE` | int | `40` | Sweep | `20–60` | Min inter-eye distance (px) to consider face embedding reliable |
| `SWAY_REID_COLOR_SPACE` | enum | `hsv` | Option | `hsv`, `lab`, `rgb` | Color space for costume histogram. HSV is more robust to lighting changes |
| `SWAY_REID_SPATIAL_DECAY` | float | `0.01` | Sweep | `0.005–0.05` | Exponential decay rate for spatial formation prior (how fast the prior weakens after formation changes) |

#### EMA gallery management

| Parameter | Type | Default | Sweep? | Range | Notes |
|-----------|------|---------|--------|-------|-------|
| `SWAY_REID_EMA_ALPHA_HIGH` | float | `0.15` | **Sweep** | `0.05–0.30` | EMA update rate when dancer is isolated + extended pose (clean embedding) |
| `SWAY_REID_EMA_ALPHA_LOW` | float | `0.00` | **Locked** | | Update rate when dancer is in cluster — frozen, no update |
| `SWAY_REID_EMA_ISOLATION_DIST` | float | `1.5` | **Sweep** | `1.0–3.0` | Min bbox-height-fractions of clearance from nearest neighbor to count as "isolated" |
| `SWAY_REID_EMA_POSE_QUALITY_THRESH` | float | `0.60` | Sweep | `0.40–0.80` | Min mean keypoint confidence to allow EMA update |

**Locked decisions:**
- `SWAY_REID_EMA_ALPHA_LOW=0.00` — gallery is NEVER updated when the dancer is in a cluster. This prevents the memory corruption that causes cascading ID swaps. Non-negotiable.
- Confidence gating is always ON — signals only contribute when their quality gate is passed.
- Group-split combinatorial re-ID (Hungarian N×N) always replaces greedy matching when ≥3 tracks are involved in a coalescence event. Greedy matching is only used for isolated 1:1 re-emergence.

**Sweep strategy:** The signal weights are the highest-impact sweep target for re-ID. Run a dedicated Phase 6 sweep with `SWAY_UNLOCK_REID_DEDUP_TUNING=1`. The 6 weights plus `SWAY_REID_EMA_ALPHA_HIGH` and `SWAY_REID_EMA_ISOLATION_DIST` form an 8-dimensional search space — appropriate for 200–400 Optuna trials.

#### Contrastive fine-tuning (pre-sweep, one-time)

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `SWAY_REID_FINETUNE_ENABLED` | bool | `0` | Run contrastive fine-tuning on own clips before sweep. One-time preprocessing step |
| `SWAY_REID_FINETUNE_PAIRS` | int | `500` | Number of same-ID / different-ID pairs to extract from GT clips |
| `SWAY_REID_FINETUNE_EPOCHS` | int | `20` | Training epochs for fine-tuning |
| `SWAY_REID_FINETUNE_LR` | float | `1e-4` | Learning rate |
| `SWAY_REID_FINETUNE_BASE_MODEL` | enum | `bpbreid` | Which model to fine-tune |

This is NOT a sweep parameter. It is a one-time preprocessing step that produces fine-tuned weights, which are then used as the default for all subsequent sweeps. Run fine-tuning first, then sweep the signal weights.

---

### Layer 4: Occlusion State Machine

| Parameter | Type | Default | Sweep? | Range | Notes |
|-----------|------|---------|--------|-------|-------|
| `SWAY_STATE_PARTIAL_MASK_FRAC` | float | `0.30` | **Sweep** | `0.15–0.50` | Mask area fraction threshold: ACTIVE → PARTIAL |
| `SWAY_STATE_DORMANT_MASK_FRAC` | float | `0.05` | Sweep | `0.01–0.15` | Mask area fraction threshold: PARTIAL → DORMANT |
| `SWAY_STATE_DORMANT_MAX_FRAMES` | int | `300` | **Sweep** | `120–600` | Frames in DORMANT before → LOST. Much longer than current max_age because enrollment gallery persists |
| `SWAY_STATE_PARTIAL_MIN_JOINTS` | int | `5` | Sweep | `3–8` | Min visible keypoints (HIGH/MEDIUM confidence) to stay in PARTIAL instead of DORMANT |
| `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH` | float | `0.70` | **Sweep** | `0.50–0.85` | Pose heatmap peak value threshold for HIGH confidence classification |
| `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` | float | `0.40` | **Sweep** | `0.25–0.55` | Pose heatmap peak value threshold for MEDIUM confidence (below this → LOW) |
| `SWAY_CONFIDENCE_TEMPORAL_WINDOW` | int | `5` | Sweep | `3–10` | Frames of temporal consistency check for confidence classification |
| `SWAY_CONFIDENCE_MASK_GATE` | bool | `1` | **Locked** | | Keypoint outside SAM2 mask → downgrade to LOW regardless of heatmap score |

**Locked decisions:**
- `SWAY_CONFIDENCE_MASK_GATE=1` — if a keypoint falls outside the person's SAM2 mask, it is downgraded regardless of its heatmap score. This prevents detecting the occluder's joints as the target's.
- No critique data is generated for joints classified as LOW or NOT_VISIBLE. This is the honest uncertainty guarantee.
- Gallery is frozen in DORMANT. Gallery updates only happen in ACTIVE (all signals) and PARTIAL (visible-part signals only).

---

### Layer 5: Backward Pass

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_BACKWARD_PASS_ENABLED` | bool | `1` | **Option** | `0` or `1` | Enable reversed-video second pass. ~2x processing time |
| `SWAY_BACKWARD_STITCH_MIN_SIMILARITY` | float | `0.60` | **Sweep** | `0.40–0.85` | Min multi-signal re-ID similarity to merge a reverse track with a forward dormant track |
| `SWAY_BACKWARD_STITCH_MAX_GAP` | int | `300` | Sweep | `120–600` | Max frame gap for forward↔reverse track stitching |
| `SWAY_BACKWARD_COI_ENABLED` | bool | `1` | **Locked** | | Run Cross-Object Interaction in reverse pass too |

**Locked decisions:**
- If backward pass is enabled, it always uses the same SAM2 + COI pipeline as the forward pass.
- DP collision resolution (Layer 7) always uses both forward + reverse evidence when backward pass is on.

**Sweep strategy:** First test `SWAY_BACKWARD_PASS_ENABLED=1` vs `0` on bigtest specifically. If it helps (expected: significant bigtest improvement), sweep `SWAY_BACKWARD_STITCH_MIN_SIMILARITY`.

---

### Layer 6: Advanced Tracker Modules (Optional Add-ons)

These are discrete options to enable/disable. Each targets a specific failure mode.

| Parameter | Type | Default | Sweep? | Notes |
|-----------|------|---------|--------|-------|
| `SWAY_MOTE_DISOCCLUSION` | bool | `0` | **Option** | Enable MOTE disocclusion matrix. Adds optical flow cost. Test on bigtest specifically |
| `SWAY_MOTE_FLOW_MODEL` | enum | `raft_small` | Option | `raft_small`, `raft_large`. Only used when MOTE enabled |
| `SWAY_MOTE_CONFIDENCE_BOOST` | float | `0.15` | Sweep (if MOTE on) | `0.05–0.30`. How much to boost re-ID confidence when MOTE spatial prediction matches |
| `SWAY_UMOT_BACKTRACK` | bool | `0` | **Option** | Enable UMOT historical backtracking for long dormancy recovery |
| `SWAY_UMOT_HISTORY_LENGTH` | int | `500` | Sweep (if UMOT on) | `200–1000`. Frames of trajectory history to maintain |
| `SWAY_SENTINEL_SBM` | bool | `0` | **Option** | Enable Sentinel Survival Boosting Mechanism |
| `SWAY_SENTINEL_GRACE_MULTIPLIER` | float | `3.0` | Sweep (if SBM on) | `1.5–5.0`. How many multiples of normal grace period for strong-history tracks |
| `SWAY_SENTINEL_WEAK_DET_CONF` | float | `0.08` | Sweep (if SBM on) | `0.03–0.15`. Min YOLO conf to use as a "weak signal" for survival boosting |

**Locked decisions:**
- All advanced modules default to OFF. They are opt-in for specific failure modes.
- MATR (predict-then-verify paradigm) is NOT a config option — it is a full architecture replacement. If needed, it replaces `SWAY_TRACKER_ENGINE` entirely.

**Sweep strategy:** Test each module independently on bigtest first (single-variable A/B). Only enable multiple simultaneously after individual impact is measured. Order of testing: MOTE (most likely to help crossovers) → Sentinel (entrance/exit cycling) → UMOT (long dormancy).

---

### Layer 7: Collision Resolution

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_COLLISION_SOLVER` | enum | `hungarian` | **Option** | `hungarian`, `dp`, `greedy` | Assignment method when ≥3 tracks are in a coalescence event. `greedy` = current behavior (baseline) |
| `SWAY_COLLISION_MIN_TRACKS` | int | `3` | Sweep | `2–5` | Min tracks in cluster to trigger combinatorial solver (below this, use greedy) |
| `SWAY_COLLISION_DP_MAX_PERMUTATIONS` | int | `120` | Locked | | Cap on DP permutation search (5! = 120). Above 5 tracks in cluster, fall back to Hungarian |
| `SWAY_COALESCENCE_IOU_THRESH` | float | `0.85` | **Sweep** | `0.70–0.95` | Carried from sweep_v3. IoU threshold to declare coalescence event |
| `SWAY_COALESCENCE_CONSECUTIVE_FRAMES` | int | `8` | **Sweep** | `4–20` | Carried from sweep_v3. Consecutive frames above IoU to confirm coalescence |

**Locked decisions:**
- DP solver is capped at 5 tracks (120 permutations). For larger clusters, Hungarian matching is used (polynomial time, near-optimal).
- Embedding snapshots are always frozen at cluster entry. No gallery updates during coalescence.

---

### Layer 8: Pose Estimation

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_POSE_MODEL` | enum | `vitpose_large` | **Option** | `vitpose_large`, `vitpose_huge`, `rtmw_l`, `rtmw_x` | Pose backbone. Default stays ViTPose-Large (current production). RTMW adds hand/foot/face keypoints |
| `SWAY_POSE_MASK_GUIDED` | bool | `1` | **Option** | `0` or `1` | Feed SAM2 mask as auxiliary input to suppress occluder keypoints |
| `SWAY_POSE_KEYPOINT_SET` | enum | `coco17` | Option | `coco17`, `wholebody133` | Keypoint format. `wholebody133` available with RTMW models |
| `SWAY_POSE_VISIBILITY_THRESHOLD` | float | `0.30` | **Sweep** | `0.15–0.50` | Carried from current pipeline. Score below this → joint marked NOT_VISIBLE |
| `SWAY_POSE_SMART_PAD` | bool | `1` | **Locked** | | Smart bbox expansion before crops. Proven in sweep_v3 |

**Locked decisions:**
- `SWAY_POSE_SMART_PAD=1` — always on (proven benefit from sweep_v3).
- `SWAY_POSE_MASK_GUIDED=1` by default when SAM2 is the primary tracker. Can be disabled for A/B testing.
- Pose estimation only runs on ACTIVE and PARTIAL tracks, never on DORMANT.

**Sweep strategy:** A/B test `vitpose_large` vs `rtmw_l` first (measures whole-body vs body-only impact on critique quality). Then sweep `SWAY_POSE_VISIBILITY_THRESHOLD` with the chosen model.

---

### Layer 9: 3D Lifting

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_LIFT_BACKEND` | enum | `motionagformer` | **Option** | `motionagformer`, `motionbert` | 3D lifter. MotionAGFormer = current production. MotionBERT = ideal target |
| `SWAY_LIFT_MULTI_PERSON` | bool | `0` | **Option** | `0` or `1` | Joint multi-person lifting on shared floor plane (requires MotionBERT) |
| `SWAY_LIFT_DEPTH_SCENE` | bool | `0` | Option | `0` or `1` | Run full-scene depth estimation for floor plane (only with `SWAY_LIFT_MULTI_PERSON=1`) |

**Locked decisions:**
- All existing MotionAGFormer configs from master lock carry forward as defaults.
- `SWAY_LIFT_MULTI_PERSON` defaults OFF until MotionBERT integration is validated.

---

### Layer 10: Critique Scoring

| Parameter | Type | Default | Sweep? | Range / Choices | Notes |
|-----------|------|---------|--------|-----------------|-------|
| `SWAY_CRITIQUE_DIMENSIONS` | str | `formation,timing,extension,smoothness,sync` | Option | Comma-separated subset | Which critique dimensions to compute. All ON by default |
| `SWAY_CRITIQUE_JERK_WINDOW` | int | `5` | Sweep | `3–15` | Smoothing window for jerk computation (smoothness dimension) |
| `SWAY_CRITIQUE_BEAT_TOLERANCE_MS` | int | `100` | Sweep | `50–200` | Tolerance window (ms) for timing precision — "on the beat" vs "early/late" |
| `SWAY_CRITIQUE_MIN_CONFIDENCE` | enum | `MEDIUM` | **Locked** | | Min per-keypoint confidence level to include in critique. LOW and NOT_VISIBLE joints are excluded |
| `SWAY_CRITIQUE_REPORT_GAPS` | bool | `1` | **Locked** | | Always report "not visible" intervals in critique output |

**Locked decisions:**
- `SWAY_CRITIQUE_MIN_CONFIDENCE=MEDIUM` — only HIGH and MEDIUM joints produce feedback. LOW/NOT_VISIBLE joints are reported as gaps, never critiqued. This is the honest uncertainty guarantee.
- `SWAY_CRITIQUE_REPORT_GAPS=1` — every output always includes explicit "no data" intervals. The user always knows what the system can and cannot see.

---

### Post-Processing (carried from current pipeline, unchanged)

These parameters retain their current master-lock values. They are not part of the future architecture changes but remain relevant.

| Parameter | Default | Sweep? | Range | From |
|-----------|---------|--------|-------|------|
| `SWAY_STITCH_MAX_FRAME_GAP` | `110` | Sweep | `80–200` | sweep_v3 |
| `SWAY_STITCH_RADIUS_BBOX_FRAC` | `0.55` | Sweep | `0.40–1.00` | sweep_v3 |
| `SWAY_SHORT_GAP_FRAMES` | `20` | Sweep | `10–40` | sweep_v3 |
| `SWAY_DORMANT_MAX_GAP` | `120` | Sweep | `80–300` | sweep_v3, extended for enrollment-aware pipeline |
| `sway_global_aflink_mode` | `neural_if_available` | **Locked** | | sweep_v3 — universal in top 50 |
| `SWAY_BOX_INTERP_MODE` | `linear` | **Locked** | | sweep_v3 — 46/50 top trials |
| `SMOOTHER_MIN_CUTOFF` | `1.0` | Locked | | Current master lock |
| `SMOOTHER_BETA` | `0.55` | Sweep | `0.30–0.80` | Current tunable |

---

### Recommended Sweep Phases

The sweep should be run in phases, each building on the locked winners of the previous phase:

| Sweep Phase | What It Tunes | Approx Trials | Stop Boundary | Metric |
|-------------|---------------|---------------|---------------|--------|
| **S1: Detection + Tracking baseline** | `SWAY_YOLO_CONF`, `SWAY_PRETRACK_NMS_IOU`, `SWAY_DETECT_SIZE`, `SWAY_SAM2_REINVOKE_STRIDE`, `SWAY_COI_MASK_IOU_THRESH`, `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA`, `SWAY_TRACK_MAX_AGE` | 300–400 | `after_phase_3` | HOTA vs GT (same as sweep_v3) |
| **S2: Re-ID weights** | `SWAY_REID_W_PART`, `W_KPR`, `W_SKELETON`, `W_FACE`, `W_COLOR`, `W_SPATIAL`, `SWAY_REID_EMA_ALPHA_HIGH`, `SWAY_REID_EMA_ISOLATION_DIST` | 200–400 | `after_phase_7` | HOTA + IDF1 (identity-focused metrics) |
| **S3: State machine + confidence** | `SWAY_STATE_PARTIAL_MASK_FRAC`, `SWAY_STATE_DORMANT_MAX_FRAMES`, `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH`, `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` | 100–200 | full pipeline | HOTA + per-keypoint accuracy vs GT |
| **S4: Backward pass + collision** | `SWAY_BACKWARD_STITCH_MIN_SIMILARITY`, `SWAY_COLLISION_MIN_TRACKS`, `SWAY_COALESCENCE_IOU_THRESH` | 100–200 | full pipeline | bigtest HOTA specifically |
| **S5: Advanced modules (A/B)** | `SWAY_MOTE_DISOCCLUSION`, `SWAY_SENTINEL_SBM`, `SWAY_UMOT_BACKTRACK` (on/off + their params) | 50–100 each | full pipeline | bigtest identity switches |
| **S6: Pose + critique** | `SWAY_POSE_MODEL` (A/B), `SWAY_POSE_VISIBILITY_THRESHOLD`, `SWAY_CRITIQUE_JERK_WINDOW`, `SWAY_CRITIQUE_BEAT_TOLERANCE_MS` | 100–200 | full pipeline | Pose quality vs GT + critique accuracy |

**Total estimated sweep budget:** ~1000–1500 trials across all phases. At ~4 min/trial on A10 GPU: ~60–100 GPU-hours.

---

### Quick Reference: All Locked Defaults (Not Sweepable)

These are architectural decisions proven by sweep_v3 or mandated by the critique-accuracy north star. Override only with `SWAY_UNLOCK_*` gates for engineering tests.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `SWAY_ENROLLMENT_ENABLED` | `1` | Core architecture — open-set → closed-set |
| `SWAY_REID_EMA_ALPHA_LOW` | `0.00` | Gallery freeze in clusters — prevents memory corruption |
| `SWAY_CONFIDENCE_MASK_GATE` | `1` | Keypoint outside mask → downgraded — prevents chimeric skeletons |
| `SWAY_CRITIQUE_MIN_CONFIDENCE` | `MEDIUM` | No feedback on uncertain joints — honest uncertainty guarantee |
| `SWAY_CRITIQUE_REPORT_GAPS` | `1` | Always report "not visible" intervals |
| `sway_global_aflink_mode` | `neural_if_available` | sweep_v3: universal in top 50 |
| `SWAY_BOX_INTERP_MODE` | `linear` | sweep_v3: 46/50 top trials |
| `SWAY_POSE_SMART_PAD` | `1` | sweep_v3: proven benefit |
| `SWAY_GROUP_VIDEO` | `1` | Crowd mode always on |
| `SWAY_BACKWARD_COI_ENABLED` | `1` | If backward pass runs, COI always runs with it |
| `SWAY_COLLISION_DP_MAX_PERMUTATIONS` | `120` | Cap DP at 5 tracks for compute safety |

---

## 7. The Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ENROLLMENT (one-time)                           │
│  User selects clear frame → click each dancer → assign name          │
│  Per dancer: part embeddings + face + skeleton gait + color          │
│              histogram + starting formation position                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FORWARD PASS — FRAME PROCESSING LOOP                                │
│                                                                      │
│  1. Detection (hybrid YOLO scout + Co-DETR precision)                │
│     └─ YOLO on every frame; Co-DETR on overlap/drift frames         │
│     └─ No NMS ambiguity on Co-DETR detections                       │
│                                                                      │
│  2. SAM2 mask propagation (primary tracking signal)                  │
│     └─ Pixel-level mask per person, propagated via temporal memory   │
│     └─ Cross-Object Interaction: quarantine memory on collision      │
│     └─ MeMoSORT motion prediction: where is this person next?       │
│     └─ Mask area → ACTIVE / PARTIAL / DORMANT state                 │
│                                                                      │
│  3. Visibility-masked pose estimation (RTMW / ViTPose-Huge)         │
│     └─ SAM2 mask as auxiliary input — suppress occluder signal       │
│     └─ Whole-body keypoints: body + hands + feet + face              │
│     └─ Per-keypoint confidence: HIGH / MEDIUM / LOW / NOT_VISIBLE   │
│     └─ Only on ACTIVE and PARTIAL tracks                             │
│                                                                      │
│  4. Gallery update via pose-gated EMA                                │
│     └─ Isolated dancer, extended pose → update aggressively          │
│     └─ Clustered, crouched → freeze gallery (no update)             │
│                                                                      │
│  5. Collision detection + group-split combinatorial re-ID            │
│     └─ Freeze N embeddings on cluster entry                          │
│     └─ Hungarian N×N assignment on cluster exit                      │
│     └─ DP solver for long/complex collisions                         │
│                                                                      │
│  6. Multi-signal re-ID (on every re-emergence event)                 │
│     └─ Part-based visual (PAFormer / BPBreID)                        │
│     └─ Keypoint-prompted (KPR) for multi-person crops                │
│     └─ Skeleton gait (MoCos, if window ≥ 30 frames)                 │
│     └─ Face (ArcFace, when detected)                                 │
│     └─ Color histogram (always, from SAM2 mask)                      │
│     └─ Spatial formation prior (always)                              │
│     └─ Confidence-gated weighted fusion                              │
│                                                                      │
│  7. Per-frame data accumulation                                      │
│     └─ 2D keypoints + per-joint confidence + visibility flags        │
│     └─ Which joints are real vs occluded (no hallucination)          │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BACKWARD PASS (reversed video)                                      │
│                                                                      │
│  Same SAM2 + Cross-Object Interaction pipeline on reversed frames    │
│  Clean re-emergence = clean identity (reverse sees cluster breaks    │
│  as clean appearances)                                               │
│  Stitch reverse tracks to forward dormant tracks via re-ID           │
│  DP solver uses both forward + reverse evidence for collisions       │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING                                                     │
│                                                                      │
│  8. 1-Euro temporal smoothing (per-joint, visibility-aware)          │
│     └─ Only smooth across ACTIVE/PARTIAL frames                      │
│     └─ Do NOT interpolate through DORMANT gaps                       │
│                                                                      │
│  9. MotionBERT 3D lifting (multi-person, shared floor plane)        │
│                                                                      │
│  10. Five-dimension critique scoring                                 │
│      └─ Formation accuracy (3D positions)                            │
│      └─ Timing precision (audio-synced beats)                        │
│      └─ Extension and line quality (joint angles vs reference)       │
│      └─ Smoothness (jerk analysis per joint)                         │
│      └─ Group synchronization (3D spatial + timing + amplitude)      │
│                                                                      │
│  11. Per-dancer critique report with confidence annotations          │
│      └─ Timestamped feedback per dimension                           │
│      └─ Explicit "not visible" intervals — no hallucinated feedback  │
│      └─ Per-keypoint confidence in every reported metric              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Expected Performance Ceiling

| Component | Current | Ideal | Expected Gain |
|-----------|---------|-------|---------------|
| Detection | YOLO26l (COCO AP ~55) | Co-DETR / Co-DINO (COCO AP ~66) | +20% detection accuracy on overlapping bodies |
| Tracking paradigm | Box-based (SolidTrack) | Mask-based (SAM2 + Cross-Object Interaction) | Eliminates box overlap confusion; prevents memory corruption |
| Motion prediction | Kalman filter | MeMoSORT memory-augmented Kalman | Handles rapid direction changes (leaps, turns) |
| Re-ID | OSNet x0.25 (single global embedding) | 6-signal fusion (part + KPR + skeleton + face + color + spatial) | ~80% → 97%+ re-ID accuracy after occlusion |
| Re-ID domain | Market-1501 (pedestrians) | Fine-tuned on own dance clips | Closes domain gap for costumes + stage lighting |
| Gallery management | Passive accumulation | Pose-gated EMA + enrollment gallery | Prevents gallery pollution from contaminated frames |
| Collision resolution | Greedy sequential matching | Hungarian N×N + DP global optimization | Eliminates cascade errors in group splits |
| Gap filling | Forward-only | Forward + backward pass + stitch | Directly addresses re-entry failure mode |
| Occlusion handling | Binary (tracked/lost) | 4-state (active/partial/dormant/lost) | Zero hallucinated critique data; explicit uncertainty |
| Pose model | ViTPose+ Large (17 keypoints) | RTMW / ViTPose-Huge (133+ keypoints) | +15% keypoint AP; hands/feet/face available |
| Pose in occlusion | No mask guidance | SAM2 mask as auxiliary input | Eliminates chimeric skeletons from occluder bleeding |
| Keypoint reporting | Score only | HIGH/MEDIUM/LOW/NOT_VISIBLE per joint | Honest uncertainty; no feedback on invisible joints |
| 3D lifting | MotionAGFormer (per-person) | MotionBERT (multi-person, shared scene) | Physically consistent relative positions |
| Critique output | Angle deviation + timing | 5-dimension biomechanical analysis + audio sync | Actionable per-dancer feedback with confidence |
| Track preservation | Fixed max_age timeout | Sentinel SBM + MOTE disocclusion prediction | Tracks survive long occlusions; predicted re-emergence |

**Estimated DanceTrack-equivalent HOTA:** 76–80 (above SAM2MOT's 75.5–75.8, driven by multi-signal re-ID that no current system has). *Note: these are internal projections, not externally benchmarked results.*

**Estimated bigtest HOTA:** 0.65–0.70 (up from 0.495 — the backward pass + combinatorial re-ID + cross-object interaction directly target the three root causes). *Note: internal projection. Validate incrementally via Implementation Gates in Section 11.*

**Re-ID accuracy after occlusion:** 97%+ (the remaining 3% flagged as "low confidence re-ID" for optional human review in the critique report).

---

## 9. Research References

| System | Venue | DanceTrack HOTA | Key Relevance |
|--------|-------|----------------|---------------|
| **SAM2MOT** | arXiv 2025 (submitted) | 75.5–75.8 | Cross-Object Interaction module for memory quarantine |
| **MATR** | 2025 | 71.3 | Best-in-literature AssA (61.6) — predict-then-verify paradigm |
| **MeMoSORT** | 2025 | 67.9 | Drop-in TBD tracker; memory-augmented Kalman + motion-adaptive IoU |
| **MOTE** | ICML 2025 | 74.2 | Disocclusion matrix — predicts where hidden objects reappear |
| **MOTIP** | CVPR 2025 | 73.7 | Tracking as ID prediction via transformer |
| **Sentinel** | Scientific Reports (Nature portfolio), 2026 | — | Survival Boosting Mechanism for long-occlusion track preservation |
| **UMOT** | 2025 | — | Historical Backtracking Module for long-term dormancy recovery |
| **KPR** | Springer, 2025 | — | Keypoint-prompted re-ID for multi-person ambiguity resolution |
| **BPBreID** | — | — | Part-based re-ID with adversarial occlusion training |
| **PAFormer** | — | — | Pose-token-guided part-based re-ID with visibility prediction. *Not clearly linked to a canonical paper — treat as placeholder; BPBreID is the validated alternative.* |
| **MoCos** | AAAI 2025 (?) | — | Skeleton-based gait re-ID via motif-guided graph transformers. *Exact name not verified in major indexes — general approach is well-supported.* |
| **Co-DETR / Co-DINO** | — | — | NMS-free transformer detection (66+ AP COCO) |
| **RT-DETR** | — | — | Real-time DETR variant — faster NMS-free detection |
| **RTMW** | — | — | Whole-body pose (body + hands + feet + face) in one pass. *Name not verified — may refer to RTMPose-WholeBody or similar.* |
| **MotionBERT** | ICCV 2023 | — | Multi-person 3D lifting with DSTformer |

---

## 10. What This Design Intentionally Does NOT Do

1. **Does not track through full occlusion.** If all pixels are hidden, the track goes dormant. No hallucinated trajectory, no interpolated position, no estimated pose. The system says "I cannot see dancer 3 from frames 450–520" and produces no critique for that interval.

2. **Does not require real-time performance.** Optimized for accuracy over speed. A 2-minute dance video may take 15–30 minutes to fully process. If real-time preview is needed, the current YOLO + SolidTrack pipeline can provide live visualization while this pipeline runs asynchronously.

3. **Does not require multiple cameras.** Every component is designed for single monocular video.

4. **Does not generate comparative judgments.** Reports facts ("15° short"), not opinions ("worse than").

5. **Does not hallucinate.** If a joint is not visible, it is reported as not visible. If re-ID confidence is low, it is flagged. The user always knows what the system can and cannot see.

---

## 11. Lean Core vs. Experiment Add-Ons — Implementation Phases

The Perplexity fact-check (March 2026) correctly identified that this document presents many overlapping modules targeting the same failure modes. The pragmatic path is to converge on a **lean, evidence-backed core** first, then selectively add one advanced module at a time — measuring impact with critique-centric metrics, not only HOTA.

### The Lean Core (implement and validate FIRST)

These components form the minimum viable future pipeline. Every one has strong published evidence and directly addresses the root causes identified in Section 1. Nothing in the "experiment add-on" list should be started until the lean core passes basic bigtest regression.

| Layer | Core Component | Why It's Core |
|-------|---------------|---------------|
| **L0** | Closed-set enrollment gallery | Converts open-set → closed-set re-ID. Single highest-value change. |
| **L1** | Hybrid YOLO scout + Co-DETR/Co-DINO precision detector | NMS-free detection on overlap frames. Proven in SAM2MOT. |
| **L2** | SAM2MOT-style mask tracking + Cross-Object Interaction | Pixel-level identity. Prevents memory corruption. HOTA 75.5–75.8 on DanceTrack. |
| **L2** | MeMoSORT motion prediction (integrated with SAM2) | Memory-augmented Kalman for rapid direction changes. 67.9 HOTA on DanceTrack. |
| **L3** | Part-based Re-ID (BPBreID) + KPR | Part embeddings for partial visibility + keypoint prompts for multi-person crops. |
| **L3** | Color histogram + spatial formation prior | Cheapest signals; always available from SAM2 masks. |
| **L3** | Pose-gated EMA gallery management | Prevents gallery pollution — the primary cause of cascading ID errors. |
| **L4** | ACTIVE/PARTIAL/DORMANT/LOST state machine | No hallucinated critique. Honest uncertainty. |
| **L8** | ViTPose-Large with SAM2 mask guidance | Eliminates chimeric skeletons. Mask gate = locked. |
| **L9** | MotionBERT 3D lifting | Better 3D than MotionAGFormer; shared floor plane for relative positions. |

**What the lean core intentionally excludes at first:**
- Backward pass (Layer 5) — adds ~2x processing time; test after core stabilizes
- MOTE disocclusion (Layer 6) — optical flow cost; test on bigtest-specific failures
- UMOT historical backtracking (Layer 6) — complex integration; test only if dormancy recovery is still weak
- Sentinel SBM (Layer 6) — test only if entrance/exit cycling persists after enrollment + COI
- MATR (Layer 6) — full architecture replacement; parallel branch experiment, not an add-on
- Skeleton gait Re-ID / MoCos (Layer 3, Signal 3) — temporal window dependency; add after core Re-ID validates
- Face Re-ID / ArcFace (Layer 3, Signal 4) — opportunistic signal; add after core Re-ID validates
- DP collision solver (Layer 7) — Hungarian is sufficient for ≤5 tracks; DP is an optimization
- RTMW whole-body pose (Layer 8) — adds hand/foot/face keypoints; test after body-only critique works
- Multi-person joint 3D lifting (Layer 9) — shared floor plane; test after single-person MotionBERT works

### Implementation Phases with Sweep-Readiness Gates

Each phase ends with a **sweep gate** — a point where you can (and should) run an Optuna sweep or manual A/B experiments before proceeding. The gate ensures the foundation is solid before adding complexity.

| Impl Phase | What to Build | Sweep Gate | Metric to Beat |
|------------|--------------|------------|----------------|
| **I1: Foundation** | Enrollment gallery (L0) + SAM2MOT mask tracking with COI (L2) + MeMoSORT motion (L2) + state machine (L4). Keep existing ViTPose + MotionAGFormer for pose/3D. | **Gate 1:** Sweep `SWAY_SAM2_REINVOKE_STRIDE`, `SWAY_COI_MASK_IOU_THRESH`, `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA`, `SWAY_TRACK_MAX_AGE` against bigtest GT. | bigtest HOTA > 0.55 (up from 0.495 baseline). If not met, debug COI/enrollment before proceeding. |
| **I2: Re-ID Upgrade** | Replace OSNet with BPBreID (L3). Add KPR for multi-person crops. Add color + spatial signals. Implement pose-gated EMA. Group-split Hungarian re-ID. | **Gate 2:** Sweep `SWAY_REID_W_*` weights + `SWAY_REID_EMA_ALPHA_HIGH` + `SWAY_REID_EMA_ISOLATION_DIST`. | Re-ID accuracy after occlusion > 90%. bigtest HOTA > 0.58. |
| **I3: Detection Upgrade** | Integrate Co-DETR/Co-DINO as precision detector. Build hybrid YOLO-scout trigger logic. | **Gate 3:** A/B test `SWAY_DETECTOR_HYBRID=1` vs `0`. Sweep `SWAY_HYBRID_OVERLAP_IOU_TRIGGER`. | Detection AP on overlap frames improves. No regression on easy sequences. |
| **I4: Pose + 3D Upgrade** | SAM2 mask-guided ViTPose (or ViTPose-Huge). MotionBERT 3D lifting. Per-keypoint confidence classification. | **Gate 4:** Sweep `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH/MED`. A/B test MotionBERT vs MotionAGFormer. | Per-keypoint confidence accuracy > 90%. Critique gap reporting works. |
| **I5: Backward Pass** | Reverse-video second pass with SAM2+COI. Forward↔reverse track stitching. | **Gate 5:** A/B test `SWAY_BACKWARD_PASS_ENABLED=1` vs `0` on bigtest. Sweep stitch similarity. | bigtest HOTA > 0.62. Identity switches drop by ≥30%. |
| **I6: Selective Advanced Modules** | Test ONE module at a time: MOTE → Sentinel → UMOT. Each gets its own A/B test + conditional sweep. | **Gate 6:** Per-module A/B on bigtest. Only enable if it measurably reduces identity switches or improves critique accuracy. | Per-module: statistically significant improvement on bigtest identity switches. |
| **I7: Critique Layer** | Five-dimension biomechanical scoring. Audio beat alignment. Full critique report with confidence annotations. | **Gate 7:** Sweep `SWAY_CRITIQUE_JERK_WINDOW`, `SWAY_CRITIQUE_BEAT_TOLERANCE_MS`. | Critique output matches human evaluation on ≥3 reference clips. |

**Critical rule:** Do NOT skip gates. If Gate 1 fails, the issue is in the tracking foundation — adding Re-ID upgrades on a broken foundation compounds errors.

### Implementation Status (Updated 2026-03-31)

All 21 standalone plans have been implemented. Code is in `sway/` and `tools/`.

| Plan | Module | Status | Code Location |
|------|--------|--------|---------------|
| PLAN_01 | Track State Machine | **IMPLEMENTED** | `sway/track_state.py` |
| PLAN_02 | Co-DETR/RT-DETR Detection | **IMPLEMENTED** | `sway/detr_detector.py`, `sway/detector_factory.py` |
| PLAN_03 | Hybrid YOLO+DETR Dispatcher | **IMPLEMENTED** | `sway/hybrid_detector.py` |
| PLAN_04 | SAM2 Primary Tracker | **IMPLEMENTED** | `sway/sam2_tracker.py`, `sway/tracker_factory.py` |
| PLAN_05 | Cross-Object Interaction | **IMPLEMENTED** | `sway/cross_object_interaction.py` |
| PLAN_06 | MeMoSORT Motion Prediction | **IMPLEMENTED** | `sway/memosort.py` |
| PLAN_07 | Enrollment Gallery | **IMPLEMENTED** | `sway/enrollment.py` |
| PLAN_08 | BPBreID Part-Based Re-ID | **IMPLEMENTED** | `sway/bpbreid_extractor.py` |
| PLAN_09 | KPR Keypoint-Prompted Re-ID | **IMPLEMENTED** | `sway/kpr_extractor.py` |
| PLAN_10 | Skeleton Gait Re-ID (add-on) | **IMPLEMENTED** | `sway/mocos_extractor.py` |
| PLAN_11 | Face Recognition (add-on) | **IMPLEMENTED** | `sway/face_reid.py` |
| PLAN_12 | Color Histogram Re-ID | **IMPLEMENTED** | `sway/color_histogram_reid.py` |
| PLAN_13 | Re-ID Fusion Engine | **IMPLEMENTED** | `sway/reid_fusion.py`, `sway/reid_factory.py` |
| PLAN_14 | Pose-Gated EMA | **IMPLEMENTED** | `sway/pose_gated_ema.py` |
| PLAN_15 | Collision Solver | **IMPLEMENTED** | `sway/collision_solver.py` |
| PLAN_16 | Backward Pass | **IMPLEMENTED** | `sway/backward_pass.py` |
| PLAN_17 | Mask-Guided Pose + Confidence | **IMPLEMENTED** | `sway/mask_guided_pose.py` |
| PLAN_18 | MotionBERT 3D Lifting | **IMPLEMENTED** | `sway/motionbert_lifter.py` |
| PLAN_19 | Five-Dimension Critique | **IMPLEMENTED** | `sway/critique_engine.py` |
| PLAN_20 | Re-ID Fine-Tuning | **IMPLEMENTED** | `tools/finetune_reid.py` |
| PLAN_21 | Advanced Trackers (MOTE/Sentinel/UMOT) | **IMPLEMENTED** | `sway/mote_disocclusion.py`, `sway/sentinel_sbm.py`, `sway/umot_backtrack.py` |

**Next steps:** Wire modules into `main.py` orchestration loop, download model weights, run Gate 1 sweep on bigtest.

---

## 12. Experimentation Architecture — Sweep and Manual A/B Readiness

This section specifies the **code architecture** that makes the parameter tables in Section 6 actually work. Without this, the `SWAY_*` env vars are just names on paper.

### 12.1 Module Registry Pattern

The current `main.py` is import-and-call: every stage is hard-coded. To support `SWAY_TRACKER_ENGINE=sam2mot` vs `solidtrack` vs `sam2_memosort_hybrid` as a runtime choice (not a code edit), the pipeline needs a **factory dispatch** layer.

**Pattern: one factory function per swappable component.**

```python
# sway/tracker_factory.py
def create_tracker(engine: str, env: dict) -> BaseTracker:
    """Dispatch to the right tracker based on SWAY_TRACKER_ENGINE."""
    if engine == "solidtrack":
        return SolidTrackWrapper(env)
    elif engine == "sam2mot":
        return SAM2MOTWrapper(env)
    elif engine == "sam2_memosort_hybrid":
        return SAM2MeMoSORTHybrid(env)
    elif engine == "memosort":
        return MeMoSORTWrapper(env)
    raise ValueError(f"Unknown tracker engine: {engine}")

# sway/reid_factory.py
def create_reid_ensemble(env: dict) -> ReIDEnsemble:
    """Build the multi-signal re-ID ensemble from SWAY_REID_* env vars."""
    signals = []
    if env.get("SWAY_REID_PART_MODEL"):
        signals.append(create_part_reid(env["SWAY_REID_PART_MODEL"]))
    if env.get("SWAY_REID_KPR_ENABLED", "1") == "1":
        signals.append(KPRSignal(env))
    # ... each signal is independently togglable
    return ReIDEnsemble(signals, weights_from_env(env))

# sway/detector_factory.py
def create_detector(env: dict) -> BaseDetector:
    """Dispatch to YOLO, Co-DETR, RT-DETR, or hybrid."""
    primary = env.get("SWAY_DETECTOR_PRIMARY", "yolo26l_dancetrack")
    hybrid = env.get("SWAY_DETECTOR_HYBRID", "0") == "1"
    # ...
```

**Where `main.py` changes:** Replace hard-coded imports with factory calls. The orchestration order stays the same — factories only change **which implementation** runs at each stage, not the stage order.

**Rule:** Every `SWAY_*_ENABLED` bool and every `SWAY_*_MODEL` enum in Section 6 must map to a factory dispatch or a conditional code path. If a parameter exists in the config table but has no code path, it is dead config — and dead config is worse than no config.

### 12.2 Evolving `auto_sweep.py` for the Future Pipeline

The current `suggest_env_for_trial()` in `tools/auto_sweep.py` is one monolithic function that suggests ~30 parameters. The future pipeline adds ~50 more. This cannot be one function.

**Pattern: per-layer suggest functions, composed by sweep phase.**

```python
# tools/auto_sweep.py — future structure

def suggest_detection_params(trial, env):
    """Layer 1 params. Used in sweep phases S1, S3."""
    env["SWAY_YOLO_CONF"] = trial.suggest_float("yolo_conf", 0.08, 0.25)
    env["SWAY_PRETRACK_NMS_IOU"] = trial.suggest_float("nms_iou", 0.55, 0.75)
    env["SWAY_DETECT_SIZE"] = trial.suggest_categorical("detect_size", [640, 800, 960])
    if env.get("SWAY_DETECTOR_HYBRID") == "1":
        env["SWAY_HYBRID_OVERLAP_IOU_TRIGGER"] = trial.suggest_float("hybrid_trigger", 0.15, 0.50)

def suggest_tracking_params(trial, env):
    """Layer 2 params. Used in sweep phase S1."""
    env["SWAY_SAM2_REINVOKE_STRIDE"] = trial.suggest_int("sam2_stride", 10, 60)
    env["SWAY_COI_MASK_IOU_THRESH"] = trial.suggest_float("coi_thresh", 0.10, 0.50)
    env["SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA"] = trial.suggest_float("memo_alpha", 0.20, 0.80)
    env["SWAY_TRACK_MAX_AGE"] = trial.suggest_int("max_age", 90, 400)

def suggest_reid_params(trial, env):
    """Layer 3 params. Used in sweep phase S2."""
    env["SWAY_REID_W_PART"] = trial.suggest_float("w_part", 0.10, 0.50)
    env["SWAY_REID_W_KPR"] = trial.suggest_float("w_kpr", 0.05, 0.30)
    # ... all 6 weights + EMA params

def suggest_advanced_module_params(trial, env):
    """Layer 6 params. Used in sweep phase S5. One module at a time."""
    module = trial.suggest_categorical("advanced_module", ["none", "mote", "sentinel", "umot"])
    env["SWAY_MOTE_DISOCCLUSION"] = "1" if module == "mote" else "0"
    env["SWAY_SENTINEL_SBM"] = "1" if module == "sentinel" else "0"
    env["SWAY_UMOT_BACKTRACK"] = "1" if module == "umot" else "0"
    if module == "mote":
        env["SWAY_MOTE_CONFIDENCE_BOOST"] = trial.suggest_float("mote_boost", 0.05, 0.30)
    # ... conditional params per module

SWEEP_PHASE_COMPOSERS = {
    "S1": [suggest_detection_params, suggest_tracking_params],
    "S2": [suggest_reid_params],
    "S3": [suggest_state_machine_params, suggest_confidence_params],
    "S4": [suggest_backward_pass_params, suggest_collision_params],
    "S5": [suggest_advanced_module_params],
    "S6": [suggest_pose_params, suggest_critique_params],
}

def suggest_env_for_trial(trial, phase="S1"):
    """Compose the env for a trial by calling phase-appropriate suggest functions."""
    env = load_locked_winners_from_previous_phases()
    for fn in SWEEP_PHASE_COMPOSERS[phase]:
        fn(trial, env)
    return env
```

**Key property:** Each sweep phase locks the winners from previous phases and only searches the new parameters. This is already described in Section 6's "Recommended Sweep Phases" but the code architecture above is what makes it real.

### 12.3 Manual A/B Experiment Workflow

For "I want to try SAM2MOT + KPR + no backward pass on bigtest right now," the system needs named experiment recipes that are trivially composable.

**Mechanism: extend `sway/pipeline_matrix_presets.py` with future-pipeline recipes.**

Each recipe is a complete `SWAY_*` env dict. Recipes compose by merging dicts (later keys win). This is the existing pattern — the future pipeline just needs more recipes.

**Concrete recipe examples to add:**

```python
# sway/pipeline_matrix_presets.py — future additions

LEAN_CORE_BASE = {
    "SWAY_ENROLLMENT_ENABLED": "1",
    "SWAY_TRACKER_ENGINE": "sam2_memosort_hybrid",
    "SWAY_SAM2_MODEL": "sam2.1_b",
    "SWAY_COI_MASK_IOU_THRESH": "0.25",  # sweep winner from Gate 1
    "SWAY_REID_PART_MODEL": "bpbreid",
    "SWAY_REID_KPR_ENABLED": "1",
    "SWAY_REID_EMA_ALPHA_LOW": "0.00",
    "SWAY_BACKWARD_PASS_ENABLED": "0",
    "SWAY_MOTE_DISOCCLUSION": "0",
    "SWAY_SENTINEL_SBM": "0",
    "SWAY_UMOT_BACKTRACK": "0",
    # ... all other defaults from Section 6
}

EXPERIMENT_RECIPES = {
    "lean_core": LEAN_CORE_BASE,

    "lean_core_backward": {
        **LEAN_CORE_BASE,
        "SWAY_BACKWARD_PASS_ENABLED": "1",
        "SWAY_BACKWARD_STITCH_MIN_SIMILARITY": "0.60",
    },

    "lean_core_mote": {
        **LEAN_CORE_BASE,
        "SWAY_MOTE_DISOCCLUSION": "1",
        "SWAY_MOTE_FLOW_MODEL": "raft_small",
        "SWAY_MOTE_CONFIDENCE_BOOST": "0.15",
    },

    "lean_core_sentinel": {
        **LEAN_CORE_BASE,
        "SWAY_SENTINEL_SBM": "1",
        "SWAY_SENTINEL_GRACE_MULTIPLIER": "3.0",
    },

    "lean_core_codet_hybrid": {
        **LEAN_CORE_BASE,
        "SWAY_DETECTOR_HYBRID": "1",
        "SWAY_DETECTOR_PRECISION": "co_dino",
        "SWAY_HYBRID_OVERLAP_IOU_TRIGGER": "0.30",
    },

    "lean_core_full_reid": {
        **LEAN_CORE_BASE,
        "SWAY_REID_SKEL_MODEL": "mocos",
        "SWAY_REID_FACE_MODEL": "arcface",
        "SWAY_REID_W_SKELETON": "0.15",
        "SWAY_REID_W_FACE": "0.15",
    },

    "matr_branch": {
        "SWAY_TRACKER_ENGINE": "matr",
        # ... completely different tracker, same Re-ID gallery
    },
}
```

**CLI usage (extends existing `tools/pipeline_matrix_runs.py`):**

```bash
# Run a single named recipe on bigtest
python -m tools.pipeline_matrix_runs \
  --recipe lean_core_backward \
  --video data/ground_truth/bigtest/BigTest.mov \
  --stop-after-boundary after_phase_3

# Run all lean_core_* recipes as a matrix (automatic comparison)
python -m tools.pipeline_matrix_runs \
  --recipe-prefix lean_core_ \
  --video data/ground_truth/bigtest/BigTest.mov

# Compare two specific recipes side-by-side
python -m tools.pipeline_matrix_runs \
  --recipes lean_core,lean_core_mote \
  --video data/ground_truth/bigtest/BigTest.mov \
  --compare
```

### 12.4 Critique-Centric Metrics (Beyond HOTA)

Per Perplexity's recommendation, sweeps should optimize for critique quality, not just generic MOT scores. The sweep objective function in `auto_sweep.py` needs additional metrics:

| Metric | What It Measures | How to Compute | When to Use |
|--------|-----------------|----------------|-------------|
| **HOTA** (existing) | Overall tracking quality | TrackEval vs GT | All sweep phases — baseline metric |
| **IDF1** (existing) | Identity preservation | TrackEval vs GT | Sweep phases S2, S4 — identity-focused |
| **Re-ID accuracy after occlusion** | % correct ID on re-emergence | Count re-emergence events in GT, check if predicted ID matches | Sweep phases S2, S4, S5 — the metric that matters most for critique |
| **Identity switches per 1000 frames** | Frequency of ID errors | Count track ID changes not present in GT | Sweep phases S1–S5 |
| **Per-keypoint confidence calibration** | When we say HIGH, is error actually ≤5px? | Compare confidence labels to GT keypoint positions | Sweep phases S3, S6 |
| **Incorrect-critique rate** | Critiques given on wrong-ID frames | Cross-reference critique output with GT identities | Sweep phases S2, S4 — the metric the user actually cares about |
| **Missing-critique rate** | Critiques omitted due to conservative visibility | Count ACTIVE frames in GT that pipeline classified as DORMANT | Sweep phase S3 — trade-off against incorrect-critique |

**Sweep objective function for future phases:**

```python
def future_sweep_objective(trial_results):
    """Weighted combination — critique quality > raw tracking."""
    hota = trial_results["hota"]
    idf1 = trial_results["idf1"]
    reid_acc = trial_results["reid_accuracy_after_occlusion"]
    incorrect_critique = trial_results["incorrect_critique_rate"]

    # Penalize incorrect critiques heavily — this is the north star
    return (0.3 * hota + 0.2 * idf1 + 0.4 * reid_acc - 0.5 * incorrect_critique)
```

### 12.5 Experiment Tracking and Comparison

Every experiment (sweep trial or manual recipe run) produces:

1. **Structured result JSON** — HOTA, IDF1, per-sequence scores, per-metric breakdowns.
2. **Phase preview videos** — visual comparison of tracking quality.
3. **Config snapshot** — the exact `SWAY_*` env dict that produced this result.

The Pipeline Lab already serves sweep status via `/api/optuna-sweep/`. Extend it with:

- **Recipe comparison view:** Select 2–4 recipe runs, see metrics side-by-side + video previews.
- **Parameter importance:** Optuna's built-in `plot_param_importances()` — show which parameters actually move the needle at each sweep phase.
- **Gate status dashboard:** For each implementation phase (I1–I7), show whether the gate metric has been met and which sweep/recipe achieved it.

### 12.6 Configuration Composition Rules

To prevent config conflicts when composing recipes:

1. **Locked params always win.** `MASTER_LOCKED_*` dicts are reapplied after recipe merge. No recipe can override a locked param without the corresponding `SWAY_UNLOCK_*` gate.
2. **Later layers inherit earlier winners.** When running sweep phase S2 (Re-ID), detection and tracking params are loaded from the best S1 trial — not re-searched.
3. **Advanced modules are mutually exclusive by default.** `SWAY_MOTE_DISOCCLUSION`, `SWAY_SENTINEL_SBM`, and `SWAY_UMOT_BACKTRACK` — at most one ON at a time in sweep phase S5. After individual impact is measured, a follow-up sweep can test pairwise combinations.
4. **Every recipe is self-contained.** A recipe dict contains ALL parameters needed to reproduce the run. No implicit dependencies on "whatever was in the env when you ran it." The `verify_sweep_env_wiring.py` tool should be extended to validate recipe completeness.

---

## 13. Complete Preset & Configuration Catalog

This section lists **every named preset and configuration** the future pipeline supports. Each is a self-contained `SWAY_*` env dict that can be run via the Pipeline Lab, `tools/pipeline_matrix_runs.py`, or a YAML `--params` file. No implicit dependencies — copy the dict and the run is fully reproducible.

All presets below are **wired** in `sway/pipeline_matrix_presets.py` (v14), with corresponding schema fields in `sway/pipeline_config_schema.py` and runtime toggles in `main.py`. CLI selectors `--recipe`, `--recipes`, `--recipe-prefix`, `--compare`, and `--stop-after-boundary` are implemented in `tools/pipeline_matrix_runs.py`.

Presets follow the existing `pipeline_matrix_presets.py` pattern: a `MatrixRecipe` with an `id`, a `recipe_name`, a `varies` tag, and a `fields` dict. The future pipeline extends the matrix from the current M01–M23 to F01–F75.

### 13.1 Baseline Carry-Forward (from current pipeline)

These existing presets remain available unchanged. They serve as **regression baselines** — every future configuration must not regress on the sequences where these presets already work.

| ID | Recipe Name | What It Tests |
|----|------------|---------------|
| `baseline` | M01_baseline_locked_stack | Current production defaults (Deep OC-SORT, ViTPose-Base, all master-locks) |
| `preset_open_competition` | M18_preset_open_floor_competition | Open Floor + Competition Grade + Sharp Hip-Hop |
| `preset_open_competition_recovery` | M23_preset_open_floor_competition_recovery | Recovery bias variant — best current config for bigtest |
| `preset_dense_hifi` | M17_preset_dense_crowd_high_fidelity | Dense Crowd + High Fidelity + Balanced Cleanup |
| `preset_ballet_fluid` | M19_preset_standard_ballet | Standard + Balanced + Fluid Ballet |
| `preset_mirror_studio` | M20_preset_mirror_studio | Mirror Studio pruning weights |
| `preset_wide_angle_maxprec` | M21_preset_wide_angle_max_precision | Wide Angle + ViTPose-Huge |
| `preset_osnet_aggressive` | M22_preset_osnet_lock_aggressive_clean | OSNet Identity Lock + Aggressive Clean |

### 13.2 Lean Core Presets (I1–I4 foundation)

These are the primary configurations for the new pipeline. Each builds on the previous, accumulating validated components.

| ID | Recipe Name | What It Configures | When Available |
|----|------------|-------------------|----------------|
| `f01_lean_core_i1` | F01_lean_core_foundation | Enrollment ON + SAM2MOT (`sam2_memosort_hybrid`) + COI + state machine. Existing ViTPose-Large + MotionAGFormer for pose/3D. All advanced modules OFF. | After I1 implementation |
| `f02_lean_core_i2` | F02_lean_core_reid_upgrade | I1 + BPBreID part-based Re-ID + KPR + color histogram + spatial prior + pose-gated EMA + Hungarian collision solver. Skeleton/face signals OFF. | After I2 implementation |
| `f03_lean_core_i3` | F03_lean_core_hybrid_detection | I2 + Hybrid YOLO scout + Co-DINO precision detector on overlap frames. | After I3 implementation |
| `f04_lean_core_i4` | F04_lean_core_pose_upgrade | I3 + Mask-guided ViTPose-Large + per-keypoint confidence + MotionBERT 3D lifting. | After I4 implementation |
| `f05_lean_core_full` | F05_lean_core_complete | The complete lean core: I1–I4 with all Gate 1–4 winners locked. This is the **new production baseline**. | After all gates pass |

**Full `fields` dict for `f05_lean_core_full` (the target production config):**

```yaml
# F05_lean_core_complete — full production config
SWAY_ENROLLMENT_ENABLED: "1"
SWAY_ENROLLMENT_PART_MODEL: bpbreid
SWAY_ENROLLMENT_COLOR_BINS: "32"

SWAY_DETECTOR_PRIMARY: yolo26l_dancetrack
SWAY_DETECTOR_HYBRID: "1"
SWAY_DETECTOR_PRECISION: co_dino
SWAY_HYBRID_OVERLAP_IOU_TRIGGER: "0.30"     # sweep S1 winner
SWAY_YOLO_CONF: "0.16"                      # sweep S1 winner
SWAY_PRETRACK_NMS_IOU: "0.65"               # sweep S1 winner
SWAY_DETECT_SIZE: "800"

SWAY_TRACKER_ENGINE: sam2_memosort_hybrid
SWAY_SAM2_MODEL: sam2.1_b
SWAY_SAM2_REINVOKE_STRIDE: "30"             # sweep S1 winner
SWAY_SAM2_CONFIDENCE_REINVOKE: "0.40"
SWAY_COI_MASK_IOU_THRESH: "0.25"            # sweep S1 winner
SWAY_COI_QUARANTINE_MODE: delete
SWAY_MEMOSORT_MEMORY_LENGTH: "30"
SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA: "0.50"    # sweep S1 winner
SWAY_TRACK_MAX_AGE: "200"                   # sweep S1 winner

SWAY_REID_PART_MODEL: bpbreid
SWAY_REID_KPR_ENABLED: "1"
SWAY_REID_W_PART: "0.30"                    # sweep S2 winner
SWAY_REID_W_KPR: "0.15"                     # sweep S2 winner
SWAY_REID_W_SKELETON: "0.00"                # OFF — not lean core
SWAY_REID_W_FACE: "0.00"                    # OFF — not lean core
SWAY_REID_W_COLOR: "0.10"                   # sweep S2 winner
SWAY_REID_W_SPATIAL: "0.05"                 # sweep S2 winner
SWAY_REID_EMA_ALPHA_HIGH: "0.15"            # sweep S2 winner
SWAY_REID_EMA_ALPHA_LOW: "0.00"             # LOCKED
SWAY_REID_EMA_ISOLATION_DIST: "1.5"         # sweep S2 winner
SWAY_REID_COLOR_SPACE: hsv

SWAY_COLLISION_SOLVER: hungarian
SWAY_COLLISION_MIN_TRACKS: "3"
SWAY_COALESCENCE_IOU_THRESH: "0.85"
SWAY_COALESCENCE_CONSECUTIVE_FRAMES: "8"

SWAY_STATE_PARTIAL_MASK_FRAC: "0.30"        # sweep S3 winner
SWAY_STATE_DORMANT_MASK_FRAC: "0.05"
SWAY_STATE_DORMANT_MAX_FRAMES: "300"        # sweep S3 winner
SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH: "0.70" # sweep S3 winner
SWAY_CONFIDENCE_HEATMAP_THRESH_MED: "0.40"  # sweep S3 winner
SWAY_CONFIDENCE_MASK_GATE: "1"              # LOCKED

SWAY_POSE_MODEL: vitpose_large
SWAY_POSE_MASK_GUIDED: "1"
SWAY_POSE_KEYPOINT_SET: coco17
SWAY_POSE_SMART_PAD: "1"                    # LOCKED

SWAY_LIFT_BACKEND: motionbert
SWAY_LIFT_MULTI_PERSON: "0"

SWAY_BACKWARD_PASS_ENABLED: "0"             # OFF in lean core
SWAY_MOTE_DISOCCLUSION: "0"
SWAY_SENTINEL_SBM: "0"
SWAY_UMOT_BACKTRACK: "0"

SWAY_CRITIQUE_DIMENSIONS: formation,timing,extension,smoothness,sync
SWAY_CRITIQUE_MIN_CONFIDENCE: MEDIUM        # LOCKED
SWAY_CRITIQUE_REPORT_GAPS: "1"              # LOCKED
```

### 13.3 Detector A/B Presets

Each isolates one detector variable against the lean core. Use these to decide which detector to lock.

| ID | Recipe Name | Varies | Description |
|----|------------|--------|-------------|
| `f10_det_yolo_only` | F10_detector_yolo_only | detection | YOLO-only, no hybrid dispatch. `SWAY_DETECTOR_HYBRID=0`. Baseline speed reference. |
| `f11_det_codetr_only` | F11_detector_codetr_only | detection | Co-DETR on every frame. `SWAY_DETECTOR_PRIMARY=co_detr`, `SWAY_DETECTOR_HYBRID=0`. Maximum detection AP, highest cost. |
| `f12_det_codino_only` | F12_detector_codino_only | detection | Co-DINO on every frame. Same as F11 but Co-DINO variant. |
| `f13_det_rtdetr_only` | F13_detector_rtdetr_only | detection | RT-DETR on every frame. Faster than Co-DETR, still NMS-free. |
| `f14_det_hybrid_codino` | F14_detector_hybrid_yolo_codino | detection | **Default lean core:** YOLO scout + Co-DINO on overlap frames. |
| `f15_det_hybrid_rtdetr` | F15_detector_hybrid_yolo_rtdetr | detection | YOLO scout + RT-DETR on overlap frames. Faster hybrid variant. |
| `f16a_det_hybrid_overlap_lo` | F16a_detector_hybrid_overlap_015 | detection | Hybrid with `SWAY_HYBRID_OVERLAP_IOU_TRIGGER=0.15` (DETR on more frames). Schema: `sway_hybrid_overlap_iou_trigger`. |
| `f16_det_hybrid_sweep` | F16_detector_hybrid_trigger_sweep | detection | Hybrid with `SWAY_HYBRID_OVERLAP_IOU_TRIGGER=0.30` (mid / default sweep point). |
| `f16c_det_hybrid_overlap_hi` | F16c_detector_hybrid_overlap_050 | detection | Hybrid with `SWAY_HYBRID_OVERLAP_IOU_TRIGGER=0.50` (DETR only on heavy overlap). |
| `f17_det_yolo_crowdhuman` | F17_detector_yolo_crowdhuman | detection | YOLO with DanceTrack+CrowdHuman weights. Tests fine-tuned YOLO as a stronger scout. |

### 13.4 Tracker Engine A/B Presets

Each swaps the tracking paradigm. All other components stay at lean core defaults.

| ID | Recipe Name | Varies | Description |
|----|------------|--------|-------------|
| `f20_trk_solidtrack` | F20_tracker_solidtrack | tracking | Current production tracker (box-based SolidTrack via BoxMOT). Regression baseline. |
| `f21_trk_sam2mot` | F21_tracker_sam2mot_only | tracking | SAM2MOT mask tracking without MeMoSORT motion. Tests mask-only tracking. |
| `f22_trk_memosort` | F22_tracker_memosort_only | tracking | MeMoSORT only (box-based, no masks). Tests motion prediction without masks. |
| `f23_trk_sam2_memosort` | F23_tracker_sam2_memosort_hybrid | tracking | **Default lean core:** SAM2 masks + MeMoSORT motion. The recommended hybrid. |
| `f24_trk_sam2_b` | F24_tracker_sam2_model_base | tracking | SAM2MOT with `sam2.1_b` checkpoint (default). |
| `f25_trk_sam2_l` | F25_tracker_sam2_model_large | tracking | SAM2MOT with `sam2.1_l` checkpoint. Better masks, slower. |
| `f26_trk_sam2_h` | F26_tracker_sam2_model_huge | tracking | SAM2MOT with `sam2.1_h` checkpoint. Best masks, slowest. |
| `f27_trk_coi_freeze` | F27_tracker_coi_freeze_mode | tracking | COI quarantine mode = `freeze` instead of `delete`. Less aggressive memory management. |

### 13.5 Re-ID Signal A/B Presets

Each tests a different combination of re-ID signals. Lean core base is always enrolled.

| ID | Recipe Name | Varies | Description |
|----|------------|--------|-------------|
| `f30_reid_osnet_baseline` | F30_reid_osnet_global_only | reid | OSNet x0.25 global embedding only (no parts). Current production Re-ID. Regression baseline. |
| `f31_reid_bpbreid_only` | F31_reid_bpbreid_parts_only | reid | BPBreID parts + color + spatial. No KPR, no face, no gait. Tests part-based improvement over OSNet. |
| `f32_reid_lean_core` | F32_reid_lean_core_4signal | reid | **Default lean core:** BPBreID + KPR + color + spatial. The 4-signal ensemble. |
| `f33_reid_plus_face` | F33_reid_lean_plus_face | reid | Lean core + ArcFace face signal. Tests whether face recognition adds value. |
| `f34_reid_plus_skeleton` | F34_reid_lean_plus_skeleton | reid | Lean core + MoCos skeleton gait signal. Tests whether gait adds value. |
| `f35_reid_full_6signal` | F35_reid_full_ensemble_6signal | reid | All 6 signals ON: part + KPR + skeleton + face + color + spatial. Maximum Re-ID power. |
| `f36_reid_finetuned` | F36_reid_bpbreid_finetuned | reid | BPBreID with contrastive fine-tuned weights (PLAN_20). Tests domain adaptation. |
| `f37_reid_finetuned_full` | F37_reid_finetuned_full_ensemble | reid | Fine-tuned BPBreID + all 6 signals. The theoretical maximum Re-ID config. |
| `f38_reid_weight_sweep_a` | F38_reid_appearance_heavy | reid | W_PART=0.45, W_KPR=0.20, W_COLOR=0.05, W_SPATIAL=0.05. Appearance-dominated weighting. |
| `f39_reid_weight_sweep_b` | F39_reid_balanced_weights | reid | Equal weights across all active signals. Tests whether learned weighting matters. |

### 13.6 Pose & 3D Lifting A/B Presets

| ID | Recipe Name | Varies | Description |
|----|------------|--------|-------------|
| `f40_pose_vitpose_large` | F40_pose_vitpose_large_masked | pose | **Default lean core:** ViTPose-Large with SAM2 mask guidance. |
| `f41_pose_vitpose_huge` | F41_pose_vitpose_huge_masked | pose | ViTPose-Huge with mask guidance. Higher accuracy, higher cost. |
| `f42_pose_vitpose_nomask` | F42_pose_vitpose_large_nomask | pose | ViTPose-Large WITHOUT mask guidance. Tests mask guidance value. |
| `f43_pose_rtmw_l` | F43_pose_rtmw_l_wholebody | pose | RTMW-L whole-body (133 keypoints). Tests hand/foot/face keypoints for critique. |
| `f44_pose_rtmw_x` | F44_pose_rtmw_x_wholebody | pose | RTMW-X whole-body. Larger, more accurate. |
| `f45_lift_motionagformer` | F45_lift_motionagformer | 3d_lift | Current production 3D lifter. Regression baseline. |
| `f46_lift_motionbert` | F46_lift_motionbert_single | 3d_lift | **Default lean core:** MotionBERT per-person 3D lifting. |
| `f47_lift_motionbert_multi` | F47_lift_motionbert_multi_person | 3d_lift | MotionBERT + multi-person shared floor plane + depth estimation. |

### 13.7 Backward Pass & Collision Presets

| ID | Recipe Name | Varies | Description |
|----|------------|--------|-------------|
| `f50_backward_off` | F50_backward_pass_disabled | backward | No backward pass. Lean core default. |
| `f51_backward_on` | F51_backward_pass_enabled | backward | Backward pass ON with default stitch similarity 0.60. ~2x processing time. |
| `f52_backward_tight` | F52_backward_stitch_tight | backward | Backward pass with tight stitch similarity (0.75). Fewer but more confident stitches. |
| `f53_backward_loose` | F53_backward_stitch_loose | backward | Backward pass with loose stitch similarity (0.45). More stitches, higher risk of wrong merges. |
| `f54_collision_greedy` | F54_collision_greedy_baseline | collision | Greedy sequential matching (current behavior). Regression baseline. |
| `f55_collision_hungarian` | F55_collision_hungarian | collision | **Default lean core:** Hungarian N×N. |
| `f56_collision_dp` | F56_collision_dp_full | collision | DP solver for ≤5 tracks + Hungarian fallback for larger clusters. |

### 13.8 Advanced Module Experiment Presets (I6)

Each enables exactly ONE advanced module on top of the lean core. For comparison, always run `f05_lean_core_full` alongside these.

| ID | Recipe Name | Varies | Description |
|----|------------|--------|-------------|
| `f60_mote` | F60_advanced_mote_disocclusion | advanced | MOTE optical flow disocclusion matrix ON. RAFT-small. Predicts re-emergence location after crossovers. |
| `f61_mote_raft_large` | F61_advanced_mote_raft_large | advanced | MOTE with RAFT-large (higher quality flow, slower). |
| `f62_sentinel` | F62_advanced_sentinel_sbm | advanced | Sentinel Survival Boosting ON. Grace multiplier 3.0, weak det conf 0.08. |
| `f63_sentinel_aggressive` | F63_advanced_sentinel_aggressive | advanced | Sentinel with grace multiplier 5.0 and weak det conf 0.05. More aggressive track preservation. |
| `f64_umot` | F64_advanced_umot_backtrack | advanced | UMOT Historical Backtracking ON. History length 500 frames. |
| `f65_umot_long_history` | F65_advanced_umot_long_history | advanced | UMOT with history length 1000 frames. For very long dormancy recovery. |
| `f66_mote_sentinel` | F66_advanced_mote_plus_sentinel | advanced | MOTE + Sentinel together. Only run after F60 and F62 show individual benefit. |
| `f67_matr_branch` | F67_advanced_matr_full_replace | advanced | MATR as tracker engine (full architecture replacement). Separate experiment branch — not composable with SAM2MOT. |

### 13.9 Full-Stack Experiment Presets (I5+)

These combine backward pass + advanced modules + full Re-ID on the lean core. Use for final production config exploration.

| ID | Recipe Name | Description |
|----|------------|-------------|
| `f70_production_candidate_a` | F70_prod_lean_backward | Lean core + backward pass. Conservative — no advanced modules. |
| `f71_production_candidate_b` | F71_prod_lean_backward_mote | Lean core + backward pass + MOTE. Tests the full occlusion-recovery stack. |
| `f72_production_candidate_c` | F72_prod_lean_backward_full_reid | Lean core + backward pass + all 6 Re-ID signals. Maximum identity accuracy. |
| `f73_production_candidate_d` | F73_prod_full_stack | Lean core + backward pass + best advanced module (winner from I6) + full Re-ID + fine-tuned weights. The theoretical maximum config. |
| `f74_production_speed` | F74_prod_lean_speed_optimized | Lean core with RT-DETR (faster) + SAM2 base + ViTPose-Large + no backward pass. Best accuracy-per-minute config. |
| `f75_production_quality` | F75_prod_lean_quality_max | Lean core with Co-DINO + SAM2 huge + ViTPose-Huge + backward pass + MOTE. Maximum quality regardless of cost. |

### 13.10 Sweep-Phase Lock Presets

These are NOT for manual A/B testing — they are the **locked base configs** loaded by `auto_sweep.py` at the start of each sweep phase. They carry forward all winners from previous gates.

| Sweep Phase | Base Config | What's Locked | What's Searched |
|-------------|------------|---------------|-----------------|
| **S1** | `f01_lean_core_i1` | Detection + tracker engine choice | `SWAY_YOLO_CONF`, `SWAY_PRETRACK_NMS_IOU`, `SWAY_DETECT_SIZE`, `SWAY_SAM2_REINVOKE_STRIDE`, `SWAY_COI_MASK_IOU_THRESH`, `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA`, `SWAY_TRACK_MAX_AGE` |
| **S2** | S1 winners + `f02_lean_core_i2` | All S1 params | `SWAY_REID_W_PART`, `W_KPR`, `W_COLOR`, `W_SPATIAL`, `SWAY_REID_EMA_ALPHA_HIGH`, `SWAY_REID_EMA_ISOLATION_DIST` |
| **S3** | S1+S2 winners + state machine | All S1+S2 params | `SWAY_STATE_PARTIAL_MASK_FRAC`, `SWAY_STATE_DORMANT_MAX_FRAMES`, `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH`, `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` |
| **S4** | S1+S2+S3 winners + backward pass | All S1–S3 params | `SWAY_BACKWARD_STITCH_MIN_SIMILARITY`, `SWAY_COLLISION_MIN_TRACKS`, `SWAY_COALESCENCE_IOU_THRESH` |
| **S5** | S1–S4 winners | All S1–S4 params | `advanced_module` (categorical: none/mote/sentinel/umot) + module-specific params |
| **S6** | S1–S5 winners | All S1–S5 params | `SWAY_POSE_MODEL` (categorical), `SWAY_POSE_VISIBILITY_THRESHOLD`, `SWAY_CRITIQUE_JERK_WINDOW`, `SWAY_CRITIQUE_BEAT_TOLERANCE_MS` |

### 13.11 Quick Reference: CLI for Every Preset Category

```bash
# --- Run a single preset ---
python -m tools.pipeline_matrix_runs \
  --recipe f05_lean_core_full \
  --video data/ground_truth/bigtest/BigTest.mov

# --- Compare detector options ---
python -m tools.pipeline_matrix_runs \
  --recipes f10_det_yolo_only,f14_det_hybrid_codino,f15_det_hybrid_rtdetr \
  --video data/ground_truth/bigtest/BigTest.mov \
  --stop-after-boundary after_phase_3 --compare

# --- §13.3 hybrid overlap trigger sweep (0.15 / 0.30 / 0.50) ---
python -m tools.pipeline_matrix_runs \
  --recipes f16a_det_hybrid_overlap_lo,f16_det_hybrid_sweep,f16c_det_hybrid_overlap_hi \
  --video data/ground_truth/bigtest/BigTest.mov \
  --stop-after-boundary after_phase_3 --compare

# --- Compare tracker engines ---
python -m tools.pipeline_matrix_runs \
  --recipes f20_trk_solidtrack,f21_trk_sam2mot,f23_trk_sam2_memosort \
  --video data/ground_truth/bigtest/BigTest.mov \
  --stop-after-boundary after_phase_3 --compare

# --- Compare Re-ID configurations ---
python -m tools.pipeline_matrix_runs \
  --recipes f30_reid_osnet_baseline,f32_reid_lean_core,f35_reid_full_6signal \
  --video data/ground_truth/bigtest/BigTest.mov --compare

# --- Test advanced modules one at a time ---
python -m tools.pipeline_matrix_runs \
  --recipes f05_lean_core_full,f60_mote,f62_sentinel,f64_umot \
  --video data/ground_truth/bigtest/BigTest.mov --compare

# --- Run ALL lean core progression (I1→I4) ---
python -m tools.pipeline_matrix_runs \
  --recipes f01_lean_core_i1,f02_lean_core_i2,f03_lean_core_i3,f04_lean_core_i4 \
  --video data/ground_truth/bigtest/BigTest.mov --compare

# --- Run production candidates head-to-head ---
python -m tools.pipeline_matrix_runs \
  --recipes f70_production_candidate_a,f71_production_candidate_b,f73_production_candidate_d \
  --video data/ground_truth/bigtest/BigTest.mov --compare

# --- Run current vs future baseline ---
python -m tools.pipeline_matrix_runs \
  --recipes preset_open_competition_recovery,f05_lean_core_full \
  --video data/ground_truth/bigtest/BigTest.mov --compare

# --- Full matrix: run ALL presets in a category ---
python -m tools.pipeline_matrix_runs \
  --recipe-prefix f3  \
  --video data/ground_truth/bigtest/BigTest.mov
  # Runs f30, f31, f32, f33, f34, f35, f36, f37, f38, f39 (all Re-ID variants)
```

### 13.12 Preset Naming Convention

| Prefix | Meaning |
|--------|---------|
| `M01–M23` | Current production presets (carry forward unchanged) |
| `f01–f09` | Lean core progression presets (I1→I4, one per implementation phase) |
| `f10–f19` | Detector A/B presets |
| `f20–f29` | Tracker engine A/B presets |
| `f30–f39` | Re-ID signal A/B presets |
| `f40–f49` | Pose + 3D lifting A/B presets |
| `f50–f59` | Backward pass + collision solver presets |
| `f60–f69` | Advanced module experiment presets (MOTE, Sentinel, UMOT, MATR) |
| `f70–f79` | Full-stack production candidate presets |

**Total wired presets:** 92 (22 carry-forward M01–M23 + 70 future-style rows, including `f16a`/`f16c` overlap sweep endpoints). All preset IDs resolve from `matrix_recipe_by_id()` and are selectable via CLI.

Each preset is a one-line command to run. Each comparison is a one-line command with `--compare`. No code changes needed to test any combination — just pick the preset IDs.

---

## 14. Supported vs Optional Models

All documented model variants are wired at runtime. Models marked **Supported** work
out of the box with fallback logic. Models marked **Optional** require manually placed
weights — the pipeline falls back gracefully with a clear log message when they are absent.

| Model / Feature | Env Key | Status | Weights Path / Install | Fallback When Absent |
|-----------------|---------|--------|------------------------|----------------------|
| BPBreID (ResNet-50 base) | `SWAY_REID_PART_MODEL=bpbreid` | **Supported** | Auto via `torchreid` / `torchvision` | ResNet-50 backbone features |
| BPBreID (fine-tuned) | `SWAY_REID_PART_MODEL=bpbreid_finetuned` | Optional | `models/bpbreid_r50_sway_finetuned.pth` — generate with `python -m tools.finetune_reid` | Falls back to base BPBreID |
| PAFormer | `SWAY_ENROLLMENT_PART_MODEL=paformer` | Optional | `models/paformer.pth` — place manually | Falls back to BPBreID |
| MoCos Skeleton Gait | `SWAY_REID_SKEL_MODEL=mocos` | Optional | `models/mocos_gait.pth` — place manually | Handcrafted gait features |
| ArcFace / AdaFace | `SWAY_REID_FACE_MODEL=arcface` | Optional | Via `pip install insightface onnxruntime-gpu` (auto-downloads `buffalo_l`) | Face re-ID disabled with clear log |
| KPR Extractor | `SWAY_REID_KPR_ENABLED=1` | **Supported** | Auto via `torchreid` / `torchvision` | KPR signal disabled |
| Color Histogram | always on | **Supported** | No weights needed | — |
| ViTPose-Large | `SWAY_POSE_MODEL=vitpose_large` | **Supported** | Auto via `tools/prefetch_models.py` | — |
| ViTPose-Huge | `SWAY_POSE_MODEL=vitpose_huge` | **Supported** | Auto via `tools/prefetch_models.py` | — |
| RTMW-L/X | `SWAY_POSE_MODEL=rtmw_l` | Optional | Requires MMPose + checkpoint | Falls back to ViTPose |
| SAM2 (base/large/huge) | `SWAY_SAM2_MODEL` | **Supported** | Auto via `tools/prefetch_models.py` | — |
| MotionAGFormer | `SWAY_LIFT_BACKEND=motionagformer` | **Supported** | Auto via `tools/prefetch_models.py` | — |
| MotionBERT | `SWAY_LIFT_BACKEND=motionbert` | **Supported** | Auto via `tools/prefetch_models.py` | — |

### Automated validation

The consolidated suite `tests/test_MASTER_suite.py` includes doc–config parity tests (`TestDocCodeParity` and related classes) which enforce:
- Every `SWAY_*` key in this document and `MASTER_PIPELINE_GUIDELINE.md` appears in at least one `.py` file.
- All newly wired future keys are in `pipeline_config_schema.py`.
- Optional model factories fall back gracefully (no unhandled exceptions).
- `tools/verify_sweep_env_wiring.py` reports 0 sweep-only keys.

Run `python -m pytest tests/test_MASTER_suite.py -k "DocCodeParity or SchemaContainsFutureKeys or RuntimeEnvResolution or ModelFactoryGracefulFallback or SweepWiring" -v` to verify.

---

*Based on sweep_v3 results (Trial #173 baseline), SOTA research survey as of March 2026, Perplexity fact-check (March 2026), and logical hybrid analysis of technology combinations. For the current production pipeline, see `MASTER_PIPELINE_GUIDELINE.md`. For sweep raw data, see `output/sweeps/optuna/sweep.db`.*
