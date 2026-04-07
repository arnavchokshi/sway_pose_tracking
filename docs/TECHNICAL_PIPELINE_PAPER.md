# Sway Pose — Technical Pipeline Paper

**A Full Explanation of Every Stage, Every Technology Choice, and Every Tuning Decision**

*Current best configuration: Trial #173, sweep_v3 — Harmonic Mean HOTA 0.6597*

---

## Preface

This document explains our video analysis pipeline from first principles. It is written for two audiences simultaneously: (1) engineers who need to understand every implementation detail, and (2) readers who may be encountering these computer vision technologies for the first time. For the latter group, each major technology has a "What is this?" section before we dive into how we use it, why we configured it the way we did, and what happened when we tried alternatives.

The pipeline takes a single video of dancers and produces, for every frame of that video, the precise location of every dancer's body, every joint on their skeleton, a stable identity label across all frames, a three-dimensional reconstruction of their movement, and a motion quality score. Achieving all of that reliably on dense group footage — where people overlap, pass behind each other, and re-enter frame — required building a deeply layered system and spending approximately 20 GPU-hours of automated search across 347 distinct parameter combinations to find the configuration described here.

---

## Table of Contents

1. [System Overview — What the Pipeline Does](#1-system-overview)
2. [Phase 1–2: Person Detection with YOLO](#2-person-detection-yolo)
3. [Phase 1–2: Hybrid SAM Overlap Refinement](#3-hybrid-sam-overlap-refinement)
4. [Phase 1–2: Multi-Object Tracking with BoxMOT](#4-multi-object-tracking)
5. [Phase 3: Post-Track Stitching](#5-post-track-stitching)
6. [Phase 4: Pre-Pose Pruning](#6-pre-pose-pruning)
7. [Phase 5: 2D Pose Estimation with ViTPose+](#7-2d-pose-estimation)
8. [Phase 6–7: Association, Re-ID, and Collision Cleanup](#8-association-and-reid)
9. [Phase 8: Post-Pose Pruning](#9-post-pose-pruning)
10. [Phase 9: Temporal Smoothing — 1-Euro Filter](#10-temporal-smoothing)
11. [3D Lifting — MotionAGFormer](#11-3d-lifting)
12. [Phase 10–11: Scoring and Export](#12-scoring-and-export)
13. [The Sweep: How We Found These Values](#13-the-sweep)
14. [Full Best-Configuration Reference (Trial #173)](#14-full-configuration-reference)
15. [What Did Not Work and Why](#15-what-did-not-work)
16. [Open Challenges and Future Directions](#16-open-challenges)
17. [Peak Design — The Maximum-Accuracy Single-Camera Architecture](#17-peak-design)

---

## 1. System Overview

### What does the pipeline do?

The pipeline ingests one video file and produces a structured, frame-by-frame record of every dancer in that video. The record includes:

- **Bounding boxes** — where in the frame each person is located, as a rectangle (x1, y1, x2, y2).
- **Persistent track IDs** — a number assigned to each individual that stays the same across all frames, even when that person is temporarily hidden.
- **2D skeletal keypoints** — 17 COCO-format joint positions (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) with per-joint confidence scores.
- **3D joint positions** — the above 2D skeleton "lifted" into three-dimensional world space.
- **Segmentation masks** (optional) — pixel-level outlines of each person when they overlap.
- **Motion scores** — deviation and synchrony metrics that describe how closely each dancer matches the group consensus motion.
- **MP4 visualizations** — annotated video exports of all the above.

### The 11-phase architecture

The pipeline runs as an ordered sequence of 11 printed phases (plus one unlabeled substep for 3D lifting). Each phase consumes the outputs of the previous one and is engineered to solve one class of problem. The architecture is linear and single-process; there is no branching execution path during a normal run.

```
Video file
  ↓
Phase 1–2:  YOLO detection + BoxMOT tracking + Hybrid SAM overlap refinement
  ↓
Phase 3:    Post-track stitching  (re-glue broken/fragmented track IDs)
  ↓
Phase 4:    Pre-pose pruning  (discard non-dancer tracks before expensive inference)
  ↓
Phase 5:    ViTPose+ 2D keypoint estimation (every single frame)
  ↓
Phase 6:    Occlusion re-ID + crossover refinement
  ↓
Phase 7:    Collision cleanup + pose deduplication
  ↓
Phase 8:    Post-pose pruning (tiered voting)
  ↓
Phase 9:    1-Euro temporal smoothing
  ↓
[3D lift]:  MotionAGFormer (unlabeled substep)
  ↓
Phase 10:   Spatio-temporal scoring
  ↓
Phase 11:   JSON + MP4 export
```

The rest of this paper walks through each phase in depth.

---

## 2. Person Detection — YOLO

### What is YOLO?

YOLO (You Only Look Once) is a real-time object detection neural network. "Detection" means: given a single image frame, output a set of bounding boxes, where each box says "I believe there is a person at this location with this confidence score." YOLO's defining feature is speed — it processes the entire image in one neural network forward pass (hence "only once"), unlike older systems that first proposed candidate regions and then classified each one separately.

A YOLO model outputs three things per detected object: the box coordinates, a class label (in our case, "person"), and a confidence score between 0 and 1. We set a threshold: boxes below the threshold are discarded.

YOLO has been developed through many generations (YOLOv1 through YOLOv11+). The version we use is described below.

### Our weights: `yolo26l_dancetrack`

**What we use:** `yolo26l_dancetrack.pt` — a large YOLO model fine-tuned on the DanceTrack dataset.

**Why this matters:** The base YOLO weights are trained on COCO, a general-purpose dataset with images of everyday objects and people in normal contexts. Dancing footage is far from "normal context." Dancers move extremely fast, strike unusual poses, are often partially cut off at the frame edges, and frequently overlap with other dancers. The DanceTrack fine-tuning exposes the model to exactly these conditions, making it substantially better at finding partially-visible, motion-blurred, awkwardly-posed bodies.

**Evidence from the sweep:** In our 347-trial optimization study, every single one of the top 50 trials used `yolo26l_dancetrack`. Alternative weights (`yolo26l_dancetrack_crowdhuman`, `yolo26l`) never appeared in the top 50. This is the single strongest signal in the entire sweep — the weight choice is not a tunable knob, it is a hard requirement.

The "l" in the model name denotes the "large" variant. YOLO models come in nano/small/medium/large/extra-large sizes. Larger models are slower but more accurate. We use "large" rather than "extra-large" because the accuracy ceiling of extra-large is not worth the latency cost at our video resolutions.

### Detection resolution: 800px

**What we use:** `SWAY_DETECT_SIZE=800` with `SWAY_GROUP_VIDEO=1`.

YOLO requires the input image to be resized to a square (letterboxed). A larger input square finds smaller, more distant people more reliably but costs more compute per frame. The parameter `SWAY_DETECT_SIZE` sets the baseline side length of this square. However, because we always have `SWAY_GROUP_VIDEO=1` (group/crowd mode), the effective size is `max(SWAY_DETECT_SIZE, 960)` — so 960px is the floor in group mode.

**Why 800?** The sweep confirmed 800px as the best all-round choice. Detection size 640 appeared in 12 of the top 50 trials (12/50), 800 in 26/50, and 960 in 12/50. The 960px size delivered no consistent accuracy gain over 800 but added approximately 25% more compute time per trial. The 640px size was slightly faster but caused a small but measurable accuracy drop (~0.5% aggregate HOTA).

### Confidence threshold: 0.16

**What we use:** `SWAY_YOLO_CONF=0.16`

This is the minimum score a detection must achieve to be kept. A value of 0 keeps every box the model produces; 1.0 keeps nothing. Counterintuitively, we want this low.

**Why low confidence?** Partially occluded dancers — someone whose upper body is hidden behind another person — may generate a low-confidence detection. If we reject all detections below 0.25 or 0.30, we lose those dancers entirely. A downstream step (pre-track NMS) handles the duplicates that result from keeping marginal detections. The sweep confirmed the sweet spot is **0.10–0.21**, with the global optimum at **0.16–0.17**. Configurations with confidence ≥ 0.25 showed measurable recall drops on partially visible dancers.

### Pre-track NMS IOU: 0.65

**What we use:** `SWAY_PRETRACK_NMS_IOU=0.65`

NMS stands for Non-Maximum Suppression. When YOLO detects someone, it often generates several overlapping boxes for the same person. NMS is the algorithm that eliminates the duplicates — it looks at pairs of overlapping boxes, computes their intersection-over-union (IoU), and suppresses the lower-confidence box if the IoU exceeds a threshold.

**Why 0.65 (looser)?** A higher IoU threshold means NMS removes fewer boxes — it only suppresses a box when the overlap is very high. This is "looser" suppression: we keep more boxes, including some that slightly overlap real detections. This sounds wasteful but is deliberately beneficial in our setting. When two dancers are close together, a tight NMS threshold (e.g. 0.40) would incorrectly merge them into one detection. With 0.65, we keep both. The tracker handles the remaining redundancy frame-to-frame.

**What did not work:** The old default NMS IOU of 0.50 (and anything lower) was definitively worse. The sweep is unambiguous: best range is **0.59–0.65**. Our previous stock OpenFloor configuration used 0.80 — extremely loose, which actually allowed too many spurious boxes through and caused downstream ID confusion.

---

## 3. Hybrid SAM Overlap Refinement

### What is SAM?

SAM (Segment Anything Model, developed by Meta) is a neural network that takes an image and a prompt (a point, a box, or a rough mask) and produces a precise pixel-level segmentation of the object at that location. Unlike detection networks that output rectangles, SAM outputs exact shapes — it can trace the outline of a person's arm even when it passes in front of another person's torso.

SAM 2.1 (the version we use) is the second generation, optimized for both images and video. We use the "base" variant (`sam2.1_b.pt`) which balances accuracy and speed.

### Why we use it here

Pure bounding box tracking has a fundamental problem: when two people's boxes overlap heavily, the tracker cannot tell which box belongs to which person. SAM solves this by computing, for each overlapping pair, exactly which pixels belong to each person. The resulting tighter boxes are then fed to the tracker instead of the raw YOLO boxes.

This is "hybrid" SAM because we do not run it on every frame for every detection — that would be too slow. We only invoke SAM when a pair of detected boxes overlaps beyond a threshold.

### Our configuration

**IoU trigger:** `SWAY_HYBRID_SAM_IOU_TRIGGER=0.42` (in the Phase 1–2 master lock) but the best trial used `b_hs_SAM_IOU=0.19` in the handshake mode.

The IoU trigger is the overlap percentage at which we decide to call SAM. A lower trigger means SAM runs on more pairs (more often). A higher trigger means we only call SAM on severely overlapping boxes.

**What we use:** In the `sway_handshake` mode, the effective trigger is very low — around **0.19–0.20**. This means SAM runs on almost any perceptible overlap. This is intentional: in dance footage, partial overlap is the norm, not the exception.

**Weak cues mode on:** `b_hs_WEAK=1` — this enables SAM to use lower-confidence cues to decide when to intervene. Without this, SAM would only fire on strong, obvious overlaps and miss the subtle crossings that matter most during fluid movement.

**Vertex stride 1:** `b_hs_VSTRIDE=1` — this means SAM overlap refinement runs on every single frame. Setting stride to 2 would skip every other frame, saving compute but allowing overlap confusion to persist for one frame before correction.

**SAM mask threshold:** `SWAY_HYBRID_SAM_MASK_THRESH=0.5` — the probability threshold for a pixel to be included in the mask. Standard value.

**Isolation scale:** `b_hs_ISO=1.6` — a scaling factor for how aggressively SAM separates overlapping regions. Values much below 1.0 would cause the segmentation to be too tight and clip dancer limbs.

**ROI crop (locked on):** Rather than running SAM on the full frame, we crop a Region of Interest around the overlapping pair plus a 25% padding fraction. This dramatically reduces compute cost since SAM processes a small crop rather than the entire 1920×1080 frame.

### What the old approach did differently

The original "stock OpenFloor" configuration used `sway_hybrid_sam_iou_trigger=0.50`. This higher trigger meant SAM was called far less often — approximately 25 refined frames per run vs. ~151 refined frames in the sweep-optimized configuration. In a side-by-side comparison on BigTest.mov, the lower trigger produced clearly better ID continuity through behind-body occlusion and re-entry. The qualitative result was decisive: the recovery variant was adopted as the new default.

---

## 4. Multi-Object Tracking — BoxMOT and SolidTrack

### What is multi-object tracking?

Detection gives us boxes per frame, but it has no memory. If you run YOLO on frame 100 and frame 101, you get two independent sets of boxes with no connection between them. Multi-object tracking (MOT) solves this: it assigns a persistent ID to each person and maintains that ID across frames, even through brief occlusions or when the person moves off screen and returns.

Modern trackers operate on an "estimate-then-associate" cycle every frame:
1. Based on previous positions and velocities, **predict** where each known track will be this frame.
2. Run the detector to get the new boxes.
3. **Associate** predicted positions with detected boxes using a cost function (typically IoU, appearance similarity, or both).
4. Update matched tracks, create new tracks for unmatched detections, and "hibernate" unmatched tracks (they may reappear soon).

### Our framework: BoxMOT

BoxMOT is a library that provides several tracker algorithms in a unified interface, all designed to work with YOLO detections. We use BoxMOT's tracker implementations throughout.

### Our tracker: SolidTrack

**What we use:** `SWAY_BOXMOT_TRACKER=solidtrack`

SolidTrack is a tracker variant that emphasizes track stability — it uses both motion (Kalman filter predictions) and appearance embeddings (visual similarity of cropped regions) to associate detections with tracks. It is the best-performing tracker in our sweep.

**Why SolidTrack won:** SolidTrack appeared in 74% of top-50 trials. It consistently outperformed alternatives on the harder test sequences (gymtest, mirrortest) where identity preservation across occlusion is most critical. SolidTrack's appearance-based re-ID component lets it recognize a returning dancer even after a gap, rather than assigning a new ID.

### Re-ID model: OSNet

**What we use:** `SWAY_BOXMOT_REID_WEIGHTS=osnet_x0_25_market1501.pt`

Re-identification (Re-ID) is the computer vision task of recognizing whether two cropped images show the same person. OSNet (Omni-Scale Network) is a re-ID backbone designed to capture both fine-grained texture details and coarse body shape simultaneously across multiple scales. The "x0_25" refers to a lightweight variant (25% of full capacity).

We use the Market-1501 pretrained weights. Market-1501 is a large pedestrian re-ID dataset — the pretrained features capture general person appearance representations that generalize reasonably well to dance footage.

**Why not the heavier model?** We tested `osnet_x1_0_msmt17` (full-size OSNet on a larger dataset). It added computation cost but showed only marginal accuracy improvement. The lightweight x0.25 model at 29/37 top SolidTrack trials is the clear best bang-for-buck choice.

### Key tracker parameters

**Max age: 135 frames** (`SWAY_BOXMOT_MAX_AGE=135`)

Max age is how many consecutive frames a track can go undetected before it is declared lost. If a dancer is occluded for longer than this many frames, the tracker gives up and the next time they appear they get a new ID.

**Why 135?** The sweet spot confirmed by the sweep is 90–180 frames, with 120–150 being the core. Our best trial used 135. Low max age (< 45) caused significant performance drops on mirrortest and bigtest, where dancers disappear behind mirrors or other bodies for extended periods. You need enough patience to wait for them to reappear. However, very high max age (> 200) allows wrong IDs to persist too long if a confusion event already occurred.

At 30fps video, 135 frames corresponds to 4.5 seconds — about the time a dancer might stand behind a mirror wall or a group cluster before re-emerging.

**Match threshold: 0.26** (`SWAY_BOXMOT_MATCH_THRESH=0.26`)

This is the maximum cost (lower = more similar) at which two candidate matches are accepted. A lower threshold means stricter matching — the tracker only associates a detection with a known track if they look very similar. A higher threshold allows looser matches.

**Why 0.26?** The sweep's sweet spot is 0.15–0.30. Our best trial's 0.26 is slightly above the center, allowing the tracker to re-associate dancers after small appearance changes (different lighting, slight clothing movement) without being so loose that two different people get merged.

**EMA: 0.91** (`b_st_EMA=0.91`)

EMA is the Exponential Moving Average weight for updating a track's appearance embedding over time. A value of 0.91 means the track's stored appearance is updated very slowly: new_appearance = 0.91 × old_appearance + 0.09 × new_embedding. This is "high stability" — the track's reference appearance changes slowly, which is good because a momentary unusual pose should not destabilize the long-term representation. The sweep confirmed high EMA (0.85–0.93) consistently wins.

### The competing trackers we evaluated

**BotSORT:** Appeared in 26% of top-50 trials. It is simpler and faster (~200s avg vs ~220s for SolidTrack) and achieved nearly the same aggregate score (e.g., Trial #310 scored 0.6592 vs Trial #173's 0.6597). BotSORT uses camera motion compensation to improve association but lacks SolidTrack's robust appearance re-ID. It is the recommended backup choice when speed matters more than the last ~0.001 in HOTA score.

**Deep OC-SORT:** This was the previous default tracker (in the "stock OpenFloor" configuration). It did not appear in the top 50 trials of sweep_v3. The old OpenFloor config used `deep_ocsort` with `max_age=120` and `match_thresh=0.35`. The stricter match threshold and lack of strong appearance re-ID caused ID loss when dancers reappeared after occlusion. The "recovery bias" improvement we discovered (lowering the threshold to 0.29 and raising max age to 165) helped Deep OC-SORT, but it still could not match SolidTrack at scale.

**ByteTrack:** Never appeared in the top 50. ByteTrack's key innovation is using "low-confidence" detections as secondary evidence, but it relies heavily on motion-only association with no strong appearance component. In group dance footage where motion patterns of different dancers frequently intersect, purely motion-based tracking fails to maintain identities across occlusions.

**DeepOCSORt (Deep OC-SORT):** Also never appeared in the top 50. Struggled especially with identity preservation across occlusions — the same fundamental problem as ByteTrack.

---

## 5. Post-Track Stitching

### Why tracking alone is not enough

Even the best tracker produces "fragmented" tracks — sequences where the same physical person ends up with multiple different IDs because the tracker lost them and then re-found them after the max age expired. In group dancing, this is nearly inevitable: a dancer might disappear completely behind a crowd for 5 seconds (150 frames), come back, and the tracker assigns a new ID because the old one aged out. From an analytics perspective, this looks like the original dancer left permanently and a new one appeared.

Phase 3 is entirely dedicated to fixing this: it stitches fragmented track segments back together into coherent long-lived identities.

### Step 1: Dormant merge

`apply_dormant_merges` identifies tracks that ended and new tracks that began shortly afterward at a compatible position. If a track disappears at frame 100 at position (500, 400) and a new track begins at frame 110 at position (510, 420), these are almost certainly the same person. They are merged.

**Max gap: 120 frames** (`SWAY_DORMANT_MAX_GAP=120`) — we will look back up to 120 frames to find a predecessor. At 30fps this is 4 seconds — long enough to cover most brief exits and mirror occlusions.

### Step 2: Neural AFLink (global track linking)

**What is AFLink?**

AFLink (Appearance-Free Link) is a learned temporal linking model. Unlike appearance-based re-ID that compares how people look, AFLink looks at the spatiotemporal patterns of track trajectories to decide which ones should be connected. It asks: "Is the motion pattern of track A's end compatible with track B's beginning, given the time gap and spatial displacement?" This is "appearance-free" because it does not use any visual features of the person's clothing or face.

**What we use:** `sway_global_aflink_mode=neural_if_available`

We use the neural AFLink model if the pretrained weights file (`models/AFLink_epoch20.pth`) is present, and fall back to the heuristic linker otherwise. The sweep confirmed this is the right choice: `neural_if_available` was used in all top-50 trials, and forcing the heuristic linker degraded scores.

**AFLink thresholds (master-locked):**
- `SWAY_AFLINK_THR_T0=0` — minimum frame gap to consider
- `SWAY_AFLINK_THR_T1=30` — maximum frame gap the linker will consider
- `SWAY_AFLINK_THR_S=75` — spatial distance threshold
- `SWAY_AFLINK_THR_P=0.05` — probability threshold for merging

These are locked and not tuned per-run because they reflect the temporal scale of the AFLink training distribution.

### Step 3: Fragment stitching

`stitch_fragmented_tracks` handles the general case of re-connecting broken track segments. It:
- Considers all pairs of (ended track, started track) within the max frame gap.
- Computes spatial compatibility: is the start position of the new track within a plausible radius of where the old track would have drifted?
- Uses the bounding box size as the radius scale: `SWAY_STITCH_RADIUS_BBOX_FRAC=0.55` means the search radius is 55% of the average bounding box height.

**Stitch max frame gap: 110 frames** (`SWAY_STITCH_MAX_FRAME_GAP=110`)

This is how far in time the stitcher will look to connect a track end to a new track start. 110 frames at 30fps = ~3.7 seconds. The sweep confirmed longer gaps help with occlusion recovery (best range: 80–130 frames).

**Short gap frames: 20** (`SWAY_SHORT_GAP_FRAMES=20`)

For very short gaps (under 20 frames), we apply a less strict spatial test — small gaps are almost certainly the same person and the looser criteria reduces false rejections.

### Step 4: Coalescence deduplication

`coalescence_deduplicate` finds pairs of tracks that are running simultaneously and have very high bounding box overlap for many consecutive frames. This catches the case where the tracker assigned two IDs to what is clearly one person.

**IoU threshold: 0.85** (`SWAY_COALESCENCE_IOU_THRESH=0.85`) — extremely high, meaning only very nearly identical boxes are merged. This avoids accidentally collapsing two close-together dancers into one.

**Consecutive frames: 8** (`SWAY_COALESCENCE_CONSECUTIVE_FRAMES=8`) — both tracks must co-exist with this high overlap for at least 8 consecutive frames. Brief accidental overlaps are not enough to trigger a merge.

### Step 5: Complementary and coexisting track merges

`merge_complementary_tracks` and `merge_coexisting_fragments` handle edge cases where a track was only partially observed — e.g., a person who was detected only in the upper half of frame because their lower half was cut off. These merge the partial observations into the most complete track for that person.

### Step 6: Box interpolation for stride gaps

If YOLO was run at stride > 1 (skipping frames for speed), the gaps between anchor detections are filled with interpolated boxes. **What we use:** `SWAY_BOX_INTERP_MODE=linear` — linear interpolation between anchor positions. The alternative (GSI, Gaussian-smoothed interpolation) was tested extensively but appeared in only 4/50 top trials vs. 46/50 for linear, with no consistent advantage. In fast, non-linear dance movement, the Gaussian smoothing introduces inertial artifacts that linear interpolation avoids.

---

## 6. Pre-Pose Pruning

### Why prune before pose estimation?

ViTPose+ (Phase 5) is computationally expensive: it runs a transformer-based neural network on a cropped image for each detected person per frame. If the tracker produced 20 IDs — some of which are the camera operator, an audience member, or a duplicate detection of a stage prop — we would be running expensive inference on non-dancers. Phase 4 cuts all tracks that are likely non-dancers before the pose phase begins.

The pruning decision is based on properties of each track that we can compute cheaply from the bounding boxes alone — no pose inference required.

### Pruning rules applied

**Duration ratio:** A track must span at least 20% of the video (`min_duration_ratio=0.20`). A dancer is in the scene for most of the video; a briefly-appearing bystander or an artifact detection is not. This is the most aggressive early cut.

**Kinetic motion:** `KINETIC_STD_FRAC=0.02`. A track's bounding box center must move over time by at least 2% of the frame size. Completely stationary detections (e.g., a reflected image in glass that stays perfectly fixed) are removed.

**Stage polygon (optional):** If a user defines a stage boundary polygon, tracks that spend most of their time outside it are pruned. Alternatively, `SWAY_AUTO_STAGE_DEPTH=1` uses a depth model to estimate the stage area automatically.

**Spatial outliers:** `SPATIAL_OUTLIER_STD_FACTOR=2.0` — if a track's average position is more than 2 standard deviations away from the cluster of all tracks, it is an outlier and probably not a dancer.

**Short tracks:** `SHORT_TRACK_MIN_FRAC=0.15` — must span at least 15% of the video.

**Audience region:** Tracks concentrated in the lower-right quadrant of the frame (where audiences typically sit) are removed.

**Late entrant short span:** A track that starts after 35% of the video has elapsed and ends before covering 17% of total duration is pruned. This removes brief intrusions.

**Bounding box size and aspect ratio:** Extremely small boxes (could be faces in a crowd) and extremely horizontal boxes (could be stage furniture) are removed.

**Geometric mirror detection:** Tracks that spend most of their time near the extreme left/right edges of the frame at consistent positions are likely mirror reflections and are pruned.

All pruning decisions are logged to `prune_log.json` so they can be reviewed and audited.

### Master-locked values

The six most critical pruning thresholds are master-locked and cannot be changed through the Lab UI. This prevents accidental mis-tuning that would flood the pose stage with garbage tracks or prune real dancers.

---

## 7. 2D Pose Estimation — ViTPose+

### What is pose estimation?

Pose estimation is the task of finding the locations of specific body joints (keypoints) on a person in an image. Given a cropped image of a person, a pose model returns 17 coordinates (for COCO format): nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee, left/right ankle — each with an x-coordinate, y-coordinate, and confidence score.

### What is ViTPose+?

ViTPose is a pose estimation model that uses a Vision Transformer (ViT) as its backbone. Transformers were originally invented for language processing but have since proven extraordinarily effective for vision tasks. A ViT divides an image into small patches, treats each patch like a "word," and uses self-attention mechanisms to understand relationships between patches across the whole image simultaneously.

The "+" in ViTPose+ refers to improvements including multi-task pretraining and enhanced multi-scale feature use. We use the "large" variant (`usyd-community/vitpose-plus-large`), which is the best balance between accuracy and speed for our use case. The "huge" variant exists but adds minimal accuracy at significant cost.

### Smart bounding box padding

Before passing a cropped person image to ViTPose, we apply `smart_expand_bbox_xyxy`. This function:
- Expands the crop box dynamically based on the person's estimated motion (adding "lead room" in the direction of movement)
- Adds extra top padding when the person appears to be lifting arms overhead (to prevent clipping high keypoints)
- Adjusts aspect ratio slightly to better match the model's expected input proportions

**Why this matters:** A ViTPose model trained on tightly-cropped person images performs poorly if the crop is too tight (keypoints near the edges get clipped) or too loose (background activations confuse the model). The smart pad adapts to each detection dynamically.

### Pose stride: 1 (every frame)

**What we use:** `pose_stride=1` — we run ViTPose on every single frame of the video.

**Why this is locked:** Early experiments considered running pose only on every other frame (stride 2) and interpolating the in-between frames. This would halve the pose inference cost. However, dance choreography involves extremely fast, high-frequency movement — a wrist can travel 100+ pixels in a single frame at performance speed. Interpolating between frames introduces artifacts: the interpolated wrist position may be on the wrong side of the body at the midpoint. We call this the "GSI speed cheat" and have permanently retired it for production.

**Linear interpolation (not GSI):** For any remaining gaps (e.g., if a track was missing for a frame), we fill with linear box interpolation rather than Gaussian-smoothed interpolation (GSI). GSI adds a physically implausible inertial smoothing effect that makes fast dance motion look robotic. 46 of the top 50 sweep trials used linear; GSI rarely outperformed it.

### Temporal keypoint refine

`apply_temporal_keypoint_smoothing` — after computing keypoints for a frame, we apply a confidence-weighted blend across the ±2 neighboring frames. This is a local temporal smoothing pass, distinct from the global 1-Euro filter in Phase 9. The purpose is to suppress single-frame keypoint jitter caused by motion blur or partial occlusion in one frame while the adjacent frames have clear visibility.

**Radius: 2 frames** (master-locked). This means each keypoint is blended with its values from 2 frames before and 2 frames after. This is a very local window — it does not significantly smooth out genuine fast motion, only eliminates single-frame spikes.

### Visibility gating

Before investing in pose inference, we check whether the person is likely visible at all. If the bounding box's predicted visibility score falls below 0.30 (`POSE_VISIBILITY_THRESHOLD=0.30`), we skip pose inference for that track in that frame. This prevents wasting compute on a box that is mostly occluded and whose keypoints would be random noise.

---

## 8. Association and Re-ID — Phases 6–7

### Phase 6: Occlusion re-ID

After pose estimation, we run `apply_occlusion_reid`. This phase addresses the case where the tracker failed to maintain identity through an occlusion even after Phase 3 stitching — for example, two tracks that Phase 3 could not confidently merge but that clearly belong to the same person based on their pose similarity.

Re-ID uses **HSV color strip embeddings** — for each person crop, we compute a compact color histogram of the torso region. If two different track IDs at different time points have very similar pose patterns (measured via OKS — Object Keypoint Similarity) and similar color embeddings, they are likely the same person and their IDs are reconciled.

**Max frame gap for re-ID: 90 frames** (master-locked `REID_MAX_FRAME_GAP=90`)

**Minimum OKS for re-ID: 0.35** (master-locked `REID_MIN_OKS=0.35`) — the pose patterns must be at least 35% similar for a re-ID merge to be considered.

Phase 6 also runs `apply_crossover_refinement` — when two persons' predicted paths cross (a "crossover event"), their IDs may have been swapped by the tracker. This step detects crossing events and corrects them.

`apply_acceleration_audit` flags tracks that show physically implausible sudden acceleration (a person cannot teleport 300 pixels in one frame). These are flagged for review.

### Phase 7: Collision cleanup

`deduplicate_collocated_poses` — after re-ID, some tracks may still have two different pose estimates for the same physical person in the same frame (a "collision"). This can happen when a re-ID merge was partial or when two tracks assigned to different IDs happen to both be fitted to the same visible person.

This is detected by comparing keypoint distances between all pairs of same-frame poses. If two poses are spatially coincident (keypoints within `COLLISION_KPT_DIST_FRAC=0.26` of the person's height), one is removed.

The collision state machine triggers entry at `COLLISION_ENTRY_IOU=0.6` (boxes overlap > 60% for 3 consecutive frames) and exits when overlap drops below 0.3. This state-aware design prevents a brief accidental overlap from causing a permanent dedup decision.

`sanitize_pose_bbox_consistency` — a final sanity check: if any keypoints are outside the bounding box (which should never happen if the pipeline is working correctly), they are clipped or flagged.

---

## 9. Post-Pose Pruning — Tiered Voting

### Why prune again after pose?

Phase 4 removed tracks that looked wrong based on box properties alone. Now that we have full pose estimates for every remaining track, we can apply richer quality checks. Some tracks survived Phase 4 (they moved plausibly and had reasonable boxes) but produced low-quality pose estimates — perhaps because they are actually reflections, or far-field background people, or technical artifacts.

Phase 8 uses a tiered voting system.

### Tier A — Confirmed humans (protected from pruning)

`compute_confirmed_human_set` identifies tracks that are definitively real dancers. A track earns Tier A status if:
- Its mean torso keypoint confidence (shoulders and hips — COCO joints 5, 6, 11, 12) is ≥ 0.5 in at least 40% of its pose frames
- It spans at least 10% of the video
- It is not concentrated near the frame edges (likely not a mirror reflection)

Tier A tracks are **skipped** for all Tier B voting. We never prune a confirmed human.

### Tier C — Ultra-low skeleton (immediate removal)

`prune_ultra_low_skeleton_tracks` removes tracks where the average skeleton confidence is below 0.15 (extremely low — essentially no valid pose) on 80% or more of frames. These are clearly not real people.

### Tier B — Weighted vote

For tracks that are not Tier A or Tier C, we run a weighted voting system. Each of several diagnostic functions votes on whether a track is bad:

| Rule | Weight | Logic |
|------|--------|-------|
| `prune_low_sync_tracks` | 0.7 | Dancer's motion is completely out of sync with the group |
| `prune_smart_mirrors` | 0.9 | Track is likely a mirror reflection (high confidence) |
| `prune_completeness_audit` | 0.6 | Track has many frames with missing keypoints |
| `prune_head_only_tracks` | 0.8 | Only the head is visible, no body pose |
| `prune_low_confidence_tracks` | 0.5 | Overall keypoint confidence is low |
| `prune_jittery_tracks` | 0.5 | Position jitter is too high relative to body size |

A track is pruned if the weighted sum of votes exceeds `PRUNE_THRESHOLD=0.65`. This threshold is not too aggressive — we need more than one weak signal before removing someone.

All pruning decisions are logged, and the prune log is included in the exported `data.json`.

---

## 10. Temporal Smoothing — 1-Euro Filter

### What is the 1-Euro filter?

The 1-Euro filter is an adaptive smoothing algorithm designed specifically for real-time noisy signals with varying speed. It was invented for interaction design (mouse and touch input) but works extraordinarily well for joint position smoothing.

The key insight is that the right amount of smoothing depends on the current speed of the signal:
- When a joint is **moving slowly** (e.g., a held pose), small jittery fluctuations are noise — apply **more smoothing**.
- When a joint is **moving fast** (e.g., a wrist during a quick gesture), the variations are probably real movement — apply **less smoothing** (preserve responsiveness).

A standard fixed-cutoff low-pass filter cannot do this. It either blurs fast movement or preserves slow jitter. The 1-Euro filter's adaptive cutoff frequency solves both problems simultaneously.

**How it works:** At each frame, the filter estimates the current speed of the signal (derivative). If the speed is high, the cutoff frequency is raised (less filtering). If the speed is low, the cutoff is lowered (more filtering). The "1-Euro" name refers to the minimum cutoff frequency of 1 Hz in the original paper.

### Our configuration

**Min cutoff: 1.0** (`SMOOTHER_MIN_CUTOFF=1.0`) — the baseline level of smoothing at low speed. This is master-locked because it was carefully tuned to the frame rates and motion speeds of dance footage.

**Beta: 0.7** (`SMOOTHER_BETA=0.7`) — controls how aggressively the cutoff increases with speed. Higher beta means the filter stays responsive at moderate speeds (good for dance) while still smoothing stationary holds.

This runs independently per joint per track. Each of the 17 COCO keypoints on each dancer track has its own 1-Euro filter instance.

**This is distinct from Phase 5 temporal refine.** Phase 5 did a local neighbor-frame blend at pose estimation time. Phase 9 applies a causal, forward-only filter over the entire track timeline. They address different sources of noise.

---

## 11. 3D Lifting — MotionAGFormer

### What is 3D lifting?

2D pose estimation gives us joint positions in pixel coordinates — a flat, screen-space representation. Real movement happens in three-dimensional space. "3D lifting" is the process of inferring the depth coordinate (z) of each joint from the 2D skeleton, producing a full 3D body pose.

This is an inherently under-constrained problem: many different 3D configurations can produce the same 2D projection. Modern neural approaches solve this by learning strong priors about how human bodies move — the model has seen millions of 3D motion captures and knows which 3D poses are physically plausible given a 2D input.

### Our model: MotionAGFormer

MotionAGFormer (Motion Adaptive Graph Former) is a graph-based transformer that processes a temporal window of 2D poses and produces 3D joint positions. The graph structure models the connectivity of the skeleton — adjacent joints (e.g., knee connected to hip) are linked with stronger weights than distant joints. The temporal window allows the model to reason about motion continuity: the z position of a joint at frame t is informed by its trajectory at t-1, t-2, etc.

**Why MotionAGFormer over alternatives?** PoseFormerV2 is also available as a backend. MotionAGFormer (`SWAY_LIFT_BACKEND=motionagformer`) is our default because it handles multi-person scenes more robustly — it processes each person independently but with their full temporal context, rather than trying to jointly model the whole scene.

### Depth integration

We use Depth Anything V2 (`SWAY_DEPTH_DYNAMIC=on`) to extract per-frame depth maps from the video. These provide approximate scene depth that is used to disambiguate the root joint's z position in world space.

**Root Z handling:** Depth maps from monocular video are not metric — they are min-max normalized per frame and do not give absolute distances. By default, we use a fixed `SWAY_DEFAULT_ROOT_Z=2.5` meters as the pelvis depth, which is a reasonable default for a stage performance at typical recording distance. Depth is used for relative depth between people in the scene, not as an absolute anchor.

**World coordinate export:** The 3D poses are exported in a unified world coordinate system (`SWAY_UNIFIED_3D_EXPORT=on`), placing all dancers in the same coordinate frame so their relative positions and movements can be directly compared.

### Running after 1-Euro, before scoring

The 3D lift runs after Phase 9 (1-Euro smoothing) and before Phase 10 (scoring). This ordering matters: we lift the smoothed 2D poses rather than the raw estimates, so the 3D output inherits the temporal stability of the 1-Euro filter. Lifting noisy 2D keypoints directly would produce very jittery 3D reconstructions.

---

## 12. Scoring and Export

### Phase 10: Spatio-temporal scoring

`process_all_frames_scoring_vectorized` computes metrics that quantify movement quality across the dancer ensemble:

**Joint angles:** For each kinematic pair (e.g., elbow → wrist → shoulder defines the elbow joint angle), compute the angle over time for each dancer.

**Consensus angles:** Compute the median angle across all dancers at each frame. This is the "group consensus" — what the choreography says should be happening at this moment.

**Deviations:** Per-dancer deviation from consensus. Low deviation means the dancer is in sync with the group; high deviation means they are executing a different shape.

**Shape errors and timing errors:** Shape error is the spatial error in joint positions; timing error is the phase offset in repetitive motifs.

### Phase 11: Export

The pipeline writes to the output directory:

- `data.json` — the complete frame-by-frame record (boxes, IDs, 2D/3D keypoints, scores, prune log)
- `{stem}_poses.mp4` — full pose visualization
- `{stem}_track_ids.mp4` — bounding boxes with ID labels
- `{stem}_skeleton.mp4` — skeleton-only rendering
- `{stem}_sam_style.mp4` — SAM-refined detection visualization
- `{stem}_3d.mp4` — 3D pose visualization when lift succeeded
- `prune_log.json` — audit trail of every pruning decision
- `track_stats.json` — per-track summary statistics
- `phase_previews/` — per-phase MP4s for debugging (when `--save-phase-previews`)

**Export visualization interpolation:** `SWAY_VIS_TEMPORAL_INTERP_MODE=linear` — box positions are linearly interpolated in the exported MP4 for smooth visual output. Linear is used here too for the same reasons as everywhere else.

---

## 13. The Sweep: How We Found These Values

### The problem

A pipeline with 30+ tunable parameters cannot be manually tuned. The interactions between parameters are too complex: changing the detection confidence affects how much work the tracker has to do, which interacts with max age, which interacts with the stitching gap thresholds, and so on. A parameter that looks optimal in isolation may perform worse when other parameters are also at their optimal individual values.

We needed a principled, automated search.

### Optuna and TPE

We used **Optuna** with the **Tree-structured Parzen Estimator (TPE)** algorithm. Optuna is a hyperparameter optimization framework; TPE is a Bayesian optimization algorithm that models the probability distribution of "good" parameter values based on past trial outcomes.

Unlike a grid search (which exhaustively tries all combinations and is computationally intractable for 30 parameters), TPE learns which regions of the parameter space are promising and focuses samples there. It is significantly more efficient than random search because it exploits past trial results.

**Study name:** `sweep_v3`  
**GPU instance:** `gpu_1x_a10` on Lambda Labs (NVIDIA A10 GPU, 24 GB VRAM, us-east-1)  
**Total trials:** 347 completed  
**Total GPU time:** ~20 hours  
**Average trial duration:** ~207 seconds (~3.5 minutes)

### The objective function

Each trial ran the full pipeline (Phases 1–3 only, stopping after stitching) on **five benchmark sequences**:

| Sequence | Character | Peak Score |
|----------|-----------|------------|
| `aditest` | Simple, few dancers, clean background | 0.899 |
| `easytest` | Low density, good lighting | 0.898 |
| `mirrortest` | Mirror walls create reflection confounds | 0.833 |
| `gymtest` | Gym environment, complex spatial arrangements | 0.659 |
| `bigtest` | Dense group, heavy occlusion, entry/exit cycling | 0.495 |

The score on each sequence is the HOTA (Higher Order Tracking Accuracy) metric from TrackEval — the standard multi-object tracking benchmark metric. HOTA jointly measures detection accuracy and association accuracy, making it sensitive to both "did you find everyone?" and "did you keep their IDs consistent?"

The objective was the **harmonic mean** of the five per-sequence HOTA scores. The harmonic mean is deliberately chosen over arithmetic mean: if one sequence scores 0.0 (complete failure), the harmonic mean is 0.0 regardless of the other scores. This forces the optimizer to find configurations that work across all conditions, not just configurations that maximize performance on easy sequences while failing on hard ones.

### Why stop at Phase 3?

Running Phases 4–11 (pose, 3D lift, scoring, export) for every trial would take 5–7× longer. The Phases 1–3 parameters (detection, tracking, stitching) are the dominant factors in tracking quality — they determine whether the right people are being tracked with stable IDs. Pose quality improvements are a separate optimization problem. Stopping at Phase 3 and scoring with TrackEval ground truth gives us the signal we need to optimize tracking parameters ~70–80% faster.

### Results

**Best trial: #173** with aggregate harmonic mean HOTA = **0.6597**

| Sequence | Score |
|----------|-------|
| aditest | 0.8854 |
| easytest | 0.8950 |
| mirrortest | 0.8256 |
| gymtest | 0.6513 |
| bigtest | 0.4876 |

The top-20 trials span only 0.0086 in aggregate score (0.6511–0.6597), meaning the optimizer converged well — the solution space near the optimum is flat, and further TPE trials with the same parameter space would not move the needle.

### Key sweep findings

**Convergence of the SolidTrack + sway_handshake combination:** SolidTrack with the `sway_handshake` Phase 1–3 mode combination won 29 of the top 50 slots. The runner-up pairing (BotSORT + standard mode) won 18 of the top 50. Everything else was absent.

**Pre-NMS IOU should be 0.65:** This is the single most actionable finding versus the old default (0.50). The improvement is consistent across all top-50 trials.

**Neural AFLink is non-negotiable:** Every top-50 trial used `neural_if_available`. Forcing the heuristic linker is definitively worse.

**bigtest is the hard ceiling:** No configuration broke through 0.50 HOTA on bigtest. The bottleneck is not a tunable parameter — it is the fundamental difficulty of maintaining track identity through dense group occlusions. This requires architectural improvements (better re-ID backbones, longer-horizon temporal reasoning) rather than parameter tuning.

---

## 14. Full Configuration Reference (Trial #173 — Current Best)

This is the complete drop-in configuration. All parameters not listed here use the master-locked defaults described in the relevant sections above.

```yaml
# ── Detection ──────────────────────────────────────────────
SWAY_YOLO_WEIGHTS: yolo26l_dancetrack        # fine-tuned on DanceTrack dataset
SWAY_YOLO_CONF: 0.16                          # low threshold — keep partial detections
SWAY_DETECT_SIZE: "800"                       # effective 960 in group mode
SWAY_PRETRACK_NMS_IOU: 0.65                   # loose NMS — keep near-overlapping dets

# ── Tracker ────────────────────────────────────────────────
SWAY_BOXMOT_TRACKER: solidtrack               # appearance + motion re-ID
SWAY_BOXMOT_MAX_AGE: 135                      # 4.5 seconds patience at 30fps
SWAY_BOXMOT_MATCH_THRESH: 0.26               # moderate strictness on association
SWAY_BOXMOT_REID_WEIGHTS: osnet_x0_25_market1501.pt   # lightweight re-ID backbone

# ── SolidTrack internal params ──────────────────────────────
b_st_TIOU: 0.3                                # IoU component weight
b_st_TEMB: 0.375                              # embedding component weight
b_st_EMA: 0.91                                # high EMA — stable appearance model

# ── Phase 1–3 Mode ─────────────────────────────────────────
SWAY_PHASE13_MODE: sway_handshake             # SAM-assisted handshake linking

# ── Hybrid SAM (handshake mode) ────────────────────────────
b_hs_SAM_IOU: 0.19                            # very low trigger — SAM fires often
b_hs_WEAK: "1"                                # weak cue mode on
b_hs_VSTRIDE: "1"                             # every frame
b_hs_ISO: 1.6                                 # isolation scale

# ── Post-track stitching ───────────────────────────────────
SWAY_STITCH_MAX_FRAME_GAP: 110               # ~3.7 sec gap tolerance
SWAY_STITCH_RADIUS_BBOX_FRAC: 0.55           # search radius = 55% of bbox height
SWAY_SHORT_GAP_FRAMES: 20                     # loose criteria for short gaps
SWAY_DORMANT_MAX_GAP: 120                     # ~4 sec dormant track patience
SWAY_COALESCENCE_IOU_THRESH: 0.85            # only merge nearly-identical boxes
SWAY_COALESCENCE_CONSECUTIVE_FRAMES: 8       # must overlap for 8 frames
SWAY_BOX_INTERP_MODE: linear                 # linear (not GSI) interpolation
sway_global_aflink_mode: neural_if_available  # use AFLink neural linker

# ── Hybrid SAM (global overlap params) ────────────────────
SWAY_HYBRID_SAM_MASK_THRESH: 0.5
SWAY_HYBRID_SAM_BBOX_PAD: 4
SWAY_HYBRID_SAM_ROI_PAD_FRAC: 0.25
```

---

## 15. What Did Not Work and Why

### ByteTrack

**What it is:** A tracker that uses both high-confidence and low-confidence detections in a two-stage association. The idea is clever: after matching high-confidence detections to tracks, use the remaining low-confidence detections as secondary evidence.

**Why it failed:** ByteTrack is fundamentally motion-based. It uses Kalman filter predictions to associate detections and does not maintain an appearance embedding per track. When two dancers swap positions during a crossover, ByteTrack frequently swaps their IDs because the positions are more similar than the identities. In group dance footage where crossing paths is frequent and deliberate, this is a fatal flaw.

### Deep OC-SORT as default (old stock OpenFloor config)

**What it is:** An extension of OC-SORT that optionally uses appearance embeddings. The "OC" stands for Observation-Centric — it uses actual detection observations rather than purely predicted positions for association, making it more robust to non-linear motion.

**Why it underperformed:** The old default configuration used `match_thresh=0.35` (strict), `max_age=120`, and `reid_on=False` (no appearance embeddings). The strict match threshold caused ID loss when a dancer reappeared after occlusion in a slightly different position. The lack of appearance re-ID meant the tracker could not recognize a returning dancer — it would assign a new ID whenever the gap was too large for motion prediction alone. We found that when we lowered the threshold to 0.29, raised max age to 165, and enabled SAM with a lower IoU trigger, Deep OC-SORT improved substantially. But it still could not match SolidTrack with the full re-ID stack.

### GSI (Gaussian-smoothed interpolation)

**What it is:** An alternative to linear interpolation for filling gaps between detection anchors. GSI uses a Gaussian process to produce a smooth curve through the anchor points, which in theory better captures the physically smooth trajectories of moving people.

**Why it failed:** Dance movement is not smooth on the scales we are interpolating. A fast hip circle can complete in 3–4 frames. Linear interpolation between two anchor positions 6 frames apart will place the midpoint halfway between them in a straight line. GSI, given a smooth prior, will actually put the midpoint even further from the truth because it will predict a smooth trajectory that a fast dancer did not follow. Linear interpolation's failure mode (straight-line motion during a curve) is less harmful than GSI's failure mode (wrong smooth curve through rapid moves). The sweep confirmed this: 46/50 top trials used linear.

### Detection size 960

**What it is:** Running YOLO on 960×960 pixel inputs rather than 800×800.

**Why it underperformed:** Larger detection resolution means YOLO can find smaller, more distant people. At 800px with group mode on (effective 960px floor), we are already at this resolution. Testing explicit 960 in the DETECT_SIZE param with group mode was redundant and simply added ~25% compute cost with no consistent accuracy improvement. 12 of 50 top trials used 960 vs 26 at 800.

### High YOLO confidence (≥ 0.25)

**What happened:** When we set `SWAY_YOLO_CONF ≥ 0.25`, recall dropped on partially visible dancers. The exact failure mode: a dancer partially behind another person generates YOLO confidence scores around 0.18–0.22. At threshold 0.25, these detections are simply discarded. The tracker never sees them. When the dancer fully emerges, the tracker assigns a new ID because it has no history. The downstream stitch can sometimes recover this but not always.

### `dancer_registry` Phase 1–3 mode

**What it is:** An alternative to `sway_handshake` that maintains an explicit registry of known dancer identities, attempting to be more conservative about assigning new IDs.

**Why it underperformed:** dancer_registry showed the highest gymtest scores (0.659 max vs 0.654 for sway_handshake) but was 25–35% slower per trial than sway_handshake. The computational cost was not justified by the marginal gymtest improvement, especially since sway_handshake was nearly equivalent on gymtest and clearly better on mirrortest.

### Very low max_age (< 45 frames)

**What happened:** When max age was set below 45 frames (~1.5 seconds), performance on mirrortest and bigtest dropped dramatically. The reason is straightforward: mirrortest involves dancers passing behind mirror walls. Even a brief mirror pass can hide someone for 60–90 frames. With max_age < 45, those tracks expire and get new IDs after re-emergence.

### Zero-value trials (11 total)

In the very early exploration phase (Trials #12, #21, #46, #61, #65, and others), entire trial runs collapsed to 0.0 HOTA on all five sequences. The cause: certain combinations of very low YOLO confidence (0.05–0.08) with very high pre-track NMS IOU (0.85+) produced an essentially empty detection set. Very low confidence means detections are marginal; very high NMS IOU means the NMS step is extremely loose and many boxes survive, but then very low tracker thresholds killed them all. The Optuna optimizer learned from these failures and quickly steered away from this parameter region.

---

## 16. Open Challenges and Future Directions

### bigtest — the hard ceiling

bigtest is a dense group video with heavy simultaneous occlusion, multiple entry/exit cycles, and complex spatial arrangements. Our best configuration achieves only 0.4876 HOTA — the ceiling appears to be around 0.50 for the current architecture. The optimizer ran 347 trials and could not break through this ceiling.

The root cause is not tunable parameters. It is architecture:

1. **Track identity loss during group occlusion:** When all five dancers overlap in a tight cluster, every tracker loses IDs. No amount of max_age tuning helps if the detections themselves are ambiguous. Better separation requires either a more powerful segmentation model or a different approach to identity maintenance during full occlusion (e.g., body part tracking rather than whole-body tracking).

2. **Entrance/exit cycling:** Dancers who leave and re-enter the frame multiple times accumulate ID assignments over time. The neural AFLink model is trained on typical gap distributions; very long gaps (dancer exits for a full minute) push beyond its reliable operating range. Longer-horizon temporal linking with a stronger appearance model is needed.

3. **Group-scale re-ID:** OSNet was trained on pedestrian datasets with relatively fixed appearances. Dancers in matching costumes under complex stage lighting frequently confuse appearance-based re-ID at the crop level. A group-aware re-ID model that explicitly models relative position within the ensemble could help.

### Diminishing returns from further Optuna tuning

The top-20 trial scores span only 0.0086 (0.6511–0.6597). The objective landscape near the optimum is very flat. Running another 347 trials with the same parameter space would produce marginal gains at best. The next performance improvement requires architectural changes: a new detection backbone, a better re-ID model, or an improved SAM integration for dense occlusion.

### Phase 5–11 sweep

The current sweep optimized only Phases 1–3. Phases 5–11 (pose estimation, association, pruning, smoothing) were not part of the search space. A secondary sweep optimizing the ViTPose model size, temporal refine radius, 1-Euro parameters, re-ID thresholds, and pruning weights against pose quality ground truth would likely yield further improvements.

### 3D lifting quality

The MotionAGFormer 3D lift is applied to each dancer independently. In group scenes, cross-person depth estimation is challenging because the monocular depth cues for one person are confounded by other people in the frame. A group-aware 3D lifting approach that jointly reasons about all visible dancers could produce more geometrically consistent world-space reconstructions.

---

## 17. Future Pipeline — Ideal Architecture

**This section has been moved to its own document.** The future pipeline design — incorporating all sweep findings, SOTA research (SAM2MOT, MeMoSORT, MATR, MOTE, Sentinel, KPR, UMOT, and others), logical hybrid strategies, and quantitative targets — lives in:

> **[`docs/FUTURE_PIPELINE.md`](./FUTURE_PIPELINE.md)**

That document is the unified ideal architecture for maximum-accuracy single-camera dance critique, built on the foundation of what sweep_v3 proved works and where it hit the architectural ceiling.

---

*End of document*

*Generated from sweep_v3 results (2026-03-30). Parameters reflect Trial #173 optimal configuration. For the raw sweep data, see `output/sweeps/optuna/sweep.db` and `output/sweeps/optuna/sweep_status.json`.*
