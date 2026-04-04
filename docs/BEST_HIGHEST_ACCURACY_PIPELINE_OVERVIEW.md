# Best Highest-Accuracy Pipeline (Paper Goal Version)

This doc answers one question only:
**What is the best highest-accuracy pipeline for Sway, and what is already implemented vs still to do?**

---

## 1) Single Best Pipeline (No Options)

For maximum accuracy in dense dance clips, the best end-state pipeline is:

1. **Strong person detection**
   - `YOLO26l_dancetrack` scout for robust recall.
   - Co-DINO precision re-detection on hard overlap/occlusion frames.

2. **Per-person pose/keypoints**
   - ViTPose-Huge with mask-guided crops and smart padding.

3. **Primary tracking**
   - SAM2 primary tracker (`sam2.1_h`) with Cross-Object Interaction memory quarantine.

4. **Identity matching (multi-signal Re-ID fusion)**
   - BPBreID (fine-tuned) + KPR + color + spatial + face + skeleton signals.
   - Pose-gated EMA gallery updates (freeze contaminated frames).

5. **Global relinking / stitching**
   - Phase-3 stitch + AFLink global link.
   - Backward pass + forward/reverse merge for re-entry recovery.

6. **Collision resolution**
   - Global assignment solver (DP/Hungarian policy).

7. **Postprocess + critique**
   - Phase-8 pruning -> Phase-9 smoothing -> 3D lift -> scoring/export.
   - Confidence-gated critique with explicit visibility gaps.

This is the highest-accuracy architecture for this repo’s trajectory.

---

## 2) What Is Already Implemented

Implemented in codebase now (modules exist and are wired to runtime paths):

- Detector stack:
  - YOLO path
  - DETR-family backends (Co-DINO/Co-DETR/RT-DETR)
  - Hybrid detector dispatcher
- Tracking stack:
  - SAM2 primary tracker
  - COI module
  - State machine
  - Tracker factory and engine switching
- Re-ID stack:
  - Enrollment gallery
  - BPBreID extractor
  - KPR extractor
  - Color-hist signal
  - Skeleton/face add-on paths
  - Re-ID fusion engine + weights + EMA controls
- Recovery / association:
  - Phase-3 stitching and AFLink integration path
  - Backward pass module
  - Collision solver module
- Pose / 3D / scoring:
  - ViTPose stack and mask-guided pose path
  - MotionAGFormer and MotionBERT backend paths
  - Critique engine and export flow
- Advanced modules:
  - MOTE, Sentinel, UMOT modules implemented as add-ons

---

## 3) What Is Still "Yet To Be" (For True Best Accuracy)

These are the remaining execution gaps between "implemented modules" and "fully optimized best pipeline":

1. **Hard production lock of one final stack**
   - Freeze one canonical best profile end-to-end (detector -> tracker -> reID -> relink -> critique).
   - Eliminate fallback drift during runs.

2. **Weight/dependency completeness**
   - Ensure all high-accuracy checkpoints are present and actually loaded:
     - SAM2 huge
     - Co-DINO
     - BPBreID fine-tuned
     - ArcFace runtime
     - MoCos
     - AFLink weights
   - Validate no silent fallback to weaker models.

3. **Gate-based validation to lock winners**
   - Run ordered gate validation on bigtest:
     - tracking foundation
     - re-ID fusion
     - detection precision trigger behavior
     - pose/3D quality
     - backward-pass identity recovery
   - Promote only measured winners to permanent defaults.

4. **Final integration polish**
   - Ensure single-path orchestration in `main.py` stays consistent with chosen best stack.
   - Keep advanced add-ons off unless they produce measured gains on your target clips.

---

## 4) Final Practical Statement

If your requirement is "highest accuracy now for Sway":

- Use the architecture above as the one true target.
- Treat the system as **mostly implemented** at module level.
- The remaining work is **final locking + full weight provisioning + rigorous gate validation**, not inventing new architecture.

