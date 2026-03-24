# Pipeline improvements roadmap

**Source analysis:** derived from `pipeline_improvements_analysis.md.resolved` (Antigravity brain `9b20b314-5e42-44ab-a69c-f847deedff0e`).  
**This file tracks what is implemented in `sway_pose_mvp` and what remains.**

---

## Status summary

| Item | Status | Notes |
|------|--------|--------|
| **ROI-cropped SAM2** (union of overlapped dancers + pad, SAM on crop only) | **Done** | `sway/hybrid_sam_refiner.py`; default on. `SWAY_HYBRID_SAM_ROI_CROP=0` restores full-frame SAM. `SWAY_HYBRID_SAM_ROI_PAD_FRAC` (default `0.1`). Lab: `sway_hybrid_sam_roi_crop`, `sway_hybrid_sam_roi_pad_frac`. |
| **Lower IoU trigger with ROI** (e.g. 0.30–0.35) | **Manual** | No default change; ROI makes lower triggers cheaper—tune `sway_hybrid_sam_iou_trigger` / env. |
| **RTMPose-L in pipeline + Lab UI** | **Done (optional backend)** | `--pose-model rtmpose`, Lab `RTMPose-L`. Requires MMPose stack (see `requirements-rtmpose.txt`). `sway/rtmpose_estimator.py`. Mask-gated ViTPose crops not ported for RTMPose yet. |
| **Sapiens qualitative test** | **Not started** | No code path; manual / separate env. |
| **Explicit YOLO + ViTPose batching** | **Not started** | Roadmap phase B. |
| **TensorRT / FP16 path (detector + pose)** | **Not started** | ViTPose already fp16 on MPS/CUDA; TRT export not done. |
| **Bidirectional tracking (forward + backward merge)** | **Not started** | Roadmap phase C. |
| **GNN group graph tracking** | **Not started** | Roadmap phase D. |
| **HMR 2.0 / 4DHumans optional 3D** | **Not started** | Roadmap phase D. |
| **Golden-set benchmarks (latency, ID stability, QA)** | **Partial** | Existing pose QA tests; no dedicated golden-set harness in repo. |

---

## Recommended order (from analysis)

1. ~~ROI-cropped SAM2~~ **implemented**  
2. ~~RTMPose-l selectable + experiment~~ **implemented (install MMPose to run)**  
3. Sapiens spot-check — **todo**  
4. Frame/crop batching — **todo**  
5. TensorRT / FP16 path — **todo**  
6. Bidirectional tracking — **todo**

---

## Original phased roadmap (A–D)

| Phase | Focus | Status |
|-------|--------|--------|
| **A** — Quick wins | ROI SAM, RTMPose toggle, Sapiens spot-checks | ROI + RTMPose UI/backend done; Sapiens not started |
| **B** — Throughput | YOLO + ViTPose batching; FP16/TensorRT on detector path | Not started |
| **C** — Robustness | Bidirectional merge; AFLink + forward/back consensus | Not started |
| **D** — Research | GNN graph; HMR 2.0 optional 3D | Not started |

---

## How to validate

- **ROI SAM:** run a clip with hybrid SAM on; compare wall time vs `SWAY_HYBRID_SAM_ROI_CROP=0` on the same video.  
- **RTMPose:** `pip install -r requirements-rtmpose.txt` (adjust mmcv for your platform), then `python main.py … --pose-model rtmpose` or Lab **RTMPose-L**.  
- **Regression:** `pytest sway_pose_mvp/tests/test_hybrid_sam_roi.py`

---

*Update this table when you complete or reprioritize items.*
