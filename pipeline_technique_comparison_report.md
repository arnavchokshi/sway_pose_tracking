# Closed-Set CV Pipeline Technique Comparison

## Sources fully reviewed

1. `Computer Vision Pipeline Improvement Plan Review.pdf`
2. `Review of a Plan to Stabilize and Optimize a Closed-Set Computer Vision Pipeline.pdf`
3. `pipeline_improvements_v23_d16f8ce0.plan.md`

---

## Executive decision

Best overall approach is a **hybrid**:
- Keep the `V23` core architecture and A-H issue framing.
- Adopt the second review's **evaluation discipline and risk controls** (HOTA/IDF1 slices, reason-code telemetry, staged rollout, guardrails for hysteresis and phase-8).
- Adopt the first review's **high-upside upgrades** selectively (DEIMv2 activation, stronger tracker backend testing, runtime acceleration stack, MLOps drift/telemetry), but only behind correctness gates.

This hybrid is strongest because:
- `V23` is closest to your actual failures and artifacts.
- Review #2 is strongest on avoiding false improvements and regression risk.
- Review #1 is strongest on expanding ceiling performance and production hardening.

---

## Problem-by-problem comparison and decision

## 1) Identity instability in overlap windows (BigTest switch bursts)

- **V23 plan:** Solution B (hysteresis/ID lock + overlap-aware penalty), Solution E (phase-8 correction tuning), optional Solution F (better tracker backend).
- **Review #1:** Strongly supports hysteresis/penalty and recommends modern tracker upgrades (BoT-SORT/StrongSORT family) as key for dense overlap.
- **Review #2:** Agrees directionally, but warns hysteresis can reduce switch count while increasing wrong-ID duration; demands explicit state-machine unlock logic + association-aware metrics.

**Best choice:**  
Use **V23 B + E**, but implemented with **Review #2 safety constraints**, then run **Solution F A/B**.

**Why this is best:**  
You need immediate stabilization (B/E), but without review #2 guardrails you can "look better" numerically while identity continuity worsens. The robust sequence is: telemetry -> guarded hysteresis -> phase-8 tuning -> tracker A/B.

---

## 2) MirrorTest handoff (ID10 -> ID11 around frame ~254)

- **V23 plan:** B (switch resistance) + A (enrollment de-dup) to eliminate upstream contention.
- **Review #1:** Same direction, emphasizes overlap-cost inflation and local continuity.
- **Review #2:** Same direction, explicitly calls this a case where lock logic needs contradiction-based unlock criteria.

**Best choice:**  
**A first, then B** (with unlock criteria and per-switch reason codes).

**Why this is best:**  
This specific failure can come from both contaminated enrollment and over-eager switching. Fixing enrollment first removes self-competition, then B cleans remaining oscillation.

---

## 3) Enrollment near-duplicate seeds (10/11, 6/8)

- **V23 plan:** Solution A spatial+appearance de-dup gate.
- **Review #1:** Strongly endorses composite spatial + appearance suppression and optional keypoint similarity refinement.
- **Review #2:** Calls this highest-confidence, high-leverage, low-risk near-term fix.

**Best choice:**  
**Directly implement Solution A** with strict logging of merge/suppress decisions.

**Why this is best:**  
All three sources converge here with minimal disagreement. This is the cleanest high-impact upstream fix.

---

## 4) Too many short-lived noisy tracks

- **V23 plan:** Solution C deterministic pruning (`<8` frames + low confidence + isolation), with dark-zone exceptions.
- **Review #1:** Supports pruning as structural noise reduction to reduce downstream Re-ID load.
- **Review #2:** Agrees but warns pruning can hide core tracking errors; recommends confirmation/delayed emission semantics and boundary-case tests.

**Best choice:**  
**Solution C with review #2 framing**: track confirmation + delayed emission, not only after-the-fact deletion.

**Why this is best:**  
You still get the clutter reduction, but you avoid masking real problems and reduce risk of deleting true brief appearances.

---

## 5) Low-confidence limb hallucination represented as valid structure

- **V23 plan:** Solution D confidence-state fields + gray rendering of low-confidence limbs.
- **Review #1:** Strong endorsement; frames this as semantic integrity and trust calibration.
- **Review #2:** Strong endorsement; aligns with standard thresholded pose visualization practice.

**Best choice:**  
**Implement D exactly** (confidence enum + UI rendering + schema propagation).

**Why this is best:**  
All three align strongly; low engineering risk, high trust and review-quality gain.

---

## 6) Runtime bottleneck (phase-2 masking + phase-5 re-ID dominate)

- **V23 plan:** Solution H (mask refresh cadence, ROI reuse, embedding cache) after accuracy fixes.
- **Review #1:** Supports H and extends with mixed precision, TensorRT/OpenVINO paths, dynamic batching, and possible multi-GPU split.
- **Review #2:** Supports H but insists on correctness-aware caching and deterministic regression checks; profile-first approach.

**Best choice:**  
Use **V23 H sequencing** plus a **Review #1 acceleration stack**, constrained by **Review #2 correctness gates**.

**Why this is best:**  
Review #1 gives the strongest throughput toolbox; review #2 prevents speedups from silently damaging quality.

---

## 7) Weak gait modality (fallback-only handcrafted gait)

- **V23 plan:** Solution G learned gait checkpoint + fusion retune.
- **Review #1:** Very supportive, strongly recommends MoCos-style learned gait embedding.
- **Review #2:** Cautious; says transfer to dance-like noncanonical motion is not guaranteed without domain validation.

**Best choice:**  
Treat G as **conditional**: run ablation on overlap slices first; deploy only if motion cue improves IDF1/HOTA without regressions.

**Why this is best:**  
High upside exists, but certainty is lower than A/C/D/B. This is where review #2 caution is crucial.

---

## 8) DEIMv2 detector path not active

- **V23 plan:** Issue 8 demands enabling DEIMv2 path.
- **Review #1:** Calls DEIMv2 activation one of the highest-impact improvements because better upstream separation reduces downstream association collapse.
- **Review #2:** Supports detector improvement but stresses evaluation discipline and staged integration.

**Best choice:**  
**Enable DEIMv2 path behind canary/feature gate immediately after telemetry baseline**, then compare against current detector on overlap slices.

**Why this is best:**  
This is likely high leverage, but should be rolled out with A/B evidence rather than full replacement.

---

## 9) Missing production lifecycle controls (not explicit in V23 core)

- **V23 plan:** Includes diagnostics additions, but not full MLOps lifecycle.
- **Review #1:** Explicitly adds telemetry, drift monitoring, versioning, automated retraining pipeline recommendations.
- **Review #2:** Strongly reinforces observability and metric/dashboard rigor as first-class.

**Best choice:**  
Adopt a **lean production layer now**: metric schema + dashboards + drift alarms + reproducible eval runner; postpone full auto-retrain until post-stabilization.

**Why this is best:**  
Gives practical operations safety now without derailing current stabilization work.

---

## Final ranked implementation order (best combined strategy)

1. **Observability baseline first** (switch reasons, phase-8 reject reasons, overlap-slice metric harness: HOTA/IDF1 + existing counters).
2. **A: enrollment de-dup**.
3. **C: confirmation/pruning with dark-zone exceptions**.
4. **D: joint confidence semantics + gray rendering**.
5. **B: hysteresis/ID-lock with strict unlock state machine**.
6. **E: phase-8 guarded tuning using reject telemetry**.
7. **Enable DEIMv2 path in canary A/B**.
8. **F: tracker backend A/B (BoT-SORT/StrongSORT/OC-SORT family)**.
9. **H: runtime optimization** (cache + cadence + compile/mixed precision/batching) with quality gates.
10. **G: learned gait deployment only if ablation proves gain on overlap-heavy slices**.

---

## Where each source is strongest

- **Best for practical in-repo fixes:** `pipeline_improvements_v23_d16f8ce0.plan.md`
- **Best for expansion and long-term performance ceiling:** `Computer Vision Pipeline Improvement Plan Review.pdf`
- **Best for avoiding false wins and regression risk:** `Review of a Plan to Stabilize and Optimize a Closed-Set Computer Vision Pipeline.pdf`

---

## Confidence statement

For your current known issues, I am highest-confidence on:
- A (enrollment de-dup),
- C (track confirmation/pruning),
- D (pose confidence semantics),
- and B/E only with strict telemetry-driven guardrails.

I am medium-confidence (high-upside but evidence-dependent on your data) on:
- G (learned gait checkpoint as a net positive in dance-heavy overlap),
- and the exact detector/tracker winner among DEIMv2 vs alternatives until A/B is run on your overlap slices.
