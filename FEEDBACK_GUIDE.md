# Sway Pose Tracking — Captain & Dancer Feedback Guide

How to use pose tracking data to give captains and individual dancers actionable feedback, optimized for **quickly seeing what's wrong** and **fixing it** as intuitively as possible.

---

## Data You Already Have

From the pipeline you produce:

| Data Type | What It Is | Where It Lives |
|-----------|------------|----------------|
| **6 Joint Angles** | left/right shoulder, elbow, knee (degrees 0–180°) | `track_angles`, `consensus_angles` |
| **Deviations** | Your angle vs group angle per joint per frame (degrees) | `deviations` |
| **Shape Errors** | cDTW-based "how wrong is the shape" per joint (degrees) | `shape_errors` |
| **Timing Errors** | Frames behind/ahead of group per joint (positive = ahead) | `timing_errors` |
| **Worst Joints** | Top 3 joints with highest mean shape error per dancer | `track_summaries.worst_joints` |
| **Sync Score** | 0–100 overall sync metric | `track_summaries.sync_score` |

---

## Captain: Group-Level Insights (High-Level)

*Goal: At a glance, see who's off and in what way.*

### 1. One-Sentence Squad Summary

**Example:** *"3 dancers are off: IDs 5 and 12 are ahead of the beat, ID 7 has shape issues in the knees."*

Show:
- Who is **off-beat** (timing only): "Ahead" / "Behind"
- Who has **shape problems** (joint angles wrong)
- Who is **most out of sync** (lowest sync score)

### 2. Heatmap Table / Grid

| Dancer ID | Overall Sync | Worst Problem | Primary Joint |
|-----------|--------------|---------------|----------------|
| 3 | 94 | — | — |
| 5 | 72 | **Ahead 4 frames** | right_knee |
| 7 | 68 | **Shape 18° off** | left_knee |
| 12 | 71 | **Behind 3 frames** | left_elbow |

Make the "Worst Problem" column clickable → jumps to that moment in the video.

### 3. Real-Time "Who's Off Right Now"

At any given moment in the video:
- List of dancer IDs currently in Red or Blue
- One-line label: *"5: ahead"*, *"7: left knee bent"*
- Optional filter: show only timing issues vs only shape issues

### 4. Formation / Spatial View

You already have bounding boxes and keypoints. Add:
- Average position per dancer vs group centroid
- Highlight if someone is consistently too far forward/back or off to the side

---

## Individual Dancers: Actionable, Joint-Specific Feedback

*Goal: Answer "what's wrong?" and "how do I fix it?" in seconds.*

### 1. Top-3 Problem Joints (Plain Language)

Instead of "left_knee_shape: 18.2", show:

- **"Left knee: 18° more bent than the group"**
- **"Right elbow: ~4 frames ahead of the beat"**

Translate angles into **movement direction** (e.g., "more bent", "straighter", "higher") so dancers know what to change.

### 2. Per-Joint Cards with Video Jump

For each of the 6 joints:

```
┌─────────────────────────────────────────┐
│ LEFT KNEE                                │
│ Shape error: 18° (aim for <20°)          │
│ Timing: ~4 frames ahead                 │
│ [Jump to worst moment in video]         │
└─────────────────────────────────────────┘
```

- Show current metric + threshold (e.g., "aim for <20°")
- **"Jump to worst moment"** → seek video to the timestamp where that joint is worst (max deviation or max shape error)

### 3. Before/After "Correct" Pose Overlay

Use the consensus pose at that moment:
- Side-by-side: **your pose** vs. **group pose** (ghost overlay)
- Red outline/highlight on the limb that's wrong
- One-line fix: *"Straighten left knee to match"* or *"Slow down right elbow by ~0.1s"*

### 4. Timing vs Shape — Clear Labels

Separate the two error types explicitly:

- **Shape:** *"Your left knee is ~18° more bent than the group."*
- **Timing:** *"You're starting the move ~4 frames (~0.13s) early."*

So the dancer knows: change the **form** vs change the **when**.

### 5. Simple Waveform for Timing

For each problem joint:
- Your angle curve vs. group curve
- Highlight regions where you're ahead (blue) or behind (orange)
- Click a spike → jump to that frame in the video

---

## Design Principles for "See It Fast, Fix It"

### 1. Lead with One Fix, Not Five

Don't overwhelm: *"You have 5 issues."*  
Instead: *"Fix left knee first — you're ~18° off and 4 frames early."*

### 2. Every Metric → One-Click Video Jump

Every number should have:
- **"See it"** — jump to the worst frame
- **"When"** — e.g., "worst around 1:23–1:28"

### 3. Human-Readable Thresholds

- "18° more bent" instead of "shape_error 18.2"
- "~0.13s early" instead of "4 frames ahead" (or both)
- Show the threshold: "20° = minor, 35° = major" so they know severity

### 4. One-Click Drill-Down

**Captain view:** Click row → drill into that dancer's detailed view  
**Dancer view:** Click metric → jump to video at worst moment

### 5. Color Consistency

Match the skeleton heatmap colors in the UI:
- **Green** = In sync
- **Blue** = Off-beat (timing)
- **Yellow** = Minor shape error
- **Red** = Major shape error
- **Gray** = Occluded / not evaluable

---

## TL;DR: Minimal "Instant Diagnosis" View

**Captain:**
- One-line squad summary
- Table: ID → worst problem + sync score
- One click → see that dancer at their worst moment

**Individual Dancer:**
- One line: *"Fix your left knee — you're ~18° off and a bit early."*
- "See it" button → video at worst timestamp
- Your pose vs. group pose overlay at that moment

---

All of this is derivable from the existing `deviations`, `shape_errors`, `timing_errors`, `worst_joints`, and `sync_score` — the improvement is in **how you surface it** and **how you link it to the video**.
