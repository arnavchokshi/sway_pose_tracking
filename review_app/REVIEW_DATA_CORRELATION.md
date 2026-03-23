# Human review ↔ pipeline data correlation

Generated from every text field in each review record (except `updated_at`), plus `metadata`, `track_summaries`, per-frame **`tracks`** counts from `data.json`, and `prune_log.json`.

- Review file: `/Users/arnavchokshi/Desktop/human_review_flight_batch (2).jsonl` (51 videos)
- Data root: `/Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/output/flight_batch`


---

## IMG_0256

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> excellent

- **problem_segments_sec[0].note:**

> Dancing comes in the right side and doesnt seem to be getting detected. Unsure why. I dont even see them in the dedup or pruned stages, they just never get tracked. This is strange because their entire body is visible and they are in the front. In previous versions of the model this person has gotten tracked.

- **ground_truth.expected_dancer_count:**

> 8 at the start, then 9 at the end

- **model_feedback.suspected_subsystems[0]:**

> pose

- **model_feedback.suspected_subsystem:**

> pose

- **model_feedback.what_went_wrong:**

> What went well: DI

- **model_feedback.what_went_well:**

> Did well at pruning the not moving person on the bottom right. Even though they are in much of the video, only their head and sometimes shoulders is shown which is god enough evidence that they should be pruned

- **manual_status:**

> done

### From `data.json`

- **num_frames:** 429 | **fps:** 29.837

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '2', '3', '4', '5', '6', '8', '9', '61']

- **Per-frame `tracks` count:** max **9** simultaneous, mean 7.68, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=428, `sync_score`=80.7
  - **2**: `frame_count`=387, `sync_score`=80.1
  - **3**: `frame_count`=407, `sync_score`=83.6
  - **4**: `frame_count`=400, `sync_score`=80.7
  - **5**: `frame_count`=425, `sync_score`=82.6
  - **6**: `frame_count`=423, `sync_score`=82.1
  - **8**: `frame_count`=364, `sync_score`=80.4
  - **9**: `frame_count`=414, `sync_score`=76.4
  - **61**: `frame_count`=45, `sync_score`=74.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 12 → `[1, 2, 3, 4, 5, 6, 8, 9, 40, 41, 58, 61]`

- **After pre-pose prune:** 9 → `[1, 2, 3, 4, 5, 6, 8, 9, 61]`

- **After post-pose prune:** 9 → `[1, 2, 3, 4, 5, 6, 8, 9, 61]`

- **Dropped before pose:** [40, 41, 58]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'audience_region': 1}

  - Track **40** → `duration/kinetic` (~4 frames)
  - Track **41** → `duration/kinetic` (~6 frames)
  - Track **58** → `audience_region` (~426 frames)

### Why the issues likely happened (comments + artifacts)

- **Never tracked** — compare **tracker ID list** in `prune_log` to expected count; if the person never appears there, failure is **YOLO recall** or **NMS** at source, not ViTPose.


---

## IMG_0870

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> There are 2 dancers all teh way in the front line (ID 7 and ID 1) whose pose estimation keeps going in and out. I dont know why this is happening, but they are very clearly visible throuhgout teh entire video, but their tracking isnt consistent.

- **ground_truth.expected_dancer_count:**

> 9 dancers always divisble

- **model_feedback.what_went_wrong:**

> What went well: Picked up

- **model_feedback.what_went_well:**

> Picked up dancers in teh back well even when they are covered

### From `data.json`

- **num_frames:** 774 | **fps:** 29.99

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '2', '3', '4', '5', '6', '7', '8', '9']

- **Per-frame `tracks` count:** max **9** simultaneous, mean 7.84, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=641, `sync_score`=70.8
  - **2**: `frame_count`=566, `sync_score`=76.0
  - **3**: `frame_count`=740, `sync_score`=79.4
  - **4**: `frame_count`=771, `sync_score`=81.0
  - **5**: `frame_count`=765, `sync_score`=81.6
  - **6**: `frame_count`=765, `sync_score`=81.1
  - **7**: `frame_count`=393, `sync_score`=78.8
  - **8**: `frame_count`=771, `sync_score`=80.9
  - **9**: `frame_count`=659, `sync_score`=78.2

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 11 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 22, 37]`

- **After pre-pose prune:** 9 → `[1, 2, 3, 4, 5, 6, 7, 8, 9]`

- **After post-pose prune:** 9 → `[1, 2, 3, 4, 5, 6, 7, 8, 9]`

- **Dropped before pose:** [22, 37]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 1, 'short_track': 1}

  - Track **22** → `duration/kinetic` (~3 frames)
  - Track **37** → `short_track` (~94 frames)

### Why the issues likely happened (comments + artifacts)

- **Pose flicker** — `frame_count` &lt; `num_frames` for a front-row ID (e.g. IMG_0870 ID 7) indicates **gaps in tracking boxes** or **visibility-gated pose skips**, not necessarily wrong final ID count.


---

## IMG_0871

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **affected_tracks_note:**

> ID 2 is on the right side of the screen and was wrongfully pruned. The dancer is fully visible throuhgout the peice so they shoudnt be pruned.

- **ground_truth.expected_dancer_count:**

> 9 dancers present in entire video, but somtimes covered making it okay if we only detect 8 at some points. (Should still detect between 8 and performers at all times)

- **notes_for_model:**

> If a dancer is in the "audience_region" we should do more checks before pruning them. We should first check the shape of their pose box. If it seems like they are standing for a decent amount (more than 40%) of the video they porbably shouldnt be pruned. Also we could check if their vitpose sketlton match closley wiht any other dancers, adn if it does, that also means they shouldnt be pruned.

### From `data.json`

- **num_frames:** 485 | **fps:** 29.997

- **Final exported tracks** (`track_summaries`): **8** — IDs ['1', '3', '4', '5', '6', '7', '8', '9']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 7.12, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=485, `sync_score`=75.4
  - **3**: `frame_count`=482, `sync_score`=72.6
  - **4**: `frame_count`=466, `sync_score`=70.9
  - **5**: `frame_count`=481, `sync_score`=75.0
  - **6**: `frame_count`=380, `sync_score`=74.0
  - **7**: `frame_count`=481, `sync_score`=75.3
  - **8**: `frame_count`=404, `sync_score`=74.1
  - **9**: `frame_count`=275, `sync_score`=70.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 11 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 34, 61]`

- **After pre-pose prune:** 8 → `[1, 3, 4, 5, 6, 7, 8, 9]`

- **After post-pose prune:** 8 → `[1, 3, 4, 5, 6, 7, 8, 9]`

- **Dropped before pose:** [2, 34, 61]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'audience_region': 1}

  - Track **34** → `duration/kinetic` (~1 frames)
  - Track **61** → `duration/kinetic` (~1 frames)
  - Track **2** → `audience_region` (~481 frames)

### Why the issues likely happened (comments + artifacts)

- **Audience-region prune** — `prune_by_stage_polygon` / `prune_audience_region` removed a track that the reviewer considers on-stage; see `audience_region` in prune log (e.g. IMG_0871 ID 2).


---

## IMG_0944

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> excellent

- **ground_truth.additional_ground_truth:**

> There are technically 4 people in this video, but the 4th person is directly behind ID2 making them not visible.

- **ground_truth_expectation:**

> There are technically 4 people in this video, but the 4th person is directly behind ID2 making them not visible.

### From `data.json`

- **num_frames:** 451 | **fps:** 29.917

- **Final exported tracks** (`track_summaries`): **3** — IDs ['1', '2', '3']

- **Per-frame `tracks` count:** max **3** simultaneous, mean 2.99, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=449, `sync_score`=84.0
  - **2**: `frame_count`=451, `sync_score`=83.0
  - **3**: `frame_count`=447, `sync_score`=85.4

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 4 → `[1, 2, 3, 7]`

- **After pre-pose prune:** 4 → `[1, 2, 3, 7]`

- **After post-pose prune:** 3 → `[1, 2, 3]`

- **Dropped after pose:** [7]

- **Prune log rules (may include diagnostics, not only final drops):** {}


### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [7] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_1305

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> poor

- **affected_tracks_note:**

> There is a person directly in the middle, who is not getting tracked. I see ID1 fails becuase of jitter. I asusme becaues the pose tracking is strulgging to lock onto a specific person. I propose that making the color of the person clothing more important to our tracking will help fix this.

### From `data.json`

- **num_frames:** 512 | **fps:** 29.985

- **Final exported tracks** (`track_summaries`): **8** — IDs ['2', '3', '5', '8', '9', '10', '66', '74']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 5.9, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=511, `sync_score`=73.2
  - **3**: `frame_count`=468, `sync_score`=74.4
  - **5**: `frame_count`=418, `sync_score`=73.6
  - **8**: `frame_count`=361, `sync_score`=75.6
  - **9**: `frame_count`=321, `sync_score`=71.8
  - **10**: `frame_count`=280, `sync_score`=64.4
  - **66**: `frame_count`=167, `sync_score`=74.0
  - **74**: `frame_count`=495, `sync_score`=72.2

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 15 → `[1, 2, 3, 5, 8, 9, 10, 11, 15, 44, 53, 66, 74, 79, 103]`

- **After pre-pose prune:** 11 → `[1, 2, 3, 5, 8, 9, 10, 15, 53, 66, 74]`

- **After post-pose prune:** 8 → `[2, 3, 5, 8, 9, 10, 66, 74]`

- **Dropped before pose:** [11, 44, 79, 103]

- **Dropped after pose:** [1, 15, 53]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 3, 'short_track': 1, 'jitter': 1}

  - Track **11** → `duration/kinetic` (~19 frames)
  - Track **44** → `duration/kinetic` (~39 frames)
  - Track **79** → `duration/kinetic` (~11 frames)
  - Track **103** → `short_track` (~59 frames)
  - Track **1** → `jitter` (~510 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [1, 15, 53] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Jitter / ID hopping** — BoT-SORT (and later crossover/dedup) **reassigns** the same numeric ID to different bodies; `prune_log` may record a **`jitter`** prune for that ID (e.g. IMG_7821 track 3) even when the viewer still sees confusion in earlier frames or in the rendered video.

- **Appearance** — reviewer wants stronger **color Re-ID**; pipeline already extracts **HSV embeddings** but may need **higher weight** or better features for costumes/lighting.


---

## IMG_1660

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> There are lots of overlapped people in this video so validly there are mistakes at this is hard.

- **model_feedback.what_went_wrong:**

> What went well: Correc

- **model_feedback.what_went_well:**

> Correctly pruned people who are far away and not relevant to the dance

- **notes_for_model:**

> I think we need to consider doing somehting about overlapped people. We will obviously get bad data is poeple overlap. Maybe in Sway we need to add a version thats light weight as can work live, so before the user presses record, Sway can warn the users that people are covering others and to move them. Similarly, if covering happens in the middle of the video we should have something in the UI that shows that the users got stacked so at that time we arent able to give accurate data.

### From `data.json`

- **num_frames:** 938 | **fps:** 29.923

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '2', '4', '5', '6', '7', '41', '143', '148']

- **Per-frame `tracks` count:** max **9** simultaneous, mean 7.44, **0.4%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=929, `sync_score`=73.9
  - **2**: `frame_count`=594, `sync_score`=70.8
  - **4**: `frame_count`=600, `sync_score`=72.9
  - **5**: `frame_count`=903, `sync_score`=78.1
  - **6**: `frame_count`=902, `sync_score`=79.9
  - **7**: `frame_count`=772, `sync_score`=78.9
  - **41**: `frame_count`=656, `sync_score`=71.9
  - **143**: `frame_count`=837, `sync_score`=78.9
  - **148**: `frame_count`=783, `sync_score`=69.9

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 20 → `[1, 2, 4, 5, 6, 7, 9, 10, 20, 33, 41, 55, 96, 132, 143, 148, 186, 237, 252, 291]`

- **After pre-pose prune:** 12 → `[1, 2, 4, 5, 6, 7, 10, 41, 55, 143, 148, 186]`

- **After post-pose prune:** 9 → `[1, 2, 4, 5, 6, 7, 41, 143, 148]`

- **Dropped before pose:** [9, 20, 33, 96, 132, 237, 252, 291]

- **Dropped after pose:** [10, 55, 186]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 6, 'spatial_outlier': 2}

  - Track **20** → `duration/kinetic` (~2 frames)
  - Track **33** → `duration/kinetic` (~6 frames)
  - Track **96** → `duration/kinetic` (~8 frames)
  - Track **237** → `duration/kinetic` (~1 frames)
  - Track **252** → `duration/kinetic` (~1 frames)
  - Track **291** → `duration/kinetic` (~8 frames)
  - Track **9** → `spatial_outlier` (~909 frames)
  - Track **132** → `spatial_outlier` (~327 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [10, 55, 186] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Close spacing** — high simultaneous `tracks` count with reviewer complaints about IDs usually points to **association errors**, **crossover OKS swaps**, or **deduplicate_collocated_poses**.


---

## IMG_1946

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> poor

- **affected_tracks_note:**

> Lots of overlap so i understand why our pipeline got this wrong.

- **notes_for_model:**

> The spatial outlier shouldnt have been pruned. Similar to how i stated as feedback in another video about audience pruning, we should check if it looks like the sptial outlier is stadning and doing moves similar to the other dancers.

### From `data.json`

- **num_frames:** 817 | **fps:** 29.912

- **Final exported tracks** (`track_summaries`): **12** — IDs ['1', '3', '4', '5', '6', '7', '8', '11', '12', '53', '120', '146']

- **Per-frame `tracks` count:** max **12** simultaneous, mean 8.65, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=809, `sync_score`=74.5
  - **3**: `frame_count`=815, `sync_score`=76.5
  - **4**: `frame_count`=736, `sync_score`=80.8
  - **5**: `frame_count`=517, `sync_score`=73.4
  - **6**: `frame_count`=806, `sync_score`=81.7
  - **7**: `frame_count`=817, `sync_score`=75.6
  - **8**: `frame_count`=809, `sync_score`=80.3
  - **11**: `frame_count`=390, `sync_score`=79.7
  - **12**: `frame_count`=566, `sync_score`=80.5
  - **53**: `frame_count`=412, `sync_score`=74.7
  - **120**: `frame_count`=202, `sync_score`=77.1
  - **146**: `frame_count`=184, `sync_score`=70.1

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 20 → `[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 49, 50, 52, 53, 82, 94, 120, 146, 179]`

- **After pre-pose prune:** 14 → `[1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 49, 53, 120, 146]`

- **After post-pose prune:** 12 → `[1, 3, 4, 5, 6, 7, 8, 11, 12, 53, 120, 146]`

- **Dropped before pose:** [2, 50, 52, 82, 94, 179]

- **Dropped after pose:** [10, 49]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 5, 'spatial_outlier': 1}

  - Track **50** → `duration/kinetic` (~23 frames)
  - Track **52** → `duration/kinetic` (~26 frames)
  - Track **82** → `duration/kinetic` (~1 frames)
  - Track **94** → `duration/kinetic` (~4 frames)
  - Track **179** → `duration/kinetic` (~26 frames)
  - Track **2** → `spatial_outlier` (~817 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [10, 49] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Audience-region prune** — `prune_by_stage_polygon` / `prune_audience_region` removed a track that the reviewer considers on-stage; see `audience_region` in prune log (e.g. IMG_0871 ID 2).

- **Close spacing** — high simultaneous `tracks` count with reviewer complaints about IDs usually points to **association errors**, **crossover OKS swaps**, or **deduplicate_collocated_poses**.


---

## IMG_2712

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> Again, lots of stacked poeple so you missed a lot of them, but thats understnadbale because they are hard to see.

### From `data.json`

- **num_frames:** 170 | **fps:** 29.634

- **Final exported tracks** (`track_summaries`): **7** — IDs ['1', '2', '3', '4', '5', '6', '8']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 5.26, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=163, `sync_score`=80.9
  - **2**: `frame_count`=168, `sync_score`=77.3
  - **3**: `frame_count`=167, `sync_score`=84.8
  - **4**: `frame_count`=19, `sync_score`=67.9
  - **5**: `frame_count`=137, `sync_score`=81.5
  - **6**: `frame_count`=145, `sync_score`=80.6
  - **8**: `frame_count`=96, `sync_score`=64.2

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 10 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 15]`

- **After pre-pose prune:** 8 → `[1, 2, 3, 4, 5, 6, 8, 9]`

- **After post-pose prune:** 7 → `[1, 2, 3, 4, 5, 6, 8]`

- **Dropped before pose:** [7, 15]

- **Dropped after pose:** [9]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 1, 'bbox_size': 1}

  - Track **15** → `duration/kinetic` (~8 frames)
  - Track **7** → `bbox_size` (~80 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [9] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_2921

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **affected_tracks_note:**

> Dancer all the way on the right is present in entire video and did not get tracked at all. Also ID5 keeps popping in and out but they shouldnt because they are fully visible thoruohg the video.

### From `data.json`

- **num_frames:** 283 | **fps:** 29.774

- **Final exported tracks** (`track_summaries`): **8** — IDs ['1', '2', '3', '5', '6', '7', '8', '9']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 5.84, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=283, `sync_score`=77.6
  - **2**: `frame_count`=181, `sync_score`=68.9
  - **3**: `frame_count`=282, `sync_score`=73.4
  - **5**: `frame_count`=61, `sync_score`=75.3
  - **6**: `frame_count`=187, `sync_score`=72.3
  - **7**: `frame_count`=283, `sync_score`=79.7
  - **8**: `frame_count`=281, `sync_score`=79.6
  - **9**: `frame_count`=96, `sync_score`=81.0

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 10 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 26]`

- **After pre-pose prune:** 8 → `[1, 2, 3, 5, 6, 7, 8, 9]`

- **After post-pose prune:** 8 → `[1, 2, 3, 5, 6, 7, 8, 9]`

- **Dropped before pose:** [4, 26]

- **Prune log rules (may include diagnostics, not only final drops):** {'spatial_outlier': 1, 'bbox_size': 1}

  - Track **4** → `spatial_outlier` (~283 frames)
  - Track **26** → `bbox_size` (~59 frames)

### Why the issues likely happened (comments + artifacts)

- **Pose flicker** — `frame_count` &lt; `num_frames` for a front-row ID (e.g. IMG_0870 ID 7) indicates **gaps in tracking boxes** or **visibility-gated pose skips**, not necessarily wrong final ID count.


---

## IMG_2942

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> ID 22 is sitting down and not doing anything. They shouldve been pruned

### From `data.json`

- **num_frames:** 525 | **fps:** 29.929

- **Final exported tracks** (`track_summaries`): **14** — IDs ['2', '3', '6', '7', '8', '9', '10', '22', '103', '109', '128', '154', '186', '215']

- **Per-frame `tracks` count:** max **11** simultaneous, mean 7.03, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=504, `sync_score`=65.6
  - **3**: `frame_count`=467, `sync_score`=68.5
  - **6**: `frame_count`=178, `sync_score`=73.2
  - **7**: `frame_count`=522, `sync_score`=73.9
  - **8**: `frame_count`=455, `sync_score`=70.4
  - **9**: `frame_count`=187, `sync_score`=81.4
  - **10**: `frame_count`=111, `sync_score`=75.0
  - **22**: `frame_count`=369, `sync_score`=41.5
  - **103**: `frame_count`=117, `sync_score`=69.0
  - **109**: `frame_count`=85, `sync_score`=67.7
  - **128**: `frame_count`=124, `sync_score`=63.0
  - **154**: `frame_count`=380, `sync_score`=71.5
  - **186**: `frame_count`=88, `sync_score`=72.1
  - **215**: `frame_count`=104, `sync_score`=63.7

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 29 → `[1, 2, 3, 4, 6, 7, 8, 9, 10, 14, 18, 22, 28, 32, 52, 99, 102, 103, 109, 128, 129, 143, 154, 176, 186, 192, 196, 215, 235]`

- **After pre-pose prune:** 18 → `[2, 3, 6, 7, 8, 9, 10, 18, 22, 32, 103, 109, 128, 143, 154, 176, 186, 215]`

- **After post-pose prune:** 14 → `[2, 3, 6, 7, 8, 9, 10, 22, 103, 109, 128, 154, 186, 215]`

- **Dropped before pose:** [1, 4, 14, 28, 52, 99, 102, 129, 192, 196, 235]

- **Dropped after pose:** [18, 32, 143, 176]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 6, 'spatial_outlier': 3, 'short_track': 1, 'audience_region': 1, 'low_sync': 1, 'low_confidence': 1}

  - Track **14** → `duration/kinetic` (~18 frames)
  - Track **28** → `duration/kinetic` (~1 frames)
  - Track **52** → `duration/kinetic` (~1 frames)
  - Track **99** → `duration/kinetic` (~4 frames)
  - Track **192** → `duration/kinetic` (~1 frames)
  - Track **235** → `duration/kinetic` (~1 frames)
  - Track **4** → `spatial_outlier` (~257 frames)
  - Track **102** → `spatial_outlier` (~83 frames)
  - Track **129** → `spatial_outlier` (~235 frames)
  - Track **196** → `short_track` (~69 frames)
  - Track **1** → `audience_region` (~504 frames)
  - Track **176** → `low_sync` (~242 frames)
  - Track **176** → `low_confidence` (~242 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [18, 32, 143, 176] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_2946

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> ID 8 is assume is being pruned beacuse it keeps moving between 3 people who are closely next to each other. It sees to keep moving around them. Although there are many times in which these dancers are far enouhg appart to where all of their boxes wouldnt even overlap, so im confused why all 3 didnt get their own boxes.

- **model_feedback.what_went_wrong:**

> What went well: Gopd

- **model_feedback.what_went_well:**

> Good job at pruning ID 6. THey ar ea person walking at the start and are not dancing. ALso yes ID 30 and 39 are correctly pruned.

- **notes_for_model:**

> We should put more priority on clothing color to help with keeping ID consistnet. Also need to igiure out why these 3 dancers on the right side all ended up having ID 8 flop between them, even thouhg they are decetnyl far aparat enough to get their own boxes.

### From `data.json`

- **num_frames:** 411 | **fps:** 29.844

- **Final exported tracks** (`track_summaries`): **7** — IDs ['1', '2', '3', '7', '9', '44', '63']

- **Per-frame `tracks` count:** max **6** simultaneous, mean 4.91, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=407, `sync_score`=76.7
  - **2**: `frame_count`=264, `sync_score`=73.3
  - **3**: `frame_count`=155, `sync_score`=69.2
  - **7**: `frame_count`=411, `sync_score`=80.0
  - **9**: `frame_count`=317, `sync_score`=79.2
  - **44**: `frame_count`=411, `sync_score`=77.6
  - **63**: `frame_count`=54, `sync_score`=76.8

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 12 → `[1, 2, 3, 6, 7, 8, 9, 10, 30, 39, 44, 63]`

- **After pre-pose prune:** 9 → `[1, 2, 3, 7, 8, 9, 10, 44, 63]`

- **After post-pose prune:** 7 → `[1, 2, 3, 7, 9, 44, 63]`

- **Dropped before pose:** [6, 30, 39]

- **Dropped after pose:** [8, 10]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 3, 'jitter': 1}

  - Track **6** → `duration/kinetic` (~10 frames)
  - Track **30** → `duration/kinetic` (~49 frames)
  - Track **39** → `duration/kinetic` (~2 frames)
  - Track **8** → `jitter` (~385 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [8, 10] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Close spacing** — high simultaneous `tracks` count with reviewer complaints about IDs usually points to **association errors**, **crossover OKS swaps**, or **deduplicate_collocated_poses**.

- **Appearance** — reviewer wants stronger **color Re-ID**; pipeline already extracts **HSV embeddings** but may need **higher weight** or better features for costumes/lighting.


---

## IMG_3070

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> Very confused on why ID 3 was pruned as part of being a spatial outlier. They are basically in the middle of the video for most of it and in the front. THey also do the same choreo as eveyrone else. So this was a very bad choice to prune ID 3. Same thing with ID 179, they are also not a spacial outlier.
> 
> JItter on ID 6 happened beacuse it swithced people half way though. Again, if we keep track of color of person we can avoid this.

- **notes_for_model:**

> Over all the way spatial outlier is done needs to be dramatically changed.

### From `data.json`

- **num_frames:** 601 | **fps:** 29.938

- **Final exported tracks** (`track_summaries`): **15** — IDs ['1', '2', '4', '5', '7', '8', '9', '11', '12', '13', '22', '145', '185', '300', '301']

- **Per-frame `tracks` count:** max **12** simultaneous, mean 8.61, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=534, `sync_score`=64.0
  - **2**: `frame_count`=387, `sync_score`=73.7
  - **4**: `frame_count`=585, `sync_score`=70.7
  - **5**: `frame_count`=148, `sync_score`=74.3
  - **7**: `frame_count`=100, `sync_score`=78.0
  - **8**: `frame_count`=517, `sync_score`=75.5
  - **9**: `frame_count`=394, `sync_score`=67.7
  - **11**: `frame_count`=569, `sync_score`=75.6
  - **12**: `frame_count`=571, `sync_score`=74.7
  - **13**: `frame_count`=401, `sync_score`=68.7
  - **22**: `frame_count`=356, `sync_score`=75.6
  - **145**: `frame_count`=139, `sync_score`=55.0
  - **185**: `frame_count`=232, `sync_score`=68.2
  - **300**: `frame_count`=124, `sync_score`=57.9
  - **301**: `frame_count`=116, `sync_score`=55.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 30 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 22, 39, 48, 67, 70, 87, 145, 179, 180, 185, 198, 222, 257, 289, 300, 301, 314, 359]`

- **After pre-pose prune:** 18 → `[1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 22, 48, 67, 145, 185, 300, 301]`

- **After post-pose prune:** 15 → `[1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 22, 145, 185, 300, 301]`

- **Dropped before pose:** [3, 39, 70, 87, 179, 180, 198, 222, 257, 289, 314, 359]

- **Dropped after pose:** [6, 48, 67]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 10, 'spatial_outlier': 2, 'jitter': 1}

  - Track **39** → `duration/kinetic` (~2 frames)
  - Track **70** → `duration/kinetic` (~77 frames)
  - Track **87** → `duration/kinetic` (~4 frames)
  - Track **180** → `duration/kinetic` (~21 frames)
  - Track **198** → `duration/kinetic` (~10 frames)
  - Track **222** → `duration/kinetic` (~2 frames)
  - Track **257** → `duration/kinetic` (~6 frames)
  - Track **289** → `duration/kinetic` (~2 frames)
  - Track **314** → `duration/kinetic` (~1 frames)
  - Track **359** → `duration/kinetic` (~1 frames)
  - Track **3** → `spatial_outlier` (~594 frames)
  - Track **179** → `spatial_outlier` (~345 frames)
  - Track **6** → `jitter` (~596 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [6, 48, 67] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Jitter / ID hopping** — BoT-SORT (and later crossover/dedup) **reassigns** the same numeric ID to different bodies; `prune_log` may record a **`jitter`** prune for that ID (e.g. IMG_7821 track 3) even when the viewer still sees confusion in earlier frames or in the rendered video.

- **Appearance** — reviewer wants stronger **color Re-ID**; pipeline already extracts **HSV embeddings** but may need **higher weight** or better features for costumes/lighting.


---

## IMG_3072

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **affected_tracks_note:**

> I see ID 8 in the back, but thats a chair.... Im confused on how both the models thought that the chair was a person with enough confidence to not prune.

- **notes_for_model:**

> We should check to make sure both models give reasonable confidence values for the pose estimation otherwise prune them, becuase in this it thinks ID 8 is a person when its a chair

- **manual_status:**

> done

### From `data.json`

- **num_frames:** 539 | **fps:** 29.931

- **Final exported tracks** (`track_summaries`): **13** — IDs ['1', '2', '3', '5', '6', '7', '8', '9', '11', '13', '47', '233', '247']

- **Per-frame `tracks` count:** max **12** simultaneous, mean 8.59, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=511, `sync_score`=66.6
  - **2**: `frame_count`=517, `sync_score`=74.9
  - **3**: `frame_count`=461, `sync_score`=74.4
  - **5**: `frame_count`=527, `sync_score`=72.5
  - **6**: `frame_count`=432, `sync_score`=66.2
  - **7**: `frame_count`=197, `sync_score`=73.0
  - **8**: `frame_count`=223, `sync_score`=67.1
  - **9**: `frame_count`=322, `sync_score`=71.2
  - **11**: `frame_count`=511, `sync_score`=71.8
  - **13**: `frame_count`=432, `sync_score`=70.7
  - **47**: `frame_count`=233, `sync_score`=73.5
  - **233**: `frame_count`=131, `sync_score`=77.3
  - **247**: `frame_count`=132, `sync_score`=66.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 29 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 19, 44, 47, 70, 95, 116, 142, 167, 200, 205, 220, 230, 233, 247, 268, 279, 288, 324]`

- **After pre-pose prune:** 18 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 19, 44, 47, 70, 230, 233, 247]`

- **After post-pose prune:** 13 → `[1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 47, 233, 247]`

- **Dropped before pose:** [95, 116, 142, 167, 200, 205, 220, 268, 279, 288, 324]

- **Dropped after pose:** [4, 19, 44, 70, 230]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 10, 'spatial_outlier': 1, 'jitter': 2}

  - Track **95** → `duration/kinetic` (~77 frames)
  - Track **116** → `duration/kinetic` (~2 frames)
  - Track **142** → `duration/kinetic` (~2 frames)
  - Track **167** → `duration/kinetic` (~6 frames)
  - Track **200** → `duration/kinetic` (~23 frames)
  - Track **220** → `duration/kinetic` (~6 frames)
  - Track **268** → `duration/kinetic` (~20 frames)
  - Track **279** → `duration/kinetic` (~1 frames)
  - Track **288** → `duration/kinetic` (~10 frames)
  - Track **324** → `duration/kinetic` (~1 frames)
  - Track **205** → `spatial_outlier` (~238 frames)
  - Track **4** → `jitter` (~197 frames)
  - Track **19** → `jitter` (~446 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [4, 19, 44, 70, 230] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Non-person boxes** — YOLO `person` class fires on props/layout; tracks survive until **duration/kinetic**, **sync**, or **bbox** rules cut them. Low **ViTPose sync** (e.g. chair) often keeps them unless post-pose pruning catches them.


---

## IMG_3331

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 2696 | **fps:** 29.97

- **Final exported tracks** (`track_summaries`): **11** — IDs ['2', '4', '5', '8', '38', '137', '139', '173', '203', '335', '649']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 5.84, **0.4%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=2328, `sync_score`=74.9
  - **4**: `frame_count`=1712, `sync_score`=70.5
  - **5**: `frame_count`=2484, `sync_score`=74.1
  - **8**: `frame_count`=1617, `sync_score`=73.2
  - **38**: `frame_count`=723, `sync_score`=72.0
  - **137**: `frame_count`=1564, `sync_score`=71.6
  - **139**: `frame_count`=2307, `sync_score`=72.2
  - **173**: `frame_count`=1157, `sync_score`=75.9
  - **203**: `frame_count`=453, `sync_score`=77.4
  - **335**: `frame_count`=592, `sync_score`=67.5
  - **649**: `frame_count`=795, `sync_score`=67.1

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 20 → `[2, 4, 5, 8, 12, 38, 137, 139, 173, 203, 243, 307, 335, 638, 649, 667, 676, 693, 708, 730]`

- **After pre-pose prune:** 13 → `[2, 4, 5, 8, 12, 38, 137, 139, 173, 203, 307, 335, 649]`

- **After post-pose prune:** 11 → `[2, 4, 5, 8, 38, 137, 139, 173, 203, 335, 649]`

- **Dropped before pose:** [243, 638, 667, 676, 693, 708, 730]

- **Dropped after pose:** [12, 307]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 4, 'spatial_outlier': 1, 'short_track': 2}

  - Track **243** → `duration/kinetic` (~233 frames)
  - Track **638** → `duration/kinetic` (~1 frames)
  - Track **676** → `duration/kinetic` (~2 frames)
  - Track **708** → `duration/kinetic` (~1 frames)
  - Track **667** → `spatial_outlier` (~419 frames)
  - Track **693** → `short_track` (~183 frames)
  - Track **730** → `short_track` (~201 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [12, 307] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3396

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **affected_tracks_note:**

> ID 1 is not a spatial outlier.

### From `data.json`

- **num_frames:** 552 | **fps:** 29.878

- **Final exported tracks** (`track_summaries`): **10** — IDs ['2', '3', '4', '5', '6', '7', '8', '10', '11', '12']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 8.11, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=549, `sync_score`=74.7
  - **3**: `frame_count`=548, `sync_score`=72.2
  - **4**: `frame_count`=278, `sync_score`=74.3
  - **5**: `frame_count`=552, `sync_score`=69.3
  - **6**: `frame_count`=512, `sync_score`=74.0
  - **7**: `frame_count`=478, `sync_score`=74.0
  - **8**: `frame_count`=485, `sync_score`=72.6
  - **10**: `frame_count`=218, `sync_score`=70.5
  - **11**: `frame_count`=329, `sync_score`=77.8
  - **12**: `frame_count`=529, `sync_score`=65.0

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 17 → `[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 54, 67, 99, 130, 167]`

- **After pre-pose prune:** 11 → `[2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14]`

- **After post-pose prune:** 10 → `[2, 3, 4, 5, 6, 7, 8, 10, 11, 12]`

- **Dropped before pose:** [1, 54, 67, 99, 130, 167]

- **Dropped after pose:** [14]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 5, 'spatial_outlier': 1}

  - Track **54** → `duration/kinetic` (~1 frames)
  - Track **67** → `duration/kinetic` (~1 frames)
  - Track **99** → `duration/kinetic` (~1 frames)
  - Track **130** → `duration/kinetic` (~5 frames)
  - Track **167** → `duration/kinetic` (~3 frames)
  - Track **1** → `spatial_outlier` (~550 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [14] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3410

### Review comments (all text fields except `updated_at`)

- **affected_tracks_note:**

> ID 1 and 8 are not spatial outliers.

- **model_feedback.what_went_wrong:**

> What went well: I

### From `data.json`

- **num_frames:** 427 | **fps:** 29.843

- **Final exported tracks** (`track_summaries`): **11** — IDs ['2', '3', '4', '7', '9', '15', '25', '108', '119', '147', '198']

- **Per-frame `tracks` count:** max **11** simultaneous, mean 7.49, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=427, `sync_score`=64.0
  - **3**: `frame_count`=402, `sync_score`=58.6
  - **4**: `frame_count`=405, `sync_score`=71.6
  - **7**: `frame_count`=292, `sync_score`=71.6
  - **9**: `frame_count`=287, `sync_score`=60.6
  - **15**: `frame_count`=339, `sync_score`=69.2
  - **25**: `frame_count`=259, `sync_score`=74.1
  - **108**: `frame_count`=59, `sync_score`=64.1
  - **119**: `frame_count`=86, `sync_score`=57.3
  - **147**: `frame_count`=216, `sync_score`=64.3
  - **198**: `frame_count`=426, `sync_score`=73.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 22 → `[1, 2, 3, 4, 6, 7, 8, 9, 15, 25, 32, 45, 72, 76, 108, 119, 147, 154, 198, 241, 305, 314]`

- **After pre-pose prune:** 14 → `[2, 3, 4, 6, 7, 9, 15, 25, 32, 45, 108, 119, 147, 198]`

- **After post-pose prune:** 11 → `[2, 3, 4, 7, 9, 15, 25, 108, 119, 147, 198]`

- **Dropped before pose:** [1, 8, 72, 76, 154, 241, 305, 314]

- **Dropped after pose:** [6, 32, 45]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 6, 'spatial_outlier': 2, 'jitter': 1}

  - Track **72** → `duration/kinetic` (~1 frames)
  - Track **76** → `duration/kinetic` (~1 frames)
  - Track **154** → `duration/kinetic` (~33 frames)
  - Track **241** → `duration/kinetic` (~4 frames)
  - Track **305** → `duration/kinetic` (~1 frames)
  - Track **314** → `duration/kinetic` (~1 frames)
  - Track **1** → `spatial_outlier` (~414 frames)
  - Track **8** → `spatial_outlier` (~427 frames)
  - Track **6** → `jitter` (~381 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [6, 32, 45] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3427

### Review comments (all text fields except `updated_at`)

- **affected_tracks_note:**

> ID 1 is not a spatial outlier and should not have been pruned

- **model_feedback.what_went_wrong:**

> What went well: Good

- **model_feedback.what_went_well:**

> Good job at not pruning ID 2. They are on the ground for a lot of the time and not moving making it seem like they might be an audicen member, but half way thru they get up and they are also closer to teh center of the video giving a good indicator that they are dancing.

### From `data.json`

- **num_frames:** 593 | **fps:** 29.887

- **Final exported tracks** (`track_summaries`): **9** — IDs ['2', '3', '4', '5', '6', '7', '9', '15', '22']

- **Per-frame `tracks` count:** max **9** simultaneous, mean 7.13, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=592, `sync_score`=64.9
  - **3**: `frame_count`=591, `sync_score`=80.7
  - **4**: `frame_count`=499, `sync_score`=76.6
  - **5**: `frame_count`=568, `sync_score`=76.2
  - **6**: `frame_count`=592, `sync_score`=74.3
  - **7**: `frame_count`=493, `sync_score`=76.1
  - **9**: `frame_count`=464, `sync_score`=78.0
  - **15**: `frame_count`=93, `sync_score`=80.3
  - **22**: `frame_count`=336, `sync_score`=79.6

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 13 → `[1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 22, 42, 87]`

- **After pre-pose prune:** 10 → `[2, 3, 4, 5, 6, 7, 9, 11, 15, 22]`

- **After post-pose prune:** 9 → `[2, 3, 4, 5, 6, 7, 9, 15, 22]`

- **Dropped before pose:** [1, 42, 87]

- **Dropped after pose:** [11]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'spatial_outlier': 1}

  - Track **42** → `duration/kinetic` (~1 frames)
  - Track **87** → `duration/kinetic` (~1 frames)
  - Track **1** → `spatial_outlier` (~530 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [11] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3463

### Review comments (all text fields except `updated_at`)

- **affected_tracks_note:**

> ID 15 (correctly pruned) is just tracking a shoe lol. How did our pose tracking lock onto JUST a shoe??
> 
> ID 26 also started as staying tracked on a shoe, and then it moved aroudn between people a lot.

### From `data.json`

- **num_frames:** 1160 | **fps:** 29.934

- **Final exported tracks** (`track_summaries`): **11** — IDs ['1', '3', '4', '5', '6', '7', '8', '9', '10', '31', '108']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 6.79, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=847, `sync_score`=73.4
  - **3**: `frame_count`=1090, `sync_score`=73.0
  - **4**: `frame_count`=1149, `sync_score`=74.9
  - **5**: `frame_count`=80, `sync_score`=68.9
  - **6**: `frame_count`=551, `sync_score`=76.6
  - **7**: `frame_count`=394, `sync_score`=72.8
  - **8**: `frame_count`=1069, `sync_score`=74.2
  - **9**: `frame_count`=1030, `sync_score`=72.6
  - **10**: `frame_count`=647, `sync_score`=74.6
  - **31**: `frame_count`=621, `sync_score`=69.9
  - **108**: `frame_count`=401, `sync_score`=68.0

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 27 → `[1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 26, 31, 39, 81, 108, 129, 164, 196, 305, 317, 335, 341, 342, 403, 444]`

- **After pre-pose prune:** 15 → `[1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 26, 31, 108, 196, 305]`

- **After post-pose prune:** 11 → `[1, 3, 4, 5, 6, 7, 8, 9, 10, 31, 108]`

- **Dropped before pose:** [13, 15, 39, 81, 129, 164, 317, 335, 341, 342, 403, 444]

- **Dropped after pose:** [12, 26, 196, 305]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 11, 'spatial_outlier': 1, 'jitter': 1}

  - Track **13** → `duration/kinetic` (~3 frames)
  - Track **39** → `duration/kinetic` (~1 frames)
  - Track **81** → `duration/kinetic` (~3 frames)
  - Track **129** → `duration/kinetic` (~2 frames)
  - Track **164** → `duration/kinetic` (~5 frames)
  - Track **317** → `duration/kinetic` (~5 frames)
  - Track **335** → `duration/kinetic` (~1 frames)
  - Track **341** → `duration/kinetic` (~3 frames)
  - Track **342** → `duration/kinetic` (~30 frames)
  - Track **403** → `duration/kinetic` (~162 frames)
  - Track **444** → `duration/kinetic` (~7 frames)
  - Track **15** → `spatial_outlier` (~1126 frames)
  - Track **26** → `jitter` (~1146 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [12, 26, 196, 305] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3528

### Review comments (all text fields except `updated_at`)

- **affected_tracks_note:**

> ID 5 should not have been pruned. We can see their full body and their movemnets match the other dancers.

- **model_feedback.what_went_wrong:**

> What went well: Good job at pruning ID

- **model_feedback.what_went_well:**

> Good job at pruning ID 64.

### From `data.json`

- **num_frames:** 288 | **fps:** 29.778

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '2', '3', '4', '6', '8', '9', '10', '67']

- **Per-frame `tracks` count:** max **9** simultaneous, mean 7.59, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=287, `sync_score`=87.7
  - **2**: `frame_count`=287, `sync_score`=81.6
  - **3**: `frame_count`=288, `sync_score`=88.1
  - **4**: `frame_count`=288, `sync_score`=84.3
  - **6**: `frame_count`=286, `sync_score`=85.6
  - **8**: `frame_count`=154, `sync_score`=87.4
  - **9**: `frame_count`=280, `sync_score`=87.0
  - **10**: `frame_count`=288, `sync_score`=86.6
  - **67**: `frame_count`=28, `sync_score`=75.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 20 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 20, 21, 24, 30, 43, 64, 67]`

- **After pre-pose prune:** 13 → `[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 30, 64, 67]`

- **After post-pose prune:** 9 → `[1, 2, 3, 4, 6, 8, 9, 10, 67]`

- **Dropped before pose:** [5, 13, 16, 20, 21, 24, 43]

- **Dropped after pose:** [7, 11, 30, 64]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 5, 'audience_region': 1, 'bbox_size': 1, 'low_sync': 1, 'completeness': 1, 'jitter': 1}

  - Track **16** → `duration/kinetic` (~1 frames)
  - Track **20** → `duration/kinetic` (~40 frames)
  - Track **21** → `duration/kinetic` (~2 frames)
  - Track **24** → `duration/kinetic` (~9 frames)
  - Track **43** → `duration/kinetic` (~2 frames)
  - Track **5** → `audience_region` (~286 frames)
  - Track **13** → `bbox_size` (~222 frames)
  - Track **7** → `low_sync` (~263 frames)
  - Track **64** → `completeness` (~12 frames)
  - Track **11** → `jitter` (~185 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [7, 11, 30, 64] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3555

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> ID 6 at the left cornor is very obvioulsy not a dancer and should be pruned. You ahve esitmated them to be on the floor adn lying down for the entire video whihc is the opposite of what all the other dancers are doing.
> 
> Also ID 10 in the back is a window reflection, but im unsure how you would eveen be able to get rid of that. Just take note of that but disregard it.

- **ground_truth.additional_ground_truth:**

> Throughout the video ther should be 5 people being trakced. 4 main in teh middle, anda. 5th of the left side.
> 
> There are an aiddiotnal 2 people on teh floor but they arent dancing and should be pruned.
> 
> When rerunning this video we should only see 5 people present.

- **ground_truth_expectation:**

> Throughout the video ther should be 5 people being trakced. 4 main in teh middle, anda. 5th of the left side.
> 
> There are an aiddiotnal 2 people on teh floor but they arent dancing and should be pruned.
> 
> When rerunning this video we should only see 5 people present.

- **model_feedback.what_went_wrong:**

> What went well: The main pai

- **model_feedback.what_went_well:**

> The main peope ID 2, 3, 1, 4, 5 are very well being posed.

### From `data.json`

- **num_frames:** 388 | **fps:** 29.831

- **Final exported tracks** (`track_summaries`): **7** — IDs ['1', '2', '3', '4', '5', '6', '10']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 6.36, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=388, `sync_score`=74.7
  - **2**: `frame_count`=388, `sync_score`=66.1
  - **3**: `frame_count`=359, `sync_score`=73.5
  - **4**: `frame_count`=346, `sync_score`=77.4
  - **5**: `frame_count`=361, `sync_score`=73.4
  - **6**: `frame_count`=386, `sync_score`=41.5
  - **10**: `frame_count`=241, `sync_score`=70.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 11 → `[1, 2, 3, 4, 5, 6, 10, 15, 32, 37, 58]`

- **After pre-pose prune:** 7 → `[1, 2, 3, 4, 5, 6, 10]`

- **After post-pose prune:** 7 → `[1, 2, 3, 4, 5, 6, 10]`

- **Dropped before pose:** [15, 32, 37, 58]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 4}

  - Track **15** → `duration/kinetic` (~4 frames)
  - Track **32** → `duration/kinetic` (~6 frames)
  - Track **37** → `duration/kinetic` (~13 frames)
  - Track **58** → `duration/kinetic` (~2 frames)

### Why the issues likely happened (comments + artifacts)

- Compare **expected dancer count** in the review to **final track count** and **max simultaneous tracks**; use `prune_log` drops to see which subsystem removed each ID.


---

## IMG_3556

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **problem_segments_sec[0].note:**

> ID 6 is being deduped but shouldnt be. They are clearly shown and unubscured so we should just see their ID and pose estiamtion like normal here.

- **affected_tracks_note:**

> ID 32 should be pruned. They are on teh side and clearly not doing the same movements the poeple int eh center are doing.
> 
> Also the ID 8 in teh back is a widnow reflection

- **model_feedback.what_went_wrong:**

> What went well: Good job at

- **model_feedback.what_went_well:**

> Good job at pruning ID 7, they are far away and not dancing. Also the tracking is decently good on the main member ID 6 2 3 1.

### From `data.json`

- **num_frames:** 378 | **fps:** 29.83

- **Final exported tracks** (`track_summaries`): **6** — IDs ['1', '2', '3', '6', '8', '32']

- **Per-frame `tracks` count:** max **6** simultaneous, mean 5.0, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=374, `sync_score`=77.4
  - **2**: `frame_count`=378, `sync_score`=77.8
  - **3**: `frame_count`=344, `sync_score`=78.2
  - **6**: `frame_count`=277, `sync_score`=76.1
  - **8**: `frame_count`=144, `sync_score`=71.4
  - **32**: `frame_count`=372, `sync_score`=63.9

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 9 → `[1, 2, 3, 6, 7, 8, 26, 32, 60]`

- **After pre-pose prune:** 6 → `[1, 2, 3, 6, 8, 32]`

- **After post-pose prune:** 6 → `[1, 2, 3, 6, 8, 32]`

- **Dropped before pose:** [7, 26, 60]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'spatial_outlier': 1}

  - Track **26** → `duration/kinetic` (~8 frames)
  - Track **60** → `duration/kinetic` (~20 frames)
  - Track **7** → `spatial_outlier` (~326 frames)

### Why the issues likely happened (comments + artifacts)

- Compare **expected dancer count** in the review to **final track count** and **max simultaneous tracks**; use `prune_log` drops to see which subsystem removed each ID.


---

## IMG_3560

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> ID 3 should have been pruned, they are on teh side adn not doing same movememnts as everyone else.

- **model_feedback.what_went_wrong:**

> What went well: Tracking on ID

- **model_feedback.what_went_well:**

> Tracking on ID 4 5 and 2 is really good

### From `data.json`

- **num_frames:** 452 | **fps:** 29.851

- **Final exported tracks** (`track_summaries`): **4** — IDs ['2', '3', '4', '5']

- **Per-frame `tracks` count:** max **4** simultaneous, mean 3.92, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=451, `sync_score`=77.3
  - **3**: `frame_count`=422, `sync_score`=69.2
  - **4**: `frame_count`=452, `sync_score`=76.8
  - **5**: `frame_count`=448, `sync_score`=77.8

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 11 → `[2, 3, 4, 5, 7, 8, 9, 11, 12, 26, 42]`

- **After pre-pose prune:** 4 → `[2, 3, 4, 5]`

- **After post-pose prune:** 4 → `[2, 3, 4, 5]`

- **Dropped before pose:** [7, 8, 9, 11, 12, 26, 42]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 6, 'aspect_ratio': 1}

  - Track **8** → `duration/kinetic` (~1 frames)
  - Track **9** → `duration/kinetic` (~1 frames)
  - Track **11** → `duration/kinetic` (~1 frames)
  - Track **12** → `duration/kinetic` (~92 frames)
  - Track **26** → `duration/kinetic` (~1 frames)
  - Track **42** → `duration/kinetic` (~3 frames)
  - Track **7** → `aspect_ratio` (~413 frames)

### Why the issues likely happened (comments + artifacts)

- Compare **expected dancer count** in the review to **final track count** and **max simultaneous tracks**; use `prune_log` drops to see which subsystem removed each ID.


---

## IMG_3577

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 1509 | **fps:** 29.935

- **Final exported tracks** (`track_summaries`): **11** — IDs ['2', '3', '4', '6', '7', '8', '54', '119', '221', '243', '332']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 7.33, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=1451, `sync_score`=71.5
  - **3**: `frame_count`=1427, `sync_score`=79.3
  - **4**: `frame_count`=1489, `sync_score`=81.2
  - **6**: `frame_count`=1483, `sync_score`=78.0
  - **7**: `frame_count`=897, `sync_score`=72.6
  - **8**: `frame_count`=1448, `sync_score`=79.8
  - **54**: `frame_count`=755, `sync_score`=76.0
  - **119**: `frame_count`=1072, `sync_score`=70.8
  - **221**: `frame_count`=266, `sync_score`=69.4
  - **243**: `frame_count`=433, `sync_score`=76.0
  - **332**: `frame_count`=346, `sync_score`=73.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 24 → `[2, 3, 4, 5, 6, 7, 8, 15, 45, 48, 54, 82, 93, 101, 119, 158, 160, 205, 221, 242, 243, 297, 332, 336]`

- **After pre-pose prune:** 14 → `[2, 3, 4, 6, 7, 8, 54, 101, 119, 158, 221, 242, 243, 332]`

- **After post-pose prune:** 11 → `[2, 3, 4, 6, 7, 8, 54, 119, 221, 243, 332]`

- **Dropped before pose:** [5, 15, 45, 48, 82, 93, 160, 205, 297, 336]

- **Dropped after pose:** [101, 158, 242]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 7, 'spatial_outlier': 2, 'short_track': 1}

  - Track **15** → `duration/kinetic` (~39 frames)
  - Track **48** → `duration/kinetic` (~123 frames)
  - Track **82** → `duration/kinetic` (~2 frames)
  - Track **93** → `duration/kinetic` (~1 frames)
  - Track **160** → `duration/kinetic` (~1 frames)
  - Track **205** → `duration/kinetic` (~2 frames)
  - Track **336** → `duration/kinetic` (~1 frames)
  - Track **5** → `spatial_outlier` (~429 frames)
  - Track **45** → `spatial_outlier` (~1197 frames)
  - Track **297** → `short_track` (~99 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [101, 158, 242] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3839

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 354 | **fps:** 29.819

- **Final exported tracks** (`track_summaries`): **12** — IDs ['1', '2', '3', '5', '6', '8', '9', '10', '20', '22', '28', '39']

- **Per-frame `tracks` count:** max **11** simultaneous, mean 9.18, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=353, `sync_score`=74.6
  - **2**: `frame_count`=353, `sync_score`=76.8
  - **3**: `frame_count`=352, `sync_score`=75.3
  - **5**: `frame_count`=344, `sync_score`=78.1
  - **6**: `frame_count`=342, `sync_score`=72.2
  - **8**: `frame_count`=333, `sync_score`=75.0
  - **9**: `frame_count`=49, `sync_score`=72.1
  - **10**: `frame_count`=284, `sync_score`=77.9
  - **20**: `frame_count`=272, `sync_score`=77.0
  - **22**: `frame_count`=354, `sync_score`=80.2
  - **28**: `frame_count`=90, `sync_score`=76.0
  - **39**: `frame_count`=124, `sync_score`=70.9

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 19 → `[1, 2, 3, 5, 6, 8, 9, 10, 12, 20, 22, 28, 30, 31, 36, 39, 41, 65, 67]`

- **After pre-pose prune:** 16 → `[1, 2, 3, 5, 6, 8, 9, 10, 12, 20, 22, 28, 30, 31, 39, 41]`

- **After post-pose prune:** 12 → `[1, 2, 3, 5, 6, 8, 9, 10, 20, 22, 28, 39]`

- **Dropped before pose:** [36, 65, 67]

- **Dropped after pose:** [12, 30, 31, 41]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'spatial_outlier': 1, 'jitter': 1}

  - Track **36** → `duration/kinetic` (~2 frames)
  - Track **67** → `duration/kinetic` (~1 frames)
  - Track **65** → `spatial_outlier` (~190 frames)
  - Track **30** → `jitter` (~311 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [12, 30, 31, 41] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3896

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 722 | **fps:** 29.903

- **Final exported tracks** (`track_summaries`): **10** — IDs ['1', '2', '4', '5', '6', '7', '8', '16', '53', '111']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 6.34, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=715, `sync_score`=80.2
  - **2**: `frame_count`=722, `sync_score`=81.4
  - **4**: `frame_count`=722, `sync_score`=79.0
  - **5**: `frame_count`=722, `sync_score`=81.0
  - **6**: `frame_count`=245, `sync_score`=77.7
  - **7**: `frame_count`=310, `sync_score`=81.5
  - **8**: `frame_count`=698, `sync_score`=73.3
  - **16**: `frame_count`=85, `sync_score`=62.3
  - **53**: `frame_count`=245, `sync_score`=73.9
  - **111**: `frame_count`=110, `sync_score`=71.9

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 17 → `[1, 2, 4, 5, 6, 7, 8, 13, 16, 17, 50, 53, 88, 93, 108, 111, 122]`

- **After pre-pose prune:** 13 → `[1, 2, 4, 5, 6, 7, 8, 13, 16, 17, 50, 53, 111]`

- **After post-pose prune:** 10 → `[1, 2, 4, 5, 6, 7, 8, 16, 53, 111]`

- **Dropped before pose:** [88, 93, 108, 122]

- **Dropped after pose:** [13, 17, 50]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 3, 'late_entrant_short_span': 1, 'completeness': 1}

  - Track **88** → `duration/kinetic` (~2 frames)
  - Track **93** → `duration/kinetic` (~1 frames)
  - Track **122** → `duration/kinetic` (~2 frames)
  - Track **108** → `late_entrant_short_span` (~117 frames)
  - Track **50** → `completeness` (~365 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [13, 17, 50] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_3967

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 414 | **fps:** 29.841

- **Final exported tracks** (`track_summaries`): **6** — IDs ['1', '2', '3', '4', '6', '34']

- **Per-frame `tracks` count:** max **6** simultaneous, mean 4.85, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=414, `sync_score`=81.1
  - **2**: `frame_count`=414, `sync_score`=79.1
  - **3**: `frame_count`=413, `sync_score`=83.7
  - **4**: `frame_count`=403, `sync_score`=83.7
  - **6**: `frame_count`=325, `sync_score`=82.4
  - **34**: `frame_count`=38, `sync_score`=69.7

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 10 → `[1, 2, 3, 4, 5, 6, 25, 34, 43, 45]`

- **After pre-pose prune:** 8 → `[1, 2, 3, 4, 5, 6, 25, 34]`

- **After post-pose prune:** 6 → `[1, 2, 3, 4, 6, 34]`

- **Dropped before pose:** [43, 45]

- **Dropped after pose:** [5, 25]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'completeness': 1}

  - Track **43** → `duration/kinetic` (~3 frames)
  - Track **45** → `duration/kinetic` (~6 frames)
  - Track **5** → `completeness` (~101 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [5, 25] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4027

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> excellent

### From `data.json`

- **num_frames:** 1118 | **fps:** 29.921

- **Final exported tracks** (`track_summaries`): **4** — IDs ['1', '2', '3', '4']

- **Per-frame `tracks` count:** max **4** simultaneous, mean 3.95, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=1118, `sync_score`=79.2
  - **2**: `frame_count`=1066, `sync_score`=78.6
  - **3**: `frame_count`=1112, `sync_score`=77.5
  - **4**: `frame_count`=1118, `sync_score`=81.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 6 → `[1, 2, 3, 4, 5, 18]`

- **After pre-pose prune:** 4 → `[1, 2, 3, 4]`

- **After post-pose prune:** 4 → `[1, 2, 3, 4]`

- **Dropped before pose:** [5, 18]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2}

  - Track **5** → `duration/kinetic` (~9 frames)
  - Track **18** → `duration/kinetic` (~2 frames)

### Why the issues likely happened (comments + artifacts)

- Compare **expected dancer count** in the review to **final track count** and **max simultaneous tracks**; use `prune_log` drops to see which subsystem removed each ID.


---

## IMG_4032

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 838 | **fps:** 29.914

- **Final exported tracks** (`track_summaries`): **8** — IDs ['1', '2', '4', '5', '6', '7', '8', '13']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 6.94, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=499, `sync_score`=73.4
  - **2**: `frame_count`=699, `sync_score`=77.5
  - **4**: `frame_count`=769, `sync_score`=74.4
  - **5**: `frame_count`=832, `sync_score`=73.5
  - **6**: `frame_count`=746, `sync_score`=76.3
  - **7**: `frame_count`=838, `sync_score`=76.8
  - **8**: `frame_count`=836, `sync_score`=74.9
  - **13**: `frame_count`=597, `sync_score`=71.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 16 → `[1, 2, 3, 4, 5, 6, 7, 8, 13, 42, 55, 64, 65, 78, 136, 141]`

- **After pre-pose prune:** 10 → `[1, 2, 4, 5, 6, 7, 8, 13, 55, 78]`

- **After post-pose prune:** 8 → `[1, 2, 4, 5, 6, 7, 8, 13]`

- **Dropped before pose:** [3, 42, 64, 65, 136, 141]

- **Dropped after pose:** [55, 78]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 5, 'short_track': 1}

  - Track **3** → `duration/kinetic` (~158 frames)
  - Track **42** → `duration/kinetic` (~1 frames)
  - Track **64** → `duration/kinetic` (~3 frames)
  - Track **65** → `duration/kinetic` (~4 frames)
  - Track **136** → `duration/kinetic` (~2 frames)
  - Track **141** → `short_track` (~32 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [55, 78] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4101

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 398 | **fps:** 29.835

- **Final exported tracks** (`track_summaries`): **13** — IDs ['1', '2', '3', '4', '6', '7', '8', '10', '11', '13', '14', '91', '98']

- **Per-frame `tracks` count:** max **13** simultaneous, mean 9.94, **4.3%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=346, `sync_score`=72.2
  - **2**: `frame_count`=379, `sync_score`=71.2
  - **3**: `frame_count`=367, `sync_score`=68.3
  - **4**: `frame_count`=361, `sync_score`=74.6
  - **6**: `frame_count`=379, `sync_score`=72.3
  - **7**: `frame_count`=334, `sync_score`=69.8
  - **8**: `frame_count`=348, `sync_score`=72.3
  - **10**: `frame_count`=375, `sync_score`=74.7
  - **11**: `frame_count`=172, `sync_score`=70.4
  - **13**: `frame_count`=253, `sync_score`=73.7
  - **14**: `frame_count`=351, `sync_score`=60.7
  - **91**: `frame_count`=200, `sync_score`=53.3
  - **98**: `frame_count`=92, `sync_score`=67.2

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 17 → `[1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 91, 98, 142, 178, 215]`

- **After pre-pose prune:** 14 → `[1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 91, 98]`

- **After post-pose prune:** 13 → `[1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 91, 98]`

- **Dropped before pose:** [142, 178, 215]

- **Dropped after pose:** [12]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 3}

  - Track **142** → `duration/kinetic` (~1 frames)
  - Track **178** → `duration/kinetic` (~6 frames)
  - Track **215** → `duration/kinetic` (~1 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [12] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4104

### Review comments (all text fields except `updated_at`)

- **affected_tracks_note:**

> The pipeline wronggully prunsed ID 1. THey are not a spacial outlier. THey are in frame with their full body teh entire time adn are doing choreo very similar to everyone else

- **model_feedback.what_went_wrong:**

> What went well: Good job p

- **model_feedback.what_went_well:**

> Good job pruning ID 11, you can only see their head so it makes sense to prune them

### From `data.json`

- **num_frames:** 506 | **fps:** 29.867

- **Final exported tracks** (`track_summaries`): **6** — IDs ['2', '3', '4', '5', '8', '12']

- **Per-frame `tracks` count:** max **6** simultaneous, mean 4.64, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=506, `sync_score`=76.1
  - **3**: `frame_count`=506, `sync_score`=76.7
  - **4**: `frame_count`=506, `sync_score`=71.5
  - **5**: `frame_count`=317, `sync_score`=80.5
  - **8**: `frame_count`=341, `sync_score`=80.6
  - **12**: `frame_count`=174, `sync_score`=75.2

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 11 → `[1, 2, 3, 4, 5, 8, 11, 12, 49, 57, 85]`

- **After pre-pose prune:** 7 → `[2, 3, 4, 5, 8, 12, 85]`

- **After post-pose prune:** 6 → `[2, 3, 4, 5, 8, 12]`

- **Dropped before pose:** [1, 11, 49, 57]

- **Dropped after pose:** [85]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'spatial_outlier': 2}

  - Track **49** → `duration/kinetic` (~1 frames)
  - Track **57** → `duration/kinetic` (~1 frames)
  - Track **1** → `spatial_outlier` (~504 frames)
  - Track **11** → `spatial_outlier` (~495 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [85] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4152

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 980 | **fps:** 29.925

- **Final exported tracks** (`track_summaries`): **13** — IDs ['1', '2', '3', '4', '5', '6', '11', '17', '22', '66', '127', '211', '225']

- **Per-frame `tracks` count:** max **11** simultaneous, mean 6.96, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=806, `sync_score`=73.4
  - **2**: `frame_count`=956, `sync_score`=71.3
  - **3**: `frame_count`=272, `sync_score`=79.4
  - **4**: `frame_count`=936, `sync_score`=77.3
  - **5**: `frame_count`=750, `sync_score`=82.1
  - **6**: `frame_count`=935, `sync_score`=79.4
  - **11**: `frame_count`=446, `sync_score`=72.4
  - **17**: `frame_count`=216, `sync_score`=73.5
  - **22**: `frame_count`=714, `sync_score`=78.6
  - **66**: `frame_count`=459, `sync_score`=77.5
  - **127**: `frame_count`=263, `sync_score`=75.4
  - **211**: `frame_count`=43, `sync_score`=69.8
  - **225**: `frame_count`=25, `sync_score`=67.0

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 33 → `[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 17, 19, 20, 22, 24, 33, 36, 38, 39, 59, 66, 75, 127, 129, 134, 137, 140, 173, 194, 211, 225, 237, 247]`

- **After pre-pose prune:** 15 → `[1, 2, 3, 4, 5, 6, 11, 17, 22, 39, 66, 75, 127, 211, 225]`

- **After post-pose prune:** 13 → `[1, 2, 3, 4, 5, 6, 11, 17, 22, 66, 127, 211, 225]`

- **Dropped before pose:** [7, 8, 10, 19, 20, 24, 33, 36, 38, 59, 129, 134, 137, 140, 173, 194, 237, 247]

- **Dropped after pose:** [39, 75]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 13, 'short_track': 4, 'bbox_size': 1}

  - Track **7** → `duration/kinetic` (~137 frames)
  - Track **10** → `duration/kinetic` (~40 frames)
  - Track **19** → `duration/kinetic` (~4 frames)
  - Track **20** → `duration/kinetic` (~7 frames)
  - Track **24** → `duration/kinetic` (~2 frames)
  - Track **33** → `duration/kinetic` (~1 frames)
  - Track **36** → `duration/kinetic` (~4 frames)
  - Track **38** → `duration/kinetic` (~3 frames)
  - Track **134** → `duration/kinetic` (~15 frames)
  - Track **137** → `duration/kinetic` (~1 frames)
  - Track **140** → `duration/kinetic` (~200 frames)
  - Track **237** → `duration/kinetic` (~1 frames)
  - Track **247** → `duration/kinetic` (~2 frames)
  - Track **59** → `short_track` (~144 frames)
  - Track **129** → `short_track` (~94 frames)
  - Track **173** → `short_track` (~127 frames)
  - Track **194** → `short_track` (~64 frames)
  - Track **8** → `bbox_size` (~607 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [39, 75] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4166

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 507 | **fps:** 29.867

- **Final exported tracks** (`track_summaries`): **10** — IDs ['3', '4', '5', '7', '8', '11', '45', '65', '89', '92']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 6.56, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **3**: `frame_count`=185, `sync_score`=81.3
  - **4**: `frame_count`=315, `sync_score`=77.9
  - **5**: `frame_count`=384, `sync_score`=51.4
  - **7**: `frame_count`=488, `sync_score`=75.2
  - **8**: `frame_count`=506, `sync_score`=70.5
  - **11**: `frame_count`=490, `sync_score`=80.3
  - **45**: `frame_count`=374, `sync_score`=70.3
  - **65**: `frame_count`=229, `sync_score`=75.7
  - **89**: `frame_count`=184, `sync_score`=74.2
  - **92**: `frame_count`=172, `sync_score`=77.4

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 17 → `[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 45, 60, 65, 89, 92, 103, 107]`

- **After pre-pose prune:** 11 → `[3, 4, 5, 7, 8, 11, 12, 45, 65, 89, 92]`

- **After post-pose prune:** 10 → `[3, 4, 5, 7, 8, 11, 45, 65, 89, 92]`

- **Dropped before pose:** [1, 2, 9, 60, 103, 107]

- **Dropped after pose:** [12]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 4, 'spatial_outlier': 2}

  - Track **9** → `duration/kinetic` (~4 frames)
  - Track **60** → `duration/kinetic` (~3 frames)
  - Track **103** → `duration/kinetic` (~2 frames)
  - Track **107** → `duration/kinetic` (~1 frames)
  - Track **1** → `spatial_outlier` (~507 frames)
  - Track **2** → `spatial_outlier` (~507 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [12] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4172

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 397 | **fps:** 30.0

- **Final exported tracks** (`track_summaries`): **7** — IDs ['1', '17', '24', '26', '30', '37', '40']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 2.59, **23.7%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=303, `sync_score`=74.5
  - **17**: `frame_count`=176, `sync_score`=66.9
  - **24**: `frame_count`=111, `sync_score`=69.3
  - **26**: `frame_count`=112, `sync_score`=72.4
  - **30**: `frame_count`=89, `sync_score`=68.4
  - **37**: `frame_count`=60, `sync_score`=74.8
  - **40**: `frame_count`=176, `sync_score`=72.6

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 17 → `[1, 2, 6, 15, 17, 24, 26, 27, 30, 33, 34, 37, 40, 41, 47, 51, 60]`

- **After pre-pose prune:** 8 → `[1, 15, 17, 24, 26, 30, 37, 40]`

- **After post-pose prune:** 7 → `[1, 17, 24, 26, 30, 37, 40]`

- **Dropped before pose:** [2, 6, 27, 33, 34, 41, 47, 51, 60]

- **Dropped after pose:** [15]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 9}

  - Track **2** → `duration/kinetic` (~8 frames)
  - Track **6** → `duration/kinetic` (~6 frames)
  - Track **27** → `duration/kinetic` (~1 frames)
  - Track **33** → `duration/kinetic` (~6 frames)
  - Track **34** → `duration/kinetic` (~1 frames)
  - Track **41** → `duration/kinetic` (~5 frames)
  - Track **47** → `duration/kinetic` (~22 frames)
  - Track **51** → `duration/kinetic` (~4 frames)
  - Track **60** → `duration/kinetic` (~1 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [15] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4175

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 279 | **fps:** 29.979

- **Final exported tracks** (`track_summaries`): **13** — IDs ['1', '2', '3', '4', '5', '7', '9', '10', '11', '12', '13', '30', '69']

- **Per-frame `tracks` count:** max **12** simultaneous, mean 9.99, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=217, `sync_score`=64.7
  - **2**: `frame_count`=275, `sync_score`=69.2
  - **3**: `frame_count`=279, `sync_score`=70.5
  - **4**: `frame_count`=195, `sync_score`=78.7
  - **5**: `frame_count`=264, `sync_score`=73.0
  - **7**: `frame_count`=274, `sync_score`=67.8
  - **9**: `frame_count`=277, `sync_score`=63.3
  - **10**: `frame_count`=154, `sync_score`=68.8
  - **11**: `frame_count`=184, `sync_score`=73.7
  - **12**: `frame_count`=250, `sync_score`=81.6
  - **13**: `frame_count`=223, `sync_score`=69.5
  - **30**: `frame_count`=151, `sync_score`=66.4
  - **69**: `frame_count`=45, `sync_score`=71.4

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 26 → `[1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 23, 24, 30, 35, 42, 47, 54, 55, 68, 69, 74, 85, 87, 96]`

- **After pre-pose prune:** 14 → `[1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 30, 54, 69]`

- **After post-pose prune:** 13 → `[1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 30, 69]`

- **Dropped before pose:** [14, 23, 24, 35, 42, 47, 55, 68, 74, 85, 87, 96]

- **Dropped after pose:** [54]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 10, 'spatial_outlier': 2}

  - Track **23** → `duration/kinetic` (~4 frames)
  - Track **24** → `duration/kinetic` (~1 frames)
  - Track **35** → `duration/kinetic` (~3 frames)
  - Track **42** → `duration/kinetic` (~2 frames)
  - Track **47** → `duration/kinetic` (~7 frames)
  - Track **68** → `duration/kinetic` (~22 frames)
  - Track **74** → `duration/kinetic` (~5 frames)
  - Track **85** → `duration/kinetic` (~1 frames)
  - Track **87** → `duration/kinetic` (~5 frames)
  - Track **96** → `duration/kinetic` (~6 frames)
  - Track **14** → `spatial_outlier` (~77 frames)
  - Track **55** → `spatial_outlier` (~101 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [54] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4179

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 1573 | **fps:** 29.968

- **Final exported tracks** (`track_summaries`): **12** — IDs ['1', '2', '3', '7', '27', '86', '123', '126', '132', '176', '183', '250']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 6.08, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=1182, `sync_score`=81.6
  - **2**: `frame_count`=1546, `sync_score`=79.9
  - **3**: `frame_count`=1125, `sync_score`=78.7
  - **7**: `frame_count`=898, `sync_score`=81.7
  - **27**: `frame_count`=1296, `sync_score`=77.5
  - **86**: `frame_count`=867, `sync_score`=79.5
  - **123**: `frame_count`=1049, `sync_score`=74.6
  - **126**: `frame_count`=275, `sync_score`=67.3
  - **132**: `frame_count`=383, `sync_score`=70.3
  - **176**: `frame_count`=550, `sync_score`=71.8
  - **183**: `frame_count`=272, `sync_score`=75.2
  - **250**: `frame_count`=115, `sync_score`=68.9

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 39 → `[1, 2, 3, 6, 7, 11, 21, 25, 27, 32, 37, 38, 48, 60, 65, 86, 101, 112, 123, 126, 132, 147, 166, 171, 172, 176, 182, 183, 212, 217, 233, 249, 250, 254, 284, 287, 292, 333, 370]`

- **After pre-pose prune:** 18 → `[1, 2, 3, 7, 11, 21, 27, 32, 86, 123, 126, 132, 147, 176, 183, 250, 284, 333]`

- **After post-pose prune:** 12 → `[1, 2, 3, 7, 27, 86, 123, 126, 132, 176, 183, 250]`

- **Dropped before pose:** [6, 25, 37, 38, 48, 60, 65, 101, 112, 166, 171, 172, 182, 212, 217, 233, 249, 254, 287, 292, 370]

- **Dropped after pose:** [11, 21, 32, 147, 284, 333]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 15, 'spatial_outlier': 1, 'short_track': 2, 'bbox_size': 3, 'completeness': 1, 'jitter': 1}

  - Track **6** → `duration/kinetic` (~95 frames)
  - Track **25** → `duration/kinetic` (~69 frames)
  - Track **37** → `duration/kinetic` (~2 frames)
  - Track **60** → `duration/kinetic` (~1 frames)
  - Track **112** → `duration/kinetic` (~5 frames)
  - Track **166** → `duration/kinetic` (~6 frames)
  - Track **171** → `duration/kinetic` (~32 frames)
  - Track **172** → `duration/kinetic` (~1 frames)
  - Track **182** → `duration/kinetic` (~106 frames)
  - Track **212** → `duration/kinetic` (~1 frames)
  - Track **233** → `duration/kinetic` (~1 frames)
  - Track **249** → `duration/kinetic` (~1 frames)
  - Track **287** → `duration/kinetic` (~34 frames)
  - Track **292** → `duration/kinetic` (~34 frames)
  - Track **370** → `duration/kinetic` (~466 frames)
  - Track **38** → `spatial_outlier` (~1421 frames)
  - Track **217** → `short_track` (~98 frames)
  - Track **254** → `short_track` (~222 frames)
  - Track **48** → `bbox_size` (~910 frames)
  - Track **65** → `bbox_size` (~679 frames)
  - Track **101** → `bbox_size` (~828 frames)
  - Track **333** → `completeness` (~442 frames)
  - Track **11** → `jitter` (~1560 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [11, 21, 32, 147, 284, 333] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4269

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> excellent

### From `data.json`

- **num_frames:** 513 | **fps:** 29.974

- **Final exported tracks** (`track_summaries`): **15** — IDs ['1', '2', '4', '5', '6', '7', '8', '10', '11', '17', '90', '112', '145', '197', '213']

- **Per-frame `tracks` count:** max **14** simultaneous, mean 9.61, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=292, `sync_score`=76.2
  - **2**: `frame_count`=507, `sync_score`=72.9
  - **4**: `frame_count`=499, `sync_score`=72.7
  - **5**: `frame_count`=476, `sync_score`=72.7
  - **6**: `frame_count`=465, `sync_score`=74.7
  - **7**: `frame_count`=513, `sync_score`=72.2
  - **8**: `frame_count`=352, `sync_score`=71.9
  - **10**: `frame_count`=197, `sync_score`=71.1
  - **11**: `frame_count`=401, `sync_score`=60.1
  - **17**: `frame_count`=476, `sync_score`=65.3
  - **90**: `frame_count`=55, `sync_score`=67.6
  - **112**: `frame_count`=147, `sync_score`=70.2
  - **145**: `frame_count`=133, `sync_score`=65.2
  - **197**: `frame_count`=237, `sync_score`=73.7
  - **213**: `frame_count`=181, `sync_score`=65.6

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 24 → `[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 17, 42, 90, 112, 115, 145, 197, 213, 225, 254, 260, 269, 290]`

- **After pre-pose prune:** 16 → `[1, 2, 4, 5, 6, 7, 8, 10, 11, 17, 90, 112, 115, 145, 197, 213]`

- **After post-pose prune:** 15 → `[1, 2, 4, 5, 6, 7, 8, 10, 11, 17, 90, 112, 145, 197, 213]`

- **Dropped before pose:** [3, 12, 42, 225, 254, 260, 269, 290]

- **Dropped after pose:** [115]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 6, 'spatial_outlier': 1, 'short_track': 1}

  - Track **3** → `duration/kinetic` (~89 frames)
  - Track **12** → `duration/kinetic` (~43 frames)
  - Track **42** → `duration/kinetic` (~12 frames)
  - Track **260** → `duration/kinetic` (~6 frames)
  - Track **269** → `duration/kinetic` (~3 frames)
  - Track **290** → `duration/kinetic` (~4 frames)
  - Track **225** → `spatial_outlier` (~107 frames)
  - Track **254** → `short_track` (~73 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [115] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4279

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **ground_truth.expected_dancer_count:**

> There are 14 dancers fuly visibile in this entire video

### From `data.json`

- **num_frames:** 873 | **fps:** 29.918

- **Final exported tracks** (`track_summaries`): **15** — IDs ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '13', '15', '54', '129']

- **Per-frame `tracks` count:** max **14** simultaneous, mean 12.93, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=873, `sync_score`=70.5
  - **2**: `frame_count`=650, `sync_score`=76.6
  - **3**: `frame_count`=873, `sync_score`=68.2
  - **4**: `frame_count`=869, `sync_score`=79.1
  - **5**: `frame_count`=849, `sync_score`=68.5
  - **6**: `frame_count`=791, `sync_score`=60.1
  - **7**: `frame_count`=860, `sync_score`=76.2
  - **8**: `frame_count`=604, `sync_score`=80.7
  - **9**: `frame_count`=809, `sync_score`=78.5
  - **10**: `frame_count`=759, `sync_score`=77.4
  - **11**: `frame_count`=855, `sync_score`=75.8
  - **13**: `frame_count`=829, `sync_score`=78.9
  - **15**: `frame_count`=586, `sync_score`=83.6
  - **54**: `frame_count`=868, `sync_score`=75.5
  - **129**: `frame_count`=210, `sync_score`=70.1

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 21 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 23, 54, 58, 75, 78, 125, 129, 131]`

- **After pre-pose prune:** 17 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 54, 75, 125, 129]`

- **After post-pose prune:** 15 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 54, 129]`

- **Dropped before pose:** [23, 58, 78, 131]

- **Dropped after pose:** [75, 125]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 3, 'short_track': 1}

  - Track **23** → `duration/kinetic` (~2 frames)
  - Track **78** → `duration/kinetic` (~2 frames)
  - Track **131** → `duration/kinetic` (~30 frames)
  - Track **58** → `short_track` (~101 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [75, 125] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4509

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> unusable

- **affected_tracks_note:**

> ID 1, 2, and 3 all got pruned even though they are real dancaers. Its listed thaey are pruned for being sptial outliers  or bbox size. We need to change these paramters because hese people are  stadning up adn doing thigns for most of the time. Yes, there ar etimes they are on the floor but that shouldnt matter. As long as they are up adn moving for more than about 30% othen bbox shouldnt matter. Especially because they are so close to teh camer and theier box sizes are so much alrger than all these other smlaler IDs in teh back whcih you didnt prune (should have been pruned)

- **ground_truth.expected_dancer_count:**

> There are 3 dancers whoa re fully viisble in teh front. All toher dancers should be pruned and we should be left with thee main 3.

- **notes_for_model:**

> The pipeline did the exact opposite of what is needed. It remove ID 1,2, adn 3 which are the ONLY dancers in teh video. These dancers are all teh way int eh front adn FULLY visible so i have no idea why they got pruned.
> 
> ALL other IDs are background people who are far away, smlal, running around the screen, barely visible, not doing similar movmemnts to teh main poeple in teh fornt, only show up for semgents of the video. These are MANY reasons to know why they are not real people.

### From `data.json`

- **num_frames:** 1054 | **fps:** 29.99

- **Final exported tracks** (`track_summaries`): **11** — IDs ['4', '6', '36', '78', '110', '116', '131', '146', '160', '200', '259']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 5.1, **0.9%** frames with zero tracks

- **Per-track summary:**
  - **4**: `frame_count`=687, `sync_score`=72.7
  - **6**: `frame_count`=528, `sync_score`=71.3
  - **36**: `frame_count`=444, `sync_score`=69.3
  - **78**: `frame_count`=211, `sync_score`=72.9
  - **110**: `frame_count`=411, `sync_score`=71.3
  - **116**: `frame_count`=366, `sync_score`=73.2
  - **131**: `frame_count`=256, `sync_score`=74.4
  - **146**: `frame_count`=648, `sync_score`=62.0
  - **160**: `frame_count`=239, `sync_score`=77.5
  - **200**: `frame_count`=725, `sync_score`=67.5
  - **259**: `frame_count`=863, `sync_score`=72.9

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 34 → `[1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 26, 36, 41, 78, 84, 101, 110, 112, 116, 131, 132, 146, 148, 153, 160, 171, 192, 200, 205, 215, 248, 250, 259, 260]`

- **After pre-pose prune:** 13 → `[4, 6, 36, 78, 84, 110, 116, 131, 146, 160, 200, 248, 259]`

- **After post-pose prune:** 11 → `[4, 6, 36, 78, 110, 116, 131, 146, 160, 200, 259]`

- **Dropped before pose:** [1, 2, 3, 5, 7, 8, 10, 20, 26, 41, 101, 112, 132, 148, 153, 171, 192, 205, 215, 250, 260]

- **Dropped after pose:** [84, 248]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 16, 'spatial_outlier': 1, 'short_track': 2, 'bbox_size': 2, 'jitter': 1}

  - Track **5** → `duration/kinetic` (~46 frames)
  - Track **7** → `duration/kinetic` (~1 frames)
  - Track **8** → `duration/kinetic` (~61 frames)
  - Track **10** → `duration/kinetic` (~34 frames)
  - Track **20** → `duration/kinetic` (~15 frames)
  - Track **26** → `duration/kinetic` (~136 frames)
  - Track **41** → `duration/kinetic` (~34 frames)
  - Track **101** → `duration/kinetic` (~24 frames)
  - Track **112** → `duration/kinetic` (~69 frames)
  - Track **132** → `duration/kinetic` (~102 frames)
  - Track **148** → `duration/kinetic` (~22 frames)
  - Track **153** → `duration/kinetic` (~78 frames)
  - Track **171** → `duration/kinetic` (~1 frames)
  - Track **192** → `duration/kinetic` (~11 frames)
  - Track **215** → `duration/kinetic` (~3 frames)
  - Track **260** → `duration/kinetic` (~1 frames)
  - Track **1** → `spatial_outlier` (~1046 frames)
  - Track **205** → `short_track` (~64 frames)
  - Track **250** → `short_track` (~98 frames)
  - Track **2** → `bbox_size` (~1047 frames)
  - Track **3** → `bbox_size` (~1048 frames)
  - Track **84** → `jitter` (~521 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [84, 248] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4758

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **affected_tracks_note:**

> ID 2 and ID 1 are NOT spatial outliars and should def not have been pruned.

### From `data.json`

- **num_frames:** 428 | **fps:** 29.989

- **Final exported tracks** (`track_summaries`): **8** — IDs ['3', '4', '5', '6', '7', '8', '10', '50']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 5.83, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **3**: `frame_count`=401, `sync_score`=76.9
  - **4**: `frame_count`=373, `sync_score`=76.7
  - **5**: `frame_count`=428, `sync_score`=72.3
  - **6**: `frame_count`=310, `sync_score`=76.7
  - **7**: `frame_count`=424, `sync_score`=76.7
  - **8**: `frame_count`=408, `sync_score`=78.4
  - **10**: `frame_count`=120, `sync_score`=67.8
  - **50**: `frame_count`=33, `sync_score`=70.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 16 → `[1, 2, 3, 4, 5, 6, 7, 8, 10, 23, 28, 47, 50, 79, 85, 86]`

- **After pre-pose prune:** 10 → `[3, 4, 5, 6, 7, 8, 10, 23, 50, 79]`

- **After post-pose prune:** 8 → `[3, 4, 5, 6, 7, 8, 10, 50]`

- **Dropped before pose:** [1, 2, 28, 47, 85, 86]

- **Dropped after pose:** [23, 79]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 3, 'spatial_outlier': 2, 'short_track': 1}

  - Track **28** → `duration/kinetic` (~2 frames)
  - Track **47** → `duration/kinetic` (~2 frames)
  - Track **85** → `duration/kinetic` (~2 frames)
  - Track **1** → `spatial_outlier` (~427 frames)
  - Track **2** → `spatial_outlier` (~427 frames)
  - Track **86** → `short_track` (~34 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [23, 79] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_4764

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **problem_segments_sec[0].note:**

> At this time ID 3 (correctly mapped) gets incorectly changed to ID 182. ID 182 is obvoulsy wrong as its just mapped onto the floor during 0-12.4 seconds. we should know then that we should maintain the already used ID of 3 rather than replacing it with a bad region of 182.

- **affected_tracks_note:**

> ID 2 and 11 are on teh deges, but in fact should not have been pruned. They are dancing and are mostly shown for all teh video, so should be suffeceinct evidence to not prune them.

### From `data.json`

- **num_frames:** 938 | **fps:** 29.912

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '3', '4', '5', '6', '7', '8', '222', '247']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 6.38, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=937, `sync_score`=71.9
  - **3**: `frame_count`=919, `sync_score`=67.6
  - **4**: `frame_count`=554, `sync_score`=67.2
  - **5**: `frame_count`=390, `sync_score`=69.0
  - **6**: `frame_count`=869, `sync_score`=70.2
  - **7**: `frame_count`=911, `sync_score`=75.3
  - **8**: `frame_count`=931, `sync_score`=73.4
  - **222**: `frame_count`=355, `sync_score`=75.9
  - **247**: `frame_count`=122, `sync_score`=63.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 22 → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 20, 26, 146, 172, 182, 196, 222, 229, 247, 257, 280, 284]`

- **After pre-pose prune:** 13 → `[1, 3, 4, 5, 6, 7, 8, 9, 26, 146, 196, 222, 247]`

- **After post-pose prune:** 9 → `[1, 3, 4, 5, 6, 7, 8, 222, 247]`

- **Dropped before pose:** [2, 11, 20, 172, 182, 229, 257, 280, 284]

- **Dropped after pose:** [9, 26, 146, 196]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 5, 'spatial_outlier': 1, 'audience_region': 2, 'late_entrant_short_span': 1}

  - Track **20** → `duration/kinetic` (~76 frames)
  - Track **172** → `duration/kinetic` (~1 frames)
  - Track **229** → `duration/kinetic` (~42 frames)
  - Track **280** → `duration/kinetic` (~1 frames)
  - Track **284** → `duration/kinetic` (~10 frames)
  - Track **2** → `spatial_outlier` (~788 frames)
  - Track **11** → `audience_region` (~926 frames)
  - Track **182** → `audience_region` (~933 frames)
  - Track **257** → `late_entrant_short_span` (~150 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [9, 26, 146, 196] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_5261

### Review comments (all text fields except `updated_at`)

- **problem_segments_sec[0].note:**

> I ntoiced that ID 4 was paired onto someone on the edge adn then it jumped to being paire to this person in the middle. Because it was orginally apired with someone at the dege it got pruned for uadince region. It should be obvious that someone sitting donw and in teh edge did not just jump and become a person dancing in the middle. Thes epoeple shouldve been given different ID numbers. Overall we need to have a muhc mroe accurate system to maintain ID numbers wihtt he correct poepl,e because things like this for incorrect prunigns can happen.

- **affected_tracks_note:**

> ID 24 and ID 96 and ID 7 are all epole sitting donw, facing away, not all their body is shown, adn are on teh edges. All these signs should coem to gether to have more than enouhg evidence that they are not in teh dance and shoudlnt be tracked

- **ground_truth.expected_dancer_count:**

> We should have only 5 dancers deterected. There are more poeple visible in the front but they should be pruned as part of audince, leving us with 5 dancers throuhguht the video.

### From `data.json`

- **num_frames:** 717 | **fps:** 29.987

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '2', '3', '5', '7', '24', '72', '96', '144']

- **Per-frame `tracks` count:** max **8** simultaneous, mean 5.86, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=572, `sync_score`=69.7
  - **2**: `frame_count`=717, `sync_score`=73.5
  - **3**: `frame_count`=716, `sync_score`=74.8
  - **5**: `frame_count`=435, `sync_score`=73.1
  - **7**: `frame_count`=707, `sync_score`=59.5
  - **24**: `frame_count`=352, `sync_score`=54.3
  - **72**: `frame_count`=236, `sync_score`=72.0
  - **96**: `frame_count`=310, `sync_score`=53.2
  - **144**: `frame_count`=155, `sync_score`=73.8

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 28 → `[1, 2, 3, 4, 5, 7, 9, 11, 15, 17, 21, 24, 29, 34, 35, 39, 44, 45, 46, 72, 84, 96, 105, 115, 121, 139, 144, 159]`

- **After pre-pose prune:** 16 → `[1, 2, 3, 5, 7, 9, 24, 29, 34, 44, 72, 96, 105, 115, 139, 144]`

- **After post-pose prune:** 9 → `[1, 2, 3, 5, 7, 24, 72, 96, 144]`

- **Dropped before pose:** [4, 11, 15, 17, 21, 35, 39, 45, 46, 84, 121, 159]

- **Dropped after pose:** [9, 29, 34, 44, 105, 115, 139]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 10, 'audience_region': 2, 'low_sync': 4, 'completeness': 3, 'low_confidence': 2}

  - Track **11** → `duration/kinetic` (~63 frames)
  - Track **15** → `duration/kinetic` (~2 frames)
  - Track **21** → `duration/kinetic` (~106 frames)
  - Track **35** → `duration/kinetic` (~100 frames)
  - Track **39** → `duration/kinetic` (~1 frames)
  - Track **45** → `duration/kinetic` (~14 frames)
  - Track **46** → `duration/kinetic` (~5 frames)
  - Track **84** → `duration/kinetic` (~6 frames)
  - Track **121** → `duration/kinetic` (~3 frames)
  - Track **159** → `duration/kinetic` (~12 frames)
  - Track **4** → `audience_region` (~716 frames)
  - Track **17** → `audience_region` (~171 frames)
  - Track **9** → `low_sync` (~716 frames)
  - Track **29** → `low_sync` (~474 frames)
  - Track **34** → `low_sync` (~717 frames)
  - Track **105** → `low_sync` (~273 frames)
  - Track **9** → `completeness` (~716 frames)
  - Track **29** → `completeness` (~474 frames)
  - Track **34** → `completeness` (~717 frames)
  - Track **9** → `low_confidence` (~716 frames)
  - Track **105** → `low_confidence` (~273 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [9, 29, 34, 44, 105, 115, 139] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Jitter / ID hopping** — BoT-SORT (and later crossover/dedup) **reassigns** the same numeric ID to different bodies; `prune_log` may record a **`jitter`** prune for that ID (e.g. IMG_7821 track 3) even when the viewer still sees confusion in earlier frames or in the rendered video.


---

## IMG_5843

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 580 | **fps:** 29.987

- **Final exported tracks** (`track_summaries`): **7** — IDs ['2', '3', '4', '5', '6', '8', '113']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 4.79, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **2**: `frame_count`=520, `sync_score`=72.5
  - **3**: `frame_count`=568, `sync_score`=78.4
  - **4**: `frame_count`=81, `sync_score`=70.9
  - **5**: `frame_count`=398, `sync_score`=78.0
  - **6**: `frame_count`=577, `sync_score`=76.5
  - **8**: `frame_count`=260, `sync_score`=70.6
  - **113**: `frame_count`=374, `sync_score`=73.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 13 → `[1, 2, 3, 4, 5, 6, 8, 13, 28, 113, 163, 196, 213]`

- **After pre-pose prune:** 11 → `[1, 2, 3, 4, 5, 6, 8, 13, 28, 113, 196]`

- **After post-pose prune:** 7 → `[2, 3, 4, 5, 6, 8, 113]`

- **Dropped before pose:** [163, 213]

- **Dropped after pose:** [1, 13, 28, 196]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'jitter': 1}

  - Track **163** → `duration/kinetic` (~5 frames)
  - Track **213** → `duration/kinetic` (~3 frames)
  - Track **1** → `jitter` (~563 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [1, 13, 28, 196] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_7094

### Review comments (all text fields except `updated_at`)

- **model_feedback.what_went_wrong:**

> What went well: good

- **model_feedback.what_went_well:**

> good job for pruning ID 66

### From `data.json`

- **num_frames:** 364 | **fps:** 29.836

- **Final exported tracks** (`track_summaries`): **8** — IDs ['1', '2', '3', '4', '5', '6', '7', '14']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 5.15, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=360, `sync_score`=79.6
  - **2**: `frame_count`=361, `sync_score`=75.1
  - **3**: `frame_count`=155, `sync_score`=72.0
  - **4**: `frame_count`=364, `sync_score`=79.2
  - **5**: `frame_count`=39, `sync_score`=80.4
  - **6**: `frame_count`=256, `sync_score`=75.9
  - **7**: `frame_count`=169, `sync_score`=77.7
  - **14**: `frame_count`=172, `sync_score`=78.3

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 12 → `[1, 2, 3, 4, 5, 6, 7, 14, 17, 66, 71, 104]`

- **After pre-pose prune:** 9 → `[1, 2, 3, 4, 5, 6, 7, 14, 17]`

- **After post-pose prune:** 8 → `[1, 2, 3, 4, 5, 6, 7, 14]`

- **Dropped before pose:** [66, 71, 104]

- **Dropped after pose:** [17]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'bbox_size': 1}

  - Track **71** → `duration/kinetic` (~4 frames)
  - Track **104** → `duration/kinetic` (~1 frames)
  - Track **66** → `bbox_size` (~132 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [17] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_7546

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **affected_tracks_note:**

> ID 29 is being pruned for being a spatial outlier but it is not.

### From `data.json`

- **num_frames:** 860 | **fps:** 29.951

- **Final exported tracks** (`track_summaries`): **10** — IDs ['1', '2', '3', '4', '6', '11', '13', '92', '109', '121']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 6.42, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=817, `sync_score`=74.4
  - **2**: `frame_count`=687, `sync_score`=71.6
  - **3**: `frame_count`=423, `sync_score`=77.9
  - **4**: `frame_count`=676, `sync_score`=79.0
  - **6**: `frame_count`=851, `sync_score`=78.4
  - **11**: `frame_count`=671, `sync_score`=73.4
  - **13**: `frame_count`=368, `sync_score`=75.1
  - **92**: `frame_count`=756, `sync_score`=71.1
  - **109**: `frame_count`=125, `sync_score`=71.2
  - **121**: `frame_count`=145, `sync_score`=61.8

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 19 → `[1, 2, 3, 4, 6, 11, 13, 29, 35, 51, 79, 84, 92, 109, 119, 121, 130, 141, 165]`

- **After pre-pose prune:** 11 → `[1, 2, 3, 4, 6, 11, 13, 35, 92, 109, 121]`

- **After post-pose prune:** 10 → `[1, 2, 3, 4, 6, 11, 13, 92, 109, 121]`

- **Dropped before pose:** [29, 51, 79, 84, 119, 130, 141, 165]

- **Dropped after pose:** [35]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 7, 'spatial_outlier': 1}

  - Track **51** → `duration/kinetic` (~4 frames)
  - Track **79** → `duration/kinetic` (~4 frames)
  - Track **84** → `duration/kinetic` (~2 frames)
  - Track **119** → `duration/kinetic` (~3 frames)
  - Track **130** → `duration/kinetic` (~2 frames)
  - Track **141** → `duration/kinetic` (~1 frames)
  - Track **165** → `duration/kinetic` (~2 frames)
  - Track **29** → `spatial_outlier` (~676 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [35] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_7821

### Review comments (all text fields except `updated_at`)

- **problem_segments_sec[0].note:**

> for ID 5 there is only a leg visible here. That is too little for us to be showing to teh user. We shoudlnt consider this as a person when there is only 1 leg, it should get pruned.

- **affected_tracks_note:**

> ID 3 is jittering a lot. UNsure why because there is ample space between the people.
> 
> Also im seieng ID 273 and ID 5. Both of which are not people. ID 273 is tracking a chair and ID 5 is tracking empty space. Again, it seems like we need to have a higher confidence threshold

- **ground_truth.expected_dancer_count:**

> We should see 7 people throuhguht the video. At all times of the vidoe there are only 7 people who are fully visible.

- **notes_for_model:**

> ID 3 is jumping around from 2 poeple who are decently far away. I thin we need to def add the color of teh IDed person as refernce so when IDing people we use that as a higher criteria. ALso, we need to make sure that for every frame we run the pose anayslis to find ALL epople with both moels and use both to check over eachother. THen we cna ID people adn use the ersons cloths color to help ID them as well as the other methods we currently have.
> 
> overall im ntoiciing jitter being a bit issue because the ID moves from person to person. this means that we are acutlly detecting the person is there, we are just incorectly putting the IDs. For ecampe, in a video im seeing 3 people somehwat near each other but only 1 person gets the ID. and teh ID mvoes between people. This emasn we acutally are detecting all 3 poeple. If this is the acse why arent we jsut giving all 3 people their own uniqiue IDs and pose estimating them for all frames. Ive noticed this esepeitally for IMG_7821

### From `data.json`

- **num_frames:** 1284 | **fps:** 29.941

- **Final exported tracks** (`track_summaries`): **7** — IDs ['1', '2', '4', '5', '7', '8', '273']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 5.5, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=1254, `sync_score`=72.6
  - **2**: `frame_count`=1284, `sync_score`=76.3
  - **4**: `frame_count`=1284, `sync_score`=72.7
  - **5**: `frame_count`=1069, `sync_score`=63.1
  - **7**: `frame_count`=1142, `sync_score`=75.7
  - **8**: `frame_count`=492, `sync_score`=50.9
  - **273**: `frame_count`=531, `sync_score`=48.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 13 → `[1, 2, 3, 4, 5, 7, 8, 22, 107, 149, 155, 273, 315]`

- **After pre-pose prune:** 11 → `[1, 2, 3, 4, 5, 7, 8, 22, 107, 155, 273]`

- **After post-pose prune:** 7 → `[1, 2, 4, 5, 7, 8, 273]`

- **Dropped before pose:** [149, 315]

- **Dropped after pose:** [3, 22, 107, 155]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'jitter': 1}

  - Track **149** → `duration/kinetic` (~1 frames)
  - Track **315** → `duration/kinetic` (~1 frames)
  - Track **3** → `jitter` (~1284 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [3, 22, 107, 155] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Jitter / ID hopping** — BoT-SORT (and later crossover/dedup) **reassigns** the same numeric ID to different bodies; `prune_log` may record a **`jitter`** prune for that ID (e.g. IMG_7821 track 3) even when the viewer still sees confusion in earlier frames or in the rendered video.

- **Non-person boxes** — YOLO `person` class fires on props/layout; tracks survive until **duration/kinetic**, **sync**, or **bbox** rules cut them. Low **ViTPose sync** (e.g. chair) often keeps them unless post-pose pruning catches them.

- **Close spacing** — high simultaneous `tracks` count with reviewer complaints about IDs usually points to **association errors**, **crossover OKS swaps**, or **deduplicate_collocated_poses**.

- **Partial body** — reviewer wants **keypoint-completeness** gates; current pipeline may still export a track with **weak lower/upper body** coverage.

- **Appearance** — reviewer wants stronger **color Re-ID**; pipeline already extracts **HSV embeddings** but may need **higher weight** or better features for costumes/lighting.


---

## IMG_7922

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> good

- **problem_segments_sec[0].note:**

> ID 12 and ID 10 are on teh side nad just stadning/walking around. They are clearly not doing the saem movememnts as eveyrone else and should be pruned

### From `data.json`

- **num_frames:** 455 | **fps:** 29.987

- **Final exported tracks** (`track_summaries`): **10** — IDs ['1', '2', '3', '5', '6', '7', '9', '12', '18', '19']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 6.56, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=378, `sync_score`=77.0
  - **2**: `frame_count`=452, `sync_score`=77.7
  - **3**: `frame_count`=228, `sync_score`=73.1
  - **5**: `frame_count`=263, `sync_score`=76.7
  - **6**: `frame_count`=402, `sync_score`=78.4
  - **7**: `frame_count`=344, `sync_score`=65.7
  - **9**: `frame_count`=77, `sync_score`=80.3
  - **12**: `frame_count`=204, `sync_score`=75.2
  - **18**: `frame_count`=455, `sync_score`=74.2
  - **19**: `frame_count`=183, `sync_score`=63.8

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 14 → `[1, 2, 3, 5, 6, 7, 9, 10, 12, 18, 19, 21, 31, 64]`

- **After pre-pose prune:** 11 → `[1, 2, 3, 5, 6, 7, 9, 12, 18, 19, 21]`

- **After post-pose prune:** 10 → `[1, 2, 3, 5, 6, 7, 9, 12, 18, 19]`

- **Dropped before pose:** [10, 31, 64]

- **Dropped after pose:** [21]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 2, 'bbox_size': 1}

  - Track **31** → `duration/kinetic` (~3 frames)
  - Track **64** → `duration/kinetic` (~2 frames)
  - Track **10** → `bbox_size` (~401 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [21] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_7940

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> ok

- **problem_segments_sec[0].note:**

> Its mapping ID2 to something in a rnadom prop in the back

- **notes_for_model:**

> I think we need to make sure on everyframe to view the entire image and check for people and then use the frames before to check which ID it should amtch to. I feel as thouhg we pikc up MANY false positives and we also fail to pikc up people a lot of the time. IF we were actually doing teh work of check for people at eveyrframe and then validating it with BOTH the yolo and vitpose model I dont undesrtand how we can how so many false positives and also be missing so many peoople as well. Overlal we might need to rehtink how we are esitimating poeple in a way that is more accurate

### From `data.json`

- **num_frames:** 511 | **fps:** 29.988

- **Final exported tracks** (`track_summaries`): **7** — IDs ['1', '2', '8', '16', '17', '19', '26']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 3.7, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=494, `sync_score`=82.1
  - **2**: `frame_count`=500, `sync_score`=83.7
  - **8**: `frame_count`=248, `sync_score`=77.8
  - **16**: `frame_count`=267, `sync_score`=83.5
  - **17**: `frame_count`=127, `sync_score`=67.4
  - **19**: `frame_count`=189, `sync_score`=85.4
  - **26**: `frame_count`=68, `sync_score`=79.5

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 11 → `[1, 2, 8, 16, 17, 19, 23, 24, 25, 26, 30]`

- **After pre-pose prune:** 10 → `[1, 2, 8, 16, 17, 19, 23, 24, 25, 26]`

- **After post-pose prune:** 7 → `[1, 2, 8, 16, 17, 19, 26]`

- **Dropped before pose:** [30]

- **Dropped after pose:** [23, 24, 25]

- **Prune log rules (may include diagnostics, not only final drops):** {'short_track': 1}

  - Track **30** → `short_track` (~71 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [23, 24, 25] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

- **Non-person boxes** — YOLO `person` class fires on props/layout; tracks survive until **duration/kinetic**, **sync**, or **bbox** rules cut them. Low **ViTPose sync** (e.g. chair) often keeps them unless post-pose pruning catches them.


---

## IMG_7961

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 514 | **fps:** 29.988

- **Final exported tracks** (`track_summaries`): **11** — IDs ['1', '2', '3', '4', '5', '6', '8', '18', '44', '49', '59']

- **Per-frame `tracks` count:** max **10** simultaneous, mean 7.31, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=495, `sync_score`=57.2
  - **2**: `frame_count`=514, `sync_score`=44.3
  - **3**: `frame_count`=254, `sync_score`=62.6
  - **4**: `frame_count`=495, `sync_score`=59.7
  - **5**: `frame_count`=491, `sync_score`=67.5
  - **6**: `frame_count`=501, `sync_score`=73.5
  - **8**: `frame_count`=482, `sync_score`=66.8
  - **18**: `frame_count`=226, `sync_score`=70.4
  - **44**: `frame_count`=36, `sync_score`=74.8
  - **49**: `frame_count`=70, `sync_score`=65.5
  - **59**: `frame_count`=192, `sync_score`=74.7

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 21 → `[1, 2, 3, 4, 5, 6, 8, 10, 13, 18, 21, 40, 44, 49, 59, 75, 100, 108, 116, 117, 119]`

- **After pre-pose prune:** 12 → `[1, 2, 3, 4, 5, 6, 8, 13, 18, 44, 49, 59]`

- **After post-pose prune:** 11 → `[1, 2, 3, 4, 5, 6, 8, 18, 44, 49, 59]`

- **Dropped before pose:** [10, 21, 40, 75, 100, 108, 116, 117, 119]

- **Dropped after pose:** [13]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 8, 'spatial_outlier': 1}

  - Track **10** → `duration/kinetic` (~3 frames)
  - Track **21** → `duration/kinetic` (~1 frames)
  - Track **75** → `duration/kinetic` (~1 frames)
  - Track **100** → `duration/kinetic` (~11 frames)
  - Track **108** → `duration/kinetic` (~1 frames)
  - Track **116** → `duration/kinetic` (~2 frames)
  - Track **117** → `duration/kinetic` (~2 frames)
  - Track **119** → `duration/kinetic` (~1 frames)
  - Track **40** → `spatial_outlier` (~216 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [13] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_8077

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 574 | **fps:** 29.984

- **Final exported tracks** (`track_summaries`): **6** — IDs ['3', '7', '112', '117', '188', '230']

- **Per-frame `tracks` count:** max **6** simultaneous, mean 2.94, **0.0%** frames with zero tracks

- **Per-track summary:**
  - **3**: `frame_count`=573, `sync_score`=74.9
  - **7**: `frame_count`=383, `sync_score`=73.8
  - **112**: `frame_count`=386, `sync_score`=68.8
  - **117**: `frame_count`=133, `sync_score`=66.4
  - **188**: `frame_count`=118, `sync_score`=65.6
  - **230**: `frame_count`=93, `sync_score`=76.7

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 24 → `[1, 3, 7, 8, 9, 14, 15, 16, 18, 23, 33, 59, 112, 117, 123, 156, 163, 167, 175, 188, 213, 230, 254, 287]`

- **After pre-pose prune:** 13 → `[1, 3, 7, 8, 15, 16, 23, 33, 112, 117, 175, 188, 230]`

- **After post-pose prune:** 6 → `[3, 7, 112, 117, 188, 230]`

- **Dropped before pose:** [9, 14, 18, 59, 123, 156, 163, 167, 213, 254, 287]

- **Dropped after pose:** [1, 8, 15, 16, 23, 33, 175]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 9, 'spatial_outlier': 1, 'short_track': 1, 'jitter': 3}

  - Track **9** → `duration/kinetic` (~73 frames)
  - Track **18** → `duration/kinetic` (~58 frames)
  - Track **59** → `duration/kinetic` (~50 frames)
  - Track **123** → `duration/kinetic` (~6 frames)
  - Track **156** → `duration/kinetic` (~1 frames)
  - Track **163** → `duration/kinetic` (~13 frames)
  - Track **167** → `duration/kinetic` (~2 frames)
  - Track **213** → `duration/kinetic` (~1 frames)
  - Track **287** → `duration/kinetic` (~1 frames)
  - Track **14** → `spatial_outlier` (~555 frames)
  - Track **254** → `short_track` (~48 frames)
  - Track **1** → `jitter` (~574 frames)
  - Track **8** → `jitter` (~324 frames)
  - Track **15** → `jitter` (~541 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [1, 8, 15, 16, 23, 33, 175] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_9563

### Review comments (all text fields except `updated_at`)

- **overall_quality:**

> poor

- **affected_tracks_note:**

> ID 7 is geteting pruned at a spatial outlier when it is not

### From `data.json`

- **num_frames:** 2038 | **fps:** 29.966

- **Final exported tracks** (`track_summaries`): **9** — IDs ['1', '2', '5', '12', '15', '22', '74', '252', '363']

- **Per-frame `tracks` count:** max **7** simultaneous, mean 4.77, **0.2%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=1205, `sync_score`=74.4
  - **2**: `frame_count`=1489, `sync_score`=70.5
  - **5**: `frame_count`=518, `sync_score`=71.8
  - **12**: `frame_count`=214, `sync_score`=77.7
  - **15**: `frame_count`=1660, `sync_score`=69.3
  - **22**: `frame_count`=1395, `sync_score`=73.6
  - **74**: `frame_count`=1095, `sync_score`=72.1
  - **252**: `frame_count`=1352, `sync_score`=74.3
  - **363**: `frame_count`=801, `sync_score`=64.6

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 34 → `[1, 2, 5, 7, 9, 12, 15, 19, 22, 74, 88, 95, 117, 130, 140, 144, 166, 182, 229, 252, 307, 308, 312, 317, 320, 330, 339, 349, 363, 408, 448, 452, 482, 495]`

- **After pre-pose prune:** 16 → `[1, 2, 5, 12, 15, 22, 74, 117, 229, 252, 307, 320, 339, 363, 408, 482]`

- **After post-pose prune:** 9 → `[1, 2, 5, 12, 15, 22, 74, 252, 363]`

- **Dropped before pose:** [7, 9, 19, 88, 95, 130, 140, 144, 166, 182, 308, 312, 317, 330, 349, 448, 452, 495]

- **Dropped after pose:** [117, 229, 307, 320, 339, 408, 482]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 16, 'spatial_outlier': 1, 'short_track': 1, 'low_sync': 1, 'low_confidence': 1, 'jitter': 1}

  - Track **9** → `duration/kinetic` (~17 frames)
  - Track **19** → `duration/kinetic` (~3 frames)
  - Track **88** → `duration/kinetic` (~76 frames)
  - Track **95** → `duration/kinetic` (~25 frames)
  - Track **130** → `duration/kinetic` (~320 frames)
  - Track **140** → `duration/kinetic` (~31 frames)
  - Track **144** → `duration/kinetic` (~2 frames)
  - Track **166** → `duration/kinetic` (~2 frames)
  - Track **182** → `duration/kinetic` (~1 frames)
  - Track **308** → `duration/kinetic` (~1 frames)
  - Track **312** → `duration/kinetic` (~22 frames)
  - Track **317** → `duration/kinetic` (~2 frames)
  - Track **330** → `duration/kinetic` (~1 frames)
  - Track **349** → `duration/kinetic` (~12 frames)
  - Track **452** → `duration/kinetic` (~1 frames)
  - Track **495** → `duration/kinetic` (~28 frames)
  - Track **7** → `spatial_outlier` (~1326 frames)
  - Track **448** → `short_track` (~183 frames)
  - Track **307** → `low_sync` (~526 frames)
  - Track **307** → `low_confidence` (~526 frames)
  - Track **117** → `jitter` (~1059 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [117, 229, 307, 320, 339, 408, 482] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.


---

## IMG_9564

### Review comments (all text fields except `updated_at`)

### From `data.json`

- **num_frames:** 790 | **fps:** 29.945

- **Final exported tracks** (`track_summaries`): **20** — IDs ['1', '10', '23', '199', '207', '216', '217', '222', '224', '261', '306', '329', '338', '351', '364', '378', '382', '401', '402', '426']

- **Per-frame `tracks` count:** max **16** simultaneous, mean 9.0, **4.2%** frames with zero tracks

- **Per-track summary:**
  - **1**: `frame_count`=449, `sync_score`=61.4
  - **10**: `frame_count`=682, `sync_score`=72.5
  - **23**: `frame_count`=559, `sync_score`=75.1
  - **199**: `frame_count`=693, `sync_score`=64.8
  - **207**: `frame_count`=575, `sync_score`=77.8
  - **216**: `frame_count`=136, `sync_score`=73.8
  - **217**: `frame_count`=286, `sync_score`=58.8
  - **222**: `frame_count`=459, `sync_score`=71.2
  - **224**: `frame_count`=215, `sync_score`=81.8
  - **261**: `frame_count`=402, `sync_score`=0.0
  - **306**: `frame_count`=250, `sync_score`=69.4
  - **329**: `frame_count`=309, `sync_score`=77.8
  - **338**: `frame_count`=226, `sync_score`=74.7
  - **351**: `frame_count`=246, `sync_score`=71.7
  - **364**: `frame_count`=222, `sync_score`=55.3
  - **378**: `frame_count`=215, `sync_score`=68.5
  - **382**: `frame_count`=532, `sync_score`=54.0
  - **401**: `frame_count`=186, `sync_score`=60.4
  - **402**: `frame_count`=185, `sync_score`=75.1
  - **426**: `frame_count`=285, `sync_score`=63.7

### From `prune_log.json`

- **IDs from raw tracker (before pipeline prune):** 76 → `[1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 23, 25, 33, 34, 37, 38, 41, 43, 44, 55, 60, 76, 84, 94, 98, 105, 107, 116, 122, 127, 128, 150, 155, 163, 165, 166, 169, 172, 174, 178, 180, 192, 199, 202, 207, 216, 217, 222, 224, 225, 230, 231, 240, 261, 268, 271, 272, 306, 316, 318, 329, 330, 338, 351, 352, 364, 378, 382, 394, 401, 402, 403, 412, 416, 426]`

- **After pre-pose prune:** 27 → `[1, 2, 8, 10, 11, 14, 23, 84, 199, 207, 216, 217, 222, 224, 261, 306, 329, 330, 338, 351, 364, 378, 382, 401, 402, 416, 426]`

- **After post-pose prune:** 20 → `[1, 10, 23, 199, 207, 216, 217, 222, 224, 261, 306, 329, 338, 351, 364, 378, 382, 401, 402, 426]`

- **Dropped before pose:** [4, 6, 7, 9, 13, 25, 33, 34, 37, 38, 41, 43, 44, 55, 60, 76, 94, 98, 105, 107, 116, 122, 127, 128, 150, 155, 163, 165, 166, 169, 172, 174, 178, 180, 192, 202, 225, 230, 231, 240, 268, 271, 272, 316, 318, 352, 394, 403, 412]

- **Dropped after pose:** [2, 8, 11, 14, 84, 330, 416]

- **Prune log rules (may include diagnostics, not only final drops):** {'duration/kinetic': 43, 'spatial_outlier': 2, 'audience_region': 1, 'bbox_size': 2, 'aspect_ratio': 1, 'low_confidence': 3, 'jitter': 3}

  - Track **6** → `duration/kinetic` (~101 frames)
  - Track **7** → `duration/kinetic` (~5 frames)
  - Track **9** → `duration/kinetic` (~1 frames)
  - Track **13** → `duration/kinetic` (~23 frames)
  - Track **25** → `duration/kinetic` (~86 frames)
  - Track **33** → `duration/kinetic` (~112 frames)
  - Track **37** → `duration/kinetic` (~110 frames)
  - Track **38** → `duration/kinetic` (~98 frames)
  - Track **41** → `duration/kinetic` (~13 frames)
  - Track **43** → `duration/kinetic` (~15 frames)
  - Track **55** → `duration/kinetic` (~88 frames)
  - Track **60** → `duration/kinetic` (~5 frames)
  - Track **76** → `duration/kinetic` (~49 frames)
  - Track **94** → `duration/kinetic` (~10 frames)
  - Track **98** → `duration/kinetic` (~8 frames)
  - Track **105** → `duration/kinetic` (~1 frames)
  - Track **107** → `duration/kinetic` (~1 frames)
  - Track **116** → `duration/kinetic` (~5 frames)
  - Track **122** → `duration/kinetic` (~4 frames)
  - Track **127** → `duration/kinetic` (~3 frames)
  - Track **128** → `duration/kinetic` (~93 frames)
  - Track **150** → `duration/kinetic` (~1 frames)
  - Track **155** → `duration/kinetic` (~26 frames)
  - Track **165** → `duration/kinetic` (~2 frames)
  - Track **166** → `duration/kinetic` (~3 frames)
  - Track **169** → `duration/kinetic` (~4 frames)
  - Track **172** → `duration/kinetic` (~1 frames)
  - Track **174** → `duration/kinetic` (~1 frames)
  - Track **178** → `duration/kinetic` (~4 frames)
  - Track **180** → `duration/kinetic` (~3 frames)
  - Track **225** → `duration/kinetic` (~1 frames)
  - Track **230** → `duration/kinetic` (~2 frames)
  - Track **231** → `duration/kinetic` (~1 frames)
  - Track **240** → `duration/kinetic` (~1 frames)
  - Track **268** → `duration/kinetic` (~17 frames)
  - Track **271** → `duration/kinetic` (~3 frames)
  - Track **272** → `duration/kinetic` (~1 frames)
  - Track **316** → `duration/kinetic` (~35 frames)
  - Track **318** → `duration/kinetic` (~11 frames)
  - Track **352** → `duration/kinetic` (~1 frames)
  - Track **394** → `duration/kinetic` (~4 frames)
  - Track **403** → `duration/kinetic` (~12 frames)
  - Track **412** → `duration/kinetic` (~17 frames)
  - Track **44** → `spatial_outlier` (~195 frames)
  - Track **163** → `spatial_outlier` (~164 frames)
  - Track **34** → `audience_region` (~225 frames)
  - Track **4** → `bbox_size` (~455 frames)
  - Track **192** → `bbox_size` (~498 frames)
  - Track **202** → `aspect_ratio` (~493 frames)
  - Track **11** → `low_confidence` (~387 frames)
  - Track **14** → `low_confidence` (~532 frames)
  - Track **330** → `low_confidence` (~206 frames)
  - Track **2** → `jitter` (~225 frames)
  - Track **8** → `jitter` (~194 frames)
  - Track **84** → `jitter` (~145 frames)

### Why the issues likely happened (comments + artifacts)

- **Post-pose removal** of IDs [2, 8, 11, 14, 84, 330, 416] — see prune rules logged for those IDs (e.g. jitter, low sync, mirror, completeness). Explains “ID vanished” or “bad ID gone” in the final MP4/JSON.

