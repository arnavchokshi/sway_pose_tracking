# Vendor (local clones)

- **MotionAGFormer** — clone here for 3D pose lift (`sway/pose_lift_3d.py`):

  ```bash
  git clone https://github.com/TaatiTeam/MotionAGFormer.git MotionAGFormer
  ```

  The `MotionAGFormer/` directory is gitignored. Also run `pip install timm` and
  `python -m tools.prefetch_models --include-3d` from the repo root for weights.

- **PoseFormerV2** — optional alternative lifter (`SWAY_LIFT_BACKEND=poseformerv2`):

  ```bash
  git clone https://github.com/QitaoZhao/PoseFormerV2.git PoseFormerV2
  ```

  Install `einops` and `torch-dct` (see `requirements.txt`). Download a checkpoint
  from the [PoseFormerV2 README](https://github.com/QitaoZhao/PoseFormerV2#evaluation)
  (e.g. `27_243_45.2.bin`) into `models/` or set `SWAY_POSEFORMERV2_WEIGHTS`.
  Official releases are **Human3.6M** (and MPI-INF-3DHP); there is no published
  AMASS checkpoint in that repo — train or fine-tune on AMASS if you need it.
