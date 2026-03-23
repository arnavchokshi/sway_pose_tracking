# What’s left for you — YOLO11x fine-tune

**Already done in this workspace (do not repeat):** DanceTrack is on disk under `datasets/dancetrack/`, and `convert_dancetrack_to_yolo.py` has been run → `datasets/dancetrack_yolo/` exists.

This file lists **only** what you still need to do for the merged **DanceTrack + CrowdHuman** train and for using the new weights in the app.

**Project root:** `sway_pose_mvp/` (folder with `main.py` and `sway/`).

```bash
cd /path/to/sway_pose_mvp
```

---

## 1. CrowdHuman (~33 GB) — you must do this

### 1.0 Google Drive often fails — use this instead

The **Google Drive** buttons on [crowdhuman.org](https://www.crowdhuman.org/download.html) frequently return:

> *Sorry, the file you have requested does not exist.*

That usually means the link is **broken, moved, or not shared anymore** — not something you can fix with a “correct URL.” **Do not rely on those Drive links.**

**Easiest fix (recommended):** download the **same files** from Hugging Face with the script below (no Baidu client needed, no Drive).

**Disk space:** you already have **DanceTrack + `dancetrack_yolo`** on disk (~22+ GB). CrowdHuman adds **~11 GB** of zips during download and briefly **another ~3 GB** while each zip is extracted and copied — the script now refuses to start unless there is **about ≥ 22 GiB free** on the same volume as `sway_pose_mvp/` (override with `--ignore-disk-check` only if you know you have room). **Free space first** (external drive, Trash, other large files), then run:

```bash
cd /path/to/sway_pose_mvp
pip install huggingface_hub
python scripts/phase2_public_training/fetch_crowdhuman_hf.py
```

If a previous run failed with **“No space left on device”**, free space, remove the partial folder `datasets/crowdhuman/Images/train/` (keep the `.odgt` files in `datasets/crowdhuman/`), delete `crowdhuman_hf/` if you want a clean re-download, and run the script again.

That pulls **[sshao0516/CrowdHuman](https://huggingface.co/datasets/sshao0516/CrowdHuman)** (community mirror with the **same** `annotation_*.odgt` and `CrowdHuman_train*.zip` / `val` names), unzips, and fills `datasets/crowdhuman/` the way `convert_crowdhuman_to_yolo.py` expects. By default it **deletes each `.zip` after** extracting (use `--keep-zips` to keep them). You still must follow **CrowdHuman’s terms** (non-commercial, no redistribution).

**Alternative — Baidu Netdisk (macOS):** install **one** Baidu client from your `.dmg` (you have two copies of the same installer, e.g. `BaiduNetdisk_mac_8.3.1_x64.dmg` and `…(1).dmg` — **pick one**, install it, delete the duplicate). Log in, use the **fetch codes** shown on the CrowdHuman download page for each file, and download the **six** items in §1.1 into the layout in §1.3.

### 1.1 What to download for *this* project

The training config here only uses **train** and **val**. You need **exactly these six** files (the HF script above gets all of them automatically):

| Download | Why |
|----------|-----|
| **`annotation_train.odgt`** | Training annotations (one JSON object per line; see below). |
| **`annotation_val.odgt`** | Validation annotations. |
| **`CrowdHuman_train01.zip`** | Training images (split 1 of 3). |
| **`CrowdHuman_train02.zip`** | Training images (split 2 of 3). |
| **`CrowdHuman_train03.zip`** | Training images (split 3 of 3). |
| **`CrowdHuman_val.zip`** | Validation images. |

**Do *not* download for this fine-tune (optional / other uses only):**

- **`CrowdHuman_test.zip`** — test-set images for CrowdHuman’s own benchmark. Our `dancetrack_crowdhuman.yaml` and `convert_crowdhuman_to_yolo.py` only need **`annotation_train` / `annotation_val`** and **`Images/train` / `Images/val`**, so **test is not used**.

### 1.2 What `.odgt` is (you don’t edit it)

Each line is one JSON with the whole annotation for one image. Our converter reads **`ID`** (image id / filename stem) and, for **`tag: "person"`**, the **`fbox`** `[x, y, w, h]` full-body box. Lines with **`tag: "mask"`** etc. are ignored for detection labels.

### 1.3 Lay out files under `sway_pose_mvp/` (manual / Baidu only)

If you used **`fetch_crowdhuman_hf.py`**, you can **skip** to §1.3 step 4 (checks) and then §1.3 step 5 (`convert_crowdhuman_to_yolo.py`). Otherwise:

1. Put the two annotation files **next to** `Images/`:

   ```text
   datasets/crowdhuman/
     annotation_train.odgt
     annotation_val.odgt
     Images/
       train/
       val/
   ```

2. **Training images:** unzip **`CrowdHuman_train01.zip`**, **`train02`**, and **`train03`** so that **every** training `.jpg` ends up in **one** folder:

   ```text
   datasets/crowdhuman/Images/train/*.jpg
   ```

   If a zip unpacks into subfolders, **move all `.jpg` files into `Images/train/`** (flat directory). The converter looks up `Images/train/<ID>.jpg` where `<ID>` matches the `"ID"` field in each `.odgt` line (same as the official layout).

3. **Validation images:** unzip **`CrowdHuman_val.zip`** so all val `.jpg` files are in:

   ```text
   datasets/crowdhuman/Images/val/*.jpg
   ```

   Again, flatten if the archive created extra directories.

4. Quick check:

   ```bash
   wc -l datasets/crowdhuman/annotation_train.odgt
   ls datasets/crowdhuman/Images/train | head
   ls datasets/crowdhuman/Images/val | head
   ```

5. Convert to YOLO format:

   ```bash
   python scripts/phase2_public_training/convert_crowdhuman_to_yolo.py
   ```

   Success ends with `Done. Output: .../datasets/crowdhuman_yolo`.

### 1.4 Terms of use (CrowdHuman)

By using the data you agree to their terms, including: **non-commercial research and educational use only**; **do not redistribute** the images; you accept responsibility for use. Read the full text on [their download page](https://www.crowdhuman.org/download.html). **DanceTrack** remains **non-commercial research** under [their README](https://github.com/DanceTrack/DanceTrack/blob/main/README.md) as well.

---

## 2. Verify layout (after CrowdHuman is in place)

```bash
python scripts/phase2_public_training/download_datasets.py
```

You want **exit code 0**. If anything is listed as missing, fix paths under `datasets/crowdhuman/` and run again.

---

## 3. Training machine — Python + CUDA

On the computer that will run training (typically Linux/Windows + NVIDIA GPU):

```bash
python3 --version    # 3.10+ is fine
```

Install **PyTorch with CUDA** using the command from [pytorch.org](https://pytorch.org) for your driver/GPU.

Then:

```bash
pip install -r requirements.txt
pip install huggingface_hub   # only if not already installed
```

Check GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If training **OOM**s, edit `train_yolo11x.py`: lower `BATCH` (e.g. `8`) or `IMGSZ`. Apple Silicon: set `DEVICE = "mps"` (slow). CPU-only is not practical.

---

## 4. Train, save weights, run the app

From `sway_pose_mvp/`:

```bash
python scripts/phase2_public_training/train_yolo11x.py
```

First run may download **`yolo11x.pt`** (~100+ MB). When it finishes:

```bash
mkdir -p models
cp runs/detect/yolo11x_dancetrack/weights/best.pt models/yolo11x_dancetrack.pt
```

**Optional** — compare COCO vs fine-tuned mAP (needs `yolo11x.pt` in repo root or let Ultralytics fetch it):

```bash
python scripts/phase2_public_training/validate_trained_model.py
```

**Use in pipeline:**

```bash
export SWAY_YOLO_WEIGHTS=models/yolo11x_dancetrack.pt
python main.py --video /path/to/your_video.mp4
```

Windows cmd: `set SWAY_YOLO_WEIGHTS=models\yolo11x_dancetrack.pt`

No change to `sway/tracker.py` — it reads `SWAY_YOLO_WEIGHTS` via `resolve_yolo_model_path()`.

---

## 5. If something fails

| Symptom | Action |
|--------|--------|
| `train_yolo11x.py` → `❌ Setup incomplete` | Run §1 step 4; ensure `datasets/crowdhuman_yolo/images/{train,val}/` exist. |
| CrowdHuman converter: `Converted 0 images` | Images must be `Images/<split>/<ID>.jpg` matching IDs in the `.odgt` lines. |
| `download_datasets.py` exit 1 | Fix whatever path it prints (usually CrowdHuman still missing or wrong names). |
| Training OOM | Smaller `BATCH` / `IMGSZ` in `train_yolo11x.py`. |

---

## 6. Done when

1. `download_datasets.py` exits **0**.  
2. `train_yolo11x.py` completed and `models/yolo11x_dancetrack.pt` exists.  
3. `SWAY_YOLO_WEIGHTS` points at that file and `main.py` looks good on your videos.

---

*If you ever need to redo DanceTrack from zero on another machine, use `fetch_dancetrack_hf.py` + `convert_dancetrack_to_yolo.py` — see `README.md` in this folder.*
