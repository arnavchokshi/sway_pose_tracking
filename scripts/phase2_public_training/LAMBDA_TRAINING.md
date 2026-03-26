# Lambda training — what you still need to do

Assume **this repo is already on GitHub** (`main`):  
`https://github.com/arnavchokshi/sway_pose_tracking`  
Clone on the GPU box goes to **`~/sway_pose_tracking`**. DanceTrack is **downloaded on Lambda** by script (nothing huge from your Mac).

**SSH user** is almost always **`ubuntu`** (the dashboard’s **SSH Login** line often shows `ubuntu@<ip>` once the VM is ready).

**Current instance IP (update this when you launch a new VM):** **`150.136.6.38`** (`gpu_1x_a10`, **us-east-1**) — the commands below use this address. If SSH fails, open the Lambda dashboard and replace it with the **IP Address** shown there (new instance ⇒ new IP).

**While status is “Booting”:** IP and SSH fields may show **—**. Wait until the instance is **running**; then refresh or reopen the instance page and copy the IP.

---

## First time only (skip if already done)

1. **Lambda:** account, payment method, GPU quota if the site asks.
2. **Lambda → SSH keys:** add your **public** key (one line, no breaks). **Skip** if `pose-tracking` is already listed.

   ```
   ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDCUPXx7ipTmzI6jXU5Ond6OU5J6sYnUXdYvfzO0ka1FsAeZ3JdMuL7XM5lPEi62/LAliw2oxHedCiWQ1nHhThk90GyNXp0Vl5h/54hfA24kAyU7NJMX0iqTHhjD66q8StQjg5uI6INnHpWZd5eUAtH4M0Bz+HW7vcuKMvyWeQeYkhOFlHorYbAe8Jh1lLa9imhKdUpckaYkhsfiGTIeFiuu02Iyk5CJAw06nI5bqGm1n/ddBEHvolJ/y8jvFwYW0Phw4jGQZ8MYAw/Bl7rydnZ6Xu5V6eOBxjuuB4w2CBvPJRplX1592rOeMMpOH7L5XQgdIDVy9CdL1K3mgsnJXaR pose-tracking
   ```

   Must match the **private** file on your Mac: **`/Users/arnavchokshi/Downloads/pose-tracking.pem`**. Never upload the `.pem` to Lambda.

3. **Mac — key permissions (once):**

   ```bash
   chmod 400 /Users/arnavchokshi/Downloads/pose-tracking.pem
   ```

---

## Before each training run

**Only if you changed code locally** — push so Lambda gets the latest:

```bash
cd /path/to/sway_pose_mvp   # your local clone (contains main.py)
git checkout main && git pull origin main
git add -A && git commit -m "Your message"   # if you have changes
git push origin main
```

### CrowdHuman only — single command on Lambda (fresh box)

You do **not** need CrowdHuman on your Mac. On the GPU instance, after `git clone` / `git pull` of this repo:

1. **Upload your DanceTrack `best.pt`** (you already trained this) so the CrowdHuman stage can fine-tune from it:

   ```bash
   mkdir -p ~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights
   # From your Mac (replace IP and local path):
   scp /path/to/your_dancetrack_best.pt ubuntu@<LAMBDA_IP>:~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt
   ```

   Or put the file anywhere on Lambda and set `YOLO_CROWDHUMAN_PARENT_WEIGHTS=/absolute/path/to/best.pt` in the shell before the one-shot.

2. **Disk:** use a volume with enough free space (CrowdHuman fetch recommends **≥ ~22 GiB** free before download).

3. **One command** from `~/sway_pose_tracking` (installs venv + PyTorch, downloads CrowdHuman from Hugging Face, converts to YOLO, trains):

   ```bash
   cd ~/sway_pose_tracking && bash scripts/phase2_public_training/run_lambda_crowdhuman_one_shot.sh
   ```

   Output weights: `~/sway_pose_tracking/runs/detect/yolo26l_crowdhuman_ft/weights/best.pt` — copy to your Mac as `models/yolo26l_dancetrack.pt` (see “Weights when done” below).

**Logs & recovery**

- **One-shot session log:** `~/sway_pose_tracking/training_one_shot_latest.log` (and a dated `training_one_shot_*.log`).
- **Pipeline log (fetch + train):** `~/sway_pose_tracking/training_full_latest.log` plus per-run `training_full_*.log`.
- **Per-run manifest:** `runs/detect/<run_name>/training_manifest.json` (hyperparameters, git hash).
- **Resume training** if the job died but `runs/detect/<run_name>/weights/last.pt` exists:  
  `YOLO_TRAIN_RESUME=1 .venv/bin/python -u scripts/phase2_public_training/train_yolo26l.py --phase crowdhuman --resume`  
  (use the same `--phase` as before; `--resume-from` optional if `last.pt` is elsewhere).

---

## On each run (do in order)

### 1. Lambda dashboard

Launch an **Ubuntu** GPU instance (e.g. **`gpu_1x_a10`**, **GH200**, or whatever is available). Attach SSH key **`pose-tracking`**. When **Booting** finishes, copy the IP into the **Current instance IP** line at the top of this file (and into your commands) if it changed.

**Hardware note:** **A10** instances are **x86_64**; **GH200** is **ARM64**. The repo’s `setup_lambda_training.sh` picks the right PyTorch wheels automatically. **A10 24GB** is usually enough for YOLO26l @ 960; if training **OOM**s, lower `BATCH` or `IMGSZ` in `train_yolo26l.py`. Use a recent **ultralytics** (YOLO26); after `git pull`, run `.venv/bin/pip install -U ultralytics` on the instance if `yolo26l.pt` fails to load.

### 2. Mac → SSH into the instance

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem ubuntu@150.136.6.38
```

### 3. On the instance — clone repo and train

```bash
sudo apt-get update -y
sudo apt-get install -y git tmux
cd ~
# If you already cloned on this disk from a previous session, skip clone and run: cd ~/sway_pose_tracking && git pull origin main
git clone https://github.com/arnavchokshi/sway_pose_tracking.git
cd ~/sway_pose_tracking
tmux new -s train
chmod +x scripts/phase2_public_training/setup_lambda_training.sh \
         scripts/phase2_public_training/run_lambda_yolo_train.sh \
         scripts/phase2_public_training/run_lambda_train_dancetrack.sh \
         scripts/phase2_public_training/run_lambda_train_dancetrack_crowdhuman.sh \
         scripts/phase2_public_training/run_lambda_train_crowdhuman.sh
bash scripts/phase2_public_training/setup_lambda_training.sh
bash scripts/phase2_public_training/run_lambda_train_dancetrack.sh
```

- **DanceTrack + CrowdHuman in one go** (recommended if you want the same `yolo26l_dancetrack.pt` slot in the app, but weights include CrowdHuman fine-tune): after `setup_lambda_training.sh`, run  
  `bash scripts/phase2_public_training/run_lambda_train_dancetrack_crowdhuman.sh`  
  This runs DanceTrack fetch/convert/train, then CrowdHuman (HF) fetch/convert/train. **Disk:** CrowdHuman adds several tens of GB — use a large volume; `fetch_crowdhuman_hf.py` aborts if free space is too low.
- **CrowdHuman only** (same instance, DanceTrack already finished and `yolo26l_dancetrack_only/weights/best.pt` exists):  
  `bash scripts/phase2_public_training/run_lambda_train_crowdhuman.sh`  
  Or: `YOLO_LAMBDA_PIPELINE=crowdhuman bash scripts/phase2_public_training/run_lambda_yolo_train.sh`
- **CrowdHuman only, DanceTrack weights from your machine:** `scp` your DanceTrack `best.pt` to the instance (e.g. under `~/sway_pose_tracking/`), then either move it to `runs/detect/yolo26l_dancetrack_only/weights/best.pt`, or set **`YOLO_CROWDHUMAN_PARENT_WEIGHTS=/absolute/path/to/your_best.pt`** when running the CrowdHuman pipeline above.

- Detach tmux (session keeps running): **Ctrl+B**, then **D**.  
- Reattach: `tmux attach -t train`

Weights when done:

- DanceTrack-only run: `~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt`
- After CrowdHuman stage: **`~/sway_pose_tracking/runs/detect/yolo26l_crowdhuman_ft/weights/best.pt`** — copy this to your Mac as `models/yolo26l_dancetrack.pt` (same env var / Lab token as before).

**Quick status from your Mac** (one-shot, safe in Cursor — replace IP if the dashboard changed):

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem -o ConnectTimeout=15 ubuntu@150.136.6.38 \
  'date -u; pgrep -c -f train_yolo26l || true; ls -la ~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt 2>/dev/null || echo "best.pt: not yet"; tail -n 3 ~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/results.csv 2>/dev/null || echo "results.csv: not yet"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

**Progress / metrics on the instance:**

- **Full terminal log:** `~/sway_pose_tracking/training_full_<timestamp>.log` (everything: fetch, convert, every epoch). Each run also updates **`~/sway_pose_tracking/training_full_latest.log`** (symlink to the active run) so you do not pick an old file by mistake.
- **Per-epoch CSV:** `~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/results.csv` (losses, mAP — open or `tail -f` while training).
- Second SSH session: `tail -f ~/sway_pose_tracking/training_full_latest.log` (or tab-complete a dated `training_full_*.log`). **Avoid `tail -f` inside Cursor’s terminal** — it streams forever and can freeze the editor; use **macOS Terminal** or the **one-shot Mac commands** below.

**Last lines of the full log (Cursor-safe)** — always take the **end** of the stream on your Mac (`tail -n`), not `head -c`. **`head -c N` keeps the first N bytes**, so if anything streams from the start of the file you will see the beginning of training (wrong) instead of the latest tqdm lines.

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem -o ConnectTimeout=15 -o BatchMode=yes -T ubuntu@150.136.6.38 'tail -n 20 "$HOME/sway_pose_tracking/training_full_latest.log" 2>/dev/null || { f=$(find "$HOME/sway_pose_tracking" -maxdepth 1 -name "training_full_*.log" -type f -printf "%T@\t%p\n" 2>/dev/null | sort -n | tail -1 | cut -f2-); [ -n "$f" ] && tail -n 20 "$f"; }' | tail -n 25
```

If `training_full_latest.log` is missing (run started before that symlink existed), the `find … sort -n | tail -1` branch picks the log with the **newest mtime**. Pipe through **`tail -n 25` locally** so Cursor never renders more than 25 lines even if the remote misbehaves.

Setup creates **`~/sway_pose_tracking/.venv`** so training does not use the system `python3` (avoids broken torch/numpy mixes on the Lambda image).

**Optional — validation metrics on the instance** (DanceTrack val mAP vs COCO baseline; needs the same `datasets/dancetrack_yolo/` the training script just built):

```bash
cd ~/sway_pose_tracking
.venv/bin/python scripts/phase2_public_training/validate_trained_model.py
# or explicit checkpoint:
# python3 scripts/phase2_public_training/validate_trained_model.py \
#   --weights runs/detect/yolo26l_dancetrack_only/weights/best.pt
```

### 4. Mac — download weights

New Terminal tab on your Mac.

**DanceTrack-only** (`run_lambda_train_dancetrack.sh`):

```bash
scp -i /Users/arnavchokshi/Downloads/pose-tracking.pem \
  ubuntu@150.136.6.38:~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt \
  ~/Desktop/yolo26l_dancetrack_only_best.pt
```

**After CrowdHuman fine-tune** (`run_lambda_train_dancetrack_crowdhuman.sh` or `run_lambda_train_crowdhuman.sh`):

```bash
scp -i /Users/arnavchokshi/Downloads/pose-tracking.pem \
  ubuntu@150.136.6.38:~/sway_pose_tracking/runs/detect/yolo26l_crowdhuman_ft/weights/best.pt \
  ~/Desktop/yolo26l_crowdhuman_ft_best.pt
```

### 5. Mac — install weights for the app

```bash
# Set SWAY_REPO to your local sway_pose_mvp directory (the folder that contains main.py).
export SWAY_REPO="$HOME/path/to/sway_pose_mvp"
mkdir -p "$SWAY_REPO/models"
# Use the Desktop filename from step 4 (DanceTrack-only or CrowdHuman-stage — same destination name):
mv ~/Desktop/yolo26l_dancetrack_only_best.pt "$SWAY_REPO/models/yolo26l_dancetrack.pt"
# If you scp'd CrowdHuman weights instead: mv ~/Desktop/yolo26l_crowdhuman_ft_best.pt "$SWAY_REPO/models/yolo26l_dancetrack.pt"
export SWAY_YOLO_WEIGHTS="$SWAY_REPO/models/yolo26l_dancetrack.pt"
```

**Optional — same validation on your Mac** (needs `datasets/dancetrack_yolo/` present locally, or only CrowdHuman / merged if you converted those):

```bash
cd "$SWAY_REPO"
python3 scripts/phase2_public_training/validate_trained_model.py
```

### 6. Lambda dashboard — stop charges

**Terminate** the instance once `best.pt` is on your Mac.

---

## If something fails

| Problem | What to do |
|--------|----------------|
| `git clone https://github.com/...` auth / 404 | Repo is public; if it becomes private, add an SSH key **on the instance** to GitHub and use `git clone git@github.com:arnavchokshi/sway_pose_tracking.git`. |
| `torch` shows `+cpu` or `cuda_available=False` | **GH200 (ARM64):** `python3 -m pip uninstall -y torch torchvision` then `python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple`. **A10 / x86:** same command but use **`cu124`** instead of **`cu128`** in the URL, or re-run `setup_lambda_training.sh`. |
| Training OOM | Edit `scripts/phase2_public_training/train_yolo26l.py`: lower `BATCH` or `IMGSZ`. |
| Hugging Face rate limit | On the instance: `export HF_TOKEN=<read token>` |

Dataset terms: **DanceTrack** / **CrowdHuman** are **non-commercial research** — see each dataset’s license.
