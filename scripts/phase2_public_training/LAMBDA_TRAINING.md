# Lambda training — what you still need to do

Assume **this repo is already on GitHub** (`main`):  
`https://github.com/arnavchokshi/sway_pose_tracking`  
Clone on the GPU box goes to **`~/sway_pose_tracking`**. DanceTrack is **downloaded on Lambda** by script (nothing huge from your Mac).

**SSH user** is almost always **`ubuntu`** (the dashboard’s **SSH Login** line often shows `ubuntu@<ip>` once the VM is ready).

**Current instance IP (update this when you launch a new VM):** **`132.145.211.165`** — the commands below use this address. If SSH fails, open the Lambda dashboard and replace it with the **IP Address** shown there (new instance ⇒ new IP).

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

---

## On each run (do in order)

### 1. Lambda dashboard

Launch an **Ubuntu** GPU instance (e.g. **`gpu_1x_a10`**, **GH200**, or whatever is available). Attach SSH key **`pose-tracking`**. When **Booting** finishes, copy the IP into the **Current instance IP** line at the top of this file (and into your commands) if it changed.

**Hardware note:** **A10** instances are **x86_64**; **GH200** is **ARM64**. The repo’s `setup_lambda_training.sh` picks the right PyTorch wheels automatically. **A10 24GB** is usually enough for YOLO26l @ 960; if training **OOM**s, lower `BATCH` or `IMGSZ` in `train_yolo26l.py`. Use a recent **ultralytics** (YOLO26); after `git pull`, run `.venv/bin/pip install -U ultralytics` on the instance if `yolo26l.pt` fails to load.

### 2. Mac → SSH into the instance

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem ubuntu@132.145.211.165
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
         scripts/phase2_public_training/run_lambda_train_dancetrack.sh
bash scripts/phase2_public_training/setup_lambda_training.sh
bash scripts/phase2_public_training/run_lambda_train_dancetrack.sh
```

- Detach tmux (session keeps running): **Ctrl+B**, then **D**.  
- Reattach: `tmux attach -t train`

Weights when done:

`~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt`

**Quick status from your Mac** (one-shot, safe in Cursor — replace IP if the dashboard changed):

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem -o ConnectTimeout=15 ubuntu@132.145.211.165 \
  'date -u; pgrep -c -f train_yolo26l || true; ls -la ~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt 2>/dev/null || echo "best.pt: not yet"; tail -n 3 ~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/results.csv 2>/dev/null || echo "results.csv: not yet"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

**Progress / metrics on the instance:**

- **Full terminal log:** `~/sway_pose_tracking/training_full_<timestamp>.log` (everything: fetch, convert, every epoch). Each run also updates **`~/sway_pose_tracking/training_full_latest.log`** (symlink to the active run) so you do not pick an old file by mistake.
- **Per-epoch CSV:** `~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/results.csv` (losses, mAP — open or `tail -f` while training).
- Second SSH session: `tail -f ~/sway_pose_tracking/training_full_latest.log` (or tab-complete a dated `training_full_*.log`). **Avoid `tail -f` inside Cursor’s terminal** — it streams forever and can freeze the editor; use **macOS Terminal** or the **one-shot Mac commands** below.

**Last lines of the full log (Cursor-safe)** — always take the **end** of the stream on your Mac (`tail -n`), not `head -c`. **`head -c N` keeps the first N bytes**, so if anything streams from the start of the file you will see the beginning of training (wrong) instead of the latest tqdm lines.

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem -o ConnectTimeout=15 -o BatchMode=yes -T ubuntu@132.145.211.165 'tail -n 20 "$HOME/sway_pose_tracking/training_full_latest.log" 2>/dev/null || { f=$(find "$HOME/sway_pose_tracking" -maxdepth 1 -name "training_full_*.log" -type f -printf "%T@\t%p\n" 2>/dev/null | sort -n | tail -1 | cut -f2-); [ -n "$f" ] && tail -n 20 "$f"; }' | tail -n 25
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

New Terminal tab on your Mac:

```bash
scp -i /Users/arnavchokshi/Downloads/pose-tracking.pem \
  ubuntu@132.145.211.165:~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt \
  ~/Desktop/yolo26l_dancetrack_only_best.pt
```

### 5. Mac — install weights for the app

```bash
# Set SWAY_REPO to your local sway_pose_mvp directory (the folder that contains main.py).
export SWAY_REPO="$HOME/path/to/sway_pose_mvp"
mkdir -p "$SWAY_REPO/models"
mv ~/Desktop/yolo26l_dancetrack_only_best.pt \
  "$SWAY_REPO/models/yolo26l_dancetrack.pt"
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
