# Lambda training — what you still need to do

Assume **this repo is already on GitHub** (`main`):  
`https://github.com/arnavchokshi/sway_pose_tracking`  
Clone on the GPU box goes to **`~/sway_pose_tracking`**. DanceTrack is **downloaded on Lambda** by script (nothing huge from your Mac).

**Replace `<LAMBDA_IP>`** with your instance **IP Address** from the Lambda dashboard. **SSH user** is almost always **`ubuntu`** (the dashboard’s **SSH Login** line often shows `ubuntu@<ip>` once the VM is ready).

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
cd /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp
git checkout main && git pull origin main
git add -A && git commit -m "Your message"   # if you have changes
git push origin main
```

---

## On each run (do in order)

### 1. Lambda dashboard

Launch an **Ubuntu** GPU instance (e.g. **`gpu_1x_a10`**, **GH200**, or whatever is available). Attach SSH key **`pose-tracking`**. When **Booting** finishes, copy **`<LAMBDA_IP>`**.

**Hardware note:** **A10** instances are **x86_64**; **GH200** is **ARM64**. The repo’s `setup_lambda_training.sh` picks the right PyTorch wheels automatically. **A10 24GB** is enough for YOLO11x @ 960; if training **OOM**s, lower `BATCH` or `IMGSZ` in `train_yolo11x.py`.

### 2. Mac → SSH into the instance

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem ubuntu@<LAMBDA_IP>
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

`~/sway_pose_tracking/runs/detect/yolo11x_dancetrack_only/weights/best.pt`

**Optional — validation metrics on the instance** (DanceTrack val mAP vs COCO baseline; needs the same `datasets/dancetrack_yolo/` the training script just built):

```bash
cd ~/sway_pose_tracking
python3 scripts/phase2_public_training/validate_trained_model.py
# or explicit checkpoint:
# python3 scripts/phase2_public_training/validate_trained_model.py \
#   --weights runs/detect/yolo11x_dancetrack_only/weights/best.pt
```

### 4. Mac — download weights

New Terminal tab on your Mac:

```bash
scp -i /Users/arnavchokshi/Downloads/pose-tracking.pem \
  ubuntu@<LAMBDA_IP>:~/sway_pose_tracking/runs/detect/yolo11x_dancetrack_only/weights/best.pt \
  ~/Desktop/yolo11x_dancetrack_only_best.pt
```

### 5. Mac — install weights for the app

```bash
mkdir -p /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/models
mv ~/Desktop/yolo11x_dancetrack_only_best.pt \
  /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/models/yolo11x_dancetrack.pt
export SWAY_YOLO_WEIGHTS=/Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/models/yolo11x_dancetrack.pt
```

**Optional — same validation on your Mac** (needs `datasets/dancetrack_yolo/` present locally, or only CrowdHuman / merged if you converted those):

```bash
cd /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp
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
| Training OOM | Edit `scripts/phase2_public_training/train_yolo11x.py`: lower `BATCH` or `IMGSZ`. |
| Hugging Face rate limit | On the instance: `export HF_TOKEN=<read token>` |

Dataset terms: **DanceTrack** / **CrowdHuman** are **non-commercial research** — see each dataset’s license.
