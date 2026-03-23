# Lambda training — what you still need to do

Assume **this repo is already on GitHub** (`main`):  
`https://github.com/arnavchokshi/sway_pose_tracking`  
Clone on the GPU box goes to **`~/sway_pose_tracking`**. DanceTrack is **downloaded on Lambda** by script (nothing huge from your Mac).

**Replace `<LAMBDA_IP>`** with your instance IP. **SSH user** is usually **`ubuntu`**.

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

Launch an **Ubuntu** GPU instance (e.g. **GH200**). Copy **`<LAMBDA_IP>`**.

### 2. Mac → SSH into the instance

```bash
ssh -i /Users/arnavchokshi/Downloads/pose-tracking.pem ubuntu@<LAMBDA_IP>
```

### 3. On the instance — clone repo and train

```bash
sudo apt-get update -y
sudo apt-get install -y git tmux
cd ~
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

### 6. Lambda dashboard — stop charges

**Terminate** the instance once `best.pt` is on your Mac.

---

## If something fails

| Problem | What to do |
|--------|----------------|
| `git clone https://github.com/...` auth / 404 | Repo is public; if it becomes private, add an SSH key **on the instance** to GitHub and use `git clone git@github.com:arnavchokshi/sway_pose_tracking.git`. |
| `torch` shows `+cpu` or `cuda_available=False` on **GH200** | On the instance: `python3 -m pip uninstall -y torch torchvision` then `python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple` |
| Training OOM | Edit `scripts/phase2_public_training/train_yolo11x.py`: lower `BATCH` or `IMGSZ`. |
| Hugging Face rate limit | On the instance: `export HF_TOKEN=<read token>` |

Dataset terms: **DanceTrack** / **CrowdHuman** are **non-commercial research** — see each dataset’s license.
