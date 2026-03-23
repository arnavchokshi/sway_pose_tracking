# Lambda GPU training — full copy-paste runbook

This runbook is for **GitHub Option A only**: you clone **`arnavchokshi/sway_pose_tracking`** on the Lambda machine. No `rsync` of the dataset from your Mac (DanceTrack is downloaded on the GPU box by script).

| Item | Value |
|------|--------|
| **GitHub repo (HTTPS)** | `https://github.com/arnavchokshi/sway_pose_tracking.git` |
| **GitHub repo (SSH)** | `git@github.com:arnavchokshi/sway_pose_tracking.git` |
| **Folder after `git clone`** | `~/sway_pose_tracking` (default clone directory name) |
| **Project root on Lambda** | `~/sway_pose_tracking` (must contain `main.py` and `sway/`) |

**Budget:** about **$15** is enough for **DanceTrack-only** training if you **terminate the instance as soon as `best.pt` is on your Mac**. At **~$1.99/hr** (typical **1× GH200**), that is **~7.5 hours** of wall time.

---

## 0. On your Mac — push the latest code to GitHub (do this first)

Lambda will clone whatever is on **`main`**. Commit and push **before** you SSH to Lambda.

Open **Terminal** on your Mac and run (your local folder is **`sway_pose_mvp`** inside `sway_test`; it is the same repo as `sway_pose_tracking` on GitHub):

```bash
cd /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp
git checkout main
git pull origin main
git status
```

If you have local changes to commit:

```bash
git add -A
git commit -m "Update training scripts and Lambda runbook"
git push origin main
```

If `git push` asks for credentials, use **GitHub CLI** (`gh auth login`), a **personal access token** (HTTPS), or switch the remote to **SSH** and use your Mac’s SSH key.

**Sanity check:** In a browser, open  
`https://github.com/arnavchokshi/sway_pose_tracking`  
and confirm you see `scripts/phase2_public_training/LAMBDA_TRAINING.md` on `main`.

---

## 1. Lambda account and billing

1. Go to **[Lambda Labs](https://lambda.ai/)** and sign up.
2. Add a **payment method** (card).
3. If the UI says you need a **GPU quota**, request it and wait if required.

---

## 2. Launch a GPU instance

1. Open **Lambda Cloud** → launch a new **GPU instance**.
2. Image: **Ubuntu** (22.04 or 24.04).
3. GPU: use what is available (often **1× GH200**, **ARM64**, **~96GB** VRAM). That hardware is **more than enough** for YOLO11x @ 960 in this repo.
4. Add your **SSH public key** from your Mac:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   If that file does not exist, create a key:

   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub
   ```

   Paste the **one-line** public key into Lambda’s **SSH keys** field when launching.

5. After launch, note from the Lambda dashboard:
   - **Instance IP** (example: `192.0.2.10`)
   - **SSH user** (almost always **`ubuntu`**)

**Replace placeholders below:**

- `<LAMBDA_IP>` → your instance’s IP (digits only, four dotted numbers).
- If Lambda ever shows a different SSH user, replace **`ubuntu`** with that username.

### GH200 / ARM64

- **`setup_lambda_training.sh`** installs **PyTorch CUDA** from the **`cu128`** wheel index on **`aarch64`**. Do not use generic `pip install torch` on GH200 without that index (you would get **CPU-only** builds).

---

## 3. On the Lambda machine — install Git and clone this repo (Option A only)

From your **Mac**, SSH in:

```bash
ssh ubuntu@<LAMBDA_IP>
```

On the **Lambda** shell, run **exactly**:

```bash
sudo apt-get update -y
sudo apt-get install -y git
cd ~
git clone https://github.com/arnavchokshi/sway_pose_tracking.git
cd ~/sway_pose_tracking
```

Verify the layout (both commands should list files, not “No such file”):

```bash
test -f main.py && test -d sway && echo "OK: repo root"
test -f scripts/phase2_public_training/train_yolo11x.py && echo "OK: training script"
```

### 3a. If the repo is private — use SSH clone instead

If `git clone https://github.com/...` fails with **authentication** or **404**, use SSH.

**On Lambda:**

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy the **full line** starting with `ssh-ed25519` into GitHub: **Settings → SSH and GPG keys → New SSH key** (title e.g. `lambda-gh200`).

Test:

```bash
ssh -T git@github.com
```

Clone:

```bash
cd ~
git clone git@github.com:arnavchokshi/sway_pose_tracking.git
cd ~/sway_pose_tracking
```

(Re-run the same **`test -f main.py`** checks as above.)

---

## 4. On Lambda — install PyTorch and run DanceTrack training

Stay in the project root **`~/sway_pose_tracking`**.

**Recommended:** use **`tmux`** so a dropped laptop Wi‑Fi does not kill training.

```bash
sudo apt-get install -y tmux
tmux new -s train
cd ~/sway_pose_tracking
chmod +x scripts/phase2_public_training/setup_lambda_training.sh
chmod +x scripts/phase2_public_training/run_lambda_train_dancetrack.sh
bash scripts/phase2_public_training/setup_lambda_training.sh
bash scripts/phase2_public_training/run_lambda_train_dancetrack.sh
```

- **Detach** from tmux (session keeps running): press **Ctrl+B**, then **D**.
- **Reattach** later: `tmux attach -t train`

**What the second script does:** downloads **DanceTrack** from Hugging Face on the instance → `convert_dancetrack_to_yolo.py` → `train_yolo11x.py --phase dancetrack`.

**When finished**, weights path on Lambda:

```text
~/sway_pose_tracking/runs/detect/yolo11x_dancetrack_only/weights/best.pt
```

---

## 5. On your Mac — copy `best.pt` off Lambda

Open a **new Terminal tab on your Mac** (not inside SSH). Run:

```bash
scp ubuntu@<LAMBDA_IP>:~/sway_pose_tracking/runs/detect/yolo11x_dancetrack_only/weights/best.pt \
  ~/Desktop/yolo11x_dancetrack_only_best.pt
```

Install the weights into your local project and point the app at them:

```bash
mkdir -p /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/models
mv ~/Desktop/yolo11x_dancetrack_only_best.pt \
  /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/models/yolo11x_dancetrack.pt
export SWAY_YOLO_WEIGHTS=/Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp/models/yolo11x_dancetrack.pt
```

Run your pipeline as usual, for example:

```bash
cd /Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp
python main.py --video /path/to/your_video.mp4
```

---

## 6. Stop billing (mandatory)

In the **Lambda web UI**, **terminate** (or stop) the instance as soon as you have **`best.pt`** on your Mac. Idle GPU time still charges **$/hr**.

---

## Troubleshooting

### ARM64 / GH200: `torch` is CPU-only (`+cpu`, `cuda_available=False`)

```bash
python3 -m pip uninstall -y torch torchvision
python3 -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If **`cu128`** fails, try **`cu126`** in both URLs (same command shape).

### x86_64 Lambda instances only

Edit `scripts/phase2_public_training/setup_lambda_training.sh`: for the x86 branch, try **`cu121`** instead of **`cu124`** if installs fail.

### Training OOM (rare on 96GB, but possible)

Edit `scripts/phase2_public_training/train_yolo11x.py`: set **`BATCH = 8`** (or **`4`**) and/or lower **`IMGSZ`**.

### Hugging Face

Usually **no token** is needed for `noahcao/dancetrack`. If you hit rate limits:

```bash
export HF_TOKEN=<your_read_token>
```

Token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Missing `python3`

```bash
sudo apt-get install -y python3 python3-pip
```

---

## Later: CrowdHuman second stage (same clone layout)

On Lambda, from **`~/sway_pose_tracking`**, after you have run CrowdHuman fetch/convert per **`REMAINING_STEPS.md`**:

```bash
cd ~/sway_pose_tracking
python3 scripts/phase2_public_training/train_yolo11x.py --phase crowdhuman \
  --weights runs/detect/yolo11x_dancetrack_only/weights/best.pt
```

---

## License reminder

**DanceTrack** / **CrowdHuman** are **non-commercial research** datasets. Read their terms before use.
