# DINOv2/DINOv3 Fast-Track (7-Hour Training)

**Time box:** ~5h (e.g. flight) + ~2h (e.g. hotel) — details in the master plan.

## Steps at a glance (do in order)

1. **Read the plan** — `docs/dinov2_dinov3_master_training_plan.md` (phases, schedule, dataset links, KPIs).
2. **Create the environment** — use [Quick launch](#quick-launch) below; in Cursor pick the **`Python (dino-fasttrack)`** kernel for each `.ipynb` (no need to run `jupyter notebook`).
3. **`00_environment_and_sanity_checks.ipynb`** — verify CPU/GPU, cache folders, download CIFAR-10, load DINO (v3-ready fallback to v2), run one embedding sanity check.
4. **`01_dinov2_image_classification_and_embeddings.ipynb`** — frozen DINO embeddings → linear classifier → metrics + nearest-neighbor retrieval; saves artifacts under `artifacts/`.
5. **`02_detection_segmentation_video_and_defect_blueprint.ipynb`** — detection/segmentation baselines, optional video + frame anomaly scores, enterprise defect-pipeline blueprint. Put test images in `data/images/` and videos in `data/video/`.
6. **Hotel / follow-up** — re-run notebooks cleanly, tune thresholds, swap in your phone-transit images per the plan.

**Dataset and test-file URLs** live in the master plan (CIFAR, Beans, MVTec, COCO, Penn-Fudan, video samples).

## Quick launch

### Why not only `requirements.txt`?

`requirements.txt` used to list `torch` / `torchvision` / `torchaudio` without a **PyTorch wheel index**. On many machines `pip install -r requirements.txt` then pulls **default CPU builds** from PyPI, or a CUDA version that doesn’t match your setup — so an RTX 8000 would silently run on CPU or fail oddly.

**Recommended:** install **PyTorch in one explicit step** (GPU or CPU), then install everything else from `requirements.txt`.

### Conda: Terms of Service error

If `conda create` fails with *Terms of Service have not been accepted*, either run the accept commands Conda prints (e.g. `conda tos accept ...` for each channel), or use **Miniforge** / **micromamba** as an alternative. After the env exists, the pip steps below are the same.

### Git Bash: `CondaError: Run 'conda init' before 'conda activate'`

Git Bash doesn’t load Conda’s shell hooks until you initialize it **once**:

```bash
conda init bash
```

Close the terminal, open a **new** Git Bash window, then `conda activate dino-fasttrack` should work.

**Alternatives if you prefer not to use `conda init`:**

- Use **PowerShell** or **cmd.exe** (Anaconda often adds `conda` there already).
- Or run commands without activating, e.g.  
  `conda run -n dino-fasttrack pip install --upgrade pip`  
  and  
  `conda run -n dino-fasttrack pip install -r requirements.txt`

### “I don’t have CUDA installed” — what you actually need

- **You do not need a separate NVIDIA CUDA Toolkit** on your machine for typical PyTorch use. The **GPU PyTorch pip package** (e.g. `cu121`) includes the **CUDA runtime** it needs.
- **For an NVIDIA GPU (e.g. RTX 8000):** install/update the **NVIDIA graphics driver** (from NVIDIA’s site or Windows Update). Run `nvidia-smi` — if it shows your GPU, you’re in good shape for GPU PyTorch after installing the `cu121` wheels.
- **If there is no NVIDIA GPU, or drivers aren’t installed / are too old:** use **CPU PyTorch** ([path B below](#2-install-pytorch--pick-one-path)). Training is slower; all notebooks still run.

### 1) Create environment

```bash
conda create -n dino-fasttrack python=3.10 -y
conda activate dino-fasttrack
pip install --upgrade pip
```

### 2) Install PyTorch — pick **one** path

**A — NVIDIA GPU (RTX 8000): CUDA 12.1 wheels** (good default; driver must be recent enough — see below)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**B — CPU only** (always works; slower training — notebooks still run)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3) Rest of the stack

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name dino-fasttrack --display-name "Python (dino-fasttrack)"
```

**Cursor / VS Code:** open any `.ipynb`, use the kernel picker (top-right) and choose **`Python (dino-fasttrack)`**. You do **not** need to start `jupyter notebook` in a terminal.

**Optional — browser Jupyter only:** if you ever want the classic web UI, install separately:  
`pip install "notebook>=6.5,<7"` then run `jupyter notebook`.

### Windows: `pip install` fails on a very long path under `...\jupyter\labextensions\...`

That happens when **`jupyter`** / **JupyterLab** (heavy `labextensions` trees) is installed. **This repo’s `requirements.txt` does not install those** — only **`ipykernel`** for Cursor/VS Code — so you usually avoid the issue entirely.

If you still hit it (e.g. you installed `jupyter` yourself):

**Option A — Permanent fix (admin):** enable **Win32 long paths** (same as [Microsoft / pip hint](https://pip.pypa.io/warnings/enable-long-paths)), then reinstall.

**Option B — Avoid the stack:** uninstall the metapackage and use Cursor only:

```bash
pip uninstall jupyter jupyterlab notebook jupyterlab-widgets -y
pip install -r requirements.txt
```

### Drivers / CUDA vs “fallback to CPU”

- **GPU:** Install/update **NVIDIA drivers** from NVIDIA (Windows: GeForce/RTX driver package or Studio). In a terminal run `nvidia-smi` — if it works, the driver is loaded.
- **PyTorch CUDA wheels** (e.g. `cu121`) ship their own CUDA **runtime**; you mainly need a **driver** new enough for that stack. If `torch.cuda.is_available()` is `False` after installing GPU wheels, either update the driver or use path **B** (CPU) and everything still runs.
- Notebooks use `device = 'cuda' if torch.cuda.is_available() else 'cpu'` — **automatic CPU fallback** if CUDA isn’t available.
