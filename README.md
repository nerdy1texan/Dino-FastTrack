# DINOv2/DINOv3 Fast-Track (7-Hour Training)

**Time box:** ~5h (e.g. flight) + ~2h (e.g. hotel) — details in the master plan.

## Steps at a glance (do in order)

1. **Read the plan** — `docs/dinov2_dinov3_master_training_plan.md` (phases, schedule, dataset links, KPIs).
2. **Create the environment** — use [Quick launch](#quick-launch) below; pick the `Python (dino-fasttrack)` kernel in Jupyter.
3. **`00_environment_and_sanity_checks.ipynb`** — verify CPU/GPU, cache folders, download CIFAR-10, load DINO (v3-ready fallback to v2), run one embedding sanity check.
4. **`01_dinov2_image_classification_and_embeddings.ipynb`** — frozen DINO embeddings → linear classifier → metrics + nearest-neighbor retrieval; saves artifacts under `artifacts/`.
5. **`02_detection_segmentation_video_and_defect_blueprint.ipynb`** — detection/segmentation baselines, optional video + frame anomaly scores, enterprise defect-pipeline blueprint. Put test images in `data/images/` and videos in `data/video/`.
6. **Hotel / follow-up** — re-run notebooks cleanly, tune thresholds, swap in your phone-transit images per the plan.

**Dataset and test-file URLs** live in the master plan (CIFAR, Beans, MVTec, COCO, Penn-Fudan, video samples).

## Quick launch

```bash
conda create -n dino-fasttrack python=3.10 -y
conda activate dino-fasttrack
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name dino-fasttrack --display-name "Python (dino-fasttrack)"
jupyter notebook
```

Use the `Python (dino-fasttrack)` kernel in each notebook.
# Dino-FastTrack
