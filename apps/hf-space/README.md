---
title: GS Mapper — Photos to 3D Gaussian Splat
emoji: 🗺️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Photos or a short video -> browser-ready 3DGS .splat
models:
  - naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
tags:
  - gaussian-splatting
  - 3d-reconstruction
  - robotics
---

# GS Mapper — Photos to 3D Gaussian Splat

Zero-install demo for [gs-mapper](https://github.com/rsasaki0109/gs-mapper):
upload 8–16 photos (or a short walkaround video) and get a browser-viewable
3D Gaussian Splat. Pose-free via DUSt3R — no COLMAP, no GPU setup.

## How it works

1. Uploads are normalized to bounded-size JPEGs (`pipeline.prepare_images`).
2. DUSt3R estimates poses + a fused point cloud (`gs_sim2real.preprocess.pose_free`).
3. gsplat trains a draft-quality 3DGS scene (`gs_sim2real.train.gsplat_trainer`).
4. The result is exported to the antimatter15 `.splat` binary
   (`gs_sim2real.viewer.web_export.ply_to_splat`) and rendered in `gr.Model3D`.

## Deploying / updating this Space

This directory is the Space contents; it is synced from the main repo by
`.github/workflows/sync-hf-space.yml` (set the `HF_TOKEN` secret, and
optionally the `HF_SPACE_ID` repository variable, in the GitHub repo).

Hardware notes:

- **ZeroGPU**: works out of the box (`spaces.GPU` is applied when available);
  tune the per-call budget via the `GS_MAPPER_GPU_DURATION` env var.
- **Custom GPU**: keep the `gsplat` wheel index in `requirements.txt` in sync
  with the torch/CUDA combo of the hardware.
- **Local**: `pip install -r requirements.txt && python app.py`. Headless
  smoke test without gradio: `python pipeline.py --images ./photos --output /tmp/out`.
