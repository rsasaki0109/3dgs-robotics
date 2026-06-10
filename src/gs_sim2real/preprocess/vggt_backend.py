"""VGGT feedforward pose-free backend.

Uses Meta's `facebook/VGGT-1B` model (CVPR 2025) for one-pass camera + depth
estimation, then exports a COLMAP-text sparse model compatible with the gsplat
trainer. This is **not** the repo-external VGGT-SLAM 2.0 artifact importer;
it runs VGGT inside GS Mapper's pose-free preprocess path.

Requirements:
- Local clone of https://github.com/facebookresearch/vggt on ``VGGT_PATH``
  (default ``/tmp/vggt``).
- A CUDA GPU is strongly recommended; CPU inference is supported but slow.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_VGGT_ROOT = Path(os.environ.get("VGGT_PATH", "/tmp/vggt"))
_DEFAULT_VGGT_HUB_ID = "facebook/VGGT-1B"
_DEFAULT_VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_VGGT_INFERENCE_RESOLUTION = 518
_VGGT_LOAD_RESOLUTION = 1024


def _add_vggt_to_path(vggt_root: Path) -> None:
    if not vggt_root.exists():
        raise FileNotFoundError(
            f"VGGT clone not found at {vggt_root}. "
            "Clone https://github.com/facebookresearch/vggt and set VGGT_PATH to point at it."
        )
    root = str(vggt_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_vggt_checkpoint(checkpoint: Path | str | None, *, hub_id: str) -> Path | str:
    if checkpoint is None:
        return hub_id
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.suffix == ".pt" and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"VGGT checkpoint not found: {checkpoint_path}. "
            f"Download facebook/VGGT-1B or pass hub id '{hub_id}' instead."
        )
    return str(checkpoint)


def _load_vggt_model(checkpoint: Path | str, *, device: str, vggt_root: Path):
    _add_vggt_to_path(vggt_root)
    import torch
    from vggt.models.vggt import VGGT

    model = VGGT()
    if isinstance(checkpoint, Path):
        logger.info("loading VGGT weights from %s", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
    else:
        hub_ref = str(checkpoint)
        logger.info("loading VGGT weights from %s", hub_ref)
        try:
            model = VGGT.from_pretrained(hub_ref)
        except Exception:
            logger.info("VGGT.from_pretrained failed; falling back to torch hub weights URL")
            state = torch.hub.load_state_dict_from_url(_DEFAULT_VGGT_MODEL_URL, map_location="cpu")
            model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _extrinsic_w2c_to_c2w(ext_w2c: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = ext_w2c
    return np.linalg.inv(w2c)


def _split_points_by_frame(
    points_3d: np.ndarray,
    points_rgb_u8: np.ndarray,
    points_xyf: np.ndarray,
    num_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frame_ids = points_xyf[:, 2].astype(np.int32)
    pts3d_per_view: list[np.ndarray] = []
    rgb_per_view: list[np.ndarray] = []
    for fidx in range(num_frames):
        mask = frame_ids == fidx
        pts3d_per_view.append(points_3d[mask].reshape(-1, 3))
        rgb_per_view.append(points_rgb_u8[mask].reshape(-1, 3).astype(np.float64) / 255.0)
    return pts3d_per_view, rgb_per_view


def run_vggt_inference(
    image_paths: Sequence[Path],
    output_dir: Path,
    *,
    checkpoint: Path | str | None = None,
    hub_id: str = _DEFAULT_VGGT_HUB_ID,
    vggt_root: Path = _DEFAULT_VGGT_ROOT,
    device: str = "cuda",
    conf_thres_value: float = 1.0,
    max_points: int = 100000,
    load_resolution: int = _VGGT_LOAD_RESOLUTION,
    inference_resolution: int = _VGGT_INFERENCE_RESOLUTION,
) -> Path:
    """Run VGGT feedforward reconstruction and write COLMAP text under ``output_dir/sparse/0``."""
    if len(image_paths) < 2:
        raise ValueError(f"VGGT needs at least 2 images, found {len(image_paths)}")

    _add_vggt_to_path(vggt_root)

    import torch
    import torch.nn.functional as F
    from gs_sim2real.preprocess.pose_free import write_colmap_sparse
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []
    for src in image_paths:
        dst = images_out / src.name
        try:
            same = dst.exists() and dst.resolve() == Path(src).resolve()
        except FileNotFoundError:
            same = False
        if not same:
            shutil.copy2(src, dst)
        staged_paths.append(dst)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "VGGT backend requested device='cuda' but no CUDA GPU is available. "
            "Pass device='cpu' for a slow smoke run, or reduce --num-frames."
        )

    dtype = (
        torch.bfloat16 if device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    resolved_checkpoint = _resolve_vggt_checkpoint(checkpoint, hub_id=hub_id)

    try:
        model = _load_vggt_model(resolved_checkpoint, device=device, vggt_root=vggt_root)
        images, _original_coords = load_and_preprocess_images_square(
            [str(path) for path in staged_paths],
            load_resolution,
        )
        images = images.to(device)

        images_infer = F.interpolate(
            images,
            size=(inference_resolution, inference_resolution),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
                predictions = model(images_infer[None])

        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_infer.shape[-2:])
        depth_tensor = predictions["depth"].squeeze(0)
        if depth_tensor.ndim == 4 and depth_tensor.shape[-1] > 1:
            depth_tensor = depth_tensor[..., 0]
        depth_map = depth_tensor.detach().cpu().numpy()
        depth_conf = predictions["depth_conf"].squeeze(0).detach().cpu().numpy()
        extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()
        intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()

        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images_infer,
            size=(inference_resolution, inference_resolution),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb = (points_rgb.detach().cpu().numpy() * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        poses = np.stack([_extrinsic_w2c_to_c2w(ext) for ext in extrinsic], axis=0)
        focals = intrinsic[:, 0, 0].reshape(-1, 1).astype(np.float32)
        imshape = (inference_resolution, inference_resolution)
        pts3d_per_view, rgb_per_view = _split_points_by_frame(points_3d, points_rgb, points_xyf, num_frames)

        sparse_dir = write_colmap_sparse(
            output_dir,
            image_paths=staged_paths,
            poses=poses,
            focals=focals,
            pts3d_per_view=pts3d_per_view,
            rgb_per_view=rgb_per_view,
            dust3r_shapes=[imshape] * num_frames,
            max_points=max_points,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "VGGT ran out of GPU memory. Reduce --num-frames (try 12–16 on a 16 GB card), "
            "close other GPU workloads, or use a smaller input batch."
        ) from exc

    logger.info(
        "VGGT wrote COLMAP sparse model: %d images, %d filtered points -> %s",
        len(staged_paths),
        len(points_3d),
        sparse_dir,
    )
    return sparse_dir
