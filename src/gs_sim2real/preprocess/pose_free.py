"""Pose-free preprocessing for 3D Gaussian Splatting.

Two real backends ship here: DUSt3R (pairwise pointmap prediction + global
alignment) and MAST3R (metric-aware descendant of DUSt3R; sparse global
alignment). VGGT (feedforward Visual Geometry Grounded Transformer) adds a
one-pass backend that exports COLMAP binaries via pycolmap. Both write models
compatible with the gsplat trainer. The "simple" fallback remains for unit tests / sanity checks when
neither clone is available — it arranges cameras in a circle so the trainer
has something to chew on, but the result is not meaningful.

The DUSt3R path expects a local clone of ``naver/dust3r`` reachable from
``DUST3R_PATH`` (default ``/tmp/dust3r``), plus its ``croco`` submodule and a
checkpoint file (``DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`` is the official
recommendation). The MAST3R path similarly expects a local clone of
``naver/mast3r`` reachable from ``MAST3R_PATH`` (default ``/tmp/mast3r``),
with its ``dust3r`` + ``croco`` submodules and the
``MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`` checkpoint.

DUSt3R pipeline:

1. ``load_images`` into DUSt3R's 512-long-side tensors.
2. ``make_pairs`` with the requested scene-graph (``complete`` for small
   batches, ``swin-N`` for larger ones).
3. ``inference`` — pairwise pointmaps via the DUSt3R network.
4. ``global_aligner(PointCloudOptimizer)`` — joint pose + depth optimization
   seeded from the MST of pair scores.
5. Write ``cameras.txt`` / ``images.txt`` / ``points3D.txt`` with per-image
   PINHOLE cameras, per-image c2w→w2c conversion, and a confidence-filtered
   subsample of the fused point cloud.

MAST3R pipeline mirrors steps 1–2 then swaps 3–4 for
``sparse_global_alignment`` which runs feature matching + a two-stage
optimizer and returns metric-scale poses, focals, and sparse points via the
``SparseGA`` container. Step 5 reuses the same ``write_colmap_sparse``
writer so downstream gsplat/exporter code does not care which backend ran.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DUST3R_ROOT = Path(os.environ.get("DUST3R_PATH", "/tmp/dust3r"))
_DEFAULT_CHECKPOINT_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
_DEFAULT_MAST3R_ROOT = Path(os.environ.get("MAST3R_PATH", "/tmp/mast3r"))
_DEFAULT_MAST3R_CHECKPOINT_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
_DEFAULT_VGGT_ROOT = Path(os.environ.get("VGGT_PATH", "/tmp/vggt"))
_DEFAULT_VGGT_HUB_ID = "facebook/VGGT-1B"


def _add_dust3r_to_path(dust3r_root: Path) -> None:
    """Make the local dust3r + croco clone importable."""
    if not dust3r_root.exists():
        raise FileNotFoundError(
            f"DUSt3R clone not found at {dust3r_root}. "
            "Clone https://github.com/naver/dust3r (with its croco submodule) "
            "and set DUST3R_PATH to point at it."
        )
    for sub in (dust3r_root, dust3r_root / "croco"):
        sub_str = str(sub)
        if sub_str not in sys.path:
            sys.path.insert(0, sub_str)


def _add_mast3r_to_path(mast3r_root: Path) -> None:
    """Make the local mast3r clone importable (mast3r ships dust3r + croco as submodules)."""
    if not mast3r_root.exists():
        raise FileNotFoundError(
            f"MAST3R clone not found at {mast3r_root}. "
            "Clone https://github.com/naver/mast3r (with its dust3r + croco submodules) "
            "and set MAST3R_PATH to point at it."
        )
    for sub in (mast3r_root, mast3r_root / "dust3r", mast3r_root / "dust3r" / "croco"):
        sub_str = str(sub)
        if sub_str not in sys.path:
            sys.path.insert(0, sub_str)


def _quat_from_rotation(R: np.ndarray) -> np.ndarray:
    """Return (qw, qx, qy, qz) for a 3x3 rotation matrix."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def _select_frames(image_paths: Sequence[Path], num_frames: int) -> list[Path]:
    if num_frames <= 0 or num_frames >= len(image_paths):
        return list(image_paths)
    idx = np.linspace(0, len(image_paths) - 1, num_frames).round().astype(int)
    return [image_paths[i] for i in idx]


def run_dust3r_inference(
    image_paths: Sequence[Path],
    checkpoint: Path,
    *,
    image_size: int = 512,
    device: str = "cuda",
    align_iters: int = 300,
    align_lr: float = 0.01,
    align_schedule: str = "cosine",
    scene_graph: str = "complete",
    dust3r_root: Path = _DEFAULT_DUST3R_ROOT,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    """Run DUSt3R inference + global alignment.

    Returns:
        poses: (N, 4, 4) camera-to-world matrices.
        focals: (N, 1) per-image focal lengths at DUSt3R's working resolution.
        pts3d_per_view: list of (Hi*Wi)x3 confidence-filtered points per view.
        rgb_per_view: list of (Hi*Wi)x3 0..1 RGB values aligned with pts3d.
        dust3r_shapes: list of (H, W) DUSt3R working resolutions per view.
    """
    _add_dust3r_to_path(dust3r_root)

    import torch  # noqa: PLC0415
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner  # noqa: PLC0415
    from dust3r.image_pairs import make_pairs  # noqa: PLC0415
    from dust3r.inference import inference  # noqa: PLC0415
    from dust3r.model import AsymmetricCroCo3DStereo  # noqa: PLC0415
    from dust3r.utils.image import load_images  # noqa: PLC0415

    # PyTorch 2.6's weights_only default trips on DUSt3R's argparse.Namespace.
    try:
        import argparse as _argparse  # noqa: PLC0415

        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:  # pragma: no cover - older torch
        pass

    logger.info("loading DUSt3R from %s", checkpoint)
    model = AsymmetricCroCo3DStereo.from_pretrained(str(checkpoint)).to(device)
    model.eval()

    logger.info("loading %d images at size=%d", len(image_paths), image_size)
    views = load_images([str(p) for p in image_paths], size=image_size, verbose=False)

    pairs = make_pairs(views, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    logger.info("running inference on %d pairs (scene_graph=%s)", len(pairs), scene_graph)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # Free model weights before the aligner allocates its stacked-pred tensor —
    # otherwise PointCloudOptimizer OOMs on 16 GB cards for >16 complete frames.
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    logger.info("running global alignment: %d iters, lr=%.4f, schedule=%s", align_iters, align_lr, align_schedule)
    loss = scene.compute_global_alignment(init="mst", niter=align_iters, schedule=align_schedule, lr=align_lr)
    logger.info("alignment final loss: %.6f", float(loss))

    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    confs = [c.detach().cpu().numpy() for c in scene.im_conf]
    imgs = [img if isinstance(img, np.ndarray) else img.cpu().numpy() for img in scene.imgs]
    imshapes = [tuple(im.shape[:2]) for im in imgs]

    filtered_pts: list[np.ndarray] = []
    filtered_rgb: list[np.ndarray] = []
    for pts, conf, img in zip(pts3d, confs, imgs, strict=True):
        mask = conf > conf.mean()
        filtered_pts.append(pts[mask])
        filtered_rgb.append(img[mask])

    return poses, focals, filtered_pts, filtered_rgb, imshapes


def run_mast3r_inference(
    image_paths: Sequence[Path],
    checkpoint: Path,
    *,
    image_size: int = 512,
    device: str = "cuda",
    cache_dir: Path | None = None,
    scene_graph: str = "complete",
    subsample: int = 8,
    mast3r_root: Path = _DEFAULT_MAST3R_ROOT,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    """Run MAST3R sparse global alignment.

    Shape matches ``run_dust3r_inference`` so ``write_colmap_sparse`` accepts
    the output directly. MAST3R returns per-view focal / pts3d / rgb from
    ``SparseGA``; we rescale focals back to match the DUSt3R working-resolution
    contract (the caller scales them up to the original image size on disk).

    Returns:
        poses: (N, 4, 4) camera-to-world matrices.
        focals: (N, 1) per-image focal lengths at MAST3R's working resolution.
        pts3d_per_view: list of (Ni, 3) sparse points per view (no confidence filter).
        rgb_per_view: list of (Ni, 3) 0..1 RGB per view, aligned to pts3d.
        mast3r_shapes: list of (H, W) MAST3R working resolutions per view.
    """
    _add_mast3r_to_path(mast3r_root)

    import torch  # noqa: PLC0415
    from dust3r.image_pairs import make_pairs  # noqa: PLC0415
    from dust3r.utils.image import load_images  # noqa: PLC0415
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: PLC0415
    from mast3r.model import AsymmetricMASt3R  # noqa: PLC0415

    try:
        import argparse as _argparse  # noqa: PLC0415

        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:  # pragma: no cover - older torch
        pass

    logger.info("loading MAST3R from %s", checkpoint)
    model = AsymmetricMASt3R.from_pretrained(str(checkpoint)).to(device)
    model.eval()

    logger.info("loading %d images at size=%d", len(image_paths), image_size)
    views = load_images([str(p) for p in image_paths], size=image_size, verbose=False)
    # Overwrite MAST3R-expected instance string with the actual path (sparse_global_alignment
    # indexes pairs by `img_path` identity).
    for view, path in zip(views, image_paths, strict=True):
        view["instance"] = str(path)

    pairs = make_pairs(views, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    logger.info("running MAST3R sparse GA on %d pairs (scene_graph=%s)", len(pairs), scene_graph)

    if cache_dir is None:
        cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "mast3r_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sga = sparse_global_alignment(
        [str(p) for p in image_paths],
        pairs,
        str(cache_dir),
        model,
        device=device,
        subsample=subsample,
    )

    poses = sga.get_im_poses().detach().cpu().numpy()
    # MAST3R's intrinsics are 3x3 matrices at the working resolution.
    focals = np.array([float(K[0, 0].detach().cpu()) for K in sga.intrinsics], dtype=np.float32).reshape(-1, 1)
    pts3d = [p.detach().cpu().numpy() for p in sga.get_sparse_pts3d()]
    colors = []
    for c in sga.get_pts3d_colors():
        colors.append(c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c))
    # MAST3R `imgs` are rendered as HxWx3 ndarrays already.
    mast3r_shapes = [tuple(im.shape[:2]) for im in sga.imgs]
    return poses, focals, pts3d, colors, mast3r_shapes


def write_colmap_sparse(
    output_dir: Path,
    image_paths: Sequence[Path],
    poses: np.ndarray,
    focals: np.ndarray,
    pts3d_per_view: Sequence[np.ndarray],
    rgb_per_view: Sequence[np.ndarray],
    dust3r_shapes: Sequence[tuple[int, int]],
    *,
    max_points: int = 100000,
    seed: int = 0,
) -> Path:
    """Write COLMAP-text sparse files + copy images to ``output_dir``.

    One PINHOLE camera is emitted per image because DUSt3R infers per-view
    focal lengths. Focals are rescaled from the DUSt3R working resolution to
    the original image resolution on disk.
    """
    from PIL import Image  # noqa: PLC0415

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    orig_shapes: list[tuple[int, int]] = []
    for src in image_paths:
        dst = images_out / src.name
        try:
            same = dst.exists() and dst.resolve() == Path(src).resolve()
        except FileNotFoundError:
            same = False
        if not same:
            shutil.copy2(src, dst)
        with Image.open(dst) as im:
            orig_shapes.append((im.size[1], im.size[0]))

    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    points_path = sparse_dir / "points3D.txt"

    with cameras_path.open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, ((h_d, w_d), (h_o, w_o)) in enumerate(zip(dust3r_shapes, orig_shapes, strict=True)):
            focal_d = float(focals[i].item() if hasattr(focals[i], "item") else focals[i][0])
            fx = focal_d * (w_o / w_d)
            fy = focal_d * (h_o / h_d)
            cx = w_o / 2.0
            cy = h_o / 2.0
            f.write(f"{i + 1} PINHOLE {w_o} {h_o} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with images_path.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (pose_c2w, img_path) in enumerate(zip(poses, image_paths, strict=True)):
            pose_w2c = np.linalg.inv(pose_c2w)
            R_w2c = pose_w2c[:3, :3]
            t_w2c = pose_w2c[:3, 3]
            q = _quat_from_rotation(R_w2c)
            f.write(
                f"{i + 1} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"{t_w2c[0]:.8f} {t_w2c[1]:.8f} {t_w2c[2]:.8f} {i + 1} {img_path.name}\n"
            )
            f.write("\n")  # no 2D keypoints

    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_per_view], axis=0)
    all_rgb = np.concatenate([c.reshape(-1, 3) for c in rgb_per_view], axis=0)
    if all_pts.shape[0] != all_rgb.shape[0]:
        raise RuntimeError(f"pts/rgb length mismatch: {all_pts.shape} vs {all_rgb.shape}")

    if max_points > 0 and all_pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(all_pts.shape[0], size=max_points, replace=False)
        all_pts = all_pts[idx]
        all_rgb = all_rgb[idx]

    rgb_u8 = np.clip(all_rgb * 255.0, 0, 255).astype(np.uint8)
    with points_path.open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (p, c) in enumerate(zip(all_pts, rgb_u8, strict=True)):
            f.write(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")

    logger.info(
        "wrote COLMAP sparse: %d cameras, %d images, %d points -> %s",
        len(image_paths),
        len(image_paths),
        all_pts.shape[0],
        sparse_dir,
    )
    return sparse_dir


class PoseFreeProcessor:
    """Pose-free preprocessing using DUSt3R with a simple-init fallback."""

    def __init__(
        self,
        method: str = "dust3r",
        *,
        checkpoint: Path | None = None,
        dust3r_root: Path | None = None,
        mast3r_root: Path | None = None,
        mast3r_cache: Path | None = None,
        vggt_root: Path | None = None,
        vggt_hub_id: str | None = None,
        vggt_conf_threshold: float = 1.0,
        num_frames: int = 30,
        image_size: int = 512,
        device: str = "cuda",
        align_iters: int = 300,
        align_lr: float = 0.01,
        align_schedule: str = "cosine",
        scene_graph: str = "complete",
        mast3r_subsample: int = 8,
        max_points: int = 100000,
    ):
        self.method = method
        self.dust3r_root = Path(dust3r_root) if dust3r_root else _DEFAULT_DUST3R_ROOT
        self.mast3r_root = Path(mast3r_root) if mast3r_root else _DEFAULT_MAST3R_ROOT
        self.mast3r_cache = Path(mast3r_cache) if mast3r_cache else None
        self.vggt_root = Path(vggt_root) if vggt_root else _DEFAULT_VGGT_ROOT
        self.vggt_hub_id = vggt_hub_id or _DEFAULT_VGGT_HUB_ID
        self.vggt_conf_threshold = vggt_conf_threshold
        if checkpoint:
            self.checkpoint = checkpoint if isinstance(checkpoint, str) and not str(checkpoint).endswith(".pt") else Path(checkpoint)
        elif method == "mast3r":
            self.checkpoint = self.mast3r_root / "checkpoints" / _DEFAULT_MAST3R_CHECKPOINT_NAME
        elif method == "vggt":
            self.checkpoint = None
        else:
            self.checkpoint = self.dust3r_root / "checkpoints" / _DEFAULT_CHECKPOINT_NAME
        self.num_frames = num_frames
        self.image_size = image_size
        self.device = device
        self.align_iters = align_iters
        self.align_lr = align_lr
        self.align_schedule = align_schedule
        self.scene_graph = scene_graph
        self.mast3r_subsample = mast3r_subsample
        self.max_points = max_points

    def estimate_poses(self, image_dir: str | Path, output_dir: str | Path) -> str:
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)
        if len(images) < 2:
            raise ValueError(f"Need at least 2 images, found {len(images)}")

        logger.info("Estimating poses for %d images using %s", len(images), self.method)

        if self.method in ("dust3r", "pose-free"):
            return self._run_dust3r(images, output_dir)
        if self.method == "mast3r":
            return self._run_mast3r(images, output_dir)
        if self.method == "vggt":
            return self._run_vggt(images, output_dir)
        if self.method == "simple":
            return self._run_simple_init(images, output_dir)
        raise ValueError(f"Unknown pose-free method: {self.method}")

    def _run_vggt(self, images: list[Path], output_dir: Path) -> str:
        from gs_sim2real.preprocess.vggt_backend import run_vggt_inference

        selected = _select_frames(images, self.num_frames)
        logger.info("VGGT using %d / %d images", len(selected), len(images))
        sparse_dir = run_vggt_inference(
            selected,
            output_dir,
            checkpoint=self.checkpoint,
            hub_id=self.vggt_hub_id,
            vggt_root=self.vggt_root,
            device=self.device,
            conf_thres_value=self.vggt_conf_threshold,
            max_points=self.max_points,
        )
        np.save(output_dir / "vggt_frame_count.npy", np.array([len(selected)], dtype=np.int32))
        return str(sparse_dir)

    def _run_mast3r(self, images: list[Path], output_dir: Path) -> str:
        try:
            selected = _select_frames(images, self.num_frames)
            logger.info("MAST3R using %d / %d images", len(selected), len(images))
            cache_dir = self.mast3r_cache or (output_dir / "mast3r_cache")
            poses, focals, pts3d_per_view, rgb_per_view, imshapes = run_mast3r_inference(
                selected,
                checkpoint=self.checkpoint,
                image_size=self.image_size,
                device=self.device,
                cache_dir=cache_dir,
                scene_graph=self.scene_graph,
                subsample=self.mast3r_subsample,
                mast3r_root=self.mast3r_root,
            )
            np.save(output_dir / "poses.npy", poses)
            np.save(output_dir / "focals.npy", focals)
            flat_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_per_view], axis=0)
            np.save(output_dir / "pts3d.npy", flat_pts)
            sparse_dir = write_colmap_sparse(
                output_dir,
                image_paths=selected,
                poses=poses,
                focals=focals,
                pts3d_per_view=pts3d_per_view,
                rgb_per_view=rgb_per_view,
                dust3r_shapes=imshapes,
                max_points=self.max_points,
            )
            return str(sparse_dir)
        except (ImportError, FileNotFoundError) as exc:
            logger.warning(
                "MAST3R not available (%s). Falling back to simple circular initialization.",
                exc,
            )
            return self._run_simple_init(images, output_dir)

    def _run_dust3r(self, images: list[Path], output_dir: Path) -> str:
        try:
            selected = _select_frames(images, self.num_frames)
            logger.info("DUSt3R using %d / %d images", len(selected), len(images))
            poses, focals, pts3d_per_view, rgb_per_view, imshapes = run_dust3r_inference(
                selected,
                checkpoint=self.checkpoint,
                image_size=self.image_size,
                device=self.device,
                align_iters=self.align_iters,
                align_lr=self.align_lr,
                align_schedule=self.align_schedule,
                scene_graph=self.scene_graph,
                dust3r_root=self.dust3r_root,
            )
            np.save(output_dir / "poses.npy", poses)
            np.save(output_dir / "focals.npy", focals)
            flat_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_per_view], axis=0)
            np.save(output_dir / "pts3d.npy", flat_pts)
            sparse_dir = write_colmap_sparse(
                output_dir,
                image_paths=selected,
                poses=poses,
                focals=focals,
                pts3d_per_view=pts3d_per_view,
                rgb_per_view=rgb_per_view,
                dust3r_shapes=imshapes,
                max_points=self.max_points,
            )
            return str(sparse_dir)
        except (ImportError, FileNotFoundError) as exc:
            logger.warning(
                "DUSt3R not available (%s). Falling back to simple circular initialization.",
                exc,
            )
            return self._run_simple_init(images, output_dir)

    def _run_simple_init(self, images: list[Path], output_dir: Path) -> str:
        """Circular camera arrangement fallback (not metrically meaningful)."""
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(images[0]))
        h, w = img.shape[:2]
        focal = max(w, h) * 1.2
        cx, cy = w / 2.0, h / 2.0

        with (sparse_dir / "cameras.txt").open("w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {w} {h} {focal} {focal} {cx} {cy}\n")

        num_images = len(images)
        with (sparse_dir / "images.txt").open("w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            for i, img_path in enumerate(images):
                angle = 2 * np.pi * i / num_images
                radius = 3.0
                tx = radius * np.cos(angle)
                ty = 0.0
                tz = radius * np.sin(angle)
                qw = np.cos(angle / 2)
                qx = 0.0
                qy = np.sin(angle / 2)
                qz = 0.0
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_path.name}\n")
                f.write("\n")

        rng = np.random.default_rng(seed=42)
        with (sparse_dir / "points3D.txt").open("w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            num_points = 1000
            for i in range(num_points):
                x = rng.uniform(-2, 2)
                y = rng.uniform(-1, 1)
                z = rng.uniform(-2, 2)
                r, g, b = rng.integers(0, 255, 3)
                f.write(f"{i + 1} {x} {y} {z} {r} {g} {b} 0.0\n")

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for img_path in images:
            shutil.copy2(img_path, images_dir / img_path.name)

        logger.info(
            "Simple initialization complete: %d cameras, %d initial points",
            num_images,
            num_points,
        )
        return str(sparse_dir)


def run_pose_free(
    image_dir: str | Path,
    output_dir: str | Path,
    method: str = "dust3r",
    **kwargs,
) -> str:
    """Run pose-free preprocessing on a directory of images."""
    processor = PoseFreeProcessor(method=method, **kwargs)
    return processor.estimate_poses(image_dir, output_dir)
