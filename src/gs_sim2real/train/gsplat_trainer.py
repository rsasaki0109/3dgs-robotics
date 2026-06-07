"""gsplat-based 3D Gaussian Splatting training wrapper.

This module provides a training loop for 3D Gaussian Splatting using the
gsplat library. It handles:
- Loading COLMAP or custom-format camera poses and images
- Initializing Gaussians from a sparse point cloud
- Running the densification / pruning / optimization loop
- Saving checkpoints and final .ply point clouds

Reference: gsplat (https://github.com/nerfstudio-project/gsplat)
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gs_sim2real.common.config import load_training_config

logger = logging.getLogger(__name__)


@dataclass
class GaussianModel:
    """Container for 3D Gaussian parameters."""

    means: Any = None  # (N, 3) positions
    scales: Any = None  # (N, 3) log-scales
    rotations: Any = None  # (N, 4) quaternions
    opacities: Any = None  # (N, 1) sigmoid-pre-activation opacities
    sh_coeffs: Any = None  # (N, C, 3) spherical harmonics coefficients
    num_gaussians: int = 0


class GsplatTrainer:
    """Trainer for 3D Gaussian Splatting using gsplat."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the trainer.

        Args:
            config: Training hyperparameters. If None, loads from configs/training.yaml.
        """
        self.config = config or load_training_config()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch is required for 3DGS training. Install it from: https://pytorch.org/get-started/locally/"
            )

        try:
            import gsplat  # noqa: F401

            self._has_gsplat = True
        except ImportError:
            self._has_gsplat = False
            logger.warning(
                "gsplat is not installed. Training will use a simplified renderer. "
                "For full performance, install gsplat: pip install gsplat"
            )

    def train(
        self,
        data_dir: Path | str,
        output_dir: Path | str,
        num_iterations: int | None = None,
    ) -> Path:
        """Train a 3D Gaussian Splatting model.

        Args:
            data_dir: Directory containing COLMAP output (sparse/ and images/).
            output_dir: Directory where training outputs will be saved.
            num_iterations: Number of training iterations. Overrides config.

        Returns:
            Path to the final exported .ply file.
        """
        import torch

        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if num_iterations is None:
            num_iterations = self.config.get("num_iterations", 30000)

        # Set random seed
        seed = self.config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")

        # Load COLMAP data
        cameras, images_meta, points3d = self._load_colmap_data(data_dir)
        print(f"Loaded {len(cameras)} cameras, {len(images_meta)} images, {len(points3d)} points")

        # Load actual images
        image_data = self._load_images(data_dir, images_meta, device)
        print(f"Loaded {len(image_data)} training images")

        if len(points3d) == 0:
            raise ValueError("No 3D points found. COLMAP reconstruction may have failed.")

        # Initialize Gaussians from point cloud
        gaussians = self._initialize_gaussians(points3d, device)
        print(f"Initialized {gaussians.num_gaussians} Gaussians")

        # Set up optimizers
        optimizers = self._create_optimizers(gaussians)

        # Training loop
        print(f"\nStarting training for {num_iterations} iterations...")
        start_time = time.time()
        save_iters = set(self.config.get("save_iterations", [7000, 15000, 30000]))

        lambda_dssim = self.config.get("lambda_dssim", 0.2)
        densify_from = self.config.get("densify_from_iter", 500)
        densify_until = self.config.get("densify_until_iter", 15000)
        densify_interval = self.config.get("densify_interval", 100)
        opacity_reset_interval = self.config.get("opacity_reset_interval", 3000)

        # Gradient accumulator for densification
        grad_accum = torch.zeros(gaussians.num_gaussians, device=device)
        grad_count = torch.zeros(gaussians.num_gaussians, device=device)

        depth_loss_weight = float(self.config.get("depth_loss_weight", 0.0))
        use_depth_loss = depth_loss_weight > 0.0 and self._has_gsplat

        appearance_enabled = bool(self.config.get("appearance_embedding", False))
        appearance_lr = float(self.config.get("appearance_lr", 0.001))
        appearance_reg = float(self.config.get("appearance_reg_weight", 0.01))
        appearance_scale = None
        appearance_bias = None
        appearance_optimizer = None
        if appearance_enabled:
            # Per-image learnable (scale, bias) RGB affine -- absorbs exposure/white-balance
            # differences between cameras and frames. Initialised at identity (scale=1, bias=0).
            appearance_scale = torch.nn.Parameter(torch.ones(len(image_data), 3, device=device, dtype=torch.float32))
            appearance_bias = torch.nn.Parameter(torch.zeros(len(image_data), 3, device=device, dtype=torch.float32))
            appearance_optimizer = torch.optim.Adam([appearance_scale, appearance_bias], lr=appearance_lr)
            print(
                f"Appearance embedding enabled: per-image (scale, bias) Adam lr={appearance_lr}, "
                f"reg_weight={appearance_reg}"
            )

        # Joint pose refinement (BA-style). Each training view learns a 6-DOF
        # delta in its camera frame: small rotation (so3) + translation applied
        # to the COLMAP-given viewmat. Helps when GNSS-seeded poses are not
        # consistent enough for 3DGS to fit.
        pose_refine_enabled = bool(self.config.get("joint_pose_refinement", False))
        pose_refine_lr = float(self.config.get("joint_pose_lr", 1e-4))
        pose_refine_reg = float(self.config.get("joint_pose_reg_weight", 0.1))
        pose_refine_start = int(self.config.get("joint_pose_start_iter", 1500))
        pose_so3 = None
        pose_t = None
        pose_refine_optimizer = None
        if pose_refine_enabled:
            pose_so3 = torch.nn.Parameter(torch.zeros(len(image_data), 3, device=device, dtype=torch.float32))
            pose_t = torch.nn.Parameter(torch.zeros(len(image_data), 3, device=device, dtype=torch.float32))
            pose_refine_optimizer = torch.optim.Adam([pose_so3, pose_t], lr=pose_refine_lr)
            print(
                f"Joint pose refinement enabled: per-image 6-DOF delta from iter {pose_refine_start}, "
                f"Adam lr={pose_refine_lr}, reg_weight={pose_refine_reg}"
            )

        for iteration in range(1, num_iterations + 1):
            # Pick a random training view
            idx = np.random.randint(len(image_data))
            entry_i = image_data[idx]
            if entry_i.get("_lazy"):
                gt_image = entry_i["image"].to(device, non_blocking=True)
            else:
                gt_image = entry_i["image"]
            viewmat = entry_i["viewmat"]
            K = entry_i["K"]
            H, W = gt_image.shape[:2]
            gt_depth = None
            if use_depth_loss and "depth" in entry_i:
                if entry_i.get("_lazy"):
                    gt_depth = entry_i["depth"].to(device, non_blocking=True)
                else:
                    gt_depth = entry_i["depth"]

            if pose_refine_enabled and iteration >= pose_refine_start and pose_so3 is not None:
                # Compose viewmat with a small learned delta: view' = Δ @ view
                so3 = pose_so3[idx]
                tvec_delta = pose_t[idx]
                theta = torch.linalg.norm(so3) + 1e-8
                axis = so3 / theta
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                ax, ay, az = axis[0], axis[1], axis[2]
                K_skew = torch.stack(
                    [
                        torch.stack([torch.zeros_like(ax), -az, ay]),
                        torch.stack([az, torch.zeros_like(ax), -ax]),
                        torch.stack([-ay, ax, torch.zeros_like(ax)]),
                    ]
                )
                R_delta = torch.eye(3, device=device) + sin_t * K_skew + (1 - cos_t) * (K_skew @ K_skew)
                T_delta = torch.eye(4, device=device, dtype=viewmat.dtype)
                T_delta[:3, :3] = R_delta
                T_delta[:3, 3] = tvec_delta
                viewmat = T_delta @ viewmat

            # Render
            if use_depth_loss and gt_depth is not None:
                rendered, rendered_depth = self._render_gsplat(gaussians, viewmat, K, H, W, device, want_depth=True)
            else:
                rendered = self._render(gaussians, viewmat, K, H, W, device)
                rendered_depth = None

            if appearance_enabled and appearance_scale is not None:
                # Apply per-image affine: rendered * scale + bias -> compare to gt
                s = appearance_scale[idx].view(1, 1, 3)
                b = appearance_bias[idx].view(1, 1, 3)
                rendered_eff = torch.clamp(rendered * s + b, 0.0, 1.0)
            else:
                rendered_eff = rendered

            # Compute loss: L1 + lambda * (1 - SSIM) [+ depth L1 where LiDAR sees]
            l1_loss = torch.abs(rendered_eff - gt_image).mean()
            ssim_loss = 1.0 - self._simple_ssim(rendered_eff, gt_image)
            loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
            if rendered_depth is not None:
                valid = gt_depth > 0
                if valid.any():
                    depth_l1 = torch.abs(rendered_depth[valid] - gt_depth[valid]).mean()
                    loss = loss + depth_loss_weight * depth_l1
            if appearance_enabled and appearance_scale is not None:
                # Regularise (scale, bias) away from drifting: penalise scale != 1, bias != 0.
                reg = ((appearance_scale[idx] - 1.0) ** 2).mean() + (appearance_bias[idx] ** 2).mean()
                loss = loss + appearance_reg * reg
            if pose_refine_enabled and iteration >= pose_refine_start and pose_so3 is not None:
                pose_reg = (pose_so3[idx] ** 2).sum() + (pose_t[idx] ** 2).sum()
                loss = loss + pose_refine_reg * pose_reg

            # Backprop
            loss.backward()

            with torch.no_grad():
                # Accumulate gradients for densification
                if gaussians.means.grad is not None and densify_from <= iteration <= densify_until:
                    grad_norms = gaussians.means.grad.norm(dim=-1)
                    grad_accum[: gaussians.num_gaussians] += grad_norms
                    grad_count[: gaussians.num_gaussians] += 1

                # Densification
                if densify_from <= iteration <= densify_until and iteration % densify_interval == 0:
                    avg_grad = grad_accum / (grad_count + 1e-8)
                    self._densify_and_prune(gaussians, avg_grad, optimizers, device)
                    # Reset accumulators
                    grad_accum = torch.zeros(gaussians.num_gaussians, device=device)
                    grad_count = torch.zeros(gaussians.num_gaussians, device=device)

                # Opacity reset
                if iteration % opacity_reset_interval == 0 and iteration < densify_until:
                    self._reset_opacity(gaussians)

                # Optimizer step
                for opt in optimizers.values():
                    opt.step()
                    opt.zero_grad()
                if appearance_optimizer is not None:
                    appearance_optimizer.step()
                    appearance_optimizer.zero_grad()
                if pose_refine_optimizer is not None and iteration >= pose_refine_start:
                    pose_refine_optimizer.step()
                    pose_refine_optimizer.zero_grad()

                # Update learning rate for position
                self._update_lr(optimizers["position"], iteration, num_iterations)

            # Logging
            if iteration % 100 == 0 or iteration == 1:
                elapsed = time.time() - start_time
                its_per_sec = iteration / elapsed if elapsed > 0 else 0
                print(
                    f"  [Iter {iteration:6d}/{num_iterations}] "
                    f"loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                    f"ssim_loss={ssim_loss.item():.4f} "
                    f"n_gaussians={gaussians.num_gaussians:,} "
                    f"({its_per_sec:.1f} it/s)"
                )

            # Save checkpoints
            if iteration in save_iters:
                ckpt_path = output_dir / f"point_cloud_iter_{iteration}.ply"
                self._save_model(ckpt_path, gaussians)
                print(f"  Checkpoint saved: {ckpt_path}")

        # Save final model
        final_path = output_dir / "point_cloud.ply"
        self._save_model(final_path, gaussians)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Final model saved to: {final_path}")
        print(f"Final Gaussians: {gaussians.num_gaussians:,}")

        return final_path

    def _load_colmap_data(self, data_dir: Path) -> tuple[dict[int, dict], dict[int, dict], np.ndarray]:
        """Load COLMAP sparse model (cameras, images, points3D).

        Supports both text and binary COLMAP formats. Looks for sparse model
        in data_dir/sparse/0/ or data_dir/undistorted/sparse/.

        Args:
            data_dir: Root data directory.

        Returns:
            Tuple of (cameras, images, points3D as Nx6 array [x,y,z,r,g,b]).
        """
        # Search for sparse model
        candidates = [
            data_dir / "sparse" / "0",
            data_dir / "undistorted" / "sparse",
            data_dir / "sparse",
            data_dir,
        ]

        sparse_dir = None
        for candidate in candidates:
            if candidate.exists() and ((candidate / "cameras.txt").exists() or (candidate / "cameras.bin").exists()):
                sparse_dir = candidate
                break

        if sparse_dir is None:
            raise FileNotFoundError(
                f"No COLMAP sparse model found in {data_dir}. "
                "Expected cameras.txt/bin, images.txt/bin, points3D.txt/bin "
                "in sparse/0/ or undistorted/sparse/."
            )

        # Try text format first, then binary
        if (sparse_dir / "cameras.txt").exists():
            cameras = self._load_cameras_txt(sparse_dir / "cameras.txt")
            images_meta = self._load_images_txt(sparse_dir / "images.txt")
            points3d = self._load_points3d_txt(sparse_dir / "points3D.txt")
        else:
            cameras = self._load_cameras_bin(sparse_dir / "cameras.bin")
            images_meta = self._load_images_bin(sparse_dir / "images.bin")
            points3d = self._load_points3d_bin(sparse_dir / "points3D.bin")

        return cameras, images_meta, points3d

    def _load_cameras_txt(self, path: Path) -> dict[int, dict]:
        """Parse COLMAP cameras.txt."""
        cameras = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(x) for x in parts[4:]]
                cameras[cam_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                }
        return cameras

    def _load_images_txt(self, path: Path) -> dict[int, dict]:
        """Parse COLMAP images.txt."""
        images = {}
        with open(path) as f:
            lines = [line.rstrip("\n") for line in f if not line.lstrip().startswith("#")]

        # images.txt has pairs of lines: metadata + 2D points. The 2D
        # points line may be blank for pose-seeded imports, so blank lines
        # must be preserved until after the paired metadata line is skipped.
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            parts = line.split()
            if len(parts) < 10:
                i += 1
                continue
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
            images[img_id] = {
                "quat": [qw, qx, qy, qz],
                "tvec": [tx, ty, tz],
                "camera_id": cam_id,
                "name": name,
            }
            i += 2
        return images

    def _load_points3d_txt(self, path: Path) -> np.ndarray:
        """Parse COLMAP points3D.txt. Returns Nx6 array [x,y,z,r,g,b]."""
        points = []
        if not path.exists():
            return np.zeros((0, 6))
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                points.append([x, y, z, r / 255.0, g / 255.0, b / 255.0])
        return np.array(points) if points else np.zeros((0, 6))

    def _load_cameras_bin(self, path: Path) -> dict[int, dict]:
        """Parse COLMAP cameras.bin."""
        cameras = {}
        CAMERA_MODELS = {
            0: ("SIMPLE_PINHOLE", 3),
            1: ("PINHOLE", 4),
            2: ("SIMPLE_RADIAL", 4),
            3: ("RADIAL", 5),
            4: ("OPENCV", 8),
            5: ("OPENCV_FISHEYE", 8),
            6: ("FULL_OPENCV", 12),
            7: ("FOV", 5),
            8: ("SIMPLE_RADIAL_FISHEYE", 4),
            9: ("RADIAL_FISHEYE", 5),
            10: ("THIN_PRISM_FISHEYE", 12),
        }
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                cam_id = struct.unpack("<i", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                model_name, num_params = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
                params = list(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
                cameras[cam_id] = {
                    "model": model_name,
                    "width": width,
                    "height": height,
                    "params": params,
                }
        return cameras

    def _load_images_bin(self, path: Path) -> dict[int, dict]:
        """Parse COLMAP images.bin."""
        images = {}
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                img_id = struct.unpack("<i", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                cam_id = struct.unpack("<i", f.read(4))[0]
                # Read null-terminated name
                name_bytes = b""
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    name_bytes += ch
                name = name_bytes.decode("utf-8")
                # Read 2D points
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                # Skip 2D point data (x, y, point3D_id per point)
                f.read(num_points2d * 24)
                images[img_id] = {
                    "quat": [qw, qx, qy, qz],
                    "tvec": [tx, ty, tz],
                    "camera_id": cam_id,
                    "name": name,
                }
        return images

    def _load_points3d_bin(self, path: Path) -> np.ndarray:
        """Parse COLMAP points3D.bin. Returns Nx6 array [x,y,z,r,g,b]."""
        points = []
        if not path.exists():
            return np.zeros((0, 6))
        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                struct.unpack("<Q", f.read(8))[0]  # point_id (unused)
                x, y, z = struct.unpack("<3d", f.read(24))
                r, g, b = struct.unpack("<3B", f.read(3))
                struct.unpack("<d", f.read(8))[0]  # error (unused)
                track_len = struct.unpack("<Q", f.read(8))[0]
                # Skip track data
                f.read(track_len * 8)
                points.append([x, y, z, r / 255.0, g / 255.0, b / 255.0])
        return np.array(points) if points else np.zeros((0, 6))

    def _load_images(self, data_dir: Path, images_meta: dict[int, dict], device: Any) -> list[dict[str, Any]]:
        """Load training images and compute view/projection matrices.

        Args:
            data_dir: Root data directory.
            images_meta: COLMAP image metadata.
            device: Torch device.

        Returns:
            List of dicts with 'image', 'viewmat', 'K' tensors.
        """
        import torch
        from PIL import Image

        # Find images directory
        img_dir_candidates = [
            data_dir / "undistorted" / "images",
            data_dir / "images",
        ]
        img_dir = None
        for candidate in img_dir_candidates:
            if candidate.exists():
                img_dir = candidate
                break
        if img_dir is None:
            raise FileNotFoundError(f"No images directory found in {data_dir}")

        # Load camera intrinsics (use first camera)
        cameras_data = self._load_colmap_data(data_dir)[0]
        if not cameras_data:
            raise ValueError("No cameras found in COLMAP model")

        # Heuristic: if the dataset is large enough to risk OOM when every
        # image (+ depth map) is pinned on GPU, keep images / depth on CPU
        # and transfer per iteration. Threshold ~1000 images or CPU device.
        image_count = len(images_meta)
        lazy_threshold = int(self.config.get("lazy_image_threshold", 1000))
        lazy_images = image_count > lazy_threshold or str(device) == "cpu"
        if lazy_images:
            print(f"Lazy image loading enabled ({image_count} > {lazy_threshold}); images stay on CPU")

        image_data = []
        for img_id, meta in images_meta.items():
            img_path = img_dir / meta["name"]
            if not img_path.exists():
                logger.warning("Image not found: %s", img_path)
                continue

            # Load image
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_tensor_cpu = torch.from_numpy(img_np)
            if lazy_images:
                # Skip pin_memory() for very large datasets: it doubles host RAM use
                # (pinned page-locked pages stay resident) and can cause CUDA OOM
                # when combined with training tensors. Regular CPU tensor + per-iter
                # .to(device) is plenty fast for our scale.
                img_tensor = img_tensor_cpu
            else:
                img_tensor = img_tensor_cpu.to(device)

            H, W = img_tensor.shape[:2]

            # Get camera intrinsics
            cam = cameras_data[meta["camera_id"]]
            K = self._make_intrinsic_matrix(cam, device)

            # Compute view matrix from quaternion + translation
            quat = meta["quat"]  # [qw, qx, qy, qz]
            tvec = meta["tvec"]  # [tx, ty, tz]
            viewmat = self._quat_tvec_to_viewmat(quat, tvec, device)

            entry = {
                "image": img_tensor,
                "viewmat": viewmat,
                "K": K,
                "name": meta["name"],
                "_lazy": lazy_images,
            }
            depth_path = data_dir / "depth" / (Path(meta["name"]).with_suffix(".npy"))
            if depth_path.exists():
                depth_np = np.load(depth_path).astype(np.float32)
                if depth_np.shape[:2] == (H, W):
                    depth_cpu = torch.from_numpy(depth_np)
                    if lazy_images:
                        entry["depth"] = depth_cpu
                    else:
                        entry["depth"] = depth_cpu.to(device)
            image_data.append(entry)

        return image_data

    def _make_intrinsic_matrix(self, cam: dict, device: Any) -> Any:
        """Build 3x3 intrinsic matrix from COLMAP camera parameters."""
        import torch

        params = cam["params"]
        model = cam["model"]

        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model in ("PINHOLE", "RADIAL", "RADIAL_FISHEYE"):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            # Fallback
            fx = fy = params[0] if params else cam["width"]
            cx, cy = cam["width"] / 2, cam["height"] / 2

        K = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        return K

    def _quat_tvec_to_viewmat(self, quat: list[float], tvec: list[float], device: Any) -> Any:
        """Convert COLMAP quaternion + translation to 4x4 view matrix."""
        import torch

        qw, qx, qy, qz = quat
        # Quaternion to rotation matrix
        R = torch.tensor(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=torch.float32,
            device=device,
        )

        t = torch.tensor(tvec, dtype=torch.float32, device=device)

        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        return viewmat

    def _initialize_gaussians(self, points3d: np.ndarray, device: Any) -> GaussianModel:
        """Initialize Gaussian parameters from COLMAP 3D points.

        Args:
            points3d: Nx6 array of [x, y, z, r, g, b].
            device: Torch device.

        Returns:
            Initialized GaussianModel.
        """
        import torch

        N = len(points3d)
        model = GaussianModel(num_gaussians=N)

        # Positions
        model.means = torch.tensor(points3d[:, :3], dtype=torch.float32, device=device)
        model.means.requires_grad_(True)

        # Compute initial scale from average nearest-neighbor distance
        from scipy.spatial import KDTree

        try:
            if N < 2:
                init_scale = np.full(N, np.log(0.01))
            else:
                tree = KDTree(points3d[:, :3])
                neighbor_count = min(4, N)
                dists, _ = tree.query(points3d[:, :3], k=neighbor_count)
                neighbor_dists = np.asarray(dists)[:, 1:]  # exclude self
                finite_dists = np.where(np.isfinite(neighbor_dists), neighbor_dists, np.nan)
                avg_dist = np.nanmean(finite_dists, axis=1)
                avg_dist = np.where(np.isfinite(avg_dist), avg_dist, 0.01)
                avg_dist = np.clip(avg_dist, 1e-7, None)
                init_scale = np.log(avg_dist)
        except Exception:
            init_scale = np.full(N, np.log(0.01))

        model.scales = torch.tensor(
            np.stack([init_scale, init_scale, init_scale], axis=-1),
            dtype=torch.float32,
            device=device,
        )
        model.scales.requires_grad_(True)

        # Rotations (identity quaternion)
        model.rotations = torch.zeros(N, 4, dtype=torch.float32, device=device)
        model.rotations[:, 0] = 1.0  # w=1, x=y=z=0
        model.rotations.requires_grad_(True)

        # Opacities (sigmoid pre-activation, init to ~0.1)
        model.opacities = torch.full((N, 1), fill_value=-2.2, dtype=torch.float32, device=device)  # sigmoid(-2.2) ≈ 0.1
        model.opacities.requires_grad_(True)

        # SH coefficients: DC component from point colors
        sh_degree = self.config.get("sh_degree", 3)
        num_sh = (sh_degree + 1) ** 2
        model.sh_coeffs = torch.zeros(N, num_sh, 3, dtype=torch.float32, device=device)
        # Set DC component (index 0) from colors (convert from [0,1] to SH space)
        colors = torch.tensor(points3d[:, 3:6], dtype=torch.float32, device=device)
        # C0 = 0.28209479177387814 (SH basis constant)
        C0 = 0.28209479177387814
        model.sh_coeffs[:, 0, :] = (colors - 0.5) / C0
        model.sh_coeffs.requires_grad_(True)

        return model

    def _create_optimizers(self, gaussians: GaussianModel) -> dict[str, Any]:
        """Create Adam optimizers for each Gaussian parameter group.

        Args:
            gaussians: The Gaussian model.

        Returns:
            Dictionary of optimizers keyed by parameter name.
        """
        import torch

        lr = self.config.get("learning_rate", {})
        optimizers = {
            "position": torch.optim.Adam([gaussians.means], lr=lr.get("position", 0.00016)),
            "feature": torch.optim.Adam([gaussians.sh_coeffs], lr=lr.get("feature", 0.0025)),
            "opacity": torch.optim.Adam([gaussians.opacities], lr=lr.get("opacity", 0.05)),
            "scaling": torch.optim.Adam([gaussians.scales], lr=lr.get("scaling", 0.005)),
            "rotation": torch.optim.Adam([gaussians.rotations], lr=lr.get("rotation", 0.001)),
        }
        return optimizers

    def _update_lr(self, optimizer: Any, iteration: int, max_iterations: int) -> None:
        """Update learning rate with exponential decay for position parameters."""
        lr_schedule = self.config.get("lr_schedule", {})
        lr_init = lr_schedule.get("position_lr_init", 0.00016)
        lr_final = lr_schedule.get("position_lr_final", 0.0000016)
        max_steps = lr_schedule.get("position_lr_max_steps", max_iterations)

        t = min(iteration / max_steps, 1.0)
        # Exponential decay
        import math

        lr = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _render(self, gaussians: GaussianModel, viewmat: Any, K: Any, H: int, W: int, device: Any) -> Any:
        """Render an image from the Gaussian model.

        Uses gsplat if available, otherwise falls back to a simplified
        splatting implementation.

        Args:
            gaussians: The Gaussian model.
            viewmat: 4x4 view matrix.
            K: 3x3 intrinsic matrix.
            H: Image height.
            W: Image width.
            device: Torch device.

        Returns:
            Rendered image tensor of shape (H, W, 3).
        """
        if self._has_gsplat:
            return self._render_gsplat(gaussians, viewmat, K, H, W, device)
        else:
            return self._render_simple(gaussians, viewmat, K, H, W, device)

    def _render_gsplat(
        self, gaussians: GaussianModel, viewmat: Any, K: Any, H: int, W: int, device: Any, want_depth: bool = False
    ) -> Any:
        """Render using the gsplat library. Returns (RGB) or (RGB, depth)."""
        import torch
        from gsplat import rasterization

        means = gaussians.means
        scales = torch.exp(gaussians.scales)
        quats = torch.nn.functional.normalize(gaussians.rotations, dim=-1)
        opacities = torch.sigmoid(gaussians.opacities).squeeze(-1)

        C0 = 0.28209479177387814
        colors = gaussians.sh_coeffs[:, 0, :] * C0 + 0.5
        colors = torch.clamp(colors, 0, 1)

        render_mode = "RGB+D" if want_depth else "RGB"
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            render_mode=render_mode,
        )
        output = rendered.squeeze(0)  # (H, W, C)
        if want_depth:
            rgb = output[..., :3]
            depth = output[..., 3]
            return rgb, depth
        return output

    def _render_simple(self, gaussians: GaussianModel, viewmat: Any, K: Any, H: int, W: int, device: Any) -> Any:
        """Simple differentiable point splatting (fallback without gsplat).

        Projects 3D Gaussians to 2D, renders with alpha blending.
        This is a simplified renderer for demonstration purposes.
        """
        import torch

        means = gaussians.means  # (N, 3)
        opacities = torch.sigmoid(gaussians.opacities).squeeze(-1)  # (N,)

        # SH DC to color
        C0 = 0.28209479177387814
        colors = gaussians.sh_coeffs[:, 0, :] * C0 + 0.5  # (N, 3)
        colors = torch.clamp(colors, 0, 1)

        # Transform to camera space
        R = viewmat[:3, :3]  # (3, 3)
        t = viewmat[:3, 3]  # (3,)
        cam_points = (means @ R.T) + t  # (N, 3)

        # Depth filtering
        depth = cam_points[:, 2]
        valid = depth > 0.01
        cam_points = cam_points[valid]
        colors_valid = colors[valid]
        opacities_valid = opacities[valid]
        depth_valid = depth[valid]

        if cam_points.shape[0] == 0:
            return torch.zeros(H, W, 3, device=device)

        # Project to image plane
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        px = (cam_points[:, 0] / cam_points[:, 2]) * fx + cx
        py = (cam_points[:, 1] / cam_points[:, 2]) * fy + cy

        # Sort by depth (back to front for alpha blending)
        sort_idx = torch.argsort(depth_valid, descending=True)
        px = px[sort_idx]
        py = py[sort_idx]
        colors_valid = colors_valid[sort_idx]
        opacities_valid = opacities_valid[sort_idx]

        # Initialize output
        output = torch.zeros(H, W, 3, device=device)
        weight_sum = torch.zeros(H, W, 1, device=device)

        # For efficiency, use scatter-based approach
        px_int = torch.round(px).long()
        py_int = torch.round(py).long()

        # Filter to valid pixel coordinates
        in_bounds = (px_int >= 0) & (px_int < W) & (py_int >= 0) & (py_int < H)
        px_int = px_int[in_bounds]
        py_int = py_int[in_bounds]
        c = colors_valid[in_bounds]
        a = opacities_valid[in_bounds].unsqueeze(-1)

        # Simple point splatting (single pixel per point for efficiency)
        pixel_idx = py_int * W + px_int
        weighted_colors = c * a
        output_flat = output.view(-1, 3)
        weight_flat = weight_sum.view(-1, 1)

        output_flat.scatter_add_(0, pixel_idx.unsqueeze(-1).expand(-1, 3), weighted_colors)
        weight_flat.scatter_add_(0, pixel_idx.unsqueeze(-1), a)

        # Normalize
        output = output_flat.view(H, W, 3)
        weight_sum = weight_flat.view(H, W, 1)
        output = output / (weight_sum + 1e-8)
        output = torch.clamp(output, 0, 1)

        return output

    def _simple_ssim(self, img1: Any, img2: Any) -> Any:
        """Compute a simplified SSIM between two images.

        Args:
            img1: First image tensor (H, W, 3).
            img2: Second image tensor (H, W, 3).

        Returns:
            Scalar SSIM value.
        """
        import torch
        import torch.nn.functional as F

        # Convert to (1, 3, H, W) for conv2d
        x = img1.permute(2, 0, 1).unsqueeze(0)
        y = img2.permute(2, 0, 1).unsqueeze(0)

        C1 = 0.01**2
        C2 = 0.03**2

        # Simple 11x11 uniform window
        window_size = 11
        pad = window_size // 2
        window = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size * window_size)
        window = window.expand(3, 1, -1, -1)

        mu_x = F.conv2d(x, window, padding=pad, groups=3)
        mu_y = F.conv2d(y, window, padding=pad, groups=3)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )

        return ssim_map.mean()

    def _densify_and_prune(self, gaussians: GaussianModel, avg_grad: Any, optimizers: dict, device: Any) -> None:
        """Densify (split/clone) and prune Gaussians based on gradients.

        Args:
            gaussians: The Gaussian model to modify.
            avg_grad: Average gradient norms for each Gaussian.
            optimizers: Dictionary of optimizers.
            device: Torch device.
        """
        import torch

        grad_threshold = self.config.get("densify_grad_threshold", 0.0002)
        min_opacity = self.config.get("min_opacity", 0.005)
        percent_dense = self.config.get("percent_dense", 0.01)
        N = gaussians.num_gaussians

        # Identify Gaussians to densify
        selected = avg_grad[:N] >= grad_threshold
        if not selected.any():
            return

        # Get scales
        scales = torch.exp(gaussians.scales)
        scale_norm = scales.norm(dim=-1)
        scene_extent = gaussians.means.detach().std().item() * 3

        # Clone small Gaussians (under-reconstruction)
        clone_mask = selected & (scale_norm < percent_dense * scene_extent)
        num_clone = clone_mask.sum().item()

        # Split large Gaussians (over-reconstruction)
        split_mask = selected & (scale_norm >= percent_dense * scene_extent)
        num_split = split_mask.sum().item()

        clone_payload = None
        if num_clone > 0:
            clone_payload = (
                gaussians.means[clone_mask].detach().clone(),
                gaussians.scales[clone_mask].detach().clone(),
                gaussians.rotations[clone_mask].detach().clone(),
                gaussians.opacities[clone_mask].detach().clone(),
                gaussians.sh_coeffs[clone_mask].detach().clone(),
            )

        split_payload = None
        if num_split > 0:
            split_means = gaussians.means[split_mask].detach()
            offset = torch.randn_like(split_means) * torch.exp(gaussians.scales[split_mask]).detach()
            split_payload = (
                split_means + offset,
                gaussians.scales[split_mask].detach().clone() - np.log(1.6),
                gaussians.rotations[split_mask].detach().clone(),
                gaussians.opacities[split_mask].detach().clone(),
                gaussians.sh_coeffs[split_mask].detach().clone(),
            )

        if clone_payload is not None:
            self._extend_gaussians(gaussians, *clone_payload, device)
        if split_payload is not None:
            self._extend_gaussians(gaussians, *split_payload, device)

        # Prune low-opacity Gaussians
        opacity_vals = torch.sigmoid(gaussians.opacities).squeeze(-1)
        prune_mask = opacity_vals > min_opacity
        if prune_mask.sum() < gaussians.num_gaussians:
            self._prune_gaussians(gaussians, prune_mask, device)

        logger.debug(
            "Densification: cloned=%d, split=%d, total=%d",
            num_clone,
            num_split,
            gaussians.num_gaussians,
        )

    def _extend_gaussians(
        self,
        gaussians: GaussianModel,
        new_means: Any,
        new_scales: Any,
        new_rotations: Any,
        new_opacities: Any,
        new_sh: Any,
        device: Any,
    ) -> None:
        """Add new Gaussians to the model."""
        import torch

        gaussians.means = torch.nn.Parameter(torch.cat([gaussians.means.data, new_means], dim=0))
        gaussians.scales = torch.nn.Parameter(torch.cat([gaussians.scales.data, new_scales], dim=0))
        gaussians.rotations = torch.nn.Parameter(torch.cat([gaussians.rotations.data, new_rotations], dim=0))
        gaussians.opacities = torch.nn.Parameter(torch.cat([gaussians.opacities.data, new_opacities], dim=0))
        gaussians.sh_coeffs = torch.nn.Parameter(torch.cat([gaussians.sh_coeffs.data, new_sh], dim=0))
        gaussians.num_gaussians = gaussians.means.shape[0]

    def _prune_gaussians(self, gaussians: GaussianModel, keep_mask: Any, device: Any) -> None:
        """Remove Gaussians that don't pass the keep mask."""
        import torch

        gaussians.means = torch.nn.Parameter(gaussians.means.data[keep_mask])
        gaussians.scales = torch.nn.Parameter(gaussians.scales.data[keep_mask])
        gaussians.rotations = torch.nn.Parameter(gaussians.rotations.data[keep_mask])
        gaussians.opacities = torch.nn.Parameter(gaussians.opacities.data[keep_mask])
        gaussians.sh_coeffs = torch.nn.Parameter(gaussians.sh_coeffs.data[keep_mask])
        gaussians.num_gaussians = gaussians.means.shape[0]

    def _reset_opacity(self, gaussians: GaussianModel) -> None:
        """Reset opacities to a low value to encourage pruning."""
        import torch

        # Reset to sigmoid^{-1}(0.01) ≈ -4.6
        new_opacity = torch.full_like(gaussians.opacities.data, -4.6)
        # Keep higher opacity for those already trained
        mask = gaussians.opacities.data > -4.6
        new_opacity[mask] = gaussians.opacities.data[mask]
        gaussians.opacities = torch.nn.Parameter(new_opacity)

    def _save_model(self, output_path: Path, gaussians: GaussianModel) -> None:
        """Save trained Gaussian model as a .ply file.

        The PLY format includes: positions, scales, rotations, opacities,
        and spherical harmonics coefficients.

        Args:
            output_path: Path to save the .ply file.
            gaussians: The trained Gaussian model.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        means = gaussians.means.detach().cpu().numpy()
        scales = gaussians.scales.detach().cpu().numpy()
        rotations = gaussians.rotations.detach().cpu().numpy()
        opacities = gaussians.opacities.detach().cpu().numpy()
        sh_coeffs = gaussians.sh_coeffs.detach().cpu().numpy()

        N = means.shape[0]
        num_sh = sh_coeffs.shape[1]

        # Build PLY header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
        ]
        # SH coefficients
        for i in range(num_sh):
            for c in range(3):
                ch = ["r", "g", "b"][c]
                header_lines.append(
                    f"property float f_dc_{i}_{ch}" if i == 0 else f"property float f_rest_{(i - 1) * 3 + c}"
                )

        # Flatten SH naming for compatibility with standard 3DGS format
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
        ]
        # DC component
        header_lines.extend(
            [
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
            ]
        )
        # Rest SH coefficients
        for i in range(1, num_sh):
            for c in range(3):
                idx = (i - 1) * 3 + c
                header_lines.append(f"property float f_rest_{idx}")

        header_lines.append("property float opacity")
        header_lines.extend(
            [
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
            ]
        )
        header_lines.extend(
            [
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
            ]
        )
        header_lines.append("end_header")

        header = "\n".join(header_lines) + "\n"

        with open(output_path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(N):
                # Position
                f.write(struct.pack("<3f", *means[i]))
                # Normals (zeros)
                f.write(struct.pack("<3f", 0, 0, 0))
                # SH DC
                f.write(struct.pack("<3f", *sh_coeffs[i, 0]))
                # SH rest
                for j in range(1, num_sh):
                    f.write(struct.pack("<3f", *sh_coeffs[i, j]))
                # Opacity
                f.write(struct.pack("<f", opacities[i, 0]))
                # Scale
                f.write(struct.pack("<3f", *scales[i]))
                # Rotation
                f.write(struct.pack("<4f", *rotations[i]))

        logger.info("Saved %d Gaussians to %s", N, output_path)


def train_gsplat(
    data_dir: Path | str,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
    num_iterations: int = 30000,
) -> Path:
    """Train a 3D Gaussian Splatting model using gsplat.

    Convenience function that creates a GsplatTrainer and runs training.

    Args:
        data_dir: Directory containing preprocessed data (images + poses).
        output_dir: Directory where training outputs will be saved.
        config: Optional dictionary of training hyperparameters.
            If None, defaults from configs/training.yaml are used.
        num_iterations: Number of training iterations.

    Returns:
        Path to the final exported .ply file.

    Raises:
        ImportError: If required dependencies are not installed.
        FileNotFoundError: If the data directory is missing required files.
    """
    trainer = GsplatTrainer(config=config)
    return trainer.train(
        data_dir=data_dir,
        output_dir=output_dir,
        num_iterations=num_iterations,
    )
