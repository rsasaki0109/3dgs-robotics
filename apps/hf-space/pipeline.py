"""Core photos/video -> .splat pipeline for the 3DGS Robotics zero-install demo.

Kept free of gradio imports so it can be smoke-tested headless:

    python pipeline.py --images ./photos --output /tmp/out --method simple --iterations 50
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DUST3R_ROOT = APP_ROOT / "third_party" / "dust3r"
DUST3R_HF_ID = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_IMAGE_SIDE = 1280

logger = logging.getLogger("gs_mapper_space")

ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineResult:
    splat_path: Path
    num_input_images: int
    num_used_frames: int
    method: str
    elapsed_sec: float


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def ensure_dust3r(root: Path | None = None) -> Path:
    """Clone naver/dust3r next to the app if it is not already available."""
    resolved = Path(root or os.environ.get("DUST3R_PATH") or DEFAULT_DUST3R_ROOT)
    if not (resolved / "dust3r").exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        logger.info("cloning naver/dust3r into %s", resolved)
        subprocess.run(
            ["git", "clone", "--recursive", "--depth", "1", "https://github.com/naver/dust3r", str(resolved)],
            check=True,
        )
    os.environ["DUST3R_PATH"] = str(resolved)
    return resolved


def preflight_dust3r(root: Path) -> None:
    """Import dust3r eagerly so missing deps fail loudly instead of silently
    falling back to the meaningless circular-camera initialization."""
    from gs_sim2real.preprocess.pose_free import _add_dust3r_to_path

    _add_dust3r_to_path(root)
    import dust3r.model  # noqa: F401, PLC0415


def prefetch_checkpoint() -> None:
    """Download the DUSt3R checkpoint into the HF cache (CPU-safe, idempotent)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(DUST3R_HF_ID)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("checkpoint prefetch failed (will retry on first run): %s", exc)


def extract_video_frames(video_path: Path, out_dir: Path, max_frames: int) -> int:
    """Sample evenly spaced frames from a video into out_dir as JPEGs."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    if total > 0:
        import numpy as np

        indices = np.linspace(0, total - 1, min(max_frames, total)).round().astype(int)
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), _resize_max_side(frame))
            saved += 1
    else:  # stream without frame count metadata
        index = 0
        while saved < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % 10 == 0:
                cv2.imwrite(str(out_dir / f"frame_{saved:04d}.jpg"), _resize_max_side(frame))
                saved += 1
            index += 1
    cap.release()
    if saved < 2:
        raise ValueError(f"Could not extract at least 2 frames from {video_path.name}")
    return saved


def _resize_max_side(image, max_side: int = MAX_IMAGE_SIDE):
    import cv2

    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def prepare_images(image_paths: list[Path], out_dir: Path) -> int:
    """Normalize uploads to bounded-size JPEGs the pipeline can consume."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src in sorted(image_paths):
        if src.suffix.lower() not in IMAGE_EXTENSIONS:
            logger.warning("skipping unsupported file: %s", src.name)
            continue
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("skipping unreadable image: %s", src.name)
            continue
        cv2.imwrite(str(out_dir / f"img_{saved:04d}.jpg"), _resize_max_side(image))
        saved += 1
    if saved < 2:
        raise ValueError(f"Need at least 2 readable images, got {saved}")
    return saved


def run_pipeline(
    images_dir: Path,
    output_dir: Path,
    *,
    method: str = "dust3r",
    num_frames: int = 10,
    iterations: int = 2000,
    align_iters: int = 150,
    scene_graph: str | None = None,
    splat_max_points: int = 400000,
    splat_normalize_extent: float | None = 17.0,
    progress: ProgressFn = _noop_progress,
) -> PipelineResult:
    """images dir -> pose-free sparse -> gsplat training -> .splat binary."""
    from gs_sim2real.preprocess.pose_free import PoseFreeProcessor
    from gs_sim2real.train.gsplat_trainer import train_gsplat
    from gs_sim2real.viewer.web_export import ply_to_splat

    start = time.time()
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_input = len([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

    if scene_graph is None:
        scene_graph = "complete" if num_frames <= 12 else "swin-3"

    processor_kwargs: dict = {
        "method": method,
        "num_frames": num_frames,
        "align_iters": align_iters,
        "scene_graph": scene_graph,
    }
    if method == "dust3r":
        dust3r_root = ensure_dust3r()
        preflight_dust3r(dust3r_root)
        processor_kwargs["dust3r_root"] = dust3r_root
        # The HF hub id is accepted verbatim by AsymmetricCroCo3DStereo.from_pretrained.
        processor_kwargs["checkpoint"] = Path(DUST3R_HF_ID)

    progress(0.05, f"Step 1/3: pose-free preprocessing ({method}, {scene_graph})")
    sparse_dir = output_dir / "sparse_input"
    processor = PoseFreeProcessor(**processor_kwargs)
    processor.estimate_poses(images_dir, sparse_dir)

    progress(0.45, f"Step 2/3: 3DGS training ({iterations} iterations)")
    ply_path = train_gsplat(
        data_dir=sparse_dir,
        output_dir=output_dir / "train",
        num_iterations=iterations,
    )

    progress(0.9, "Step 3/3: exporting browser .splat")
    splat_path = output_dir / "scene.splat"
    ply_to_splat(
        ply_path,
        splat_path,
        max_points=splat_max_points,
        # 17.0 matches the docs/splat.html viewer defaults (same as the CLI).
        normalize_target_extent=splat_normalize_extent,
        min_opacity=0.02,
        max_scale=2.0,
    )

    used = min(num_frames, num_input) if num_frames > 0 else num_input
    elapsed = time.time() - start
    progress(1.0, f"Done in {elapsed:.0f}s")
    return PipelineResult(
        splat_path=splat_path,
        num_input_images=num_input,
        num_used_frames=used,
        method=method,
        elapsed_sec=elapsed,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Headless smoke test for the Space pipeline")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="dust3r", choices=["dust3r", "simple"])
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    staged = Path(args.output) / "staged_images"
    images = [p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    prepare_images(images, staged)
    result = run_pipeline(
        staged,
        Path(args.output),
        method=args.method,
        num_frames=args.num_frames,
        iterations=args.iterations,
        progress=lambda f, m: print(f"[{f * 100:5.1f}%] {m}"),
    )
    print(f"splat: {result.splat_path} ({result.splat_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
