#!/usr/bin/env python3
"""End-to-end VGGT feedforward pose-free preprocessing.

Runs facebook/VGGT-1B on a directory of images and exports a COLMAP sparse model
ready for gsplat training. Requires a local clone of facebookresearch/vggt.

    export VGGT_PATH=/tmp/vggt
    python scripts/run_vggt.py \\
        --image-dir outputs/my_frames/images \\
        --output    outputs/my_frames_vggt \\
        --num-frames 24
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VGGT and export a COLMAP sparse model.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="VGGT checkpoint .pt path or Hugging Face hub id (default: facebook/VGGT-1B)",
    )
    parser.add_argument("--vggt-root", type=Path, default=Path(os.environ.get("VGGT_PATH", "/tmp/vggt")))
    parser.add_argument("--num-frames", type=int, default=24, help="0 = keep all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=100000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

    processor = PoseFreeProcessor(
        method="vggt",
        checkpoint=args.checkpoint,
        vggt_root=args.vggt_root,
        num_frames=args.num_frames,
        device=args.device,
        max_points=args.max_points,
    )
    sparse_dir = processor.estimate_poses(args.image_dir, args.output)
    print(f"COLMAP sparse model: {sparse_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
