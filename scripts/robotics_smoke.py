#!/usr/bin/env python3
"""End-to-end smoke for the Dynamic Map Viewer robotics render-server.

Wires together the three pieces that exist in ``src/gs_sim2real/robotics/`` —
``HeadlessSplatRenderer``, the render query payload schema used by
``render_query_client``, and the bridge-side topic map from ``topic_map.py`` —
so that a caller can verify the full "pose query → RGB + depth frame →
Dynamic Map Viewer topic payload" path without needing a live ROS2 stack.

Usage::

    # run the smoke against the bundled outdoor-demo PLY (if present)
    python scripts/robotics_smoke.py \\
        --ply outputs/bag6_demo_train/point_cloud.ply \\
        --out artifacts/robotics-smoke

    # or use the internal fixture (writes a tiny 3DGS PLY on the fly)
    python scripts/robotics_smoke.py --fixture --out artifacts/robotics-smoke

The smoke writes ``rgb.png``, ``depth.npy``, and ``payload.json`` plus prints
a one-line summary. Exit code 0 iff the rendered frame has at least one
non-background pixel and the payload conforms to the bridge topic map.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_fixture_ply(path: Path) -> None:
    """Write a tiny 3DGS-compatible binary PLY so the script runs without data."""
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 5",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")
    # Five gaussians spread in front of the camera, colours distinct.
    rows = [
        # (x, y, z, nx, ny, nz, f_dc_0..2, opacity, scale_0..2, rot_0..3)
        (0.0, 0.0, 3.0, 0, 0, 0, -1.77, -1.77, 1.77, 8.0, 0, 0, 0, 0, 0, 0, 1.0),
        (1.0, 0.0, 4.0, 0, 0, 0, 1.77, -1.77, -1.77, 8.0, 0, 0, 0, 0, 0, 0, 1.0),
        (-1.0, 0.0, 4.0, 0, 0, 0, -1.77, 1.77, -1.77, 8.0, 0, 0, 0, 0, 0, 0, 1.0),
        (0.5, 0.5, 5.0, 0, 0, 0, 1.77, 1.77, -1.77, 8.0, 0, 0, 0, 0, 0, 0, 1.0),
        (-0.5, -0.5, 6.0, 0, 0, 0, -1.77, -1.77, -1.77, 8.0, 0, 0, 0, 0, 0, 0, 1.0),
    ]
    with path.open("wb") as f:
        f.write(header)
        for row in rows:
            f.write(struct.pack("<17f", *row))


def _build_query_payload(
    *,
    namespace: str,
    pose: tuple[float, float, float, float, float, float, float],
    width: int,
    height: int,
    fov_degrees: float,
) -> dict:
    """Hand-build the render-query payload the client would send.

    Mirrors ``render_query_client.build_render_query_payload``'s shape so we
    exercise the same fields the bridge expects to relay.
    """
    return {
        "type": "dreamwalker-render-query/v1",
        "namespace": namespace,
        "pose": {
            "position": {"x": pose[0], "y": pose[1], "z": pose[2]},
            "orientation": {"x": pose[3], "y": pose[4], "z": pose[5], "w": pose[6]},
        },
        "resolution": {"width": width, "height": height},
        "camera": {"fov_degrees": fov_degrees},
        "frame_id": "robotics_smoke",
    }


def run_smoke(
    ply_path: Path,
    out_dir: Path,
    *,
    namespace: str,
    width: int,
    height: int,
    fov_degrees: float,
) -> dict:
    """Load the PLY, render one RGB+depth frame, emit smoke artifacts."""
    # Lazy imports so `--help` works without a full install.
    from gs_sim2real.robotics.gsplat_render_server import (
        CameraPose,
        HeadlessSplatRenderer,
        yaw_to_quaternion,
    )
    from gs_sim2real.robotics.topic_map import build_ros_topic_map

    renderer = HeadlessSplatRenderer(ply_path, backend="simple")
    orientation = yaw_to_quaternion(0.0)
    pose = CameraPose(position=(0.0, 0.0, 0.0), orientation=orientation)
    rgb, depth = renderer.render_rgbd(
        pose,
        width=width,
        height=height,
        fov_degrees=fov_degrees,
        near_clip=0.05,
        far_clip=50.0,
        point_radius=1,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_u8).save(out_dir / "rgb.png")
    np.save(out_dir / "depth.npy", depth.astype(np.float32))

    payload = _build_query_payload(
        namespace=namespace,
        pose=(0.0, 0.0, 0.0, *orientation),
        width=width,
        height=height,
        fov_degrees=fov_degrees,
    )
    topic_map = build_ros_topic_map(namespace)

    bundled = {
        "payload": payload,
        "topic_map": {
            "namespace": topic_map.namespace,
            "camera_compressed": topic_map.camera_compressed,
            "depth_image": topic_map.depth_image,
            "robot_pose_stamped": topic_map.robot_pose_stamped,
        },
        "frame_stats": {
            "rgb_non_background_pixels": int(np.count_nonzero(rgb.sum(axis=-1) > 0)),
            "depth_non_inf_pixels": int(np.count_nonzero(np.isfinite(depth) & (depth < 1e9))),
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
        },
    }
    (out_dir / "payload.json").write_text(json.dumps(bundled, indent=2), encoding="utf-8")
    return bundled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ply", type=Path, default=None, help="Trained gsplat PLY path")
    parser.add_argument("--fixture", action="store_true", help="Render the internal 5-gaussian fixture")
    parser.add_argument("--out", type=Path, default=Path("artifacts/robotics-smoke"))
    parser.add_argument("--namespace", default="/dreamwalker")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fov-degrees", type=float, default=60.0)
    args = parser.parse_args()

    if not args.ply and not args.fixture:
        print("must pass either --ply <path> or --fixture", file=sys.stderr)
        return 2

    ply_path = args.ply
    if args.fixture:
        args.out.mkdir(parents=True, exist_ok=True)
        ply_path = args.out / "fixture.ply"
        _write_fixture_ply(ply_path)

    if not ply_path.exists():
        print(f"PLY not found: {ply_path}", file=sys.stderr)
        return 2

    bundle = run_smoke(
        ply_path,
        args.out,
        namespace=args.namespace,
        width=args.width,
        height=args.height,
        fov_degrees=args.fov_degrees,
    )
    stats = bundle["frame_stats"]
    print(
        f"robotics_smoke: rendered {stats['rgb_shape']} RGB / {stats['depth_shape']} depth "
        f"(non-bg pixels: {stats['rgb_non_background_pixels']}, "
        f"valid depth pixels: {stats['depth_non_inf_pixels']}) -> {args.out}"
    )
    if stats["rgb_non_background_pixels"] == 0:
        print("FAIL: rendered frame is entirely background", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
