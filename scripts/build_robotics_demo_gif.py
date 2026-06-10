#!/usr/bin/env python3
"""Compose the robotics closed-loop GIF from a live-mapping session.

One GIF, all four robotics pillars: a virtual camera (GS camera simulator)
flies the mapped trajectory while the 3DGS localizer re-estimates its pose
from the rendered pixels alone, plotted over the nav2 occupancy grid built
from the same map.

Top panel: the simulated camera view rendered with gsplat at the session's
calibrated intrinsics. Bottom panel: the occupancy grid (free corridor +
obstacles), the ground-truth trajectory, the camera marker, and the
localizer's estimates accumulating as orange dots.

    python3 scripts/build_robotics_demo_gif.py \
        --session outputs/e2e_bag_live3 \
        --output docs/images/robotics/robotics-loop.gif
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gs_sim2real.robotics.camera_sim_node import (  # noqa: E402
    camera_intrinsics_from_colmap,
    render_optical,
    replay_poses,
    scale_intrinsics,
)
from gs_sim2real.robotics.gsplat_render_server import CameraPose, HeadlessSplatRenderer  # noqa: E402
from gs_sim2real.robotics.localize import _slerp_quat  # noqa: E402
from gs_sim2real.robotics.occupancy_grid import GridParams, build_occupancy_grid  # noqa: E402

FRAME_DURATION_MS = 350
LAST_FRAME_HOLD_MS = 2200

INK = (236, 240, 246)
PANEL_BG = (12, 16, 24)
GT_TRAIL = (110, 200, 130)
ESTIMATE_DOT = (255, 150, 60)
CAMERA_DOT = (96, 205, 255)
GRID_FREE = (252, 252, 252)
GRID_OCCUPIED = (212, 80, 80)
GRID_UNKNOWN = (52, 58, 70)


def interpolate_poses(poses: list[tuple[str, CameraPose]], steps_between: int) -> list[tuple[str, CameraPose, bool]]:
    """Insert ``steps_between`` interpolated poses between consecutive keyframes.

    Returns (label, pose, is_keyframe); interpolated poses lerp the position
    and slerp the orientation.
    """
    if steps_between <= 0 or len(poses) < 2:
        return [(name, pose, True) for name, pose in poses]
    out: list[tuple[str, CameraPose, bool]] = []
    for (name_a, pose_a), (_name_b, pose_b) in zip(poses[:-1], poses[1:]):
        out.append((name_a, pose_a, True))
        pos_a = np.asarray(pose_a.position, dtype=np.float64)
        pos_b = np.asarray(pose_b.position, dtype=np.float64)
        for step in range(1, steps_between + 1):
            t = step / (steps_between + 1)
            position = tuple((1.0 - t) * pos_a + t * pos_b)
            orientation = _slerp_quat(pose_a.orientation, pose_b.orientation, t)
            out.append((f"{name_a}+{step}", CameraPose(position=position, orientation=orientation), False))
    out.append((poses[-1][0], poses[-1][1], True))
    return out


@dataclass
class GridView:
    """Occupancy grid rendered to RGB plus the world -> pixel mapping."""

    image: np.ndarray  # (H, W, 3) uint8, row 0 = min grid-y
    basis2: np.ndarray  # (2, 3) world -> grid-plane coords
    origin: tuple[float, float]
    resolution: float

    def world_to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        xy = np.asarray(point, dtype=np.float64) @ self.basis2.T
        col = int(round((xy[0] - self.origin[0]) / self.resolution))
        row = int(round((xy[1] - self.origin[1]) / self.resolution))
        height, width = self.image.shape[:2]
        return min(max(col, 0), width - 1), min(max(row, 0), height - 1)


def build_grid_view(points, opacities, camera_centers, camera_qvecs) -> GridView:
    grid = build_occupancy_grid(points, opacities, camera_centers, camera_qvecs, params=GridParams())
    image = np.empty((*grid.data.shape, 3), dtype=np.uint8)
    image[:] = GRID_UNKNOWN
    image[grid.data == 0] = GRID_FREE
    image[grid.data == 100] = GRID_OCCUPIED
    return GridView(image=image, basis2=grid.basis[:2], origin=grid.origin, resolution=grid.resolution)


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_frame(
    camera_rgb: np.ndarray,
    grid_view: GridView,
    *,
    gt_centers: np.ndarray,
    estimates: list[np.ndarray],
    current: np.ndarray,
    width: int,
) -> Image.Image:
    """Stack the simulated camera view over the annotated occupancy grid."""
    font = _load_font(16)
    camera = Image.fromarray(camera_rgb)
    if camera.width != width:
        camera = camera.resize((width, int(round(camera.height * width / camera.width))), Image.LANCZOS)

    grid = Image.fromarray(grid_view.image[::-1])  # row 0 = max grid-y for display
    map_scale = width / grid.width
    grid = grid.resize((width, max(int(round(grid.height * map_scale)), 1)), Image.NEAREST)
    draw = ImageDraw.Draw(grid)

    def to_display(point: np.ndarray) -> tuple[float, float]:
        col, row = grid_view.world_to_pixel(point)
        return col * map_scale, (grid_view.image.shape[0] - 1 - row) * map_scale

    trail = [to_display(center) for center in gt_centers]
    if len(trail) >= 2:
        draw.line(trail, fill=GT_TRAIL, width=2)
    for estimate in estimates:
        x, y = to_display(estimate)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=ESTIMATE_DOT)
    x, y = to_display(current)
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=CAMERA_DOT, width=3)

    caption_h = 26
    total = Image.new("RGB", (width, camera.height + grid.height + caption_h * 2), PANEL_BG)
    total.paste(camera, (0, caption_h))
    total.paste(grid, (0, caption_h + camera.height + caption_h))
    head = ImageDraw.Draw(total)
    head.text((10, 5), "3DGS camera simulator - virtual view in the splat map", fill=INK, font=font)
    head.text(
        (10, caption_h + camera.height + 5),
        "nav2 occupancy grid - ground truth (green) vs 3DGS localizer (orange)",
        fill=INK,
        font=font,
    )
    return total


def build_gif(args: argparse.Namespace) -> dict:
    import cv2

    from gs_sim2real.robotics.gsplat_render_server import sigmoid
    from gs_sim2real.robotics.localize import (
        LocalizeConfig,
        SessionLocalizer,
        _load_cameras_txt,
        load_mapped_records,
        resolve_live_map_session,
    )
    from gs_sim2real.viewer.web_viewer import load_ply

    session = resolve_live_map_session(Path(args.session), round_index=args.round)
    records = load_mapped_records(session)
    cameras = _load_cameras_txt(session.round.cameras_txt)
    cam = cameras[records[0].camera_id]
    native_w, native_h, fx, fy, cx, cy = camera_intrinsics_from_colmap(cam)
    render_w = args.width
    render_h = int(round(native_h * render_w / native_w))
    intrinsics = scale_intrinsics((fx, fy, cx, cy), from_size=(native_w, native_h), to_size=(render_w, render_h))

    renderer = HeadlessSplatRenderer(session.round.ply_path, backend="auto", max_points=None)
    localizer = SessionLocalizer(
        Path(args.session),
        round_index=args.round,
        config=LocalizeConfig(refine_iters=args.refine_iters, pyramid_scales=(0.25, 0.5)),
    )

    ply = load_ply(session.round.ply_path)
    opacities = (
        sigmoid(np.asarray(ply.opacities, dtype=np.float32))
        if ply.opacities is not None
        else np.ones(len(ply.positions))
    )
    gt_centers = np.asarray([record.center for record in records], dtype=np.float64)
    grid_view = build_grid_view(np.asarray(ply.positions), opacities, gt_centers, [record.qvec for record in records])

    poses = interpolate_poses(replay_poses(records), args.steps_between)
    frames: list[Image.Image] = []
    estimates: list[np.ndarray] = []
    errors: list[float] = []
    for index, (label, pose, is_keyframe) in enumerate(poses):
        rgb, _depth = render_optical(
            renderer,
            pose,
            width=render_w,
            height=render_h,
            intrinsics=intrinsics,
            near_clip=0.001,
            far_clip=500.0,
            point_radius=1,
        )
        current = np.asarray(pose.position, dtype=np.float64)
        status = "render only"
        # interpolated views sit too far from any retrieval seed for the
        # photometric refinement basin — localize the keyframe views only,
        # with the localizer node's seed-distance gate
        if is_keyframe:
            result = localizer.localize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), query_name=label)
            if result.seed_distance <= args.max_seed_distance:
                estimates.append(np.asarray(result.center, dtype=np.float64))
                errors.append(float(np.linalg.norm(estimates[-1] - current)))
                status = f"error {errors[-1]:.4f} (gauge units)"
            else:
                status = f"rejected (seed distance {result.seed_distance:.3f})"
        frames.append(
            compose_frame(
                rgb,
                grid_view,
                gt_centers=gt_centers,
                estimates=estimates,
                current=current,
                width=args.width,
            )
        )
        print(f"[{index + 1}/{len(poses)}] {label}: {status}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [FRAME_DURATION_MS] * (len(frames) - 1) + [LAST_FRAME_HOLD_MS]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    spacing = float(np.median(np.linalg.norm(np.diff(gt_centers, axis=0), axis=1))) if len(gt_centers) > 1 else 1.0
    median_error = float(np.median(errors)) if errors else float("nan")
    return {
        "output": str(output),
        "frames": len(frames),
        "localized": len(errors),
        "median_error": median_error,
        "median_error_vs_spacing": median_error / spacing,
        "size_kb": output.stat().st_size / 1024.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Live mapping session workdir (with rounds/)")
    parser.add_argument("--round", type=int, default=None, help="Round to use (default: last successful)")
    parser.add_argument("--output", default=str(REPO / "docs/images/robotics/robotics-loop.gif"))
    parser.add_argument("--width", type=int, default=830, help="GIF width in pixels")
    parser.add_argument("--steps-between", type=int, default=1, help="Interpolated poses between keyframes")
    parser.add_argument("--refine-iters", type=int, default=40, help="Localizer refinement iterations per scale")
    parser.add_argument(
        "--max-seed-distance", type=float, default=0.5, help="Reject estimates whose retrieval seed is this far"
    )
    return parser.parse_args()


def main() -> int:
    summary = build_gif(_parse_args())
    print(
        f"wrote {summary['output']} ({summary['frames']} frames, {summary['size_kb']:.0f} KB, "
        f"median localization error {summary['median_error']:.4f} gauge units "
        f"= {summary['median_error_vs_spacing']:.2f} keyframe spacings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
