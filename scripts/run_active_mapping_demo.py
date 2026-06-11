#!/usr/bin/env python3
"""Replay an image folder into the active-mapping driver."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gs_sim2real.robotics.active_mapping import ActiveMappingConfig, ActiveMappingDriver  # noqa: E402
from gs_sim2real.robotics.gauge_alignment import apply_to_points, read_gauge_transform  # noqa: E402
from gs_sim2real.robotics.live_mapping import LiveMapperConfig, LiveMappingSession  # noqa: E402
from gs_sim2real.viewer.web_viewer import load_ply  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Folder of replay frames in recorded drive order")
    parser.add_argument("--workdir", required=True, help="Live mapping work directory")
    parser.add_argument("--initial-frames", type=int, default=15, help="Frames fed before the bootstrap rebuild")
    parser.add_argument("--batch-frames", type=int, default=8, help="Recorded frames fed per frontier capture")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum active rebuild rounds after bootstrap")
    parser.add_argument("--method", default="vggt", choices=("vggt", "dust3r", "mast3r"), help="Pose-free backend")
    parser.add_argument("--num-frames", type=int, default=24, help="Frames sampled by each reconstruction round")
    parser.add_argument("--iterations", type=int, default=1500, help="gsplat training iterations per round")
    parser.add_argument("--align-iters", type=int, default=300, help="Pose-free alignment iterations")
    parser.add_argument("--scene-graph", default="swin", help="Pose-free scene graph")
    parser.add_argument("--dust3r-root", default=None, help="Local clone of naver/dust3r")
    parser.add_argument("--vggt-root", default=None, help="Local clone of facebookresearch/vggt")
    parser.add_argument("--dust3r-checkpoint", default=None, help="DUSt3R checkpoint path or HF hub id")
    parser.add_argument("--vggt-checkpoint", default=None, help="VGGT checkpoint .pt path or Hugging Face hub id")
    parser.add_argument("--gif", default=None, help="Optional active-mapping top-down GIF output path")
    parser.add_argument("--width", type=int, default=1600, help="GIF frame width")
    return parser


def iter_folder_frames(images_dir: Path):
    import cv2

    frame_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(frame_paths) < 2:
        print(f"Error: need at least 2 frames in {images_dir}", file=sys.stderr)
        sys.exit(2)

    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("skipping unreadable frame %s", frame_path.name)
            continue
        yield image, float(index)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()

    workdir = Path(args.workdir)
    config = LiveMapperConfig(
        workdir=workdir,
        method=args.method,
        num_frames=args.num_frames,
        iterations=args.iterations,
        align_iters=args.align_iters,
        scene_graph=args.scene_graph,
        checkpoint=(
            args.vggt_checkpoint
            if args.method == "vggt" and args.vggt_checkpoint
            else (Path(args.dust3r_checkpoint) if args.dust3r_checkpoint else None)
        ),
        dust3r_root=Path(args.dust3r_root) if args.dust3r_root else None,
        vggt_root=Path(args.vggt_root) if args.vggt_root else None,
    )
    session = LiveMappingSession(config)
    driver = ActiveMappingDriver(
        session,
        iter_folder_frames(Path(args.images)),
        ActiveMappingConfig(batch_frames=args.batch_frames, max_rounds=args.max_rounds),
    )

    bootstrap = driver.bootstrap(args.initial_frames)
    result = driver.run()
    payload = {
        "config": {
            "images": str(Path(args.images)),
            "workdir": str(workdir),
            "initial_frames": args.initial_frames,
            "batch_frames": args.batch_frames,
            "max_rounds": args.max_rounds,
            "method": args.method,
            "num_frames": args.num_frames,
            "iterations": args.iterations,
            "align_iters": args.align_iters,
            "scene_graph": args.scene_graph,
        },
        "bootstrap": bootstrap,
        "result": result,
        "robot_trail": [point.tolist() for point in driver.robot_trail],
    }
    log_path = workdir / "active_mapping_log.json"
    log_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.gif:
        write_active_mapping_gif(workdir, driver, bootstrap, result, Path(args.gif), image_width=args.width)

    print_summary(bootstrap, result)


def print_summary(bootstrap: dict, result: dict) -> None:
    entries = result["entries"]
    print("step  round  driven  fed  accepted  built  new_kf  grew   gaussians")
    for index, entry in enumerate(entries, start=1):
        print(
            f"{index:>4}  {entry['round_index']:>5}  {str(entry['driven']):>6}  "
            f"{entry['frames_fed']:>3}  {entry['frames_accepted']:>8}  {str(entry['built']):>5}  "
            f"{entry['new_keyframes']:>6}  {str(entry['grew']):>5}  {entry['gaussians']:>10}"
        )

    final_gaussians = entries[-1]["gaussians"] if entries else bootstrap["gaussians"]
    print(
        f"Totals: {len(entries)} active rounds, {final_gaussians:,} final gaussians, "
        f"stop_reason={result['stop_reason']}"
    )


def write_active_mapping_gif(
    workdir: Path,
    driver: ActiveMappingDriver,
    bootstrap: dict,
    result: dict,
    output_path: Path,
    *,
    image_width: int,
) -> None:
    items: list[tuple[int, dict | None, int]] = [(int(bootstrap["round_index"]), None, 1)]
    trail_count = 1
    for entry in result["entries"]:
        if entry["driven"]:
            trail_count += 1
        if entry["built"]:
            items.append((int(entry["round_index"]), entry, trail_count))

    if not items:
        return

    final_positions = _load_round_positions(workdir, items[-1][0])
    axes = _top_down_axes(final_positions)
    final_xy = _project(final_positions, axes)
    if len(final_xy) == 0:
        min_xy = np.zeros(2, dtype=np.float64)
        max_xy = np.ones(2, dtype=np.float64)
    else:
        min_xy = np.percentile(final_xy, 0.5, axis=0)
        max_xy = np.percentile(final_xy, 99.5, axis=0)

    frames: list[Image.Image] = []
    durations: list[int] = []
    for round_index, entry, trail_prefix in items:
        positions = _load_round_positions(workdir, round_index)
        caption = f"round {round_index} - {len(positions):,} gaussians"
        if entry is not None and not entry["grew"]:
            caption += " - frontier exhausted"
        frames.append(
            _draw_gif_frame(
                positions,
                axes,
                (min_xy, max_xy),
                driver.robot_trail[:trail_prefix],
                entry,
                caption,
                image_width=image_width,
            )
        )
        durations.append(900)

    durations[-1] = 2500
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def _load_round_positions(workdir: Path, round_index: int) -> np.ndarray:
    round_dir = workdir / "rounds" / f"round_{round_index:03d}"
    ply = load_ply(str(round_dir / "train" / "point_cloud.ply"))
    positions = np.asarray(ply.positions, dtype=np.float64)
    loaded = read_gauge_transform(round_dir)
    transform = loaded[0] if loaded is not None else (1.0, np.eye(3), np.zeros(3))
    return apply_to_points(transform, positions)


def _top_down_axes(positions: np.ndarray) -> tuple[int, int]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        return 0, 1
    drop_axis = int(np.argmin(np.var(positions, axis=0)))
    keep_axes = [axis for axis in range(3) if axis != drop_axis]
    spread = np.var(positions[:, keep_axes], axis=0)
    if spread[1] > spread[0]:
        keep_axes = [keep_axes[1], keep_axes[0]]
    return int(keep_axes[0]), int(keep_axes[1])


def _project(points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return points.reshape((-1, 3))[:, list(axes)]


def _draw_gif_frame(
    positions: np.ndarray,
    axes: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    trail: list[np.ndarray],
    entry: dict | None,
    caption: str,
    *,
    image_width: int,
) -> Image.Image:
    xy = _project(positions, axes)
    min_xy, max_xy = bounds
    span = np.maximum(max_xy - min_xy, 1e-9)
    width = max(int(image_width), 1)
    scale = (width - 1) / span[0]
    height = max(int(np.ceil(span[1] * scale)) + 1, 80)

    def to_pixels(points: np.ndarray) -> list[tuple[int, int]]:
        projected = _project(points, axes)
        if len(projected) == 0:
            return []
        cols = np.clip(((projected[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, width - 1)
        rows = np.clip(((projected[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)
        return [(int(col), int(row)) for row, col in zip(rows, cols)]

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(xy):
        cols = np.clip(((xy[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, width - 1)
        rows = np.clip(((xy[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)
        image[rows, cols] = (95, 125, 160)

    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame)

    trail_points = to_pixels(np.asarray(trail, dtype=np.float64)) if trail else []
    if len(trail_points) >= 2:
        draw.line(trail_points, fill=(30, 150, 80), width=5)
    for point in trail_points:
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(30, 150, 80))

    if entry is not None:
        frontier = np.asarray([entry["frontier_world"]], dtype=np.float64)
        frontier_points = to_pixels(frontier)
        if frontier_points:
            x, y = frontier_points[0]
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(230, 130, 20), width=4)

    draw.rectangle((0, 0, width, 38), fill=(255, 255, 255))
    draw.text((12, 11), caption, fill=(20, 20, 20))
    return frame


if __name__ == "__main__":
    main()
