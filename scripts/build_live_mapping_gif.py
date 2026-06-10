#!/usr/bin/env python3
"""Compose the "map grows as the robot drives" GIF from a live-mapping session.

Input is the workdir produced by ``scripts/run_live_mapping_demo.py`` (or the
ROS 2 node): every successful rebuild round leaves ``rounds/round_NNN/`` with
the trained gaussians (``train/point_cloud.ply``, full precision, same gauge as
the round's COLMAP poses) and the poses themselves. Each round is a full
pose-free rebuild, so rounds live in different gauges; this script aligns every
round onto the last one by chaining per-pair similarity transforms (rotation
from shared cameras' orientations, so two shared keyframes suffice), then
renders the growing map as a fixed top-down orthographic gsplat view — the
mapped street reads like a map strip that extends as the robot drives — and
overlays the driving frame, the trajectory so far, and a round/keyframe HUD.

    python3 scripts/build_live_mapping_gif.py \
        --session outputs/live_demo_kitti0056/session \
        --output docs/images/live-mapping/live-mapping-grow.gif
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gs_sim2real.robotics.gauge_alignment import (  # noqa: E402
    MIN_SHARED_CAMERAS,
    compose,
    invert,
    parse_images_txt,
    quat_multiply,
    quat_to_rotation,
    read_gauge_transform,
    rotation_to_quat,
    similarity_from_poses,
)

FRAME_SIZE = (960, 420)
FRAME_DURATION_MS = 900
LAST_FRAME_HOLD_MS = 2600
SCALE_PERCENTILE = 98.0
MIN_OPACITY = 0.06

INK = (236, 240, 246)
ACCENT = (96, 205, 255)
TRAIL = (80, 190, 255)
CAMERA_DOT = (255, 214, 90)
PANEL_BG = (12, 16, 24)


# ----------------------------------------------------------------- session parsing


@dataclass
class RoundData:
    index: int
    ply_path: Path
    names: list[str]
    centers: np.ndarray  # (N, 3) camera centers, this round's gauge
    rotations: np.ndarray  # (N, 3, 3) world-from-camera


# Shared gauge-alignment math lives in gs_sim2real.robotics.gauge_alignment;
# these aliases keep this script's historical local names working.
_quat_to_rot = quat_to_rotation
_parse_images_txt = parse_images_txt
_rot_to_quat = rotation_to_quat
_quat_multiply = quat_multiply
_compose = compose


def load_rounds(session: Path) -> list[RoundData]:
    rounds: list[RoundData] = []
    for round_dir in sorted(session.glob("rounds/round_*")):
        ply = round_dir / "train" / "point_cloud.ply"
        images_txt = round_dir / "sparse_input" / "sparse" / "0" / "images.txt"
        if not ply.is_file() or not images_txt.is_file() or not (round_dir / "scene.splat").is_file():
            continue  # scene.splat marks the round as successfully published
        match = re.search(r"round_(\d+)", round_dir.name)
        names, centers, rotations = _parse_images_txt(images_txt)
        if len(names) < 2:
            continue
        rounds.append(RoundData(int(match.group(1)), ply, names, centers, rotations))
    return rounds


# ----------------------------------------------------------------- alignment


def load_runtime_transforms(rounds: list[RoundData]) -> list[tuple[float, np.ndarray, np.ndarray]] | None:
    """Fast path: reuse the per-round session-gauge transforms the live runtime persisted.

    Returns transforms re-anchored onto the last round (matching
    :func:`align_to_anchor`'s convention), or ``None`` when any round lacks
    ``gauge_transform.json`` or the session gauge was rebased mid-run.
    """
    to_session: list[tuple[float, np.ndarray, np.ndarray]] = []
    for position, rnd in enumerate(rounds):
        entry = read_gauge_transform(rnd.ply_path.parent.parent)
        if entry is None:
            return None
        transform, rebased = entry
        if rebased and position > 0:
            return None  # chain restarted mid-session; recompute from poses instead
        to_session.append(transform)
    session_from_last_inv = invert(to_session[-1])
    return [_compose(session_from_last_inv, transform) for transform in to_session]


def align_to_anchor(rounds: list[RoundData]) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Per-round similarity mapping that round's gauge onto the last round's.

    Consecutive rounds share far more keyframes than distant ones (each rebuild
    re-strides the growing keyframe list), so align each round to the next and
    compose the chain up to the final round.
    """
    identity = (1.0, np.eye(3), np.zeros(3))
    transforms = [identity]
    for nxt, rnd in zip(rounds[::-1], rounds[-2::-1]):
        nxt_index = {name: i for i, name in enumerate(nxt.names)}
        shared = [(i, nxt_index[name]) for i, name in enumerate(rnd.names) if name in nxt_index]
        if len(shared) < MIN_SHARED_CAMERAS:
            raise SystemExit(f"rounds {rnd.index}->{nxt.index}: only {len(shared)} shared cameras; cannot align")
        src_ids = [i for i, _ in shared]
        dst_ids = [j for _, j in shared]
        step = similarity_from_poses(
            rnd.centers[src_ids], rnd.rotations[src_ids], nxt.centers[dst_ids], nxt.rotations[dst_ids]
        )
        transforms.append(_compose(transforms[-1], step))
    return transforms[::-1]


# ----------------------------------------------------------------- rendering


def load_ply_aligned(path: Path, transform: tuple[float, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Trained gaussians (full precision), mapped into the anchor round's gauge."""
    from gs_sim2real.viewer.web_viewer import load_ply

    scale_f, rotation, translation = transform
    data = load_ply(str(path))
    pos = np.asarray(data.positions, np.float64) @ rotation.T * scale_f + translation
    scales = np.exp(np.asarray(data.scales, np.float64)) * scale_f
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(data.opacities, np.float64).reshape(-1)))
    colors = np.clip(np.asarray(data.colors, np.float64), 0.0, 1.0)
    quats = np.asarray(data.rotations, np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
    quats = _quat_multiply(np.broadcast_to(_rot_to_quat(rotation), quats.shape), quats)
    scale_max = scales.max(axis=1)
    keep = (scale_max <= np.percentile(scale_max, SCALE_PERCENTILE)) & (opacities >= MIN_OPACITY)
    return {
        "pos": pos[keep].astype(np.float32),
        "scale": scales[keep].astype(np.float32),
        "quat": quats[keep].astype(np.float32),
        "opacity": opacities[keep].astype(np.float32),
        "color": colors[keep].astype(np.float32),
        "total": int(len(pos)),
    }


@dataclass
class ViewCamera:
    viewmat: np.ndarray  # (4, 4) world-to-camera
    intrinsics: np.ndarray  # (3, 3)
    width: int
    height: int

    def project(self, points: np.ndarray) -> np.ndarray:
        """Orthographic projection (world -> pixel), depth kept for culling."""
        cam = points @ self.viewmat[:3, :3].T + self.viewmat[:3, 3]
        px = cam[:, :2] * [self.intrinsics[0, 0], self.intrinsics[1, 1]]
        px += [self.intrinsics[0, 2], self.intrinsics[1, 2]]
        return np.concatenate([px, cam[:, 2:3]], axis=1)


def fit_view_camera(anchor: RoundData, anchor_splat: dict[str, np.ndarray], size: tuple[int, int]) -> ViewCamera:
    """Fixed top-down orthographic camera framing the final round's whole map.

    Image +x follows the drive direction so the mapped street reads as a
    horizontal strip that extends to the right as rounds complete.
    """
    centers = anchor.centers
    up = -anchor.rotations[:, :, 1].mean(axis=0)  # camera +y looks down in OpenCV convention
    up /= np.linalg.norm(up).clip(1e-8)
    flat = centers - np.outer(centers @ up, up)
    _u, _s, vt = np.linalg.svd(flat - flat.mean(axis=0))  # dominant drive axis, not just endpoints
    forward = vt[0]
    if forward @ (centers[-1] - centers[0]) < 0:
        forward = -forward
    forward -= up * (forward @ up)
    forward /= np.linalg.norm(forward).clip(1e-8)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right).clip(1e-8)

    rotation = np.stack([forward, -right, -up])  # world-to-camera rows, looking straight down
    # frame on the dense part of the final map (ignore far-flung floaters)
    in_cam = anchor_splat["pos"].astype(np.float64) @ rotation.T
    lo = np.percentile(in_cam[:, :2], 2.0, axis=0)
    hi = np.percentile(in_cam[:, :2], 98.0, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    mid = (lo + hi) / 2.0

    width, height = size
    focal = min(width / (span[0] * 1.08), height / (span[1] * 1.25))
    viewmat = np.eye(4)
    viewmat[:3, :3] = rotation
    # center the dense map in camera xy; +10 keeps every gaussian at positive depth
    viewmat[:3, 3] = np.array([-mid[0], -mid[1], 10.0])

    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    return ViewCamera(viewmat, intrinsics, width, height)


def render_splat(splat: dict[str, np.ndarray], camera: ViewCamera) -> Image.Image:
    import torch
    from gsplat import rasterization

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image, _alpha, _meta = rasterization(
        torch.from_numpy(splat["pos"]).to(device),
        torch.from_numpy(splat["quat"]).to(device),
        torch.from_numpy(splat["scale"]).to(device),
        torch.from_numpy(splat["opacity"]).to(device),
        torch.from_numpy(splat["color"]).to(device),
        torch.from_numpy(camera.viewmat.astype(np.float32)).to(device)[None],
        torch.from_numpy(camera.intrinsics.astype(np.float32)).to(device)[None],
        camera.width,
        camera.height,
        camera_model="ortho",
        near_plane=0.001,
        far_plane=100.0,
    )
    array = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    frame = Image.fromarray(array)
    frame = ImageEnhance.Brightness(frame).enhance(1.22)
    frame = ImageEnhance.Contrast(frame).enhance(1.05)
    frame = ImageEnhance.Color(frame).enhance(1.1)
    return frame


# ----------------------------------------------------------------- composition


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_trajectory(draw: ImageDraw.ImageDraw, camera: ViewCamera, centers: np.ndarray) -> None:
    projected = camera.project(centers)
    visible = projected[:, 2] > 0
    points = [(float(p[0]), float(p[1])) for p, v in zip(projected, visible) if v]
    if len(points) >= 2:
        draw.line(points, fill=TRAIL, width=4, joint="curve")
    if points:
        x, y = points[-1]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=CAMERA_DOT, outline=(30, 30, 30), width=2)


def compose_frame(
    map_render: Image.Image,
    camera: ViewCamera,
    centers_so_far: np.ndarray,
    driving_frame: Image.Image | None,
    *,
    round_index: int,
    total_rounds: int,
    keyframes: int,
    gaussians: int,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    frame = Image.new("RGB", size, PANEL_BG)
    frame.paste(map_render.resize(size, Image.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    _draw_trajectory(draw, camera, centers_so_far)

    # driving-camera inset, top-right
    if driving_frame is not None:
        inset_w = 318
        inset_h = max(1, round(inset_w * driving_frame.height / driving_frame.width))
        inset = driving_frame.resize((inset_w, inset_h), Image.LANCZOS)
        ix, iy = width - inset_w - 14, 14
        draw.rectangle((ix - 3, iy - 3, ix + inset_w + 2, iy + inset_h + 18), fill=(10, 14, 20, 235))
        frame.paste(inset, (ix, iy))
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.text((ix, iy + inset_h + 3), "onboard camera (KITTI)", font=_load_font(13), fill=ACCENT)

    # HUD banner, bottom
    font_big, font_small = _load_font(22), _load_font(15)
    banner_h = 58
    draw.rectangle((0, height - banner_h, width, height), fill=(8, 12, 18, 225))
    draw.text(
        (16, height - banner_h + 8),
        f"live mapping — rebuild round {round_index}/{total_rounds}",
        font=font_big,
        fill=INK,
    )
    draw.text(
        (16, height - banner_h + 36),
        f"{keyframes} keyframes · {gaussians / 1e6:.1f}M gaussians · the map grows as the robot drives",
        font=font_small,
        fill=ACCENT,
    )
    progress = round_index / max(total_rounds, 1)
    draw.rectangle((0, height - 4, int(width * progress), height), fill=ACCENT)
    return frame


def latest_driving_frame(round_dir: Path) -> Image.Image | None:
    staged = sorted((round_dir / "images").glob("*"))
    if not staged:
        return None
    return Image.open(staged[-1]).convert("RGB")


# ----------------------------------------------------------------- main


def build_gif(session: Path, output: Path, size: tuple[int, int]) -> dict:
    rounds = load_rounds(session)
    if len(rounds) < 2:
        raise SystemExit(f"need at least 2 successful rounds under {session}/rounds, found {len(rounds)}")
    transforms = load_runtime_transforms(rounds)
    if transforms is None:
        transforms = align_to_anchor(rounds)  # legacy sessions without runtime gauge transforms
    anchor = rounds[-1]
    anchor_splat = load_ply_aligned(anchor.ply_path, transforms[-1])
    camera = fit_view_camera(anchor, anchor_splat, size)

    frames: list[Image.Image] = []
    for rnd, transform in zip(rounds, transforms):
        splat = anchor_splat if rnd is anchor else load_ply_aligned(rnd.ply_path, transform)
        render = render_splat(splat, camera)
        # trajectory from the anchor's (cleanest) poses, cut at this round's last keyframe
        seen = max(rnd.names)
        centers = anchor.centers[[i for i, name in enumerate(anchor.names) if name <= seen]]
        frames.append(
            compose_frame(
                render,
                camera,
                centers,
                latest_driving_frame(rnd.ply_path.parent.parent),
                round_index=rnd.index,
                total_rounds=anchor.index,
                keyframes=len(rnd.names),
                gaussians=len(splat["pos"]),
                size=size,
            )
        )

    durations = [FRAME_DURATION_MS] * len(frames)
    durations[-1] = LAST_FRAME_HOLD_MS
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return {
        "rounds": len(frames),
        "keyframesFinal": len(anchor.names),
        "bytes": output.stat().st_size,
        "size": list(size),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Live mapping session workdir (with rounds/)")
    parser.add_argument("--output", default=str(REPO / "docs/images/live-mapping/live-mapping-grow.gif"))
    parser.add_argument("--width", type=int, default=FRAME_SIZE[0])
    parser.add_argument("--height", type=int, default=FRAME_SIZE[1])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_gif(Path(args.session), Path(args.output), (args.width, args.height))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    main()
    sys.exit(0)
