#!/usr/bin/env python3
"""Build a README GIF that proves map quality from the actual .splat data."""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
ASSET_DIR = DOCS / "assets" / "outdoor-demo"
OUTPUT = DOCS / "images" / "demo-sweep" / "map-quality.gif"
FRAME_SIZE = (960, 540)
SPLAT_RECORD_BYTES = 32
FRAMES_PER_SCENE = 5
FRAME_DURATION_MS = 170
MAX_POINTS_PER_SCENE = 170_000


@dataclass(frozen=True, slots=True)
class MapProofScene:
    asset: str
    label: str
    axes: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SplatPoint:
    xyz: tuple[float, float, float]
    rgba: tuple[int, int, int, int]


MAP_PROOF_SCENES = (
    MapProofScene(
        asset="bag6-pi3x-20-15k.splat",
        label="bag6 Pi3X external-SLAM",
        axes=(0, 2),
    ),
    MapProofScene(
        asset="mcd-ntu-day02-supervised.splat",
        label="MCD ntu_day_02 supervised",
        axes=(0, 1),
    ),
    MapProofScene(
        asset="outdoor-demo.splat",
        label="Autoware fused supervised",
        axes=(1, 0),
    ),
    MapProofScene(
        asset="bag6-mast3r.splat",
        label="bag6 MASt3R pose-free",
        axes=(0, 2),
    ),
)


def build_map_quality_gif(
    output: Path = OUTPUT,
    *,
    asset_dir: Path = ASSET_DIR,
    size: tuple[int, int] = FRAME_SIZE,
    frames_per_scene: int = FRAMES_PER_SCENE,
) -> Path:
    """Render a map-inspection GIF from the shipped .splat binaries."""

    frames: list[Image.Image] = []
    for scene_index, scene in enumerate(MAP_PROOF_SCENES, start=1):
        path = asset_dir / scene.asset
        points = _read_splat_points(path, max_points=MAX_POINTS_PER_SCENE)
        bounds = _projection_bounds(points, scene.axes, size)
        for frame_index in range(frames_per_scene):
            progress = frame_index / max(1, frames_per_scene - 1)
            zoom = 1.0 + 0.23 * _ease_in_out(progress)
            frame = _render_scene_frame(
                points,
                scene=scene,
                scene_index=scene_index,
                scene_count=len(MAP_PROOF_SCENES),
                axes=scene.axes,
                bounds=_zoom_bounds(bounds, zoom),
                size=size,
                gaussian_count=_gaussian_count(path),
            )
            frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output


def _read_splat_points(path: Path, *, max_points: int) -> list[SplatPoint]:
    data = path.read_bytes()
    count = len(data) // SPLAT_RECORD_BYTES
    stride = max(1, count // max_points)
    points: list[SplatPoint] = []
    for index in range(0, count, stride):
        offset = index * SPLAT_RECORD_BYTES
        xyz = struct.unpack_from("<fff", data, offset)
        rgba = struct.unpack_from("<BBBB", data, offset + 24)
        if rgba[3] < 18 or max(rgba[:3]) < 12:
            continue
        if not all(math.isfinite(value) for value in xyz):
            continue
        points.append(SplatPoint(xyz=xyz, rgba=rgba))
    if not points:
        raise ValueError(f"{path} did not yield any renderable splat points")
    return points


def _gaussian_count(path: Path) -> int:
    return path.stat().st_size // SPLAT_RECORD_BYTES


def _projection_bounds(
    points: list[SplatPoint],
    axes: tuple[int, int],
    size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x_values = sorted(point.xyz[axes[0]] for point in points)
    y_values = sorted(point.xyz[axes[1]] for point in points)
    left, right = _robust_span(x_values)
    bottom, top = _robust_span(y_values)
    width, height = size
    span_x = right - left
    span_y = top - bottom
    target_aspect = width / height
    if span_x / span_y < target_aspect:
        extra = (span_y * target_aspect - span_x) / 2.0
        left -= extra
        right += extra
    else:
        extra = (span_x / target_aspect - span_y) / 2.0
        bottom -= extra
        top += extra
    return (left, right, bottom, top)


def _robust_span(values: list[float]) -> tuple[float, float]:
    count = len(values)
    low = values[int(count * 0.01)]
    high = values[min(count - 1, int(count * 0.99))]
    if high <= low:
        low = values[0]
        high = values[-1]
    padding = max(0.1, (high - low) * 0.08)
    return low - padding, high + padding


def _zoom_bounds(bounds: tuple[float, float, float, float], zoom: float) -> tuple[float, float, float, float]:
    left, right, bottom, top = bounds
    center_x = (left + right) / 2.0
    center_y = (bottom + top) / 2.0
    half_width = (right - left) / (2.0 * zoom)
    half_height = (top - bottom) / (2.0 * zoom)
    return (center_x - half_width, center_x + half_width, center_y - half_height, center_y + half_height)


def _render_scene_frame(
    points: list[SplatPoint],
    *,
    scene: MapProofScene,
    scene_index: int,
    scene_count: int,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
    gaussian_count: int,
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (5, 8, 11))
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_grid(draw, bounds, size)
    pixels = image.load()
    left, right, bottom, top = bounds
    for point in points:
        px = (point.xyz[axes[0]] - left) / (right - left)
        py = (point.xyz[axes[1]] - bottom) / (top - bottom)
        x = int(px * (width - 1))
        y = int((1.0 - py) * (height - 1))
        if not (0 <= x < width and 0 <= y < height):
            continue
        color = _point_color(point.rgba)
        _max_blend_point(pixels, x, y, color, size)
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = ImageEnhance.Sharpness(image).enhance(1.05)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    _draw_frame_label(
        overlay_draw,
        scene=scene,
        scene_index=scene_index,
        scene_count=scene_count,
        gaussian_count=gaussian_count,
        sampled_count=len(points),
        size=size,
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _point_color(rgba: tuple[int, int, int, int]) -> tuple[int, int, int]:
    red, green, blue, alpha = rgba
    gain = 0.46 + 0.58 * (alpha / 255.0)
    return (
        int(min(255, red * gain + 24)),
        int(min(255, green * gain + 24)),
        int(min(255, blue * gain + 24)),
    )


def _max_blend_point(
    pixels: Image.PixelAccess,
    x: int,
    y: int,
    color: tuple[int, int, int],
    size: tuple[int, int],
) -> None:
    width, height = size
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx = x + dx
        yy = y + dy
        if not (0 <= xx < width and 0 <= yy < height):
            continue
        old = pixels[xx, yy]
        pixels[xx, yy] = (
            max(old[0], color[0]),
            max(old[1], color[1]),
            max(old[2], color[2]),
        )


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
) -> None:
    width, height = size
    left, right, bottom, top = bounds
    for index in range(1, 6):
        x = int(width * index / 6)
        y = int(height * index / 6)
        draw.line((x, 0, x, height), fill=(255, 255, 255, 14), width=1)
        draw.line((0, y, width, y), fill=(255, 255, 255, 14), width=1)
    label_font = _load_font(14)
    draw.text(
        (18, height - 30),
        f"orthographic .splat map view  x=[{left:.1f},{right:.1f}]",
        font=label_font,
        fill=(160, 176, 190, 180),
    )
    draw.text(
        (width - 176, height - 30),
        f"y=[{bottom:.1f},{top:.1f}]",
        font=label_font,
        fill=(160, 176, 190, 180),
    )


def _draw_frame_label(
    draw: ImageDraw.ImageDraw,
    *,
    scene: MapProofScene,
    scene_index: int,
    scene_count: int,
    gaussian_count: int,
    sampled_count: int,
    size: tuple[int, int],
) -> None:
    width, _ = size
    title_font = _load_font(28)
    meta_font = _load_font(17)
    chip_font = _load_font(15)
    draw.rounded_rectangle((18, 16, 650, 91), radius=6, fill=(4, 9, 15, 205), outline=(255, 255, 255, 36))
    draw.text((34, 29), scene.label, font=title_font, fill=(244, 249, 255, 255))
    draw.text(
        (36, 65),
        f"actual .splat XYZ render / {gaussian_count // 1000}k gaussians / {sampled_count // 1000}k sampled",
        font=meta_font,
        fill=(203, 218, 232, 240),
    )
    draw.rounded_rectangle(
        (width - 166, 20, width - 24, 54), radius=5, fill=(4, 9, 15, 205), outline=(91, 232, 120, 70)
    )
    draw.text((width - 148, 29), f"MAP {scene_index:02d}/{scene_count:02d}", font=chip_font, fill=(91, 232, 120, 255))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        candidate = Path(path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _ease_in_out(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = build_map_quality_gif(output=args.output, asset_dir=args.asset_dir)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
