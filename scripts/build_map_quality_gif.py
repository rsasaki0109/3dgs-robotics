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
FOV_DEGREES = 58.0
MINIMAP_SIZE = (244, 138)


@dataclass(frozen=True, slots=True)
class MapProofScene:
    asset: str
    label: str
    axes: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CameraPose:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]


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
        asset="bag6-mast3r-slam-20-15k.splat",
        label="bag6 MASt3R-SLAM external-SLAM",
        axes=(0, 2),
    ),
    MapProofScene(
        asset="bag6-vggt-slam-20-15k.splat",
        label="bag6 VGGT-SLAM external-SLAM",
        axes=(0, 2),
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
        cameras = _camera_path(points, axes=scene.axes, bounds=bounds, frame_count=frames_per_scene)
        for camera in cameras:
            frame = _render_scene_frame(
                points,
                scene=scene,
                scene_index=scene_index,
                scene_count=len(MAP_PROOF_SCENES),
                axes=scene.axes,
                bounds=bounds,
                camera=camera,
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
    stride = max(1, math.ceil(count / max_points))
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


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile from an empty list")
    index = int((len(sorted_values) - 1) * max(0.0, min(1.0, fraction)))
    return sorted_values[index]


def _vertical_axis(axes: tuple[int, int]) -> int:
    return ({0, 1, 2} - set(axes)).pop()


def _plane_median(points: list[SplatPoint], axes: tuple[int, int]) -> tuple[float, float]:
    first = sorted(point.xyz[axes[0]] for point in points)
    second = sorted(point.xyz[axes[1]] for point in points)
    return (_percentile(first, 0.50), _percentile(second, 0.50))


def _principal_direction(
    points: list[SplatPoint],
    *,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    left, right, bottom, top = bounds
    step = max(1, len(points) // 50_000)
    values: list[tuple[float, float]] = []
    for point in points[::step]:
        x = point.xyz[axes[0]]
        y = point.xyz[axes[1]]
        if left <= x <= right and bottom <= y <= top:
            values.append((x, y))
    if len(values) < 2:
        return (1.0, 0.0)
    mean_x = sum(value[0] for value in values) / len(values)
    mean_y = sum(value[1] for value in values) / len(values)
    var_x = sum((value[0] - mean_x) ** 2 for value in values)
    var_y = sum((value[1] - mean_y) ** 2 for value in values)
    cov_xy = sum((value[0] - mean_x) * (value[1] - mean_y) for value in values)
    if var_x + var_y <= 1e-9:
        return (1.0, 0.0)
    angle = 0.5 * math.atan2(2.0 * cov_xy, var_x - var_y)
    return _normalize_2d((math.cos(angle), math.sin(angle)))


def _plane_projection(
    xyz: tuple[float, float, float],
    axes: tuple[int, int],
    center: tuple[float, float],
    direction: tuple[float, float],
) -> float:
    return (xyz[axes[0]] - center[0]) * direction[0] + (xyz[axes[1]] - center[1]) * direction[1]


def _compose_xyz(
    axes: tuple[int, int],
    vertical_axis: int,
    plane: tuple[float, float],
    height: float,
) -> tuple[float, float, float]:
    values = [0.0, 0.0, 0.0]
    values[axes[0]] = plane[0]
    values[axes[1]] = plane[1]
    values[vertical_axis] = height
    return (values[0], values[1], values[2])


def _camera_path(
    points: list[SplatPoint],
    *,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
    frame_count: int,
) -> list[CameraPose]:
    vertical_axis = _vertical_axis(axes)
    direction = _principal_direction(points, axes=axes, bounds=bounds)
    side = (-direction[1], direction[0])
    center = _plane_median(points, axes)
    projections = sorted(_plane_projection(point.xyz, axes, center, direction) for point in points)
    low = _percentile(projections, 0.08)
    high = _percentile(projections, 0.48)
    target_low = _percentile(projections, 0.42)
    target_high = _percentile(projections, 0.72)
    travel = max(0.3, high - low)
    target_travel = max(0.3, target_high - target_low)
    side_span = max(0.2, min(bounds[1] - bounds[0], bounds[3] - bounds[2]))

    heights = sorted(point.xyz[vertical_axis] for point in points)
    height_low = _percentile(heights, 0.20)
    height_mid = _percentile(heights, 0.54)
    height_high = _percentile(heights, 0.92)
    eye_height = height_mid + max(0.16, (height_high - height_low) * 0.24)
    target_height = height_mid + max(0.02, (height_high - height_low) * 0.04)

    cameras: list[CameraPose] = []
    for index in range(frame_count):
        progress = index / max(1, frame_count - 1)
        eased = _ease_in_out(progress)
        current = low + travel * eased
        lookahead = max(current + 0.25, target_low + target_travel * eased)
        lateral = math.sin(progress * math.pi) * side_span * 0.06
        eye_plane = (
            center[0] + direction[0] * current + side[0] * lateral,
            center[1] + direction[1] * current + side[1] * lateral,
        )
        target_plane = (
            center[0] + direction[0] * lookahead + side[0] * lateral * 0.35,
            center[1] + direction[1] * lookahead + side[1] * lateral * 0.35,
        )
        cameras.append(
            CameraPose(
                eye=_compose_xyz(axes, vertical_axis, eye_plane, eye_height),
                target=_compose_xyz(axes, vertical_axis, target_plane, target_height),
            )
        )
    return cameras


def _camera_basis(
    camera: CameraPose,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    forward = _normalize_3d(_sub(camera.target, camera.eye))
    world_up = (0.0, 1.0, 0.0)
    right = _normalize_3d(_cross(forward, world_up))
    if _dot(right, right) <= 1e-9:
        right = (1.0, 0.0, 0.0)
    up = _normalize_3d(_cross(right, forward))
    return right, up, forward


def _camera_far_distance(bounds: tuple[float, float, float, float]) -> float:
    left, right, bottom, top = bounds
    return max(2.0, math.hypot(right - left, top - bottom) * 1.25)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize_2d(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        return (1.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _normalize_3d(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-9:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _render_scene_frame(
    points: list[SplatPoint],
    *,
    scene: MapProofScene,
    scene_index: int,
    scene_count: int,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
    camera: CameraPose,
    size: tuple[int, int],
    gaussian_count: int,
) -> Image.Image:
    width, height = size
    image = _viewer_background(size)
    pixels = image.load()
    z_buffer = [math.inf] * (width * height)
    right, up, forward = _camera_basis(camera)
    focal = width / (2.0 * math.tan(math.radians(FOV_DEGREES) / 2.0))
    far = _camera_far_distance(bounds)
    rendered = 0
    for point in points:
        rel = _sub(point.xyz, camera.eye)
        depth = _dot(rel, forward)
        if depth < 0.08 or depth > far:
            continue
        camera_x = _dot(rel, right)
        camera_y = _dot(rel, up)
        x = int(width * 0.5 + (camera_x / depth) * focal)
        y = int(height * 0.56 - (camera_y / depth) * focal)
        if not (0 <= x < width and 0 <= y < height):
            continue
        color = _perspective_color(point.rgba, depth=depth, far=far)
        radius = 2 if depth < far * 0.18 else 1
        _depth_blend_point(pixels, z_buffer, x, y, depth, color, size, radius=radius)
        rendered += 1
    image = ImageEnhance.Contrast(image).enhance(1.10)
    image = ImageEnhance.Sharpness(image).enhance(1.05)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    _draw_viewer_grid(overlay_draw, size)
    _draw_minimap(overlay_draw, points, axes=axes, bounds=bounds, camera=camera, size=size)
    _draw_frame_label(
        overlay_draw,
        scene=scene,
        scene_index=scene_index,
        scene_count=scene_count,
        gaussian_count=gaussian_count,
        sampled_count=len(points),
        rendered_count=rendered,
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


def _perspective_color(
    rgba: tuple[int, int, int, int],
    *,
    depth: float,
    far: float,
) -> tuple[int, int, int]:
    red, green, blue, alpha = rgba
    fog = max(0.18, min(1.0, 1.0 - depth / far))
    alpha_gain = 0.56 + 0.48 * (alpha / 255.0)
    gain = alpha_gain * (0.55 + 0.65 * fog)
    return (
        int(min(255, red * gain + 16 + 26 * fog)),
        int(min(255, green * gain + 18 + 28 * fog)),
        int(min(255, blue * gain + 20 + 30 * fog)),
    )


def _depth_blend_point(
    pixels: Image.PixelAccess,
    z_buffer: list[float],
    x: int,
    y: int,
    depth: float,
    color: tuple[int, int, int],
    size: tuple[int, int],
    *,
    radius: int,
) -> None:
    width, height = size
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xx = x + dx
            yy = y + dy
            if not (0 <= xx < width and 0 <= yy < height):
                continue
            index = yy * width + xx
            if depth >= z_buffer[index]:
                continue
            z_buffer[index] = depth
            old = pixels[xx, yy]
            pixels[xx, yy] = (
                max(old[0], color[0]),
                max(old[1], color[1]),
                max(old[2], color[2]),
            )


def _viewer_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (6, 9, 13))
    pixels = image.load()
    for y in range(height):
        t = y / max(1, height - 1)
        if y < height * 0.55:
            sky = int(17 + 20 * (1.0 - t))
            color = (8, 13 + sky // 4, 18 + sky)
        else:
            floor = int(34 * (t - 0.55) / 0.45)
            color = (7 + floor // 5, 10 + floor // 4, 13 + floor // 3)
        for x in range(width):
            pixels[x, y] = color
    return image


def _draw_viewer_grid(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    horizon = int(height * 0.56)
    draw.line((0, horizon, width, horizon), fill=(255, 255, 255, 22), width=1)
    for index in range(1, 9):
        y = horizon + int((height - horizon) * (index / 9) ** 1.75)
        draw.line((0, y, width, y), fill=(255, 255, 255, 10), width=1)
    for index in range(-8, 9):
        x = width * 0.5 + index * width * 0.075
        draw.line((width * 0.5, horizon, x, height), fill=(255, 255, 255, 8), width=1)


def _draw_minimap(
    draw: ImageDraw.ImageDraw,
    points: list[SplatPoint],
    *,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
    camera: CameraPose,
    size: tuple[int, int],
) -> None:
    width, height = size
    left, right, bottom, top = bounds
    mini_width, mini_height = MINIMAP_SIZE
    x0 = width - mini_width - 24
    y0 = height - mini_height - 24
    x1 = x0 + mini_width
    y1 = y0 + mini_height
    draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=(4, 9, 15, 205), outline=(255, 255, 255, 36))
    for index in range(1, 4):
        x = x0 + mini_width * index / 4
        y = y0 + mini_height * index / 4
        draw.line((x, y0, x, y1), fill=(255, 255, 255, 16), width=1)
        draw.line((x0, y, x1, y), fill=(255, 255, 255, 16), width=1)
    step = max(1, len(points) // 4500)
    for point in points[::step]:
        x, y = _minimap_xy(point.xyz, axes=axes, bounds=bounds, box=(x0, y0, x1, y1))
        draw.point((x, y), fill=(*_point_color(point.rgba), 96))
    eye_x, eye_y = _minimap_xy(camera.eye, axes=axes, bounds=bounds, box=(x0, y0, x1, y1))
    target_x, target_y = _minimap_xy(camera.target, axes=axes, bounds=bounds, box=(x0, y0, x1, y1))
    draw.line((eye_x, eye_y, target_x, target_y), fill=(88, 236, 130, 255), width=3)
    draw.ellipse((eye_x - 5, eye_y - 5, eye_x + 5, eye_y + 5), fill=(88, 236, 130, 255))
    label_font = _load_font(13)
    draw.text((x0 + 10, y0 + 8), "top-down trace + camera", font=label_font, fill=(213, 230, 244, 230))
    draw.text(
        (x0 + 10, y1 - 24),
        f"x=[{left:.1f},{right:.1f}]  y=[{bottom:.1f},{top:.1f}]",
        font=label_font,
        fill=(160, 176, 190, 190),
    )


def _minimap_xy(
    xyz: tuple[float, float, float],
    *,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, right, bottom, top = bounds
    x0, y0, x1, y1 = box
    x_ratio = (xyz[axes[0]] - left) / (right - left)
    y_ratio = (xyz[axes[1]] - bottom) / (top - bottom)
    x = int(x0 + max(0.0, min(1.0, x_ratio)) * (x1 - x0))
    y = int(y1 - max(0.0, min(1.0, y_ratio)) * (y1 - y0))
    return x, y


def _draw_frame_label(
    draw: ImageDraw.ImageDraw,
    *,
    scene: MapProofScene,
    scene_index: int,
    scene_count: int,
    gaussian_count: int,
    sampled_count: int,
    rendered_count: int,
    size: tuple[int, int],
) -> None:
    width, _ = size
    title_font = _load_font(28)
    meta_font = _load_font(17)
    chip_font = _load_font(15)
    draw.rounded_rectangle((18, 16, 718, 92), radius=6, fill=(4, 9, 15, 205), outline=(255, 255, 255, 36))
    draw.text((34, 29), scene.label, font=title_font, fill=(244, 249, 255, 255))
    draw.text(
        (36, 65),
        (
            "actual .splat FPS render / "
            f"{gaussian_count // 1000}k gaussians / {sampled_count // 1000}k sampled / {rendered_count // 1000}k visible"
        ),
        font=meta_font,
        fill=(203, 218, 232, 240),
    )
    draw.rounded_rectangle(
        (width - 176, 20, width - 24, 54), radius=5, fill=(4, 9, 15, 205), outline=(91, 232, 120, 70)
    )
    draw.text(
        (width - 158, 29),
        f"INSIDE {scene_index:02d}/{scene_count:02d}",
        font=chip_font,
        fill=(91, 232, 120, 255),
    )


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
