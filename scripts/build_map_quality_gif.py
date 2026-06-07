#!/usr/bin/env python3
"""Build a README GIF that proves map quality from the actual .splat data."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
ASSET_DIR = DOCS / "assets" / "outdoor-demo"
OUTPUT = DOCS / "images" / "demo-sweep" / "map-quality.gif"
MAP_MATERIAL_OUTPUT = DOCS / "images" / "demo-sweep" / "dynamic-map-material.png"
FRAME_SIZE = (960, 540)
MAP_MATERIAL_SIZE = (1280, 720)
SPLAT_RECORD_BYTES = 32
FRAMES_PER_SCENE = 12
FRAME_DURATION_MS = 260
MAX_POINTS_PER_SCENE = 170_000
FOV_DEGREES = 64.0
MAP_PANEL_SIZE = (424, 468)
MAP_TILE_COLUMNS = 8
MAP_TILE_ROWS = 4


@dataclass(frozen=True, slots=True)
class MapProofScene:
    asset: str
    label: str
    axes: tuple[int, int]
    catalog: str | None = None


@dataclass(frozen=True, slots=True)
class CameraPose:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SplatPoint:
    xyz: tuple[float, float, float]
    rgba: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RouteMapProjection:
    axes: tuple[int, int]
    center: tuple[float, float]
    direction: tuple[float, float]
    side: tuple[float, float]
    progress_bounds: tuple[float, float]
    lateral_bounds: tuple[float, float]


MAP_PROOF_SCENES = (
    MapProofScene(
        asset="outdoor-production-grid-large-tile-catalog.json",
        label="34-tile outdoor production grid",
        axes=(0, 2),
        catalog="apps/dreamwalker-web/public/manifests/outdoor-production-grid-large-tile-catalog.json",
    ),
)


def build_map_quality_gif(
    output: Path = OUTPUT,
    *,
    asset_dir: Path = ASSET_DIR,
    size: tuple[int, int] = FRAME_SIZE,
    frames_per_scene: int = FRAMES_PER_SCENE,
    map_material_output: Path | None = MAP_MATERIAL_OUTPUT,
) -> Path:
    """Render a map-inspection GIF from the shipped .splat binaries."""

    frames: list[Image.Image] = []
    for scene_index, scene in enumerate(MAP_PROOF_SCENES, start=1):
        if scene.catalog is None:
            path = asset_dir / scene.asset
            points = _read_splat_points(path, max_points=MAX_POINTS_PER_SCENE)
            gaussian_count = _gaussian_count(path)
            source_asset = scene.asset
        else:
            path = REPO / scene.catalog
            points = _read_catalog_splat_points(path, max_points=MAX_POINTS_PER_SCENE)
            gaussian_count = _catalog_gaussian_count(path)
            source_asset = Path(scene.catalog).name
        bounds = _projection_bounds(points, scene.axes, size)
        projection = _route_map_projection(points, axes=scene.axes, bounds=bounds)
        cameras = _camera_path(points, axes=scene.axes, bounds=bounds, frame_count=frames_per_scene)
        if scene_index == 1 and map_material_output is not None:
            material = _render_dynamic_map_material(
                points,
                projection=projection,
                camera=cameras[len(cameras) // 2],
                route=cameras,
                size=MAP_MATERIAL_SIZE,
                bounds=bounds,
                full_material=True,
                source_asset=source_asset,
                gaussian_count=gaussian_count,
            )
            map_material_output.parent.mkdir(parents=True, exist_ok=True)
            material.convert("RGB").save(map_material_output)
        for frame_index, camera in enumerate(cameras, start=1):
            frame = _render_map_loading_frame(
                points,
                scene=scene,
                scene_index=frame_index,
                scene_count=len(cameras),
                bounds=bounds,
                projection=projection,
                camera=camera,
                route=cameras,
                size=size,
                gaussian_count=gaussian_count,
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


def _read_catalog_splat_points(catalog_path: Path, *, max_points: int) -> list[SplatPoint]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tiles = [
        tile for tile in catalog.get("tiles", []) if tile.get("splatUrl") and tile.get("status") != "missing-splat"
    ]
    if not tiles:
        raise ValueError(f"{catalog_path} has no ready splat tiles")

    max_points_per_tile = max(1, math.ceil(max_points / len(tiles)))
    points: list[SplatPoint] = []
    for tile in tiles:
        points.extend(_read_splat_points(_catalog_tile_path(catalog_path, tile), max_points=max_points_per_tile))

    if len(points) > max_points:
        stride = max(1, math.ceil(len(points) / max_points))
        points = points[::stride]
    if not points:
        raise ValueError(f"{catalog_path} did not yield any renderable splat points")
    return points


def _catalog_tile_path(catalog_path: Path, tile: dict[str, object]) -> Path:
    public_path = tile.get("publicPath")
    if isinstance(public_path, str) and public_path:
        candidate = Path(public_path)
        return candidate if candidate.is_absolute() else REPO / candidate

    splat_url = tile.get("splatUrl")
    if isinstance(splat_url, str) and splat_url.startswith("/"):
        return REPO / "apps" / "dreamwalker-web" / "public" / splat_url.lstrip("/")
    if isinstance(splat_url, str) and splat_url:
        return catalog_path.parent / splat_url
    raise ValueError(f"{catalog_path} tile is missing splatUrl/publicPath")


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


def _catalog_gaussian_count(catalog_path: Path) -> int:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    summary = catalog.get("summary") if isinstance(catalog.get("summary"), dict) else {}
    count = summary.get("inputSplatCount") or summary.get("tiledSplatCount") or 0
    return int(count)


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
    low = _percentile(projections, 0.05)
    high = _percentile(projections, 0.54)
    target_low = _percentile(projections, 0.18)
    target_high = _percentile(projections, 0.66)
    travel = max(0.3, high - low)
    target_travel = max(0.3, target_high - target_low)
    side_span = max(0.2, min(bounds[1] - bounds[0], bounds[3] - bounds[2]))

    heights = sorted(point.xyz[vertical_axis] for point in points)
    height_low = _percentile(heights, 0.20)
    height_mid = _percentile(heights, 0.54)
    height_high = _percentile(heights, 0.92)
    height_span = height_high - height_low
    eye_height = height_mid + max(0.24, height_span * 0.34)
    target_height = height_mid + max(0.04, height_span * 0.08)

    cameras: list[CameraPose] = []
    for index in range(frame_count):
        progress = index / max(1, frame_count - 1)
        eased = _ease_in_out(progress)
        current = low + travel * eased
        lookahead = max(current + 0.25, target_low + target_travel * eased)
        lateral = math.sin(progress * math.pi) * side_span * 0.035
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
    bounds: tuple[float, float, float, float],
    projection: RouteMapProjection,
    camera: CameraPose,
    route: list[CameraPose],
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
    _draw_minimap(
        overlay,
        overlay_draw,
        points,
        bounds=bounds,
        projection=projection,
        camera=camera,
        route=route,
        size=size,
    )
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


def _render_map_loading_frame(
    points: list[SplatPoint],
    *,
    scene: MapProofScene,
    scene_index: int,
    scene_count: int,
    bounds: tuple[float, float, float, float],
    projection: RouteMapProjection,
    camera: CameraPose,
    route: list[CameraPose],
    size: tuple[int, int],
    gaussian_count: int,
) -> Image.Image:
    frame = _render_dynamic_map_material(
        points,
        projection=projection,
        camera=camera,
        route=route,
        size=size,
        bounds=bounds,
        full_material=True,
        source_asset=scene.asset,
        gaussian_count=gaussian_count,
        labels=False,
    )
    draw = ImageDraw.Draw(frame)
    _draw_hero_progress(draw, scene_index=scene_index, scene_count=scene_count, size=size)
    return frame.convert("RGB")


def _draw_hero_progress(
    draw: ImageDraw.ImageDraw, *, scene_index: int, scene_count: int, size: tuple[int, int]
) -> None:
    width, height = size
    bar_width = 132
    bar_height = 6
    x0 = width - bar_width - 28
    y0 = height - 30
    x1 = x0 + bar_width
    y1 = y0 + bar_height
    progress = max(0.0, min(1.0, scene_index / max(1, scene_count)))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=3, fill=(5, 13, 18, 190))
    draw.rounded_rectangle((x0, y0, x0 + int(bar_width * progress), y1), radius=3, fill=(91, 232, 120, 230))


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
    overlay: Image.Image,
    draw: ImageDraw.ImageDraw,
    points: list[SplatPoint],
    *,
    bounds: tuple[float, float, float, float],
    projection: RouteMapProjection,
    camera: CameraPose,
    route: list[CameraPose],
    size: tuple[int, int],
) -> None:
    width, height = size
    left, right, bottom, top = bounds
    panel_width, panel_height = MAP_PANEL_SIZE
    x0 = width - panel_width - 18
    y0 = 54
    x1 = x0 + panel_width
    y1 = min(height - 18, y0 + panel_height)
    map_top = y0 + 64
    map_bottom = y1 - 66
    map_box = (x0 + 16, map_top, x1 - 16, map_bottom)
    label_font = _load_font(12)
    title_font = _load_font(18)
    metric_font = _load_font(13)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(4, 9, 15, 226), outline=(255, 255, 255, 44))
    draw.text((x0 + 15, y0 + 12), "dynamic map loading material", font=title_font, fill=(244, 249, 255, 255))
    draw.text(
        (x0 + 17, y0 + 38),
        "resident + preload tiles over actual .splat density",
        font=label_font,
        fill=(173, 192, 210, 225),
    )

    map_material = _render_dynamic_map_material(
        points,
        projection=projection,
        camera=camera,
        route=route,
        size=(map_box[2] - map_box[0], map_box[3] - map_box[1]),
        bounds=bounds,
        full_material=False,
    )
    overlay.alpha_composite(map_material, dest=(map_box[0], map_box[1]))

    map_width_m = right - left
    map_height_m = top - bottom
    tile_column, tile_row = _active_tile(camera.eye, projection=projection)
    draw.text(
        (x0 + 15, y1 - 42),
        f"footprint {map_width_m:.1f} x {map_height_m:.1f} m / {len(points) // 1000}k real splats",
        font=metric_font,
        fill=(204, 223, 238, 235),
    )
    draw.text(
        (x0 + 15, y1 - 22),
        f"loaded window follows route / active tile C{tile_column + 1} R{tile_row + 1}",
        font=label_font,
        fill=(150, 171, 190, 205),
    )


def _render_dynamic_map_material(
    points: list[SplatPoint],
    *,
    projection: RouteMapProjection,
    camera: CameraPose,
    route: list[CameraPose],
    size: tuple[int, int],
    bounds: tuple[float, float, float, float],
    full_material: bool,
    source_asset: str | None = None,
    gaussian_count: int | None = None,
    labels: bool = True,
) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (5, 10, 14, 255))
    draw = ImageDraw.Draw(image)
    _draw_material_background(draw, size)

    if full_material and labels:
        map_box = (48, 118, width - 48, height - 96)
        title_font = _load_font(32 if width < 1100 else 34)
        subtitle_font = _load_font(15 if width < 1100 else 17)
        source_name = source_asset or "actual shipped .splat"
        gaussian_text = f" / {gaussian_count // 1000}k gaussians" if gaussian_count is not None else ""
        draw.text((48, 32), "Dynamic map loading", font=title_font, fill=(245, 250, 255, 255))
        draw.text(
            (52, 82),
            f"PCD cells + route overlay over {source_name} footprint{gaussian_text}",
            font=subtitle_font,
            fill=(176, 197, 214, 235),
        )
    else:
        margin = 22 if full_material else 0
        map_box = (margin, margin, width - margin, height - margin)

    draw.rounded_rectangle(
        map_box, radius=8 if full_material else 5, fill=(7, 15, 20, 255), outline=(255, 255, 255, 42)
    )
    _draw_dynamic_tile_layer(
        draw, projection=projection, camera=camera, box=map_box, full_material=full_material, labels=labels
    )
    _draw_density_material(image, points, projection=projection, box=map_box, full_material=full_material)
    _draw_route_overlay_layer(
        draw, projection=projection, route=route, box=map_box, full_material=full_material, labels=labels
    )
    _draw_load_radius(
        draw, projection=projection, camera=camera, box=map_box, full_material=full_material, labels=labels
    )
    _draw_route_material(
        draw, projection=projection, camera=camera, route=route, box=map_box, full_material=full_material, labels=labels
    )
    _draw_material_scale(draw, projection=projection, box=map_box, full_material=full_material, labels=labels)

    if full_material and labels:
        _draw_material_legend(draw, box=map_box)
        _draw_map_loading_status_hud(
            draw, projection=projection, camera=camera, box=map_box, gaussian_count=gaussian_count
        )
        _draw_material_footer(
            draw,
            points,
            projection=projection,
            bounds=bounds,
            box=map_box,
            size=size,
            source_asset=source_asset,
            gaussian_count=gaussian_count,
        )

    return image


def _draw_material_background(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    for y in range(height):
        t = y / max(1, height - 1)
        color = (5 + int(5 * t), 10 + int(7 * t), 15 + int(11 * t), 255)
        draw.line((0, y, width, y), fill=color)


def _draw_dynamic_tile_layer(
    draw: ImageDraw.ImageDraw,
    *,
    projection: RouteMapProjection,
    camera: CameraPose,
    box: tuple[int, int, int, int],
    full_material: bool,
    labels: bool,
) -> None:
    loaded_tiles, preload_tiles, active_tile = _tile_residency_sets(camera, projection=projection)
    tile_font = _load_font(13 if full_material else 9)
    for column in range(MAP_TILE_COLUMNS):
        for row in range(MAP_TILE_ROWS):
            rect = _tile_rect(column, row, box=box)
            fill = (5, 13, 18, 176)
            outline = (80, 147, 190, 55)
            if (column, row) in preload_tiles:
                fill = (224, 162, 54, 58)
                outline = (242, 190, 82, 120)
            if (column, row) in loaded_tiles:
                fill = (34, 132, 112, 76)
                outline = (91, 232, 120, 132)
            if (column, row) == active_tile:
                fill = (91, 232, 120, 112)
                outline = (181, 255, 192, 210)
            draw.rectangle(rect, fill=fill, outline=outline, width=2 if (column, row) == active_tile else 1)

            if labels and (full_material or (row in (0, MAP_TILE_ROWS - 1) and column % 2 == 0)):
                label = f"PCD {column + 1}{row + 1}" if full_material else f"P{column + 1}{row + 1}"
                draw.text((rect[0] + 7, rect[1] + 7), label, font=tile_font, fill=(157, 190, 210, 150))

    loaded_bounds = _tile_group_bounds(loaded_tiles, box=box)
    if loaded_bounds is not None:
        draw.rounded_rectangle(loaded_bounds, radius=8, outline=(91, 232, 120, 230), width=3 if full_material else 2)
        if full_material and labels:
            draw.text(
                (loaded_bounds[0] + 12, loaded_bounds[1] + 10),
                "RESIDENT TILE WINDOW",
                font=_load_font(16),
                fill=(194, 255, 206, 245),
            )

    preload_bounds = _tile_group_bounds(preload_tiles, box=box)
    if full_material and labels and preload_bounds is not None:
        draw.rounded_rectangle(preload_bounds, radius=8, outline=(242, 190, 82, 210), width=2)
        draw.text(
            (preload_bounds[0] + 12, preload_bounds[1] + 10), "PRELOAD", font=_load_font(15), fill=(255, 221, 139, 240)
        )


def _draw_route_overlay_layer(
    draw: ImageDraw.ImageDraw,
    *,
    projection: RouteMapProjection,
    route: list[CameraPose],
    box: tuple[int, int, int, int],
    full_material: bool,
    labels: bool,
) -> None:
    route_xy = _dynamic_route_xy(projection=projection, route=route, box=box)
    if len(route_xy) < 2:
        return

    lane_width = max(18, int((box[3] - box[1]) * (0.075 if full_material else 0.060)))
    left_lane = _offset_polyline(route_xy, lane_width)
    right_lane = _offset_polyline(route_xy, -lane_width)
    shoulder_left = _offset_polyline(route_xy, int(lane_width * 1.55))
    shoulder_right = _offset_polyline(route_xy, int(-lane_width * 1.55))

    draw.line(shoulder_left, fill=(78, 216, 178, 95), width=2 if full_material else 1, joint="curve")
    draw.line(shoulder_right, fill=(78, 216, 178, 95), width=2 if full_material else 1, joint="curve")
    draw.line(left_lane, fill=(230, 244, 236, 210), width=3 if full_material else 2, joint="curve")
    draw.line(right_lane, fill=(230, 244, 236, 210), width=3 if full_material else 2, joint="curve")
    _draw_dashed_polyline(draw, route_xy, fill=(244, 204, 72, 230), width=2 if full_material else 1, dash=16)

    stop_index = min(len(route_xy) - 2, max(1, int(len(route_xy) * 0.72)))
    stop_start, stop_end = _perpendicular_segment(route_xy, stop_index, lane_width * 1.25)
    draw.line((*stop_start, *stop_end), fill=(255, 82, 82, 230), width=4 if full_material else 2)
    if full_material and labels:
        label_x = int((stop_start[0] + stop_end[0]) / 2) + 8
        label_y = int((stop_start[1] + stop_end[1]) / 2) - 24
        draw.text((label_x, label_y), "route / stop marker", font=_load_font(14), fill=(230, 244, 236, 220))


def _draw_load_radius(
    draw: ImageDraw.ImageDraw,
    *,
    projection: RouteMapProjection,
    camera: CameraPose,
    box: tuple[int, int, int, int],
    full_material: bool,
    labels: bool,
) -> None:
    eye_x, eye_y = _route_map_xy(camera.eye, projection=projection, box=box)
    radius = int((box[3] - box[1]) * (0.23 if full_material else 0.18))
    draw.ellipse(
        (eye_x - radius, eye_y - radius, eye_x + radius, eye_y + radius),
        outline=(91, 232, 120, 105),
        width=3 if full_material else 2,
    )
    inner_radius = int(radius * 0.58)
    draw.ellipse(
        (eye_x - inner_radius, eye_y - inner_radius, eye_x + inner_radius, eye_y + inner_radius),
        outline=(96, 178, 255, 80),
        width=2 if full_material else 1,
    )
    if full_material and labels:
        draw.text((eye_x + radius + 8, eye_y - 10), "load radius", font=_load_font(14), fill=(190, 240, 204, 220))


def _draw_density_material(
    image: Image.Image,
    points: list[SplatPoint],
    *,
    projection: RouteMapProjection,
    box: tuple[int, int, int, int],
    full_material: bool,
) -> None:
    density = Image.new("L", image.size, 0)
    density_draw = ImageDraw.Draw(density)
    step = max(1, len(points) // (70_000 if full_material else 24_000))
    radius = 2 if full_material else 1
    for point in points[::step]:
        x, y = _route_map_xy(point.xyz, projection=projection, box=box)
        brightness = int(min(230, max(44, (point.rgba[3] * 0.55) + (max(point.rgba[:3]) * 0.35))))
        density_draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=brightness)

    blur_radius = 1.6 if full_material else 1.0
    density = density.filter(ImageFilter.GaussianBlur(blur_radius))
    density = ImageEnhance.Contrast(density).enhance(2.2 if full_material else 1.8)
    alpha = density.point(lambda value: min(210, int(value * (0.82 if full_material else 0.72))))
    colored = ImageOps.colorize(density, black=(8, 16, 18), white=(232, 246, 238)).convert("RGBA")
    colored.putalpha(alpha)
    image.alpha_composite(colored)

    crisp_draw = ImageDraw.Draw(image)
    crisp_step = max(1, len(points) // (20_000 if full_material else 8_500))
    for point in points[::crisp_step]:
        x, y = _route_map_xy(point.xyz, projection=projection, box=box)
        color = _point_color(point.rgba)
        fill = (min(255, color[0] + 24), min(255, color[1] + 26), min(255, color[2] + 30), 120)
        crisp_draw.point((x, y), fill=fill)


def _draw_route_material(
    draw: ImageDraw.ImageDraw,
    *,
    projection: RouteMapProjection,
    camera: CameraPose,
    route: list[CameraPose],
    box: tuple[int, int, int, int],
    full_material: bool,
    labels: bool,
) -> None:
    route_xy = _dynamic_route_xy(projection=projection, route=route, box=box)
    if len(route_xy) > 1:
        box_height = box[3] - box[1]
        corridor_width = max(22, int(box_height * (0.13 if full_material else 0.11)))
        route_width = max(8, int(corridor_width * 0.30))
        draw.line(route_xy, fill=(22, 65, 75, 168), width=corridor_width, joint="curve")
        draw.line(route_xy, fill=(10, 25, 32, 235), width=max(16, int(corridor_width * 0.68)), joint="curve")
        draw.line(route_xy, fill=(47, 137, 221, 245), width=route_width, joint="curve")
        draw.line(route_xy, fill=(156, 220, 255, 245), width=max(2, route_width // 3), joint="curve")
        marker_step = max(1, len(route_xy) // 6)
        marker_radius = 7 if full_material else 4
        for waypoint_x, waypoint_y in route_xy[::marker_step]:
            draw.ellipse(
                (
                    waypoint_x - marker_radius,
                    waypoint_y - marker_radius,
                    waypoint_x + marker_radius,
                    waypoint_y + marker_radius,
                ),
                fill=(162, 211, 255, 245),
            )

    eye_x, eye_y = _route_map_xy(camera.eye, projection=projection, box=box)
    target_x, target_y = _route_map_xy(camera.target, projection=projection, box=box)
    _draw_ego_vehicle_marker(
        draw,
        eye=(eye_x, eye_y),
        target=(target_x, target_y),
        full_material=full_material,
        labels=labels,
    )


def _draw_material_scale(
    draw: ImageDraw.ImageDraw,
    *,
    projection: RouteMapProjection,
    box: tuple[int, int, int, int],
    full_material: bool,
    labels: bool,
) -> None:
    scale_width = min(180 if full_material else 96, max(54, int((box[2] - box[0]) * 0.18)))
    scale_x0 = box[0] + (28 if full_material else 14)
    scale_y = box[3] - (32 if full_material else 18)
    line_width = 4 if full_material else 3
    draw.line((scale_x0, scale_y, scale_x0 + scale_width, scale_y), fill=(226, 239, 250, 225), width=line_width)
    draw.line((scale_x0, scale_y - 6, scale_x0, scale_y + 6), fill=(226, 239, 250, 225), width=2)
    draw.line(
        (scale_x0 + scale_width, scale_y - 6, scale_x0 + scale_width, scale_y + 6),
        fill=(226, 239, 250, 225),
        width=2,
    )
    if labels:
        draw.text(
            (scale_x0, scale_y + 10),
            f"{_route_span_meters(projection) * scale_width / max(1, box[2] - box[0]):.1f} m",
            font=_load_font(13 if full_material else 10),
            fill=(226, 239, 250, 210),
        )


def _offset_polyline(points: list[tuple[int, int]], offset: int) -> list[tuple[int, int]]:
    shifted: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        if index == 0:
            previous_point = points[index]
            next_point = points[index + 1]
        elif index == len(points) - 1:
            previous_point = points[index - 1]
            next_point = points[index]
        else:
            previous_point = points[index - 1]
            next_point = points[index + 1]
        dx = next_point[0] - previous_point[0]
        dy = next_point[1] - previous_point[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            normal = (0.0, -1.0)
        else:
            normal = (-dy / length, dx / length)
        shifted.append((int(point[0] + normal[0] * offset), int(point[1] + normal[1] * offset)))
    return shifted


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
) -> None:
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-6:
            continue
        steps = max(1, int(segment_length / dash))
        for step_index in range(steps):
            if step_index % 2:
                continue
            t0 = step_index / steps
            t1 = min(1.0, (step_index + 0.72) / steps)
            x0 = int(start[0] + dx * t0)
            y0 = int(start[1] + dy * t0)
            x1 = int(start[0] + dx * t1)
            y1 = int(start[1] + dy * t1)
            draw.line((x0, y0, x1, y1), fill=fill, width=width)


def _perpendicular_segment(
    points: list[tuple[int, int]],
    index: int,
    half_width: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    point = points[index]
    next_point = points[min(len(points) - 1, index + 1)]
    dx = next_point[0] - point[0]
    dy = next_point[1] - point[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        normal = (0.0, -1.0)
    else:
        normal = (-dy / length, dx / length)
    return (
        (int(point[0] - normal[0] * half_width), int(point[1] - normal[1] * half_width)),
        (int(point[0] + normal[0] * half_width), int(point[1] + normal[1] * half_width)),
    )


def _draw_ego_vehicle_marker(
    draw: ImageDraw.ImageDraw,
    *,
    eye: tuple[int, int],
    target: tuple[int, int],
    full_material: bool,
    labels: bool,
) -> None:
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        direction = (1.0, 0.0)
    else:
        direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    vehicle_length = 42 if full_material else 26
    vehicle_width = 22 if full_material else 14
    front = (eye[0] + direction[0] * vehicle_length * 0.58, eye[1] + direction[1] * vehicle_length * 0.58)
    rear = (eye[0] - direction[0] * vehicle_length * 0.42, eye[1] - direction[1] * vehicle_length * 0.42)
    corners = (
        (front[0], front[1]),
        (eye[0] + normal[0] * vehicle_width * 0.5, eye[1] + normal[1] * vehicle_width * 0.5),
        (rear[0] + normal[0] * vehicle_width * 0.5, rear[1] + normal[1] * vehicle_width * 0.5),
        (rear[0] - normal[0] * vehicle_width * 0.5, rear[1] - normal[1] * vehicle_width * 0.5),
        (eye[0] - normal[0] * vehicle_width * 0.5, eye[1] - normal[1] * vehicle_width * 0.5),
    )
    polygon = [(int(x), int(y)) for x, y in corners]
    draw.polygon(polygon, fill=(91, 232, 120, 220), outline=(210, 255, 218, 245))
    axis_length = 28 if full_material else 16
    draw.line(
        (eye[0], eye[1], int(eye[0] + direction[0] * axis_length), int(eye[1] + direction[1] * axis_length)),
        fill=(255, 82, 82, 245),
        width=3 if full_material else 2,
    )
    draw.line(
        (eye[0], eye[1], int(eye[0] + normal[0] * axis_length), int(eye[1] + normal[1] * axis_length)),
        fill=(88, 178, 255, 235),
        width=3 if full_material else 2,
    )
    if full_material and labels:
        draw.text((eye[0] + 24, eye[1] - 12), "base_link", font=_load_font(15), fill=(196, 255, 206, 245))


def _draw_material_legend(draw: ImageDraw.ImageDraw, *, box: tuple[int, int, int, int]) -> None:
    legend_font = _load_font(15)
    x = box[2] - 358
    y = box[1] + 18
    items = (
        ((91, 232, 120, 150), "resident PCD cells"),
        ((242, 190, 82, 150), "preload request"),
        ((230, 244, 236, 190), "route overlay"),
        ((232, 246, 238, 165), ".splat footprint density"),
    )
    draw.rounded_rectangle((x - 16, y - 12, box[2] - 18, y + len(items) * 28 + 8), radius=8, fill=(4, 10, 14, 205))
    for index, (color, label) in enumerate(items):
        yy = y + index * 28
        draw.rounded_rectangle((x, yy, x + 26, yy + 14), radius=3, fill=color)
        draw.text((x + 38, yy - 2), label, font=legend_font, fill=(218, 232, 242, 235))


def _draw_map_loading_status_hud(
    draw: ImageDraw.ImageDraw,
    *,
    projection: RouteMapProjection,
    camera: CameraPose,
    box: tuple[int, int, int, int],
    gaussian_count: int | None,
) -> None:
    loaded_tiles, preload_tiles, active_tile = _tile_residency_sets(camera, projection=projection)
    panel_width = 382
    panel_height = 118
    x0 = box[2] - panel_width - 18
    y0 = box[3] - panel_height - 18
    x1 = x0 + panel_width
    y1 = y0 + panel_height
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(3, 8, 12, 214), outline=(87, 158, 206, 92))
    title_font = _load_font(15)
    text_font = _load_font(13)
    total = f"{gaussian_count // 1000}k" if gaussian_count is not None else "n/a"
    lines = (
        "/map/pointcloud_map  GetPartialPointCloudMap",
        "/route_overlay       debug marker layer",
        f"loaded {len(loaded_tiles):02d} PCD cells / preload {len(preload_tiles):02d} / src {total}",
        f"active cell pcd_{active_tile[0] + 1:02d}_{active_tile[1] + 1:02d}  frame_id=map -> base_link",
    )
    draw.text((x0 + 14, y0 + 12), "map_loader debug view", font=title_font, fill=(196, 225, 243, 245))
    for index, line in enumerate(lines):
        draw.text((x0 + 14, y0 + 36 + index * 18), line, font=text_font, fill=(155, 185, 205, 230))


def _draw_material_footer(
    draw: ImageDraw.ImageDraw,
    points: list[SplatPoint],
    *,
    projection: RouteMapProjection,
    bounds: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    source_asset: str | None,
    gaussian_count: int | None,
) -> None:
    left, right, bottom, top = bounds
    width, height = size
    footer_font = _load_font(18)
    meta_font = _load_font(14 if width < 1100 else 16)
    source_name = source_asset or "actual shipped .splat"
    if width < 1100:
        gaussian_text = f"{gaussian_count // 1000}k gaussians" if gaussian_count is not None else "sampled splats"
        source_line = f"source: {source_name} / {gaussian_text} / partial map loading"
    else:
        gaussian_text = f"{gaussian_count // 1000}k total gaussians / " if gaussian_count is not None else ""
        source_line = (
            f"source: {source_name} / {gaussian_text}sampled {len(points) // 1000}k splats / partial map loading"
        )
    draw.text(
        (box[0], height - 68),
        f"PCD footprint {right - left:.1f} x {top - bottom:.1f} m / route span {_route_span_meters(projection):.1f} m",
        font=footer_font,
        fill=(225, 238, 247, 240),
    )
    draw.text(
        (box[0], height - 38),
        source_line,
        font=meta_font,
        fill=(162, 184, 202, 220),
    )


def _tile_residency_sets(
    camera: CameraPose,
    *,
    projection: RouteMapProjection,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    active_column, active_row = _active_tile(
        camera.eye, projection=projection, columns=MAP_TILE_COLUMNS, rows=MAP_TILE_ROWS
    )
    loaded_tiles: set[tuple[int, int]] = set()
    for column in range(active_column - 1, active_column + 2):
        for row in range(active_row - 1, active_row + 2):
            if 0 <= column < MAP_TILE_COLUMNS and 0 <= row < MAP_TILE_ROWS:
                loaded_tiles.add((column, row))

    preload_tiles: set[tuple[int, int]] = set()
    for column in (active_column + 2,):
        for row in range(active_row - 1, active_row + 2):
            if 0 <= column < MAP_TILE_COLUMNS and 0 <= row < MAP_TILE_ROWS:
                preload_tiles.add((column, row))

    return loaded_tiles, preload_tiles, (active_column, active_row)


def _tile_rect(column: int, row: int, *, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    left = int(x0 + (x1 - x0) * column / MAP_TILE_COLUMNS)
    right = int(x0 + (x1 - x0) * (column + 1) / MAP_TILE_COLUMNS)
    top = int(y0 + (y1 - y0) * row / MAP_TILE_ROWS)
    bottom = int(y0 + (y1 - y0) * (row + 1) / MAP_TILE_ROWS)
    return left, top, right, bottom


def _tile_group_bounds(
    tiles: set[tuple[int, int]], *, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    if not tiles:
        return None
    rects = [_tile_rect(column, row, box=box) for column, row in tiles]
    return (
        min(rect[0] for rect in rects) + 4,
        min(rect[1] for rect in rects) + 4,
        max(rect[2] for rect in rects) - 4,
        max(rect[3] for rect in rects) - 4,
    )


def _route_map_projection(
    points: list[SplatPoint],
    *,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> RouteMapProjection:
    center = _plane_median(points, axes)
    direction = _principal_direction(points, axes=axes, bounds=bounds)
    side = (-direction[1], direction[0])
    left, right, bottom, top = bounds
    step = max(1, len(points) // 70_000)
    route_values: list[tuple[float, float]] = []
    for point in points[::step]:
        x = point.xyz[axes[0]]
        y = point.xyz[axes[1]]
        if not (left <= x <= right and bottom <= y <= top):
            continue
        route_values.append(_route_coords(point.xyz, axes=axes, center=center, direction=direction, side=side))

    if len(route_values) < 2:
        return RouteMapProjection(
            axes=axes,
            center=center,
            direction=direction,
            side=side,
            progress_bounds=(-1.0, 1.0),
            lateral_bounds=(-1.0, 1.0),
        )

    progress_values = sorted(value[0] for value in route_values)
    lateral_values = sorted(value[1] for value in route_values)
    progress_low, progress_high = _robust_span(progress_values)
    lateral_low, lateral_high = _robust_span(lateral_values)
    progress_span = max(1.0, progress_high - progress_low)
    minimum_lateral_span = max(6.0, progress_span * 0.22)
    if lateral_high - lateral_low < minimum_lateral_span:
        lateral_mid = _percentile(lateral_values, 0.50)
        lateral_low = lateral_mid - minimum_lateral_span / 2.0
        lateral_high = lateral_mid + minimum_lateral_span / 2.0

    return RouteMapProjection(
        axes=axes,
        center=center,
        direction=direction,
        side=side,
        progress_bounds=(progress_low, progress_high),
        lateral_bounds=(lateral_low, lateral_high),
    )


def _route_coords(
    xyz: tuple[float, float, float],
    *,
    axes: tuple[int, int],
    center: tuple[float, float],
    direction: tuple[float, float],
    side: tuple[float, float],
) -> tuple[float, float]:
    delta_x = xyz[axes[0]] - center[0]
    delta_y = xyz[axes[1]] - center[1]
    progress = delta_x * direction[0] + delta_y * direction[1]
    lateral = delta_x * side[0] + delta_y * side[1]
    return progress, lateral


def _route_map_xy(
    xyz: tuple[float, float, float],
    *,
    projection: RouteMapProjection,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    progress, lateral = _route_coords(
        xyz,
        axes=projection.axes,
        center=projection.center,
        direction=projection.direction,
        side=projection.side,
    )
    return _route_coords_to_xy(progress, lateral, projection=projection, box=box)


def _route_coords_to_xy(
    progress: float,
    lateral: float,
    *,
    projection: RouteMapProjection,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    progress_low, progress_high = projection.progress_bounds
    lateral_low, lateral_high = projection.lateral_bounds
    x0, y0, x1, y1 = box
    x_ratio = (progress - progress_low) / (progress_high - progress_low)
    y_ratio = (lateral - lateral_low) / (lateral_high - lateral_low)
    x = int(x0 + max(0.0, min(1.0, x_ratio)) * (x1 - x0))
    y = int(y1 - max(0.0, min(1.0, y_ratio)) * (y1 - y0))
    return x, y


def _dynamic_route_xy(
    *,
    projection: RouteMapProjection,
    route: list[CameraPose],
    box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    progress_low, progress_high = projection.progress_bounds
    lateral_low, lateral_high = projection.lateral_bounds
    route_laterals = [
        _route_coords(
            pose.eye,
            axes=projection.axes,
            center=projection.center,
            direction=projection.direction,
            side=projection.side,
        )[1]
        for pose in route
    ]
    lateral_center = sum(route_laterals) / max(1, len(route_laterals))
    lateral_swing = (lateral_high - lateral_low) * 0.10
    points: list[tuple[int, int]] = []
    for index in range(36):
        progress = index / 35
        eased = _ease_in_out(progress)
        route_progress = progress_low + (progress_high - progress_low) * eased
        route_lateral = lateral_center + math.sin(progress * math.pi * 1.12) * lateral_swing
        points.append(_route_coords_to_xy(route_progress, route_lateral, projection=projection, box=box))
    return points


def _route_span_meters(projection: RouteMapProjection) -> float:
    return projection.progress_bounds[1] - projection.progress_bounds[0]


def _active_tile(
    xyz: tuple[float, float, float],
    *,
    projection: RouteMapProjection,
    columns: int = 6,
    rows: int = 4,
) -> tuple[int, int]:
    progress, lateral = _route_coords(
        xyz,
        axes=projection.axes,
        center=projection.center,
        direction=projection.direction,
        side=projection.side,
    )
    progress_low, progress_high = projection.progress_bounds
    lateral_low, lateral_high = projection.lateral_bounds
    column_ratio = (progress - progress_low) / (progress_high - progress_low)
    row_ratio = 1.0 - (lateral - lateral_low) / (lateral_high - lateral_low)
    column = int(max(0, min(columns - 1, math.floor(column_ratio * columns))))
    row = int(max(0, min(rows - 1, math.floor(row_ratio * rows))))
    return column, row


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
    title_font = _load_font(23)
    meta_font = _load_font(14)
    chip_font = _load_font(15)
    draw.rounded_rectangle((18, 16, 512, 96), radius=6, fill=(4, 9, 15, 215), outline=(255, 255, 255, 36))
    draw.text((34, 29), scene.label, font=title_font, fill=(244, 249, 255, 255))
    draw.text(
        (36, 65),
        (f"dynamic map route / {gaussian_count // 1000}k gaussians / {rendered_count // 1000}k visible"),
        font=meta_font,
        fill=(203, 218, 232, 240),
    )
    draw.rounded_rectangle(
        (width - 176, 20, width - 24, 54), radius=5, fill=(4, 9, 15, 205), outline=(91, 232, 120, 70)
    )
    draw.text(
        (width - 158, 29),
        f"COURSE {scene_index:02d}/{scene_count:02d}",
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
