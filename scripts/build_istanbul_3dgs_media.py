#!/usr/bin/env python3
"""Render Istanbul large-scale 3DGS README media from promoted viewer PLY tiles."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "apps/dreamwalker-web/public/manifests/istanbul-bag6-pilot-tile-catalog.json"
ROUTE = REPO / "apps/dreamwalker-web/public/robot-routes/istanbul-bag6-pilot-route.json"
OUTPUT_DIR = REPO / "docs/images/istanbul-bag6-pilot"
SH_C0 = 0.28209479177387814
MAX_POINTS = 520_000
STILL_SIZE = (1280, 720)
GIF_SIZE = (960, 540)


@dataclass(frozen=True, slots=True)
class TileCloud:
    tile_id: str
    points: np.ndarray
    colors: np.ndarray
    alpha: np.ndarray
    bounds: dict[str, float]
    source_count: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--route", type=Path, default=ROUTE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    route = json.loads(args.route.read_text(encoding="utf-8"))
    tiles = _load_tiles(args.catalog, catalog)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_points = np.concatenate([tile.points for tile in tiles], axis=0)
    bounds = _combined_bounds(tiles, all_points)
    route_points = _route_points(route)
    tile_order = [tile.tile_id for tile in tiles]

    still = _render_frame(
        tiles,
        bounds=bounds,
        route_points=route_points,
        active_tile=tile_order[0],
        resident_tiles=set(tile_order[:5]),
        preload_tiles=set(tile_order[1:3]),
        size=STILL_SIZE,
        title="Istanbul Bag6 Large-scale 3DGS",
        subtitle=f"{len(tiles)} ready tiles / {_source_count(tiles):,} viewer Gaussians",
        progress=1.0,
        rich_labels=True,
    )
    still_path = output_dir / "large-scale-3dgs-result.png"
    still.save(still_path)

    frames: list[Image.Image] = []
    for index, tile_id in enumerate(tile_order):
        resident = set(tile_order[max(0, index - 2) : index + 3])
        preload = set(tile_order[index + 1 : index + 3])
        frame = _render_frame(
            tiles,
            bounds=bounds,
            route_points=route_points,
            active_tile=tile_id,
            resident_tiles=resident,
            preload_tiles=preload,
            size=GIF_SIZE,
            title="Dynamic Map Viewer",
            subtitle="3DGS tile residency",
            progress=(index + 1) / len(tile_order),
            rich_labels=False,
        )
        frames.extend([frame] * 2)

    gif_path = output_dir / "dynamic-map-viewer.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        optimize=True,
        disposal=2,
    )

    thumb_path = output_dir / "dynamic-map-viewer-still.png"
    frames[-1].save(thumb_path)
    print(f"wrote {still_path.relative_to(REPO)}")
    print(f"wrote {gif_path.relative_to(REPO)}")
    print(f"wrote {thumb_path.relative_to(REPO)}")


def _load_tiles(catalog_path: Path, catalog: dict[str, object]) -> list[TileCloud]:
    raw_tiles = [tile for tile in catalog.get("tiles", []) if isinstance(tile, dict) and tile.get("status") == "ready"]
    if not raw_tiles:
        raise ValueError(f"{catalog_path} has no ready tiles")

    per_tile = max(1, math.ceil(MAX_POINTS / len(raw_tiles)))
    clouds: list[TileCloud] = []
    for tile in raw_tiles:
        path = _tile_viewer_path(catalog_path, tile)
        points, colors, alpha, source_count = _read_viewer_ply(path, max_points=per_tile)
        bounds = tile.get("viewerCoreBounds") or tile.get("viewerExpandedBounds") or {}
        clouds.append(
            TileCloud(
                tile_id=str(tile["id"]),
                points=points,
                colors=colors,
                alpha=alpha,
                bounds={key: float(value) for key, value in bounds.items()},
                source_count=source_count,
            )
        )
    return sorted(clouds, key=lambda item: (item.bounds.get("minZ", 0.0), item.bounds.get("minX", 0.0)))


def _tile_viewer_path(catalog_path: Path, tile: dict[str, object]) -> Path:
    for key in ("viewerPublicPath", "publicViewerSplat", "publicPath"):
        value = tile.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else REPO / path
    url = tile.get("viewerSplatUrl") or tile.get("splatUrl")
    if isinstance(url, str) and url.startswith("/"):
        return REPO / "apps/dreamwalker-web/public" / url.lstrip("/")
    if isinstance(url, str) and url:
        return catalog_path.parent / url
    raise ValueError(f"tile {tile.get('id')} has no viewer asset")


def _read_viewer_ply(path: Path, *, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    props: list[tuple[str, str]] = []
    vertex_count = 0
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[0] == "property":
            props.append((parts[1], parts[2]))
    fmt_map = {"float": "f", "double": "d", "uchar": "B", "uint8": "B", "int": "i", "uint": "I"}
    fmt = "<" + "".join(fmt_map[prop_type] for prop_type, _ in props)
    stride = struct.calcsize(fmt)
    index = {name: prop_index for prop_index, (_, name) in enumerate(props)}
    sample_stride = max(1, math.ceil(vertex_count / max_points))
    rows = math.ceil(vertex_count / sample_stride)
    points = np.empty((rows, 3), dtype=np.float32)
    colors = np.empty((rows, 3), dtype=np.float32)
    alpha = np.empty((rows,), dtype=np.float32)

    row_index = 0
    for vertex_index in range(0, vertex_count, sample_stride):
        row = struct.unpack_from(fmt, data, header_end + vertex_index * stride)
        points[row_index] = [row[index["x"]], row[index["y"]], row[index["z"]]]
        colors[row_index] = [
            max(0.0, min(1.0, row[index["f_dc_0"]] * SH_C0 + 0.5)),
            max(0.0, min(1.0, row[index["f_dc_1"]] * SH_C0 + 0.5)),
            max(0.0, min(1.0, row[index["f_dc_2"]] * SH_C0 + 0.5)),
        ]
        alpha[row_index] = 1.0 / (1.0 + math.exp(-row[index["opacity"]]))
        row_index += 1
    return points[:row_index], colors[:row_index], alpha[:row_index], vertex_count


def _combined_bounds(tiles: list[TileCloud], points: np.ndarray) -> dict[str, float]:
    min_x = min(
        (tile.bounds.get("minX") for tile in tiles if "minX" in tile.bounds), default=float(np.min(points[:, 0]))
    )
    max_x = max(
        (tile.bounds.get("maxX") for tile in tiles if "maxX" in tile.bounds), default=float(np.max(points[:, 0]))
    )
    min_z = min(
        (tile.bounds.get("minZ") for tile in tiles if "minZ" in tile.bounds), default=float(np.min(points[:, 2]))
    )
    max_z = max(
        (tile.bounds.get("maxZ") for tile in tiles if "maxZ" in tile.bounds), default=float(np.max(points[:, 2]))
    )
    span_x = max(1.0, max_x - min_x)
    span_z = max(1.0, max_z - min_z)
    pad = max(span_x, span_z) * 0.08
    return {"minX": min_x - pad, "maxX": max_x + pad, "minZ": min_z - pad, "maxZ": max_z + pad}


def _route_points(route: dict[str, object]) -> list[tuple[float, float, float]]:
    points = []
    for item in route.get("route", []):
        position = item.get("position") if isinstance(item, dict) else None
        if isinstance(position, list) and len(position) >= 3:
            points.append((float(position[0]), float(position[1]), float(position[2])))
    return points


def _source_count(tiles: list[TileCloud]) -> int:
    return sum(tile.source_count for tile in tiles)


def _render_frame(
    tiles: list[TileCloud],
    *,
    bounds: dict[str, float],
    route_points: list[tuple[float, float, float]],
    active_tile: str,
    resident_tiles: set[str],
    preload_tiles: set[str],
    size: tuple[int, int],
    title: str,
    subtitle: str,
    progress: float,
    rich_labels: bool,
) -> Image.Image:
    image = _render_cloud(tiles, bounds=bounds, size=size)
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_vignette(draw, size)
    _draw_tiles(
        draw,
        tiles,
        bounds=bounds,
        size=size,
        active_tile=active_tile,
        resident_tiles=resident_tiles,
        preload_tiles=preload_tiles,
    )
    _draw_route(draw, route_points, bounds=bounds, size=size, progress=progress)
    _draw_header(draw, size=size, title=title, subtitle=subtitle, rich_labels=rich_labels)
    return image


def _render_cloud(tiles: list[TileCloud], *, bounds: dict[str, float], size: tuple[int, int]) -> Image.Image:
    width, height = size
    bg = np.zeros((height, width, 3), dtype=np.float32)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    bg[..., 0] = 6 + 8 * yy
    bg[..., 1] = 16 + 16 * yy
    bg[..., 2] = 18 + 22 * yy
    accum = np.zeros_like(bg)
    weight = np.zeros((height, width), dtype=np.float32)

    min_x, max_x = bounds["minX"], bounds["maxX"]
    min_z, max_z = bounds["minZ"], bounds["maxZ"]
    for tile in tiles:
        points = tile.points
        px = ((points[:, 0] - min_x) / (max_x - min_x) * (width - 96) + 48).astype(np.int32)
        py = ((1.0 - (points[:, 2] - min_z) / (max_z - min_z)) * (height - 96) + 48).astype(np.int32)
        valid = (px >= 2) & (px < width - 2) & (py >= 2) & (py < height - 2)
        px = px[valid]
        py = py[valid]
        colors = tile.colors[valid]
        alpha = np.clip(tile.alpha[valid] * 2.4 + 0.1, 0.12, 1.0)
        height_boost = np.clip((points[valid, 1] + 6.0) / 28.0, 0.0, 1.0)[:, None]
        colors = np.clip(colors * (0.78 + height_boost * 0.52), 0.0, 1.0)
        for ox, oy, scale in (
            (0, 0, 1.0),
            (1, 0, 0.5),
            (-1, 0, 0.5),
            (0, 1, 0.5),
            (0, -1, 0.5),
            (1, 1, 0.24),
            (1, -1, 0.24),
            (-1, 1, 0.24),
            (-1, -1, 0.24),
        ):
            x = px + ox
            y = py + oy
            w = alpha * scale
            np.add.at(accum, (y, x), colors * w[:, None] * 255.0)
            np.add.at(weight, (y, x), w)

    mask = weight > 0
    cloud = np.zeros_like(bg)
    cloud[mask] = accum[mask] / np.maximum(weight[mask, None], 1e-6)
    intensity = np.clip(weight * 1.18, 0.0, 1.0) ** 0.42
    out = bg * (1.0 - intensity[..., None]) + cloud * intensity[..., None]
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = ImageEnhance.Color(image).enhance(1.16)
    return image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=90, threshold=3)).convert("RGBA")


def _world_to_screen(x: float, z: float, *, bounds: dict[str, float], size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    sx = int((x - bounds["minX"]) / (bounds["maxX"] - bounds["minX"]) * (width - 96) + 48)
    sy = int((1.0 - (z - bounds["minZ"]) / (bounds["maxZ"] - bounds["minZ"])) * (height - 96) + 48)
    return sx, sy


def _draw_tiles(
    draw: ImageDraw.ImageDraw,
    tiles: list[TileCloud],
    *,
    bounds: dict[str, float],
    size: tuple[int, int],
    active_tile: str,
    resident_tiles: set[str],
    preload_tiles: set[str],
) -> None:
    for tile in tiles:
        if not {"minX", "maxX", "minZ", "maxZ"}.issubset(tile.bounds):
            continue
        x0, y0 = _world_to_screen(tile.bounds["minX"], tile.bounds["maxZ"], bounds=bounds, size=size)
        x1, y1 = _world_to_screen(tile.bounds["maxX"], tile.bounds["minZ"], bounds=bounds, size=size)
        color = (150, 172, 176, 105)
        width = 1
        strip = None
        if tile.tile_id in preload_tiles:
            color = (247, 190, 88, 210)
            strip = color
            width = 3
        if tile.tile_id in resident_tiles:
            color = (101, 205, 158, 215)
            strip = color
            width = 3
        if tile.tile_id == active_tile:
            color = (94, 224, 226, 255)
            strip = color
            width = 4
        draw.rounded_rectangle((x0, y0, x1, y1), radius=7, outline=color, width=width)
        if strip is not None:
            strip_height = max(4, min(8, (y1 - y0) // 13))
            draw.rectangle((x0 + width, y0 + width, x1 - width, y0 + width + strip_height), fill=strip)
            if tile.tile_id == active_tile:
                draw.rectangle((x0 + width, y0 + width, x0 + width + strip_height, y1 - width), fill=strip)


def _draw_route(
    draw: ImageDraw.ImageDraw,
    route_points: list[tuple[float, float, float]],
    *,
    bounds: dict[str, float],
    size: tuple[int, int],
    progress: float,
) -> None:
    if len(route_points) < 2:
        return
    screen = [_world_to_screen(point[0], point[2], bounds=bounds, size=size) for point in route_points]
    draw.line(screen, fill=(236, 244, 239, 118), width=3, joint="curve")
    active_count = max(2, min(len(screen), math.ceil(progress * len(screen))))
    draw.line(screen[:active_count], fill=(255, 212, 94, 245), width=5, joint="curve")
    x, y = screen[active_count - 1]
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(255, 236, 140, 255), outline=(30, 38, 36, 220), width=2)


def _draw_header(
    draw: ImageDraw.ImageDraw,
    *,
    size: tuple[int, int],
    title: str,
    subtitle: str,
    rich_labels: bool,
) -> None:
    width, height = size
    title_font = _font(34 if width >= 1200 else 26, bold=True)
    label_font = _font(18 if width >= 1200 else 14, bold=False)
    small_font = _font(15 if width >= 1200 else 12, bold=False)
    draw.text((46, 38), title, font=title_font, fill=(0, 0, 0, 210))
    draw.text((44, 36), title, font=title_font, fill=(232, 250, 250, 248))
    draw.text((47, 82), subtitle, font=label_font, fill=(0, 0, 0, 190))
    draw.text((45, 80), subtitle, font=label_font, fill=(181, 209, 208, 235))
    legend_x = width - 318
    draw.rounded_rectangle(
        (legend_x, height - 104, width - 28, height - 28),
        radius=8,
        fill=(5, 16, 18, 172),
        outline=(194, 218, 207, 90),
        width=1,
    )
    for idx, (name, color) in enumerate(
        (
            ("active", (94, 224, 226, 255)),
            ("resident", (101, 205, 158, 255)),
            ("preload", (247, 190, 88, 255)),
        )
    ):
        x = legend_x + 22 + idx * 90
        y = height - 76
        draw.rounded_rectangle((x, y, x + 18, y + 18), radius=4, fill=color)
        draw.text((x + 25, y - 1), name, font=small_font, fill=(220, 234, 229, 230))
    if rich_labels:
        draw.line((48, height - 50, 238, height - 50), fill=(224, 239, 232, 205), width=3)
        draw.text((48, height - 74), "30 m tile grid", font=small_font, fill=(188, 210, 205, 218))


def _draw_vignette(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    draw.rectangle((0, 0, width, 22), fill=(0, 0, 0, 80))
    draw.rectangle((0, height - 42, width, height), fill=(0, 0, 0, 95))
    draw.rectangle((0, 0, 24, height), fill=(0, 0, 0, 80))
    draw.rectangle((width - 24, 0, width, height), fill=(0, 0, 0, 80))


def _font(size: int, *, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
