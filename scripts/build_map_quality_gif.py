#!/usr/bin/env python3
"""Build the README dynamic-map GIF from the actual .splat tile data.

Every map element is derived from shipped assets: each grid cell shows its
tile's real `.splat` content rendered top-down (alpha-weighted color
accumulation), the route polyline is the actual robot-route waypoint list, and
the resident/preload window animates along the real tile visit sequence. No
decorative geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
ASSET_DIR = DOCS / "assets" / "outdoor-demo"
OUTPUT = DOCS / "images" / "demo-sweep" / "map-quality.gif"
MAP_MATERIAL_OUTPUT = DOCS / "images" / "demo-sweep" / "dynamic-map-material.png"
FRAME_SIZE = (960, 540)
MAP_MATERIAL_SIZE = (1280, 720)
SPLAT_RECORD_BYTES = 32
FRAMES_PER_SCENE = 12
FRAME_DURATION_MS = 420

RESIDENT_LIMIT = 6
PRELOAD_LIMIT = 3
SUPERSAMPLE = 3

BG = (8, 12, 18)
PANEL_BG = (11, 16, 24)
GRID_COLOR = (32, 48, 66)
TEXT = (230, 237, 243)
TEXT_DIM = (139, 148, 158)
RESIDENT_COLOR = (63, 185, 80)
PRELOAD_COLOR = (210, 153, 34)
LOADED_COLOR = (88, 110, 134)
ROUTE_DONE = (88, 166, 255)
ROUTE_AHEAD = (60, 74, 92)
ROBOT_COLOR = (255, 255, 255)

SPLAT_DTYPE = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])


@dataclass(frozen=True, slots=True)
class MapProofScene:
    asset: str
    label: str
    axes: tuple[int, int]
    catalog: str | None = None


MAP_PROOF_SCENES = (
    MapProofScene(
        asset="outdoor-production-grid-large-tile-catalog.json",
        label="87-tile outdoor production regional mosaic",
        axes=(0, 2),
        catalog="apps/dreamwalker-web/public/manifests/outdoor-production-grid-large-tile-catalog.json",
    ),
)


@dataclass(frozen=True, slots=True)
class Tile:
    tile_id: str
    splat_path: Path
    bounds: tuple[float, float, float, float]  # minX, maxX, minZ, maxZ


@dataclass(frozen=True, slots=True)
class MosaicScene:
    label: str
    tiles: dict[str, Tile]
    tile_sequence: list[str]
    route: list[tuple[float, float]]  # (x, z) waypoints
    world_bounds: tuple[float, float, float, float]
    source_gaussians: int
    tiled_gaussians: int


class MapTransform:
    """World (x, z) meters -> map panel pixels, north-up, uniform scale."""

    def __init__(self, world: tuple[float, float, float, float], box: tuple[int, int, int, int]):
        min_x, max_x, min_z, max_z = world
        x0, y0, x1, y1 = box
        span_x = max(max_x - min_x, 1e-6)
        span_z = max(max_z - min_z, 1e-6)
        self.scale = min((x1 - x0) / span_x, (y1 - y0) / span_z)
        used_w = span_x * self.scale
        used_h = span_z * self.scale
        self.origin_px = (x0 + ((x1 - x0) - used_w) / 2.0, y0 + ((y1 - y0) - used_h) / 2.0)
        self.world_min = (min_x, min_z)
        self.world_span = (span_x, span_z)

    def to_px(self, x: float, z: float) -> tuple[float, float]:
        u = self.origin_px[0] + (x - self.world_min[0]) * self.scale
        v = self.origin_px[1] + (self.world_span[1] - (z - self.world_min[1])) * self.scale
        return u, v

    def rect_px(self, bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        min_x, max_x, min_z, max_z = bounds
        u0, v0 = self.to_px(min_x, max_z)
        u1, v1 = self.to_px(max_x, min_z)
        return int(round(u0)), int(round(v0)), int(round(u1)), int(round(v1))


def load_mosaic_scene(scene: MapProofScene) -> MosaicScene:
    catalog_path = REPO / scene.catalog if scene.catalog else ASSET_DIR / scene.asset
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tiles: dict[str, Tile] = {}
    for tile in catalog["tiles"]:
        core = tile["coreBounds"]
        tiles[tile["id"]] = Tile(
            tile_id=tile["id"],
            splat_path=REPO / tile["publicPath"],
            bounds=(core["minX"], core["maxX"], core["minZ"], core["maxZ"]),
        )
    world = catalog["tiling"]["worldBounds"]
    route_path = REPO / "apps/dreamwalker-web/public/robot-routes/outdoor-production-grid-large-route.json"
    route_doc = json.loads(route_path.read_text(encoding="utf-8"))
    route = [(float(p[0]), float(p[2])) for p in route_doc["route"]]
    sequence = [tile_id for tile_id in route_doc["tileSequence"] if tile_id in tiles]
    summary = catalog["summary"]
    return MosaicScene(
        label=scene.label,
        tiles=tiles,
        tile_sequence=sequence,
        route=route,
        world_bounds=(world["minX"], world["maxX"], world["minZ"], world["maxZ"]),
        source_gaussians=int(summary.get("inputSplatCount", 0)),
        tiled_gaussians=int(summary.get("tiledSplatCount", 0)),
    )


@lru_cache(maxsize=None)
def _tile_chip(splat_path: str, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> Image.Image:
    """Top-down ortho rendering of one tile's actual splat content."""
    width, height = size
    ss_w, ss_h = max(width * SUPERSAMPLE, 2), max(height * SUPERSAMPLE, 2)
    raw = np.fromfile(splat_path, dtype=SPLAT_DTYPE)
    min_x, max_x, min_z, max_z = bounds
    x = raw["pos"][:, 0]
    z = raw["pos"][:, 2]
    inside = (x >= min_x) & (x <= max_x) & (z >= min_z) & (z <= max_z)
    raw = raw[inside]
    if len(raw) == 0:
        return Image.new("RGB", size, PANEL_BG)
    x = raw["pos"][:, 0]
    z = raw["pos"][:, 2]
    rgba = raw["rgba"].astype(np.float32) / 255.0
    u = np.clip((x - min_x) / (max_x - min_x) * (ss_w - 1), 0, ss_w - 1).astype(np.int64)
    v = np.clip((max_z - z) / (max_z - min_z) * (ss_h - 1), 0, ss_h - 1).astype(np.int64)
    flat = v * ss_w + u
    weight = rgba[:, 3]
    acc = np.zeros((ss_h * ss_w, 3), dtype=np.float64)
    wgt = np.zeros(ss_h * ss_w, dtype=np.float64)
    for channel in range(3):
        np.add.at(acc[:, channel], flat, rgba[:, channel] * weight)
    np.add.at(wgt, flat, weight)
    color = np.zeros_like(acc)
    filled = wgt > 0
    color[filled] = acc[filled] / wgt[filled, None]
    density_norm = np.percentile(wgt[filled], 60) if filled.any() else 1.0
    density = np.tanh(wgt / max(density_norm, 1e-6))
    shaded = color * (0.45 + 0.55 * density[:, None])
    # per-chip auto gain so sparse tiles still read as map content
    luminance = shaded[filled].mean(axis=1)
    if luminance.size:
        gain = 0.85 / max(float(np.percentile(luminance, 95)), 0.05)
        shaded = shaded * min(gain, 3.0)
    base = np.array(PANEL_BG, dtype=np.float64) / 255.0
    out = base + (shaded - base) * density[:, None].clip(0.0, 1.0)
    out_img = Image.fromarray((np.clip(out.reshape(ss_h, ss_w, 3), 0, 1) * 255).astype(np.uint8)).resize(
        size, Image.LANCZOS
    )
    return ImageEnhance.Color(out_img).enhance(1.35)


def _residency(scene: MosaicScene, visited: int) -> tuple[list[str], list[str], list[str]]:
    sequence = scene.tile_sequence
    visited = max(0, min(visited, len(sequence)))
    resident = sequence[max(0, visited - RESIDENT_LIMIT) : visited]
    loaded = sequence[: max(0, visited - RESIDENT_LIMIT)]
    preload = sequence[visited : visited + PRELOAD_LIMIT]
    return resident, preload, loaded


def _dim(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


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


def _robot_pose(route: list[tuple[float, float]], progress_index: float) -> tuple[float, float, float]:
    """Interpolated robot (x, z, heading) along the waypoint list."""
    whole = min(int(progress_index), len(route) - 1)
    frac = progress_index - int(progress_index)
    if whole < len(route) - 1:
        cur, nxt = route[whole], route[whole + 1]
        x = cur[0] + (nxt[0] - cur[0]) * frac
        z = cur[1] + (nxt[1] - cur[1]) * frac
        heading = math.atan2(-(nxt[1] - cur[1]), nxt[0] - cur[0])
    else:
        x, z = route[-1]
        prev = route[-2] if len(route) > 1 else route[-1]
        heading = math.atan2(-(z - prev[1]), x - prev[0])
    return x, z, heading


def _draw_route(
    draw: ImageDraw.ImageDraw,
    transform: MapTransform,
    route: list[tuple[float, float]],
    progress_index: float,
    *,
    trail: int = 10,
    width: int = 3,
) -> tuple[float, float, float]:
    """Faint full history + bright recent trail; returns robot pose in px."""
    points = [transform.to_px(x, z) for x, z in route]
    whole = min(int(progress_index), len(points) - 1)
    x, z, heading_px = _robot_pose(route, progress_index)  # atan2(-dz, dx) == px-space heading
    robot = transform.to_px(x, z)

    traveled = [*points[: whole + 1], robot]
    if len(traveled) >= 2:
        draw.line(traveled, fill=ROUTE_AHEAD, width=1)
    recent = traveled[-(trail + 1) :]
    if len(recent) >= 2:
        draw.line(recent, fill=ROUTE_DONE, width=width)
    return robot[0], robot[1], heading_px


def _draw_robot(draw: ImageDraw.ImageDraw, x: float, y: float, heading: float, scale: float = 1.0) -> None:
    size = 7.0 * scale
    points = []
    for angle, radius in ((0.0, 1.6), (2.5, 1.0), (math.pi, 0.35), (-2.5, 1.0)):
        points.append((x + math.cos(heading + angle) * size * radius, y + math.sin(heading + angle) * size * radius))
    draw.polygon(points, fill=ROBOT_COLOR, outline=(20, 28, 38))
    ring = 12.0 * scale
    draw.ellipse([x - ring, y - ring, x + ring, y + ring], outline=RESIDENT_COLOR, width=max(1, int(scale)))


def _tile_states(scene: MosaicScene, progress_index: float) -> tuple[dict[str, str], tuple[int, int, int]]:
    visited = int(progress_index) + 1
    resident, preload, loaded = _residency(scene, visited)
    state: dict[str, str] = {tile_id: "loaded" for tile_id in loaded}
    state.update({tile_id: "resident" for tile_id in resident})
    state.update({tile_id: "preload" for tile_id in preload})
    return state, (len(resident), len(preload), len(loaded))


def _viewport_bounds(
    scene: MosaicScene, progress_index: float, viewport_m: float, aspect: float
) -> tuple[float, float, float, float]:
    """viewport_m-tall window centered on the robot, clamped to the world."""
    x, z, _ = _robot_pose(scene.route, progress_index)
    min_x, max_x, min_z, max_z = scene.world_bounds
    half_h = viewport_m / 2.0
    half_w = viewport_m * aspect / 2.0
    cx = min(max(x, min_x + half_w), max_x - half_w) if max_x - min_x > 2 * half_w else (min_x + max_x) / 2
    cz = min(max(z, min_z + half_h), max_z - half_h) if max_z - min_z > 2 * half_h else (min_z + max_z) / 2
    return cx - half_w, cx + half_w, cz - half_h, cz + half_h


def _render_map_panel(
    scene: MosaicScene,
    panel_size: tuple[int, int],
    progress_index: float,
    *,
    viewport_m: float,
    line_scale: float = 1.0,
) -> Image.Image:
    """Robot-following viewport over the tile mosaic (own image, self-clipping)."""
    panel = Image.new("RGB", panel_size, PANEL_BG)
    draw = ImageDraw.Draw(panel)
    width, height = panel_size
    viewport = _viewport_bounds(scene, progress_index, viewport_m, width / height)
    transform = MapTransform(viewport, (0, 0, width, height))

    grid_step_m = 8.0
    min_x, max_x, min_z, max_z = viewport
    gx = math.floor(min_x / grid_step_m) * grid_step_m
    while gx <= max_x:
        u0, v0 = transform.to_px(gx, max_z)
        u1, v1 = transform.to_px(gx, min_z)
        draw.line([(u0, v0), (u1, v1)], fill=GRID_COLOR, width=1)
        gx += grid_step_m
    gz = math.floor(min_z / grid_step_m) * grid_step_m
    while gz <= max_z:
        u0, v0 = transform.to_px(min_x, gz)
        u1, v1 = transform.to_px(max_x, gz)
        draw.line([(u0, v0), (u1, v1)], fill=GRID_COLOR, width=1)
        gz += grid_step_m

    state, _counts = _tile_states(scene, progress_index)
    brightness = {"unvisited": 0.5, "loaded": 0.8, "preload": 0.92, "resident": 1.0}
    outline = {
        "resident": (RESIDENT_COLOR, max(2, int(2 * line_scale))),
        "preload": (PRELOAD_COLOR, max(2, int(2 * line_scale))),
        "loaded": (LOADED_COLOR, 1),
        "unvisited": ((30, 44, 62), 1),
    }
    # fixed px size per tile cell -> chip cache hits across frames
    for tile_id, tile in scene.tiles.items():
        t_min_x, t_max_x, t_min_z, t_max_z = tile.bounds
        if t_max_x < min_x or t_min_x > max_x or t_max_z < min_z or t_min_z > max_z:
            continue
        rect = transform.rect_px(tile.bounds)
        cell_w = max(int(round((t_max_x - t_min_x) * transform.scale)), 2)
        cell_h = max(int(round((t_max_z - t_min_z) * transform.scale)), 2)
        chip = _tile_chip(str(tile.splat_path), tile.bounds, (cell_w, cell_h))
        tile_state = state.get(tile_id, "unvisited")
        panel.paste(_dim(chip, brightness[tile_state]), (rect[0], rect[1]))
        color, line_width = outline[tile_state]
        draw.rectangle(rect, outline=color, width=line_width)

    robot_x, robot_y, heading = _draw_route(
        draw, transform, scene.route, progress_index, width=max(3, int(3 * line_scale))
    )
    _draw_robot(draw, robot_x, robot_y, heading, scale=line_scale * 1.3)
    return panel


def _render_overview_inset(
    scene: MosaicScene,
    size: tuple[int, int],
    progress_index: float,
    viewport: tuple[float, float, float, float],
) -> Image.Image:
    """Whole-mosaic context map with the live viewport rectangle."""
    inset = Image.new("RGB", size, PANEL_BG)
    draw = ImageDraw.Draw(inset)
    transform = MapTransform(scene.world_bounds, (4, 4, size[0] - 4, size[1] - 4))
    state, _counts = _tile_states(scene, progress_index)
    fill = {
        "resident": RESIDENT_COLOR,
        "preload": PRELOAD_COLOR,
        "loaded": (52, 72, 96),
        "unvisited": (24, 34, 48),
    }
    for tile_id, tile in scene.tiles.items():
        rect = transform.rect_px(tile.bounds)
        draw.rectangle(rect, fill=fill[state.get(tile_id, "unvisited")])
    points = [transform.to_px(x, z) for x, z in scene.route]
    draw.line(points, fill=(58, 84, 116), width=1)
    x, z, _ = _robot_pose(scene.route, progress_index)
    rx, ry = transform.to_px(x, z)
    draw.ellipse([rx - 3, ry - 3, rx + 3, ry + 3], fill=ROBOT_COLOR)
    vp = transform.rect_px(viewport)
    draw.rectangle(vp, outline=TEXT, width=1)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=GRID_COLOR)
    return inset


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    transform_scale: float,
    anchor: tuple[int, int],
    *,
    meters: float = 40.0,
    font: ImageFont.ImageFont,
) -> None:
    x, y = anchor
    length = meters * transform_scale
    draw.line([(x, y), (x + length, y)], fill=TEXT, width=2)
    for tick_x in (x, x + length):
        draw.line([(tick_x, y - 4), (tick_x, y + 4)], fill=TEXT, width=2)
    draw.text((x, y - 22), f"{meters:.0f} m", font=font, fill=TEXT)


def _legend_entries() -> list[tuple[tuple[int, int, int], str]]:
    return [
        (RESIDENT_COLOR, "resident 3DGS tiles"),
        (PRELOAD_COLOR, "preload request"),
        (LOADED_COLOR, "loaded (evicted) tiles"),
        (ROUTE_DONE, "robot route traveled"),
    ]


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
    *,
    row_height: int = 24,
    chip: int = 14,
) -> None:
    x, y = origin
    for color, label in _legend_entries():
        draw.rectangle([x, y + 2, x + chip, y + 2 + chip], fill=color)
        draw.text((x + chip + 9, y), label, font=font, fill=TEXT)
        y += row_height


VIEWPORT_M = 72.0


def render_frame(scene: MosaicScene, progress: float, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    big = width >= 1100
    title_font = _load_font(34 if big else 25)
    sub_font = _load_font(17 if big else 13)
    small_font = _load_font(15 if big else 12)

    header_h = 78 if big else 58
    footer_h = 50 if big else 36
    side_w = 300 if big else 234
    margin = 16
    map_box = (margin, header_h, width - side_w, height - footer_h)
    panel_size = (map_box[2] - map_box[0], map_box[3] - map_box[1])

    progress_index = progress * (len(scene.route) - 1)
    panel = _render_map_panel(
        scene,
        panel_size,
        progress_index,
        viewport_m=VIEWPORT_M,
        line_scale=1.4 if big else 1.0,
    )
    image.paste(panel, (map_box[0], map_box[1]))
    draw.rectangle((map_box[0] - 1, map_box[1] - 1, map_box[2], map_box[3]), outline=GRID_COLOR)

    _state, (resident, preload, loaded) = _tile_states(scene, progress_index)
    draw.text((margin, 12), "Regional 3DGS dynamic map", font=title_font, fill=TEXT)
    visited = int(progress_index) + 1
    draw.text(
        (margin, (50 if big else 38)),
        f"{len(scene.tiles)} streamed .splat tiles - route step {visited:02d}/{len(scene.route)} - "
        f"resident {resident:02d} / preload {preload:02d} / evicted {loaded:02d}",
        font=sub_font,
        fill=TEXT_DIM,
    )

    legend_x = width - side_w + 18
    draw.text((legend_x, header_h + 2), "tile residency", font=sub_font, fill=TEXT)
    _draw_legend(
        draw,
        (legend_x, header_h + (30 if big else 24)),
        small_font,
        row_height=28 if big else 22,
        chip=16 if big else 12,
    )
    stats_y = header_h + (158 if big else 122)
    span_x = scene.world_bounds[1] - scene.world_bounds[0]
    span_z = scene.world_bounds[3] - scene.world_bounds[2]
    for line in (
        "map_loader live state",
        f"footprint {span_x:.0f} x {span_z:.0f} m",
        f"source {scene.source_gaussians / 1e6:.2f}M gaussians",
        f"tiled {scene.tiled_gaussians / 1e6:.2f}M gaussians",
        "tile cells: real .splat content",
    ):
        fill = TEXT if line == "map_loader live state" else TEXT_DIM
        draw.text((legend_x, stats_y), line, font=small_font, fill=fill)
        stats_y += 26 if big else 20

    inset_w = side_w - 36
    inset_h = int(inset_w * span_z / span_x)
    inset_y = height - footer_h - inset_h - 8
    viewport = _viewport_bounds(scene, progress_index, VIEWPORT_M, panel_size[0] / panel_size[1])
    inset = _render_overview_inset(scene, (inset_w, inset_h), progress_index, viewport)
    image.paste(inset, (legend_x, inset_y))
    draw.text((legend_x, inset_y - (22 if big else 18)), "regional overview", font=small_font, fill=TEXT)

    panel_scale = panel_size[1] / VIEWPORT_M
    _draw_scale_bar(
        draw,
        panel_scale,
        (map_box[0] + 14, map_box[3] - 16),
        meters=20.0,
        font=small_font,
    )
    draw.text(
        (margin, height - footer_h + (12 if big else 8)),
        "source: outdoor-production-grid-large-tile-catalog.json - every cell, route point, and "
        "load state comes from the shipped assets",
        font=small_font,
        fill=TEXT_DIM,
    )
    return image


def build_map_quality_gif(
    output: Path = OUTPUT,
    *,
    size: tuple[int, int] = FRAME_SIZE,
    frames_per_scene: int = FRAMES_PER_SCENE,
    map_material_output: Path | None = MAP_MATERIAL_OUTPUT,
) -> Path:
    frames: list[Image.Image] = []
    for proof_scene in MAP_PROOF_SCENES:
        scene = load_mosaic_scene(proof_scene)
        for index in range(frames_per_scene):
            progress = index / max(frames_per_scene - 1, 1)
            frames.append(render_frame(scene, progress, size))
        if map_material_output is not None:
            material = render_frame(scene, 0.62, MAP_MATERIAL_SIZE)
            map_material_output.parent.mkdir(parents=True, exist_ok=True)
            material.save(map_material_output)
            print(f"map material: {map_material_output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [FRAME_DURATION_MS] * len(frames)
    durations[-1] = 1400  # hold the fully-loaded map before looping
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"gif: {output} ({output.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--material-output", type=Path, default=MAP_MATERIAL_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    build_map_quality_gif(args.output, map_material_output=args.material_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
