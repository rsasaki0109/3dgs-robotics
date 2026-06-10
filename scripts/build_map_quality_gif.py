#!/usr/bin/env python3
"""Build the README dynamic-map GIF from real rosbag-derived .splat tiles.

The base layer is a true orthographic gsplat render of the Istanbul Bag6
pilot scene (real rosbag2 data) viewed top-down — an actual aerial map of the
mapped street. The resident/preload tile window moves along a continuous
camera trajectory recovered from the data itself (tile adjacency chain pulled
onto the splat-density centerline of the road), so the loading boxes follow
the drive instead of teleporting between grid cells.

Two stages:

    # 1. (GPU + gsplat required, run once) re-render the ortho base map
    python3 scripts/build_map_quality_gif.py --render-base

    # 2. (CPU-only) compose the GIF + material PNG from the committed base
    python3 scripts/build_map_quality_gif.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
ASSET_DIR = DOCS / "assets" / "outdoor-demo"
OUTPUT = DOCS / "images" / "demo-sweep" / "map-quality.gif"
MAP_MATERIAL_OUTPUT = DOCS / "images" / "demo-sweep" / "dynamic-map-material.png"
BASE_MAP_OUTPUT = DOCS / "images" / "demo-sweep" / "istanbul-bag6-ortho-base.png"
BASE_MAP_META_OUTPUT = DOCS / "images" / "demo-sweep" / "istanbul-bag6-ortho-base.json"
FRAME_SIZE = (960, 540)
MAP_MATERIAL_SIZE = (1280, 720)
SPLAT_RECORD_BYTES = 32
FRAMES_PER_SCENE = 12
FRAME_DURATION_MS = 460

RESIDENT_LIMIT = 3
PRELOAD_LIMIT = 1
BASE_MAP_RESOLUTION_M = 0.08  # meters per pixel of the ortho base render
BASE_MAP_SCALE_PERCENTILE = 95.0
BASE_MAP_MIN_OPACITY = 0.08

BG = (8, 12, 18)
PANEL_BG = (11, 16, 24)
GRID_COLOR = (32, 48, 66)
TEXT = (230, 237, 243)
TEXT_DIM = (139, 148, 158)
RESIDENT_COLOR = (63, 185, 80)
PRELOAD_COLOR = (210, 153, 34)
LOADED_COLOR = (96, 120, 146)
ROUTE_DONE = (88, 166, 255)
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
        asset="istanbul-bag6-pilot-tile-catalog.json",
        label="Istanbul Bag6 real rosbag2 3DGS pilot",
        axes=(0, 1),
        catalog="apps/dreamwalker-web/public/manifests/istanbul-bag6-pilot-tile-catalog.json",
    ),
)

ROUTE_JSON = "apps/dreamwalker-web/public/robot-routes/istanbul-bag6-pilot-route.json"


@dataclass(frozen=True, slots=True)
class Tile:
    tile_id: str
    splat_path: Path
    bounds: tuple[float, float, float, float]  # minX, maxX, minY, maxY (splat x/y plane)


@dataclass(frozen=True, slots=True)
class PilotScene:
    label: str
    tiles: dict[str, Tile]
    tile_sequence: list[str]
    route: list[tuple[float, float]]  # (x, y) waypoints in the splat x/y plane
    gaussian_count: int
    splat_bytes: int


def load_pilot_scene(scene: MapProofScene) -> PilotScene:
    catalog_path = REPO / scene.catalog if scene.catalog else ASSET_DIR / scene.asset
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tiles: dict[str, Tile] = {}
    for tile in catalog["tiles"]:
        core = tile["coreBounds"]
        tiles[tile["id"]] = Tile(
            tile_id=tile["id"],
            splat_path=REPO / tile["publicPath"],
            bounds=(core["minX"], core["maxX"], core["minY"], core["maxY"]),
        )
    route_doc = json.loads((REPO / ROUTE_JSON).read_text(encoding="utf-8"))
    # robot-route waypoints are [x, 0, z] in viewer coords; viewer z == splat y here
    route = [(float(p[0]), float(p[2])) for p in route_doc["route"]]
    sequence = [tile_id for tile_id in route_doc["tileSequence"] if tile_id in tiles]
    summary = catalog["summary"]
    return PilotScene(
        label=scene.label,
        tiles=tiles,
        tile_sequence=sequence,
        route=route,
        gaussian_count=int(summary.get("viewerGaussianCount", 0)),
        splat_bytes=int(summary.get("viewerSplatBytes", summary.get("splatBytes", 0))),
    )


# --------------------------------------------------------------------------- base map


def render_base_map(scene: PilotScene) -> tuple[Image.Image, dict]:
    """True orthographic top-down gsplat render of the whole pilot scene (GPU)."""
    import torch
    from gsplat import rasterization

    raw = np.concatenate([np.fromfile(t.splat_path, dtype=SPLAT_DTYPE) for t in scene.tiles.values()])
    pos = raw["pos"].astype(np.float32)
    rgba = raw["rgba"].astype(np.float32) / 255.0
    scales_np = raw["scale"].astype(np.float32)
    scale_max = scales_np.max(axis=1)
    keep = (scale_max <= np.percentile(scale_max, BASE_MAP_SCALE_PERCENTILE)) & (rgba[:, 3] >= BASE_MAP_MIN_OPACITY)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    means = torch.from_numpy(pos[keep]).to(device)
    scales = torch.from_numpy(scales_np[keep]).to(device)
    quats_np = (raw["rot"][keep].astype(np.float32) - 128.0) / 128.0
    quats_np /= np.linalg.norm(quats_np, axis=1, keepdims=True).clip(1e-8)
    quats = torch.from_numpy(quats_np).to(device)
    opacities = torch.from_numpy(rgba[keep, 3]).to(device)
    colors = torch.from_numpy(rgba[keep, :3]).to(device)

    min_x, max_x = float(pos[:, 0].min()), float(pos[:, 0].max())
    min_y, max_y = float(pos[:, 1].min()), float(pos[:, 1].max())
    width = int((max_x - min_x) / BASE_MAP_RESOLUTION_M)
    height = int((max_y - min_y) / BASE_MAP_RESOLUTION_M)

    # camera above the scene looking along -z (splat z is height); image x = +x (east),
    # image y = -y so north (+y) points up
    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    eye = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 60.0])
    viewmat = np.eye(4)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ eye
    viewmats = torch.from_numpy(viewmat.astype(np.float32)).to(device)[None]
    focal = 1.0 / BASE_MAP_RESOLUTION_M  # ortho: pixels per world meter
    intrinsics = torch.tensor(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=torch.float32, device=device
    )[None]
    image, _alpha, _meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width,
        height,
        camera_model="ortho",
        near_plane=0.01,
        far_plane=500.0,
    )
    array = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    base = Image.fromarray(array)
    base = ImageEnhance.Brightness(base).enhance(1.3)
    base = ImageEnhance.Contrast(base).enhance(1.08)
    base = ImageEnhance.Color(base).enhance(1.15)
    meta = {
        "minX": min_x,
        "maxX": max_x,
        "minY": min_y,
        "maxY": max_y,
        "resolution": BASE_MAP_RESOLUTION_M,
        "scalePercentile": BASE_MAP_SCALE_PERCENTILE,
        "minOpacity": BASE_MAP_MIN_OPACITY,
        "gaussiansRendered": int(keep.sum()),
    }
    return base, meta


def load_base_map() -> tuple[Image.Image, dict]:
    if not BASE_MAP_OUTPUT.is_file() or not BASE_MAP_META_OUTPUT.is_file():
        raise FileNotFoundError(
            f"Base map not found ({BASE_MAP_OUTPUT}). Run with --render-base on a GPU machine first."
        )
    return Image.open(BASE_MAP_OUTPUT).convert("RGB"), json.loads(BASE_MAP_META_OUTPUT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- compose


class MapTransform:
    """World (x, y) meters -> map panel pixels (north up, uniform scale)."""

    def __init__(self, meta: dict, panel_size: tuple[int, int]):
        self.min_x, self.max_x = meta["minX"], meta["maxX"]
        self.min_y, self.max_y = meta["minY"], meta["maxY"]
        span_x = self.max_x - self.min_x
        span_y = self.max_y - self.min_y
        self.scale = min(panel_size[0] / span_x, panel_size[1] / span_y)
        self.offset = (
            (panel_size[0] - span_x * self.scale) / 2.0,
            (panel_size[1] - span_y * self.scale) / 2.0,
        )

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        u = self.offset[0] + (x - self.min_x) * self.scale
        v = self.offset[1] + (self.max_y - y) * self.scale
        return u, v

    def rect_px(self, bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        min_x, max_x, min_y, max_y = bounds
        u0, v0 = self.to_px(min_x, max_y)
        u1, v1 = self.to_px(max_x, min_y)
        return int(round(u0)), int(round(v0)), int(round(u1)), int(round(v1))


def _tile_center(scene: PilotScene, tile_id: str) -> tuple[float, float]:
    b = scene.tiles[tile_id].bounds
    return (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0


def _continuous_tile_chain(scene: PilotScene) -> list[str]:
    """Order the visited tiles into a continuous drive (adjacent cells only).

    The shipped route JSON lists tile centers in grid order, which teleports
    between columns; the camera obviously drove a connected path, so rebuild
    it from tile adjacency, starting at the northernmost endpoint.
    """
    ids = list(scene.tile_sequence)
    if len(ids) <= 2:
        return ids
    centers = {tile_id: _tile_center(scene, tile_id) for tile_id in ids}
    tile_step = max(
        scene.tiles[ids[0]].bounds[1] - scene.tiles[ids[0]].bounds[0],
        scene.tiles[ids[0]].bounds[3] - scene.tiles[ids[0]].bounds[2],
    )

    def adjacent(a: str, b: str) -> bool:
        ax, ay = centers[a]
        bx, by = centers[b]
        return abs(ax - bx) + abs(ay - by) <= tile_step * 1.05

    neighbours = {t: [u for u in ids if u != t and adjacent(t, u)] for t in ids}
    endpoints = [t for t in ids if len(neighbours[t]) == 1]
    start = max(endpoints or ids, key=lambda t: centers[t][1])
    chain = [start]
    seen = {start}
    while len(chain) < len(ids):
        options = [u for u in neighbours[chain[-1]] if u not in seen]
        if not options:
            remaining = [u for u in ids if u not in seen]
            options = [
                min(
                    remaining,
                    key=lambda u: (
                        (centers[u][0] - centers[chain[-1]][0]) ** 2 + (centers[u][1] - centers[chain[-1]][1]) ** 2
                    ),
                )
            ]
        chain.append(options[0])
        seen.add(options[0])
    return chain


def build_camera_trajectory(scene: PilotScene, *, samples_per_tile: int = 10, radius_m: float = 14.0) -> np.ndarray:
    """Estimate the continuous camera path through the mapped street.

    Resample the continuous tile chain, then pull every sample toward the
    alpha-weighted centroid of the surrounding splat mass — the densest band
    is the driven road — and smooth. Returns (N, 2) world x/y points.
    """
    chain = _continuous_tile_chain(scene)
    anchors = np.array([_tile_center(scene, tile_id) for tile_id in chain], dtype=np.float64)

    seg = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    count = max(samples_per_tile * len(chain), 16)
    targets = np.linspace(0.0, total, count)
    traj = np.stack([np.interp(targets, cum, anchors[:, 0]), np.interp(targets, cum, anchors[:, 1])], axis=1)

    positions = []
    weights = []
    for tile in scene.tiles.values():
        raw = np.fromfile(tile.splat_path, dtype=SPLAT_DTYPE)
        positions.append(raw["pos"][:, :2].astype(np.float64))
        weights.append(raw["rgba"][:, 3].astype(np.float64) / 255.0)
    points = np.concatenate(positions)
    alpha = np.concatenate(weights)

    for _ in range(2):
        refined = traj.copy()
        for index, sample in enumerate(traj):
            d2 = ((points - sample) ** 2).sum(axis=1)
            mask = d2 <= radius_m * radius_m
            if mask.sum() < 50:
                continue
            w = alpha[mask]
            centroid = (points[mask] * w[:, None]).sum(axis=0) / w.sum()
            refined[index] = 0.45 * sample + 0.55 * centroid
        traj = refined
    kernel = np.ones(5) / 5.0
    for axis in range(2):
        padded = np.concatenate([np.repeat(traj[0, axis], 2), traj[:, axis], np.repeat(traj[-1, axis], 2)])
        traj[:, axis] = np.convolve(padded, kernel, mode="valid")
    return traj


def _arc_lengths(traj: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _camera_pose(traj: np.ndarray, progress: float) -> tuple[float, float, float]:
    """Camera (x, y, px-space heading) at eased arc-length progress 0..1."""
    cum = _arc_lengths(traj)
    s = progress * cum[-1]
    x = float(np.interp(s, cum, traj[:, 0]))
    y = float(np.interp(s, cum, traj[:, 1]))
    ahead = min(s + 2.0, cum[-1])
    behind = max(s - 2.0, 0.0)
    dx = float(np.interp(ahead, cum, traj[:, 0]) - np.interp(behind, cum, traj[:, 0]))
    dy = float(np.interp(ahead, cum, traj[:, 1]) - np.interp(behind, cum, traj[:, 1]))
    heading = math.atan2(-dy, dx)  # px y grows downward
    return x, y, heading


def _tile_at(scene: PilotScene, x: float, y: float) -> str | None:
    for tile_id, tile in scene.tiles.items():
        min_x, max_x, min_y, max_y = tile.bounds
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return tile_id
    return None


def _tile_states(
    scene: PilotScene, traj: np.ndarray, progress: float, *, lookahead_m: float = 35.0
) -> tuple[dict[str, str], tuple[int, int, int]]:
    """Residency derived from the camera trajectory: tiles the camera has
    entered so far (latest RESIDENT_LIMIT stay resident) plus the next tiles
    the path enters within lookahead_m as preload requests."""
    cum = _arc_lengths(traj)
    s = progress * cum[-1]
    visited: list[str] = []
    for sample, dist in zip(traj, cum, strict=True):
        if dist > s:
            break
        tile_id = _tile_at(scene, sample[0], sample[1])
        if tile_id is not None and tile_id not in visited:
            visited.append(tile_id)
    preload: list[str] = []
    for sample, dist in zip(traj, cum, strict=True):
        if dist <= s or dist > s + lookahead_m:
            continue
        tile_id = _tile_at(scene, sample[0], sample[1])
        if tile_id is not None and tile_id not in visited and tile_id not in preload:
            preload.append(tile_id)
        if len(preload) >= PRELOAD_LIMIT:
            break
    resident = visited[-RESIDENT_LIMIT:]
    evicted = visited[:-RESIDENT_LIMIT] if len(visited) > RESIDENT_LIMIT else []
    state: dict[str, str] = {tile_id: "loaded" for tile_id in evicted}
    state.update({tile_id: "resident" for tile_id in resident})
    state.update({tile_id: "preload" for tile_id in preload})
    return state, (len(resident), len(preload), len(evicted))


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


def _render_map_panel(
    scene: PilotScene,
    base: Image.Image,
    meta: dict,
    panel_size: tuple[int, int],
    traj: np.ndarray,
    progress: float,
    *,
    line_scale: float = 1.0,
) -> Image.Image:
    panel = Image.new("RGB", panel_size, PANEL_BG)
    transform = MapTransform(meta, panel_size)
    span_x = meta["maxX"] - meta["minX"]
    span_y = meta["maxY"] - meta["minY"]
    fitted = base.resize((max(int(span_x * transform.scale), 1), max(int(span_y * transform.scale), 1)), Image.LANCZOS)
    origin = (int(round(transform.offset[0])), int(round(transform.offset[1])))

    # unloaded world: heavily dimmed base map
    panel.paste(ImageEnhance.Brightness(fitted).enhance(0.22), origin)

    state, _counts = _tile_states(scene, traj, progress)
    brightness = {"loaded": 0.72, "preload": 0.5, "resident": 1.0}
    draw = ImageDraw.Draw(panel)
    for tile_id, tile in scene.tiles.items():
        tile_state = state.get(tile_id)
        if tile_state is None:
            continue
        rect = transform.rect_px(tile.bounds)
        crop_box = (rect[0] - origin[0], rect[1] - origin[1], rect[2] - origin[0], rect[3] - origin[1])
        crop_box = (
            max(crop_box[0], 0),
            max(crop_box[1], 0),
            min(crop_box[2], fitted.width),
            min(crop_box[3], fitted.height),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        chip = ImageEnhance.Brightness(fitted.crop(crop_box)).enhance(brightness[tile_state])
        panel.paste(chip, (crop_box[0] + origin[0], crop_box[1] + origin[1]))

    outline = {
        "resident": (RESIDENT_COLOR, max(2, int(2 * line_scale))),
        "preload": (PRELOAD_COLOR, max(2, int(2 * line_scale))),
        "loaded": (LOADED_COLOR, 1),
    }
    for tile_id, tile in scene.tiles.items():
        tile_state = state.get(tile_id)
        if tile_state is None:
            continue
        color, width = outline[tile_state]
        draw.rectangle(transform.rect_px(tile.bounds), outline=color, width=width)

    cum = _arc_lengths(traj)
    s = progress * cum[-1]
    rx, ry, heading = _camera_pose(traj, progress)
    robot = transform.to_px(rx, ry)
    traveled = [transform.to_px(p[0], p[1]) for p, d in zip(traj, cum, strict=True) if d <= s]
    traveled.append(robot)
    if len(traveled) >= 2:
        draw.line(traveled, fill=ROUTE_DONE, width=max(3, int(3 * line_scale)), joint="curve")

    size = 8.0 * line_scale
    tip = []
    for angle, radius in ((0.0, 1.6), (2.5, 1.0), (math.pi, 0.35), (-2.5, 1.0)):
        tip.append(
            (
                robot[0] + math.cos(heading + angle) * size * radius,
                robot[1] + math.sin(heading + angle) * size * radius,
            )
        )
    draw.polygon(tip, fill=ROBOT_COLOR, outline=(20, 28, 38))
    ring = 14.0 * line_scale
    draw.ellipse(
        [robot[0] - ring, robot[1] - ring, robot[0] + ring, robot[1] + ring],
        outline=RESIDENT_COLOR,
        width=max(1, int(line_scale)),
    )
    return panel


def _legend_entries() -> list[tuple[tuple[int, int, int], str]]:
    return [
        (RESIDENT_COLOR, "resident 3DGS tiles"),
        (PRELOAD_COLOR, "preload request"),
        (LOADED_COLOR, "loaded (evicted) tiles"),
        (ROUTE_DONE, "camera trajectory"),
    ]


def render_frame(
    scene: PilotScene,
    base: Image.Image,
    meta: dict,
    traj: np.ndarray,
    progress: float,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    big = width >= 1100
    title_font = _load_font(34 if big else 25)
    sub_font = _load_font(17 if big else 13)
    small_font = _load_font(15 if big else 12)

    header_h = 78 if big else 58
    footer_h = 50 if big else 36
    side_w = 312 if big else 246
    margin = 16
    panel_box = (margin, header_h, width - side_w, height - footer_h)
    panel_size = (panel_box[2] - panel_box[0], panel_box[3] - panel_box[1])

    panel = _render_map_panel(scene, base, meta, panel_size, traj, progress, line_scale=1.4 if big else 1.0)
    image.paste(panel, (panel_box[0], panel_box[1]))
    draw.rectangle((panel_box[0] - 1, panel_box[1] - 1, panel_box[2], panel_box[3]), outline=GRID_COLOR)

    _state, (resident, preload, loaded) = _tile_states(scene, traj, progress)
    draw.text((margin, 12), "Real rosbag2 -> 3DGS dynamic map (Istanbul Bag6)", font=title_font, fill=TEXT)
    total_m = float(_arc_lengths(traj)[-1])
    draw.text(
        (margin, (50 if big else 38)),
        f"{len(scene.tiles)} streamed .splat tiles - camera at {progress * total_m:.0f}/{total_m:.0f} m - "
        f"resident {resident} / preload {preload} / evicted {loaded}",
        font=sub_font,
        fill=TEXT_DIM,
    )

    legend_x = width - side_w + 18
    draw.text((legend_x, header_h + 2), "tile residency", font=sub_font, fill=TEXT)
    y = header_h + (32 if big else 26)
    chip = 16 if big else 12
    for color, label in _legend_entries():
        draw.rectangle([legend_x, y + 2, legend_x + chip, y + 2 + chip], fill=color)
        draw.text((legend_x + chip + 9, y), label, font=small_font, fill=TEXT)
        y += 28 if big else 22

    stats_y = y + (18 if big else 12)
    span_x = meta["maxX"] - meta["minX"]
    span_y = meta["maxY"] - meta["minY"]
    for line in (
        "map_loader live state",
        f"footprint {span_x:.0f} x {span_y:.0f} m",
        f"{scene.gaussian_count / 1e3:.0f}k tiled gaussians",
        "base: true top-down gsplat",
        "render of the shipped tiles",
        "(GNSS-seeded, 291 frames)",
    ):
        fill = TEXT if line == "map_loader live state" else TEXT_DIM
        draw.text((legend_x, stats_y), line, font=small_font, fill=fill)
        stats_y += 26 if big else 20

    transform = MapTransform(meta, panel_size)
    bar_m = 20.0
    bar_px = bar_m * transform.scale
    bar_x = panel_box[0] + 14
    bar_y = panel_box[3] - 16
    draw.line([(bar_x, bar_y), (bar_x + bar_px, bar_y)], fill=TEXT, width=2)
    for tick in (bar_x, bar_x + bar_px):
        draw.line([(tick, bar_y - 4), (tick, bar_y + 4)], fill=TEXT, width=2)
    draw.text((bar_x, bar_y - 22), f"{bar_m:.0f} m", font=small_font, fill=TEXT)

    draw.text(
        (margin, height - footer_h + (12 if big else 8)),
        "source: istanbul-bag6-pilot-tile-catalog.json - the map is a real top-down render; "
        "the loading window follows the camera trajectory",
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
        scene = load_pilot_scene(proof_scene)
        base, meta = load_base_map()
        traj = build_camera_trajectory(scene)
        for index in range(frames_per_scene):
            progress = index / max(frames_per_scene - 1, 1)
            frames.append(render_frame(scene, base, meta, traj, progress, size))
        if map_material_output is not None:
            material = render_frame(scene, base, meta, traj, 0.6, MAP_MATERIAL_SIZE)
            map_material_output.parent.mkdir(parents=True, exist_ok=True)
            material.save(map_material_output)
            print(f"map material: {map_material_output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [FRAME_DURATION_MS] * len(frames)
    durations[-1] = 1600  # hold the fully-loaded map before looping
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
    parser.add_argument(
        "--render-base",
        action="store_true",
        help="Re-render the orthographic base map (requires GPU + gsplat) before composing",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.render_base:
        scene = load_pilot_scene(MAP_PROOF_SCENES[0])
        base, meta = render_base_map(scene)
        BASE_MAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        base.save(BASE_MAP_OUTPUT)
        BASE_MAP_META_OUTPUT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"base map: {BASE_MAP_OUTPUT} ({base.size[0]}x{base.size[1]})")
    build_map_quality_gif(args.output, map_material_output=args.material_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
