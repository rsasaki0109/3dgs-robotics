"""Large-scale 3DGS chunk planning for COLMAP-style training data."""

from __future__ import annotations

import json
import math
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import numpy as np

from gs_sim2real.preprocess.colmap_ready import require_colmap_sparse_model


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class ColmapImageRecord:
    image_id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    center: tuple[float, float, float]
    metadata_line: str
    points2d_line: str


@dataclass(frozen=True)
class ColmapPointRecord:
    point_id: int
    xyz: tuple[float, float, float]
    line: str


@dataclass(frozen=True)
class LargeScale3DGSOptions:
    data_dir: Path
    output_dir: Path
    tile_size: float = 30.0
    overlap: float = 5.0
    axes: str = "xy"
    min_images: int = 8
    iterations: int = 30000
    config: str | None = None
    export_max_points: int = 400000
    splat_min_opacity: float = 0.02
    splat_max_scale: float | None = 2.0
    splat_max_scale_percentile: float | None = 98.0
    materialize: bool = False
    link_mode: str = "symlink"


@dataclass(frozen=True)
class LargeScale3DGSRunOptions:
    plan_path: Path
    report_path: Path | None = None
    max_chunks: int | None = None
    resume: bool = True
    dry_run: bool = False
    fail_fast: bool = True


@dataclass(frozen=True)
class LargeScale3DGSPreflightOptions:
    data_dir: Path
    output_dir: Path
    axes: str = "xy"
    tile_sizes: tuple[float, ...] = (20.0, 30.0, 50.0)
    overlap: float = 5.0
    min_images: int = 8
    target_images_per_chunk: int = 48
    iterations: int = 30000
    config: str | None = "configs/training_ba.yaml"


@dataclass(frozen=True)
class LargeScale3DGSCatalogOptions:
    plan_path: Path
    output_path: Path | None = None
    run_report_path: Path | None = None
    scene_id: str = "large-scale-3dgs"
    label: str = "Large-scale 3DGS"
    public_root: Path | None = None
    public_url_prefix: str = "/splats"
    link_mode: str = "symlink"
    require_splats: bool = False
    web_app_dir: Path | None = Path("apps/dreamwalker-web")
    site_url: str = "http://localhost:5173/"
    tile_preload: str = "metadata"
    route_path: str | Path | None = None
    route_playback: bool = False
    route_playback_ms: int | None = None
    route_playback_loop: bool = False


@dataclass(frozen=True)
class LargeScale3DGSRouteOptions:
    catalog_path: Path
    output_path: Path | None = None
    label: str | None = None
    description: str | None = None
    fragment_id: str = "residency"
    fragment_label: str = "Residency"
    frame_id: str = "dreamwalker_map"
    asset_label: str | None = None
    zone_map_url: str = "/manifests/robotics-residency.zones.json"
    world_splat_url: str = ""
    collider_mesh_url: str = ""
    default_y: float = 0.0
    order: str = "spiral"
    include_missing_splats: bool = False


@dataclass(frozen=True)
class LargeScale3DGSSmokeDataOptions:
    output_dir: Path
    axes: str = "xz"
    grid_width: int = 3
    grid_height: int = 2
    tile_size: float = 8.0
    images_per_tile: int = 2
    points_per_tile: int = 12
    image_size: int = 48


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _read_data_lines(path: Path) -> list[str]:
    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(value) for value in qvec]
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qx * qz + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qx * qz - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def _camera_center_from_qvec_tvec(qvec: Sequence[float], tvec: Sequence[float]) -> tuple[float, float, float]:
    rotation = _qvec_to_rotmat(qvec)
    translation = np.array([float(value) for value in tvec], dtype=np.float64)
    center = -(rotation.T @ translation)
    return (float(center[0]), float(center[1]), float(center[2]))


def load_colmap_images_text(path: Path) -> list[ColmapImageRecord]:
    """Parse COLMAP text image records while preserving the paired 2D line."""
    raw_lines = [
        line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
    ]
    records: list[ColmapImageRecord] = []
    index = 0

    while index < len(raw_lines):
        metadata_line = raw_lines[index].strip()
        if not metadata_line:
            index += 1
            continue

        parts = metadata_line.split()
        if len(parts) < 10:
            index += 1
            continue

        image_id = int(parts[0])
        qvec = tuple(float(value) for value in parts[1:5])
        tvec = tuple(float(value) for value in parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        points2d_line = raw_lines[index + 1].strip() if index + 1 < len(raw_lines) else ""

        records.append(
            ColmapImageRecord(
                image_id=image_id,
                camera_id=camera_id,
                name=name,
                qvec=qvec,  # type: ignore[arg-type]
                tvec=tvec,  # type: ignore[arg-type]
                center=_camera_center_from_qvec_tvec(qvec, tvec),
                metadata_line=metadata_line,
                points2d_line=points2d_line,
            )
        )
        index += 2

    return records


def load_colmap_points_text(path: Path) -> list[ColmapPointRecord]:
    """Parse enough of points3D.txt for spatial filtering."""
    records: list[ColmapPointRecord] = []

    if not path.exists():
        return records

    for line in _read_data_lines(path):
        parts = line.split()
        if len(parts) < 7:
            continue
        records.append(
            ColmapPointRecord(
                point_id=int(parts[0]),
                xyz=(float(parts[1]), float(parts[2]), float(parts[3])),
                line=line,
            )
        )

    return records


def _validate_axes(axes: str) -> tuple[str, str]:
    normalized = axes.strip().lower()
    if len(normalized) != 2 or normalized[0] == normalized[1]:
        raise ValueError("--axes must contain two distinct axes, e.g. xy, xz, yz")
    if normalized[0] not in _AXIS_INDEX or normalized[1] not in _AXIS_INDEX:
        raise ValueError("--axes must contain only x, y, z")
    return normalized[0], normalized[1]


def _axis_value(position: Sequence[float], axis: str) -> float:
    return float(position[_AXIS_INDEX[axis]])


def _bounds_for_records(records: Sequence[ColmapImageRecord], axes: tuple[str, str]) -> dict[str, float]:
    a_values = [_axis_value(record.center, axes[0]) for record in records]
    b_values = [_axis_value(record.center, axes[1]) for record in records]
    return {
        f"min{axes[0].upper()}": float(min(a_values)),
        f"max{axes[0].upper()}": float(max(a_values)),
        f"min{axes[1].upper()}": float(min(b_values)),
        f"max{axes[1].upper()}": float(max(b_values)),
    }


def _in_bounds(position: Sequence[float], bounds: dict[str, float], axes: tuple[str, str]) -> bool:
    a = _axis_value(position, axes[0])
    b = _axis_value(position, axes[1])
    return (
        bounds[f"min{axes[0].upper()}"] <= a <= bounds[f"max{axes[0].upper()}"]
        and bounds[f"min{axes[1].upper()}"] <= b <= bounds[f"max{axes[1].upper()}"]
    )


def _format_command(parts: Iterable[str | Path | int | float]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def parse_large_scale_3dgs_tile_sizes(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse a comma-separated tile-size list for large-scale planning."""
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = [str(part).strip() for part in value]

    tile_sizes = tuple(float(part) for part in raw_values)
    if not tile_sizes:
        raise ValueError("--tile-sizes must include at least one positive value")
    if any(tile_size <= 0 for tile_size in tile_sizes):
        raise ValueError("--tile-sizes must contain only positive values")
    return tile_sizes


def _split_command(command: str) -> list[str]:
    return shlex.split(command)


def _default_command_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    command = list(args)
    try:
        return subprocess.run(command, check=False)
    except FileNotFoundError:
        if command and command[0] == "gs-mapper":
            return subprocess.run([sys.executable, "-m", "gs_sim2real.cli", *command[1:]], check=False)
        raise


def _build_train_command(
    *,
    chunk_data_dir: Path,
    train_dir: Path,
    iterations: int,
    config: str | None,
) -> str:
    parts: list[str | Path | int] = [
        "gs-mapper",
        "train",
        "--data",
        chunk_data_dir,
        "--output",
        train_dir,
        "--method",
        "gsplat",
        "--iterations",
        iterations,
    ]
    if config:
        parts.extend(["--config", config])
    return _format_command(parts)


def _build_export_command(
    *,
    train_dir: Path,
    splat_path: Path,
    export_max_points: int,
    splat_min_opacity: float,
    splat_max_scale: float | None,
    splat_max_scale_percentile: float | None,
) -> str:
    parts: list[str | Path | int | float] = [
        "gs-mapper",
        "export",
        "--model",
        train_dir / "point_cloud.ply",
        "--format",
        "splat",
        "--output",
        splat_path,
        "--max-points",
        export_max_points,
        "--splat-min-opacity",
        splat_min_opacity,
    ]
    if splat_max_scale is not None:
        parts.extend(["--splat-max-scale", splat_max_scale])
    if splat_max_scale_percentile is not None:
        parts.extend(["--splat-max-scale-percentile", splat_max_scale_percentile])
    return _format_command(parts)


def _find_images_root(data_dir: Path) -> Path | None:
    for candidate in (data_dir / "undistorted" / "images", data_dir / "images"):
        if candidate.exists():
            return candidate
    return None


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "none" or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode != "symlink":
        raise ValueError("--link-mode must be symlink, copy, or none")
    dst.symlink_to(src.resolve())


def _position_from_tile_axes(
    axes: tuple[str, str], a_value: float, b_value: float, fallback_z: float
) -> tuple[float, float, float]:
    position = [0.0, 0.0, float(fallback_z)]
    position[_AXIS_INDEX[axes[0]]] = float(a_value)
    position[_AXIS_INDEX[axes[1]]] = float(b_value)
    return (position[0], position[1], position[2])


def _write_ppm(path: Path, *, width: int, height: int, base_rgb: tuple[int, int, int]) -> None:
    data = bytearray()
    for y in range(height):
        for x in range(width):
            shade = int(30 * (x / max(1, width - 1)) + 24 * (y / max(1, height - 1)))
            data.extend(min(255, channel + shade) for channel in base_rgb)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(data))


def write_large_scale_3dgs_smoke_data(options: LargeScale3DGSSmokeDataOptions) -> dict[str, Any]:
    """Write a deterministic multi-tile COLMAP text fixture for large-scale 3DGS smoke runs."""
    if options.grid_width < 1 or options.grid_height < 1:
        raise ValueError("--grid-width and --grid-height must be >= 1")
    if options.tile_size <= 0:
        raise ValueError("--tile-size must be > 0")
    if options.images_per_tile < 1:
        raise ValueError("--images-per-tile must be >= 1")
    if options.points_per_tile < 1:
        raise ValueError("--points-per-tile must be >= 1")
    if options.image_size < 8:
        raise ValueError("--image-size must be >= 8")

    axes = _validate_axes(options.axes)
    output_dir = Path(options.output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    width = int(options.image_size)
    height = int(options.image_size)
    focal = float(options.image_size) * 0.9
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list for deterministic large-scale 3DGS smoke data\n"
        f"1 PINHOLE {width} {height} {focal:.6f} {focal:.6f} {width / 2:.6f} {height / 2:.6f}\n",
        encoding="utf-8",
    )

    spacing = float(options.tile_size) * 1.1
    origin = float(options.tile_size) * 0.5
    view_span = min(float(options.tile_size) * 0.16, 1.0)
    point_span = min(float(options.tile_size) * 0.24, 1.4)
    depth_offset = min(float(options.tile_size) * 0.35, 3.2)
    image_lines = ["# Image list for deterministic large-scale 3DGS smoke data"]
    point_lines = ["# 3D point list for deterministic large-scale 3DGS smoke data"]
    tile_records: list[dict[str, Any]] = []
    image_id = 1
    point_id = 1

    for tile_a in range(options.grid_width):
        for tile_b in range(options.grid_height):
            center_a = origin + tile_a * spacing
            center_b = origin + tile_b * spacing
            tile_image_names: list[str] = []
            tile_point_ids: list[int] = []
            base_rgb = (
                48 + (tile_a * 53 + tile_b * 19) % 160,
                58 + (tile_a * 31 + tile_b * 61) % 150,
                70 + (tile_a * 23 + tile_b * 47) % 140,
            )

            for view_index in range(options.images_per_tile):
                offset_ratio = (view_index + 0.5) / options.images_per_tile - 0.5
                camera_a = center_a + offset_ratio * view_span
                camera_b = center_b - offset_ratio * view_span
                center = _position_from_tile_axes(axes, camera_a, camera_b, fallback_z=0.0)
                tvec = (-center[0], -center[1], -center[2])
                image_name = f"tile_{axes[0]}{tile_a:03d}_{axes[1]}{tile_b:03d}_view{view_index:03d}.ppm"
                _write_ppm(images_dir / image_name, width=width, height=height, base_rgb=base_rgb)
                image_lines.append(f"{image_id} 1 0 0 0 {tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f} 1 {image_name}")
                image_lines.append("")
                tile_image_names.append(image_name)
                image_id += 1

            side = max(1, math.ceil(math.sqrt(options.points_per_tile)))
            for point_index in range(options.points_per_tile):
                u = (point_index % side + 0.5) / side
                v = (point_index // side + 0.5) / side
                point_a = center_a + u * point_span
                point_b = center_b + v * point_span
                if axes[0] == "z":
                    point_a += depth_offset
                if axes[1] == "z":
                    point_b += depth_offset
                point = _position_from_tile_axes(axes, point_a, point_b, fallback_z=depth_offset)
                red = min(255, max(0, base_rgb[0] + int((u - 0.5) * 40)))
                green = min(255, max(0, base_rgb[1] + int((v - 0.5) * 40)))
                blue = min(255, max(0, base_rgb[2] + (point_index * 7) % 50))
                point_lines.append(f"{point_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {red} {green} {blue} 0.10")
                tile_point_ids.append(point_id)
                point_id += 1

            tile_records.append(
                {
                    "tileIndex": {axes[0]: tile_a, axes[1]: tile_b},
                    "center": {
                        axes[0]: round(center_a, 6),
                        axes[1]: round(center_b, 6),
                    },
                    "imageNames": tile_image_names,
                    "pointIds": tile_point_ids,
                }
            )

    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")

    manifest = {
        "version": 1,
        "type": "large-scale-3dgs-smoke-data",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataDir": str(output_dir),
        "sparseDir": str(sparse_dir),
        "imagesDir": str(images_dir),
        "axes": "".join(axes),
        "tileSize": float(options.tile_size),
        "grid": {
            "width": int(options.grid_width),
            "height": int(options.grid_height),
        },
        "summary": {
            "tileCount": int(options.grid_width * options.grid_height),
            "imageCount": int(options.grid_width * options.grid_height * options.images_per_tile),
            "points3DCount": int(options.grid_width * options.grid_height * options.points_per_tile),
            "imagesPerTile": int(options.images_per_tile),
            "pointsPerTile": int(options.points_per_tile),
            "imageSize": int(options.image_size),
        },
        "tiles": tile_records,
    }
    (output_dir / "large_scale_3dgs_smoke_data.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def format_large_scale_3dgs_smoke_data_text(manifest: dict[str, Any]) -> str:
    data_dir = Path(manifest["dataDir"])
    suggested_output = data_dir.parent / f"{data_dir.name}_large_scale"
    summary = manifest["summary"]
    min_images = max(1, int(summary["imagesPerTile"]))
    plan_command = _format_command(
        [
            "gs-mapper",
            "large-scale-3dgs-plan",
            "--data",
            data_dir,
            "--output",
            suggested_output,
            "--tile-size",
            manifest["tileSize"],
            "--overlap",
            0,
            "--axes",
            manifest["axes"],
            "--min-images",
            min_images,
            "--iterations",
            5,
            "--materialize",
            "--link-mode",
            "copy",
        ]
    )
    run_command = _format_command(
        [
            "gs-mapper",
            "large-scale-3dgs-run",
            "--plan",
            suggested_output / "large_scale_3dgs_plan.json",
        ]
    )

    return "\n".join(
        [
            "Large-scale 3DGS smoke data",
            f"  data: {manifest['dataDir']}",
            f"  sparse: {manifest['sparseDir']}",
            f"  tiles: {summary['tileCount']} ({manifest['grid']['width']}x{manifest['grid']['height']})",
            f"  images: {summary['imageCount']} / points: {summary['points3DCount']}",
            f"  axes: {manifest['axes']} tile_size={manifest['tileSize']}",
            f"  manifest: {data_dir / 'large_scale_3dgs_smoke_data.json'}",
            f"  next plan: {plan_command}",
            f"  next run: {run_command}",
        ]
    )


def _slugify(value: str, fallback: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(value).strip())
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or fallback


def _join_public_url(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if str(part).strip("/")]
    return "/" + "/".join(cleaned)


def _write_chunk_sparse(
    chunk_dir: Path,
    cameras_txt: Path,
    image_records: Sequence[ColmapImageRecord],
    point_records: Sequence[ColmapPointRecord],
) -> None:
    sparse_dir = chunk_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_ids = {record.camera_id for record in image_records}
    camera_lines = [
        line for line in _read_data_lines(cameras_txt) if line.split() and int(line.split()[0]) in camera_ids
    ]
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list for one large-scale 3DGS chunk\n" + "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )

    image_lines: list[str] = ["# Image list for one large-scale 3DGS chunk"]
    for record in image_records:
        image_lines.append(record.metadata_line)
        image_lines.append(record.points2d_line)
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    point_lines = ["# 3D point list for one large-scale 3DGS chunk"]
    point_lines.extend(record.line for record in point_records)
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def _materialize_chunk_data(
    *,
    chunk_dir: Path,
    data_dir: Path,
    sparse_dir: Path,
    image_records: Sequence[ColmapImageRecord],
    point_records: Sequence[ColmapPointRecord],
    link_mode: str,
) -> None:
    images_root = _find_images_root(data_dir)
    _write_chunk_sparse(chunk_dir, sparse_dir / "cameras.txt", image_records, point_records)

    if images_root is None:
        return

    for record in image_records:
        source_image = images_root / record.name
        _link_or_copy(source_image, chunk_dir / "images" / record.name, link_mode)

        source_depth = data_dir / "depth" / str(Path(record.name).with_suffix(".npy"))
        _link_or_copy(source_depth, chunk_dir / "depth" / str(Path(record.name).with_suffix(".npy")), link_mode)


def build_large_scale_3dgs_plan(options: LargeScale3DGSOptions) -> dict[str, Any]:
    """Build a tile-based training/export plan for large COLMAP sparse inputs."""
    if options.tile_size <= 0:
        raise ValueError("--tile-size must be > 0")
    if options.overlap < 0:
        raise ValueError("--overlap must be >= 0")
    if options.min_images < 1:
        raise ValueError("--min-images must be >= 1")

    data_dir = Path(options.data_dir)
    output_dir = Path(options.output_dir)
    axes = _validate_axes(options.axes)
    sparse_dir = require_colmap_sparse_model(data_dir)

    if not (sparse_dir / "cameras.txt").exists():
        raise ValueError("large-scale 3DGS planning currently requires COLMAP text sparse files")

    image_records = load_colmap_images_text(sparse_dir / "images.txt")
    point_records = load_colmap_points_text(sparse_dir / "points3D.txt")
    if not image_records:
        raise ValueError("No registered images found in COLMAP images.txt")

    world_bounds = _bounds_for_records(image_records, axes)
    axis_a, axis_b = axes
    min_a = world_bounds[f"min{axis_a.upper()}"]
    min_b = world_bounds[f"min{axis_b.upper()}"]
    max_a = world_bounds[f"max{axis_a.upper()}"]
    max_b = world_bounds[f"max{axis_b.upper()}"]
    num_a = max(1, math.floor((max_a - min_a) / options.tile_size) + 1)
    num_b = max(1, math.floor((max_b - min_b) / options.tile_size) + 1)
    chunks: list[dict[str, Any]] = []

    for tile_a in range(num_a):
        for tile_b in range(num_b):
            core_bounds = {
                f"min{axis_a.upper()}": min_a + tile_a * options.tile_size,
                f"max{axis_a.upper()}": min_a + (tile_a + 1) * options.tile_size,
                f"min{axis_b.upper()}": min_b + tile_b * options.tile_size,
                f"max{axis_b.upper()}": min_b + (tile_b + 1) * options.tile_size,
            }
            expanded_bounds = {
                f"min{axis_a.upper()}": core_bounds[f"min{axis_a.upper()}"] - options.overlap,
                f"max{axis_a.upper()}": core_bounds[f"max{axis_a.upper()}"] + options.overlap,
                f"min{axis_b.upper()}": core_bounds[f"min{axis_b.upper()}"] - options.overlap,
                f"max{axis_b.upper()}": core_bounds[f"max{axis_b.upper()}"] + options.overlap,
            }
            core_images = [record for record in image_records if _in_bounds(record.center, core_bounds, axes)]
            chunk_images = [record for record in image_records if _in_bounds(record.center, expanded_bounds, axes)]
            if not core_images:
                continue
            chunk_points = [record for record in point_records if _in_bounds(record.xyz, expanded_bounds, axes)]
            chunk_id = f"tile_{axis_a}{tile_a:03d}_{axis_b}{tile_b:03d}"
            chunk_data_dir = output_dir / "chunks" / chunk_id
            train_dir = output_dir / "train" / chunk_id
            splat_path = output_dir / "splats" / f"{chunk_id}.splat"
            trainable = len(core_images) >= options.min_images
            train_command = _build_train_command(
                chunk_data_dir=chunk_data_dir,
                train_dir=train_dir,
                iterations=options.iterations,
                config=options.config,
            )
            export_command = _build_export_command(
                train_dir=train_dir,
                splat_path=splat_path,
                export_max_points=options.export_max_points,
                splat_min_opacity=options.splat_min_opacity,
                splat_max_scale=options.splat_max_scale,
                splat_max_scale_percentile=options.splat_max_scale_percentile,
            )

            if options.materialize:
                _materialize_chunk_data(
                    chunk_dir=chunk_data_dir,
                    data_dir=data_dir,
                    sparse_dir=sparse_dir,
                    image_records=chunk_images,
                    point_records=chunk_points,
                    link_mode=options.link_mode,
                )

            chunks.append(
                {
                    "id": chunk_id,
                    "status": "ready" if trainable else "too-few-images",
                    "tileIndex": {axis_a: tile_a, axis_b: tile_b},
                    "axes": "".join(axes),
                    "coreBounds": core_bounds,
                    "expandedBounds": expanded_bounds,
                    "coreImageCount": len(core_images),
                    "imageCount": len(chunk_images),
                    "pointCount": len(chunk_points),
                    "cameraCount": len({record.camera_id for record in chunk_images}),
                    "dataDir": str(chunk_data_dir),
                    "trainOutputDir": str(train_dir),
                    "splatOutput": str(splat_path),
                    "trainCommand": train_command if trainable else "",
                    "exportCommand": export_command if trainable else "",
                    "imageNames": [record.name for record in chunk_images],
                }
            )

    ready_chunks = [chunk for chunk in chunks if chunk["status"] == "ready"]
    return {
        "version": 1,
        "type": "large-scale-3dgs-plan",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataDir": str(data_dir),
        "sparseDir": str(sparse_dir),
        "outputDir": str(output_dir),
        "materialized": bool(options.materialize),
        "tiling": {
            "strategy": "camera-center-grid",
            "axes": "".join(axes),
            "tileSize": float(options.tile_size),
            "overlap": float(options.overlap),
            "minImages": int(options.min_images),
            "worldBounds": world_bounds,
        },
        "training": {
            "method": "gsplat",
            "iterations": int(options.iterations),
            "config": options.config,
            "exportMaxPoints": int(options.export_max_points),
            "splatMinOpacity": float(options.splat_min_opacity),
            "splatMaxScale": options.splat_max_scale,
            "splatMaxScalePercentile": options.splat_max_scale_percentile,
        },
        "summary": {
            "registeredImageCount": len(image_records),
            "points3DCount": len(point_records),
            "chunkCount": len(chunks),
            "readyChunkCount": len(ready_chunks),
            "tooFewImageChunkCount": len(chunks) - len(ready_chunks),
        },
        "chunks": chunks,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    rank = max(0.0, min(1.0, percentile / 100.0)) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_stats(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    numeric_values = [float(value) for value in values]
    return {
        "min": round(min(numeric_values), 3),
        "p50": round(_percentile(numeric_values, 50.0), 3),
        "p90": round(_percentile(numeric_values, 90.0), 3),
        "max": round(max(numeric_values), 3),
    }


def _image_size_index(data_dir: Path, image_names: Iterable[str]) -> tuple[str, dict[str, int]]:
    image_root = _find_images_root(data_dir)
    if image_root is None:
        return "", {}

    sizes: dict[str, int] = {}
    for image_name in sorted(set(image_names)):
        image_path = image_root / image_name
        sizes[image_name] = image_path.stat().st_size if image_path.exists() else 0
    return str(image_root), sizes


def _format_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _preflight_candidate_from_plan(
    plan: dict[str, Any],
    *,
    tile_size: float,
    overlap: float,
    image_sizes: dict[str, int],
    target_images_per_chunk: int,
) -> dict[str, Any]:
    chunks = list(plan.get("chunks", []))
    chunk_image_bytes = [
        sum(image_sizes.get(image_name, 0) for image_name in set(chunk.get("imageNames", []))) for chunk in chunks
    ]
    ready_count = int(plan["summary"]["readyChunkCount"])
    chunk_count = int(plan["summary"]["chunkCount"])
    ready_ratio = ready_count / chunk_count if chunk_count else 0.0
    core_stats = _numeric_stats([chunk["coreImageCount"] for chunk in chunks])
    image_stats = _numeric_stats([chunk["imageCount"] for chunk in chunks])
    point_stats = _numeric_stats([chunk["pointCount"] for chunk in chunks])
    byte_stats = _numeric_stats(chunk_image_bytes)
    target_delta = abs(core_stats["p50"] - float(target_images_per_chunk))

    return {
        "tileSize": float(tile_size),
        "overlap": float(overlap),
        "chunkCount": chunk_count,
        "readyChunkCount": ready_count,
        "tooFewImageChunkCount": int(plan["summary"]["tooFewImageChunkCount"]),
        "readyRatio": round(ready_ratio, 3),
        "coreImagesPerChunk": core_stats,
        "imagesPerChunk": image_stats,
        "pointsPerChunk": point_stats,
        "sourceImageBytesPerChunk": byte_stats,
        "targetImagesPerChunkDelta": round(target_delta, 3),
    }


def _recommend_preflight_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No large-scale 3DGS preflight candidates were generated")

    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate["readyRatio"]),
            float(candidate["targetImagesPerChunkDelta"]),
            int(candidate["chunkCount"]),
            float(candidate["tileSize"]),
        ),
    )


def _build_preflight_plan_command(options: LargeScale3DGSPreflightOptions, recommendation: dict[str, Any]) -> str:
    parts: list[str | Path | int | float] = [
        "gs-mapper",
        "large-scale-3dgs-plan",
        "--data",
        options.data_dir,
        "--output",
        options.output_dir,
        "--tile-size",
        recommendation["tileSize"],
        "--overlap",
        recommendation["overlap"],
        "--axes",
        options.axes,
        "--min-images",
        options.min_images,
        "--iterations",
        options.iterations,
    ]
    if options.config:
        parts.extend(["--config", options.config])
    parts.extend(["--materialize"])
    return _format_command(parts)


def build_large_scale_3dgs_preflight(options: LargeScale3DGSPreflightOptions) -> dict[str, Any]:
    """Inspect a COLMAP scene and recommend a large-scale 3DGS tiling setup."""
    if options.overlap < 0:
        raise ValueError("--overlap must be >= 0")
    if options.min_images < 1:
        raise ValueError("--min-images must be >= 1")
    if options.target_images_per_chunk < 1:
        raise ValueError("--target-images-per-chunk must be >= 1")

    tile_sizes = parse_large_scale_3dgs_tile_sizes(options.tile_sizes)
    data_dir = Path(options.data_dir)
    output_dir = Path(options.output_dir)
    axes = _validate_axes(options.axes)
    sparse_dir = require_colmap_sparse_model(data_dir)
    image_records = load_colmap_images_text(sparse_dir / "images.txt")
    point_records = load_colmap_points_text(sparse_dir / "points3D.txt")
    if not image_records:
        raise ValueError("No registered images found in COLMAP images.txt")

    image_root, image_sizes = _image_size_index(data_dir, (record.name for record in image_records))
    world_bounds = _bounds_for_records(image_records, axes)
    candidates: list[dict[str, Any]] = []

    for tile_size in tile_sizes:
        plan = build_large_scale_3dgs_plan(
            LargeScale3DGSOptions(
                data_dir=data_dir,
                output_dir=output_dir,
                tile_size=tile_size,
                overlap=options.overlap,
                axes="".join(axes),
                min_images=options.min_images,
                iterations=options.iterations,
                config=options.config,
                materialize=False,
            )
        )
        candidates.append(
            _preflight_candidate_from_plan(
                plan,
                tile_size=tile_size,
                overlap=options.overlap,
                image_sizes=image_sizes,
                target_images_per_chunk=options.target_images_per_chunk,
            )
        )

    recommended = _recommend_preflight_candidate(candidates)
    candidates = [{**candidate, "recommended": candidate is recommended} for candidate in candidates]
    recommended = next(candidate for candidate in candidates if candidate["recommended"])
    plan_path = output_dir / "large_scale_3dgs_plan.json"
    run_report_path = output_dir / "large_scale_3dgs_run_report.json"
    plan_command = _build_preflight_plan_command(options, recommended)

    return {
        "version": 1,
        "type": "large-scale-3dgs-preflight",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataDir": str(data_dir),
        "sparseDir": str(sparse_dir),
        "imageRoot": image_root,
        "outputDir": str(output_dir),
        "axes": "".join(axes),
        "targetImagesPerChunk": int(options.target_images_per_chunk),
        "summary": {
            "registeredImageCount": len(image_records),
            "points3DCount": len(point_records),
            "sourceImageBytes": int(sum(image_sizes.values())),
            "worldBounds": world_bounds,
            "worldSpan": {
                axes[0]: round(world_bounds[f"max{axes[0].upper()}"] - world_bounds[f"min{axes[0].upper()}"], 3),
                axes[1]: round(world_bounds[f"max{axes[1].upper()}"] - world_bounds[f"min{axes[1].upper()}"], 3),
            },
        },
        "candidates": candidates,
        "recommendation": {
            "tileSize": recommended["tileSize"],
            "overlap": recommended["overlap"],
            "chunkCount": recommended["chunkCount"],
            "readyChunkCount": recommended["readyChunkCount"],
            "coreImagesPerChunk": recommended["coreImagesPerChunk"],
            "sourceImageBytesPerChunk": recommended["sourceImageBytesPerChunk"],
        },
        "next": {
            "planCommand": plan_command,
            "runCommand": _format_command(["gs-mapper", "large-scale-3dgs-run", "--plan", plan_path]),
            "catalogCommand": _format_command(
                [
                    "gs-mapper",
                    "large-scale-3dgs-catalog",
                    "--plan",
                    plan_path,
                    "--run-report",
                    run_report_path,
                ]
            ),
        },
    }


def write_large_scale_3dgs_preflight(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "large_scale_3dgs_preflight.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def format_large_scale_3dgs_preflight_text(report: dict[str, Any], report_path: Path | None = None) -> str:
    summary = report["summary"]
    recommendation = report["recommendation"]
    lines = [
        "Large-scale 3DGS preflight",
        f"  data: {report['dataDir']}",
        f"  sparse: {report['sparseDir']}",
        f"  images: {summary['registeredImageCount']} ({_format_bytes(summary['sourceImageBytes'])})",
        f"  points3D: {summary['points3DCount']}",
        f"  span: {report['axes'][0]}={summary['worldSpan'][report['axes'][0]]} / "
        f"{report['axes'][1]}={summary['worldSpan'][report['axes'][1]]}",
        f"  recommended: tile_size={recommendation['tileSize']} overlap={recommendation['overlap']} "
        f"chunks={recommendation['readyChunkCount']}/{recommendation['chunkCount']} ready",
    ]
    if report_path is not None:
        lines.append(f"  report: {report_path}")
    lines.append("  candidates:")
    for candidate in report["candidates"]:
        marker = "*" if candidate.get("recommended") else "-"
        lines.append(
            f"    {marker} tile={candidate['tileSize']} overlap={candidate['overlap']} "
            f"ready={candidate['readyChunkCount']}/{candidate['chunkCount']} "
            f"core_p50={candidate['coreImagesPerChunk']['p50']} "
            f"image_bytes_p90={_format_bytes(candidate['sourceImageBytesPerChunk']['p90'])}"
        )
    lines.extend(
        [
            f"  next plan: {report['next']['planCommand']}",
            f"  next run: {report['next']['runCommand']}",
            f"  next catalog: {report['next']['catalogCommand']}",
        ]
    )
    return "\n".join(lines)


def write_large_scale_3dgs_plan(plan: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "large_scale_3dgs_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan_path


def load_large_scale_3dgs_plan(plan_path: Path) -> dict[str, Any]:
    return json.loads(Path(plan_path).read_text(encoding="utf-8"))


def _resolve_report_path(plan: dict[str, Any], options: LargeScale3DGSRunOptions) -> Path:
    if options.report_path is not None:
        return Path(options.report_path)
    return Path(plan["outputDir"]) / "large_scale_3dgs_run_report.json"


def _run_plan_command(command: str, runner: CommandRunner) -> dict[str, Any]:
    started_at = time.time()
    args = _split_command(command)
    result = runner(args)
    elapsed = time.time() - started_at
    return {
        "command": command,
        "args": args,
        "returnCode": int(result.returncode),
        "durationSec": round(elapsed, 3),
        "status": "ok" if result.returncode == 0 else "failed",
    }


def run_large_scale_3dgs_plan(
    options: LargeScale3DGSRunOptions,
    *,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Execute train/export commands from a large-scale 3DGS plan."""
    plan = load_large_scale_3dgs_plan(Path(options.plan_path))
    runner = command_runner or _default_command_runner
    ready_chunks = [chunk for chunk in plan.get("chunks", []) if chunk.get("status") == "ready"]
    if options.max_chunks is not None:
        ready_chunks = ready_chunks[: max(0, int(options.max_chunks))]

    chunk_reports: list[dict[str, Any]] = []
    failed = False

    for chunk in ready_chunks:
        splat_output = Path(chunk["splatOutput"])
        train_ply = Path(chunk["trainOutputDir"]) / "point_cloud.ply"

        if options.resume and splat_output.exists():
            chunk_reports.append(
                {
                    "id": chunk["id"],
                    "status": "skipped",
                    "reason": "splat-exists",
                    "splatOutput": str(splat_output),
                }
            )
            continue

        if options.dry_run:
            chunk_reports.append(
                {
                    "id": chunk["id"],
                    "status": "planned",
                    "trainCommand": chunk["trainCommand"],
                    "exportCommand": chunk["exportCommand"],
                    "splatOutput": str(splat_output),
                }
            )
            continue

        train_report = None
        if options.resume and train_ply.exists():
            train_report = {
                "command": chunk["trainCommand"],
                "returnCode": 0,
                "durationSec": 0.0,
                "status": "skipped",
                "reason": "point-cloud-exists",
            }
        else:
            train_report = _run_plan_command(chunk["trainCommand"], runner)

        if train_report["status"] == "failed":
            failed = True
            chunk_reports.append(
                {
                    "id": chunk["id"],
                    "status": "failed",
                    "stage": "train",
                    "train": train_report,
                    "splatOutput": str(splat_output),
                }
            )
            if options.fail_fast:
                break
            continue

        export_report = _run_plan_command(chunk["exportCommand"], runner)
        if export_report["status"] == "failed":
            failed = True
            chunk_reports.append(
                {
                    "id": chunk["id"],
                    "status": "failed",
                    "stage": "export",
                    "train": train_report,
                    "export": export_report,
                    "splatOutput": str(splat_output),
                }
            )
            if options.fail_fast:
                break
            continue

        chunk_reports.append(
            {
                "id": chunk["id"],
                "status": "done",
                "train": train_report,
                "export": export_report,
                "splatOutput": str(splat_output),
            }
        )

    report = {
        "version": 1,
        "type": "large-scale-3dgs-run-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "planPath": str(options.plan_path),
        "dryRun": bool(options.dry_run),
        "resume": bool(options.resume),
        "summary": {
            "selectedChunkCount": len(ready_chunks),
            "doneCount": sum(1 for chunk in chunk_reports if chunk["status"] == "done"),
            "skippedCount": sum(1 for chunk in chunk_reports if chunk["status"] == "skipped"),
            "plannedCount": sum(1 for chunk in chunk_reports if chunk["status"] == "planned"),
            "failedCount": sum(1 for chunk in chunk_reports if chunk["status"] == "failed"),
            "status": "failed" if failed else "ok",
        },
        "chunks": chunk_reports,
    }

    report_path = _resolve_report_path(plan, options)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["reportPath"] = str(report_path)
    return report


def format_large_scale_3dgs_shell(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# {plan['summary']['readyChunkCount']} ready chunk(s) from {plan['summary']['chunkCount']} planned chunk(s)",
    ]
    for chunk in plan["chunks"]:
        if chunk["status"] != "ready":
            lines.append(f"# skip {chunk['id']}: {chunk['status']} ({chunk['coreImageCount']} core images)")
            continue
        lines.append(chunk["trainCommand"])
        lines.append(chunk["exportCommand"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_large_scale_3dgs_text(plan: dict[str, Any], plan_path: Path | None = None) -> str:
    summary = plan["summary"]
    lines = [
        "Large-scale 3DGS plan",
        f"  data: {plan['dataDir']}",
        f"  sparse: {plan['sparseDir']}",
        f"  chunks: {summary['readyChunkCount']} ready / {summary['chunkCount']} total",
        f"  tile: {plan['tiling']['tileSize']}m overlap={plan['tiling']['overlap']}m axes={plan['tiling']['axes']}",
    ]
    if plan_path is not None:
        lines.append(f"  plan: {plan_path}")
    return "\n".join(lines)


def format_large_scale_3dgs_run_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "Large-scale 3DGS run",
            f"  status: {summary['status']}",
            f"  chunks: {summary['doneCount']} done / {summary['skippedCount']} skipped / {summary['failedCount']} failed",
            f"  report: {report['reportPath']}",
        ]
    )


def _run_report_done_ids(run_report: dict[str, Any] | None) -> set[str]:
    if not run_report:
        return set()
    return {chunk["id"] for chunk in run_report.get("chunks", []) if chunk.get("status") in {"done", "skipped"}}


def build_large_scale_3dgs_catalog(options: LargeScale3DGSCatalogOptions) -> dict[str, Any]:
    """Build a web-facing tile catalog from a large-scale 3DGS plan."""
    plan = load_large_scale_3dgs_plan(Path(options.plan_path))
    run_report = (
        json.loads(Path(options.run_report_path).read_text(encoding="utf-8")) if options.run_report_path else None
    )
    done_ids = _run_report_done_ids(run_report)
    scene_id = _slugify(options.scene_id, "large-scale-3dgs")
    tiles: list[dict[str, Any]] = []

    for chunk in plan.get("chunks", []):
        if chunk.get("status") != "ready":
            continue

        source_splat = Path(chunk["splatOutput"])
        has_splat = source_splat.exists()
        if options.require_splats and not has_splat:
            continue

        public_url = str(source_splat)
        public_path = None
        if options.public_root is not None:
            public_path = (
                Path(options.public_root) / options.public_url_prefix.strip("/") / scene_id / source_splat.name
            )
            if has_splat:
                _link_or_copy(source_splat, public_path, options.link_mode)
            public_url = _join_public_url(options.public_url_prefix, scene_id, source_splat.name)

        tiles.append(
            {
                "id": chunk["id"],
                "label": chunk["id"].replace("_", " "),
                "status": "ready" if has_splat else "missing-splat",
                "runStatus": "done" if chunk["id"] in done_ids else "unknown",
                "splatUrl": public_url,
                "sourceSplat": str(source_splat),
                "publicPath": str(public_path) if public_path is not None else "",
                "coreBounds": chunk["coreBounds"],
                "expandedBounds": chunk["expandedBounds"],
                "tileIndex": chunk["tileIndex"],
                "axes": chunk["axes"],
                "imageCount": chunk["imageCount"],
                "coreImageCount": chunk["coreImageCount"],
                "pointCount": chunk["pointCount"],
            }
        )

    return {
        "version": 1,
        "type": "large-scale-3dgs-tile-catalog",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sceneId": scene_id,
        "label": options.label,
        "planPath": str(options.plan_path),
        "runReportPath": str(options.run_report_path) if options.run_report_path else "",
        "tiling": plan["tiling"],
        "summary": {
            "tileCount": len(tiles),
            "readyTileCount": sum(1 for tile in tiles if tile["status"] == "ready"),
            "missingSplatTileCount": sum(1 for tile in tiles if tile["status"] == "missing-splat"),
        },
        "tiles": tiles,
    }


def write_large_scale_3dgs_catalog(catalog: dict[str, Any], options: LargeScale3DGSCatalogOptions) -> Path:
    if options.output_path is not None:
        output_path = Path(options.output_path)
    else:
        plan = load_large_scale_3dgs_plan(Path(options.plan_path))
        output_path = Path(plan["outputDir"]) / "large_scale_3dgs_tile_catalog.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return output_path


def _catalog_route_axes(catalog: dict[str, Any]) -> tuple[str, str]:
    tiling = catalog.get("tiling") if isinstance(catalog.get("tiling"), dict) else {}
    axes = str(tiling.get("axes") or "").strip().lower()
    if not axes:
        for tile in catalog.get("tiles", []):
            axes = str(tile.get("axes") or "").strip().lower()
            if axes:
                break
    return _validate_axes(axes or "xz")


def _ready_catalog_route_tiles(catalog: dict[str, Any], *, include_missing_splats: bool) -> list[dict[str, Any]]:
    tiles = catalog.get("tiles", [])
    if not isinstance(tiles, list):
        return []

    ready_tiles: list[dict[str, Any]] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        if not include_missing_splats and (tile.get("status") == "missing-splat" or not tile.get("splatUrl")):
            continue
        ready_tiles.append(tile)
    return ready_tiles


def _tile_axis_center(tile: dict[str, Any], axis: str) -> float:
    min_key = f"min{axis.upper()}"
    max_key = f"max{axis.upper()}"
    for bounds_key in ("coreBounds", "expandedBounds"):
        bounds = tile.get(bounds_key)
        if not isinstance(bounds, dict):
            continue
        minimum = bounds.get(min_key)
        maximum = bounds.get(max_key)
        if minimum is not None and maximum is not None:
            return (float(minimum) + float(maximum)) / 2.0
    raise ValueError(f"tile {tile.get('id', '<unknown>')} is missing {min_key}/{max_key} bounds")


def _tile_center_position(tile: dict[str, Any], axes: tuple[str, str], *, default_y: float) -> list[float]:
    position = [0.0, float(default_y), 0.0]
    for axis in axes:
        position[_AXIS_INDEX[axis]] = _tile_axis_center(tile, axis)
    return [round(float(value), 6) for value in position]


def _tile_index_pair(tile: dict[str, Any], axes: tuple[str, str]) -> tuple[int, int] | None:
    tile_index = tile.get("tileIndex")
    if not isinstance(tile_index, dict):
        return None

    values = [tile_index.get(axis) for axis in axes]
    if not all(isinstance(value, int) for value in values):
        return None
    return int(values[0]), int(values[1])


def _sort_catalog_tiles_by_center(
    tiles: list[dict[str, Any]], axes: tuple[str, str], *, default_y: float
) -> list[dict[str, Any]]:
    return sorted(
        tiles,
        key=lambda tile: (
            _tile_center_position(tile, axes, default_y=default_y)[_AXIS_INDEX[axes[0]]],
            _tile_center_position(tile, axes, default_y=default_y)[_AXIS_INDEX[axes[1]]],
            str(tile.get("id") or ""),
        ),
    )


def _spiral_tile_index_keys(indexed_tiles: dict[tuple[int, int], dict[str, Any]]) -> list[tuple[int, int]]:
    first_values = [key[0] for key in indexed_tiles]
    second_values = [key[1] for key in indexed_tiles]
    left = min(first_values)
    right = max(first_values)
    bottom = min(second_values)
    top = max(second_values)
    keys: list[tuple[int, int]] = []

    while left <= right and bottom <= top:
        for second in range(bottom, top + 1):
            keys.append((left, second))
        for first in range(left + 1, right + 1):
            keys.append((first, top))
        if left < right:
            for second in range(top - 1, bottom - 1, -1):
                keys.append((right, second))
        if bottom < top:
            for first in range(right - 1, left, -1):
                keys.append((first, bottom))

        left += 1
        right -= 1
        bottom += 1
        top -= 1

    return keys


def _order_catalog_route_tiles(
    tiles: list[dict[str, Any]],
    axes: tuple[str, str],
    *,
    default_y: float,
    order: str,
) -> list[dict[str, Any]]:
    indexed_pairs = [(tile, _tile_index_pair(tile, axes)) for tile in tiles]
    if not all(pair is not None for _, pair in indexed_pairs):
        return _sort_catalog_tiles_by_center(tiles, axes, default_y=default_y)

    indexed_tiles = {pair: tile for tile, pair in indexed_pairs if pair is not None}
    if len(indexed_tiles) != len(tiles):
        return _sort_catalog_tiles_by_center(tiles, axes, default_y=default_y)

    normalized_order = order if order in {"spiral", "snake", "row-major"} else "spiral"
    first_values = sorted({pair[0] for pair in indexed_tiles})
    second_values = sorted({pair[1] for pair in indexed_tiles})

    if normalized_order == "row-major":
        keys = [(first, second) for first in first_values for second in second_values]
    elif normalized_order == "snake":
        keys = []
        for first_index, first in enumerate(first_values):
            row_seconds = second_values if first_index % 2 == 0 else list(reversed(second_values))
            keys.extend((first, second) for second in row_seconds)
    else:
        keys = _spiral_tile_index_keys(indexed_tiles)

    return [indexed_tiles[key] for key in keys if key in indexed_tiles]


def _yaw_degrees_from_segment(
    current_position: Sequence[float], next_position: Sequence[float], fallback: float = 0.0
) -> int:
    dx = float(next_position[0]) - float(current_position[0])
    dz = float(next_position[2]) - float(current_position[2])
    if math.hypot(dx, dz) < 1e-6:
        return int(round(fallback)) % 360
    return int(round((math.degrees(math.atan2(-dx, -dz))) % 360))


def build_large_scale_3dgs_route(options: LargeScale3DGSRouteOptions) -> dict[str, Any]:
    """Build a DreamWalker robot route through ready tiles in a large-scale 3DGS catalog."""
    catalog_path = Path(options.catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    axes = _catalog_route_axes(catalog)
    tiles = _ready_catalog_route_tiles(catalog, include_missing_splats=options.include_missing_splats)
    if not tiles:
        raise ValueError("tile catalog has no ready tiles to route through")

    ordered_tiles = _order_catalog_route_tiles(tiles, axes, default_y=options.default_y, order=options.order)
    route_positions = [_tile_center_position(tile, axes, default_y=options.default_y) for tile in ordered_tiles]
    pose_position = route_positions[-1]
    yaw_degrees = _yaw_degrees_from_segment(route_positions[-2], route_positions[-1]) if len(route_positions) > 1 else 0
    scene_id = str(catalog.get("sceneId") or "large-scale-3dgs")
    catalog_label = str(catalog.get("label") or scene_id)
    route_label = options.label or f"{catalog_label} Route"
    description = options.description or f"auto-generated route through {len(route_positions)} large-scale 3DGS tile(s)"
    asset_label = options.asset_label or catalog_label

    return {
        "version": 1,
        "protocol": "dreamwalker-robot-route/v1",
        "label": route_label,
        "description": description,
        "fragmentId": options.fragment_id,
        "fragmentLabel": options.fragment_label,
        "frameId": options.frame_id,
        "world": {
            "fragmentId": options.fragment_id,
            "fragmentLabel": options.fragment_label,
            "assetLabel": asset_label,
            "splatUrl": options.world_splat_url,
            "colliderMeshUrl": options.collider_mesh_url,
            "frameId": options.frame_id,
            "zoneMapUrl": options.zone_map_url,
            "usesDemoFallback": False,
        },
        "pose": {
            "position": pose_position,
            "yawDegrees": yaw_degrees,
        },
        "route": route_positions,
        "tileSequence": [str(tile.get("id") or "") for tile in ordered_tiles],
        "sourceCatalog": {
            "path": str(catalog_path),
            "sceneId": scene_id,
            "label": catalog_label,
            "axes": "".join(axes),
            "order": options.order,
        },
    }


def _default_large_scale_3dgs_route_path(route: dict[str, Any], catalog_path: Path) -> Path:
    source_catalog = route.get("sourceCatalog") if isinstance(route.get("sourceCatalog"), dict) else {}
    scene_id = _slugify(
        str(source_catalog.get("sceneId") or route.get("label") or "large-scale-3dgs"), "large-scale-3dgs"
    )
    if catalog_path.parent.name == "manifests":
        return catalog_path.parent.parent / "robot-routes" / f"{scene_id}-route.json"
    return catalog_path.with_name(f"{scene_id}-route.json")


def write_large_scale_3dgs_route(route: dict[str, Any], options: LargeScale3DGSRouteOptions) -> Path:
    output_path = (
        Path(options.output_path)
        if options.output_path is not None
        else _default_large_scale_3dgs_route_path(
            route,
            Path(options.catalog_path),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    return output_path


def format_large_scale_3dgs_route_text(route: dict[str, Any], route_path: Path) -> str:
    source_catalog = route.get("sourceCatalog") if isinstance(route.get("sourceCatalog"), dict) else {}
    tile_sequence = route.get("tileSequence") if isinstance(route.get("tileSequence"), list) else []
    lines = [
        "Large-scale 3DGS robot route",
        f"  label: {route.get('label', '')}",
        f"  catalog: {source_catalog.get('path', '')}",
        f"  route: {route_path}",
        f"  points: {len(route.get('route', []))}",
    ]
    if tile_sequence:
        lines.append(f"  tiles: {' -> '.join(str(tile_id) for tile_id in tile_sequence)}")
    lines.append(
        f"  pose: {route.get('pose', {}).get('position', [])} yaw={route.get('pose', {}).get('yawDegrees', 0)}"
    )
    return "\n".join(lines)


def _path_to_public_url(path: Path, public_root: Path | None) -> str:
    if public_root is None:
        return ""

    try:
        relative_path = path.resolve().relative_to(public_root.resolve())
    except ValueError:
        return ""

    return "/" + "/".join(quote(part) for part in relative_path.parts)


def _resolve_public_input_url(input_value: str | Path | None, public_root: Path | None) -> str:
    if input_value is None:
        return ""

    normalized = str(input_value).strip()
    if not normalized:
        return ""

    if normalized.startswith(("http://", "https://")):
        return normalized

    if normalized.startswith("/") and not normalized.startswith("//"):
        public_url = _path_to_public_url(Path(normalized), public_root)
        if public_url:
            return public_url
        if Path(normalized).exists():
            return ""
        return normalized

    return _path_to_public_url(Path(normalized), public_root)


def _positive_int_or_none(value: int | str | None) -> int | None:
    if value is None:
        return None

    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None

    return normalized if normalized > 0 else None


def _build_url_with_query(site_url: str, query: dict[str, str]) -> str:
    split_url = urlsplit(site_url)
    query_pairs = parse_qsl(split_url.query, keep_blank_values=True)
    query_pairs.extend(query.items())
    return urlunsplit(
        (
            split_url.scheme,
            split_url.netloc,
            split_url.path or "/",
            urlencode(query_pairs),
            split_url.fragment,
        )
    )


def build_large_scale_3dgs_web_runbook(
    catalog_path: Path,
    options: LargeScale3DGSCatalogOptions,
) -> dict[str, str]:
    """Build follow-up commands for validating and loading a tile catalog in DreamWalker."""
    preload_mode = options.tile_preload if options.tile_preload in {"off", "metadata", "cache"} else "metadata"
    public_root = Path(options.public_root) if options.public_root is not None else None
    catalog_url = _path_to_public_url(Path(catalog_path), public_root)
    route_input = str(options.route_path).strip() if options.route_path is not None else ""
    route_url = _resolve_public_input_url(route_input, public_root)
    route_playback_ms = _positive_int_or_none(options.route_playback_ms)
    validate_parts: list[str | Path | int] = [
        "npm",
    ]
    if options.web_app_dir is not None:
        validate_parts.extend(["--prefix", options.web_app_dir])
    validate_parts.extend(
        [
            "run",
            "validate:dynamic-map-catalog",
            "--",
            catalog_path,
        ]
    )
    if public_root is not None:
        validate_parts.extend(["--public-root", public_root])
    validate_parts.extend(["--site-url", options.site_url, "--preload-mode", preload_mode])
    if route_input:
        validate_parts.extend(["--route", route_input])
    if options.route_playback:
        validate_parts.extend(["--route-playback", "1"])
    if route_playback_ms is not None:
        validate_parts.extend(["--route-playback-ms", route_playback_ms])
    if options.route_playback_loop:
        validate_parts.extend(["--route-playback-loop", "1"])

    launch_url = ""
    if catalog_url:
        launch_query = {
            "tileCatalog": catalog_url,
            "tilePreload": preload_mode,
        }
        if route_url:
            launch_query["robotRoute"] = route_url
        if options.route_playback or options.route_playback_loop:
            launch_query["robotRoutePlayback"] = "1"
        if route_playback_ms is not None:
            launch_query["robotRoutePlaybackMs"] = str(route_playback_ms)
        if options.route_playback_loop:
            launch_query["robotRoutePlaybackLoop"] = "1"

        launch_url = _build_url_with_query(
            options.site_url,
            launch_query,
        )

    return {
        "catalogUrl": catalog_url,
        "routeUrl": route_url,
        "validateCommand": _format_command(validate_parts),
        "launchUrl": launch_url,
    }


def format_large_scale_3dgs_catalog_text(
    catalog: dict[str, Any],
    catalog_path: Path,
    options: LargeScale3DGSCatalogOptions | None = None,
) -> str:
    summary = catalog["summary"]
    lines = [
        "Large-scale 3DGS tile catalog",
        f"  scene: {catalog['sceneId']} / {catalog['label']}",
        f"  tiles: {summary['readyTileCount']} ready / {summary['tileCount']} total",
        f"  catalog: {catalog_path}",
    ]

    if options is not None:
        runbook = build_large_scale_3dgs_web_runbook(catalog_path, options)
        lines.append(f"  validate: {runbook['validateCommand']}")
        if runbook["launchUrl"]:
            lines.append(f"  launch: {runbook['launchUrl']}")
        else:
            lines.append("  launch: unavailable until --output is inside --public-root")

    return "\n".join(lines)
