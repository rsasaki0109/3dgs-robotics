"""PLY to web format conversion for browser-based rendering.

Converts trained Gaussian Splat PLY files to JSON or compact binary
formats that can be rendered in the browser using the existing Three.js
viewer on GitHub Pages.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)
SPLAT_RECORD_BYTES = 32
_SPLAT_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class SplatFilterReport:
    """Summary of a direct .splat cleanup pass."""

    input_count: int
    output_count: int
    min_opacity: float
    max_scale: float | None
    max_scale_percentile: float | None
    adaptive_max_scale: float | None
    max_points: int | None

    @property
    def kept_ratio(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.output_count / self.input_count


@dataclass(frozen=True, slots=True)
class SplatStatsReport:
    """Opacity and scale summary for an antimatter15 .splat binary."""

    input_count: int
    size_bytes: int
    low_opacity_threshold: float
    low_opacity_count: int
    opacity_min: float
    opacity_p10: float
    opacity_p50: float
    opacity_p90: float
    opacity_max: float
    scale_min: float
    scale_p50: float
    scale_p95: float
    scale_p98: float
    scale_p99: float
    scale_max: float

    @property
    def low_opacity_ratio(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.low_opacity_count / self.input_count

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000

    @property
    def scale_tail_ratio(self) -> float:
        if self.scale_p98 <= 0.0:
            return 0.0
        return self.scale_max / self.scale_p98

    def as_dict(self) -> dict[str, float | int]:
        return {
            "input_count": self.input_count,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_mb,
            "low_opacity_threshold": self.low_opacity_threshold,
            "low_opacity_count": self.low_opacity_count,
            "low_opacity_ratio": self.low_opacity_ratio,
            "opacity_min": self.opacity_min,
            "opacity_p10": self.opacity_p10,
            "opacity_p50": self.opacity_p50,
            "opacity_p90": self.opacity_p90,
            "opacity_max": self.opacity_max,
            "scale_min": self.scale_min,
            "scale_p50": self.scale_p50,
            "scale_p95": self.scale_p95,
            "scale_p98": self.scale_p98,
            "scale_p99": self.scale_p99,
            "scale_max": self.scale_max,
            "scale_tail_ratio": self.scale_tail_ratio,
        }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile for an empty sequence")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("max_scale_percentile must be in the range (0, 100]")
    index = round((len(sorted_values) - 1) * (percentile / 100.0))
    return sorted_values[min(len(sorted_values) - 1, max(0, index))]


def filter_splat_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_opacity: float = 0.0,
    max_scale: float | None = None,
    max_scale_percentile: float | None = None,
    max_points: int | None = None,
) -> SplatFilterReport:
    """Filter an existing antimatter15 .splat binary without requiring the source PLY.

    This is useful when a browser export looks cloudy: oversized or nearly
    transparent gaussians can dominate the WebGL blend even though they carry
    little map structure. Records are kept in their original order so exports
    that were already sorted by importance remain stable.
    """

    src = Path(input_path)
    dst = Path(output_path)
    data = src.read_bytes()
    if len(data) % SPLAT_RECORD_BYTES:
        raise ValueError(f"{src} is not a 32-byte-per-gaussian .splat file")

    count = len(data) // SPLAT_RECORD_BYTES
    scale_max_values: list[float] = []
    for index in range(count):
        offset = index * SPLAT_RECORD_BYTES + 12
        scale_max_values.append(max(struct.unpack_from("<fff", data, offset)))
    adaptive_max_scale = None
    if max_scale_percentile is not None:
        adaptive_max_scale = _percentile(sorted(scale_max_values), float(max_scale_percentile))

    kept = bytearray()
    written = 0
    for index, scale_max in enumerate(scale_max_values):
        offset = index * SPLAT_RECORD_BYTES
        opacity = data[offset + 27] / 255.0
        if opacity < min_opacity:
            continue
        if max_scale is not None and max_scale > 0.0 and scale_max > max_scale:
            continue
        if adaptive_max_scale is not None and scale_max > adaptive_max_scale:
            continue
        kept.extend(data[offset : offset + SPLAT_RECORD_BYTES])
        written += 1
        if max_points is not None and max_points > 0 and written >= max_points:
            break

    if written == 0:
        raise ValueError(
            "No splats survived filtering "
            f"(min_opacity={min_opacity}, max_scale={max_scale}, max_scale_percentile={max_scale_percentile})"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(kept)
    return SplatFilterReport(
        input_count=count,
        output_count=written,
        min_opacity=float(min_opacity),
        max_scale=max_scale,
        max_scale_percentile=max_scale_percentile,
        adaptive_max_scale=adaptive_max_scale,
        max_points=max_points,
    )


def inspect_splat_file(
    input_path: str | Path,
    *,
    low_opacity_threshold: float = 0.08,
) -> SplatStatsReport:
    """Inspect an existing antimatter15 .splat binary for cleanup decisions."""

    src = Path(input_path)
    data = src.read_bytes()
    if len(data) % SPLAT_RECORD_BYTES:
        raise ValueError(f"{src} is not a 32-byte-per-gaussian .splat file")

    count = len(data) // SPLAT_RECORD_BYTES
    if count == 0:
        raise ValueError(f"{src} does not contain any splats")

    opacities: list[float] = []
    scale_max_values: list[float] = []
    low_opacity_count = 0
    for index in range(count):
        offset = index * SPLAT_RECORD_BYTES
        opacity = data[offset + 27] / 255.0
        scale_max = max(struct.unpack_from("<fff", data, offset + 12))
        opacities.append(opacity)
        scale_max_values.append(scale_max)
        if opacity < low_opacity_threshold:
            low_opacity_count += 1

    opacities.sort()
    scale_max_values.sort()
    return SplatStatsReport(
        input_count=count,
        size_bytes=len(data),
        low_opacity_threshold=float(low_opacity_threshold),
        low_opacity_count=low_opacity_count,
        opacity_min=opacities[0],
        opacity_p10=_percentile(opacities, 10.0),
        opacity_p50=_percentile(opacities, 50.0),
        opacity_p90=_percentile(opacities, 90.0),
        opacity_max=opacities[-1],
        scale_min=scale_max_values[0],
        scale_p50=_percentile(scale_max_values, 50.0),
        scale_p95=_percentile(scale_max_values, 95.0),
        scale_p98=_percentile(scale_max_values, 98.0),
        scale_p99=_percentile(scale_max_values, 99.0),
        scale_max=scale_max_values[-1],
    )


def _validate_splat_axes(axes: str) -> tuple[str, str]:
    normalized = str(axes or "").strip().lower()
    if len(normalized) != 2 or normalized[0] == normalized[1]:
        raise ValueError("--axes must contain two distinct axes, e.g. xy, xz, yz")
    if normalized[0] not in _SPLAT_AXIS_INDEX or normalized[1] not in _SPLAT_AXIS_INDEX:
        raise ValueError("--axes must contain only x, y, z")
    return normalized[0], normalized[1]


def _splat_axis_value(data: bytes, record_index: int, axis: str) -> float:
    return struct.unpack_from("<f", data, record_index * SPLAT_RECORD_BYTES + _SPLAT_AXIS_INDEX[axis] * 4)[0]


def _record_in_axis_bounds(
    data: bytes,
    record_index: int,
    *,
    axes: tuple[str, str],
    bounds: dict[str, float],
) -> bool:
    first = _splat_axis_value(data, record_index, axes[0])
    second = _splat_axis_value(data, record_index, axes[1])
    return (
        bounds[f"min{axes[0].upper()}"] <= first <= bounds[f"max{axes[0].upper()}"]
        and bounds[f"min{axes[1].upper()}"] <= second <= bounds[f"max{axes[1].upper()}"]
    )


def _splat_world_bounds(data: bytes, count: int, axes: tuple[str, str]) -> dict[str, float]:
    first_values = [_splat_axis_value(data, index, axes[0]) for index in range(count)]
    second_values = [_splat_axis_value(data, index, axes[1]) for index in range(count)]
    return {
        f"min{axes[0].upper()}": float(min(first_values)),
        f"max{axes[0].upper()}": float(max(first_values)),
        f"min{axes[1].upper()}": float(min(second_values)),
        f"max{axes[1].upper()}": float(max(second_values)),
    }


def splat_to_tile_catalog(
    input_path: str | Path,
    catalog_path: str | Path,
    *,
    public_root: str | Path,
    scene_id: str | None = None,
    label: str | None = None,
    tile_size: float = 10.0,
    overlap: float = 2.0,
    axes: str = "xz",
    min_splats: int = 1,
    public_url_prefix: str = "/splats",
) -> dict[str, Any]:
    """Split an existing browser .splat into tile files and a dynamic-map catalog."""
    if tile_size <= 0:
        raise ValueError("--tile-size must be > 0")
    if overlap < 0:
        raise ValueError("--overlap must be >= 0")
    if min_splats < 1:
        raise ValueError("--min-splats must be >= 1")

    src = Path(input_path)
    data = src.read_bytes()
    if len(data) % SPLAT_RECORD_BYTES:
        raise ValueError(f"{src} is not a 32-byte-per-gaussian .splat file")
    count = len(data) // SPLAT_RECORD_BYTES
    if count == 0:
        raise ValueError(f"{src} does not contain any splats")

    split_axes = _validate_splat_axes(axes)
    catalog = Path(catalog_path)
    root = Path(public_root)
    normalized_scene_id = _sanitize_scene_id(scene_id or src.stem)
    tile_dir = root / public_url_prefix.strip("/") / normalized_scene_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    world_bounds = _splat_world_bounds(data, count, split_axes)
    axis_a, axis_b = split_axes
    min_a = world_bounds[f"min{axis_a.upper()}"]
    min_b = world_bounds[f"min{axis_b.upper()}"]
    max_a = world_bounds[f"max{axis_a.upper()}"]
    max_b = world_bounds[f"max{axis_b.upper()}"]
    num_a = max(1, int((max_a - min_a) // tile_size) + 1)
    num_b = max(1, int((max_b - min_b) // tile_size) + 1)
    tiles: list[dict[str, Any]] = []

    for tile_a in range(num_a):
        for tile_b in range(num_b):
            core_bounds = {
                f"min{axis_a.upper()}": min_a + tile_a * tile_size,
                f"max{axis_a.upper()}": min_a + (tile_a + 1) * tile_size,
                f"min{axis_b.upper()}": min_b + tile_b * tile_size,
                f"max{axis_b.upper()}": min_b + (tile_b + 1) * tile_size,
            }
            expanded_bounds = {
                f"min{axis_a.upper()}": core_bounds[f"min{axis_a.upper()}"] - overlap,
                f"max{axis_a.upper()}": core_bounds[f"max{axis_a.upper()}"] + overlap,
                f"min{axis_b.upper()}": core_bounds[f"min{axis_b.upper()}"] - overlap,
                f"max{axis_b.upper()}": core_bounds[f"max{axis_b.upper()}"] + overlap,
            }
            core_indices = [
                index
                for index in range(count)
                if _record_in_axis_bounds(data, index, axes=split_axes, bounds=core_bounds)
            ]
            if len(core_indices) < min_splats:
                continue

            expanded_indices = [
                index
                for index in range(count)
                if _record_in_axis_bounds(data, index, axes=split_axes, bounds=expanded_bounds)
            ]
            tile_id = f"tile_{axis_a}{tile_a:03d}_{axis_b}{tile_b:03d}"
            tile_path = tile_dir / f"{tile_id}.splat"
            tile_path.write_bytes(
                b"".join(
                    data[index * SPLAT_RECORD_BYTES : (index + 1) * SPLAT_RECORD_BYTES] for index in expanded_indices
                )
            )
            tiles.append(
                {
                    "id": tile_id,
                    "label": tile_id.replace("_", " "),
                    "status": "ready",
                    "runStatus": "tiled",
                    "splatUrl": "/"
                    + "/".join(
                        part for part in [public_url_prefix.strip("/"), normalized_scene_id, tile_path.name] if part
                    ),
                    "sourceSplat": str(src),
                    "publicPath": str(tile_path),
                    "coreBounds": core_bounds,
                    "expandedBounds": expanded_bounds,
                    "tileIndex": {
                        axis_a: tile_a,
                        axis_b: tile_b,
                    },
                    "axes": "".join(split_axes),
                    "imageCount": 0,
                    "coreImageCount": 0,
                    "pointCount": len(expanded_indices),
                    "coreSplatCount": len(core_indices),
                    "splatCount": len(expanded_indices),
                }
            )

    catalog_payload = {
        "version": 1,
        "type": "large-scale-3dgs-tile-catalog",
        "sceneId": normalized_scene_id,
        "label": label or src.stem.replace("_", " ").replace("-", " "),
        "planPath": f"{src}:splat-tiling",
        "runReportPath": "",
        "tiling": {
            "strategy": "existing-splat-axis-grid",
            "axes": "".join(split_axes),
            "tileSize": float(tile_size),
            "overlap": float(overlap),
            "minSplats": int(min_splats),
            "worldBounds": world_bounds,
        },
        "summary": {
            "tileCount": len(tiles),
            "readyTileCount": len(tiles),
            "missingSplatTileCount": 0,
            "inputSplatCount": count,
            "inputBytes": len(data),
            "tiledSplatCount": sum(tile["splatCount"] for tile in tiles),
        },
        "tiles": tiles,
    }
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(json.dumps(catalog_payload, indent=2) + "\n", encoding="utf-8")
    return catalog_payload


def _sanitize_scene_id(value: str) -> str:
    text = str(value or "").strip().lower()
    normalized: list[str] = []
    last_was_dash = False
    for char in text:
        if char.isalnum():
            normalized.append(char)
            last_was_dash = False
        elif not last_was_dash:
            normalized.append("-")
            last_was_dash = True
    scene_id = "".join(normalized).strip("-")
    return scene_id or "scene"


def _load_web_point_data(ply_path: str, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    from gs_sim2real.viewer.web_viewer import load_ply

    ply_data = load_ply(ply_path)
    positions = np.asarray(ply_data.positions, dtype=np.float32)
    colors = np.asarray(ply_data.colors, dtype=np.float32)
    n = len(positions)

    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        indices.sort()
        positions = positions[indices]
        colors = colors[indices]

    return positions, colors


def _compute_bounds(positions: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    }


def _estimate_camera(bounds: dict[str, list[float]]) -> dict[str, list[float]]:
    import numpy as np

    minimum = np.asarray(bounds["min"], dtype=np.float32)
    maximum = np.asarray(bounds["max"], dtype=np.float32)
    center = (minimum + maximum) * 0.5
    extents = np.maximum(maximum - minimum, 1e-3)
    radius = float(max(np.linalg.norm(extents), extents.max()) * 0.9)
    position = center + np.array([radius * 1.4, radius * 0.75, radius * 1.4], dtype=np.float32)
    return {
        "position": position.astype(np.float32).tolist(),
        "target": center.astype(np.float32).tolist(),
        "up": [0.0, 1.0, 0.0],
    }


def points_to_scene_bundle(
    positions: np.ndarray,
    colors: np.ndarray,
    output_dir: str,
    *,
    asset_format: str = "binary",
    scene_id: str = "scene",
    label: str = "Scene",
    description: str = "",
    camera: dict[str, list[float]] | None = None,
) -> str:
    """Write positions/colors directly as a static web scene bundle."""
    import numpy as np

    normalized_asset_format = str(asset_format or "binary").strip().lower()
    if normalized_asset_format not in {"json", "binary"}:
        raise ValueError("asset_format must be one of: json, binary")

    positions_array = np.asarray(positions, dtype=np.float32)
    colors_array = np.asarray(colors, dtype=np.float32)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:
        raise ValueError("positions must be an array of shape (N, 3)")
    if colors_array.ndim != 2 or colors_array.shape[1] != 3:
        raise ValueError("colors must be an array of shape (N, 3)")
    if len(positions_array) != len(colors_array):
        raise ValueError("positions and colors must contain the same number of points")
    if len(positions_array) == 0:
        raise ValueError("scene bundle requires at least one point")

    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    resolved_scene_id = _sanitize_scene_id(scene_id)
    resolved_label = str(label or "Scene").strip() or "Scene"
    asset_name = (
        f"{resolved_scene_id}.points.json" if normalized_asset_format == "json" else f"{resolved_scene_id}.points.bin"
    )
    asset_path = bundle_dir / asset_name
    if normalized_asset_format == "json":
        _write_json_asset(asset_path, positions_array, colors_array)
    else:
        _write_binary_asset(asset_path, positions_array, colors_array)

    bounds = _compute_bounds(positions_array)
    manifest = {
        "version": "gs-sim2real-web-scene/v1",
        "type": "web-scene-manifest",
        "sceneId": resolved_scene_id,
        "label": resolved_label,
        "description": str(description or ""),
        "asset": {
            "href": asset_name,
            "format": normalized_asset_format,
        },
        "count": int(len(positions_array)),
        "bounds": bounds,
        "camera": camera or _estimate_camera(bounds),
    }
    manifest_path = bundle_dir / "scene.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Exported scene bundle to %s", manifest_path)
    return str(manifest_path)


def _write_json_asset(output_path: str | Path, positions: np.ndarray, colors: np.ndarray) -> str:
    bounds = _compute_bounds(positions)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "positions": positions.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "count": int(len(positions)),
        "bounds": bounds,
    }
    with open(out, "w", encoding="utf-8") as file:
        json.dump(data, file)
    return str(out)


def _write_binary_asset(output_path: str | Path, positions: np.ndarray, colors: np.ndarray) -> str:
    import numpy as np

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bounds = np.concatenate([positions.min(axis=0), positions.max(axis=0)]).astype(np.float32)
    with open(out, "wb") as file:
        file.write(struct.pack("<I", int(len(positions))))
        file.write(bounds.tobytes())
        combined = np.hstack([positions, colors]).astype(np.float32)
        file.write(combined.tobytes())
    return str(out)


def ply_to_json(ply_path: str, output_path: str, max_points: int = 100000) -> str:
    """Convert PLY point cloud to JSON format for web viewer.

    Outputs a JSON file with:
    {
        "positions": [x1,y1,z1, x2,y2,z2, ...],
        "colors": [r1,g1,b1, r2,g2,b2, ...],  // normalized 0-1
        "count": N,
        "bounds": {"min": [x,y,z], "max": [x,y,z]}
    }

    Args:
        ply_path: Path to the input PLY file.
        output_path: Path to the output JSON file.
        max_points: Maximum number of points to include (subsampled if exceeded).

    Returns:
        Path to the written output file as a string.
    """
    positions, colors = _load_web_point_data(ply_path, max_points)
    result = _write_json_asset(output_path, positions, colors)
    logger.info("Exported %d points to %s", len(positions), result)
    return result


def ply_to_binary(ply_path: str, output_path: str, max_points: int = 100000) -> str:
    """Convert PLY to compact binary format for faster web loading.

    Binary format:
    - 4 bytes: uint32 num_points
    - 24 bytes: float32[6] bounds (min_x, min_y, min_z, max_x, max_y, max_z)
    - num_points * 24 bytes: float32[6] per point (x, y, z, r, g, b)

    Args:
        ply_path: Path to the input PLY file.
        output_path: Path to the output binary file.
        max_points: Maximum number of points to include (subsampled if exceeded).

    Returns:
        Path to the written output file as a string.
    """
    positions, colors = _load_web_point_data(ply_path, max_points)
    result = _write_binary_asset(output_path, positions, colors)
    size_kb = Path(result).stat().st_size / 1024
    logger.info("Exported %d points to %s (%.1f KB)", len(positions), result, size_kb)
    return result


def ply_to_scene_bundle(
    ply_path: str,
    output_dir: str,
    *,
    asset_format: str = "binary",
    scene_id: str | None = None,
    label: str | None = None,
    description: str = "",
    max_points: int = 100000,
) -> str:
    """Export a self-contained scene bundle for static hosting on GitHub Pages.

    The output directory contains:
    - ``scene.json``: metadata + relative asset pointer
    - ``<scene-id>.points.json`` or ``<scene-id>.points.bin``: point data
    """
    positions, colors = _load_web_point_data(ply_path, max_points)
    return points_to_scene_bundle(
        positions,
        colors,
        output_dir,
        asset_format=asset_format,
        scene_id=scene_id or Path(ply_path).stem,
        label=label or Path(ply_path).stem.replace("_", " ").replace("-", " "),
        description=description,
    )


SH_C0 = 0.28209479177387814


def ply_to_splat(
    ply_path: str | Path,
    output_path: str | Path,
    max_points: int | None = None,
    normalize_target_extent: float | None = None,
    min_opacity: float = 0.0,
    max_scale: float | None = None,
    max_scale_percentile: float | None = None,
) -> str:
    """Convert a gsplat PLY to the antimatter15/splat 32-byte-per-gaussian binary.

    Per-gaussian layout (little-endian native float32 / uint8, matching the
    upstream WebGL viewer):
      - position   : float32 x 3  (bytes  0..11)
      - scale      : float32 x 3 as exp(log_scale)  (bytes 12..23)
      - color RGBA : uint8   x 4, RGB = (0.5 + SH_C0 * f_dc).clip, A = sigmoid(opacity)  (24..27)
      - rotation   : uint8   x 4, normalized quat * 128 + 128  (28..31)

    Gaussians are sorted by ``exp(sum(scale_logs)) * sigmoid(opacity)`` descending
    before writing so the viewer renders larger, more opaque splats first.

    If ``normalize_target_extent`` is set, positions are centered at the
    scene centroid and rescaled so that the largest XYZ extent equals the
    target (gaussian scales are divided by the same factor to preserve
    visual shape). This lets world-metric scenes render inside viewers that
    assume unit-ish scale.
    """
    import numpy as np

    from gs_sim2real.viewer.web_viewer import load_ply

    src = Path(ply_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    data = load_ply(str(src))
    positions = np.asarray(data.positions, dtype=np.float32)
    scales_log = np.asarray(data.scales, dtype=np.float32) if data.scales is not None else None
    rotations = np.asarray(data.rotations, dtype=np.float32) if data.rotations is not None else None
    opacities = np.asarray(data.opacities, dtype=np.float32).reshape(-1) if data.opacities is not None else None
    colors = np.asarray(data.colors, dtype=np.float32) if data.colors is not None else None
    if any(x is None for x in (scales_log, rotations, opacities, colors)):
        raise ValueError(
            "PLY is missing gaussian parameters required for .splat (need scales, rotations, opacities, f_dc)"
        )

    n = len(positions)
    sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacities))
    scales_world = np.exp(scales_log)
    scale_max = scales_world.max(axis=1)
    keep = np.ones(n, dtype=bool)
    if min_opacity > 0.0:
        keep &= sigmoid_opacity >= float(min_opacity)
    if max_scale is not None and max_scale > 0.0:
        keep &= scale_max <= float(max_scale)
    adaptive_max_scale = None
    if max_scale_percentile is not None:
        percentile = float(max_scale_percentile)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("max_scale_percentile must be in the range (0, 100]")
        adaptive_max_scale = float(np.percentile(scale_max, percentile))
        keep &= scale_max <= adaptive_max_scale
    kept_idx = np.nonzero(keep)[0]
    if kept_idx.size == 0:
        raise ValueError(
            "No gaussians survived filtering "
            f"(min_opacity={min_opacity}, max_scale={max_scale}, max_scale_percentile={max_scale_percentile})"
        )
    score = np.exp(scales_log[kept_idx].sum(axis=1)) * sigmoid_opacity[kept_idx]
    order = kept_idx[np.argsort(-score)]
    if max_points is not None and max_points > 0 and order.size > max_points:
        order = order[:max_points]
    n_out = int(order.shape[0])
    logger.info(
        "ply_to_splat: %d/%d gaussians after opacity>=%.2f / scale<=%s / scale_percentile<=%s (%s) (written: %d)",
        int(kept_idx.size),
        n,
        float(min_opacity),
        max_scale,
        max_scale_percentile,
        f"{adaptive_max_scale:.4f}" if adaptive_max_scale is not None else "off",
        n_out,
    )

    rot_raw = rotations[order].astype(np.float64)
    rot_norm = np.linalg.norm(rot_raw, axis=1, keepdims=True)
    rot_norm = np.where(rot_norm == 0.0, 1.0, rot_norm)
    rot_u8 = np.clip(rot_raw / rot_norm * 128.0 + 128.0, 0, 255).astype(np.uint8)

    rgba = np.empty((n_out, 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors[order], 0.0, 1.0)
    rgba[:, 3] = np.clip(sigmoid_opacity[order], 0.0, 1.0)
    rgba_u8 = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)

    pos = positions[order].astype(np.float32)
    scale_shift = 0.0
    if normalize_target_extent is not None and normalize_target_extent > 0:
        centroid = pos.mean(axis=0)
        centered = pos - centroid
        extent = float(np.max(centered.max(axis=0) - centered.min(axis=0)))
        if extent > 0:
            factor = extent / float(normalize_target_extent)
            pos = (centered / factor).astype(np.float32)
            scale_shift = float(-np.log(factor))
            logger.info(
                "ply_to_splat: normalized scene extent %.2f -> %.2f (factor %.3f)",
                extent,
                float(normalize_target_extent),
                factor,
            )
    scale = np.exp(scales_log[order] + scale_shift).astype(np.float32)

    dtype = np.dtype(
        [
            ("pos", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rot", "u1", 4),
        ]
    )
    packed = np.empty(n_out, dtype=dtype)
    packed["pos"] = pos
    packed["scale"] = scale
    packed["rgba"] = rgba_u8
    packed["rot"] = rot_u8
    with open(dst, "wb") as f:
        f.write(packed.tobytes())
    logger.info("Exported %d gaussians to %s (%.1f KB)", n_out, dst, dst.stat().st_size / 1024)
    return str(dst)
