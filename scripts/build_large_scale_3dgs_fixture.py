#!/usr/bin/env python3
"""Build a regional browser .splat fixture from the shipped outdoor results.

The generated fixture is intentionally synthetic: it preserves the original
Gaussian records from production outdoor splats, samples each source
deterministically, and translates the records into a configurable X/Z mosaic.
The output can then be passed to ``3dgs-robotics splat-tile-catalog`` to create a
dynamic-map tile catalog with substantially more spatial coverage than a single
scene.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = REPO / "docs" / "assets" / "outdoor-demo"
DEFAULT_OUTPUT = REPO / "outputs" / "large-scale-3dgs" / "outdoor-production-grid.splat"
DEFAULT_REPORT = REPO / "outputs" / "large-scale-3dgs" / "outdoor-production-grid.report.json"
SPLAT_RECORD_BYTES = 32
DEFAULT_GRID_WIDTH = 5
DEFAULT_GRID_HEIGHT = 5


@dataclass(frozen=True, slots=True)
class SourceScene:
    source: str
    label: str


@dataclass(frozen=True, slots=True)
class SourcePlacement:
    source: str
    label: str
    grid_x: int
    grid_z: int
    source_index: int


PRODUCTION_SOURCES: tuple[SourceScene, ...] = (
    SourceScene("outdoor-demo.splat", "Autoware fused supervised"),
    SourceScene("outdoor-demo-dust3r.splat", "DUSt3R outdoor"),
    SourceScene("bag6-mast3r.splat", "MAST3R bag6"),
    SourceScene("mcd-tuhh-day04.splat", "DUSt3R MCD tuhh_day_04"),
    SourceScene("mcd-tuhh-day04-mast3r.splat", "MAST3R MCD tuhh_day_04"),
    SourceScene("mcd-ntu-day02-supervised.splat", "MCD ntu_day_02 supervised"),
    SourceScene("bag6-vggt-slam-20-15k.splat", "VGGT-SLAM bag6"),
    SourceScene("bag6-mast3r-slam-20-15k.splat", "MASt3R-SLAM bag6"),
    SourceScene("bag6-pi3x-20-15k.splat", "Pi3X bag6"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--grid-width", type=int, default=DEFAULT_GRID_WIDTH)
    parser.add_argument("--grid-height", type=int, default=DEFAULT_GRID_HEIGHT)
    parser.add_argument(
        "--max-splats-per-cell",
        "--max-splats-per-source",
        dest="max_splats_per_cell",
        type=int,
        default=70_000,
        help=(
            "Deterministic sample cap for each mosaic cell. Use 0 to keep all records for every cell. "
            "--max-splats-per-source is kept as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=42.0,
        help="Spacing between mosaic cell centers in X/Z coordinate units.",
    )
    parser.add_argument("--seed", type=int, default=20260608)
    return parser.parse_args()


def _bounds(values: np.ndarray) -> dict[str, float]:
    mins = values.min(axis=0)
    maxes = values.max(axis=0)
    return {
        "minX": float(mins[0]),
        "minY": float(mins[1]),
        "minZ": float(mins[2]),
        "maxX": float(maxes[0]),
        "maxY": float(maxes[1]),
        "maxZ": float(maxes[2]),
    }


def _repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _robust_xz_center(positions: np.ndarray) -> tuple[float, float]:
    x_low, z_low = np.percentile(positions[:, [0, 2]], 1.0, axis=0)
    x_high, z_high = np.percentile(positions[:, [0, 2]], 99.0, axis=0)
    return float((x_low + x_high) * 0.5), float((z_low + z_high) * 0.5)


def _select_records(data: bytes, max_splats: int, rng: np.random.Generator) -> bytearray:
    count = len(data) // SPLAT_RECORD_BYTES
    records = np.frombuffer(data, dtype=np.uint8).reshape(count, SPLAT_RECORD_BYTES)
    if 0 < max_splats < count:
        indices = rng.choice(count, size=max_splats, replace=False)
        indices.sort()
        records = records[indices]
    return bytearray(records.copy().tobytes())


def build_source_placements(*, grid_width: int, grid_height: int) -> tuple[SourcePlacement, ...]:
    """Create a deterministic broad mosaic from the shipped production source list."""
    if grid_width < 1 or grid_height < 1:
        raise ValueError("--grid-width and --grid-height must be >= 1")

    placements: list[SourcePlacement] = []
    for grid_z in range(grid_height):
        for grid_x in range(grid_width):
            source_index = (grid_x + grid_z * 3) % len(PRODUCTION_SOURCES)
            source = PRODUCTION_SOURCES[source_index]
            placements.append(
                SourcePlacement(
                    source=source.source,
                    label=source.label,
                    grid_x=grid_x,
                    grid_z=grid_z,
                    source_index=source_index,
                )
            )
    return tuple(placements)


def _translate_records(
    records: bytearray,
    *,
    target_x: float,
    target_z: float,
) -> tuple[dict[str, float], dict[str, float]]:
    count = len(records) // SPLAT_RECORD_BYTES
    positions = np.ndarray((count, 3), dtype="<f4", buffer=records, offset=0, strides=(SPLAT_RECORD_BYTES, 4))
    source_center_x, source_center_z = _robust_xz_center(positions)
    source_bounds = _bounds(positions.copy())
    positions[:, 0] += np.float32(target_x - source_center_x)
    positions[:, 2] += np.float32(target_z - source_center_z)
    return source_bounds, _bounds(positions.copy())


def build_fixture(
    *,
    asset_dir: Path,
    output: Path,
    report_path: Path,
    grid_width: int,
    grid_height: int,
    max_splats_per_cell: int,
    spacing: float,
    seed: int,
) -> dict[str, Any]:
    if max_splats_per_cell < 0:
        raise ValueError("--max-splats-per-cell must be >= 0")
    if spacing <= 0:
        raise ValueError("--spacing must be > 0")

    rng = np.random.default_rng(seed)
    source_placements = build_source_placements(grid_width=grid_width, grid_height=grid_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    placements: list[dict[str, Any]] = []
    combined_bounds: list[dict[str, float]] = []
    source_total = 0
    kept_total = 0
    max_grid_x = grid_width - 1
    max_grid_z = grid_height - 1

    with output.open("wb") as destination:
        for placement_index, placement in enumerate(source_placements):
            source_path = asset_dir / placement.source
            data = source_path.read_bytes()
            if len(data) % SPLAT_RECORD_BYTES:
                raise ValueError(f"{source_path} is not a 32-byte-per-gaussian .splat file")
            input_count = len(data) // SPLAT_RECORD_BYTES
            records = _select_records(data, max_splats_per_cell, rng)
            kept_count = len(records) // SPLAT_RECORD_BYTES
            target_x = (placement.grid_x - max_grid_x / 2.0) * spacing
            target_z = (placement.grid_z - max_grid_z / 2.0) * spacing
            source_bounds, translated_bounds = _translate_records(
                records,
                target_x=target_x,
                target_z=target_z,
            )
            destination.write(records)
            source_total += input_count
            kept_total += kept_count
            combined_bounds.append(translated_bounds)
            placements.append(
                {
                    "source": _repo_relative_or_absolute(source_path),
                    "label": placement.label,
                    "placementIndex": placement_index,
                    "sourceIndex": placement.source_index,
                    "gridIndex": {"x": placement.grid_x, "z": placement.grid_z},
                    "targetCenter": {"x": target_x, "z": target_z},
                    "inputSplatCount": input_count,
                    "keptSplatCount": kept_count,
                    "sourceBounds": source_bounds,
                    "translatedBounds": translated_bounds,
                }
            )

    mins = {axis: min(bounds[f"min{axis}"] for bounds in combined_bounds) for axis in ("X", "Y", "Z")}
    maxes = {axis: max(bounds[f"max{axis}"] for bounds in combined_bounds) for axis in ("X", "Y", "Z")}
    report = {
        "type": "large-scale-3dgs-composite-fixture",
        "output": _repo_relative_or_absolute(output),
        "sourceCount": len(PRODUCTION_SOURCES),
        "placementCount": len(source_placements),
        "inputSplatCount": source_total,
        "compositeSplatCount": kept_total,
        "outputBytes": output.stat().st_size,
        "sampling": {
            "maxSplatsPerCell": max_splats_per_cell,
            "seed": seed,
        },
        "grid": {
            "columns": grid_width,
            "rows": grid_height,
            "spacing": spacing,
            "worldBounds": {
                "minX": mins["X"],
                "maxX": maxes["X"],
                "minY": mins["Y"],
                "maxY": maxes["Y"],
                "minZ": mins["Z"],
                "maxZ": maxes["Z"],
            },
        },
        "placements": placements,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = _parse_args()
    report = build_fixture(
        asset_dir=args.asset_dir,
        output=args.output,
        report_path=args.report,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        max_splats_per_cell=args.max_splats_per_cell,
        spacing=args.spacing,
        seed=args.seed,
    )
    print(f"wrote {report['output']}")
    print(
        "composite: "
        f"{report['compositeSplatCount']:,} splats / "
        f"{report['placementCount']} placements from {report['sourceCount']} sources / "
        f"{report['outputBytes'] / 1_000_000:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
