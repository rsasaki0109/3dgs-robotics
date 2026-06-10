"""Merge two 3DGS maps of the same place into one splat (collaborative mapping).

Two robots (or two patrols) each build their own pose-free map; this module
aligns map B onto map A's gauge with the same Sim3 machinery as change
detection — shared keyframes when the rounds overlap, the 3DGS localizer
across independent sessions — transforms B's gaussians (positions, rotations,
log-scales), optionally drops B gaussians that duplicate A's coverage, and
writes one merged gsplat PLY in map A's gauge.

The merge happens at the raw PLY property level so every attribute survives.
One caveat: spherical-harmonic rest coefficients (``f_rest_*``) are copied
unrotated — after a large gauge rotation B's view-dependent shading is
slightly off (the base color ``f_dc`` is rotation-invariant). ``--dc-only``
zeroes the rest coefficients for a fully consistent (if more matte) result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gs_sim2real.robotics.gauge_alignment import (
    Sim3,
    quat_multiply,
    rotation_to_quat,
)
from gs_sim2real.viewer.web_viewer import _read_ply_ascii, _read_ply_binary_le

logger = logging.getLogger(__name__)


@dataclass
class RawGaussianPly:
    """A gsplat PLY as raw property columns."""

    properties: list[str]
    data: np.ndarray  # (N, P) float32

    def column(self, name: str) -> np.ndarray:
        return self.data[:, self.properties.index(name)]

    def columns(self, names: list[str]) -> np.ndarray:
        indices = [self.properties.index(name) for name in names]
        return self.data[:, indices]

    def set_columns(self, names: list[str], values: np.ndarray) -> None:
        indices = [self.properties.index(name) for name in names]
        self.data[:, indices] = values

    def __len__(self) -> int:
        return int(self.data.shape[0])


def read_raw_gaussian_ply(path: Path) -> RawGaussianPly:
    """Read every vertex property of a PLY into a float matrix."""
    path = Path(path)
    with open(path, "rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = "ascii"
        num_vertices = 0
        properties: list[tuple[str, str]] = []
        for line in header:
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[2], parts[1]))
        if fmt == "ascii":
            data = _read_ply_ascii(handle, num_vertices, len(properties))
        elif fmt == "binary_little_endian":
            data = _read_ply_binary_le(handle, num_vertices, properties)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    return RawGaussianPly(properties=[name for name, _ in properties], data=np.asarray(data, dtype=np.float32))


def write_raw_gaussian_ply(path: Path, raw: RawGaussianPly) -> Path:
    """Write the raw property matrix as a binary little-endian float PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(raw)}"]
    header += [f"property float {name}" for name in raw.properties]
    header += ["end_header", ""]
    with open(path, "wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(np.ascontiguousarray(raw.data, dtype="<f4").tobytes())
    return path


def transform_raw_gaussians(raw: RawGaussianPly, transform: Sim3) -> RawGaussianPly:
    """Apply a Sim3 to positions/normals/rotations/log-scales in place-like copy."""
    scale, rotation, translation = transform
    out = RawGaussianPly(properties=list(raw.properties), data=raw.data.copy())

    positions = out.columns(["x", "y", "z"]).astype(np.float64)
    out.set_columns(["x", "y", "z"], positions @ rotation.T * scale + translation)

    if all(name in out.properties for name in ("nx", "ny", "nz")):
        normals = out.columns(["nx", "ny", "nz"]).astype(np.float64)
        out.set_columns(["nx", "ny", "nz"], normals @ rotation.T)

    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(name in out.properties for name in rot_names):
        quats = out.columns(rot_names).astype(np.float64)  # (w, x, y, z)
        q_rotation = rotation_to_quat(rotation)
        out.set_columns(rot_names, quat_multiply(np.broadcast_to(q_rotation, quats.shape), quats))

    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(name in out.properties for name in scale_names):
        out.set_columns(scale_names, out.columns(scale_names) + np.log(scale))

    return out


def duplicate_mask(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radius: float,
) -> np.ndarray:
    """True for B points whose voxel neighborhood already holds an A point."""
    if radius <= 0 or len(points_a) == 0 or len(points_b) == 0:
        return np.zeros(len(points_b), dtype=bool)
    cells_a = {tuple(cell) for cell in np.floor(np.asarray(points_a, dtype=np.float64) / radius).astype(np.int64)}
    cells_b = np.floor(np.asarray(points_b, dtype=np.float64) / radius).astype(np.int64)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    mask = np.zeros(len(points_b), dtype=bool)
    for index, cell in enumerate(map(tuple, cells_b)):
        for dx, dy, dz in offsets:
            if (cell[0] + dx, cell[1] + dy, cell[2] + dz) in cells_a:
                mask[index] = True
                break
    return mask


def merge_raw_gaussians(
    raw_a: RawGaussianPly,
    raw_b: RawGaussianPly,
    transform: Sim3,
    *,
    dedup_radius: float = 0.0,
    dc_only_b: bool = False,
) -> tuple[RawGaussianPly, int]:
    """Transform B onto A's gauge and concatenate. Returns (merged, deduped count)."""
    if raw_a.properties != raw_b.properties:
        raise ValueError("the two PLYs have different property layouts; merge needs maps trained by the same pipeline")
    moved_b = transform_raw_gaussians(raw_b, transform)

    dropped = 0
    if dedup_radius > 0:
        mask = duplicate_mask(
            raw_a.columns(["x", "y", "z"]),
            moved_b.columns(["x", "y", "z"]),
            dedup_radius,
        )
        dropped = int(mask.sum())
        moved_b = RawGaussianPly(properties=moved_b.properties, data=moved_b.data[~mask])

    if dc_only_b:
        rest = [name for name in moved_b.properties if name.startswith("f_rest_")]
        if rest:
            moved_b.set_columns(rest, np.zeros((len(moved_b), len(rest)), dtype=np.float32))

    merged = RawGaussianPly(
        properties=list(raw_a.properties),
        data=np.vstack([raw_a.data, moved_b.data]),
    )
    return merged, dropped


def merge_sessions(
    session_a_dir: Path,
    session_b_dir: Path,
    output_ply: Path,
    *,
    round_a: int | None = None,
    round_b: int | None = None,
    align: str = "auto",
    dedup_radius_camera_heights: float = 0.0,
    dc_only_b: bool = False,
    localize_config: Any | None = None,
) -> dict[str, Any]:
    """Align session B's round onto session A's and write one merged PLY."""
    from gs_sim2real.robotics.change_detection import align_by_localization, align_shared_keyframes
    from gs_sim2real.robotics.localize import load_mapped_records, resolve_live_map_session
    from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame

    session_a = resolve_live_map_session(Path(session_a_dir), round_index=round_a)
    session_b = resolve_live_map_session(Path(session_b_dir), round_index=round_b)
    if session_a.round.round_dir == session_b.round.round_dir:
        raise ValueError("both inputs resolve to the same round; nothing to merge")
    records_a = load_mapped_records(session_a)
    records_b = load_mapped_records(session_b)

    if align == "auto":
        try:
            transform, matched = align_shared_keyframes(session_a.round.images_txt, session_b.round.images_txt)
            align = "shared"
        except ValueError:
            transform, matched = align_by_localization(
                Path(session_a_dir), session_b, records_b, round_a=round_a, config=localize_config
            )
            align = "localize"
    elif align == "shared":
        transform, matched = align_shared_keyframes(session_a.round.images_txt, session_b.round.images_txt)
    elif align == "localize":
        transform, matched = align_by_localization(
            Path(session_a_dir), session_b, records_b, round_a=round_a, config=localize_config
        )
    else:
        raise ValueError(f"unknown alignment mode: {align!r}")

    raw_a = read_raw_gaussian_ply(session_a.round.ply_path)
    raw_b = read_raw_gaussian_ply(session_b.round.ply_path)

    dedup_radius = 0.0
    if dedup_radius_camera_heights > 0:
        centers = np.asarray([record.center for record in records_a], dtype=np.float64)
        positions_a = raw_a.columns(["x", "y", "z"]).astype(np.float64)
        _, _, _, camera_height = estimate_ground_frame(
            centers,
            [record.qvec for record in records_a],
            lambda up: positions_a @ up,
            ground_percentile=30.0,
        )
        dedup_radius = dedup_radius_camera_heights * camera_height

    merged, dropped = merge_raw_gaussians(raw_a, raw_b, transform, dedup_radius=dedup_radius, dc_only_b=dc_only_b)
    output = write_raw_gaussian_ply(Path(output_ply), merged)
    scale, _rotation, _translation = transform
    return {
        "output": str(output),
        "gaussians_a": len(raw_a),
        "gaussians_b": len(raw_b),
        "deduplicated": dropped,
        "merged": len(merged),
        "alignment": {"mode": align, "matched_keyframes": matched, "scale": float(scale)},
        "dedup_radius": dedup_radius,
    }
