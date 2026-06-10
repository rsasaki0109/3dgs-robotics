"""Detect scene changes between two trained 3DGS maps (inspection).

Aligns two maps of the same place into one gauge, voxelizes both gaussian
clouds, and reports where solid voxels exist in one map but not the other:
``appeared`` (new in B) and ``disappeared`` (gone from A). Typical uses:
two patrols of the same route, or two rebuild rounds of one live-mapping
session.

Alignment options:

- ``shared``: Sim3 from keyframes present in both rounds (same session, or
  any pair of rounds whose ``images.txt`` share filenames) — pure numpy,
  reuses :mod:`gs_sim2real.robotics.gauge_alignment`.
- ``localize``: localize map B's keyframe images against map A with the 3DGS
  localizer, then fit the Sim3 from the matched camera poses — works across
  independent sessions of the same place (GPU).
- ``none``: maps already share a gauge.

All metric knobs are in camera-height units (the same non-metric-gauge
anchor as the occupancy-grid export); heights are measured along the up
vector estimated from map A's camera poses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from gs_sim2real.robotics.gauge_alignment import (
    Sim3,
    apply_to_points,
    quat_to_rotation,
    similarity_from_poses,
)
from gs_sim2real.robotics.gsplat_render_server import sigmoid
from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame
from gs_sim2real.viewer.web_viewer import load_ply

logger = logging.getLogger(__name__)

_NEIGHBORS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


@dataclass(frozen=True)
class ChangeParams:
    """Tuning knobs; sizes and heights are in camera-height units."""

    voxel_size: float | None = None  # gauge units per voxel (default: camera height / 4)
    min_opacity: float = 0.3
    height_band: tuple[float, float] = (0.1, 3.0)
    min_count: int = 3  # gaussians per voxel to call it solid
    min_cluster_voxels: int = 4


@dataclass
class ChangeCluster:
    """One connected blob of changed voxels."""

    kind: str  # "appeared" | "disappeared"
    voxels: int
    points: int
    centroid: tuple[float, float, float]  # world coords (map A gauge)
    extent: tuple[float, float, float]  # bounding box size, gauge units

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "voxels": self.voxels,
            "points": self.points,
            "centroid": list(self.centroid),
            "extent": list(self.extent),
        }


@dataclass
class ChangeReport:
    """Voxel-diff summary in map A's gauge."""

    clusters: list[ChangeCluster] = field(default_factory=list)
    voxel_size: float = 0.0
    camera_height: float = 0.0
    up: np.ndarray = field(default_factory=lambda: np.zeros(3))
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))
    alignment: dict[str, Any] = field(default_factory=dict)

    @property
    def appeared(self) -> list[ChangeCluster]:
        return [c for c in self.clusters if c.kind == "appeared"]

    @property
    def disappeared(self) -> list[ChangeCluster]:
        return [c for c in self.clusters if c.kind == "disappeared"]

    def to_json(self) -> dict[str, Any]:
        return {
            "voxel_size": self.voxel_size,
            "camera_height": self.camera_height,
            "up": self.up.tolist(),
            "basis": self.basis.tolist(),
            "alignment": self.alignment,
            "appeared": [c.to_json() for c in self.appeared],
            "disappeared": [c.to_json() for c in self.disappeared],
            "note": "coordinates are in map A's reconstruction gauge (not metres unless mapped with metric poses)",
        }


def _voxel_counts(coords: np.ndarray, voxel_size: float) -> dict[tuple[int, int, int], int]:
    cells = np.floor(coords / voxel_size).astype(np.int64)
    counts: dict[tuple[int, int, int], int] = {}
    for cell in map(tuple, cells):
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def _cluster_voxels(voxels: set[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Connected components over the 26-neighborhood."""
    remaining = set(voxels)
    clusters: list[list[tuple[int, int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            x, y, z = frontier.pop()
            for dx, dy, dz in _NEIGHBORS:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    frontier.append(neighbor)
        clusters.append(component)
    return clusters


def detect_changes(
    points_a: np.ndarray,
    opacities_a: np.ndarray,
    points_b: np.ndarray,
    opacities_b: np.ndarray,
    camera_centers_a: np.ndarray,
    camera_qvecs_a: Sequence[Sequence[float]],
    *,
    params: ChangeParams | None = None,
) -> ChangeReport:
    """Voxel-diff two aligned gaussian clouds (both already in map A's gauge)."""
    params = params or ChangeParams()
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError("both maps need gaussians to compare")

    up, forward, ground, camera_height = estimate_ground_frame(
        np.asarray(camera_centers_a, dtype=np.float64),
        camera_qvecs_a,
        lambda candidate_up: points_a @ candidate_up,
        ground_percentile=30.0,
    )
    basis = np.stack([forward, np.cross(up, forward), up])
    voxel_size = params.voxel_size if params.voxel_size is not None else camera_height / 4.0
    if voxel_size <= 0:
        raise ValueError("voxel size must be positive")

    def solid_coords(points: np.ndarray, opacities: np.ndarray) -> np.ndarray:
        opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
        heights = points @ up - ground
        lo, hi = (v * camera_height for v in params.height_band)
        keep = (opacities >= params.min_opacity) & (heights >= lo) & (heights <= hi)
        return points[keep]

    kept_a = solid_coords(points_a, opacities_a)
    kept_b = solid_coords(points_b, opacities_b)
    counts_a = _voxel_counts(kept_a, voxel_size)
    counts_b = _voxel_counts(kept_b, voxel_size)

    solid_a = {cell for cell, count in counts_a.items() if count >= params.min_count}
    solid_b = {cell for cell, count in counts_b.items() if count >= params.min_count}
    appeared = {cell for cell in solid_b if cell not in counts_a}
    disappeared = {cell for cell in solid_a if cell not in counts_b}

    report = ChangeReport(voxel_size=float(voxel_size), camera_height=float(camera_height), up=up, basis=basis)
    for kind, voxels, counts in (("appeared", appeared, counts_b), ("disappeared", disappeared, counts_a)):
        for component in _cluster_voxels(voxels):
            if len(component) < params.min_cluster_voxels:
                continue
            cells = np.asarray(component, dtype=np.float64)
            centers = (cells + 0.5) * voxel_size
            centroid = centers.mean(axis=0)
            extent = (cells.max(axis=0) - cells.min(axis=0) + 1.0) * voxel_size
            report.clusters.append(
                ChangeCluster(
                    kind=kind,
                    voxels=len(component),
                    points=int(sum(counts[cell] for cell in component)),
                    centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                    extent=(float(extent[0]), float(extent[1]), float(extent[2])),
                )
            )
    report.clusters.sort(key=lambda c: c.voxels, reverse=True)
    return report


def align_shared_keyframes(images_txt_a: Path, images_txt_b: Path) -> tuple[Sim3, int]:
    """Sim3 mapping map B's gauge onto map A's via keyframes named in both."""
    from gs_sim2real.robotics.gauge_alignment import RoundPoses, shared_camera_indices

    poses_a = RoundPoses.from_images_txt(Path(images_txt_a))
    poses_b = RoundPoses.from_images_txt(Path(images_txt_b))
    b_ids, a_ids = shared_camera_indices(poses_b.names, poses_a.names)
    if len(b_ids) < 2:
        raise ValueError(
            f"only {len(b_ids)} shared keyframes between the two rounds; use --align localize for independent sessions"
        )
    transform = similarity_from_poses(
        poses_b.centers[b_ids],
        poses_b.rotations[b_ids],
        poses_a.centers[a_ids],
        poses_a.rotations[a_ids],
    )
    return transform, len(b_ids)


def align_by_localization(
    session_a_dir: Path,
    session_b: Any,
    records_b: Sequence[Any],
    *,
    round_a: int | None = None,
    max_keyframes: int = 8,
    max_seed_distance: float = 0.5,
    config: Any | None = None,
    localizer: Any | None = None,
) -> tuple[Sim3, int]:
    """Sim3 mapping map B's gauge onto map A's by localizing B's keyframes in A.

    ``localizer`` (an object with ``localize(image_bgr)``) is injectable for
    tests; by default a :class:`~gs_sim2real.robotics.localize.SessionLocalizer`
    is built for session A.
    """
    import cv2

    if localizer is None:
        from gs_sim2real.robotics.localize import SessionLocalizer

        localizer = SessionLocalizer(Path(session_a_dir), round_index=round_a, config=config)

    records = list(records_b)
    if len(records) > max_keyframes:
        indices = np.linspace(0, len(records) - 1, num=max_keyframes, dtype=np.int64)
        records = [records[i] for i in indices]

    centers_b: list[np.ndarray] = []
    rotations_b: list[np.ndarray] = []
    centers_a: list[np.ndarray] = []
    rotations_a: list[np.ndarray] = []
    for record in records:
        image = cv2.imread(str(Path(session_b.keyframes_dir) / record.name))
        if image is None:
            continue
        result = localizer.localize(image, query_name=record.name)
        if result.seed_distance > max_seed_distance:
            logger.info("skipping %s: seed distance %.3f (off-map)", record.name, result.seed_distance)
            continue
        r_cw_b = quat_to_rotation(np.asarray(record.qvec, dtype=np.float64))
        centers_b.append(np.asarray(record.center, dtype=np.float64))
        rotations_b.append(r_cw_b.T)
        r_cw_a = quat_to_rotation(np.asarray(result.qvec, dtype=np.float64))
        centers_a.append(np.asarray(result.center, dtype=np.float64))
        rotations_a.append(r_cw_a.T)

    if len(centers_b) < 2:
        raise ValueError(f"only {len(centers_b)} of map B's keyframes localized in map A; cannot fit an alignment")
    transform = similarity_from_poses(
        np.asarray(centers_b),
        np.asarray(rotations_b),
        np.asarray(centers_a),
        np.asarray(rotations_a),
    )
    return transform, len(centers_b)


def write_change_preview(
    report: ChangeReport,
    points_a: np.ndarray,
    points_b: np.ndarray,
    output_path: Path,
    *,
    image_width: int = 1600,
) -> Path:
    """Top-down preview: map A density in gray, appeared red, disappeared blue."""
    from PIL import Image

    basis2 = report.basis[:2]
    xy_a = np.asarray(points_a, dtype=np.float64) @ basis2.T
    xy_b = np.asarray(points_b, dtype=np.float64) @ basis2.T
    stack = np.vstack([xy_a, xy_b])
    min_xy = stack.min(axis=0)
    max_xy = stack.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    scale = (image_width - 1) / span[0]
    height = max(int(np.ceil(span[1] * scale)) + 1, 1)

    def to_pixels(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.clip(((xy[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, image_width - 1)
        rows = np.clip(((xy[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)
        return rows, cols

    image = np.full((height, image_width, 3), 245, dtype=np.uint8)
    rows, cols = to_pixels(xy_a)
    image[rows, cols] = (190, 190, 190)

    half = report.voxel_size * 0.5
    for cluster in report.clusters:
        color = (220, 40, 40) if cluster.kind == "appeared" else (40, 80, 220)
        centroid = np.asarray(cluster.centroid, dtype=np.float64)
        extent = np.asarray(cluster.extent, dtype=np.float64)
        corner_min = (centroid @ basis2.T) - extent[:2] * 0.5 - half
        corner_max = (centroid @ basis2.T) + extent[:2] * 0.5 + half
        c0 = int(np.clip((corner_min[0] - min_xy[0]) * scale, 0, image_width - 1))
        c1 = int(np.clip((corner_max[0] - min_xy[0]) * scale, 0, image_width - 1))
        r0 = int(np.clip((corner_min[1] - min_xy[1]) * scale, 0, height - 1))
        r1 = int(np.clip((corner_max[1] - min_xy[1]) * scale, 0, height - 1))
        image[r0 : r1 + 1, c0] = color
        image[r0 : r1 + 1, c1] = color
        image[r0, c0 : c1 + 1] = color
        image[r1, c0 : c1 + 1] = color

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[::-1]).save(output_path)
    return output_path


def _load_map(session_dir: Path, round_index: int | None):
    from gs_sim2real.robotics.localize import load_mapped_records, resolve_live_map_session

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    records = load_mapped_records(session)
    ply = load_ply(session.round.ply_path)
    points = np.asarray(ply.positions, dtype=np.float64)
    opacities = (
        sigmoid(np.asarray(ply.opacities, dtype=np.float32))
        if ply.opacities is not None
        else np.ones(len(points), dtype=np.float32)
    )
    return session, records, points, np.asarray(opacities, dtype=np.float64)


def detect_session_changes(
    session_a_dir: Path,
    session_b_dir: Path,
    *,
    round_a: int | None = None,
    round_b: int | None = None,
    align: str = "auto",
    params: ChangeParams | None = None,
    localize_config: Any | None = None,
    align_fn: Callable[..., tuple[Sim3, int]] | None = None,
) -> tuple[ChangeReport, np.ndarray, np.ndarray]:
    """Align two session rounds and diff them. Returns (report, points_a, aligned_points_b)."""
    session_a, records_a, points_a, opac_a = _load_map(session_a_dir, round_a)
    session_b, records_b, points_b, opac_b = _load_map(session_b_dir, round_b)
    if session_a.round.round_dir == session_b.round.round_dir:
        raise ValueError("both inputs resolve to the same round; nothing to compare")

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
        align_call = align_fn or align_by_localization
        transform, matched = align_call(
            Path(session_a_dir), session_b, records_b, round_a=round_a, config=localize_config
        )
    elif align == "none":
        from gs_sim2real.robotics.gauge_alignment import identity_sim3

        transform, matched = identity_sim3(), 0
    else:
        raise ValueError(f"unknown alignment mode: {align!r}")

    aligned_b = apply_to_points(transform, points_b)
    report = detect_changes(
        points_a,
        opac_a,
        aligned_b,
        opac_b,
        np.asarray([record.center for record in records_a], dtype=np.float64),
        [record.qvec for record in records_a],
        params=params,
    )
    scale, _rotation, _translation = transform
    report.alignment = {"mode": align, "matched_keyframes": matched, "scale": float(scale)}
    return report, points_a, aligned_b
