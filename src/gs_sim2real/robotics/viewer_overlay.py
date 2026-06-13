"""Export robot results as a browser splat-viewer overlay.

The Pages viewer (``splat.html``) renders ``scene.splat`` files whose
positions were sim3-chained onto the session gauge and then normalized
(``(p - centroid) / factor``, frozen on the session's first or last rebased
round — see ``LiveMapBuilder._session_normalize_params``). This module
rebuilds that mapping from session artifacts and projects robot results —
the mapped trajectory, a planned navigation path (``nav_result.json``),
open-vocabulary query hits (``query.json``) — into the splat frame, writing
one overlay JSON that ``splat.html?overlay=<url>`` draws on top of the
gaussians.

Caveat: rounds whose ``gauge_transform.json`` was rewritten by pose-graph
refinement *after* their ``scene.splat`` was exported can be slightly off;
the latest round (the one the live viewer shows) always matches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TRAJECTORY_COLOR = "#4da3ff"
PATH_COLOR = "#2ecc71"
GOAL_COLOR = "#f5c542"
HIT_COLOR = "#ff5050"
APPEARED_COLOR = "#3ad29f"  # Dynamic diff: present now, absent in the baseline
DISAPPEARED_COLOR = "#ff9f43"  # Dynamic diff: in the baseline, gone now


@dataclass
class SplatFrameMapper:
    """Maps round-gauge points into the round's ``scene.splat`` frame."""

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    centroid: np.ndarray  # (3,)
    factor: float

    def points(self, points: np.ndarray) -> np.ndarray:
        moved = np.asarray(points, dtype=np.float64) @ self.rotation.T * self.scale + self.translation
        return (moved - self.centroid) / self.factor

    def distance(self, distance: float) -> float:
        return float(distance) * self.scale / self.factor


def splat_frame_mapper(
    session: Any,
    *,
    target_extent: float = 17.0,
) -> SplatFrameMapper:
    """Rebuild the gauge -> scene.splat mapping for a resolved session round.

    The normalization anchor is the last rebased round at or before this one
    (the live mapper freezes the viewer frame there); its at-freeze transform
    is the identity, so the anchor's raw PLY drives the normalization.
    """
    from gs_sim2real.robotics.gauge_alignment import identity_sim3, read_gauge_transform
    from gs_sim2real.viewer.web_export import compute_splat_normalization
    from gs_sim2real.viewer.web_viewer import load_ply

    round_dir = session.round.round_dir
    loaded = read_gauge_transform(round_dir)
    scale, rotation, translation = loaded[0] if loaded is not None else identity_sim3()

    candidates = sorted(
        child for child in round_dir.parent.iterdir() if child.is_dir() and child.name <= round_dir.name
    )
    anchor_dir = None
    for candidate in candidates:
        entry = read_gauge_transform(candidate)
        if entry is not None and entry[1]:  # the chain rebased here
            anchor_dir = candidate
    if anchor_dir is None:
        anchor_dir = candidates[0] if candidates else round_dir

    anchor_ply = anchor_dir / "train" / "point_cloud.ply"
    if not anchor_ply.is_file():
        anchor_ply = session.round.ply_path
    positions = np.asarray(load_ply(str(anchor_ply)).positions, dtype=np.float64)
    centroid, factor = compute_splat_normalization(positions, float(target_extent))

    return SplatFrameMapper(
        scale=float(scale),
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        centroid=np.asarray(centroid, dtype=np.float64),
        factor=float(factor),
    )


def _extent_corners(centroid: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """The eight corners of an axis-aligned gauge-frame extent box around a centroid.

    Mapped into the splat frame these become a (possibly rotated) wireframe the
    viewer draws instead of a flat screen-space circle.
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    half = np.asarray(extent, dtype=np.float64) / 2.0
    return np.asarray(
        [
            centroid + (sx * half[0], sy * half[1], sz * half[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )


def _lift_plane_points(
    plane_xy: np.ndarray,
    basis: np.ndarray,
    centers: np.ndarray,
    camera_height: float,
    *,
    height_offset: float = 0.15,
) -> np.ndarray:
    """Grid-plane (x, y) -> 3D gauge points hugging the local road level.

    Heights follow the nearest mapped keyframe along the trajectory (sloping
    streets!), dropped by one camera height to the ground plus a small
    offset so the line stays visible above the road surface.
    """
    e1, e2, up = basis
    plane_xy = np.asarray(plane_xy, dtype=np.float64)
    centers_xy = centers @ np.stack([e1, e2]).T
    center_heights = centers @ up
    lifted = []
    for x, y in plane_xy:
        nearest = int(np.argmin(np.linalg.norm(centers_xy - (x, y), axis=1)))
        height = center_heights[nearest] - camera_height * (1.0 - height_offset)
        lifted.append(e1 * x + e2 * y + up * height)
    return np.asarray(lifted)


def build_overlay(
    session_dir: Path,
    output_json: Path,
    *,
    round_index: int | None = None,
    nav_json: Path | None = None,
    query_json: Path | None = None,
    changes_json: Path | None = None,
    target_extent: float = 17.0,
    include_trajectory: bool = True,
) -> dict[str, Any]:
    """Write a splat-frame overlay JSON; returns its summary stats."""
    from gs_sim2real.robotics.localize import load_mapped_records, resolve_live_map_session
    from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame
    from gs_sim2real.viewer.web_viewer import load_ply

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    records = load_mapped_records(session)
    centers = np.asarray([record.center for record in records], dtype=np.float64)
    mapper = splat_frame_mapper(session, target_extent=target_extent)

    polylines: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    ground_frame: tuple[np.ndarray, float] | None = None

    def grid_basis() -> tuple[np.ndarray, float]:
        nonlocal ground_frame
        if ground_frame is None:
            points = np.asarray(load_ply(session.round.ply_path).positions, dtype=np.float64)
            up, forward, _ground, camera_height = estimate_ground_frame(
                centers,
                [record.qvec for record in records],
                lambda candidate_up: points @ candidate_up,
                ground_percentile=30.0,
            )
            basis = np.stack([forward, np.cross(up, forward), up])
            ground_frame = (basis, float(camera_height))
        return ground_frame

    if include_trajectory:
        polylines.append(
            {
                "label": "mapped trajectory",
                "color": TRAJECTORY_COLOR,
                "points": mapper.points(centers).tolist(),
            }
        )

    if nav_json is not None:
        nav = json.loads(Path(nav_json).read_text(encoding="utf-8"))
        basis, camera_height = grid_basis()
        path = _lift_plane_points(np.asarray(nav["path_vertices"], dtype=np.float64), basis, centers, camera_height)
        polylines.append(
            {
                "label": "planned path",
                "color": PATH_COLOR,
                "points": mapper.points(path).tolist(),
            }
        )
        goal = _lift_plane_points(np.asarray([nav["goal"]], dtype=np.float64), basis, centers, camera_height)
        markers.append(
            {
                "label": "goal",
                "color": GOAL_COLOR,
                "position": mapper.points(goal)[0].tolist(),
                "radius": mapper.distance(camera_height * 0.5),
            }
        )

    if query_json is not None:
        query = json.loads(Path(query_json).read_text(encoding="utf-8"))
        for rank, hit in enumerate(query.get("hits", []), start=1):
            centroid = np.asarray(hit["centroid"], dtype=np.float64)
            extent = np.asarray(hit["extent"], dtype=np.float64)
            radius = max(float(np.mean(extent)) * 0.5, query.get("camera_height", 0.0) * 0.25)
            corners = _extent_corners(centroid, extent)
            markers.append(
                {
                    "label": f"{query['prompt']} #{rank} ({hit['mean_score']:.2f})",
                    "color": HIT_COLOR,
                    "position": mapper.points(np.asarray([centroid]))[0].tolist(),
                    "radius": mapper.distance(radius),
                    "box": mapper.points(corners).tolist(),
                }
            )

    if changes_json is not None:
        changes = json.loads(Path(changes_json).read_text(encoding="utf-8"))
        camera_height = float(changes.get("camera_height", 0.0))
        # Dynamic diff clusters live in map A's gauge — the same round-gauge the
        # mapper normalizes — so appeared/disappeared boxes land on the served
        # splat. Colour keeps the two kinds apart for the viewer.
        for kind, color in (("appeared", APPEARED_COLOR), ("disappeared", DISAPPEARED_COLOR)):
            for rank, cluster in enumerate(changes.get(kind, []), start=1):
                centroid = np.asarray(cluster["centroid"], dtype=np.float64)
                extent = np.asarray(cluster["extent"], dtype=np.float64)
                radius = max(float(np.mean(extent)) * 0.5, camera_height * 0.25)
                corners = _extent_corners(centroid, extent)
                markers.append(
                    {
                        "label": f"{kind} #{rank} ({int(cluster.get('points', 0))} pts)",
                        "color": color,
                        "position": mapper.points(np.asarray([centroid]))[0].tolist(),
                        "radius": mapper.distance(radius),
                        "box": mapper.points(corners).tolist(),
                    }
                )

    payload = {
        "frame": "splat",
        "session": str(session_dir),
        "round": session.round.round_dir.name,
        "polylines": polylines,
        "markers": markers,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output_json),
        "polylines": len(polylines),
        "markers": len(markers),
        "splat": str(session.round.round_dir / "scene.splat"),
    }
