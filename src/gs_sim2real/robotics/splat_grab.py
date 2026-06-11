"""Grab and paste language-selected objects between 3DGS maps.

``splat-grab`` reuses the same language scoring and clustered mask as ``splat-clean``, but keeps the selected rows as a
standalone object splat. ``splat-paste`` places that object into a target map using the same raw-Ply Sim3 merge path as
``merge-maps``. Coordinates and scales remain in reconstruction gauge; the grab sidecar stores camera height and up
direction so paste can auto-scale across maps and land the object on the target ground plane.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gs_sim2real.robotics.localize import load_mapped_records, resolve_live_map_session
from gs_sim2real.robotics.map_merge import RawGaussianPly, merge_raw_gaussians, read_raw_gaussian_ply
from gs_sim2real.robotics.map_merge import write_raw_gaussian_ply
from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame
from gs_sim2real.robotics.splat_clean import CleanParams, removal_mask


def _xyz(raw: RawGaussianPly) -> np.ndarray:
    return raw.columns(["x", "y", "z"]).astype(np.float64)


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} has near-zero length")
    return vector / norm


def _keep_cluster_nearest(points: np.ndarray, mask: np.ndarray, voxel_size: float, target: np.ndarray) -> np.ndarray:
    """Restrict ``mask`` to the connected voxel component nearest to ``target``.

    A prompt like "car" often matches every sighting of a moving object along
    the drive; grabbing means taking ONE object, so keep only the component
    closest to the best query hit.
    """
    from collections import deque

    indices = np.nonzero(mask)[0]
    if len(indices) == 0 or voxel_size <= 0:
        return mask
    cells = np.floor(points[indices] / voxel_size).astype(np.int64)
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for position, cell in zip(indices, map(tuple, cells)):
        by_cell.setdefault(cell, []).append(int(position))

    seen: set[tuple[int, int, int]] = set()
    best_component: list[int] = []
    best_distance = float("inf")
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for start in by_cell:
        if start in seen:
            continue
        queue: deque[tuple[int, int, int]] = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            cell = queue.popleft()
            component.extend(by_cell[cell])
            for dx, dy, dz in offsets:
                neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if neighbor in by_cell and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        distance = float(np.linalg.norm(points[component].mean(axis=0) - target))
        if distance < best_distance:
            best_distance = distance
            best_component = component

    restricted = np.zeros_like(mask)
    restricted[best_component] = True
    return restricted


def grab_map(
    session_dir: Path,
    prompt: str,
    output_ply: Path,
    *,
    round_index: int | None = None,
    params: CleanParams | None = None,
    heatmap_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    device: str = "cuda",
    best_cluster: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Keep prompt-matching gaussians from a session round as a standalone object splat.

    Returns ``(stats, points, mask)`` so callers can render the same preview style as ``splat-clean``. The output object
    keeps the source map gauge; its sidecar stores enough gauge metadata for paste-time auto-scaling. With
    ``best_cluster`` (default), only the connected component nearest the best query hit is grabbed - a moving object
    sighted along the whole drive otherwise smears into one giant "object".
    """
    from gs_sim2real.robotics.language_query import QueryParams, query_map

    session_dir = Path(session_dir)
    output_ply = Path(output_ply)
    params = params or CleanParams()
    query_params = QueryParams(
        score_threshold=params.score_threshold,
        min_views=params.min_views,
        max_keyframes=params.max_keyframes,
        voxel_size=params.voxel_size,
        min_cluster_gaussians=params.min_cluster_gaussians,
    )
    result, points = query_map(
        session_dir,
        prompt,
        round_index=round_index,
        params=query_params,
        heatmap_fn=heatmap_fn,
        device=device,
    )

    session = resolve_live_map_session(session_dir, round_index=round_index)
    raw = read_raw_gaussian_ply(session.round.ply_path)
    if len(raw) != len(points):
        raise ValueError(f"PLY row mismatch: scored {len(points)} gaussians but raw file holds {len(raw)}")

    voxel_size = params.voxel_size if params.voxel_size is not None else result.camera_height / 4.0
    dilate_radius = params.dilate_camera_heights * result.camera_height
    mask = removal_mask(
        points,
        result.scores,
        result.views,
        voxel_size=voxel_size,
        dilate_radius=dilate_radius,
        params=params,
    )
    if best_cluster and result.hits and mask.any():
        target = np.asarray(result.hits[0].centroid, dtype=np.float64)
        cluster_voxel = max(voxel_size, dilate_radius)
        mask = _keep_cluster_nearest(np.asarray(points, dtype=np.float64), mask, cluster_voxel, target)
    grabbed = int(mask.sum())
    if grabbed == 0:
        raise ValueError(f'nothing matched "{prompt}" at threshold {params.score_threshold} - lower the threshold')

    object_raw = RawGaussianPly(properties=list(raw.properties), data=raw.data[mask].copy())
    output = write_raw_gaussian_ply(output_ply, object_raw)

    selected = np.asarray(points, dtype=np.float64)[mask]
    basis = np.asarray(result.basis, dtype=np.float64)
    up = _normalize(basis[2], name="source up")
    centroid = selected.mean(axis=0)
    bottom = float(np.min(selected @ up))
    sidecar = output.with_suffix(".json")
    payload = {
        "prompt": prompt,
        "source_session": str(session_dir),
        "source_round": int(session.round.round_index),
        "gaussians": grabbed,
        "camera_height": float(result.camera_height),
        "up": up.tolist(),
        "centroid": centroid.tolist(),
        "bottom": bottom,
        "created": datetime.now(timezone.utc).isoformat(),
        "clusters": len(result.hits),
        "basis": basis.tolist(),
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats = {
        "output": str(output),
        "sidecar": str(sidecar),
        "prompt": prompt,
        "grabbed": grabbed,
        "total": len(raw),
        "clusters": len(result.hits),
        "camera_height": float(result.camera_height),
        "basis": basis.tolist(),
    }
    return stats, np.asarray(points, dtype=np.float64), mask


def load_grab_sidecar(object_ply: Path) -> dict[str, Any]:
    """Load the JSON sidecar written next to a grabbed object PLY."""
    sidecar = Path(object_ply).with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"grab sidecar missing: {sidecar}. Re-run splat-grab; paste needs it for gauge-aware scaling."
        )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _rotation_about(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotating column vectors about ``axis``."""
    axis = _normalize(axis, name="rotation axis")
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    identity = np.eye(3)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * identity + s * skew + (1.0 - c) * np.outer(axis, axis)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the shortest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = _normalize(a, name="source vector")
    b = _normalize(b, name="target vector")
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3)
    if dot < -1.0 + 1e-10:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(a @ reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, reference), name="antiparallel axis")
        return _rotation_about(axis, math.pi)

    v = np.cross(a, b)
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    s2 = float(v @ v)
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / s2)


def placement_sim3(
    sidecar: dict[str, Any],
    *,
    up_t: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    ground_t: float,
    camera_height_t: float,
    at_xy: tuple[float, float],
    yaw_deg: float,
    scale: float | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the paste Sim3 in target gauge.

    ``at_xy`` is in the target grid-plane coordinates used by query and navigation. If ``scale`` is omitted, the source
    object is scaled by the ratio of target to source camera height.
    """
    src_up = _normalize(np.asarray(sidecar["up"], dtype=np.float64), name="source up")
    up_t = _normalize(up_t, name="target up")
    e1 = np.asarray(e1, dtype=np.float64)
    e1 = _normalize(e1 - float(e1 @ up_t) * up_t, name="target e1")
    e2 = _normalize(np.asarray(e2, dtype=np.float64), name="target e2")

    source_camera_height = float(sidecar["camera_height"])
    if source_camera_height <= 0:
        raise ValueError("grab sidecar camera_height must be positive")
    s = float(scale) if scale is not None else float(camera_height_t) / source_camera_height
    if s <= 0:
        raise ValueError("paste scale must be positive")

    align = _rotation_between(src_up, up_t)
    yaw = _rotation_about(up_t, math.radians(float(yaw_deg)))
    rotation = yaw @ align

    centroid = np.asarray(sidecar["centroid"], dtype=np.float64).reshape(3)
    bottom = float(sidecar["bottom"])
    anchor = centroid - (float(centroid @ src_up) - bottom) * src_up
    target_world = float(at_xy[0]) * e1 + float(at_xy[1]) * e2 + float(ground_t) * up_t
    translation = target_world - s * (rotation @ anchor)
    return s, rotation, translation


def paste_map(
    object_ply: Path,
    session_dir: Path,
    output_ply: Path,
    *,
    at_xy: tuple[float, float],
    yaw_deg: float = 0.0,
    scale: float | None = None,
    round_index: int | None = None,
    dc_only: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Paste a grabbed object into a target map.

    The merged map keeps the target session gauge. When ``scale`` is omitted, the grab sidecar's camera height is matched
    to the target camera height before concatenating the raw gaussian rows.
    """
    object_ply = Path(object_ply)
    session_dir = Path(session_dir)
    output_ply = Path(output_ply)

    sidecar = load_grab_sidecar(object_ply)
    object_raw = read_raw_gaussian_ply(object_ply)
    session = resolve_live_map_session(session_dir, round_index=round_index)
    records = load_mapped_records(session)
    camera_centers = np.asarray([record.center for record in records], dtype=np.float64)
    camera_qvecs = [record.qvec for record in records]

    target_raw = read_raw_gaussian_ply(session.round.ply_path)
    target_xyz = _xyz(target_raw)
    up_t, forward_t, ground_t, camera_height_t = estimate_ground_frame(
        camera_centers,
        camera_qvecs,
        lambda candidate_up: target_xyz @ candidate_up,
        ground_percentile=30.0,
    )
    up_t = _normalize(up_t, name="target up")
    e1 = _normalize(forward_t - float(forward_t @ up_t) * up_t, name="target forward")
    e2 = _normalize(np.cross(up_t, e1), name="target right")
    basis2 = np.stack([e1, e2])

    sim3 = placement_sim3(
        sidecar,
        up_t=up_t,
        e1=e1,
        e2=e2,
        ground_t=ground_t,
        camera_height_t=camera_height_t,
        at_xy=at_xy,
        yaw_deg=yaw_deg,
        scale=scale,
    )
    merged, dropped = merge_raw_gaussians(target_raw, object_raw, sim3, dedup_radius=0.0, dc_only_b=dc_only)
    output = write_raw_gaussian_ply(output_ply, merged)

    s, rotation, translation = sim3
    object_xyz_transformed = _xyz(object_raw) @ rotation.T * s + translation
    stats = {
        "output": str(output),
        "object": str(object_ply),
        "gaussians_target": len(target_raw),
        "gaussians_object": len(object_raw),
        "merged": len(merged),
        "dropped": dropped,
        "scale": float(s),
        "yaw_deg": float(yaw_deg),
        "at": [float(at_xy[0]), float(at_xy[1])],
        "note": "merged map keeps the target session's gauge",
    }
    return stats, target_xyz, object_xyz_transformed, basis2


def write_paste_preview(
    target_positions: np.ndarray,
    object_positions_transformed: np.ndarray,
    basis2_rows: np.ndarray,
    output_path: Path,
    *,
    image_width: int = 1600,
) -> Path:
    """Top-down preview: target map in gray, pasted object in orange."""
    from PIL import Image

    target_positions = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
    object_positions_transformed = np.asarray(object_positions_transformed, dtype=np.float64).reshape(-1, 3)
    basis2_rows = np.asarray(basis2_rows, dtype=np.float64).reshape(2, 3)

    target_xy = target_positions @ basis2_rows.T
    object_xy = object_positions_transformed @ basis2_rows.T
    stacks = [xy for xy in (target_xy, object_xy) if len(xy)]
    if not stacks:
        raise ValueError("need target or pasted object positions to render a preview")
    union = np.vstack(stacks)

    min_xy = np.percentile(union, 0.5, axis=0)
    max_xy = np.percentile(union, 99.5, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    pixel_scale = (image_width - 1) / span[0]
    height = max(int(np.ceil(span[1] * pixel_scale)) + 1, 1)

    image = np.full((height, image_width, 3), 245, dtype=np.uint8)
    if len(target_xy):
        cols = np.clip(((target_xy[:, 0] - min_xy[0]) * pixel_scale).astype(np.int64), 0, image_width - 1)
        rows = np.clip(((target_xy[:, 1] - min_xy[1]) * pixel_scale).astype(np.int64), 0, height - 1)
        image[rows, cols] = (185, 185, 185)
    if len(object_xy):
        cols = np.clip(((object_xy[:, 0] - min_xy[0]) * pixel_scale).astype(np.int64), 0, image_width - 1)
        rows = np.clip(((object_xy[:, 1] - min_xy[1]) * pixel_scale).astype(np.int64), 0, height - 1)
        image[rows, cols] = (235, 120, 30)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[::-1]).save(output_path)
    return output_path
