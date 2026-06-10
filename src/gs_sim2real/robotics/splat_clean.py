"""Remove objects from a trained 3DGS map by language prompt.

Dynamic objects are the classic 3DGS mapping artifact: a car driving ahead
of the mapping rig smears into a ghost streak along the road. This module
turns the open-vocabulary query machinery into an eraser — CLIPSeg scores
every gaussian against a prompt (``language_query.query_map``), high-scoring
gaussians are grouped into spatial clusters, the clusters are dilated to
catch the low-opacity smear around them, and the matching rows are dropped
from the raw gsplat PLY so every surviving attribute is untouched.

``splat-clean "car"`` therefore writes a cleaned map in the same gauge —
ready for ``navigate``, ``export-grid``, or another mapping round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanParams:
    """Selection knobs; radii in camera-height units."""

    score_threshold: float = 0.5  # CLIPSeg sigmoid relevance in [0, 1]
    min_views: int = 1
    max_keyframes: int = 12
    voxel_size: float | None = None  # default: camera height / 4
    min_cluster_gaussians: int = 20
    dilate_camera_heights: float = 0.125  # pulls the transparent smear in too


def removal_mask(
    points: np.ndarray,
    scores: np.ndarray,
    views: np.ndarray,
    *,
    voxel_size: float,
    dilate_radius: float,
    params: CleanParams,
) -> np.ndarray:
    """True for gaussians to delete: clustered prompt hits plus a dilation shell."""
    from gs_sim2real.robotics.change_detection import _cluster_voxels
    from gs_sim2real.robotics.map_merge import duplicate_mask

    points = np.asarray(points, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    selected = (np.asarray(scores) >= params.score_threshold) & (np.asarray(views) >= params.min_views)
    if not selected.any():
        return mask

    selected_indices = np.flatnonzero(selected)
    cells = np.floor(points[selected_indices] / voxel_size).astype(np.int64)
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for local, cell in enumerate(map(tuple, cells)):
        by_cell.setdefault(cell, []).append(local)
    for component in _cluster_voxels(set(by_cell)):
        members = [local for cell in component for local in by_cell[cell]]
        if len(members) < params.min_cluster_gaussians:
            continue
        mask[selected_indices[members]] = True

    if dilate_radius > 0 and mask.any():
        mask |= duplicate_mask(points[mask], points, dilate_radius)
    return mask


def clean_map(
    session_dir: Path,
    prompt: str,
    output_ply: Path,
    *,
    round_index: int | None = None,
    params: CleanParams | None = None,
    heatmap_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    device: str = "cuda",
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Erase prompt-matching gaussians from a session round.

    Returns (stats, points, removed mask) so callers can render previews.
    """
    from gs_sim2real.robotics.language_query import QueryParams, query_map
    from gs_sim2real.robotics.localize import resolve_live_map_session
    from gs_sim2real.robotics.map_merge import RawGaussianPly, read_raw_gaussian_ply, write_raw_gaussian_ply

    params = params or CleanParams()
    query_params = QueryParams(
        score_threshold=params.score_threshold,
        min_views=params.min_views,
        max_keyframes=params.max_keyframes,
        voxel_size=params.voxel_size,
        min_cluster_gaussians=params.min_cluster_gaussians,
    )
    result, points = query_map(
        Path(session_dir),
        prompt,
        round_index=round_index,
        params=query_params,
        heatmap_fn=heatmap_fn,
        device=device,
    )

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
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

    cleaned = RawGaussianPly(properties=list(raw.properties), data=raw.data[~mask])
    output = write_raw_gaussian_ply(Path(output_ply), cleaned)
    stats = {
        "output": str(output),
        "prompt": prompt,
        "gaussians": len(raw),
        "removed": int(mask.sum()),
        "kept": len(cleaned),
        "clusters": len(result.hits),
        "camera_height": float(result.camera_height),
        "dilate_radius": float(dilate_radius),
        "basis": result.basis.tolist(),
    }
    return stats, points, mask


def write_clean_preview(
    points: np.ndarray,
    mask: np.ndarray,
    basis: np.ndarray,
    output_path: Path,
    *,
    image_width: int = 1600,
) -> Path:
    """Top-down preview: surviving map in gray, removed gaussians in red."""
    from PIL import Image

    xy = np.asarray(points, dtype=np.float64) @ np.asarray(basis)[:2].T
    min_xy = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - min_xy, 1e-9)
    scale = (image_width - 1) / span[0]
    height = max(int(np.ceil(span[1] * scale)) + 1, 1)

    cols = np.clip(((xy[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, image_width - 1)
    rows = np.clip(((xy[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)
    image = np.full((height, image_width, 3), 245, dtype=np.uint8)
    image[rows[~mask], cols[~mask]] = (190, 190, 190)
    image[rows[mask], cols[mask]] = (220, 40, 40)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[::-1]).save(output_path)
    return output_path
