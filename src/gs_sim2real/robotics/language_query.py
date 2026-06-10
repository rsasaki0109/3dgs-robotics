"""Open-vocabulary queries against a trained 3DGS map.

"Where is the car?" — CLIPSeg scores every mapped keyframe image against a
free-text prompt, the per-pixel relevance is lifted onto the gaussians by
projecting each center into the keyframes it is visible in (COLMAP poses +
intrinsics, occlusion ignored), and high-scoring gaussians are clustered
into 3D hits. The best hit doubles as a navigation goal, so
``query-map "car"`` composes with ``navigate --goal`` into
language-directed autonomous driving inside the map.

CLIPSeg (``CIDAS/clipseg-rd64-refined``) is lazy-loaded via ``transformers``
(install separately: ``pip install transformers``); everything else is pure
numpy and unit-testable without the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"


@dataclass(frozen=True)
class QueryParams:
    """Lifting/clustering knobs; sizes in camera-height units."""

    score_threshold: float = 0.4  # CLIPSeg sigmoid relevance in [0, 1]
    min_views: int = 1
    max_keyframes: int = 12
    voxel_size: float | None = None  # default: camera height / 4
    min_cluster_gaussians: int = 20


@dataclass
class QueryHit:
    """One spatial cluster of prompt-relevant gaussians."""

    centroid: tuple[float, float, float]  # map gauge
    extent: tuple[float, float, float]
    gaussians: int
    mean_score: float
    goal_xy: tuple[float, float]  # grid-plane coords for `navigate --goal`

    def to_json(self) -> dict[str, Any]:
        return {
            "centroid": list(self.centroid),
            "extent": list(self.extent),
            "gaussians": self.gaussians,
            "mean_score": self.mean_score,
            "goal_xy": list(self.goal_xy),
        }


@dataclass
class QueryResult:
    """All hits for one prompt, best first."""

    prompt: str
    hits: list[QueryHit] = field(default_factory=list)
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    views: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    camera_height: float = 0.0
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "camera_height": self.camera_height,
            "basis": self.basis.tolist(),
            "hits": [hit.to_json() for hit in self.hits],
            "scored_gaussians": int((self.views > 0).sum()),
            "note": "coordinates are in the map's reconstruction gauge; goal_xy feeds `navigate --goal x,y`",
        }


def clipseg_heatmap_fn(device: str = "cuda") -> Callable[[np.ndarray, str], np.ndarray]:
    """Build a (BGR image, prompt) -> relevance map [0, 1] callable via CLIPSeg."""
    import torch
    from PIL import Image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained(_CLIPSEG_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(_CLIPSEG_MODEL).to(torch_device).eval()

    def heatmap(image_bgr: np.ndarray, prompt: str) -> np.ndarray:
        import cv2

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = processor(text=[prompt], images=[pil], return_tensors="pt").to(torch_device)
        with torch.no_grad():
            logits = model(**inputs).logits
        relevance = torch.sigmoid(logits.squeeze()).cpu().numpy().astype(np.float32)
        return cv2.resize(relevance, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    return heatmap


def project_scores(
    points: np.ndarray,
    records: Sequence[Any],
    heatmaps: dict[str, np.ndarray],
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Lift per-keyframe relevance onto gaussian centers.

    Returns (mean score per gaussian, number of keyframes it projected into).
    Heatmaps are keyed by keyframe name and sized like the mapping camera.
    """
    points = np.asarray(points, dtype=np.float64)
    fx, fy, cx, cy = intrinsics
    totals = np.zeros(len(points))
    views = np.zeros(len(points), dtype=np.int64)
    for record in records:
        heatmap = heatmaps.get(record.name)
        if heatmap is None:
            continue
        height, width = heatmap.shape[:2]
        w, x, y, z = (float(v) for v in record.qvec)
        norm = float(np.sqrt(w * w + x * x + y * y + z * z)) or 1.0
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        r_cw = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        cam = points @ r_cw.T + np.asarray(record.tvec, dtype=np.float64)
        depth = cam[:, 2]
        valid = depth > 1e-6
        u = np.full(len(points), -1.0)
        v = np.full(len(points), -1.0)
        u[valid] = cam[valid, 0] / depth[valid] * fx + cx
        v[valid] = cam[valid, 1] / depth[valid] * fy + cy
        inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        cols = u[inside].astype(np.int64)
        rows = v[inside].astype(np.int64)
        totals[inside] += heatmap[rows, cols]
        views[inside] += 1
    scores = np.where(views > 0, totals / np.maximum(views, 1), 0.0)
    return scores, views


def cluster_hits(
    points: np.ndarray,
    scores: np.ndarray,
    views: np.ndarray,
    basis: np.ndarray,
    *,
    voxel_size: float,
    params: QueryParams,
) -> list[QueryHit]:
    """Group prompt-relevant gaussians into ranked spatial hits."""
    from gs_sim2real.robotics.change_detection import _cluster_voxels

    selected = (scores >= params.score_threshold) & (views >= params.min_views)
    if not selected.any():
        return []
    chosen = np.asarray(points, dtype=np.float64)[selected]
    chosen_scores = scores[selected]
    cells = np.floor(chosen / voxel_size).astype(np.int64)
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, cells)):
        by_cell.setdefault(cell, []).append(index)

    hits: list[QueryHit] = []
    for component in _cluster_voxels(set(by_cell)):
        member_indices = [index for cell in component for index in by_cell[cell]]
        if len(member_indices) < params.min_cluster_gaussians:
            continue
        members = chosen[member_indices]
        centroid = members.mean(axis=0)
        extent = members.max(axis=0) - members.min(axis=0)
        goal = centroid @ np.asarray(basis)[:2].T
        hits.append(
            QueryHit(
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                extent=(float(extent[0]), float(extent[1]), float(extent[2])),
                gaussians=len(member_indices),
                mean_score=float(chosen_scores[member_indices].mean()),
                goal_xy=(float(goal[0]), float(goal[1])),
            )
        )
    hits.sort(key=lambda hit: hit.mean_score * hit.gaussians, reverse=True)
    return hits


def query_map(
    session_dir: Path,
    prompt: str,
    *,
    round_index: int | None = None,
    params: QueryParams | None = None,
    heatmap_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    device: str = "cuda",
) -> tuple[QueryResult, np.ndarray]:
    """Run one open-vocabulary query against a session round. Returns (result, points)."""
    import cv2

    from gs_sim2real.robotics.camera_sim_node import camera_intrinsics_from_colmap
    from gs_sim2real.robotics.gsplat_render_server import sigmoid
    from gs_sim2real.robotics.localize import _load_cameras_txt, load_mapped_records, resolve_live_map_session
    from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame
    from gs_sim2real.viewer.web_viewer import load_ply

    params = params or QueryParams()
    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    records = load_mapped_records(session)
    if len(records) > params.max_keyframes:
        indices = np.linspace(0, len(records) - 1, num=params.max_keyframes, dtype=np.int64)
        records = [records[i] for i in indices]

    cam = _load_cameras_txt(session.round.cameras_txt)[records[0].camera_id]
    width, height, fx, fy, cx, cy = camera_intrinsics_from_colmap(cam)

    heatmap_fn = heatmap_fn or clipseg_heatmap_fn(device)
    heatmaps: dict[str, np.ndarray] = {}
    for record in records:
        image = cv2.imread(str(session.keyframes_dir / record.name))
        if image is None:
            continue
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        heatmaps[record.name] = heatmap_fn(image, prompt)
        logger.info("scored %s", record.name)

    ply = load_ply(session.round.ply_path)
    points = np.asarray(ply.positions, dtype=np.float64)
    opacities = (
        sigmoid(np.asarray(ply.opacities, dtype=np.float32)) if ply.opacities is not None else np.ones(len(points))
    )
    solid = np.asarray(opacities, dtype=np.float64) >= 0.3
    scores = np.zeros(len(points))
    views = np.zeros(len(points), dtype=np.int64)
    scores[solid], views[solid] = project_scores(points[solid], records, heatmaps, (fx, fy, cx, cy))

    all_records = load_mapped_records(session)
    centers = np.asarray([record.center for record in all_records], dtype=np.float64)
    up, forward, _ground, camera_height = estimate_ground_frame(
        centers,
        [record.qvec for record in all_records],
        lambda candidate_up: points @ candidate_up,
        ground_percentile=30.0,
    )
    basis = np.stack([forward, np.cross(up, forward), up])
    voxel_size = params.voxel_size if params.voxel_size is not None else camera_height / 4.0

    result = QueryResult(
        prompt=prompt,
        scores=scores,
        views=views,
        camera_height=float(camera_height),
        basis=basis,
    )
    result.hits = cluster_hits(points, scores, views, basis, voxel_size=voxel_size, params=params)
    return result, points


def write_query_preview(
    result: QueryResult,
    points: np.ndarray,
    output_path: Path,
    *,
    image_width: int = 1600,
) -> Path:
    """Top-down preview: map in gray, prompt relevance in red, best hit boxed."""
    from PIL import Image, ImageDraw

    basis2 = result.basis[:2]
    xy = np.asarray(points, dtype=np.float64) @ basis2.T
    min_xy = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - min_xy, 1e-9)
    scale = (image_width - 1) / span[0]
    height = max(int(np.ceil(span[1] * scale)) + 1, 1)

    cols = np.clip(((xy[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, image_width - 1)
    rows = np.clip(((xy[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)
    image = np.full((height, image_width, 3), 245, dtype=np.uint8)
    image[rows, cols] = (190, 190, 190)

    hot = result.scores >= 0.2
    if hot.any():
        intensity = np.clip(result.scores[hot], 0.0, 1.0)
        red = (255 * intensity).astype(np.uint8)
        image[rows[hot], cols[hot]] = np.stack(
            [
                np.full_like(red, 220),
                200 - (160 * intensity).astype(np.uint8),
                200 - (160 * intensity).astype(np.uint8),
            ],
            axis=1,
        )

    pil = Image.fromarray(image[::-1])
    draw = ImageDraw.Draw(pil)
    for rank, hit in enumerate(result.hits[:3]):
        center_xy = np.asarray(hit.centroid) @ basis2.T
        half = np.asarray(hit.extent[:2]) * 0.5 + result.camera_height * 0.1
        c0 = (center_xy[0] - half[0] - min_xy[0]) * scale
        c1 = (center_xy[0] + half[0] - min_xy[0]) * scale
        r0 = height - 1 - (center_xy[1] + half[1] - min_xy[1]) * scale
        r1 = height - 1 - (center_xy[1] - half[1] - min_xy[1]) * scale
        color = (200, 30, 30) if rank == 0 else (240, 150, 40)
        draw.rectangle([c0, r0, c1, r1], outline=color, width=3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    return output_path
