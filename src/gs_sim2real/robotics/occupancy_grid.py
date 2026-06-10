"""Export a trained 3DGS map as a nav2-compatible occupancy grid.

Projects the gaussians of a live-mapping session round onto the ground plane
and writes the standard ``map.pgm`` + ``map.yaml`` pair that nav2's
``map_server`` (and the classic ROS map tooling) loads directly.

Pose-free maps live in an arbitrary reconstruction gauge, so all metric knobs
are expressed in units of the **camera height above ground** — the one
physical anchor every drive or walk has. The ground plane itself is
estimated from the mapped camera poses: the world up vector is the mean of
the cameras' optical ``-y`` axes, and the ground level is a low percentile
of the gaussian heights below the trajectory.

Cells are occupied where enough gaussians fall inside the obstacle height
band, free where ground-level gaussians or the camera trajectory provide
evidence, and unknown elsewhere (trinary map, ``-1 / 0 / 100``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gs_sim2real.robotics.gsplat_render_server import sigmoid
from gs_sim2real.viewer.web_viewer import load_ply

_UNKNOWN = -1
_FREE = 0
_OCCUPIED = 100


@dataclass(frozen=True)
class GridParams:
    """Tuning knobs; heights and distances are in camera-height units."""

    resolution: float | None = None  # gauge units per cell (default: camera height / 20)
    min_opacity: float = 0.3
    obstacle_band: tuple[float, float] = (0.2, 2.0)
    free_band_max: float = 0.1
    ground_percentile: float = 30.0
    min_points_per_cell: int = 2
    free_radius: float = 1.0
    padding: float = 1.0
    # keep the swept camera corridor free even where gaussians mark obstacles —
    # the robot physically drove there, so occupied cells inside it are map
    # noise (draft-round floaters). Used by the navigation planner.
    trajectory_wins: bool = False


@dataclass
class OccupancyGridMap:
    """A trinary occupancy grid plus the 3D frame it was projected from."""

    data: np.ndarray  # (H, W) int8 with row 0 at min grid-y; -1 / 0 / 100
    resolution: float
    origin: tuple[float, float]  # grid-plane coords of the (0, 0) cell corner
    up: np.ndarray  # world up vector (unit)
    basis: np.ndarray  # 3x3 rows (e1, e2, up): world -> grid-plane coords
    ground_height: float  # along up, in gauge units
    camera_height: float  # camera height above ground, in gauge units

    @property
    def occupied_cells(self) -> int:
        return int(np.count_nonzero(self.data == _OCCUPIED))

    @property
    def free_cells(self) -> int:
        return int(np.count_nonzero(self.data == _FREE))


def camera_axes(qvec: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP world-to-camera qvec (wxyz) -> world-frame (up, forward) axes.

    The camera follows the optical convention, so up is ``-y`` and forward
    is ``+z`` of the camera frame expressed in world coordinates.
    """
    w, x, y, z = (float(v) for v in qvec)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z)) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    r_cw = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    r_wc = r_cw.T
    return -r_wc[:, 1], r_wc[:, 2]


def estimate_ground_frame(
    camera_centers: np.ndarray,
    camera_qvecs: Sequence[Sequence[float]],
    point_heights_fn,
    *,
    ground_percentile: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate (up, forward, ground_height, camera_height) from the trajectory.

    ``point_heights_fn(up)`` must return the gaussian heights along a
    candidate up vector; it is a callable so callers can avoid materializing
    the projection twice.
    """
    ups = []
    forwards = []
    for qvec in camera_qvecs:
        up, forward = camera_axes(qvec)
        ups.append(up)
        forwards.append(forward)
    up = np.mean(np.asarray(ups), axis=0)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        raise ValueError("camera orientations do not agree on an up direction")
    up /= up_norm

    camera_level = float(np.median(np.asarray(camera_centers, dtype=np.float64) @ up))
    heights = np.asarray(point_heights_fn(up), dtype=np.float64)
    below = heights[heights < camera_level]
    if below.size == 0:
        raise ValueError("no gaussians below the camera trajectory; cannot estimate the ground")
    ground = float(np.percentile(below, ground_percentile))
    camera_height = camera_level - ground
    if camera_height <= 0:
        raise ValueError("estimated ground is above the cameras; check the map")

    forward = np.mean(np.asarray(forwards), axis=0)
    forward -= float(forward @ up) * up
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:  # e.g. a camera looking straight down
        forward = np.cross(up, np.array([1.0, 0.0, 0.0]))
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-6:
            forward = np.cross(up, np.array([0.0, 1.0, 0.0]))
            forward_norm = float(np.linalg.norm(forward))
    forward /= forward_norm
    return up, forward, ground, camera_height


def _densify_polyline(xy: np.ndarray, step: float) -> np.ndarray:
    """Sample a polyline every ``step`` so disk stamps form a continuous sweep."""
    if len(xy) < 2:
        return xy
    samples = [xy[:1]]
    for start, end in zip(xy[:-1], xy[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(int(np.ceil(distance / step)), 1)
        ts = np.linspace(0.0, 1.0, count + 1)[1:, None]
        samples.append(start[None, :] + ts * (end - start)[None, :])
    return np.vstack(samples)


def build_occupancy_grid(
    points: np.ndarray,
    opacities: np.ndarray,
    camera_centers: np.ndarray,
    camera_qvecs: Sequence[Sequence[float]],
    *,
    params: GridParams | None = None,
) -> OccupancyGridMap:
    """Project gaussians onto the estimated ground plane as a trinary grid."""
    params = params or GridParams()
    points = np.asarray(points, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    if len(points) == 0 or len(camera_centers) == 0:
        raise ValueError("need gaussians and camera poses to build a grid")

    up, forward, ground, camera_height = estimate_ground_frame(
        camera_centers,
        camera_qvecs,
        lambda candidate_up: points @ candidate_up,
        ground_percentile=params.ground_percentile,
    )

    e1 = forward
    e2 = np.cross(up, e1)
    basis = np.stack([e1, e2, up])

    solid = opacities >= params.min_opacity
    heights = points @ up - ground
    band_lo, band_hi = (value * camera_height for value in params.obstacle_band)
    obstacle = solid & (heights >= band_lo) & (heights <= band_hi)
    ground_evidence = solid & (heights < params.free_band_max * camera_height)

    obstacle_xy = points[obstacle] @ basis[:2].T
    ground_xy = points[ground_evidence] @ basis[:2].T
    camera_xy = camera_centers @ basis[:2].T

    resolution = params.resolution if params.resolution is not None else camera_height / 20.0
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    margin = params.padding * camera_height
    stack = [camera_xy] + [xy for xy in (obstacle_xy, ground_xy) if len(xy)]
    all_xy = np.vstack(stack)
    min_xy = all_xy.min(axis=0) - margin
    max_xy = all_xy.max(axis=0) + margin
    width = max(int(np.ceil((max_xy[0] - min_xy[0]) / resolution)), 1)
    height = max(int(np.ceil((max_xy[1] - min_xy[1]) / resolution)), 1)

    def to_cells(xy: np.ndarray) -> np.ndarray:
        cells = np.floor((xy - min_xy) / resolution).astype(np.int64)
        cells[:, 0] = np.clip(cells[:, 0], 0, width - 1)
        cells[:, 1] = np.clip(cells[:, 1], 0, height - 1)
        return cells

    data = np.full((height, width), _UNKNOWN, dtype=np.int8)

    if len(ground_xy):
        cells = to_cells(ground_xy)
        data[cells[:, 1], cells[:, 0]] = _FREE

    radius_cells = max(int(round(params.free_radius * camera_height / resolution)), 1)
    offsets = np.arange(-radius_cells, radius_cells + 1)
    ox, oy = np.meshgrid(offsets, offsets)
    disk = np.stack([ox.ravel(), oy.ravel()], axis=1)
    disk = disk[(disk**2).sum(axis=1) <= radius_cells * radius_cells]
    camera_cells = to_cells(_densify_polyline(camera_xy, resolution))
    swept = (camera_cells[:, None, :] + disk[None, :, :]).reshape(-1, 2)
    swept[:, 0] = np.clip(swept[:, 0], 0, width - 1)
    swept[:, 1] = np.clip(swept[:, 1], 0, height - 1)
    data[swept[:, 1], swept[:, 0]] = _FREE

    if len(obstacle_xy):
        cells = to_cells(obstacle_xy)
        counts = np.zeros((height, width), dtype=np.int64)
        np.add.at(counts, (cells[:, 1], cells[:, 0]), 1)
        data[counts >= params.min_points_per_cell] = _OCCUPIED

    if params.trajectory_wins:
        data[swept[:, 1], swept[:, 0]] = _FREE

    return OccupancyGridMap(
        data=data,
        resolution=float(resolution),
        origin=(float(min_xy[0]), float(min_xy[1])),
        up=up,
        basis=basis,
        ground_height=ground,
        camera_height=float(camera_height),
    )


def write_map_files(grid: OccupancyGridMap, yaml_path: Path) -> tuple[Path, Path, Path]:
    """Write ``map.pgm`` / ``map.yaml`` (map_server format) + a frame sidecar.

    Returns (yaml_path, pgm_path, json_path). The PGM uses the standard
    trinary colors: 254 free, 0 occupied, 205 unknown; row 0 is the top of
    the image (max grid-y), and the yaml origin is the lower-left corner.
    """
    yaml_path = Path(yaml_path)
    if yaml_path.suffix != ".yaml":
        raise ValueError(f"output must be a .yaml path (map_server format): {yaml_path}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    pgm_path = yaml_path.with_suffix(".pgm")
    json_path = yaml_path.with_suffix(".json")

    image = np.full(grid.data.shape, 205, dtype=np.uint8)
    image[grid.data == _FREE] = 254
    image[grid.data == _OCCUPIED] = 0
    image = image[::-1]  # row 0 = max grid-y

    height, width = image.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    pgm_path.write_bytes(header + image.tobytes())

    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                "mode: trinary",
                f"resolution: {grid.resolution:.6f}",
                f"origin: [{grid.origin[0]:.6f}, {grid.origin[1]:.6f}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(
            {
                "up": grid.up.tolist(),
                "basis": grid.basis.tolist(),
                "ground_height": grid.ground_height,
                "camera_height": grid.camera_height,
                "resolution": grid.resolution,
                "origin": list(grid.origin),
                "note": (
                    "grid coords = world point @ basis[:2].T; heights along `up`. "
                    "Units are the map's reconstruction gauge (not metres) unless "
                    "the map was built with metric poses."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path, pgm_path, json_path


def export_occupancy_grid(
    session_dir: Path,
    yaml_path: Path,
    *,
    round_index: int | None = None,
    params: GridParams | None = None,
) -> tuple[OccupancyGridMap, Path]:
    """Build and write a nav2 map from a live-mapping session round."""
    from gs_sim2real.robotics.localize import load_mapped_records, resolve_live_map_session

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    records = load_mapped_records(session)
    ply = load_ply(session.round.ply_path)
    opacities = (
        sigmoid(np.asarray(ply.opacities, dtype=np.float32))
        if ply.opacities is not None
        else np.ones(len(ply.positions), dtype=np.float32)
    )
    grid = build_occupancy_grid(
        np.asarray(ply.positions),
        opacities,
        np.asarray([record.center for record in records], dtype=np.float64),
        [record.qvec for record in records],
        params=params,
    )
    yaml_out, _, _ = write_map_files(grid, Path(yaml_path))
    return grid, yaml_out
