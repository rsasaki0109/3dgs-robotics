"""Replay-driven active mapping over a growing live 3DGS session.

This module closes the v1 exploration loop around the current live map: it
finds map frontiers, drives to a selected frontier, feeds a batch of recorded
camera frames to the live mapper, rebuilds, and checks whether new keyframes
actually grew toward the chased frontier.

The capture source in this demo is honest replay, not a real camera controller:
imagery is consumed in recorded order from a prior drive. The autonomous part is
the map-space decision of which frontier to chase, whether navigation can reach
it, and whether the next rebuild extended the session-gauge map toward that
frontier. If the replay has no useful remaining footage for a frontier, the
frontier is marked exhausted and avoided.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .gauge_alignment import apply_to_points, quat_multiply, read_gauge_transform, rotation_to_quat
from .localize import load_mapped_records, resolve_live_map_session
from .occupancy_grid import GridParams, OccupancyGridMap, build_occupancy_grid
from .splat_explore import Frontier, reachable_free_mask, select_frontier
from .splat_nav import NavParams, Pose2D, plan_path, run_navigation


@dataclass(frozen=True)
class ActiveMappingConfig:
    """Active mapping knobs; distances are in camera-height gauge units."""

    batch_frames: int = 8
    max_rounds: int = 12
    min_frontier_cells: int = 8
    growth_tolerance: float = 3.0
    nav: NavParams = field(default_factory=NavParams)


@dataclass
class SessionGrid:
    """Current live-map occupancy grid and session-gauge keyframe centers."""

    grid: OccupancyGridMap
    centers: np.ndarray
    round_index: int
    gaussian_count: int


GridLoader = Callable[[Path], SessionGrid]


def load_session_grid(session_dir: Path, *, round_index: int | None = None) -> SessionGrid:
    """Load the current live map as a session-gauge occupancy grid."""
    from .gsplat_render_server import sigmoid
    from gs_sim2real.viewer.web_viewer import load_ply

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    records = load_mapped_records(session)
    ply = load_ply(str(session.round.ply_path))

    positions = np.asarray(ply.positions, dtype=np.float64)
    opacities = (
        sigmoid(np.asarray(ply.opacities, dtype=np.float32))
        if ply.opacities is not None
        else np.ones(len(positions), dtype=np.float32)
    )

    loaded = read_gauge_transform(session.round.round_dir)
    transform = loaded[0] if loaded is not None else (1.0, np.eye(3), np.zeros(3))
    positions = apply_to_points(transform, positions)

    centers = np.asarray([record.center for record in records], dtype=np.float64)
    centers = apply_to_points(transform, centers)

    rotation = np.asarray(transform[1], dtype=np.float64)
    gauge_qvec = rotation_to_quat(rotation.T)
    qvecs = []
    for record in records:
        qvec = quat_multiply(np.asarray(record.qvec, dtype=np.float64), gauge_qvec)
        norm = float(np.linalg.norm(qvec))
        qvecs.append(qvec / norm if norm > 0 else qvec)

    grid = build_occupancy_grid(
        positions,
        opacities,
        centers,
        qvecs,
        params=GridParams(trajectory_wins=True),
    )
    return SessionGrid(
        grid=grid,
        centers=centers,
        round_index=session.round.round_index,
        gaussian_count=int(len(positions)),
    )


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(int(steps), 0)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def map_frontier_clusters(
    grid: OccupancyGridMap, reachable: np.ndarray, *, min_cells: int, near_steps: int = 3
) -> list[Frontier]:
    """Cluster free cells that border unknown map cells near reachable space.

    Robot-radius inflation blocks the band of free cells along the unknown
    boundary (``inflate_obstacles`` dilates unknown too), so frontier cells are
    not required to be reachable themselves — only within ``near_steps`` cells
    of the reachable region. The planner later snaps the goal to the nearest
    drivable cell.
    """
    data = np.asarray(grid.data)
    reachable = np.asarray(reachable, dtype=bool)
    height, width = data.shape

    unknown = data == -1
    adjacent = np.zeros(data.shape, dtype=bool)
    adjacent[1:, :] |= unknown[:-1, :]
    adjacent[:-1, :] |= unknown[1:, :]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    frontier = _dilate(reachable, near_steps) & (data == 0) & adjacent

    seen = np.zeros(data.shape, dtype=bool)
    clusters: list[Frontier] = []
    for start_row, start_col in zip(*np.nonzero(frontier)):
        start = (int(start_row), int(start_col))
        if seen[start]:
            continue
        cells: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([start])
        seen[start] = True
        while queue:
            cell = queue.popleft()
            cells.append(cell)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (cell[0] + dr, cell[1] + dc)
                    if (
                        0 <= neighbor[0] < height
                        and 0 <= neighbor[1] < width
                        and frontier[neighbor]
                        and not seen[neighbor]
                    ):
                        seen[neighbor] = True
                        queue.append(neighbor)
        if len(cells) < min_cells:
            continue
        centers = np.asarray([_cell_center_xy(grid, cell) for cell in cells], dtype=np.float64)
        clusters.append(Frontier(cells=cells, centroid_xy=np.mean(centers, axis=0), size=len(cells)))

    return clusters


class ActiveMappingDriver:
    """Decide, drive, capture replay frames, rebuild, and verify map growth."""

    def __init__(
        self,
        session: Any,
        frames: Iterator[tuple[np.ndarray, float]],
        config: ActiveMappingConfig,
        *,
        session_dir: Path | None = None,
        grid_loader: GridLoader = load_session_grid,
    ) -> None:
        self.session = session
        self.frames = iter(frames)
        self.config = config
        self.session_dir = Path(session_dir) if session_dir is not None else Path(session.workdir)
        self.grid_loader = grid_loader

        self.entries: list[dict[str, Any]] = []
        self.stop_reason = ""
        self.exhausted_frontiers: list[np.ndarray] = []
        self.robot_trail: list[np.ndarray] = []
        self.robot_world: np.ndarray | None = None
        self.robot_yaw = 0.0

    def bootstrap(self, initial_frames: int) -> dict[str, Any]:
        """Feed initial replay frames and build the first active-mapping round."""
        fed = 0
        accepted = 0
        for _ in range(max(int(initial_frames), 0)):
            try:
                image, timestamp = next(self.frames)
            except StopIteration:
                break
            fed += 1
            if self.session.add_frame(image, timestamp):
                accepted += 1

        if not self.session.build_pending_round():
            raise RuntimeError(
                "active mapping bootstrap failed: no round was built. Feed at least two accepted keyframes "
                "and check the reconstruction backend/keyframes for errors."
            )

        state = self.grid_loader(self.session_dir)
        if len(state.centers) == 0:
            raise RuntimeError("active mapping bootstrap failed: the rebuilt round has no mapped keyframe centers.")

        self.robot_world = np.asarray(state.centers[0], dtype=np.float64)
        self.robot_yaw = 0.0
        self.robot_trail = [self.robot_world.copy()]
        return {
            "frames_fed": fed,
            "frames_accepted": accepted,
            "built": True,
            "round_index": state.round_index,
            "keyframes": int(len(state.centers)),
            "gaussians": state.gaussian_count,
            "robot_world": self.robot_world.tolist(),
        }

    def step(self) -> dict[str, Any] | None:
        """Run one active-mapping decide-drive-capture-rebuild-verify cycle."""
        if self.robot_world is None:
            raise RuntimeError("call bootstrap() before step()")

        state = self.grid_loader(self.session_dir)
        grid = state.grid
        robot_xy = self.robot_world @ grid.basis[:2].T
        robot_pose = Pose2D(float(robot_xy[0]), float(robot_xy[1]), float(self.robot_yaw))
        robot_cell = _xy_to_cell(grid, robot_xy)

        reachable = reachable_free_mask(grid, robot_cell, robot_radius=self.config.nav.robot_radius)
        radius_cells = int(np.ceil(self.config.nav.robot_radius * grid.camera_height / grid.resolution))
        frontiers = map_frontier_clusters(
            grid, reachable, min_cells=self.config.min_frontier_cells, near_steps=radius_cells + 2
        )
        frontiers = [frontier for frontier in frontiers if not self._is_exhausted(grid, frontier)]

        goal = select_frontier(frontiers, robot_xy, scale=grid.camera_height)
        if goal is None:
            self.stop_reason = "no-frontiers"
            return None

        frontier_world = _xy_to_world(grid, goal.centroid_xy)
        entry: dict[str, Any] = {
            "round_index": state.round_index,
            "frontier_world": frontier_world.tolist(),
            "frontier_size": int(goal.size),
            "nav_reached": False,
            "driven": False,
            "frames_fed": 0,
            "frames_accepted": 0,
            "built": False,
            "new_keyframes": 0,
            "frontier_distance": None,
            "grew": False,
            "gaussians": state.gaussian_count,
        }

        goal_xy = np.asarray(goal.centroid_xy, dtype=np.float64)
        try:
            planned = plan_path(grid, robot_pose.position(), goal_xy, params=self.config.nav)
        except ValueError:
            self.exhausted_frontiers.append(frontier_world.copy())
            self.entries.append(entry)
            return entry

        nav_result = run_navigation(grid, robot_pose, goal_xy, params=self.config.nav, path_xy=planned)
        if nav_result.frames:
            last = nav_result.frames[-1].true_pose
            final_xy = np.array([last.x, last.y], dtype=np.float64)
            self.robot_yaw = float(last.yaw)
        else:
            final_xy = robot_xy
        self.robot_world = _xy_to_world(grid, final_xy)
        self.robot_trail.append(self.robot_world.copy())

        entry["nav_reached"] = bool(nav_result.reached)
        entry["driven"] = True

        fed = 0
        accepted = 0
        for _ in range(max(int(self.config.batch_frames), 0)):
            try:
                image, timestamp = next(self.frames)
            except StopIteration:
                break
            fed += 1
            if self.session.add_frame(image, timestamp):
                accepted += 1

        if fed == 0:
            self.stop_reason = "frames-exhausted"
            return None

        built = bool(self.session.build_pending_round())
        next_state = self.grid_loader(self.session_dir)
        # rounds re-stride the whole keyframe history, so the mapped-center set is
        # not append-only — growth is "the new round has a keyframe near the
        # chased frontier", not a tail-slice of the center list
        threshold = float(self.config.growth_tolerance) * float(grid.camera_height)
        if len(next_state.centers):
            frontier_distance = float(np.linalg.norm(next_state.centers - frontier_world, axis=1).min())
        else:
            frontier_distance = float("inf")
        grew = frontier_distance <= threshold
        if not grew:
            self.exhausted_frontiers.append(frontier_world.copy())

        entry.update(
            {
                "round_index": next_state.round_index,
                "frames_fed": fed,
                "frames_accepted": accepted,
                "built": built,
                "new_keyframes": max(int(len(next_state.centers)) - int(len(state.centers)), 0),
                "frontier_distance": frontier_distance,
                "grew": bool(grew),
                "gaussians": next_state.gaussian_count,
            }
        )
        self.entries.append(entry)
        return entry

    def run(self) -> dict[str, Any]:
        """Run active mapping until the configured cap or a stop condition."""
        while len(self.entries) < self.config.max_rounds:
            entry = self.step()
            if entry is None:
                break

        if not self.stop_reason and len(self.entries) >= self.config.max_rounds:
            self.stop_reason = "max-rounds"

        return {
            "entries": self.entries,
            "stop_reason": self.stop_reason,
            "exhausted_frontiers": [point.tolist() for point in self.exhausted_frontiers],
        }

    def _is_exhausted(self, grid: OccupancyGridMap, frontier: Frontier) -> bool:
        if not self.exhausted_frontiers:
            return False
        world = _xy_to_world(grid, frontier.centroid_xy)
        threshold = float(self.config.growth_tolerance) * float(grid.camera_height)
        return _any_point_near(np.asarray(self.exhausted_frontiers), world, threshold)


def _cell_center_xy(grid: OccupancyGridMap, cell: tuple[int, int]) -> np.ndarray:
    row, col = cell
    return np.array(
        [
            float(grid.origin[0]) + (float(col) + 0.5) * float(grid.resolution),
            float(grid.origin[1]) + (float(row) + 0.5) * float(grid.resolution),
        ],
        dtype=np.float64,
    )


def _xy_to_cell(grid: OccupancyGridMap, xy: np.ndarray) -> tuple[int, int]:
    xy = np.asarray(xy, dtype=np.float64)
    col = int(np.floor((xy[0] - float(grid.origin[0])) / float(grid.resolution)))
    row = int(np.floor((xy[1] - float(grid.origin[1])) / float(grid.resolution)))
    height, width = grid.data.shape
    return int(np.clip(row, 0, height - 1)), int(np.clip(col, 0, width - 1))


def _xy_to_world(grid: OccupancyGridMap, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    basis = np.asarray(grid.basis, dtype=np.float64)
    return xy[0] * basis[0] + xy[1] * basis[1] + float(grid.ground_height) * np.asarray(grid.up, dtype=np.float64)


def _any_point_near(points: np.ndarray, target: np.ndarray, threshold: float) -> bool:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return False
    points = points.reshape((-1, 3))
    distances = np.linalg.norm(points - np.asarray(target, dtype=np.float64), axis=1)
    return bool(np.any(distances <= float(threshold)))
