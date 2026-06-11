"""Frontier-based autonomous exploration inside a static 3DGS occupancy map.

The occupancy grid is static: exploration grows the robot's observed region,
not the map itself. Coverage is measured over free cells reachable from the
start after robot-radius inflation, so unreachable pockets do not count.

All distances are in camera-height gauge units unless the reconstruction was
built with metric poses.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from gs_sim2real.robotics.occupancy_grid import OccupancyGridMap
from gs_sim2real.robotics.splat_nav import (
    NavParams,
    NavResult,
    Pose2D,
    cells_to_xy,
    grid_to_rgb,
    inflate_obstacles,
    nearest_free_cell,
    plan_path,
    run_navigation,
    xy_to_cell,
)

_FREE = 0
_UNKNOWN = -1
_OCCUPIED = 100


@dataclass(frozen=True)
class ExploreParams:
    """Exploration knobs; distances are in camera-height gauge units."""

    sensor_range: float = 4.0
    ray_count: int = 180
    observe_every: int = 5
    min_frontier_cells: int = 8
    coverage_target: float = 0.95
    max_goals: int = 30
    nav: NavParams = field(default_factory=NavParams)


@dataclass
class Frontier:
    """A connected frontier component."""

    cells: list[tuple[int, int]]
    centroid_xy: np.ndarray
    size: int


@dataclass
class ExploreResult:
    """Outcome of an autonomous exploration run."""

    segments: list[NavResult] = field(default_factory=list)
    waypoints: list[dict] = field(default_factory=list)
    observed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    reachable: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    coverage_fraction: float = 0.0
    coverage_history: list[tuple[int, float]] = field(default_factory=list)
    scans: list[tuple[int, float, float]] = field(default_factory=list)
    total_steps: int = 0
    distance: float = 0.0
    stop_reason: str = ""

    @property
    def goals_chosen(self) -> int:
        return len(self.waypoints)


def _radius_cells(grid: OccupancyGridMap, robot_radius: float) -> int:
    return max(int(math.ceil(float(robot_radius) * grid.camera_height / grid.resolution)), 0)


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def reachable_free_mask(grid: OccupancyGridMap, start_cell: tuple[int, int], *, robot_radius: float) -> np.ndarray:
    """Return 4-connected drivable free cells reachable from ``start_cell``."""
    blocked = inflate_obstacles(grid.data, _radius_cells(grid, robot_radius))
    start = nearest_free_cell(blocked, start_cell)
    reachable = np.zeros(grid.data.shape, dtype=bool)
    if start is None:
        return reachable

    queue: deque[tuple[int, int]] = deque([start])
    reachable[start] = True
    height, width = grid.data.shape
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + dr, col + dc)
            if (
                0 <= neighbor[0] < height
                and 0 <= neighbor[1] < width
                and not blocked[neighbor]
                and not reachable[neighbor]
            ):
                reachable[neighbor] = True
                queue.append(neighbor)
    return reachable


def visible_cells(grid: OccupancyGridMap, xy: np.ndarray, *, params: ExploreParams) -> np.ndarray:
    """Raycast a visibility scan from ``xy``.

    Occupied and unknown cells are marked visible, then stop the ray: the
    simulated camera cannot see through walls or unmapped fog.
    """
    xy = np.asarray(xy, dtype=np.float64)
    visible = np.zeros(grid.data.shape, dtype=bool)
    max_range = max(float(params.sensor_range) * grid.camera_height, 0.0)
    if max_range <= 0.0 or params.ray_count <= 0:
        visible[xy_to_cell(grid, xy)] = True
        return visible

    step = max(grid.resolution * 0.5, 1e-6)
    distances = np.arange(0.0, max_range + step, step)
    last_cell_by_ray: tuple[int, int] | None
    for angle in np.linspace(0.0, 2.0 * math.pi, int(params.ray_count), endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        last_cell_by_ray = None
        for distance in distances:
            cell = xy_to_cell(grid, xy + direction * distance)
            if cell == last_cell_by_ray:
                continue
            last_cell_by_ray = cell
            if not _in_bounds(grid.data.shape, cell):
                break
            visible[cell] = True
            value = int(grid.data[cell])
            if value == _OCCUPIED or value == _UNKNOWN:
                break
    return visible


def frontier_clusters(
    grid: OccupancyGridMap,
    observed: np.ndarray,
    reachable: np.ndarray,
    *,
    min_cells: int,
) -> list[Frontier]:
    """Find 8-connected frontier clusters.

    A frontier cell is reachable free space that has not yet been observed
    and is 4-adjacent to an observed free cell.
    """
    free = grid.data == _FREE
    observed_free = observed & free
    frontier = reachable & free & ~observed

    adjacent = np.zeros_like(frontier, dtype=bool)
    adjacent[1:, :] |= observed_free[:-1, :]
    adjacent[:-1, :] |= observed_free[1:, :]
    adjacent[:, 1:] |= observed_free[:, :-1]
    adjacent[:, :-1] |= observed_free[:, 1:]
    frontier &= adjacent

    seen = np.zeros_like(frontier, dtype=bool)
    clusters: list[Frontier] = []
    height, width = frontier.shape
    for start_row, start_col in zip(*np.nonzero(frontier)):
        start = (int(start_row), int(start_col))
        if seen[start]:
            continue
        cells: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([start])
        seen[start] = True
        while queue:
            row, col = queue.popleft()
            cells.append((row, col))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (row + dr, col + dc)
                    if (
                        0 <= neighbor[0] < height
                        and 0 <= neighbor[1] < width
                        and frontier[neighbor]
                        and not seen[neighbor]
                    ):
                        seen[neighbor] = True
                        queue.append(neighbor)
        if len(cells) >= min_cells:
            xy = cells_to_xy(grid, cells)
            clusters.append(Frontier(cells=cells, centroid_xy=xy.mean(axis=0), size=len(cells)))
    return clusters


def select_frontier(frontiers: list[Frontier], current_xy: np.ndarray, *, scale: float = 1.0) -> Frontier | None:
    """Pick the highest utility frontier: larger and nearer is better.

    ``scale`` softens the distance penalty and should be the map's camera
    height so the trade-off is gauge-independent.
    """
    if not frontiers:
        return None
    current_xy = np.asarray(current_xy, dtype=np.float64)

    def utility(frontier: Frontier) -> float:
        distance = float(np.linalg.norm(frontier.centroid_xy - current_xy))
        return frontier.size / (distance + max(scale, 1e-9))

    return max(frontiers, key=utility)


def _coverage(grid: OccupancyGridMap, observed: np.ndarray, reachable: np.ndarray, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    covered = int(np.count_nonzero(observed & reachable & (grid.data == _FREE)))
    return covered / denominator


def _scan(
    grid: OccupancyGridMap,
    observed: np.ndarray,
    pose: Pose2D,
    global_step: int,
    scans: list[tuple[int, float, float]],
    params: ExploreParams,
) -> None:
    observed |= visible_cells(grid, pose.position(), params=params)
    scans.append((int(global_step), float(pose.x), float(pose.y)))


def _segment_distance(segment: NavResult) -> float:
    if len(segment.frames) < 2:
        return 0.0
    points = np.asarray([[frame.true_pose.x, frame.true_pose.y] for frame in segment.frames], dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def run_exploration(
    grid: OccupancyGridMap,
    start: Pose2D,
    *,
    params: ExploreParams,
    observe_fn: Callable[[Pose2D, int], Pose2D | None] | None = None,
) -> ExploreResult:
    """Explore reachable free space by repeatedly choosing frontier goals."""
    observed = np.zeros(grid.data.shape, dtype=bool)
    start_cell = xy_to_cell(grid, start.position())
    reachable = reachable_free_mask(grid, start_cell, robot_radius=params.nav.robot_radius)
    denominator = int(np.count_nonzero(reachable & (grid.data == _FREE)))

    result = ExploreResult(observed=observed, reachable=reachable)
    current_pose = Pose2D(start.x, start.y, start.yaw)
    global_step = 0
    skipped: set[tuple[int, int]] = set()
    consecutive_failures = 0

    _scan(grid, observed, current_pose, global_step, result.scans, params)
    result.coverage_fraction = _coverage(grid, observed, reachable, denominator)
    result.coverage_history.append((global_step, result.coverage_fraction))

    goals_consumed = 0
    while goals_consumed < params.max_goals:
        coverage = _coverage(grid, observed, reachable, denominator)
        result.coverage_fraction = coverage
        if result.coverage_history[-1] != (global_step, coverage):
            result.coverage_history.append((global_step, coverage))
        if coverage >= params.coverage_target:
            result.stop_reason = "coverage-target"
            break

        frontiers = [
            frontier
            for frontier in frontier_clusters(grid, observed, reachable, min_cells=params.min_frontier_cells)
            if not all(cell in skipped for cell in frontier.cells)
        ]
        if not frontiers:
            result.stop_reason = "all-frontiers-unreachable" if skipped else "no-frontiers"
            break

        goal = select_frontier(frontiers, current_pose.position(), scale=grid.camera_height)
        if goal is None:
            result.stop_reason = "no-frontiers"
            break

        try:
            path_xy = plan_path(grid, current_pose.position(), goal.centroid_xy, params=params.nav)
        except ValueError:
            skipped.update(goal.cells)
            if all(all(cell in skipped for cell in frontier.cells) for frontier in frontiers):
                result.stop_reason = "all-frontiers-unreachable"
                break
            continue

        goals_consumed += 1
        segment = run_navigation(
            grid,
            current_pose,
            goal.centroid_xy,
            observe_fn=observe_fn,
            params=params.nav,
            path_xy=path_xy,
        )
        result.segments.append(segment)
        result.total_steps += segment.steps
        result.distance += _segment_distance(segment)

        for index, frame in enumerate(segment.frames):
            step = global_step + index + 1
            if index % max(int(params.observe_every), 1) == 0 or index == len(segment.frames) - 1:
                _scan(grid, observed, frame.true_pose, step, result.scans, params)
        global_step += segment.steps

        if segment.frames:
            last = segment.frames[-1].true_pose
            current_pose = Pose2D(last.x, last.y, last.yaw)

        coverage = _coverage(grid, observed, reachable, denominator)
        result.coverage_fraction = coverage
        result.coverage_history.append((global_step, coverage))
        result.waypoints.append(
            {
                "goal_xy": goal.centroid_xy.tolist(),
                "frontier_cells": goal.size,
                "reached": bool(segment.reached),
                "steps": segment.steps,
                "coverage": coverage,
            }
        )

        if segment.reached:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                result.stop_reason = "stuck"
                break
    else:
        result.stop_reason = "max-goals"

    if not result.stop_reason:
        result.stop_reason = "max-goals"
    result.coverage_fraction = _coverage(grid, observed, reachable, denominator)
    result.observed = observed
    result.reachable = reachable
    return result


def _frontier_pixels(grid: OccupancyGridMap, observed: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    clusters = frontier_clusters(grid, observed, reachable, min_cells=1)
    mask = np.zeros(grid.data.shape, dtype=bool)
    for cluster in clusters:
        for cell in cluster.cells:
            mask[cell] = True
    return mask


def explore_trace_image(grid: OccupancyGridMap, result: ExploreResult, *, width: int = 1600):
    """Top-down PIL image of observed space, driven trail, and chosen goals."""
    from PIL import Image, ImageDraw

    base = grid_to_rgb(grid)
    observed_free = result.observed & (grid.data == _FREE)
    blended = base.astype(np.float32)
    blended[observed_free] = 0.55 * blended[observed_free] + 0.45 * np.array([130, 220, 150], dtype=np.float32)

    frontier = _frontier_pixels(grid, result.observed, result.reachable)
    blended[frontier] = np.array([255, 225, 60], dtype=np.float32)

    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)[::-1])
    scale = width / image.width
    image = image.resize((width, max(int(round(image.height * scale)), 1)), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    height_cells = grid.data.shape[0]

    def to_px(xy) -> tuple[float, float]:
        col = (xy[0] - grid.origin[0]) / grid.resolution
        row = (xy[1] - grid.origin[1]) / grid.resolution
        return col * scale, (height_cells - 1 - row) * scale

    first_pose: tuple[float, float] | None = None
    for segment in result.segments:
        trail = [to_px((frame.true_pose.x, frame.true_pose.y)) for frame in segment.frames]
        if first_pose is None and segment.frames:
            first_pose = (segment.frames[0].true_pose.x, segment.frames[0].true_pose.y)
        if len(trail) >= 2:
            draw.line(trail, fill=(75, 180, 95), width=3)

    if first_pose is None and result.scans:
        first_pose = (result.scans[0][1], result.scans[0][2])
    if first_pose is not None:
        x, y = to_px(first_pose)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(96, 205, 255), width=3)

    for index, waypoint in enumerate(result.waypoints):
        x, y = to_px(waypoint["goal_xy"])
        radius = 8
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 150, 50), width=3)
        draw.text((x + radius + 2, y - radius), str(index + 1), fill=(255, 150, 50))

    return image


def draw_explore_trace(grid: OccupancyGridMap, result: ExploreResult, output_path, *, width: int = 1600):
    """Write :func:`explore_trace_image` to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explore_trace_image(grid, result, width=width).save(output_path)
    return output_path


def _partial_trail(result: ExploreResult, global_step: int) -> list[tuple[float, float]]:
    trail: list[tuple[float, float]] = []
    offset = 0
    for segment in result.segments:
        for frame in segment.frames:
            if offset + frame.step + 1 <= global_step:
                trail.append((frame.true_pose.x, frame.true_pose.y))
        offset += segment.steps
    return trail


def explore_gif(
    grid: OccupancyGridMap,
    result: ExploreResult,
    output_path,
    *,
    params: ExploreParams | None = None,
    width: int = 900,
    scan_stride: int = 2,
) -> Path:
    """Write a GIF replay of progressively accumulated exploration scans."""
    from PIL import Image, ImageDraw

    params = params or ExploreParams()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not result.scans:
        explore_trace_image(grid, result, width=width).save(output_path)
        return output_path

    observed = np.zeros_like(result.observed, dtype=bool)
    frames = []
    selected = result.scans[:: max(int(scan_stride), 1)]
    if selected[-1] != result.scans[-1]:
        selected.append(result.scans[-1])

    height_cells = grid.data.shape[0]
    for global_step, x_world, y_world in selected:
        pose_xy = np.array([x_world, y_world], dtype=np.float64)
        observed |= visible_cells(grid, pose_xy, params=params)
        base = grid_to_rgb(grid)
        observed_free = observed & (grid.data == _FREE)
        blended = base.astype(np.float32)
        blended[observed_free] = 0.55 * blended[observed_free] + 0.45 * np.array([130, 220, 150], dtype=np.float32)

        image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)[::-1])
        scale = width / image.width
        image = image.resize((width, max(int(round(image.height * scale)), 1)), Image.NEAREST)
        draw = ImageDraw.Draw(image)

        def to_px(xy) -> tuple[float, float]:
            col = (xy[0] - grid.origin[0]) / grid.resolution
            row = (xy[1] - grid.origin[1]) / grid.resolution
            return col * scale, (height_cells - 1 - row) * scale

        trail = [to_px(point) for point in _partial_trail(result, global_step)]
        if len(trail) >= 2:
            draw.line(trail, fill=(75, 180, 95), width=3)
        px, py = to_px((x_world, y_world))
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 150, 50))
        frames.append(image.convert("P", palette=Image.ADAPTIVE))

    durations = [120] * (len(frames) - 1) + [2000] if len(frames) > 1 else [2000]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return output_path
