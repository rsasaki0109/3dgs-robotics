"""Autonomous navigation inside a trained 3DGS map — no external simulator.

The full loop runs against repo-built artifacts only: the occupancy grid
(:mod:`gs_sim2real.robotics.occupancy_grid`) supplies the planning space, an
A* planner and a pure-pursuit follower drive a unicycle robot, the GS camera
simulator renders what the robot sees, and the 3DGS localizer closes the
loop — **control consumes the localizer's estimate**, dead-reckoning on the
commanded motion between fixes, while the true pose is used only to render
observations (the simulation's ground truth).

All distance parameters are in camera-height units (the repo's standard
anchor for non-metric gauges); poses live in the grid plane (gauge units).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from gs_sim2real.robotics.gsplat_render_server import CameraPose
from gs_sim2real.robotics.occupancy_grid import OccupancyGridMap

_FREE = 0
_DIAG = math.sqrt(2.0)


@dataclass(frozen=True)
class NavParams:
    """Planner/controller knobs; distances in camera-height units, times in seconds."""

    robot_radius: float = 0.4
    lookahead: float = 1.2
    speed: float = 1.5  # camera heights per second
    dt: float = 0.1
    goal_tolerance: float = 1.0
    localize_every: int = 25  # control steps between localization fixes (0 = never)
    max_steps: int = 3000
    max_yaw_rate: float = 1.5  # rad/s
    # innovation gate: ignore fixes further than this from the current
    # estimate (visual aliasing on self-similar streets); 0 disables
    max_innovation: float = 5.0
    # wheel-slip simulation: per-step noise on the true motion (fractional on
    # velocity, rad/s on yaw rate). The estimate dead-reckons on the clean
    # commands, so it drifts — which is what the localization fixes correct.
    odom_noise: float = 0.0
    # how strongly an accepted fix pulls the estimate (1 = replace)
    fix_blend: float = 1.0
    seed: int = 7


@dataclass
class Pose2D:
    """Robot pose in grid-plane coords (gauge units)."""

    x: float
    y: float
    yaw: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class NavFrame:
    """One control step of the simulation."""

    step: int
    true_pose: Pose2D
    est_pose: Pose2D
    velocity: float
    yaw_rate: float
    localized: bool


@dataclass
class NavResult:
    """Outcome of a navigation run."""

    reached: bool
    frames: list[NavFrame] = field(default_factory=list)
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    localization_count: int = 0

    @property
    def steps(self) -> int:
        return len(self.frames)

    def cross_track_errors(self) -> np.ndarray:
        """Distance from each true pose to the nearest planned-path segment."""
        if not self.frames or len(self.path_xy) == 0:
            return np.zeros(0)
        poses = np.asarray([[f.true_pose.x, f.true_pose.y] for f in self.frames])
        if len(self.path_xy) < 2:
            return np.linalg.norm(poses - self.path_xy[0], axis=1)
        starts = self.path_xy[:-1][None, :, :]
        spans = (self.path_xy[1:] - self.path_xy[:-1])[None, :, :]
        lengths_sq = np.maximum((spans**2).sum(axis=2), 1e-12)
        t = np.clip(((poses[:, None, :] - starts) * spans).sum(axis=2) / lengths_sq, 0.0, 1.0)
        closest = starts + t[:, :, None] * spans
        return np.linalg.norm(poses[:, None, :] - closest, axis=2).min(axis=1)


# ----------------------------------------------------------------- planning


def inflate_obstacles(data: np.ndarray, radius_cells: int) -> np.ndarray:
    """Blocked mask with occupied/unknown cells dilated by the robot radius."""
    blocked = np.asarray(data) != _FREE
    if radius_cells <= 0:
        return blocked
    height, width = blocked.shape
    inflated = blocked.copy()
    rows, cols = np.nonzero(blocked)
    offsets = np.arange(-radius_cells, radius_cells + 1)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    disk = (oy**2 + ox**2) <= radius_cells**2
    dy = oy[disk]
    dx = ox[disk]
    for row, col in zip(rows, cols):
        rr = np.clip(row + dy, 0, height - 1)
        cc = np.clip(col + dx, 0, width - 1)
        inflated[rr, cc] = True
    return inflated


def nearest_free_cell(blocked: np.ndarray, cell: tuple[int, int], max_radius: int = 50) -> tuple[int, int] | None:
    """Closest non-blocked (row, col) to ``cell``, searched in growing rings."""
    height, width = blocked.shape
    row0 = min(max(cell[0], 0), height - 1)
    col0 = min(max(cell[1], 0), width - 1)
    if not blocked[row0, col0]:
        return (row0, col0)
    for radius in range(1, max_radius + 1):
        best: tuple[float, tuple[int, int]] | None = None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                row, col = row0 + dr, col0 + dc
                if 0 <= row < height and 0 <= col < width and not blocked[row, col]:
                    dist = float(dr * dr + dc * dc)
                    if best is None or dist < best[0]:
                        best = (dist, (row, col))
        if best is not None:
            return best[1]
    return None


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8-connected A* over the free cells of a blocked mask. Cells are (row, col)."""
    height, width = blocked.shape
    if blocked[start] or blocked[goal]:
        return None
    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, _DIAG),
        (-1, 1, _DIAG),
        (1, -1, _DIAG),
        (1, 1, _DIAG),
    ]

    def heuristic(cell: tuple[int, int]) -> float:
        dr = abs(cell[0] - goal[0])
        dc = abs(cell[1] - goal[1])
        return max(dr, dc) + (_DIAG - 1.0) * min(dr, dc)

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        closed.add(current)
        for dr, dc, cost in moves:
            row, col = current[0] + dr, current[1] + dc
            if not (0 <= row < height and 0 <= col < width) or blocked[row, col]:
                continue
            neighbor = (row, col)
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), neighbor))
    return None


def _line_of_sight(blocked: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """All cells on the segment a-b are free (dense sampling)."""
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
    for t in np.linspace(0.0, 1.0, steps):
        row = int(round(a[0] + t * (b[0] - a[0])))
        col = int(round(a[1] + t * (b[1] - a[1])))
        if blocked[row, col]:
            return False
    return True


def shortcut_path(blocked: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy line-of-sight shortcutting of an A* cell path."""
    if len(path) <= 2:
        return path
    out = [path[0]]
    index = 0
    while index < len(path) - 1:
        best = index + 1
        for candidate in range(len(path) - 1, index, -1):
            if _line_of_sight(blocked, path[index], path[candidate]):
                best = candidate
                break
        out.append(path[best])
        index = best
    return out


def cells_to_xy(grid: OccupancyGridMap, cells: list[tuple[int, int]]) -> np.ndarray:
    """(row, col) cells -> grid-plane coords at cell centers (gauge units)."""
    array = np.asarray(cells, dtype=np.float64)
    xs = grid.origin[0] + (array[:, 1] + 0.5) * grid.resolution
    ys = grid.origin[1] + (array[:, 0] + 0.5) * grid.resolution
    return np.stack([xs, ys], axis=1)


def xy_to_cell(grid: OccupancyGridMap, xy: np.ndarray) -> tuple[int, int]:
    col = int(math.floor((xy[0] - grid.origin[0]) / grid.resolution))
    row = int(math.floor((xy[1] - grid.origin[1]) / grid.resolution))
    height, width = grid.data.shape
    return (min(max(row, 0), height - 1), min(max(col, 0), width - 1))


def plan_path(grid: OccupancyGridMap, start_xy: np.ndarray, goal_xy: np.ndarray, *, params: NavParams) -> np.ndarray:
    """Plan a smoothed grid-plane path between two coordinates (gauge units)."""
    radius_cells = max(int(round(params.robot_radius * grid.camera_height / grid.resolution)), 0)
    blocked = inflate_obstacles(grid.data, radius_cells)
    start = nearest_free_cell(blocked, xy_to_cell(grid, np.asarray(start_xy, dtype=np.float64)))
    goal = nearest_free_cell(blocked, xy_to_cell(grid, np.asarray(goal_xy, dtype=np.float64)))
    if start is None or goal is None:
        raise ValueError("start or goal has no reachable free cell nearby")
    cells = astar(blocked, start, goal)
    if cells is None:
        raise ValueError("no path between start and goal on the occupancy grid")
    return cells_to_xy(grid, shortcut_path(blocked, cells))


# ----------------------------------------------------------------- control


def step_unicycle(pose: Pose2D, velocity: float, yaw_rate: float, dt: float) -> Pose2D:
    """Integrate a unicycle model one step."""
    yaw = pose.yaw + yaw_rate * dt
    return Pose2D(
        x=pose.x + velocity * dt * math.cos(yaw),
        y=pose.y + velocity * dt * math.sin(yaw),
        yaw=math.atan2(math.sin(yaw), math.cos(yaw)),
    )


class PurePursuit:
    """Track a polyline path with a lookahead point."""

    def __init__(self, path_xy: np.ndarray, *, lookahead: float, speed: float, max_yaw_rate: float) -> None:
        self.path = np.asarray(path_xy, dtype=np.float64)
        self.lookahead = float(lookahead)
        self.speed = float(speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self._dense = self._densify(self.path, self.lookahead / 4.0)
        self._progress = 0

    @staticmethod
    def _densify(path: np.ndarray, step: float) -> np.ndarray:
        if len(path) < 2:
            return path
        samples = [path[:1]]
        for a, b in zip(path[:-1], path[1:]):
            distance = float(np.linalg.norm(b - a))
            count = max(int(math.ceil(distance / max(step, 1e-9))), 1)
            ts = np.linspace(0.0, 1.0, count + 1)[1:, None]
            samples.append(a[None, :] + ts * (b - a)[None, :])
        return np.vstack(samples)

    def target(self, pose: Pose2D) -> np.ndarray:
        """Lookahead point: first dense sample beyond the lookahead distance."""
        position = pose.position()
        # advance monotonic progress to the nearest dense sample ahead
        window = self._dense[self._progress :]
        distances = np.linalg.norm(window - position, axis=1)
        self._progress += int(np.argmin(distances))
        for index in range(self._progress, len(self._dense)):
            if np.linalg.norm(self._dense[index] - position) >= self.lookahead:
                return self._dense[index]
        return self._dense[-1]

    def command(self, pose: Pose2D) -> tuple[float, float]:
        """(velocity, yaw rate) toward the lookahead point."""
        target = self.target(pose)
        heading = math.atan2(target[1] - pose.y, target[0] - pose.x)
        alpha = math.atan2(math.sin(heading - pose.yaw), math.cos(heading - pose.yaw))
        distance = max(float(np.linalg.norm(target - pose.position())), 1e-9)
        yaw_rate = 2.0 * self.speed * math.sin(alpha) / distance
        yaw_rate = min(max(yaw_rate, -self.max_yaw_rate), self.max_yaw_rate)
        velocity = self.speed * max(math.cos(alpha), 0.1)
        return velocity, yaw_rate


# ----------------------------------------------------------------- 2D <-> 3D


def pose2d_to_camera_pose(
    pose: Pose2D,
    grid: OccupancyGridMap,
    *,
    height_above_ground: float = 1.0,
    height: float | None = None,
) -> CameraPose:
    """Grid-plane robot pose -> optical-convention camera pose in world coords.

    ``height_above_ground`` is in camera-height units (1.0 = same height as
    the mapping camera). ``height`` overrides it with an absolute height
    along ``up`` in gauge units — needed when the mapped ground is not
    planar (a sloping street) and the camera must follow the local road
    level. The camera looks along the robot heading.
    """
    e1, e2, up = grid.basis
    if height is None:
        height = grid.ground_height + height_above_ground * grid.camera_height
    world = pose.x * e1 + pose.y * e2 + height * up
    forward = math.cos(pose.yaw) * e1 + math.sin(pose.yaw) * e2
    down = -up
    right = np.cross(down, forward)
    rotation = np.stack([right, down, forward], axis=1)  # world-from-camera columns
    qw = math.sqrt(max(0.0, 1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])) / 2.0
    if qw < 1e-9:
        qx, qy, qz = 1.0, 0.0, 0.0
    else:
        qx = (rotation[2, 1] - rotation[1, 2]) / (4 * qw)
        qy = (rotation[0, 2] - rotation[2, 0]) / (4 * qw)
        qz = (rotation[1, 0] - rotation[0, 1]) / (4 * qw)
    return CameraPose(
        position=(float(world[0]), float(world[1]), float(world[2])),
        orientation=(float(qx), float(qy), float(qz), float(qw)),
    )


def world_to_pose2d(center: np.ndarray, rotation_wc: np.ndarray, grid: OccupancyGridMap) -> Pose2D:
    """World camera center + world-from-camera rotation -> grid-plane pose."""
    e1, e2, _up = grid.basis
    center = np.asarray(center, dtype=np.float64)
    forward = np.asarray(rotation_wc, dtype=np.float64)[:, 2]
    return Pose2D(
        x=float(center @ e1),
        y=float(center @ e2),
        yaw=math.atan2(float(forward @ e2), float(forward @ e1)),
    )


# ----------------------------------------------------------------- simulation


def run_navigation(
    grid: OccupancyGridMap,
    start: Pose2D,
    goal_xy: np.ndarray,
    *,
    observe_fn: Callable[[Pose2D, int], Pose2D | None] | None = None,
    params: NavParams | None = None,
    path_xy: np.ndarray | None = None,
    est_start: Pose2D | None = None,
) -> NavResult:
    """Drive a simulated robot along a planned path, closing the loop on estimates.

    ``observe_fn(true_pose, step)`` returns a localization fix (or ``None``
    to skip); between fixes the estimate dead-reckons on the commanded
    motion. The controller only ever sees the estimate.
    """
    params = params or NavParams()
    scale = grid.camera_height
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    if path_xy is None:
        path_xy = plan_path(grid, start.position(), goal_xy, params=params)
    follower = PurePursuit(
        path_xy,
        lookahead=params.lookahead * scale,
        speed=params.speed * scale,
        max_yaw_rate=params.max_yaw_rate,
    )

    result = NavResult(reached=False, path_xy=np.asarray(path_xy, dtype=np.float64))
    true_pose = Pose2D(start.x, start.y, start.yaw)
    est_pose = (
        Pose2D(est_start.x, est_start.y, est_start.yaw)
        if est_start is not None
        else Pose2D(start.x, start.y, start.yaw)
    )
    tolerance = params.goal_tolerance * scale
    rng = np.random.default_rng(params.seed)
    blend = min(max(params.fix_blend, 0.0), 1.0)
    for step in range(params.max_steps):
        localized = False
        if observe_fn is not None and params.localize_every > 0 and step % params.localize_every == 0:
            fix = observe_fn(true_pose, step)
            if fix is not None:
                innovation = float(np.linalg.norm(fix.position() - est_pose.position()))
                if params.max_innovation <= 0 or innovation <= params.max_innovation * scale:
                    sin_yaw = (1.0 - blend) * math.sin(est_pose.yaw) + blend * math.sin(fix.yaw)
                    cos_yaw = (1.0 - blend) * math.cos(est_pose.yaw) + blend * math.cos(fix.yaw)
                    est_pose = Pose2D(
                        x=(1.0 - blend) * est_pose.x + blend * fix.x,
                        y=(1.0 - blend) * est_pose.y + blend * fix.y,
                        yaw=math.atan2(sin_yaw, cos_yaw),
                    )
                    localized = True

        velocity, yaw_rate = follower.command(est_pose)
        # wheel slip: the true motion deviates from the commanded one; the
        # estimate integrates the clean command and therefore drifts
        true_velocity = velocity
        true_yaw_rate = yaw_rate
        if params.odom_noise > 0:
            true_velocity *= 1.0 + float(rng.normal(0.0, params.odom_noise))
            true_yaw_rate += float(rng.normal(0.0, params.odom_noise))
        true_pose = step_unicycle(true_pose, true_velocity, true_yaw_rate, params.dt)
        est_pose = step_unicycle(est_pose, velocity, yaw_rate, params.dt)
        result.frames.append(
            NavFrame(
                step=step,
                true_pose=true_pose,
                est_pose=est_pose,
                velocity=velocity,
                yaw_rate=yaw_rate,
                localized=localized,
            )
        )
        if localized:
            result.localization_count += 1
        if float(np.linalg.norm(true_pose.position() - goal_xy)) <= tolerance:
            result.reached = True
            break
    return result


def make_localize_observer(
    grid: OccupancyGridMap,
    renderer: Any,
    localizer: Any,
    *,
    render_size: tuple[int, int],
    intrinsics: tuple[float, float, float, float],
    camera_centers: np.ndarray | None = None,
    camera_height_units: float = 1.0,
    max_seed_distance: float = 0.5,
    frame_callback: Callable[[np.ndarray, Pose2D, Pose2D | None, int], None] | None = None,
) -> Callable[[Pose2D, int], Pose2D | None]:
    """Observer that renders the robot's view and localizes it in the map.

    ``camera_centers`` (the mapped keyframe centers, world coords) lets the
    virtual camera follow the local road level on non-planar ground: its
    height along ``up`` is taken from the nearest keyframe instead of a
    global ground+offset. ``frame_callback(rgb, true_pose, fix_or_none,
    step)`` receives every rendered observation (e.g. for a demo GIF).
    """
    import cv2

    from gs_sim2real.robotics.camera_sim_node import render_optical
    from gs_sim2real.robotics.gauge_alignment import quat_to_rotation

    keyframe_xy: np.ndarray | None = None
    keyframe_heights: np.ndarray | None = None
    if camera_centers is not None and len(camera_centers):
        centers = np.asarray(camera_centers, dtype=np.float64)
        keyframe_xy = centers @ grid.basis[:2].T
        keyframe_heights = centers @ grid.basis[2]

    def observe(true_pose: Pose2D, step: int) -> Pose2D | None:
        height = None
        if keyframe_xy is not None and keyframe_heights is not None:
            nearest = int(np.argmin(np.linalg.norm(keyframe_xy - true_pose.position(), axis=1)))
            base = float(keyframe_heights[nearest])
            height = base + (camera_height_units - 1.0) * grid.camera_height
        camera_pose = pose2d_to_camera_pose(true_pose, grid, height_above_ground=camera_height_units, height=height)
        rgb, _depth = render_optical(
            renderer,
            camera_pose,
            width=render_size[0],
            height=render_size[1],
            intrinsics=intrinsics,
            near_clip=0.001,
            far_clip=500.0,
            point_radius=1,
        )
        result = localizer.localize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), query_name=f"nav_{step:05d}")
        fix: Pose2D | None = None
        if result.seed_distance <= max_seed_distance:
            rotation_wc = quat_to_rotation(np.asarray(result.qvec, dtype=np.float64)).T
            fix = world_to_pose2d(np.asarray(result.center), rotation_wc, grid)
        if frame_callback is not None:
            frame_callback(rgb, true_pose, fix, step)
        return fix

    return observe


# ----------------------------------------------------------------- visualization


def grid_to_rgb(grid: OccupancyGridMap) -> np.ndarray:
    """Occupancy grid -> RGB (free white, occupied red, unknown gray); row 0 = min y."""
    image = np.empty((*grid.data.shape, 3), dtype=np.uint8)
    image[:] = (52, 58, 70)
    image[grid.data == 0] = (252, 252, 252)
    image[grid.data == 100] = (212, 80, 80)
    return image


def navigation_trace_image(
    grid: OccupancyGridMap,
    result: NavResult,
    *,
    width: int = 1600,
):
    """Top-down PIL image: planned path (cyan), driven trajectory (green), fixes (orange)."""
    from PIL import Image, ImageDraw

    base = grid_to_rgb(grid)[::-1]  # display with max grid-y on top
    image = Image.fromarray(base)
    scale = width / image.width
    image = image.resize((width, max(int(round(image.height * scale)), 1)), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    height_cells = grid.data.shape[0]

    def to_px(xy) -> tuple[float, float]:
        col = (xy[0] - grid.origin[0]) / grid.resolution
        row = (xy[1] - grid.origin[1]) / grid.resolution
        return col * scale, (height_cells - 1 - row) * scale

    if len(result.path_xy) >= 2:
        draw.line([to_px(p) for p in result.path_xy], fill=(80, 200, 255), width=3)
    trail = [to_px((f.true_pose.x, f.true_pose.y)) for f in result.frames]
    if len(trail) >= 2:
        draw.line(trail, fill=(110, 200, 130), width=3)
    for frame in result.frames:
        if frame.localized:
            x, y = to_px((frame.est_pose.x, frame.est_pose.y))
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 150, 60))
    if result.frames:
        x, y = to_px((result.frames[0].true_pose.x, result.frames[0].true_pose.y))
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(96, 205, 255), width=3)
        x, y = to_px((result.frames[-1].true_pose.x, result.frames[-1].true_pose.y))
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255), width=3)
    return image


def draw_navigation_trace(
    grid: OccupancyGridMap,
    result: NavResult,
    output_path,
    *,
    width: int = 1600,
):
    """Write :func:`navigation_trace_image` to ``output_path``."""
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    navigation_trace_image(grid, result, width=width).save(output_path)
    return output_path
