"""Multi-goal inspection patrol inside a 3DGS occupancy map."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .occupancy_grid import OccupancyGridMap
from .splat_nav import NavParams, NavResult, Pose2D, grid_to_rgb, plan_path, run_navigation


@dataclass(frozen=True)
class PatrolParams:
    """Patrol knobs; distances are in camera-height gauge units."""

    return_to_start: bool = False
    nav: NavParams = field(default_factory=NavParams)


@dataclass(frozen=True)
class PatrolWaypoint:
    """One requested inspection stop in grid-plane coordinates."""

    goal_xy: tuple[float, float]
    source: str
    label: str


@dataclass
class PatrolResult:
    """Outcome of a multi-stop patrol."""

    stops: list[dict[str, Any]] = field(default_factory=list)
    segments: list[NavResult] = field(default_factory=list)
    total_steps: int = 0
    distance: float = 0.0
    reached_count: int = 0
    localization_fixes: int = 0


def evenly_spaced_keyframe_waypoints(
    keyframe_xy: np.ndarray,
    count: int,
    *,
    skip_first: int = 1,
) -> list[PatrolWaypoint]:
    """Choose evenly spaced mapped keyframes as inspection stops."""
    keyframe_xy = np.asarray(keyframe_xy, dtype=np.float64)
    if count <= 0 or len(keyframe_xy) == 0:
        return []
    start = min(max(int(skip_first), 0), len(keyframe_xy) - 1)
    indices = np.linspace(start, len(keyframe_xy) - 1, count).round().astype(int)
    unique_indices = list(dict.fromkeys(int(index) for index in indices))
    return [
        PatrolWaypoint(
            goal_xy=(float(keyframe_xy[index, 0]), float(keyframe_xy[index, 1])),
            source="keyframe",
            label=f"kf {index}",
        )
        for index in unique_indices
    ]


def waypoints_from_changes(
    report: dict[str, Any],
    grid: OccupancyGridMap,
    *,
    kinds: tuple[str, ...] = ("appeared",),
    limit: int | None = None,
) -> list[PatrolWaypoint]:
    """Convert change clusters to inspection stops by projecting world centroids onto the grid plane.

    With ``limit``, the largest clusters (by voxel count) win across all kinds.
    """
    candidates: list[tuple[int, PatrolWaypoint]] = []
    for kind in kinds:
        for index, cluster in enumerate(report.get(kind, []), start=1):
            if "centroid" not in cluster:
                continue
            centroid = np.asarray(cluster["centroid"], dtype=np.float64)
            if centroid.shape != (3,):
                continue
            xy = centroid @ grid.basis[:2].T
            waypoint = PatrolWaypoint(
                goal_xy=(float(xy[0]), float(xy[1])),
                source=kind,
                label=f"{kind} #{index}",
            )
            candidates.append((int(cluster.get("voxels", 0)), waypoint))
    if limit is not None and limit > 0 and len(candidates) > limit:
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]
    return [waypoint for _voxels, waypoint in candidates]


def parse_xy_waypoints(text: str) -> list[PatrolWaypoint]:
    """Parse ``x,y;x,y`` waypoint text into patrol stops."""
    waypoints: list[PatrolWaypoint] = []
    if not text or not text.strip():
        raise ValueError('expected --goals as "x,y;x,y" with at least one stop')

    for index, chunk in enumerate(text.split(";"), start=1):
        item = chunk.strip()
        if not item:
            raise ValueError('malformed --goals: empty stop; expected "x,y;x,y"')
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f'malformed --goals stop {index}: expected "x,y", got {item!r}')
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError as error:
            raise ValueError(f"malformed --goals stop {index}: coordinates must be numbers") from error
        waypoints.append(PatrolWaypoint(goal_xy=(x, y), source="xy", label=f"stop {index}"))

    return waypoints


def _segment_distance(start: Pose2D, frames: list[Any]) -> float:
    previous = start.position()
    distance = 0.0
    for frame in frames:
        current = frame.true_pose.position()
        distance += float(np.linalg.norm(current - previous))
        previous = current
    return distance


def run_patrol(
    grid: OccupancyGridMap,
    start: Pose2D,
    waypoints: list[PatrolWaypoint],
    *,
    params: PatrolParams,
    observe_fn: Callable[[Pose2D, int], Pose2D | None] | None = None,
    capture_fn: Callable[[int, PatrolWaypoint, Pose2D], str | None] | None = None,
) -> PatrolResult:
    """Drive a simulated robot through each requested inspection stop."""
    patrol_waypoints = list(waypoints)
    if params.return_to_start:
        patrol_waypoints.append(PatrolWaypoint((float(start.x), float(start.y)), "xy", "home"))

    result = PatrolResult()
    current_pose = Pose2D(float(start.x), float(start.y), float(start.yaw))

    for index, waypoint in enumerate(patrol_waypoints):
        goal_xy = np.asarray(waypoint.goal_xy, dtype=np.float64)
        stop = {
            "label": waypoint.label,
            "source": waypoint.source,
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "planned": False,
            "reached": False,
            "steps": 0,
            "capture": None,
        }

        try:
            path = plan_path(grid, current_pose.position(), goal_xy, params=params.nav)
        except ValueError:
            result.stops.append(stop)
            continue

        segment = run_navigation(
            grid,
            current_pose,
            goal_xy,
            observe_fn=observe_fn,
            params=params.nav,
            path_xy=path,
        )
        result.segments.append(segment)
        result.total_steps += segment.steps
        result.distance += _segment_distance(current_pose, segment.frames)
        result.localization_fixes += int(segment.localization_count)

        if segment.frames:
            last_pose = segment.frames[-1].true_pose
            current_pose = Pose2D(float(last_pose.x), float(last_pose.y), float(last_pose.yaw))

        capture = capture_fn(index, waypoint, current_pose) if capture_fn is not None else None
        stop.update(
            {
                "planned": True,
                "reached": bool(segment.reached),
                "steps": int(segment.steps),
                "capture": capture,
            }
        )
        if segment.reached:
            result.reached_count += 1
        result.stops.append(stop)

    return result


def patrol_trace_image(
    grid: OccupancyGridMap,
    result: PatrolResult,
    waypoints: list[PatrolWaypoint],
    *,
    width: int = 1600,
):
    """Top-down PIL image of planned paths, driven trails, and patrol stop numbers."""
    from PIL import Image, ImageDraw

    base = grid_to_rgb(grid)[::-1]
    image = Image.fromarray(base)
    scale = width / image.width
    image = image.resize((width, max(int(round(image.height * scale)), 1)), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    height_cells = grid.data.shape[0]

    def to_px(xy: tuple[float, float] | np.ndarray) -> tuple[float, float]:
        point = np.asarray(xy, dtype=np.float64)
        col = (point[0] - grid.origin[0]) / grid.resolution
        row = (point[1] - grid.origin[1]) / grid.resolution
        return float(col * scale), float((height_cells - 1 - row) * scale)

    for segment in result.segments:
        if len(segment.path_xy) >= 2:
            draw.line([to_px(point) for point in segment.path_xy], fill=(80, 200, 255), width=2)
    for segment in result.segments:
        trail = [to_px((frame.true_pose.x, frame.true_pose.y)) for frame in segment.frames]
        if len(trail) >= 2:
            draw.line(trail, fill=(110, 200, 130), width=3)

    start_xy = None
    for segment in result.segments:
        if len(segment.path_xy):
            start_xy = segment.path_xy[0]
            break
        if segment.frames:
            start_xy = segment.frames[0].true_pose.position()
            break
    if start_xy is not None:
        x, y = to_px(start_xy)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(80, 200, 255), width=3)

    colors = {
        "keyframe": (80, 200, 255),
        "language": (255, 150, 50),
        "appeared": (235, 80, 80),
        "disappeared": (90, 120, 235),
        "xy": (255, 255, 255),
    }
    drawn_waypoints = list(waypoints)
    if len(result.stops) > len(drawn_waypoints):
        for stop in result.stops[len(drawn_waypoints) :]:
            drawn_waypoints.append(
                PatrolWaypoint(
                    goal_xy=(float(stop["goal_xy"][0]), float(stop["goal_xy"][1])),
                    source=str(stop["source"]),
                    label=str(stop["label"]),
                )
            )

    for index, waypoint in enumerate(drawn_waypoints, start=1):
        x, y = to_px(waypoint.goal_xy)
        color = colors.get(waypoint.source, (255, 255, 255))
        radius = 9
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0), width=2)
        draw.text((x + radius + 3, y - radius - 2), str(index), fill=(0, 0, 0))

    return image


def draw_patrol_trace(
    grid: OccupancyGridMap,
    result: PatrolResult,
    waypoints: list[PatrolWaypoint],
    output_path: str | Path,
    *,
    width: int = 1600,
) -> Path:
    """Save a patrol trace PNG."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = patrol_trace_image(grid, result, waypoints, width=width)
    image.save(path)
    return path
