"""Tests for multi-goal inspection patrol inside a 3DGS map."""

from __future__ import annotations

import numpy as np
import pytest

from gs_sim2real.robotics.occupancy_grid import OccupancyGridMap
from gs_sim2real.robotics.patrol import (
    PatrolParams,
    PatrolWaypoint,
    evenly_spaced_keyframe_waypoints,
    parse_xy_waypoints,
    patrol_trace_image,
    run_patrol,
    waypoints_from_changes,
)
from gs_sim2real.robotics.splat_nav import NavParams, Pose2D


def _corridor_grid(height: int = 21, width: int = 61, *, camera_height: float = 1.0) -> OccupancyGridMap:
    """Free corridor with occupied walls on the long sides, unknown beyond."""
    data = np.full((height, width), -1, dtype=np.int8)
    data[3:-3, :] = 0
    data[3, :] = 100
    data[-4, :] = 100
    return OccupancyGridMap(
        data=data,
        resolution=0.25,
        origin=(0.0, 0.0),
        up=np.array([0.0, 0.0, 1.0]),
        basis=np.eye(3),
        ground_height=0.0,
        camera_height=camera_height,
    )


def _patrol_params(**kwargs) -> PatrolParams:
    nav = NavParams(
        robot_radius=0.0,
        speed=1.5,
        goal_tolerance=0.35,
        localize_every=0,
        max_steps=250,
        seed=3,
    )
    return PatrolParams(nav=nav, **kwargs)


def test_parse_xy_waypoints_ok_and_malformed() -> None:
    waypoints = parse_xy_waypoints("1.0,2; 3.5,4.25")

    assert [waypoint.goal_xy for waypoint in waypoints] == [(1.0, 2.0), (3.5, 4.25)]
    assert [waypoint.source for waypoint in waypoints] == ["xy", "xy"]
    assert [waypoint.label for waypoint in waypoints] == ["stop 1", "stop 2"]

    with pytest.raises(ValueError, match="expected"):
        parse_xy_waypoints("1,2,3")


def test_evenly_spaced_keyframe_waypoints_count_ordering_and_skip_first() -> None:
    keyframe_xy = np.array([[float(index), float(index + 10)] for index in range(8)])

    waypoints = evenly_spaced_keyframe_waypoints(keyframe_xy, 4, skip_first=2)

    assert [waypoint.label for waypoint in waypoints] == ["kf 2", "kf 4", "kf 5", "kf 7"]
    assert [waypoint.goal_xy for waypoint in waypoints] == [(2.0, 12.0), (4.0, 14.0), (5.0, 15.0), (7.0, 17.0)]


def test_waypoints_from_changes_projects_centroids_and_filters_kinds() -> None:
    grid = _corridor_grid()
    report = {
        "appeared": [
            {"centroid": [1.0, 2.0, 9.0]},
            {"centroid": [3.0, 4.0, 8.0]},
        ],
        "disappeared": [
            {"centroid": [5.0, 6.0, 7.0]},
        ],
    }

    waypoints = waypoints_from_changes(report, grid, kinds=("disappeared", "appeared"))

    assert [waypoint.source for waypoint in waypoints] == ["disappeared", "appeared", "appeared"]
    assert [waypoint.label for waypoint in waypoints] == ["disappeared #1", "appeared #1", "appeared #2"]
    assert [waypoint.goal_xy for waypoint in waypoints] == [(5.0, 6.0), (1.0, 2.0), (3.0, 4.0)]


def test_run_patrol_reaches_two_waypoints_and_advances() -> None:
    grid = _corridor_grid()
    start = Pose2D(1.0, 2.5, 0.0)
    waypoints = [
        PatrolWaypoint((4.0, 2.5), "xy", "stop 1"),
        PatrolWaypoint((8.0, 2.5), "xy", "stop 2"),
    ]

    result = run_patrol(grid, start, waypoints, params=_patrol_params())

    assert result.reached_count == 2
    assert len(result.stops) == 2
    assert len(result.segments) == 2
    assert result.total_steps > 0
    assert result.distance > 0.0
    assert result.segments[-1].frames[-1].true_pose.x > start.x


def test_unreachable_waypoint_records_unplanned_and_patrol_continues() -> None:
    grid = _corridor_grid()
    grid.data[4:-4, 30] = 100
    start = Pose2D(1.0, 2.5, 0.0)
    waypoints = [
        PatrolWaypoint((12.0, 2.5), "xy", "blocked"),
        PatrolWaypoint((3.0, 2.5), "xy", "reachable"),
    ]

    result = run_patrol(grid, start, waypoints, params=_patrol_params())

    assert result.stops[0]["planned"] is False
    assert result.stops[0]["reached"] is False
    assert result.stops[0]["steps"] == 0
    assert result.stops[1]["planned"] is True
    assert result.stops[1]["reached"] is True
    assert len(result.segments) == 1


def test_capture_fn_runs_once_per_driven_stop_and_stores_return() -> None:
    grid = _corridor_grid()
    start = Pose2D(1.0, 2.5, 0.0)
    waypoints = [
        PatrolWaypoint((4.0, 2.5), "xy", "first"),
        PatrolWaypoint((7.0, 2.5), "xy", "second"),
    ]
    calls: list[tuple[int, str, float]] = []

    def capture(index: int, waypoint: PatrolWaypoint, pose: Pose2D) -> str:
        calls.append((index, waypoint.label, pose.x))
        return f"capture-{index}.png"

    result = run_patrol(grid, start, waypoints, params=_patrol_params(), capture_fn=capture)

    assert [(index, label) for index, label, _x in calls] == [(0, "first"), (1, "second")]
    assert calls[0][2] > start.x
    assert [stop["capture"] for stop in result.stops] == ["capture-0.png", "capture-1.png"]


def test_return_to_start_appends_home_stop() -> None:
    grid = _corridor_grid()
    start = Pose2D(1.0, 2.5, 0.0)
    waypoints = [PatrolWaypoint((4.0, 2.5), "xy", "away")]

    result = run_patrol(grid, start, waypoints, params=_patrol_params(return_to_start=True))

    assert [stop["label"] for stop in result.stops] == ["away", "home"]
    assert result.stops[-1]["goal_xy"] == [1.0, 2.5]


def test_trace_image_smoke_width() -> None:
    pytest.importorskip("PIL")
    grid = _corridor_grid()
    start = Pose2D(1.0, 2.5, 0.0)
    waypoints = [PatrolWaypoint((4.0, 2.5), "xy", "stop 1")]
    result = run_patrol(grid, start, waypoints, params=_patrol_params())

    image = patrol_trace_image(grid, result, waypoints, width=320)

    assert image.width == 320


def test_waypoints_from_changes_limit_keeps_largest() -> None:
    grid = _corridor_grid()
    report = {
        "appeared": [
            {"centroid": [1.0, 1.0, 0.0], "voxels": 5},
            {"centroid": [2.0, 2.0, 0.0], "voxels": 40},
        ],
        "disappeared": [
            {"centroid": [3.0, 3.0, 0.0], "voxels": 20},
        ],
    }
    waypoints = waypoints_from_changes(report, grid, kinds=("appeared", "disappeared"), limit=2)

    assert [waypoint.label for waypoint in waypoints] == ["appeared #2", "disappeared #1"]
