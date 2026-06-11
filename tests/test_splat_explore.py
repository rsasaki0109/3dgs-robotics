"""Tests for frontier-based autonomous exploration inside a 3DGS map."""

from __future__ import annotations

import numpy as np
import pytest

from gs_sim2real.robotics.occupancy_grid import OccupancyGridMap
from gs_sim2real.robotics.splat_explore import (
    ExploreParams,
    Frontier,
    frontier_clusters,
    explore_trace_image,
    reachable_free_mask,
    run_exploration,
    select_frontier,
    visible_cells,
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


class TestVisibility:
    def test_rays_stop_at_occupied_walls(self):
        grid = _corridor_grid()
        params = ExploreParams(sensor_range=4.0, ray_count=360)
        seen = visible_cells(grid, np.array([2.0, 2.5]), params=params)

        assert seen[3, 8]
        assert not seen[2, 8]

    def test_rays_stop_at_unknown(self):
        data = np.zeros((7, 12), dtype=np.int8)
        data[3, 6] = -1
        grid = OccupancyGridMap(
            data=data,
            resolution=1.0,
            origin=(0.0, 0.0),
            up=np.array([0.0, 0.0, 1.0]),
            basis=np.eye(3),
            ground_height=0.0,
            camera_height=1.0,
        )
        seen = visible_cells(grid, np.array([2.5, 3.5]), params=ExploreParams(sensor_range=8.0, ray_count=360))

        assert seen[3, 6]
        assert not seen[3, 8]

    def test_corridor_center_marks_corridor_within_range(self):
        grid = _corridor_grid()
        seen = visible_cells(grid, np.array([3.0, 2.5]), params=ExploreParams(sensor_range=2.0, ray_count=360))

        assert seen[10, 12]
        assert seen[10, 18]


class TestReachability:
    def test_walled_off_free_pocket_is_not_reachable(self):
        grid = _corridor_grid()
        grid.data[4:-4, 30] = 100
        reachable = reachable_free_mask(grid, (10, 5), robot_radius=0.0)

        assert reachable[10, 5]
        assert not reachable[10, 45]


class TestFrontiers:
    def test_partially_observed_corridor_yields_boundary_frontier(self):
        grid = _corridor_grid()
        reachable = grid.data == 0
        observed = np.zeros_like(reachable)
        observed[4:17, :10] = True

        clusters = frontier_clusters(grid, observed, reachable, min_cells=1)

        assert clusters
        assert any(any(col == 10 for _row, col in cluster.cells) for cluster in clusters)

    def test_min_cells_filter_drops_small_clusters(self):
        grid = _corridor_grid()
        reachable = grid.data == 0
        observed = np.zeros_like(reachable)
        observed[10, 10] = True

        assert frontier_clusters(grid, observed, reachable, min_cells=10) == []

    def test_select_frontier_prefers_large_near_cluster(self):
        near = Frontier(cells=[(0, i) for i in range(20)], centroid_xy=np.array([1.0, 0.0]), size=20)
        far = Frontier(cells=[(0, i) for i in range(4)], centroid_xy=np.array([20.0, 0.0]), size=4)

        assert select_frontier([far, near], np.array([0.0, 0.0])) is near


class TestRunExploration:
    def test_corridor_reaches_coverage_target(self):
        grid = _corridor_grid()
        params = ExploreParams(
            sensor_range=2.5,
            min_frontier_cells=2,
            coverage_target=0.9,
            max_goals=12,
            nav=NavParams(robot_radius=0.0, goal_tolerance=0.5, localize_every=0, max_steps=800),
        )

        result = run_exploration(grid, Pose2D(1.0, 2.5, 0.0), params=params)

        assert result.coverage_fraction >= params.coverage_target
        assert result.goals_chosen > 0
        assert result.stop_reason == "coverage-target"
        assert [coverage for _step, coverage in result.coverage_history] == sorted(
            coverage for _step, coverage in result.coverage_history
        )

    def test_start_scan_can_satisfy_tiny_target(self):
        grid = _corridor_grid()
        params = ExploreParams(
            sensor_range=1.0,
            coverage_target=0.05,
            max_goals=5,
            nav=NavParams(robot_radius=0.0, localize_every=0),
        )

        result = run_exploration(grid, Pose2D(1.0, 2.5, 0.0), params=params)

        assert result.goals_chosen == 0
        assert result.stop_reason == "coverage-target"


class TestVisualization:
    def test_trace_image_smoke(self):
        pytest.importorskip("PIL")
        grid = _corridor_grid()
        params = ExploreParams(
            sensor_range=1.0,
            coverage_target=0.05,
            nav=NavParams(robot_radius=0.0, localize_every=0),
        )
        result = run_exploration(grid, Pose2D(1.0, 2.5, 0.0), params=params)

        image = explore_trace_image(grid, result, width=320)

        assert image.width == 320
