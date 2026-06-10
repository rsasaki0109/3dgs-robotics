"""Tests for autonomous navigation inside a 3DGS map (CPU only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gs_sim2real.robotics.gsplat_render_server import quaternion_to_rotation_matrix
from gs_sim2real.robotics.occupancy_grid import OccupancyGridMap
from gs_sim2real.robotics.splat_nav import (
    NavParams,
    Pose2D,
    PurePursuit,
    astar,
    inflate_obstacles,
    nearest_free_cell,
    plan_path,
    pose2d_to_camera_pose,
    run_navigation,
    shortcut_path,
    step_unicycle,
    world_to_pose2d,
)


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


class TestGridPrimitives:
    def test_inflate_blocks_unknown_and_dilates(self):
        data = np.zeros((7, 7), dtype=np.int8)
        data[3, 3] = 100
        data[0, 0] = -1
        blocked = inflate_obstacles(data, 1)
        assert blocked[0, 0]
        assert blocked[3, 3] and blocked[2, 3] and blocked[3, 2]
        assert not blocked[1, 1]

    def test_nearest_free_cell(self):
        blocked = np.zeros((5, 5), dtype=bool)
        blocked[2, 2] = True
        assert nearest_free_cell(blocked, (2, 2)) in {(1, 2), (3, 2), (2, 1), (2, 3)}
        assert nearest_free_cell(np.ones((3, 3), dtype=bool), (1, 1), max_radius=2) is None

    def test_astar_goes_through_the_gap(self):
        blocked = np.zeros((9, 9), dtype=bool)
        blocked[:, 4] = True
        blocked[4, 4] = False  # gap
        path = astar(blocked, (0, 0), (8, 8))
        assert path is not None
        assert (4, 4) in path
        assert path[0] == (0, 0) and path[-1] == (8, 8)

    def test_astar_unreachable_returns_none(self):
        blocked = np.zeros((5, 5), dtype=bool)
        blocked[:, 2] = True
        assert astar(blocked, (0, 0), (0, 4)) is None

    def test_shortcut_collapses_straight_runs(self):
        blocked = np.zeros((5, 12), dtype=bool)
        path = [(2, col) for col in range(12)]
        assert shortcut_path(blocked, path) == [(2, 0), (2, 11)]


class TestPlanPath:
    def test_plan_along_corridor(self):
        grid = _corridor_grid()
        path = plan_path(grid, np.array([1.0, 2.5]), np.array([14.0, 2.5]), params=NavParams(robot_radius=0.25))
        assert len(path) >= 2
        assert path[0] == pytest.approx([1.0, 2.5], abs=grid.resolution)
        assert path[-1] == pytest.approx([14.0, 2.5], abs=grid.resolution)
        # the corridor's free band is rows 4..16 -> y in (1.0, 4.25)
        assert np.all(path[:, 1] > 1.0) and np.all(path[:, 1] < 4.25)

    def test_unreachable_goal_raises(self):
        grid = _corridor_grid()
        grid.data[:, 29:32] = 100  # full wall across the corridor
        with pytest.raises(ValueError, match="no path"):
            plan_path(grid, np.array([1.0, 2.5]), np.array([14.0, 2.5]), params=NavParams(robot_radius=0.25))


class TestControl:
    def test_step_unicycle_straight_and_turn(self):
        pose = step_unicycle(Pose2D(0.0, 0.0, 0.0), velocity=1.0, yaw_rate=0.0, dt=0.5)
        assert (pose.x, pose.y) == pytest.approx((0.5, 0.0))
        turned = step_unicycle(Pose2D(0.0, 0.0, 0.0), velocity=0.0, yaw_rate=math.pi, dt=0.5)
        assert turned.yaw == pytest.approx(math.pi / 2)

    def test_pure_pursuit_converges_to_straight_path(self):
        path = np.array([[0.0, 0.0], [20.0, 0.0]])
        follower = PurePursuit(path, lookahead=1.0, speed=1.0, max_yaw_rate=2.0)
        pose = Pose2D(0.0, 1.5, 0.0)  # offset from the path
        for _ in range(400):
            velocity, yaw_rate = follower.command(pose)
            pose = step_unicycle(pose, velocity, yaw_rate, 0.1)
            if pose.x > 18.0:
                break
        assert abs(pose.y) < 0.2


class TestPoseConversions:
    def test_roundtrip_through_camera_pose(self):
        grid = _corridor_grid()
        grid.basis = np.array(
            [
                [0.0, 0.0, 1.0],  # e1
                [-1.0, 0.0, 0.0],  # e2
                [0.0, -1.0, 0.0],  # up (KITTI-like optical world)
            ]
        )
        grid.up = grid.basis[2]
        pose = Pose2D(3.0, -1.2, 0.7)
        camera = pose2d_to_camera_pose(pose, grid, height_above_ground=1.0)
        rotation_wc = quaternion_to_rotation_matrix(camera.orientation)
        back = world_to_pose2d(np.asarray(camera.position), rotation_wc, grid)
        assert (back.x, back.y, back.yaw) == pytest.approx((pose.x, pose.y, pose.yaw), abs=1e-5)
        # height above ground along up
        assert np.asarray(camera.position) @ grid.up == pytest.approx(grid.ground_height + grid.camera_height)

    def test_identity_world_gives_identity_quaternion(self):
        grid = _corridor_grid()
        grid.basis = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        grid.up = grid.basis[2]
        camera = pose2d_to_camera_pose(Pose2D(0.0, 0.0, 0.0), grid)
        assert np.asarray(camera.orientation) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


class TestRunNavigation:
    def test_reaches_goal_with_dead_reckoning(self):
        grid = _corridor_grid()
        result = run_navigation(
            grid,
            Pose2D(1.0, 2.5, 0.0),
            np.array([14.0, 2.5]),
            params=NavParams(robot_radius=0.25, goal_tolerance=0.5, localize_every=0),
        )
        assert result.reached
        assert result.steps < 3000
        assert result.cross_track_errors().max() < 1.0

    def test_localization_fix_corrects_a_bad_initial_estimate(self):
        grid = _corridor_grid()
        params = NavParams(robot_radius=0.25, goal_tolerance=0.5, localize_every=10, max_steps=4000)
        start = Pose2D(1.0, 2.5, 0.0)
        bad_estimate = Pose2D(1.0, 1.4, 0.4)

        blind = run_navigation(
            grid,
            start,
            np.array([14.0, 2.5]),
            params=NavParams(robot_radius=0.25, localize_every=0),
            est_start=bad_estimate,
        )
        corrected = run_navigation(
            grid,
            start,
            np.array([14.0, 2.5]),
            observe_fn=lambda true_pose, step: Pose2D(true_pose.x, true_pose.y, true_pose.yaw),
            params=params,
            est_start=bad_estimate,
        )
        assert corrected.reached
        assert corrected.localization_count > 0
        assert corrected.cross_track_errors().max() < blind.cross_track_errors().max()

    def test_observer_none_fix_is_skipped(self):
        grid = _corridor_grid()
        result = run_navigation(
            grid,
            Pose2D(1.0, 2.5, 0.0),
            np.array([14.0, 2.5]),
            observe_fn=lambda true_pose, step: None,
            params=NavParams(robot_radius=0.25, localize_every=10),
        )
        assert result.reached
        assert result.localization_count == 0

    def test_localization_beats_drifting_odometry(self):
        """With wheel slip, dead reckoning veers off; fixes keep it on track."""
        grid = _corridor_grid()
        start = Pose2D(1.0, 2.5, 0.0)
        goal = np.array([14.0, 2.5])
        noisy = dict(robot_radius=0.25, goal_tolerance=0.5, odom_noise=0.2, seed=3, max_steps=4000)

        blind = run_navigation(grid, start, goal, params=NavParams(localize_every=0, **noisy))
        fixed = run_navigation(
            grid,
            start,
            goal,
            observe_fn=lambda true_pose, step: Pose2D(true_pose.x, true_pose.y, true_pose.yaw),
            params=NavParams(localize_every=10, fix_blend=0.7, **noisy),
        )
        assert fixed.reached
        assert fixed.cross_track_errors().max() < blind.cross_track_errors().max()

    def test_innovation_gate_rejects_aliased_fixes(self):
        grid = _corridor_grid()
        wild = Pose2D(50.0, 50.0, 0.0)  # visual-aliasing teleport
        result = run_navigation(
            grid,
            Pose2D(1.0, 2.5, 0.0),
            np.array([14.0, 2.5]),
            observe_fn=lambda true_pose, step: wild,
            params=NavParams(robot_radius=0.25, localize_every=10, max_innovation=5.0),
        )
        assert result.reached
        assert result.localization_count == 0  # every fix gated out
