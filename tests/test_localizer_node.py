"""Tests for the rclpy-free pieces of the 3DGS localizer ROS 2 node."""

from __future__ import annotations

import numpy as np
import pytest

from gs_sim2real.robotics.localize import viewmat_to_colmap
from gs_sim2real.robotics.localizer_node import (
    LatestFrame,
    build_parser,
    colmap_to_ros_pose,
    parse_pyramid_scales,
)


def _quat_xyzw_to_rotation(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestColmapToRosPose:
    def test_identity_pose(self):
        position, orientation = colmap_to_ros_pose((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(position, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(orientation, [0.0, 0.0, 0.0, 1.0])

    def test_position_is_camera_center(self):
        # identity rotation, camera at (1, 2, 3): tvec = -center
        position, _ = colmap_to_ros_pose((1.0, 0.0, 0.0, 0.0), (-1.0, -2.0, -3.0))
        np.testing.assert_allclose(position, [1.0, 2.0, 3.0])

    def test_consistent_with_viewmat_to_colmap(self):
        rng = np.random.default_rng(3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = 0.8
        k_skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * k_skew + (1 - np.cos(angle)) * (k_skew @ k_skew)
        viewmat = np.eye(4)
        viewmat[:3, :3] = rotation
        viewmat[:3, 3] = [0.4, -0.2, 1.5]

        qvec, tvec, center = viewmat_to_colmap(viewmat)
        position, orientation = colmap_to_ros_pose(qvec, tvec)
        np.testing.assert_allclose(position, center, atol=1e-9)
        # orientation must be world-from-camera = R_cw.T
        np.testing.assert_allclose(_quat_xyzw_to_rotation(orientation), rotation.T, atol=1e-6)


class TestParsePyramidScales:
    def test_parses_comma_separated(self):
        assert parse_pyramid_scales("0.25,0.5,1.0") == (0.25, 0.5, 1.0)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            parse_pyramid_scales("0.5,2.0")
        with pytest.raises(ValueError):
            parse_pyramid_scales("")


class TestLatestFrame:
    def test_take_returns_newest_and_counts_drops(self):
        latest = LatestFrame()
        latest.put(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)
        latest.put(np.ones((2, 2, 3), dtype=np.uint8), 2.0)
        frame = latest.take(timeout=0.01)
        assert frame is not None
        image, timestamp = frame
        assert timestamp == 2.0
        assert image.max() == 1
        assert latest.dropped == 1

    def test_take_times_out_when_empty(self):
        assert LatestFrame().take(timeout=0.01) is None

    def test_take_clears_slot(self):
        latest = LatestFrame()
        latest.put(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)
        assert latest.take(timeout=0.01) is not None
        assert latest.take(timeout=0.01) is None


def test_parser_defaults_are_sane():
    args = build_parser().parse_args(["--map", "outputs/session"])
    assert args.session == "outputs/session"
    assert args.image_topic.endswith("/compressed")
    assert args.pose_topic == "/gs_localizer/pose"
    assert args.map_frame == "map"
    assert not args.follow_latest
    assert args.max_seed_distance > 0


def test_parser_requires_map():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
