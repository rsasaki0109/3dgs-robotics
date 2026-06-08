"""Tests for Dynamic Map Viewer robotics topic helpers."""

from __future__ import annotations

import json
import math

import pytest

from gs_sim2real.robotics.ros2_bridge_node import build_parser, quaternion_to_yaw_radians
from gs_sim2real.robotics.topic_map import (
    build_camera_command_payload,
    build_ros_topic_map,
    build_teleop_command_payload,
    normalize_namespace,
    summarize_robot_state_payload,
)


def test_normalize_namespace_adds_single_leading_slash() -> None:
    assert normalize_namespace("dreamwalker") == "/dreamwalker"
    assert normalize_namespace("/dreamwalker/robot") == "/dreamwalker/robot"
    assert normalize_namespace("///dreamwalker///") == "/dreamwalker"


def test_build_ros_topic_map_uses_namespace_prefix() -> None:
    topic_map = build_ros_topic_map("sandbox/world")

    assert topic_map.namespace == "/sandbox/world"
    assert topic_map.camera_compressed == "/sandbox/world/camera/compressed"
    assert topic_map.camera_info == "/sandbox/world/camera/camera_info"
    assert topic_map.depth_image == "/sandbox/world/depth/image"
    assert topic_map.robot_state_json == "/sandbox/world/robot_state_json"
    assert topic_map.robot_pose_stamped == "/sandbox/world/robot_pose_stamped"
    assert topic_map.robot_route_path == "/sandbox/world/robot_route_path"
    assert topic_map.semantic_costmap == "/sandbox/world/semantic_costmap"
    assert topic_map.current_zone_json == "/sandbox/world/current_zone_json"
    assert topic_map.cmd_pose2d == "/sandbox/world/cmd_pose2d"


def test_command_payload_helpers_produce_expected_json() -> None:
    camera_payload = json.loads(build_camera_command_payload("top"))
    teleop_payload = json.loads(build_teleop_command_payload("forward"))

    assert camera_payload == {"type": "set-camera", "cameraId": "top"}
    assert teleop_payload == {"type": "teleop", "action": "forward"}


def test_summarize_robot_state_payload_formats_pose() -> None:
    summary = summarize_robot_state_payload(
        {
            "fragmentId": "residency",
            "pose": {
                "position": [1.25, 0.0, 7.1],
                "yawDegrees": 90,
            },
            "routeNodeCount": 3,
            "robotCameraId": "front",
        }
    )

    assert "fragment=residency" in summary
    assert "pose=(1.25, 7.10)" in summary
    assert "yaw=90.0deg" in summary
    assert "route_nodes=3" in summary
    assert "camera=front" in summary


def test_quaternion_to_yaw_radians_returns_planar_heading() -> None:
    yaw = quaternion_to_yaw_radians(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))

    assert yaw == pytest.approx(math.pi / 2)


def test_build_parser_supports_enable_image_relay() -> None:
    args = build_parser().parse_args(["--enable-image-relay"])

    assert args.enable_image_relay is True
