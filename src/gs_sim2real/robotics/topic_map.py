"""Topic map and payload helpers for Dynamic Map Viewer robotics integration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


def normalize_namespace(namespace: str) -> str:
    """Normalize ROS topic namespace to a single leading slash and no trailing slash."""
    cleaned = (namespace or "dreamwalker").strip().strip("/")
    return f"/{cleaned}" if cleaned else "/dreamwalker"


@dataclass(frozen=True)
class DreamwalkerRosTopicMap:
    """Fixed ROS topic map for Dynamic Map Viewer relay and ROS2 nodes."""

    namespace: str
    camera_compressed: str
    camera_info: str
    depth_image: str
    robot_state_json: str
    robot_pose2d: str
    robot_pose_stamped: str
    robot_waypoint: str
    robot_waypoint_json: str
    robot_goal_pose_stamped: str
    robot_route_json: str
    robot_route_path: str
    semantic_zone_summary_json: str
    current_zone_json: str
    semantic_costmap: str
    cmd_json: str
    cmd_pose2d: str
    cmd_waypoint: str
    cmd_vel: str
    request_state: str


def build_ros_topic_map(namespace: str = "/dreamwalker") -> DreamwalkerRosTopicMap:
    """Build the fixed ROS topic map under a namespace."""
    root = normalize_namespace(namespace)
    return DreamwalkerRosTopicMap(
        namespace=root,
        camera_compressed=f"{root}/camera/compressed",
        camera_info=f"{root}/camera/camera_info",
        depth_image=f"{root}/depth/image",
        robot_state_json=f"{root}/robot_state_json",
        robot_pose2d=f"{root}/robot_pose2d",
        robot_pose_stamped=f"{root}/robot_pose_stamped",
        robot_waypoint=f"{root}/robot_waypoint",
        robot_waypoint_json=f"{root}/robot_waypoint_json",
        robot_goal_pose_stamped=f"{root}/robot_goal_pose_stamped",
        robot_route_json=f"{root}/robot_route_json",
        robot_route_path=f"{root}/robot_route_path",
        semantic_zone_summary_json=f"{root}/semantic_zone_summary_json",
        current_zone_json=f"{root}/current_zone_json",
        semantic_costmap=f"{root}/semantic_costmap",
        cmd_json=f"{root}/cmd_json",
        cmd_pose2d=f"{root}/cmd_pose2d",
        cmd_waypoint=f"{root}/cmd_waypoint",
        cmd_vel=f"{root}/cmd_vel",
        request_state=f"{root}/request_state",
    )


def build_command_payload(command_type: str, **fields: Any) -> str:
    """Build a JSON command payload for the relay command topics."""
    payload = {
        "type": command_type,
        **fields,
    }
    return json.dumps(payload, ensure_ascii=True)


def build_camera_command_payload(camera_id: str) -> str:
    """Build a set-camera command payload."""
    return build_command_payload("set-camera", cameraId=camera_id)


def build_teleop_command_payload(action: str) -> str:
    """Build a teleop command payload."""
    return build_command_payload("teleop", action=action)


def summarize_robot_state_payload(payload: str | dict[str, Any]) -> str:
    """Create a concise log summary from robot-state JSON."""
    data: dict[str, Any]
    if isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload

    pose = data.get("pose") or {}
    position = pose.get("position") or [0, 0, 0]
    yaw = pose.get("yawDegrees", 0)
    fragment_id = data.get("fragmentId", "unknown")
    route_count = data.get("routeNodeCount", 0)
    camera_id = data.get("robotCameraId", "unknown")
    return (
        f"fragment={fragment_id} "
        f"pose=({float(position[0]):.2f}, {float(position[2]):.2f}) "
        f"yaw={float(yaw):.1f}deg "
        f"route_nodes={int(route_count)} "
        f"camera={camera_id}"
    )
