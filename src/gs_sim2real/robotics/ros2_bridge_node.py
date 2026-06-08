"""ROS2 consumer node scaffold for Dynamic Map Viewer robotics relay topics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from .topic_map import (
    build_camera_command_payload,
    build_ros_topic_map,
    build_teleop_command_payload,
    summarize_robot_state_payload,
)
from .zones import (
    build_zone_hit_payload,
    build_zone_summary_payload,
    find_zone_hits,
    load_semantic_zone_map,
    rasterize_costmap,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI for the ROS2 bridge node."""
    parser = argparse.ArgumentParser(
        prog="gs-sim2real robotics-node",
        description="ROS2 node scaffold for Dynamic Map Viewer robotics relay topics",
    )
    parser.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    parser.add_argument("--node-name", default="dreamwalker_bridge_node", help="ROS2 node name")
    parser.add_argument("--frame-id", default="dreamwalker_map", help="Expected map frame id")
    parser.add_argument("--log-period", type=float, default=2.0, help="Summary log period in seconds")
    parser.add_argument("--zones-file", default=None, help="Optional semantic zone JSON file")
    parser.add_argument(
        "--costmap-period", type=float, default=10.0, help="Costmap republish period in seconds (0 to disable)"
    )
    parser.add_argument("--request-state-on-start", action="store_true", help="Publish request_state once on startup")
    parser.add_argument(
        "--enable-image-relay",
        action="store_true",
        help="Subscribe to camera relay topics and log received frames",
    )
    parser.add_argument(
        "--demo-teleop",
        choices=["forward", "backward", "turn-left", "turn-right"],
        default=None,
        help="Publish one teleop command on startup",
    )
    parser.add_argument(
        "--demo-camera",
        choices=["front", "chase", "top"],
        default=None,
        help="Publish one camera command on startup",
    )
    parser.add_argument(
        "--demo-waypoint",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Publish one waypoint command on startup",
    )
    parser.add_argument(
        "--demo-pose2d",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA_RAD"),
        default=None,
        help="Publish one Pose2D command on startup",
    )
    return parser


def _import_ros2() -> dict[str, Any]:
    try:
        import rclpy
        from geometry_msgs.msg import Point, Pose2D, PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Path
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image
        from std_msgs.msg import Empty, String
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real ROS2 env
        raise RuntimeError(
            "ROS2 runtime not available. Source your ROS2 environment so `rclpy` and message packages are importable."
        ) from exc

    return {
        "rclpy": rclpy,
        "Node": Node,
        "String": String,
        "Pose2D": Pose2D,
        "PoseStamped": PoseStamped,
        "Point": Point,
        "OccupancyGrid": OccupancyGrid,
        "Path": Path,
        "Twist": Twist,
        "Empty": Empty,
        "CameraInfo": CameraInfo,
        "CompressedImage": CompressedImage,
        "Image": Image,
    }


def quaternion_to_yaw_radians(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to planar yaw."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def format_ros_timestamp(message: Any) -> str:
    """Format a ROS message header timestamp for logs."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return "unknown"
    return f"{int(sec)}.{int(nanosec):09d}"


def _build_node_class(ros2: dict[str, Any]) -> type:
    Node = ros2["Node"]
    String = ros2["String"]
    Pose2D = ros2["Pose2D"]
    PoseStamped = ros2["PoseStamped"]
    Point = ros2["Point"]
    OccupancyGrid = ros2["OccupancyGrid"]
    Path = ros2["Path"]
    Twist = ros2["Twist"]
    Empty = ros2["Empty"]
    CameraInfo = ros2["CameraInfo"]
    CompressedImage = ros2["CompressedImage"]
    Image = ros2["Image"]

    class DreamwalkerBridgeNode(Node):
        """Minimal ROS2 node that consumes and produces Dynamic Map Viewer relay topics."""

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args.node_name)
            self.args = args
            self.topic_map = build_ros_topic_map(args.namespace)
            self.latest_state_json: str | None = None
            self.latest_pose_summary = "pose: none"
            self.latest_goal_summary = "goal: none"
            self.latest_route_summary = "route: none"
            self.latest_zone_summary = "zones: none"
            self.latest_camera_resolution: tuple[int, int] | None = None
            self.zone_map = load_semantic_zone_map(args.zones_file) if args.zones_file else None

            self.cmd_json_pub = self.create_publisher(String, self.topic_map.cmd_json, 10)
            self.cmd_pose2d_pub = self.create_publisher(Pose2D, self.topic_map.cmd_pose2d, 10)
            self.cmd_waypoint_pub = self.create_publisher(Point, self.topic_map.cmd_waypoint, 10)
            self.cmd_vel_pub = self.create_publisher(Twist, self.topic_map.cmd_vel, 10)
            self.request_state_pub = self.create_publisher(Empty, self.topic_map.request_state, 10)
            self.semantic_zone_summary_pub = self.create_publisher(
                String, self.topic_map.semantic_zone_summary_json, 10
            )
            self.current_zone_pub = self.create_publisher(String, self.topic_map.current_zone_json, 10)
            self.semantic_costmap_pub = self.create_publisher(OccupancyGrid, self.topic_map.semantic_costmap, 10)

            self.create_subscription(String, self.topic_map.robot_state_json, self._on_robot_state_json, 10)
            self.create_subscription(Pose2D, self.topic_map.robot_pose2d, self._on_robot_pose2d, 10)
            self.create_subscription(PoseStamped, self.topic_map.robot_pose_stamped, self._on_robot_pose_stamped, 10)
            self.create_subscription(Point, self.topic_map.robot_waypoint, self._on_robot_waypoint, 10)
            self.create_subscription(
                PoseStamped,
                self.topic_map.robot_goal_pose_stamped,
                self._on_robot_goal_pose_stamped,
                10,
            )
            self.create_subscription(Path, self.topic_map.robot_route_path, self._on_robot_route_path, 10)
            if args.enable_image_relay:
                self.create_subscription(
                    CompressedImage,
                    self.topic_map.camera_compressed,
                    self._on_camera_compressed,
                    10,
                )
                self.create_subscription(
                    CameraInfo,
                    self.topic_map.camera_info,
                    self._on_camera_info,
                    10,
                )
                self.create_subscription(
                    Image,
                    self.topic_map.depth_image,
                    self._on_depth_image,
                    10,
                )

            self.summary_timer = self.create_timer(max(args.log_period, 0.25), self._log_summary)
            self.costmap_timer = None
            if self.zone_map and args.costmap_period > 0:
                self.costmap_timer = self.create_timer(max(args.costmap_period, 0.5), self._publish_semantic_costmap)
            self._emit_startup_commands()
            self._publish_zone_summary()
            self._publish_semantic_costmap()
            self.get_logger().info(
                f"Dynamic Map Viewer bridge node ready namespace={self.topic_map.namespace} "
                f"frame={self.args.frame_id} image_relay={self.args.enable_image_relay}"
            )

        def _emit_startup_commands(self) -> None:
            if self.args.request_state_on_start:
                self.request_state_pub.publish(Empty())

            if self.args.demo_teleop:
                message = String()
                message.data = build_teleop_command_payload(self.args.demo_teleop)
                self.cmd_json_pub.publish(message)

            if self.args.demo_camera:
                message = String()
                message.data = build_camera_command_payload(self.args.demo_camera)
                self.cmd_json_pub.publish(message)

            if self.args.demo_waypoint:
                x, y, z = self.args.demo_waypoint
                waypoint = Point()
                waypoint.x = float(x)
                waypoint.y = float(y)
                waypoint.z = float(z)
                self.cmd_waypoint_pub.publish(waypoint)

            if self.args.demo_pose2d:
                x, y, theta = self.args.demo_pose2d
                pose = Pose2D()
                pose.x = float(x)
                pose.y = float(y)
                pose.theta = float(theta)
                self.cmd_pose2d_pub.publish(pose)

        def _on_robot_state_json(self, message: Any) -> None:
            self.latest_state_json = message.data

        def _on_robot_pose2d(self, message: Any) -> None:
            self.latest_pose_summary = (
                f"pose2d=({message.x:.2f}, {message.y:.2f}) theta={math.degrees(message.theta):.1f}deg"
            )
            self._update_current_zones(message.x, message.y)

        def _on_robot_pose_stamped(self, message: Any) -> None:
            yaw = math.degrees(
                quaternion_to_yaw_radians(
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                )
            )
            self.latest_pose_summary = (
                f"pose=({message.pose.position.x:.2f}, {message.pose.position.y:.2f}) "
                f"yaw={yaw:.1f}deg frame={message.header.frame_id}"
            )
            self._update_current_zones(message.pose.position.x, message.pose.position.y)

        def _on_robot_waypoint(self, message: Any) -> None:
            self.latest_goal_summary = f"waypoint=({message.x:.2f}, {message.z:.2f})"

        def _on_robot_goal_pose_stamped(self, message: Any) -> None:
            self.latest_goal_summary = (
                f"goal=({message.pose.position.x:.2f}, {message.pose.position.y:.2f}) frame={message.header.frame_id}"
            )

        def _on_robot_route_path(self, message: Any) -> None:
            self.latest_route_summary = f"path_nodes={len(message.poses)} frame={message.header.frame_id}"

        def _on_camera_compressed(self, message: Any) -> None:
            resolution = "unknown"
            if self.latest_camera_resolution:
                width, height = self.latest_camera_resolution
                resolution = f"{width}x{height}"

            self.get_logger().info(
                f"camera/compressed bytes={len(message.data)} format={message.format or 'unknown'} "
                f"stamp={format_ros_timestamp(message)} resolution={resolution}"
            )

        def _on_camera_info(self, message: Any) -> None:
            self.latest_camera_resolution = (int(message.width), int(message.height))
            frame_id = getattr(message.header, "frame_id", "") or "unknown"
            fx = float(message.k[0]) if len(message.k) > 0 else 0.0
            fy = float(message.k[4]) if len(message.k) > 4 else 0.0
            self.get_logger().info(
                f"camera/camera_info stamp={format_ros_timestamp(message)} "
                f"resolution={message.width}x{message.height} frame={frame_id} "
                f"fx={fx:.2f} fy={fy:.2f}"
            )

        def _on_depth_image(self, message: Any) -> None:
            frame_id = getattr(message.header, "frame_id", "") or "unknown"
            self.get_logger().info(
                f"depth/image stamp={format_ros_timestamp(message)} "
                f"resolution={message.width}x{message.height} frame={frame_id} "
                f"encoding={message.encoding or 'unknown'} step={message.step} bytes={len(message.data)}"
            )

        def _log_summary(self) -> None:
            if self.latest_state_json:
                try:
                    summary = summarize_robot_state_payload(self.latest_state_json)
                except json.JSONDecodeError:
                    summary = "robot-state: invalid json"
            else:
                summary = "robot-state: waiting"

            self.get_logger().info(
                f"{summary} | {self.latest_pose_summary} | {self.latest_goal_summary} | "
                f"{self.latest_route_summary} | {self.latest_zone_summary}"
            )

        def _publish_zone_summary(self) -> None:
            if not self.zone_map:
                return

            message = String()
            message.data = json.dumps(build_zone_summary_payload(self.zone_map))
            self.semantic_zone_summary_pub.publish(message)

        def _publish_semantic_costmap(self) -> None:
            if not self.zone_map:
                return

            rasterized = rasterize_costmap(self.zone_map)
            message = OccupancyGrid()
            message.header.frame_id = rasterized.frame_id
            message.info.resolution = float(rasterized.resolution)
            message.info.width = rasterized.width
            message.info.height = rasterized.height
            message.info.origin.position.x = float(rasterized.origin_x)
            message.info.origin.position.y = float(rasterized.origin_z)
            message.info.origin.position.z = 0.0
            message.info.origin.orientation.w = 1.0
            message.data = list(rasterized.data)
            self.semantic_costmap_pub.publish(message)

        def _update_current_zones(self, x: float, z: float) -> None:
            if not self.zone_map:
                return

            zone_hits = find_zone_hits(self.zone_map, float(x), float(z))
            zone_labels = ", ".join(zone.label for zone in zone_hits) if zone_hits else "none"
            self.latest_zone_summary = f"zones={zone_labels}"

            message = String()
            message.data = json.dumps(build_zone_hit_payload(zone_hits, float(x), float(z)))
            self.current_zone_pub.publish(message)

    return DreamwalkerBridgeNode


def run_cli(args: argparse.Namespace) -> None:
    """Run the ROS2 bridge node from parsed CLI args."""
    ros2 = _import_ros2()
    rclpy = ros2["rclpy"]
    node_class = _build_node_class(ros2)

    rclpy.init(args=None)
    node = node_class(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the ROS2 bridge node."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
