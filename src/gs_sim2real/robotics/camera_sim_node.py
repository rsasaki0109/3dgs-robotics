"""ROS 2 GS camera simulator node: pose in, photorealistic camera topic out.

Renders a trained 3DGS map (a live-mapping session round or any standard
gsplat PLY) from camera poses and publishes ``sensor_msgs`` camera topics —
a lightweight, ROS-2-only counterpart to the Isaac Sim export
(:mod:`gs_sim2real.robotics.isaac_export`). Poses come either from a
``geometry_msgs/PoseStamped`` topic or from ``--replay``, which replays the
session's mapped keyframe trajectory so the node is self-contained.

Together with the localizer node this closes the loop inside one repo::

    # terminal 1: virtual camera flying the mapped trajectory
    3dgs-robotics-camera-sim --map outputs/live_mapping --replay --loop

    # terminal 2: localize the simulated stream against the same map
    3dgs-robotics-localizer --map outputs/live_mapping \
        --image-topic /gs_camera_sim/image_raw/compressed

Poses follow the localizer's convention: map frame is the round's COLMAP
world (arbitrary gauge), camera frame is optical (x right, y down, z
forward). In replay mode the ground-truth pose is also published so the
localizer's estimate can be compared against it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gs_sim2real.robotics.gsplat_render_server import (
    CameraPose,
    HeadlessSplatRenderer,
    compute_camera_intrinsics,
    encode_rgb_to_jpeg,
)
from gs_sim2real.robotics.localizer_node import colmap_to_ros_pose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3dgs-robotics-camera-sim",
        description="ROS 2 GS camera simulator (PoseStamped in -> rendered camera topics out)",
    )
    parser.add_argument("--node-name", default="gs_camera_sim", help="ROS 2 node name")
    parser.add_argument("--map", dest="session", default=None, help="Live-mapping session directory (workdir)")
    parser.add_argument("--round", type=int, default=None, help="Pin one rebuild round (default: last successful)")
    parser.add_argument("--ply", default=None, help="Render any standard 3DGS PLY instead of a session")
    parser.add_argument(
        "--pose-topic",
        default="/gs_camera_sim/pose",
        help="geometry_msgs/PoseStamped input (map frame, optical-convention camera)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay the session's mapped keyframe trajectory instead of subscribing to poses",
    )
    parser.add_argument("--loop", action="store_true", help="Restart the replay when it reaches the last keyframe")
    parser.add_argument(
        "--gt-pose-topic",
        default="/gs_camera_sim/gt_pose",
        help="PoseStamped ground-truth output while replaying (compare against the localizer estimate)",
    )
    parser.add_argument(
        "--image-topic",
        default="/gs_camera_sim/image_raw/compressed",
        help="Camera output; topics ending in /compressed publish sensor_msgs/CompressedImage",
    )
    parser.add_argument(
        "--camera-info-topic", default="/gs_camera_sim/camera_info", help="sensor_msgs/CameraInfo output topic"
    )
    parser.add_argument(
        "--depth-topic",
        default="",
        help="Optional sensor_msgs/Image 32FC1 depth output topic (empty = disabled)",
    )
    parser.add_argument("--map-frame", default="map", help="frame_id for ground-truth poses")
    parser.add_argument("--frame-id", default="camera", help="frame_id for published images")
    parser.add_argument("--fps", type=float, default=10.0, help="Publish rate in Hz")
    parser.add_argument("--width", type=int, default=None, help="Render width (default: session camera or 640)")
    parser.add_argument("--height", type=int, default=None, help="Render height (default: session camera or 480)")
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=None,
        help="Vertical field of view (default: session camera intrinsics or 60)",
    )
    parser.add_argument("--near-clip", type=float, default=0.01, help="Near clip plane (map gauge units)")
    parser.add_argument("--far-clip", type=float, default=500.0, help="Far clip plane (map gauge units)")
    parser.add_argument(
        "--renderer",
        choices=["auto", "simple", "gsplat"],
        default="auto",
        help="Rasterization backend. auto uses gsplat when CUDA and Gaussian parameters are available",
    )
    parser.add_argument(
        "--max-points", type=int, default=0, help="Subsample the PLY to at most this many points (0 = all)"
    )
    parser.add_argument("--point-radius", type=int, default=1, help="Point footprint radius for the simple backend")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for compressed output")
    parser.add_argument("--log-period", type=float, default=5.0, help="Status log period in seconds")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject argument combinations the node cannot serve."""
    if bool(args.session) == bool(args.ply):
        raise SystemExit("exactly one of --map or --ply is required")
    if args.replay and not args.session:
        raise SystemExit("--replay needs --map (the trajectory comes from the session's images.txt)")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")


def camera_intrinsics_from_colmap(cam: dict[str, Any]) -> tuple[int, int, float, float, float, float]:
    """COLMAP cameras.txt entry -> (width, height, fx, fy, cx, cy)."""
    params = cam["params"]
    model = cam["model"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in (
        "PINHOLE",
        "RADIAL",
        "RADIAL_FISHEYE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "THIN_PRISM_FISHEYE",
    ):
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    else:
        fx = fy = params[0] if params else float(cam["width"])
        cx, cy = cam["width"] / 2.0, cam["height"] / 2.0
    return int(cam["width"]), int(cam["height"]), float(fx), float(fy), float(cx), float(cy)


def scale_intrinsics(
    intrinsics: tuple[float, float, float, float],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Rescale (fx, fy, cx, cy) when rendering at a different resolution."""
    fx, fy, cx, cy = intrinsics
    sx = to_size[0] / float(from_size[0])
    sy = to_size[1] / float(from_size[1])
    return (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)


def vertical_fov_degrees(fy: float, height: int) -> float:
    """Vertical field of view implied by a pinhole fy."""
    return math.degrees(2.0 * math.atan(height / (2.0 * fy)))


def replay_poses(records: Sequence[Any]) -> list[tuple[str, CameraPose]]:
    """Mapped COLMAP records -> (keyframe name, optical-convention world pose)."""
    poses: list[tuple[str, CameraPose]] = []
    for record in records:
        position, orientation = colmap_to_ros_pose(record.qvec, record.tvec)
        poses.append((record.name, CameraPose(position=position, orientation=orientation)))
    return poses


def render_optical(
    renderer: HeadlessSplatRenderer,
    pose: CameraPose,
    *,
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
    near_clip: float,
    far_clip: float,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render an optical-convention (x right, y down, z forward) camera pose.

    The gsplat backend already uses the optical convention; the simple
    fallback projects with image-up = +y, so its output is mirrored
    vertically to match.
    """
    rgb, depth = renderer.render_rgbd(
        pose,
        width=width,
        height=height,
        fov_degrees=vertical_fov_degrees(intrinsics[1], height),
        near_clip=near_clip,
        far_clip=far_clip,
        point_radius=point_radius,
        intrinsics=intrinsics,
    )
    if renderer.backend != "gsplat":
        rgb = np.ascontiguousarray(rgb[::-1])
        depth = np.ascontiguousarray(depth[::-1])
    return rgb, depth


def _import_ros2() -> dict[str, Any]:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real ROS2 env
        raise RuntimeError(
            "ROS2 runtime not available. Source your ROS2 environment so `rclpy` and message packages are importable."
        ) from exc

    return {
        "rclpy": rclpy,
        "Node": Node,
        "PoseStamped": PoseStamped,
        "CameraInfo": CameraInfo,
        "CompressedImage": CompressedImage,
        "Image": Image,
    }


def _build_node_class(ros2: dict[str, Any]) -> type:
    Node = ros2["Node"]
    PoseStamped = ros2["PoseStamped"]
    CameraInfo = ros2["CameraInfo"]
    CompressedImage = ros2["CompressedImage"]
    Image = ros2["Image"]

    class CameraSimNode(Node):
        """Publishes rendered camera frames for poses in a trained splat map."""

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args.node_name)
            self.args = args

            if args.session:
                from gs_sim2real.robotics.localize import (
                    _load_cameras_txt,
                    load_mapped_records,
                    resolve_live_map_session,
                )

                session = resolve_live_map_session(Path(args.session), round_index=args.round)
                ply_path = session.round.ply_path
                records = load_mapped_records(session)
                cameras = _load_cameras_txt(session.round.cameras_txt)
                cam = cameras[records[0].camera_id] if records else next(iter(cameras.values()))
                native_w, native_h, fx, fy, cx, cy = camera_intrinsics_from_colmap(cam)
                self.width = args.width or native_w
                self.height = args.height or native_h
                if args.fov_degrees is not None:
                    self.intrinsics = compute_camera_intrinsics(self.width, self.height, args.fov_degrees)
                else:
                    self.intrinsics = scale_intrinsics(
                        (fx, fy, cx, cy), from_size=(native_w, native_h), to_size=(self.width, self.height)
                    )
                self._replay = replay_poses(records) if args.replay else []
                source = f"{args.session} round {session.round.round_index}"
            else:
                ply_path = Path(args.ply)
                self.width = args.width or 640
                self.height = args.height or 480
                self.intrinsics = compute_camera_intrinsics(self.width, self.height, args.fov_degrees or 60.0)
                self._replay = []
                source = str(ply_path)

            self.renderer = HeadlessSplatRenderer(ply_path, backend=args.renderer, max_points=args.max_points or None)

            self.compressed = args.image_topic.endswith("/compressed")
            image_type = CompressedImage if self.compressed else Image
            self.image_pub = self.create_publisher(image_type, args.image_topic, 10)
            self.camera_info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, 10)
            self.depth_pub = self.create_publisher(Image, args.depth_topic, 10) if args.depth_topic else None
            self.gt_pose_pub = self.create_publisher(PoseStamped, args.gt_pose_topic, 10) if args.replay else None

            self._pose: CameraPose | None = None
            self._replay_index = 0
            self._replay_name = ""
            self._frames_published = 0
            self._waiting_logged = False
            if not args.replay:
                self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, 10)

            self.create_timer(1.0 / args.fps, self._on_timer)
            self.create_timer(max(args.log_period, 1.0), self._log_status)
            self.get_logger().info(
                f"camera sim ready: map={source} points={self.renderer.num_points} "
                f"backend={self.renderer.backend} {self.width}x{self.height} fps={args.fps:.1f} "
                f"pose_source={'replay' if args.replay else args.pose_topic} image={args.image_topic}"
            )

        def _on_pose(self, message: Any) -> None:
            self._pose = CameraPose(
                position=(
                    float(message.pose.position.x),
                    float(message.pose.position.y),
                    float(message.pose.position.z),
                ),
                orientation=(
                    float(message.pose.orientation.x),
                    float(message.pose.orientation.y),
                    float(message.pose.orientation.z),
                    float(message.pose.orientation.w),
                ),
            )

        def _advance_replay(self) -> None:
            if not self._replay:
                return
            if self._replay_index >= len(self._replay):
                if self.args.loop:
                    self._replay_index = 0
                else:
                    self._replay_name, self._pose = self._replay[-1]
                    return
            self._replay_name, self._pose = self._replay[self._replay_index]
            self._replay_index += 1

        def _on_timer(self) -> None:
            if self.args.replay:
                self._advance_replay()
            if self._pose is None:
                if not self._waiting_logged:
                    self.get_logger().info(f"waiting for poses on {self.args.pose_topic}")
                    self._waiting_logged = True
                return
            self._waiting_logged = False

            rgb, depth = render_optical(
                self.renderer,
                self._pose,
                width=self.width,
                height=self.height,
                intrinsics=self.intrinsics,
                near_clip=self.args.near_clip,
                far_clip=self.args.far_clip,
                point_radius=self.args.point_radius,
            )
            self._publish(rgb, depth)

        def _publish(self, rgb: np.ndarray, depth: np.ndarray) -> None:
            stamp = self.get_clock().now().to_msg()

            if self.compressed:
                image = CompressedImage()
                image.header.stamp = stamp
                image.header.frame_id = self.args.frame_id
                image.format = "jpeg"
                image.data = encode_rgb_to_jpeg(rgb, quality=self.args.jpeg_quality)
            else:
                image = Image()
                image.header.stamp = stamp
                image.header.frame_id = self.args.frame_id
                image.height = int(self.height)
                image.width = int(self.width)
                image.encoding = "rgb8"
                image.is_bigendian = 0
                image.step = int(self.width * 3)
                image.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
            self.image_pub.publish(image)

            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = self.args.frame_id
            fx, fy, cx, cy = self.intrinsics
            info.width = int(self.width)
            info.height = int(self.height)
            info.distortion_model = "plumb_bob"
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.camera_info_pub.publish(info)

            if self.depth_pub is not None:
                depth_image = Image()
                depth_image.header.stamp = stamp
                depth_image.header.frame_id = self.args.frame_id
                depth_image.height = int(self.height)
                depth_image.width = int(self.width)
                depth_image.encoding = "32FC1"
                depth_image.is_bigendian = 0
                depth_image.step = int(self.width * 4)
                depth_image.data = np.asarray(depth, dtype="<f4").tobytes()
                self.depth_pub.publish(depth_image)

            if self.gt_pose_pub is not None and self._pose is not None:
                gt = PoseStamped()
                gt.header.stamp = stamp
                gt.header.frame_id = self.args.map_frame
                gt.pose.position.x, gt.pose.position.y, gt.pose.position.z = self._pose.position
                (
                    gt.pose.orientation.x,
                    gt.pose.orientation.y,
                    gt.pose.orientation.z,
                    gt.pose.orientation.w,
                ) = self._pose.orientation
                self.gt_pose_pub.publish(gt)

            self._frames_published += 1

        def _log_status(self) -> None:
            position = "-"
            if self._pose is not None:
                position = "({:.2f}, {:.2f}, {:.2f})".format(*self._pose.position)
            replay = f" replay={self._replay_name}" if self.args.replay else ""
            self.get_logger().info(f"frames={self._frames_published} pose={position}{replay}")

    return CameraSimNode


def run_cli(args: argparse.Namespace) -> None:
    validate_args(args)
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
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
