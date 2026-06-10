"""ROS 2 3DGS localizer node: camera topic in, pose in the splat map out.

Subscribes to a camera stream and localizes each frame against a finished (or
still growing, with ``--follow-latest``) live-mapping session using thumbnail
retrieval + differentiable gsplat photometric refinement
(:mod:`gs_sim2real.robotics.localize`). Publishes ``geometry_msgs/PoseStamped``,
an accumulated ``nav_msgs/Path`` for RViz/Foxglove, and (optionally) a
``map -> camera`` TF.

Localization runs in a background worker that always picks the most recent
frame, so a fast camera topic never queues up behind the GPU.

Usage (after sourcing the ROS 2 environment)::

    3dgs-robotics-localizer --map outputs/live_mapping/session \
        --image-topic /camera/image_raw/compressed

Poses are expressed in the map round's reconstruction gauge (COLMAP world
frame; arbitrary scale unless the map was built with metric poses). The camera
frame follows the optical convention: x right, y down, z forward.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from gs_sim2real.robotics.live_mapper_node import (
    _stamp_to_seconds,
    decode_compressed_image,
    decode_raw_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3dgs-robotics-localizer",
        description="ROS 2 3DGS localizer node (camera topic -> PoseStamped in the splat map)",
    )
    parser.add_argument("--node-name", default="gs_mapper_localizer", help="ROS 2 node name")
    parser.add_argument("--map", dest="session", required=True, help="Live-mapping session directory (workdir)")
    parser.add_argument("--round", type=int, default=None, help="Pin one rebuild round (default: last successful)")
    parser.add_argument(
        "--follow-latest",
        action="store_true",
        help="Reload the map whenever the live-mapping session finishes a newer round",
    )
    parser.add_argument(
        "--image-topic",
        default="/camera/image_raw/compressed",
        help="Camera topic; topics ending in /compressed subscribe sensor_msgs/CompressedImage",
    )
    parser.add_argument("--pose-topic", default="/gs_localizer/pose", help="geometry_msgs/PoseStamped output topic")
    parser.add_argument("--path-topic", default="/gs_localizer/path", help="nav_msgs/Path output topic (trajectory)")
    parser.add_argument("--map-frame", default="map", help="frame_id for published poses and TF parent")
    parser.add_argument("--camera-frame", default="camera", help="TF child frame (optical convention: z forward)")
    parser.add_argument("--no-tf", action="store_true", help="Disable the map -> camera TF broadcast")
    parser.add_argument("--max-path-poses", type=int, default=500, help="Trajectory poses kept in the Path message")
    parser.add_argument(
        "--min-period",
        type=float,
        default=0.0,
        help="Minimum seconds between localizations (0 = back-to-back, latest frame wins)",
    )
    parser.add_argument(
        "--max-seed-distance",
        type=float,
        default=0.5,
        help="Drop estimates whose retrieval seed thumbnail distance exceeds this (lost / off-map)",
    )
    parser.add_argument("--device", default="cuda", help="torch device for gsplat refinement")
    parser.add_argument("--refine-iters", type=int, default=40, help="Photometric refinement iterations per scale")
    parser.add_argument("--refine-lr", type=float, default=0.005, help="Refinement learning rate")
    parser.add_argument(
        "--pyramid-scales",
        default="0.25,0.5",
        help="Comma-separated render scales for coarse-to-fine refinement (streaming default favours speed)",
    )
    parser.add_argument("--log-period", type=float, default=5.0, help="Status log period in seconds")
    return parser


def parse_pyramid_scales(text: str) -> tuple[float, ...]:
    """``"0.25,0.5"`` -> ``(0.25, 0.5)``, validating each scale."""
    scales = tuple(float(part) for part in text.split(",") if part.strip())
    if not scales or any(not 0.0 < scale <= 1.0 for scale in scales):
        raise ValueError(f"pyramid scales must be in (0, 1]: {text!r}")
    return scales


def colmap_to_ros_pose(
    qvec: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """COLMAP world-to-camera qvec (wxyz) / tvec -> ROS position + orientation (xyzw).

    The returned pose is world-from-camera: position is the camera center,
    orientation rotates camera-frame vectors into the map frame.
    """
    w, x, y, z = (float(v) for v in qvec)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z)) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    r_cw = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    center = -(r_cw.T @ np.asarray(tvec, dtype=np.float64))
    position = (float(center[0]), float(center[1]), float(center[2]))
    orientation_xyzw = (-x, -y, -z, w)  # conjugate = inverse for unit quaternions
    return position, orientation_xyzw


class LatestFrame:
    """Thread-safe slot holding the newest camera frame (older frames are dropped)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame: tuple[np.ndarray, float] | None = None
        self.dropped = 0

    def put(self, image_bgr: np.ndarray, timestamp: float) -> None:
        with self._lock:
            if self._frame is not None:
                self.dropped += 1
            self._frame = (image_bgr, timestamp)
        self._event.set()

    def take(self, timeout: float = 0.2) -> tuple[np.ndarray, float] | None:
        if not self._event.wait(timeout):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._event.clear()
        return frame


def _import_ros2() -> dict[str, Any]:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from nav_msgs.msg import Path as PathMsg
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage, Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real ROS2 env
        raise RuntimeError(
            "ROS2 runtime not available. Source your ROS2 environment so `rclpy` and message packages are importable."
        ) from exc

    return {
        "rclpy": rclpy,
        "Node": Node,
        "Image": Image,
        "CompressedImage": CompressedImage,
        "PoseStamped": PoseStamped,
        "TransformStamped": TransformStamped,
        "PathMsg": PathMsg,
    }


def _build_node_class(ros2: dict[str, Any]) -> type:
    Node = ros2["Node"]
    Image = ros2["Image"]
    CompressedImage = ros2["CompressedImage"]
    PoseStamped = ros2["PoseStamped"]
    TransformStamped = ros2["TransformStamped"]
    PathMsg = ros2["PathMsg"]

    class LocalizerNode(Node):
        """Localizes camera frames against a live-mapping splat map."""

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args.node_name)
            from gs_sim2real.robotics.localize import LocalizeConfig, SessionLocalizer

            config = LocalizeConfig(
                device=args.device,
                refine_iters=args.refine_iters,
                refine_lr=args.refine_lr,
                pyramid_scales=parse_pyramid_scales(args.pyramid_scales),
            )
            self.args = args
            self.localizer = SessionLocalizer(Path(args.session), round_index=args.round, config=config)
            self.get_logger().info(
                f"map loaded: {args.session} round {self.localizer.round_index} "
                f"({len(self.localizer.mapped_records)} mapped keyframes)"
            )

            self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 10)
            self.path_pub = self.create_publisher(PathMsg, args.path_topic, 10)
            self.path_msg = PathMsg()
            self.path_msg.header.frame_id = args.map_frame
            self.tf_broadcaster = None
            if not args.no_tf:
                from tf2_ros import TransformBroadcaster

                self.tf_broadcaster = TransformBroadcaster(self)

            if args.image_topic.endswith("/compressed"):
                self.create_subscription(CompressedImage, args.image_topic, self._on_compressed_image, 1)
            else:
                self.create_subscription(Image, args.image_topic, self._on_raw_image, 1)

            self._latest = LatestFrame()
            self._frames_received = 0
            self._localized = 0
            self._rejected = 0
            self._last_result_line = "no pose yet"
            self._stop = threading.Event()
            self._worker = threading.Thread(target=self._work_loop, name="gs-localizer", daemon=True)
            self._worker.start()

            self.create_timer(max(args.log_period, 1.0), self._log_status)
            self.get_logger().info(
                f"localizer ready: image={args.image_topic} pose={args.pose_topic} "
                f"tf={'off' if args.no_tf else f'{args.map_frame}->{args.camera_frame}'} "
                f"follow_latest={args.follow_latest}"
            )

        def _now_seconds(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _on_compressed_image(self, message: Any) -> None:
            self._ingest(decode_compressed_image(message), message)

        def _on_raw_image(self, message: Any) -> None:
            self._ingest(decode_raw_image(message), message)

        def _ingest(self, image: np.ndarray | None, message: Any) -> None:
            if image is None:
                return
            self._frames_received += 1
            self._latest.put(image, _stamp_to_seconds(message, fallback=self._now_seconds()))

        def _work_loop(self) -> None:
            while not self._stop.is_set():
                frame = self._latest.take(timeout=0.2)
                if frame is None:
                    continue
                image, timestamp = frame
                try:
                    if self.args.follow_latest and self.localizer.maybe_reload_latest():
                        self.get_logger().info(f"map reloaded: round {self.localizer.round_index}")
                    result = self.localizer.localize(image)
                except Exception as error:  # noqa: BLE001 - keep the stream alive
                    self.get_logger().error(f"localization failed: {error}")
                    continue
                if result.seed_distance > self.args.max_seed_distance:
                    self._rejected += 1
                    self._last_result_line = f"rejected (seed distance {result.seed_distance:.3f})"
                    continue
                self._publish(result, timestamp)
                self._localized += 1
                self._last_result_line = (
                    f"seed {result.seed_keyframe} d={result.seed_distance:.3f} "
                    f"loss={result.refine_loss:.4f} center=({result.center[0]:.2f}, "
                    f"{result.center[1]:.2f}, {result.center[2]:.2f})"
                )
                if self.args.min_period > 0:
                    self._stop.wait(self.args.min_period)

        def _publish(self, result: Any, timestamp: float) -> None:
            position, orientation = colmap_to_ros_pose(result.qvec, result.tvec)
            stamp_sec = int(timestamp)
            stamp_nanosec = int(round((timestamp - stamp_sec) * 1e9))

            pose = PoseStamped()
            pose.header.frame_id = self.args.map_frame
            pose.header.stamp.sec = stamp_sec
            pose.header.stamp.nanosec = stamp_nanosec
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ) = orientation
            self.pose_pub.publish(pose)

            self.path_msg.header.stamp = pose.header.stamp
            self.path_msg.poses.append(pose)
            if len(self.path_msg.poses) > self.args.max_path_poses:
                self.path_msg.poses = self.path_msg.poses[-self.args.max_path_poses :]
            self.path_pub.publish(self.path_msg)

            if self.tf_broadcaster is not None:
                tf = TransformStamped()
                tf.header = pose.header
                tf.child_frame_id = self.args.camera_frame
                tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = position
                (
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ) = orientation
                self.tf_broadcaster.sendTransform(tf)

        def _log_status(self) -> None:
            self.get_logger().info(
                f"frames={self._frames_received} localized={self._localized} "
                f"rejected={self._rejected} dropped={self._latest.dropped} | {self._last_result_line}"
            )

        def shutdown(self) -> None:
            self._stop.set()
            self._worker.join(timeout=5.0)

    return LocalizerNode


def run_cli(args: argparse.Namespace) -> None:
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
        node.shutdown()
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
