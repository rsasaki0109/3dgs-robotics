"""ROS 2 live 3DGS mapping node: camera topic in, growing browser .splat out.

Subscribes to a camera stream (and optionally odometry for translation-gated
keyframes), feeds a :class:`gs_sim2real.robotics.live_mapping.LiveMappingSession`,
and serves ``live/latest.splat`` + a polling web viewer over HTTP — watch the
map grow in the browser while the robot (or a rosbag replay) drives.

Usage (after sourcing the ROS 2 environment)::

    gs-mapper-live-mapper --image-topic /camera/image_raw/compressed \
        --odom-topic /odom --workdir outputs/live_mapping --port 8765

The reconstruction backend mirrors ``photos-to-splat`` (DUSt3R pose-free +
gsplat draft training); ``--method simple`` exercises the plumbing without
the DUSt3R checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gs-mapper-live-mapper",
        description="ROS 2 live 3DGS mapping node (camera topic -> growing .splat)",
    )
    parser.add_argument("--node-name", default="gs_mapper_live_mapper", help="ROS 2 node name")
    parser.add_argument(
        "--image-topic",
        default="/camera/image_raw/compressed",
        help="Camera topic; topics ending in /compressed subscribe sensor_msgs/CompressedImage",
    )
    parser.add_argument("--odom-topic", default=None, help="Optional nav_msgs/Odometry topic for motion gating")
    parser.add_argument("--workdir", default="outputs/live_mapping", help="Session output directory")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the live viewer (0 disables)")
    parser.add_argument("--method", default="dust3r", choices=["dust3r", "mast3r", "simple"])
    parser.add_argument("--iterations", type=int, default=1500, help="gsplat iterations per rebuild round")
    parser.add_argument("--align-iters", type=int, default=150, help="DUSt3R global alignment iterations per round")
    parser.add_argument("--scene-graph", default="swin-3", help="DUSt3R pair graph (sequential streams: swin-N)")
    parser.add_argument("--num-frames", type=int, default=24, help="Frame cap per rebuild (strided over the run)")
    parser.add_argument("--rebuild-min-new", type=int, default=4, help="New keyframes required to trigger a rebuild")
    parser.add_argument("--max-keyframes", type=int, default=512, help="Hard cap on stored keyframes")
    parser.add_argument("--min-keyframe-gap", type=float, default=1.0, help="Minimum seconds between keyframes")
    parser.add_argument(
        "--min-keyframe-motion",
        type=float,
        default=0.04,
        help="Minimum image change (0..1 thumbnail diff) between keyframes when no odometry is available",
    )
    parser.add_argument(
        "--min-translation", type=float, default=0.5, help="Minimum odometry translation (m) between keyframes"
    )
    parser.add_argument("--dust3r-checkpoint", default=None, help="DUSt3R checkpoint path or HF hub id")
    parser.add_argument("--dust3r-root", default=None, help="Local clone of naver/dust3r")
    parser.add_argument("--mast3r-root", default=None, help="Local clone of naver/mast3r")
    parser.add_argument(
        "--viewer-html",
        default=None,
        help="Viewer page copied to live/index.html (default: bundled docs/splat_live.html when found)",
    )
    parser.add_argument("--log-period", type=float, default=5.0, help="Status log period in seconds")
    return parser


def _import_ros2() -> dict[str, Any]:
    try:
        import rclpy
        from nav_msgs.msg import Odometry
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
        "Odometry": Odometry,
    }


def decode_compressed_image(message: Any) -> np.ndarray | None:
    """sensor_msgs/CompressedImage -> BGR ndarray (no cv_bridge required)."""
    import cv2

    buffer = np.frombuffer(bytes(message.data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def decode_raw_image(message: Any) -> np.ndarray | None:
    """sensor_msgs/Image (bgr8/rgb8/mono8/bgra8/rgba8) -> BGR ndarray."""
    import cv2

    encoding = (message.encoding or "").lower()
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1, "bgra8": 4, "rgba8": 4}.get(encoding)
    if channels is None:
        return None
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    row = message.step if message.step else message.width * channels
    try:
        image = data.reshape(message.height, row)[:, : message.width * channels]
        image = image.reshape(message.height, message.width, channels) if channels > 1 else image
    except ValueError:
        return None
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _stamp_to_seconds(message: Any, fallback: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None or (sec == 0 and nanosec == 0):
        return fallback
    return float(sec) + float(nanosec) * 1e-9


def _default_viewer_html() -> Path | None:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        viewer = parent / "docs" / "splat_live.html"
        if viewer.is_file():
            return viewer
    return None


def _build_node_class(ros2: dict[str, Any]) -> type:
    Node = ros2["Node"]
    Image = ros2["Image"]
    CompressedImage = ros2["CompressedImage"]
    Odometry = ros2["Odometry"]

    class LiveMapperNode(Node):
        """Feeds camera frames into a LiveMappingSession and logs progress."""

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args.node_name)
            from gs_sim2real.robotics.live_mapping import (
                LiveMapperConfig,
                LiveMappingSession,
                serve_live_dir,
            )

            viewer_html = Path(args.viewer_html) if args.viewer_html else _default_viewer_html()
            config = LiveMapperConfig(
                workdir=Path(args.workdir),
                method=args.method,
                min_keyframe_gap_s=args.min_keyframe_gap,
                min_keyframe_motion=args.min_keyframe_motion,
                min_translation_m=args.min_translation,
                rebuild_min_new_keyframes=args.rebuild_min_new,
                max_keyframes=args.max_keyframes,
                num_frames=args.num_frames,
                iterations=args.iterations,
                align_iters=args.align_iters,
                scene_graph=args.scene_graph,
                checkpoint=Path(args.dust3r_checkpoint) if args.dust3r_checkpoint else None,
                dust3r_root=Path(args.dust3r_root) if args.dust3r_root else None,
                mast3r_root=Path(args.mast3r_root) if args.mast3r_root else None,
                viewer_html=viewer_html,
            )
            self.session = LiveMappingSession(config)
            self.session.start()
            self.http_server = None
            if args.port > 0:
                self.http_server = serve_live_dir(self.session.live_dir, args.port)
                self.get_logger().info(f"live viewer: http://localhost:{args.port}/")

            self._latest_position: np.ndarray | None = None
            self._frames_received = 0
            self._frames_dropped = 0

            if args.image_topic.endswith("/compressed"):
                self.create_subscription(CompressedImage, args.image_topic, self._on_compressed_image, 5)
            else:
                self.create_subscription(Image, args.image_topic, self._on_raw_image, 5)
            if args.odom_topic:
                self.create_subscription(Odometry, args.odom_topic, self._on_odometry, 20)

            self.create_timer(max(args.log_period, 1.0), self._log_status)
            self.get_logger().info(
                f"live mapper ready: image={args.image_topic} odom={args.odom_topic or 'none'} "
                f"method={args.method} workdir={args.workdir}"
            )

        def _now_seconds(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _on_compressed_image(self, message: Any) -> None:
            self._ingest(decode_compressed_image(message), message)

        def _on_raw_image(self, message: Any) -> None:
            image = decode_raw_image(message)
            if image is None:
                self._frames_dropped += 1
                if self._frames_dropped == 1:
                    self.get_logger().warning(
                        f"unsupported image encoding '{message.encoding}'; supported: bgr8/rgb8/mono8/bgra8/rgba8"
                    )
                return
            self._ingest(image, message)

        def _ingest(self, image: np.ndarray | None, message: Any) -> None:
            if image is None:
                self._frames_dropped += 1
                return
            self._frames_received += 1
            timestamp = _stamp_to_seconds(message, fallback=self._now_seconds())
            self.session.add_frame(image, timestamp, position=self._latest_position)

        def _on_odometry(self, message: Any) -> None:
            p = message.pose.pose.position
            self._latest_position = np.array([p.x, p.y, p.z], dtype=np.float64)

        def _log_status(self) -> None:
            rounds = self.session.rounds
            last = next((r for r in reversed(rounds) if r.error is None), None)
            last_summary = (
                f"round {last.round_index}: {last.keyframes_used} kf -> "
                f"{last.splat_bytes / 1024:.0f} KB in {last.build_seconds:.1f}s"
                if last
                else "no map yet"
            )
            self.get_logger().info(
                f"frames={self._frames_received} (dropped={self._frames_dropped}) "
                f"keyframes={len(self.session.keyframes)} rounds={len(rounds)} | {last_summary}"
            )

        def shutdown(self) -> None:
            if self.http_server is not None:
                self.http_server.shutdown()
            self.session.stop(wait=True, timeout=30.0)

    return LiveMapperNode


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
