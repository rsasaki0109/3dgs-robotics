#!/usr/bin/env python3
"""Replay an image folder or a rosbag through the live mapping session — no ROS required.

Simulates a camera stream so the incremental "map grows while the robot
drives" demo (and its GIF) can be produced anywhere:

    python3 scripts/run_live_mapping_demo.py \
        --images data/my_drive_frames --fps 2 --port 8765

    # rosbag2 (.db3/.mcap) or ROS 1 .bag, paced by the recorded timestamps
    python3 scripts/run_live_mapping_demo.py \
        --bag data/my_drive_bag --image-topic /camera/image_raw --port 8765

Then open http://localhost:8765/ (status page) or the polling viewer:
docs/splat.html?url=http://localhost:8765/latest.splat&refresh=2

Per-round splats are kept under <workdir>/rounds/ for offline GIF timelines.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gs_sim2real.robotics.live_mapping import (  # noqa: E402
    LiveMapperConfig,
    LiveMappingSession,
    serve_live_dir,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
logger = logging.getLogger("live_mapping_demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", help="Directory of frames replayed in filename order")
    source.add_argument("--bag", help="rosbag input (.bag / .db3 / .mcap file, or rosbag2 directory)")
    parser.add_argument("--image-topic", default=None, help="Image topic to replay from --bag (auto when unique)")
    parser.add_argument("--workdir", default="outputs/live_mapping_demo", help="Session output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Replay rate for --images (frames per second)")
    parser.add_argument("--rate", type=float, default=1.0, help="Replay speed multiplier for --bag timestamps")
    parser.add_argument("--no-realtime", action="store_true", help="Feed frames without sleeping between them")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the live viewer (0 disables)")
    parser.add_argument("--method", default="dust3r", choices=["dust3r", "mast3r", "vggt", "simple"])
    parser.add_argument("--iterations", type=int, default=1500, help="gsplat iterations per rebuild round")
    parser.add_argument("--align-iters", type=int, default=150, help="DUSt3R alignment iterations per round")
    parser.add_argument("--scene-graph", default="swin-3", help="DUSt3R pair graph for sequential frames")
    parser.add_argument("--num-frames", type=int, default=24, help="Frame cap per rebuild")
    parser.add_argument("--rebuild-min-new", type=int, default=4, help="New keyframes per rebuild round")
    parser.add_argument("--min-keyframe-gap", type=float, default=0.4, help="Min seconds between keyframes")
    parser.add_argument("--min-keyframe-motion", type=float, default=0.02, help="Min thumbnail diff (0..1)")
    parser.add_argument("--dust3r-checkpoint", default=None, help="DUSt3R checkpoint path or HF hub id")
    parser.add_argument("--dust3r-root", default=None, help="Local clone of naver/dust3r")
    parser.add_argument(
        "--vggt-checkpoint",
        default=None,
        help="VGGT checkpoint .pt path or Hugging Face hub id (default: facebook/VGGT-1B)",
    )
    parser.add_argument("--vggt-root", default=None, help="Local clone of facebookresearch/vggt")
    parser.add_argument("--hold", action="store_true", help="Keep serving after the replay until Ctrl+C")
    return parser


def iter_folder_frames(images_dir: Path, fps: float, realtime: bool):
    """Yield (image_bgr, timestamp, sleep_before_s) for an image-folder replay."""
    import cv2

    frames = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(frames) < 2:
        print(f"Error: need at least 2 frames in {images_dir}", file=sys.stderr)
        sys.exit(2)
    interval = 1.0 / max(fps, 0.01) if realtime else 0.0
    for index, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("skipping unreadable frame %s", frame_path.name)
            continue
        yield image, index / max(fps, 0.01), interval if index > 0 else 0.0


def iter_rosbag_replay(bag: str, image_topic: str | None, rate: float, realtime: bool):
    """Yield (image_bgr, timestamp, sleep_before_s) paced by the bag's own timestamps."""
    from gs_sim2real.datasets.rosbag_frames import iter_bag_frames

    first_ts: float | None = None
    prev_ts: float | None = None
    for frame in iter_bag_frames(bag, image_topic):
        if first_ts is None:
            first_ts = frame.timestamp_sec
        sleep_s = 0.0
        if realtime and prev_ts is not None:
            sleep_s = max(0.0, (frame.timestamp_sec - prev_ts) / max(rate, 1e-6))
        prev_ts = frame.timestamp_sec
        yield frame.image_bgr, frame.timestamp_sec - first_ts, sleep_s


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()

    if args.bag:
        from gs_sim2real.datasets.rosbag_frames import resolve_image_topic

        try:
            selected = resolve_image_topic(args.bag, args.image_topic)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"Replaying {selected.msgcount} frames from {selected.topic} ({args.bag})")
        frame_source = iter_rosbag_replay(args.bag, selected.topic, args.rate, not args.no_realtime)
    else:
        frame_source = iter_folder_frames(Path(args.images), args.fps, not args.no_realtime)

    viewer_html = REPO_ROOT / "docs" / "splat_live.html"
    config = LiveMapperConfig(
        workdir=Path(args.workdir),
        method=args.method,
        min_keyframe_gap_s=args.min_keyframe_gap,
        min_keyframe_motion=args.min_keyframe_motion,
        rebuild_min_new_keyframes=args.rebuild_min_new,
        num_frames=args.num_frames,
        iterations=args.iterations,
        align_iters=args.align_iters,
        scene_graph=args.scene_graph,
        checkpoint=(
            args.vggt_checkpoint
            if args.method == "vggt" and args.vggt_checkpoint
            else (Path(args.dust3r_checkpoint) if args.dust3r_checkpoint else None)
        ),
        dust3r_root=Path(args.dust3r_root) if args.dust3r_root else None,
        vggt_root=Path(args.vggt_root) if args.vggt_root else None,
        viewer_html=viewer_html if viewer_html.is_file() else None,
    )
    session = LiveMappingSession(config)
    session.start()
    server = serve_live_dir(session.live_dir, args.port) if args.port > 0 else None
    if server is not None:
        print(f"Live status page: http://localhost:{args.port}/")
        print(f"Polling 3D viewer: docs/splat.html?url=http://localhost:{args.port}/latest.splat&refresh=2")

    start = time.time()
    fed = 0
    accepted = 0
    try:
        for image, timestamp, sleep_before in frame_source:
            if sleep_before > 0:
                time.sleep(sleep_before)
            fed += 1
            if session.add_frame(image, timestamp):
                accepted += 1
        logger.info(
            "replay done: %d frames fed, %d keyframes accepted in %.1fs",
            fed,
            accepted,
            time.time() - start,
        )
        if args.hold:
            print("Replay finished; still serving (Ctrl+C to stop)...")
            while True:
                time.sleep(1.0)
        else:
            logger.info("waiting for the final rebuild round...")
    except KeyboardInterrupt:
        pass
    finally:
        session.stop(wait=True, timeout=600.0)
        if server is not None:
            server.shutdown()

    successful = [r for r in session.rounds if r.error is None]
    failed = [r for r in session.rounds if r.error is not None]
    print(
        f"Session summary: {len(session.keyframes)} keyframes, "
        f"{len(successful)} successful rounds, {len(failed)} failed rounds"
    )
    if successful:
        print(f"Final map: {session.live_dir / 'latest.splat'}")
        print(f"Per-round timeline: {session.rounds_dir}/round_*/scene.splat")
    if failed:
        print(f"First failure: {failed[0].error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
