#!/usr/bin/env python3
"""Replay an image folder through the live mapping session — no ROS required.

Simulates a camera stream so the incremental "map grows while the robot
drives" demo (and its GIF) can be produced anywhere:

    python3 scripts/run_live_mapping_demo.py \
        --images data/my_drive_frames --fps 2 --port 8765

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
    parser.add_argument("--images", required=True, help="Directory of frames replayed in filename order")
    parser.add_argument("--workdir", default="outputs/live_mapping_demo", help="Session output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Replay rate (frames per second)")
    parser.add_argument("--no-realtime", action="store_true", help="Feed frames without sleeping between them")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the live viewer (0 disables)")
    parser.add_argument("--method", default="dust3r", choices=["dust3r", "mast3r", "simple"])
    parser.add_argument("--iterations", type=int, default=1500, help="gsplat iterations per rebuild round")
    parser.add_argument("--align-iters", type=int, default=150, help="DUSt3R alignment iterations per round")
    parser.add_argument("--scene-graph", default="swin-3", help="DUSt3R pair graph for sequential frames")
    parser.add_argument("--num-frames", type=int, default=24, help="Frame cap per rebuild")
    parser.add_argument("--rebuild-min-new", type=int, default=4, help="New keyframes per rebuild round")
    parser.add_argument("--min-keyframe-gap", type=float, default=0.4, help="Min seconds between keyframes")
    parser.add_argument("--min-keyframe-motion", type=float, default=0.02, help="Min thumbnail diff (0..1)")
    parser.add_argument("--dust3r-checkpoint", default=None, help="DUSt3R checkpoint path or HF hub id")
    parser.add_argument("--dust3r-root", default=None, help="Local clone of naver/dust3r")
    parser.add_argument("--hold", action="store_true", help="Keep serving after the replay until Ctrl+C")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()

    import cv2

    images_dir = Path(args.images)
    frames = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(frames) < 2:
        print(f"Error: need at least 2 frames in {images_dir}", file=sys.stderr)
        sys.exit(2)

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
        checkpoint=Path(args.dust3r_checkpoint) if args.dust3r_checkpoint else None,
        dust3r_root=Path(args.dust3r_root) if args.dust3r_root else None,
        viewer_html=viewer_html if viewer_html.is_file() else None,
    )
    session = LiveMappingSession(config)
    session.start()
    server = serve_live_dir(session.live_dir, args.port) if args.port > 0 else None
    if server is not None:
        print(f"Live status page: http://localhost:{args.port}/")
        print(f"Polling 3D viewer: docs/splat.html?url=http://localhost:{args.port}/latest.splat&refresh=2")

    interval = 0.0 if args.no_realtime else 1.0 / max(args.fps, 0.01)
    start = time.time()
    accepted = 0
    try:
        for index, frame_path in enumerate(frames):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("skipping unreadable frame %s", frame_path.name)
                continue
            timestamp = index / max(args.fps, 0.01)
            if session.add_frame(image, timestamp):
                accepted += 1
            if interval > 0:
                time.sleep(interval)
        logger.info(
            "replay done: %d frames fed, %d keyframes accepted in %.1fs",
            len(frames),
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
