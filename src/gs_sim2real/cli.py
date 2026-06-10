"""Command-line interface for GS Mapper.

Provides subcommands for the full 3DGS pipeline:
- download: Download datasets from supported sources
- preprocess: Run COLMAP or frame extraction on raw data
- train: Train a 3DGS model using gsplat or nerfstudio
- view: Launch the web viewer for a trained model
- run: Run the full pipeline end-to-end
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_PREPROCESS_METHOD_CHOICES = [
    "colmap",
    "frames",
    "pose-free",
    "dust3r",
    "mast3r",
    "vggt",
    "simple",
    "waymo",
    "mcd",
    "lidar-slam",
    "external-slam",
]
_PIPELINE_PREPROCESS_METHOD_CHOICES = [
    "colmap",
    "pose-free",
    "dust3r",
    "mast3r",
    "vggt",
    "simple",
    "waymo",
    "mcd",
    "lidar-slam",
    "external-slam",
]
_EXTERNAL_SLAM_SYSTEM_CHOICES = ["generic", "mast3r-slam", "vggt-slam", "loger", "pi3"]
_PHOTOS_TO_SPLAT_DEFAULTS = {
    "num_frames": 20,
    "align_iters": 300,
    "iterations": 3000,
    "mast3r_subsample": 8,
    "splat_max_points": 400000,
    "splat_min_opacity": 0.02,
    "splat_max_scale": 2.0,
    "splat_max_scale_percentile": None,
}
_PHOTOS_TO_SPLAT_QUALITY_PRESETS = {
    "draft": {},
    "balanced": {
        "num_frames": 32,
        "align_iters": 500,
        "iterations": 10000,
        "mast3r_subsample": 6,
        "splat_min_opacity": 0.025,
        "splat_max_scale": 1.25,
        "splat_max_scale_percentile": 99.0,
    },
    "clean": {
        "num_frames": 64,
        "align_iters": 800,
        "iterations": 30000,
        "mast3r_subsample": 4,
        "splat_max_points": 600000,
        "splat_min_opacity": 0.035,
        "splat_max_scale": 1.0,
        "splat_max_scale_percentile": 98.0,
    },
    "hero": {
        "num_frames": 0,
        "align_iters": 1200,
        "iterations": 50000,
        "mast3r_subsample": 2,
        "splat_max_points": 800000,
        "splat_min_opacity": 0.04,
        "splat_max_scale": 0.8,
        "splat_max_scale_percentile": 96.0,
    },
}


def _add_photos_to_splat_arguments(
    parser: argparse.ArgumentParser,
    *,
    num_frames_default: int = 20,
    output_default: str | None = "outputs/photos_splat",
) -> None:
    """Register shared photos/video-to-splat pipeline flags."""
    if output_default is None:
        output_help = "Root output directory (default: outputs/<video-stem>_splat)"
    else:
        output_help = f"Root output directory (default: {output_default})"
    parser.add_argument("--output", default=output_default, help=output_help)
    parser.add_argument(
        "--preprocess",
        choices=["dust3r", "mast3r", "vggt", "simple"],
        default="dust3r",
        help="Pose-estimation backend. 'mast3r' uses naver/mast3r (newer, metric-aware). "
        "'vggt' uses facebook/VGGT-1B feedforward reconstruction (requires VGGT clone). "
        "'simple' is a non-metric circular fallback for smoke tests.",
    )
    parser.add_argument(
        "--quality",
        choices=list(_PHOTOS_TO_SPLAT_QUALITY_PRESETS),
        default="draft",
        help=(
            "Quality preset. draft preserves the fast legacy defaults; clean/hero use more frames, "
            "more alignment/training iterations, and stricter export filtering for cleaner maps."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=num_frames_default,
        help=f"Frame cap for pose-free reconstruction (0 = all). Default {num_frames_default}.",
    )
    parser.add_argument(
        "--scene-graph", default="complete", help="DUSt3R/MAST3R pair graph (complete / swin-N / oneref-K)"
    )
    parser.add_argument("--dust3r-checkpoint", default=None, help="DUSt3R checkpoint .pth path")
    parser.add_argument(
        "--dust3r-root", default=None, help="Local clone of naver/dust3r (default: DUST3R_PATH env or /tmp/dust3r)"
    )
    parser.add_argument("--mast3r-checkpoint", default=None, help="MAST3R checkpoint .pth path")
    parser.add_argument(
        "--mast3r-root", default=None, help="Local clone of naver/mast3r (default: MAST3R_PATH env or /tmp/mast3r)"
    )
    parser.add_argument(
        "--vggt-checkpoint",
        default=None,
        help="VGGT checkpoint .pt path or Hugging Face hub id (default: facebook/VGGT-1B)",
    )
    parser.add_argument(
        "--vggt-root",
        default=None,
        help="Local clone of facebookresearch/vggt (default: VGGT_PATH env or /tmp/vggt)",
    )
    parser.add_argument("--mast3r-subsample", type=int, default=8, help="MAST3R pointcloud subsample stride")
    parser.add_argument("--align-iters", type=int, default=300, help="DUSt3R global alignment iterations")
    parser.add_argument("--iterations", type=int, default=3000, help="gsplat training iterations")
    parser.add_argument("--config", default=None, help="Training config YAML override")
    parser.add_argument(
        "--splat-max-points", type=int, default=400000, help="Max gaussians in .splat output (default: 400k)"
    )
    parser.add_argument(
        "--splat-normalize-extent",
        type=float,
        default=17.0,
        help="Rescale so the scene max-axis extent matches this (matches docs/splat.html defaults).",
    )
    parser.add_argument("--splat-min-opacity", type=float, default=0.02, help="Drop gaussians below this opacity")
    parser.add_argument("--splat-max-scale", type=float, default=2.0, help="Drop gaussians above this scale (meters)")
    parser.add_argument(
        "--splat-max-scale-percentile",
        type=float,
        default=None,
        help="Drop gaussians above this adaptive max-scale percentile during .splat export.",
    )
    parser.add_argument("--skip-data-check", action="store_true", help="Skip COLMAP sparse preflight before training")


def _register_video_to_splat_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``video-to-splat`` and the shorter ``map`` alias."""
    for name, help_text in (
        ("video-to-splat", "One-shot: mp4/mov -> frame extract -> pose-free -> gsplat train -> .splat file"),
        ("map", "Alias for video-to-splat: one video file -> browser-ready .splat"),
    ):
        parser = subparsers.add_parser(name, help=help_text)
        parser.add_argument("video", help="Input video file (.mp4, .mov, .mkv, ...)")
        _add_photos_to_splat_arguments(parser, num_frames_default=32, output_default=None)
        parser.add_argument(
            "--open-viewer",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Start a local HTTP server and open docs/splat.html when training finishes (default: on)",
        )
        parser.add_argument(
            "--viewer-port",
            type=int,
            default=8000,
            help="Port for the local docs HTTP server used by --open-viewer (default: 8000)",
        )


def _add_external_slam_args(parser: argparse.ArgumentParser, *, context: str) -> None:
    parser.add_argument(
        "--external-slam-system",
        default="generic",
        help=(
            f"External SLAM artifact convention for {context}: "
            f"{', '.join(_EXTERNAL_SLAM_SYSTEM_CHOICES)} (default: generic; common aliases accepted)"
        ),
    )
    parser.add_argument(
        "--external-slam-output",
        default=None,
        help=f"Directory containing external SLAM outputs for {context}; used to auto-discover trajectory/point cloud",
    )
    parser.add_argument(
        "--pinhole-calib",
        default=None,
        help=f"Optional PINHOLE calibration JSON for {context} trajectory import",
    )
    if context == "preprocess":
        parser.add_argument(
            "--external-slam-dry-run",
            action="store_true",
            help="Resolve external SLAM artifacts and print an import manifest without writing COLMAP files",
        )
        parser.add_argument(
            "--external-slam-manifest-format",
            choices=["text", "json"],
            default="text",
            help="Manifest format for --external-slam-dry-run (default: text)",
        )
        parser.add_argument(
            "--external-slam-fail-on-dry-run-gate",
            action="store_true",
            help="Exit with status 2 when the external SLAM dry-run manifest gate fails",
        )
        parser.add_argument(
            "--external-slam-min-aligned-frames",
            type=int,
            default=2,
            help="Minimum aligned image/pose pairs required by the dry-run gate",
        )
        parser.add_argument(
            "--external-slam-allow-dropped-images",
            action="store_true",
            help="Allow image frames without matching poses in the dry-run gate",
        )
        parser.add_argument(
            "--external-slam-require-pointcloud",
            action="store_true",
            help="Require a resolved point cloud in the dry-run gate",
        )
        parser.add_argument(
            "--external-slam-min-point-count",
            type=int,
            default=0,
            help="Minimum point count required by the dry-run gate when point count is known",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="gs-mapper",
        description="Large-scale 3D Gaussian Splatting mapper for robotics and driving datasets",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>", help="Available commands")

    # download
    dl = subparsers.add_parser("download", help="Download a dataset")
    dl.add_argument("--dataset", default=None, help="Dataset name (e.g. covla, mcd, autoware_leo_drive_bagN)")
    dl.add_argument("--output", default=None, help="Output directory (default: data/)")
    dl.add_argument("--max-samples", type=int, default=None, help="Max samples to download")
    dl.add_argument("--sample-images", action="store_true", help="Download sample images for quick testing")

    # preprocess
    pp = subparsers.add_parser("preprocess", help="Preprocess images with COLMAP or frame extraction")
    pp.add_argument("--images", required=True, help="Input image directory or video file")
    pp.add_argument("--output", default="outputs/colmap", help="Output directory")
    pp.add_argument(
        "--method",
        choices=_PREPROCESS_METHOD_CHOICES,
        default="colmap",
        help="Preprocessing method (default: colmap). "
        "'pose-free' and 'dust3r' use DUSt3R for pose estimation; "
        "'mast3r' uses MAST3R sparse global alignment; "
        "'vggt' uses facebook/VGGT-1B feedforward reconstruction; "
        "'simple' uses circular camera initialization; "
        "'waymo' extracts frames from Waymo tfrecord files; "
        "'mcd' extracts images and optional sensors from MCD rosbags; "
        "'lidar-slam' imports an external trajectory; "
        "'external-slam' imports artifacts from MASt3R-SLAM, VGGT-SLAM 2.0, LoGeR, Pi3, or another front-end.",
    )
    pp.add_argument("--fps", type=float, default=2.0, help="FPS for frame extraction (default: 2)")
    pp.add_argument("--max-frames", type=int, default=100, help="Max frames to extract (default: 100)")
    pp.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy (default: exhaustive)",
    )
    pp.add_argument("--colmap-path", default="colmap", help="Path to the COLMAP executable (default: colmap)")
    pp.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP")
    pp.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera to extract (default: FRONT)",
    )
    pp.add_argument("--every-n", type=int, default=1, help="Extract every N-th frame for Waymo/MCD (default: 1)")
    pp.add_argument("--image-topic", default=None, help="ROS image topic for MCD preprocessing")
    pp.add_argument("--lidar-topic", default=None, help="ROS PointCloud2 topic for MCD preprocessing")
    pp.add_argument("--imu-topic", default=None, help="ROS IMU topic for MCD preprocessing")
    pp.add_argument(
        "--list-topics",
        action="store_true",
        help="For MCD preprocessing, print bag topics with inferred roles and exit",
    )
    pp.add_argument(
        "--extract-lidar-depth",
        action="store_true",
        help="For Waymo preprocessing, also project TOP lidar into per-frame depth maps",
    )
    pp.add_argument(
        "--extract-dynamic-masks",
        action="store_true",
        help="For Waymo preprocessing, also generate per-frame dynamic-object masks from camera labels",
    )
    pp.add_argument(
        "--extract-lidar",
        action="store_true",
        help="For MCD preprocessing, also export PointCloud2 frames to lidar/*.npy",
    )
    pp.add_argument(
        "--extract-imu",
        action="store_true",
        help="For MCD preprocessing, also export IMU measurements to imu.csv",
    )
    pp.add_argument(
        "--gnss-topic",
        default=None,
        help="For MCD preprocessing, NavSatFix topic used with --mcd-seed-poses-from-gnss (default: try /gnss/fix)",
    )
    pp.add_argument(
        "--mcd-flatten-gnss-altitude",
        action="store_true",
        help="For MCD GNSS seeding, project NavSatFix altitude to the median valid altitude before ENU conversion",
    )
    pp.add_argument(
        "--mcd-start-offset-sec",
        type=float,
        default=0.0,
        help="For MCD preprocessing, skip the first N seconds of image/LiDAR/GNSS streams",
    )
    pp.add_argument(
        "--mcd-seed-poses-from-gnss",
        action="store_true",
        help="For MCD preprocessing, write COLMAP sparse from GNSS (NavSatFix) + images; single --image-topic only",
    )
    pp.add_argument(
        "--mcd-base-frame",
        default="base_link",
        help="For MCD GNSS seeding, parent frame for /tf_static lookup (default: base_link)",
    )
    pp.add_argument(
        "--mcd-static-calibration",
        default="",
        help="MCDVIRAL rig calibration YAML (body→sensor 4×4 T) when bags lack /tf_static",
    )
    pp.add_argument(
        "--mcd-camera-frame",
        default=None,
        help="For MCD GNSS seeding, camera frame id (default: CameraInfo header.frame_id)",
    )
    pp.add_argument(
        "--mcd-disable-tf-extrinsics",
        action="store_true",
        help="For MCD GNSS seeding, ignore /tf_static (GNSS-only trajectory at vehicle frame)",
    )
    pp.add_argument(
        "--mcd-include-tf-dynamic",
        action="store_true",
        help="For MCD GNSS seeding, merge /tf into TF map after /tf_static (slower on large bags)",
    )
    pp.add_argument(
        "--mcd-gnss-antenna-offset-enu",
        nargs=3,
        type=float,
        default=None,
        metavar=("E", "N", "U"),
        help="Subtract (East, North, Up) meters from each NavSat fix in ENU (approx. base vs antenna)",
    )
    pp.add_argument(
        "--mcd-gnss-antenna-offset-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Antenna position in base_link (x forward, y left, z up, metres); "
            "subtract using per-sample heading from GNSS motion (do not combine with --mcd-gnss-antenna-offset-enu)"
        ),
    )
    pp.add_argument(
        "--mcd-tf-use-image-stamps",
        action="store_true",
        help="Multi-camera GNSS seed: resolve TF at each image time (/tf + /tf_static topology)",
    )
    pp.add_argument(
        "--mcd-lidar-frame",
        default="",
        help="For MCD GNSS seeding, LiDAR frame id under base_link (empty = identity T_base_lidar)",
    )
    pp.add_argument(
        "--mcd-skip-lidar-seed",
        action="store_true",
        help="For MCD GNSS seeding, skip merging LiDAR frames to world as points3D seed",
    )
    pp.add_argument(
        "--mcd-skip-lidar-colorize",
        action="store_true",
        help="Skip the image->LiDAR RGB projection that seeds points3D.txt with real colors",
    )
    pp.add_argument(
        "--mcd-export-depth",
        action="store_true",
        help="Project the world LiDAR cloud into each training image as sparse depth .npy (for depth_loss_weight > 0)",
    )
    pp.add_argument(
        "--mcd-reference-origin",
        default="",
        help="Share the ENU origin across bags. 'lat,lon,alt' in WGS84 degrees/metres.",
    )
    pp.add_argument(
        "--mcd-reference-bag",
        default="",
        help="Use the ENU origin recorded under <path>/pose/origin_wgs84.json from a previously preprocessed bag.",
    )
    pp.add_argument(
        "--mcd-imu-csv",
        default="",
        help="Path to an imu.csv with orientation_* columns. Interpolated into each TUM row as the base_link quaternion "
        "(falls back to the motion-inferred yaw if the column is constant).",
    )
    pp.add_argument(
        "--mcd-skip-imu-orientation",
        action="store_true",
        help="Ignore any imu.csv and keep the default motion-inferred yaw.",
    )
    pp.add_argument("--trajectory", default=None, help="SLAM trajectory file (for lidar-slam method)")
    pp.add_argument(
        "--trajectory-format",
        choices=["tum", "kitti", "nmea"],
        default="tum",
        help="Trajectory format (default: tum)",
    )
    pp.add_argument(
        "--nmea-time-offset-sec",
        type=float,
        default=0.0,
        help="Fixed seconds added to NMEA-derived timestamps to realign against a drifted logger clock.",
    )
    pp.add_argument("--pointcloud", default=None, help="Point cloud file for lidar-slam (.ply/.npy/.pcd)")
    _add_external_slam_args(pp, context="preprocess")
    pp.add_argument(
        "--vggt-checkpoint",
        default=None,
        help="For --method vggt, checkpoint .pt path or Hugging Face hub id (default: facebook/VGGT-1B)",
    )
    pp.add_argument(
        "--vggt-root",
        default=None,
        help="For --method vggt, local clone of facebookresearch/vggt (default: VGGT_PATH env or /tmp/vggt)",
    )

    # train
    tr = subparsers.add_parser("train", help="Train a 3DGS model")
    tr.add_argument("--data", required=True, help="Preprocessed data directory")
    tr.add_argument("--output", default="outputs/train", help="Training output directory")
    tr.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    tr.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    tr.add_argument("--config", default=None, help="Path to training config YAML override")
    tr.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat training (not recommended)",
    )

    lsgd = subparsers.add_parser(
        "large-scale-3dgs-discover",
        help="Discover COLMAP, rosbag, and splat inputs before large-scale 3DGS map runs",
    )
    lsgd.add_argument("--root", default=".", help="Root directory to scan for map inputs")
    lsgd.add_argument(
        "--output",
        default=None,
        help="Optional discovery report JSON path",
    )
    lsgd.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Axes used in generated preflight commands",
    )
    lsgd.add_argument(
        "--tile-sizes",
        default="20,30,50",
        help="Comma-separated candidate tile sizes used in generated preflight commands",
    )
    lsgd.add_argument(
        "--target-images-per-chunk",
        type=int,
        default=48,
        help="Target images per chunk used in generated preflight commands",
    )
    lsgd.add_argument("--pilot-chunks", type=int, default=6, help="Pilot chunk count used in generated commands")
    lsgd.add_argument(
        "--route-start-image",
        type=int,
        default=0,
        help="Zero-based COLMAP image index used in generated pilot commands",
    )
    lsgd.add_argument("--max-depth", type=int, default=8, help="Maximum relative directory depth to scan")
    lsgd.add_argument("--max-results", type=int, default=20, help="Maximum results per discovered input kind")
    lsgd.add_argument(
        "--include-chunk-models",
        action="store_true",
        help="Include already materialized chunk sparse models in COLMAP discovery",
    )
    lsgd.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the discovery report",
    )

    lsgb = subparsers.add_parser(
        "large-scale-3dgs-bootstrap",
        help="Discover inputs and write the first large-scale 3DGS pilot plan when possible",
    )
    lsgb.add_argument("--root", default=".", help="Root directory to scan for map inputs")
    lsgb.add_argument(
        "--output",
        default=None,
        help="Optional output root for generated preflight, pilot, and plan files",
    )
    lsgb.add_argument("--report", default=None, help="Optional bootstrap report JSON path")
    lsgb.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Axes used for preflight and pilot planning",
    )
    lsgb.add_argument("--tile-sizes", default="20,30,50", help="Comma-separated candidate tile sizes")
    lsgb.add_argument("--overlap", type=float, default=5.0, help="Tile overlap in scene units/metres")
    lsgb.add_argument("--min-images", type=int, default=8, help="Minimum core images for a trainable tile")
    lsgb.add_argument(
        "--target-images-per-chunk",
        type=int,
        default=48,
        help="Preferred median core image count used to pick the recommended tile size",
    )
    lsgb.add_argument("--pilot-chunks", type=int, default=6, help="Ready route chunks to include in the pilot")
    lsgb.add_argument(
        "--route-start-image",
        type=int,
        default=0,
        help="Zero-based COLMAP image order index where pilot chunk selection starts",
    )
    lsgb.add_argument("--iterations", type=int, default=30000, help="Iterations for generated train commands")
    lsgb.add_argument(
        "--config",
        default="configs/training_ba.yaml",
        help="Training config path used in generated train commands",
    )
    lsgb.add_argument(
        "--write-plan",
        action="store_true",
        help="Also write the full large_scale_3dgs_plan.json next to the pilot plan",
    )
    lsgb.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How selected chunks place images/depth below the output root",
    )
    lsgb.add_argument("--max-depth", type=int, default=8, help="Maximum relative directory depth to scan")
    lsgb.add_argument("--max-results", type=int, default=20, help="Maximum discovery results per input kind")
    lsgb.add_argument(
        "--include-chunk-models",
        action="store_true",
        help="Include already materialized chunk sparse models in discovery",
    )
    lsgb.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the bootstrap report",
    )

    lsgs = subparsers.add_parser(
        "large-scale-3dgs-smoke-data",
        help="Generate a deterministic multi-tile COLMAP fixture for large-scale gsplat smoke runs",
    )
    lsgs.add_argument(
        "--output",
        default="outputs/large_scale_3dgs_smoke_data",
        help="Output directory for generated COLMAP sparse files and images",
    )
    lsgs.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xz",
        help="Axes used for the generated tile grid",
    )
    lsgs.add_argument("--grid-width", type=int, default=3, help="Number of tiles along the first axis")
    lsgs.add_argument("--grid-height", type=int, default=2, help="Number of tiles along the second axis")
    lsgs.add_argument("--tile-size", type=float, default=8.0, help="Nominal tile size in scene units")
    lsgs.add_argument("--images-per-tile", type=int, default=2, help="Generated camera images per tile")
    lsgs.add_argument("--points-per-tile", type=int, default=12, help="Generated sparse points per tile")
    lsgs.add_argument("--image-size", type=int, default=48, help="Generated square image size in pixels")
    lsgs.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the smoke data manifest",
    )

    lsgp = subparsers.add_parser(
        "large-scale-3dgs-preflight",
        help="Inspect a COLMAP scene and recommend tile settings before large-scale gsplat runs",
    )
    lsgp.add_argument("--data", required=True, help="Preprocessed COLMAP data directory")
    lsgp.add_argument(
        "--output",
        default="outputs/large_scale_3dgs",
        help="Output root used for the preflight report and suggested plan",
    )
    lsgp.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Horizontal axes used for candidate tiling",
    )
    lsgp.add_argument(
        "--tile-sizes",
        default="20,30,50",
        help="Comma-separated candidate tile sizes in scene units/metres",
    )
    lsgp.add_argument("--overlap", type=float, default=5.0, help="Tile overlap in scene units/metres")
    lsgp.add_argument("--min-images", type=int, default=8, help="Minimum core images for a trainable tile")
    lsgp.add_argument(
        "--target-images-per-chunk",
        type=int,
        default=48,
        help="Preferred median core image count used to pick the recommended tile size",
    )
    lsgp.add_argument("--iterations", type=int, default=30000, help="Iterations used in the suggested plan command")
    lsgp.add_argument(
        "--config",
        default="configs/training_ba.yaml",
        help="Training config path used in the suggested plan command",
    )
    lsgp.add_argument(
        "--write-plan",
        action="store_true",
        help="Also write large_scale_3dgs_plan.json using the recommended tile size",
    )
    lsgp.add_argument(
        "--write-pilot",
        action="store_true",
        help="Also write route-contiguous pilot report and plan using the recommended tile size",
    )
    lsgp.add_argument("--pilot-chunks", type=int, default=6, help="Ready route chunks to include in --write-pilot")
    lsgp.add_argument(
        "--route-start-image",
        type=int,
        default=0,
        help="Zero-based COLMAP image order index where --write-pilot chunk selection starts",
    )
    lsgp.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How --write-plan or --write-pilot materializes images/depth into per-tile directories",
    )
    lsgp.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the preflight report",
    )

    lsgpilot = subparsers.add_parser(
        "large-scale-3dgs-pilot",
        help="Build a route-contiguous pilot plan for a real large-scale COLMAP scene",
    )
    lsgpilot.add_argument("--data", required=True, help="Preprocessed COLMAP data directory")
    lsgpilot.add_argument(
        "--output",
        default="outputs/large_scale_3dgs_pilot",
        help="Output root for the pilot report, plan, and selected chunk data",
    )
    lsgpilot.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Horizontal axes used for tiling",
    )
    lsgpilot.add_argument("--tile-size", type=float, default=30.0, help="Tile width in scene units/metres")
    lsgpilot.add_argument("--overlap", type=float, default=5.0, help="Tile overlap in scene units/metres")
    lsgpilot.add_argument("--min-images", type=int, default=8, help="Minimum core images for a trainable tile")
    lsgpilot.add_argument("--pilot-chunks", type=int, default=6, help="Ready route chunks to include in the pilot")
    lsgpilot.add_argument(
        "--route-start-image",
        type=int,
        default=0,
        help="Zero-based COLMAP image order index where pilot chunk selection starts",
    )
    lsgpilot.add_argument(
        "--target-images-per-chunk",
        type=int,
        default=48,
        help="Preferred core image count recorded in the pilot report",
    )
    lsgpilot.add_argument("--iterations", type=int, default=30000, help="Iterations for generated train commands")
    lsgpilot.add_argument(
        "--config",
        default="configs/training_ba.yaml",
        help="Training config path used in generated train commands",
    )
    lsgpilot.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How selected chunks place images/depth below --output/chunks",
    )
    lsgpilot.add_argument("--export-max-points", type=int, default=400000, help="Max points per exported tile splat")
    lsgpilot.add_argument("--splat-min-opacity", type=float, default=0.02, help="Opacity filter for tile exports")
    lsgpilot.add_argument("--splat-max-scale", type=float, default=2.0, help="Max scale filter for tile exports")
    lsgpilot.add_argument(
        "--splat-max-scale-percentile",
        type=float,
        default=98.0,
        help="Adaptive max-scale percentile for tile exports",
    )
    lsgpilot.add_argument(
        "--format",
        choices=["text", "json", "shell"],
        default="text",
        help="Stdout format after writing the pilot report and plan",
    )

    # large-scale-3dgs-plan
    lsg = subparsers.add_parser(
        "large-scale-3dgs-plan",
        help="Plan tile/chunk training jobs for a large COLMAP scene before running gsplat",
    )
    lsg.add_argument("--data", required=True, help="Preprocessed COLMAP data directory")
    lsg.add_argument(
        "--output",
        default="outputs/large_scale_3dgs",
        help="Output root for the plan and optional chunk data",
    )
    lsg.add_argument("--tile-size", type=float, default=30.0, help="Tile width in scene units/metres")
    lsg.add_argument("--overlap", type=float, default=5.0, help="Tile overlap in scene units/metres")
    lsg.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Horizontal axes used for tiling (default: xy for ENU/COLMAP robotics data)",
    )
    lsg.add_argument("--min-images", type=int, default=8, help="Minimum core images for a trainable tile")
    lsg.add_argument("--iterations", type=int, default=30000, help="Iterations for generated train commands")
    lsg.add_argument(
        "--config",
        default="configs/training_ba.yaml",
        help="Training config path used in generated train commands",
    )
    lsg.add_argument("--export-max-points", type=int, default=400000, help="Max points per exported tile splat")
    lsg.add_argument("--splat-min-opacity", type=float, default=0.02, help="Opacity filter for tile exports")
    lsg.add_argument("--splat-max-scale", type=float, default=2.0, help="Max scale filter for tile exports")
    lsg.add_argument(
        "--splat-max-scale-percentile",
        type=float,
        default=98.0,
        help="Adaptive max-scale percentile for tile exports",
    )
    lsg.add_argument(
        "--materialize",
        action="store_true",
        help="Write per-tile COLMAP sparse files plus image/depth links under --output/chunks",
    )
    lsg.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How --materialize places images/depth into each tile directory",
    )
    lsg.add_argument(
        "--format",
        choices=["text", "json", "shell"],
        default="text",
        help="Stdout format after writing the plan JSON",
    )

    lsgr = subparsers.add_parser(
        "large-scale-3dgs-run",
        help="Run ready train/export chunks from a large-scale 3DGS plan with resume support",
    )
    lsgr.add_argument("--plan", required=True, help="Path to large_scale_3dgs_plan.json")
    lsgr.add_argument("--report", default=None, help="Optional run report JSON path")
    lsgr.add_argument("--max-chunks", type=int, default=None, help="Run at most N ready chunks")
    lsgr.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip chunks with existing splat outputs or point_cloud.ply",
    )
    lsgr.add_argument("--dry-run", action="store_true", help="Write a run report without executing commands")
    lsgr.add_argument("--no-fail-fast", action="store_true", help="Continue after a chunk command fails")
    lsgr.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the run report",
    )

    lsgc = subparsers.add_parser(
        "large-scale-3dgs-catalog",
        help="Build a web-facing tile catalog from a large-scale 3DGS plan and generated splats",
    )
    lsgc.add_argument("--plan", required=True, help="Path to large_scale_3dgs_plan.json")
    lsgc.add_argument("--output", default=None, help="Optional tile catalog JSON output path")
    lsgc.add_argument("--run-report", default=None, help="Optional large_scale_3dgs_run_report.json")
    lsgc.add_argument("--scene-id", default="large-scale-3dgs", help="Scene id used in catalog URLs")
    lsgc.add_argument("--label", default="Large-scale 3DGS", help="Human-readable catalog label")
    lsgc.add_argument(
        "--public-root",
        default=None,
        help="Optional web public root; generated splats are linked or copied below it",
    )
    lsgc.add_argument(
        "--public-url-prefix",
        default="/splats",
        help="URL prefix corresponding to --public-root (default: /splats)",
    )
    lsgc.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How existing splats are placed below --public-root",
    )
    lsgc.add_argument(
        "--require-splats",
        action="store_true",
        help="Only include tiles whose .splat output already exists",
    )
    lsgc.add_argument(
        "--web-app-dir",
        default="apps/dreamwalker-web",
        help="Dynamic Map Viewer web app directory used in the printed validation command",
    )
    lsgc.add_argument(
        "--site-url",
        default="http://localhost:5173/",
        help="Dynamic Map Viewer site URL used in the printed launch URL",
    )
    lsgc.add_argument(
        "--tile-preload",
        choices=["off", "metadata", "cache"],
        default="metadata",
        help="Dynamic map tile preload mode used in the printed launch URL",
    )
    lsgc.add_argument(
        "--route",
        default=None,
        help="Optional robot route JSON path or public URL for validation and launch URL output",
    )
    lsgc.add_argument(
        "--route-playback",
        action="store_true",
        help="Include robotRoutePlayback=1 in the printed launch URL and validation command",
    )
    lsgc.add_argument(
        "--route-playback-ms",
        type=int,
        default=None,
        help="Robot route playback interval in milliseconds for the printed launch URL",
    )
    lsgc.add_argument(
        "--route-playback-loop",
        action="store_true",
        help="Include robotRoutePlaybackLoop=1 in the printed launch URL and validation command",
    )
    lsgc.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the tile catalog",
    )

    lsgrt = subparsers.add_parser(
        "large-scale-3dgs-route",
        help="Build a Dynamic Map Viewer robot route from a large-scale 3DGS tile catalog",
    )
    lsgrt.add_argument("--catalog", required=True, help="Path to a large-scale 3DGS tile catalog JSON")
    lsgrt.add_argument("--output", default=None, help="Optional robot route JSON output path")
    lsgrt.add_argument("--label", default=None, help="Optional robot route label")
    lsgrt.add_argument("--description", default=None, help="Optional robot route description")
    lsgrt.add_argument("--fragment-id", default="residency", help="Dynamic Map Viewer fragment id for the route")
    lsgrt.add_argument("--fragment-label", default="Residency", help="Dynamic Map Viewer fragment label for the route")
    lsgrt.add_argument("--frame-id", default="dreamwalker_map", help="Route coordinate frame id")
    lsgrt.add_argument("--asset-label", default=None, help="Optional world asset label stored in the route")
    lsgrt.add_argument(
        "--zone-map-url",
        default="/manifests/robotics-residency.zones.json",
        help="Semantic zone map URL stored in the route world context",
    )
    lsgrt.add_argument("--world-splat-url", default="", help="Optional world splat URL stored in the route")
    lsgrt.add_argument("--collider-mesh-url", default="", help="Optional collider mesh URL stored in the route")
    lsgrt.add_argument("--default-y", type=float, default=0.0, help="Y coordinate when catalog axes do not include y")
    lsgrt.add_argument(
        "--order",
        choices=["spiral", "snake", "row-major"],
        default="spiral",
        help="Tile traversal order for indexed catalogs",
    )
    lsgrt.add_argument(
        "--include-missing-splats",
        action="store_true",
        help="Include tiles marked missing-splat in the route",
    )
    lsgrt.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the robot route",
    )

    lsgpr = subparsers.add_parser(
        "large-scale-3dgs-promote",
        help="Promote large-scale 3DGS run outputs into Dynamic Map Viewer public assets",
    )
    lsgpr.add_argument("--bootstrap", default=None, help="Optional large_scale_3dgs_bootstrap.json")
    lsgpr.add_argument("--plan", default=None, help="Optional large_scale_3dgs_plan.json or pilot plan JSON")
    lsgpr.add_argument("--run-report", default=None, help="Optional large_scale_3dgs_run_report.json")
    lsgpr.add_argument("--report", default=None, help="Optional promotion report JSON path")
    lsgpr.add_argument(
        "--public-root",
        default="apps/dreamwalker-web/public",
        help="Dynamic Map Viewer public root where catalog, route, and splats are staged",
    )
    lsgpr.add_argument("--catalog", default=None, help="Optional tile catalog output path")
    lsgpr.add_argument("--route", default=None, help="Optional robot route output path")
    lsgpr.add_argument("--scene-id", default="large-scale-3dgs", help="Scene id used in catalog and route filenames")
    lsgpr.add_argument("--label", default="Large-scale 3DGS", help="Human-readable catalog label")
    lsgpr.add_argument(
        "--public-url-prefix",
        default="/splats",
        help="URL prefix corresponding to staged splat files under --public-root",
    )
    lsgpr.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How existing splats are staged below --public-root",
    )
    lsgpr.add_argument(
        "--allow-missing-splats",
        action="store_true",
        help="Include tiles without generated .splat files in the catalog",
    )
    lsgpr.add_argument("--full-plan", action="store_true", help="When --bootstrap is used, prefer the full plan")
    lsgpr.add_argument("--no-route", action="store_true", help="Only write the tile catalog")
    lsgpr.add_argument(
        "--web-app-dir",
        default="apps/dreamwalker-web",
        help="Dynamic Map Viewer web app directory used in validation commands",
    )
    lsgpr.add_argument(
        "--site-url",
        default="http://localhost:5173/",
        help="Dynamic Map Viewer site URL used in the generated launch URL",
    )
    lsgpr.add_argument(
        "--tile-preload",
        choices=["off", "metadata", "cache"],
        default="metadata",
        help="Dynamic map tile preload mode used in the generated launch URL",
    )
    lsgpr.add_argument("--no-route-playback", action="store_true", help="Do not enable route playback in launch URLs")
    lsgpr.add_argument(
        "--route-playback-ms",
        type=int,
        default=1200,
        help="Robot route playback interval in milliseconds",
    )
    lsgpr.add_argument(
        "--no-route-playback-loop",
        action="store_true",
        help="Do not loop route playback in launch URLs",
    )
    lsgpr.add_argument("--route-label", default=None, help="Optional robot route label")
    lsgpr.add_argument("--route-description", default=None, help="Optional robot route description")
    lsgpr.add_argument("--fragment-id", default="residency", help="Dynamic Map Viewer fragment id for the route")
    lsgpr.add_argument("--fragment-label", default="Residency", help="Dynamic Map Viewer fragment label")
    lsgpr.add_argument("--frame-id", default="dreamwalker_map", help="Route coordinate frame id")
    lsgpr.add_argument("--asset-label", default=None, help="Optional world asset label stored in the route")
    lsgpr.add_argument(
        "--zone-map-url",
        default="/manifests/robotics-residency.zones.json",
        help="Semantic zone map URL stored in the route world context",
    )
    lsgpr.add_argument("--world-splat-url", default="", help="Optional world splat URL stored in the route")
    lsgpr.add_argument("--collider-mesh-url", default="", help="Optional collider mesh URL stored in the route")
    lsgpr.add_argument("--default-y", type=float, default=0.0, help="Y coordinate when catalog axes do not include y")
    lsgpr.add_argument(
        "--route-order",
        choices=["spiral", "snake", "row-major"],
        default="spiral",
        help="Tile traversal order for the generated route",
    )
    lsgpr.add_argument(
        "--include-missing-splats-in-route",
        action="store_true",
        help="Allow generated routes to traverse catalog tiles without .splat files",
    )
    lsgpr.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Stdout format after writing the promotion report",
    )

    # view
    vw = subparsers.add_parser("view", help="Launch the web viewer")
    vw.add_argument("--model", required=True, help="Path to the .ply file or COLMAP sparse dir")
    vw.add_argument("--host", default="0.0.0.0", help="Viewer host (default: 0.0.0.0)")
    vw.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    vw.add_argument("--colmap", action="store_true", help="View COLMAP sparse model instead of PLY")

    # export
    ex = subparsers.add_parser("export", help="Export PLY to web-friendly format")
    ex.add_argument("--model", required=True, help="Path to the .ply file")
    ex.add_argument(
        "--format",
        choices=["json", "binary", "scene-bundle", "splat"],
        default="json",
        help="Output format (default: json). 'splat' writes the antimatter15/splat 32-byte-per-gauss "
        "binary that docs/splat.html renders directly.",
    )
    ex.add_argument("--output", required=True, help="Output file path")
    ex.add_argument("--max-points", type=int, default=100000, help="Max points to export (default: 100000)")
    ex.add_argument(
        "--bundle-asset-format",
        choices=["json", "binary"],
        default="binary",
        help="Asset format used inside --format scene-bundle (default: binary)",
    )
    ex.add_argument("--scene-id", default=None, help="Optional scene id for --format scene-bundle")
    ex.add_argument("--label", default=None, help="Optional scene label for --format scene-bundle")
    ex.add_argument("--description", default="", help="Optional scene description for --format scene-bundle")
    ex.add_argument(
        "--splat-normalize-extent",
        type=float,
        default=None,
        help="For --format splat: rescale the scene so its max-axis extent equals this value (meters). "
        "17.0 matches docs/splat.html's default camera. Leave unset to keep original world scale.",
    )
    ex.add_argument(
        "--splat-min-opacity",
        type=float,
        default=0.0,
        help="For --format splat: drop gaussians below this sigmoid(opacity) threshold (default: 0, no filter).",
    )
    ex.add_argument(
        "--splat-max-scale",
        type=float,
        default=None,
        help="For --format splat: drop gaussians whose max exp(log_scale) exceeds this (meters).",
    )
    ex.add_argument(
        "--splat-max-scale-percentile",
        type=float,
        default=None,
        help=(
            "For --format splat: adaptively drop gaussians above this max-scale percentile. "
            "Useful for removing giant blurry splats from browser exports."
        ),
    )

    # photos-to-splat (one-shot: image dir -> DUSt3R -> gsplat -> .splat)
    p2s = subparsers.add_parser(
        "photos-to-splat",
        help="One-shot: a folder of JPG/PNG -> DUSt3R pose-free -> gsplat train -> .splat file",
    )
    p2s.add_argument("--images", required=True, help="Directory of input images (jpg/png)")
    _add_photos_to_splat_arguments(p2s)

    _register_video_to_splat_parser(subparsers)

    # splat-filter
    sf = subparsers.add_parser(
        "splat-filter",
        help="Clean an existing antimatter15 .splat by removing blurry low-opacity / oversized gaussians",
    )
    sf.add_argument("--input", required=True, help="Input .splat file")
    sf.add_argument("--output", required=True, help="Output .splat file")
    sf.add_argument("--min-opacity", type=float, default=0.0, help="Drop splats below this alpha/255 opacity")
    sf.add_argument("--max-scale", type=float, default=None, help="Drop splats whose max scale exceeds this value")
    sf.add_argument(
        "--max-scale-percentile",
        type=float,
        default=None,
        help="Drop splats above this adaptive max-scale percentile, e.g. 98",
    )
    sf.add_argument("--max-points", type=int, default=None, help="Cap output splat count after filtering")

    # splat-inspect
    si = subparsers.add_parser(
        "splat-inspect",
        help="Report opacity and scale stats for an existing antimatter15 .splat",
    )
    si.add_argument("--input", required=True, help="Input .splat file")
    si.add_argument(
        "--low-opacity-threshold",
        type=float,
        default=0.08,
        help="Opacity threshold used for the low-opacity summary",
    )
    si.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    stc = subparsers.add_parser(
        "splat-tile-catalog",
        help="Split an existing browser .splat into dynamic-map tile splats and a tile catalog",
    )
    stc.add_argument("--input", required=True, help="Input 32-byte-per-gaussian .splat file")
    stc.add_argument("--output", required=True, help="Output tile catalog JSON path")
    stc.add_argument(
        "--public-root",
        default="apps/dreamwalker-web/public",
        help="Web public root where tile .splat files are written",
    )
    stc.add_argument("--scene-id", default=None, help="Scene id for the generated tile catalog")
    stc.add_argument("--label", default=None, help="Human-readable catalog label")
    stc.add_argument("--tile-size", type=float, default=10.0, help="Tile size in .splat coordinate units")
    stc.add_argument("--overlap", type=float, default=2.0, help="Tile overlap in .splat coordinate units")
    stc.add_argument("--axes", choices=["xy", "xz", "yz"], default="xz", help="Axes used for tile splitting")
    stc.add_argument("--min-splats", type=int, default=1, help="Minimum core splats required to write a tile")
    stc.add_argument(
        "--public-url-prefix",
        default="/splats",
        help="URL prefix under --public-root for generated tile .splat files",
    )
    stc.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    # benchmark
    bm = subparsers.add_parser("benchmark", help="Benchmark training backends")
    bm.add_argument("--data", required=True, help="Data directory for training")
    bm.add_argument("--iterations", type=int, default=1000, help="Number of training iterations (default: 1000)")
    bm.add_argument("--output", default="outputs/benchmark", help="Benchmark output directory")
    bm.add_argument("--dataset-name", default="default", help="Dataset name label (default: default)")
    bm.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio", "both"],
        default="both",
        help="Backend to benchmark (default: both)",
    )
    bm.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat benchmark (not recommended)",
    )

    # run (full pipeline)
    rn = subparsers.add_parser("run", help="Run the full pipeline end-to-end")
    rn.add_argument("--images", required=True, help="Input image directory")
    rn.add_argument("--output", default="outputs", help="Root output directory")
    rn.add_argument(
        "--max-frames", type=int, default=100, help="Max frames to extract for dataset-specific preprocessors"
    )
    rn.add_argument(
        "--every-n", type=int, default=1, help="Extract every N-th frame for dataset-specific preprocessors"
    )
    rn.add_argument("--colmap-path", default="colmap", help="Path to the COLMAP executable (default: colmap)")
    rn.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy for COLMAP-based preprocessors (default: exhaustive)",
    )
    rn.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP-based preprocessing")
    rn.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    rn.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    rn.add_argument("--config", default=None, help="Path to training config YAML override")
    rn.add_argument(
        "--preprocess-method",
        choices=_PIPELINE_PREPROCESS_METHOD_CHOICES,
        default="colmap",
        help="Preprocessing method (default: colmap)",
    )
    rn.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera for --preprocess-method waymo (default: FRONT)",
    )
    rn.add_argument(
        "--extract-lidar-depth",
        action="store_true",
        help="For --preprocess-method waymo, also project TOP lidar into per-frame depth maps",
    )
    rn.add_argument(
        "--extract-dynamic-masks",
        action="store_true",
        help="For --preprocess-method waymo, also generate per-frame dynamic-object masks from camera labels",
    )
    rn.add_argument("--image-topic", default=None, help="ROS image topic for --preprocess-method mcd")
    rn.add_argument("--lidar-topic", default=None, help="ROS PointCloud2 topic for --preprocess-method mcd")
    rn.add_argument("--imu-topic", default=None, help="ROS IMU topic for --preprocess-method mcd")
    rn.add_argument(
        "--extract-lidar",
        action="store_true",
        help="For --preprocess-method mcd, also export PointCloud2 frames to lidar/*.npy",
    )
    rn.add_argument(
        "--extract-imu",
        action="store_true",
        help="For --preprocess-method mcd, also export IMU measurements to imu.csv",
    )
    rn.add_argument(
        "--gnss-topic",
        default=None,
        help="For --preprocess-method mcd, NavSatFix topic for --mcd-seed-poses-from-gnss (default: try /gnss/fix)",
    )
    rn.add_argument(
        "--mcd-flatten-gnss-altitude",
        action="store_true",
        help="For MCD GNSS seeding, project NavSatFix altitude to the median valid altitude before ENU conversion",
    )
    rn.add_argument(
        "--mcd-start-offset-sec",
        type=float,
        default=0.0,
        help="For MCD preprocessing, skip the first N seconds of image/LiDAR/GNSS streams",
    )
    rn.add_argument(
        "--mcd-seed-poses-from-gnss",
        action="store_true",
        help="For --preprocess-method mcd, COLMAP sparse from GNSS + images; single --image-topic only",
    )
    rn.add_argument(
        "--mcd-base-frame",
        default="base_link",
        help="For MCD GNSS seeding, parent frame for /tf_static lookup (default: base_link)",
    )
    rn.add_argument(
        "--mcd-static-calibration",
        default="",
        help="MCDVIRAL rig calibration YAML (body→sensor 4×4 T) when bags lack /tf_static",
    )
    rn.add_argument(
        "--mcd-camera-frame",
        default=None,
        help="For MCD GNSS seeding, camera frame id (default: CameraInfo header.frame_id)",
    )
    rn.add_argument(
        "--mcd-disable-tf-extrinsics",
        action="store_true",
        help="For MCD GNSS seeding, ignore /tf_static (GNSS-only trajectory at vehicle frame)",
    )
    rn.add_argument(
        "--mcd-include-tf-dynamic",
        action="store_true",
        help="For MCD GNSS seeding, merge /tf into TF map after /tf_static (slower on large bags)",
    )
    rn.add_argument(
        "--mcd-gnss-antenna-offset-enu",
        nargs=3,
        type=float,
        default=None,
        metavar=("E", "N", "U"),
        help="Subtract (East, North, Up) meters from each NavSat fix in ENU (approx. base vs antenna)",
    )
    rn.add_argument(
        "--mcd-gnss-antenna-offset-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Antenna in base_link (x forward, y left, z up, metres); heading from GNSS motion "
            "(do not combine with --mcd-gnss-antenna-offset-enu)"
        ),
    )
    rn.add_argument(
        "--mcd-tf-use-image-stamps",
        action="store_true",
        help="Multi-camera GNSS seed: resolve TF at each image time (/tf + /tf_static topology)",
    )
    rn.add_argument(
        "--mcd-lidar-frame",
        default="",
        help="For MCD GNSS seeding, LiDAR frame id under base_link (empty = identity T_base_lidar)",
    )
    rn.add_argument(
        "--mcd-skip-lidar-seed",
        action="store_true",
        help="For MCD GNSS seeding, skip merging LiDAR frames to world as points3D seed",
    )
    rn.add_argument(
        "--mcd-skip-lidar-colorize",
        action="store_true",
        help="Skip the image->LiDAR RGB projection that seeds points3D.txt with real colors",
    )
    rn.add_argument(
        "--mcd-export-depth",
        action="store_true",
        help="Project the world LiDAR cloud into each training image as sparse depth .npy",
    )
    rn.add_argument(
        "--mcd-reference-origin",
        default="",
        help="Share the ENU origin across bags. 'lat,lon,alt' in WGS84 degrees/metres.",
    )
    rn.add_argument(
        "--mcd-reference-bag",
        default="",
        help="Use the ENU origin recorded under <path>/pose/origin_wgs84.json from a previously preprocessed bag.",
    )
    rn.add_argument(
        "--mcd-imu-csv",
        default="",
        help="Path to an imu.csv with orientation_* columns. Interpolated into each TUM row as the base_link quaternion.",
    )
    rn.add_argument(
        "--mcd-skip-imu-orientation",
        action="store_true",
        help="Ignore any imu.csv and keep the default motion-inferred yaw.",
    )
    rn.add_argument("--trajectory", default=None, help="Trajectory file for --preprocess-method lidar-slam")
    rn.add_argument(
        "--trajectory-format",
        choices=["tum", "kitti", "nmea"],
        default="tum",
        help="Trajectory format for --preprocess-method lidar-slam (default: tum)",
    )
    rn.add_argument("--pointcloud", default=None, help="Point cloud file for --preprocess-method lidar-slam")
    rn.add_argument(
        "--nmea-time-offset-sec",
        type=float,
        default=0.0,
        help="Fixed seconds added to NMEA-derived timestamps for trajectory import.",
    )
    _add_external_slam_args(rn, context="run")
    rn.add_argument("--skip-preprocess", action="store_true", help="Skip COLMAP preprocessing")
    rn.add_argument("--no-viewer", action="store_true", help="Skip launching the viewer")
    rn.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    rn.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat training (not recommended)",
    )

    # demo (end-to-end: images -> splat -> Dynamic Map Viewer teleop)
    dm = subparsers.add_parser("demo", help="End-to-end demo: images -> 3DGS -> Dynamic Map Viewer robot teleop")
    dm.add_argument("--images", default=None, help="Input image directory or video file")
    dm.add_argument("--ply", default=None, help="Skip training, stage an existing PLY file directly")
    dm.add_argument("--output", default="outputs", help="Root output directory (default: outputs)")
    dm.add_argument(
        "--max-frames", type=int, default=100, help="Max frames to extract for dataset-specific preprocessors"
    )
    dm.add_argument(
        "--every-n", type=int, default=1, help="Extract every N-th frame for dataset-specific preprocessors"
    )
    dm.add_argument("--colmap-path", default="colmap", help="Path to the COLMAP executable (default: colmap)")
    dm.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy for COLMAP-based preprocessors (default: exhaustive)",
    )
    dm.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP-based preprocessing")
    dm.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    dm.add_argument("--iterations", type=int, default=1000, help="Training iterations (default: 1000)")
    dm.add_argument("--config", default=None, help="Path to training config YAML override")
    dm.add_argument(
        "--preprocess-method",
        choices=_PIPELINE_PREPROCESS_METHOD_CHOICES,
        default="colmap",
        help="Preprocessing method (default: colmap)",
    )
    dm.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera for --preprocess-method waymo (default: FRONT)",
    )
    dm.add_argument(
        "--extract-lidar-depth",
        action="store_true",
        help="For --preprocess-method waymo, also project TOP lidar into per-frame depth maps",
    )
    dm.add_argument(
        "--extract-dynamic-masks",
        action="store_true",
        help="For --preprocess-method waymo, also generate per-frame dynamic-object masks from camera labels",
    )
    dm.add_argument("--image-topic", default=None, help="ROS image topic for --preprocess-method mcd")
    dm.add_argument("--lidar-topic", default=None, help="ROS PointCloud2 topic for --preprocess-method mcd")
    dm.add_argument("--imu-topic", default=None, help="ROS IMU topic for --preprocess-method mcd")
    dm.add_argument(
        "--extract-lidar",
        action="store_true",
        help="For --preprocess-method mcd, also export PointCloud2 frames to lidar/*.npy",
    )
    dm.add_argument(
        "--extract-imu",
        action="store_true",
        help="For --preprocess-method mcd, also export IMU measurements to imu.csv",
    )
    dm.add_argument(
        "--gnss-topic",
        default=None,
        help="For --preprocess-method mcd, NavSatFix topic for --mcd-seed-poses-from-gnss (default: try /gnss/fix)",
    )
    dm.add_argument(
        "--mcd-flatten-gnss-altitude",
        action="store_true",
        help="For MCD GNSS seeding, project NavSatFix altitude to the median valid altitude before ENU conversion",
    )
    dm.add_argument(
        "--mcd-start-offset-sec",
        type=float,
        default=0.0,
        help="For MCD preprocessing, skip the first N seconds of image/LiDAR/GNSS streams",
    )
    dm.add_argument(
        "--mcd-seed-poses-from-gnss",
        action="store_true",
        help="For --preprocess-method mcd, COLMAP sparse from GNSS + images; single --image-topic only",
    )
    dm.add_argument(
        "--mcd-base-frame",
        default="base_link",
        help="For MCD GNSS seeding, parent frame for /tf_static lookup (default: base_link)",
    )
    dm.add_argument(
        "--mcd-static-calibration",
        default="",
        help="MCDVIRAL rig calibration YAML (body→sensor 4×4 T) when bags lack /tf_static",
    )
    dm.add_argument(
        "--mcd-camera-frame",
        default=None,
        help="For MCD GNSS seeding, camera frame id (default: CameraInfo header.frame_id)",
    )
    dm.add_argument(
        "--mcd-disable-tf-extrinsics",
        action="store_true",
        help="For MCD GNSS seeding, ignore /tf_static (GNSS-only trajectory at vehicle frame)",
    )
    dm.add_argument(
        "--mcd-include-tf-dynamic",
        action="store_true",
        help="For MCD GNSS seeding, merge /tf into TF map after /tf_static (slower on large bags)",
    )
    dm.add_argument(
        "--mcd-gnss-antenna-offset-enu",
        nargs=3,
        type=float,
        default=None,
        metavar=("E", "N", "U"),
        help="Subtract (East, North, Up) meters from each NavSat fix in ENU (approx. base vs antenna)",
    )
    dm.add_argument(
        "--mcd-gnss-antenna-offset-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Antenna in base_link (x forward, y left, z up, metres); heading from GNSS motion "
            "(do not combine with --mcd-gnss-antenna-offset-enu)"
        ),
    )
    dm.add_argument(
        "--mcd-tf-use-image-stamps",
        action="store_true",
        help="Multi-camera GNSS seed: resolve TF at each image time (/tf + /tf_static topology)",
    )
    dm.add_argument(
        "--mcd-lidar-frame",
        default="",
        help="For MCD GNSS seeding, LiDAR frame id under base_link (empty = identity T_base_lidar)",
    )
    dm.add_argument(
        "--mcd-skip-lidar-seed",
        action="store_true",
        help="For MCD GNSS seeding, skip merging LiDAR frames to world as points3D seed",
    )
    dm.add_argument(
        "--mcd-skip-lidar-colorize",
        action="store_true",
        help="Skip the image->LiDAR RGB projection that seeds points3D.txt with real colors",
    )
    dm.add_argument(
        "--mcd-export-depth",
        action="store_true",
        help="Project the world LiDAR cloud into each training image as sparse depth .npy",
    )
    dm.add_argument(
        "--mcd-reference-origin",
        default="",
        help="Share the ENU origin across bags. 'lat,lon,alt' in WGS84 degrees/metres.",
    )
    dm.add_argument(
        "--mcd-reference-bag",
        default="",
        help="Use the ENU origin recorded under <path>/pose/origin_wgs84.json from a previously preprocessed bag.",
    )
    dm.add_argument(
        "--mcd-imu-csv",
        default="",
        help="Path to an imu.csv with orientation_* columns. Interpolated into each TUM row as the base_link quaternion.",
    )
    dm.add_argument(
        "--mcd-skip-imu-orientation",
        action="store_true",
        help="Ignore any imu.csv and keep the default motion-inferred yaw.",
    )
    dm.add_argument("--trajectory", default=None, help="Trajectory file for --preprocess-method lidar-slam")
    dm.add_argument(
        "--trajectory-format",
        choices=["tum", "kitti", "nmea"],
        default="tum",
        help="Trajectory format for --preprocess-method lidar-slam (default: tum)",
    )
    dm.add_argument("--pointcloud", default=None, help="Point cloud file for --preprocess-method lidar-slam")
    dm.add_argument(
        "--nmea-time-offset-sec",
        type=float,
        default=0.0,
        help="Fixed seconds added to NMEA-derived timestamps for trajectory import.",
    )
    _add_external_slam_args(dm, context="demo")
    dm.add_argument("--fragment", default="residency", help="Dynamic Map Viewer fragment name (default: residency)")
    dm.add_argument("--no-launch", action="store_true", help="Skip launching the Vite dev server")
    dm.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat training (not recommended)",
    )

    # robotics ROS2 node
    rb = subparsers.add_parser("robotics-node", help="Launch the Dynamic Map Viewer ROS2 bridge node scaffold")
    rb.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    rb.add_argument("--node-name", default="dreamwalker_bridge_node", help="ROS2 node name")
    rb.add_argument("--frame-id", default="dreamwalker_map", help="Expected map frame id")
    rb.add_argument("--log-period", type=float, default=2.0, help="Summary log period in seconds")
    rb.add_argument("--zones-file", default=None, help="Optional semantic zone JSON file")
    rb.add_argument("--costmap-period", type=float, default=10.0, help="Costmap republish period in seconds")
    rb.add_argument("--request-state-on-start", action="store_true", help="Publish request_state on startup")
    rb.add_argument(
        "--enable-image-relay",
        action="store_true",
        help="Subscribe to camera relay topics and log received frames",
    )
    rb.add_argument(
        "--demo-teleop",
        choices=["forward", "backward", "turn-left", "turn-right"],
        default=None,
        help="Publish one teleop command on startup",
    )
    rb.add_argument(
        "--demo-camera",
        choices=["front", "chase", "top"],
        default=None,
        help="Publish one camera command on startup",
    )
    rb.add_argument(
        "--demo-waypoint",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Publish one waypoint command on startup",
    )
    rb.add_argument(
        "--demo-pose2d",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA_RAD"),
        default=None,
        help="Publish one Pose2D command on startup",
    )

    # headless PLY render server
    sr = subparsers.add_parser(
        "sim2real-server",
        help="Headless PLY renderer that publishes RGB + depth to Dynamic Map Viewer ROS2 topics",
    )
    sr.add_argument("--ply", required=True, help="Path to the trained PLY point cloud")
    sr.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    sr.add_argument("--node-name", default="dreamwalker_sim2real_server", help="ROS2 node name")
    sr.add_argument("--frame-id", default="dreamwalker_map", help="Camera frame id for published messages")
    sr.add_argument("--width", type=int, default=640, help="Render width in pixels")
    sr.add_argument("--height", type=int, default=480, help="Render height in pixels")
    sr.add_argument("--fps", type=float, default=5.0, help="Publish rate in Hz")
    sr.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    sr.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    sr.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    sr.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    sr.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for camera output")
    sr.add_argument(
        "--renderer",
        choices=["auto", "simple", "gsplat"],
        default="auto",
        help="Rasterization backend. auto uses gsplat only when CUDA and Gaussian PLY parameters are available",
    )
    sr.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of points to load from the PLY for rendering",
    )
    sr.add_argument(
        "--pose-source",
        choices=["static", "robot_pose_stamped", "robot_pose2d", "query"],
        default="robot_pose_stamped",
        help="Source of camera poses used for rendering",
    )
    sr.add_argument(
        "--static-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Static camera position used when --pose-source static",
    )
    sr.add_argument(
        "--static-orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=(0.0, 0.0, 0.0, 1.0),
        help="Static camera orientation quaternion used when --pose-source static",
    )
    sr.add_argument(
        "--pose2d-z",
        type=float,
        default=0.0,
        help="Z position to use when pose source is robot_pose2d",
    )
    sr.add_argument(
        "--query-transport",
        choices=["auto", "none", "zmq", "ws"],
        default="none",
        help="Optional request-response transport for ad-hoc render queries",
    )
    sr.add_argument(
        "--query-endpoint",
        default="tcp://127.0.0.1:5588",
        help=(
            "Bind endpoint for the query transport when enabled. "
            "Defaults: tcp://127.0.0.1:5588 for zmq, ws://127.0.0.1:8781/sim2real for ws"
        ),
    )
    sr.add_argument(
        "--query-poll-period",
        type=float,
        default=0.01,
        help="Polling period in seconds for the query transport",
    )
    sr.add_argument("--run-once", action="store_true", help="Publish one frame and exit")

    # sim2real query client
    sq = subparsers.add_parser(
        "sim2real-query",
        help="Send one pose-based render query to a sim2real headless render server",
    )
    sq.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5588",
        help="Query endpoint to connect to. Supports tcp://... (ZMQ) and ws://... (WebSocket)",
    )
    sq.add_argument("--timeout-ms", type=int, default=10000, help="Request timeout in milliseconds")
    sq.add_argument(
        "--position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="World-space camera position",
    )
    orientation = sq.add_mutually_exclusive_group()
    orientation.add_argument(
        "--orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=None,
        help="World-space camera orientation quaternion",
    )
    orientation.add_argument(
        "--yaw-degrees",
        type=float,
        default=None,
        help="Planar yaw in degrees, converted to a quaternion around +Z",
    )
    sq.add_argument("--width", type=int, default=640, help="Render width in pixels")
    sq.add_argument("--height", type=int, default=480, help="Render height in pixels")
    sq.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    sq.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    sq.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    sq.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    sq.add_argument("--jpeg-out", default=None, help="Optional path for the returned JPEG frame")
    sq.add_argument("--depth-out", default=None, help="Optional path for the returned depth .npy file")
    sq.add_argument("--camera-info-out", default=None, help="Optional path for the returned cameraInfo JSON")
    sq.add_argument("--response-out", default=None, help="Optional path for the full raw response JSON")

    # sim2real localization image benchmark
    sb = subparsers.add_parser(
        "sim2real-benchmark-images",
        help="Render estimate poses through sim2real-server and compare RGB frames against captured ground truth",
    )
    sb.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5588",
        help="Query endpoint to connect to. Supports tcp://... (ZMQ) and ws://... (WebSocket)",
    )
    sb.add_argument(
        "--run",
        default=None,
        help="Optional localization-run-snapshot JSON exported from the web panel",
    )
    sb.add_argument(
        "--ground-truth",
        default=None,
        help="Path to a route-capture-bundle JSON file. Required unless --run already embeds it",
    )
    sb.add_argument(
        "--estimate",
        default=None,
        help="Path to a localization estimate JSON or TUM/ORB-SLAM text trajectory. Required unless --run embeds it",
    )
    sb.add_argument(
        "--alignment",
        choices=["auto", "index", "timestamp"],
        default="auto",
        help="Pose matching mode before rendering estimate frames",
    )
    sb.add_argument(
        "--metrics",
        nargs="+",
        choices=["psnr", "ssim", "lpips"],
        default=["psnr", "ssim", "lpips"],
        help="Image metrics to compute for each matched frame",
    )
    sb.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone used when --metrics includes lpips",
    )
    sb.add_argument(
        "--device",
        default="cpu",
        help="Torch device for LPIPS. Use auto to prefer CUDA when available",
    )
    sb.add_argument("--timeout-ms", type=int, default=10000, help="Per-frame render timeout in milliseconds")
    sb.add_argument("--max-frames", type=int, default=None, help="Optional cap on the number of matched frames")
    sb.add_argument("--output", default=None, help="Optional path for the full benchmark report JSON")

    # route policy benchmark
    rpb = subparsers.add_parser(
        "route-policy-benchmark",
        help="Fit or evaluate route policy imitation baselines in the Physical AI simulator",
    )
    source_group = rpb.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--transitions-jsonl", default=None, help="Replay transition JSONL used to fit imitation")
    source_group.add_argument("--dataset-json", default=None, help="Replay episode dataset JSON used to fit imitation")
    source_group.add_argument("--model", default=None, help="Previously saved imitation model JSON")
    source_group.add_argument("--policy-registry", default=None, help="Policy registry JSON with named policies")
    rpb.add_argument("--model-output", default=None, help="Optional path to write the fitted imitation model JSON")
    rpb.add_argument(
        "--output", default="outputs/route_policy_benchmark/report.json", help="Benchmark report JSON path"
    )
    rpb.add_argument("--markdown-output", default=None, help="Optional Markdown summary output path")
    rpb.add_argument("--scene-catalog", default="docs/scenes-list.json", help="Scene picker catalog JSON")
    rpb.add_argument(
        "--site-url", default="https://rsasaki0109.github.io/gs-mapper/", help="Base site URL for scene assets"
    )
    rpb.add_argument(
        "--scene-id", default=None, help="Scene id to evaluate (default: outdoor-demo or first catalog scene)"
    )
    rpb.add_argument("--benchmark-id", default="route-policy-benchmark", help="Benchmark/evaluation id")
    rpb.add_argument("--policy-name", default="imitation", help="Policy name for the fitted or loaded imitation model")
    rpb.add_argument("--episode-count", type=int, default=16, help="Number of evaluation episodes")
    rpb.add_argument("--seed-start", type=int, default=100, help="First evaluation seed")
    rpb.add_argument("--max-steps", type=int, default=None, help="Override route policy max steps")
    rpb.add_argument(
        "--goal",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Fixed goal position; repeat for a goal suite",
    )
    rpb.add_argument("--goal-suite", default=None, help="Named route policy goal-suite JSON")
    rpb.add_argument("--neighbor-count", type=int, default=1, help="k for fitted k-NN imitation")
    rpb.add_argument("--action-keys", nargs="+", default=None, help="Pinned replay action keys for target decoding")
    rpb.add_argument("--include-direct-baseline", action="store_true", help="Compare against a direct-goal baseline")
    rpb.add_argument("--min-success-rate", type=float, default=None, help="Optional quality threshold")
    rpb.add_argument("--max-collision-rate", type=float, default=None, help="Optional quality threshold")
    rpb.add_argument("--max-truncation-rate", type=float, default=None, help="Optional quality threshold")
    rpb.add_argument("--min-episode-count", type=int, default=None, help="Optional quality threshold")
    rpb.add_argument("--min-transition-count", type=int, default=None, help="Optional quality threshold")

    # route policy benchmark history
    rph = subparsers.add_parser(
        "route-policy-benchmark-history",
        help="Aggregate route policy benchmark reports and apply regression gates",
    )
    rph.add_argument("--report", action="append", required=True, help="Benchmark report JSON; repeat in trend order")
    rph.add_argument("--baseline-report", default=None, help="Blessed baseline report JSON for regression gates")
    rph.add_argument("--history-id", default="route-policy-benchmark-history", help="Benchmark history id")
    rph.add_argument(
        "--output", default="outputs/route_policy_benchmark/history.json", help="Benchmark history JSON path"
    )
    rph.add_argument("--markdown-output", default=None, help="Optional Markdown summary output path")
    rph.add_argument(
        "--max-success-rate-drop",
        type=float,
        default=0.0,
        help="Allowed success-rate drop from the baseline for each matching policy",
    )
    rph.add_argument(
        "--max-collision-rate-increase",
        type=float,
        default=0.0,
        help="Allowed collision-rate increase from the baseline for each matching policy",
    )
    rph.add_argument(
        "--max-truncation-rate-increase",
        type=float,
        default=0.0,
        help="Allowed truncation-rate increase from the baseline for each matching policy",
    )
    rph.add_argument(
        "--max-mean-reward-drop",
        type=float,
        default=None,
        help="Optional allowed mean-reward drop from the baseline for each matching policy",
    )
    rph.add_argument(
        "--allow-missing-policies",
        action="store_true",
        help="Do not fail the regression gate when a baseline policy is absent from a current report",
    )
    rph.add_argument(
        "--allow-report-failures",
        action="store_true",
        help="Do not fail the history gate when an input benchmark report itself failed",
    )
    rph.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 2 when the history regression gate fails",
    )

    # route policy scenario set
    rps = subparsers.add_parser(
        "route-policy-scenario-set",
        help="Run one policy registry across a versioned route policy scenario set",
    )
    rps.add_argument("--scenario-set", required=True, help="Route policy scenario-set JSON")
    rps.add_argument(
        "--policy-registry",
        default=None,
        help="Policy registry JSON override; defaults to policyRegistryPath in the scenario set",
    )
    rps.add_argument(
        "--report-dir",
        default="outputs/route_policy_scenarios/reports",
        help="Directory for per-scenario benchmark reports",
    )
    rps.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_set_run.json",
        help="Scenario-set run report JSON path",
    )
    rps.add_argument("--markdown-output", default=None, help="Optional scenario-set Markdown summary path")
    rps.add_argument(
        "--history-output",
        default="outputs/route_policy_scenarios/history.json",
        help="Benchmark history JSON path for generated scenario reports",
    )
    rps.add_argument("--history-markdown-output", default=None, help="Optional benchmark history Markdown path")
    rps.add_argument("--baseline-report", default=None, help="Blessed baseline report JSON for history gates")
    rps.add_argument("--max-success-rate-drop", type=float, default=0.0, help="Allowed baseline success-rate drop")
    rps.add_argument(
        "--max-collision-rate-increase",
        type=float,
        default=0.0,
        help="Allowed baseline collision-rate increase",
    )
    rps.add_argument(
        "--max-truncation-rate-increase",
        type=float,
        default=0.0,
        help="Allowed baseline truncation-rate increase",
    )
    rps.add_argument(
        "--max-mean-reward-drop",
        type=float,
        default=None,
        help="Optional allowed baseline mean-reward drop",
    )
    rps.add_argument(
        "--allow-missing-policies",
        action="store_true",
        help="Do not fail the history gate when a baseline policy is absent from a scenario report",
    )
    rps.add_argument(
        "--allow-report-failures",
        action="store_true",
        help="Do not fail the history gate when a scenario benchmark report itself failed",
    )
    rps.add_argument("--no-markdown", action="store_true", help="Skip per-scenario Markdown benchmark summaries")
    rps.add_argument(
        "--correlation-report",
        action="append",
        default=None,
        help=(
            "Pre-computed real-vs-sim correlation report JSON to attach to the scenario-set run "
            "report (can be passed multiple times)"
        ),
    )
    rps.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 2 when a scenario or history regression gate fails",
    )

    # route policy scenario matrix
    rpm = subparsers.add_parser(
        "route-policy-scenario-matrix",
        help="Expand a compact route policy scenario matrix into scenario-set JSON files",
    )
    rpm.add_argument("--matrix", required=True, help="Route policy scenario-matrix JSON")
    rpm.add_argument(
        "--output-dir",
        default="outputs/route_policy_scenarios/generated",
        help="Directory for generated scenario-set JSON files",
    )
    rpm.add_argument(
        "--index-output",
        default="outputs/route_policy_scenarios/scenario_matrix_expansion.json",
        help="Scenario-matrix expansion index JSON path",
    )
    rpm.add_argument("--markdown-output", default=None, help="Optional scenario-matrix Markdown summary path")

    # route policy scenario shards
    rpsh = subparsers.add_parser(
        "route-policy-scenario-shards",
        help="Split generated route policy scenario sets into CI-sized shard JSON files",
    )
    rpsh.add_argument("--expansion", required=True, help="Route policy scenario-matrix expansion JSON")
    rpsh.add_argument(
        "--max-scenarios-per-shard",
        type=int,
        default=4,
        help="Maximum scenarios to include in each generated shard scenario-set",
    )
    rpsh.add_argument("--shard-plan-id", default=None, help="Optional shard plan id")
    rpsh.add_argument(
        "--output-dir",
        default="outputs/route_policy_scenarios/shards",
        help="Directory for generated shard scenario-set JSON files",
    )
    rpsh.add_argument(
        "--index-output",
        default="outputs/route_policy_scenarios/scenario_shard_plan.json",
        help="Scenario shard plan JSON path",
    )
    rpsh.add_argument("--markdown-output", default=None, help="Optional scenario shard plan Markdown path")

    # route policy scenario shard merge
    rpshm = subparsers.add_parser(
        "route-policy-scenario-shard-merge",
        help="Merge independently executed route policy scenario shard runs",
    )
    rpshm.add_argument("--run", action="append", required=True, help="Scenario-set shard run JSON; repeat per shard")
    rpshm.add_argument("--merge-id", default="route-policy-scenario-shard-merge", help="Shard merge id")
    rpshm.add_argument("--baseline-report", default=None, help="Blessed baseline report JSON for history gates")
    rpshm.add_argument(
        "--history-output",
        default="outputs/route_policy_scenarios/shard_history.json",
        help="Merged benchmark history JSON path",
    )
    rpshm.add_argument("--history-markdown-output", default=None, help="Optional merged history Markdown path")
    rpshm.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_shard_merge.json",
        help="Scenario shard merge JSON path",
    )
    rpshm.add_argument("--markdown-output", default=None, help="Optional scenario shard merge Markdown path")
    rpshm.add_argument("--max-success-rate-drop", type=float, default=0.0, help="Allowed baseline success-rate drop")
    rpshm.add_argument(
        "--max-collision-rate-increase",
        type=float,
        default=0.0,
        help="Allowed baseline collision-rate increase",
    )
    rpshm.add_argument(
        "--max-truncation-rate-increase",
        type=float,
        default=0.0,
        help="Allowed baseline truncation-rate increase",
    )
    rpshm.add_argument(
        "--max-mean-reward-drop",
        type=float,
        default=None,
        help="Optional allowed baseline mean-reward drop",
    )
    rpshm.add_argument(
        "--allow-missing-policies",
        action="store_true",
        help="Do not fail the merged history gate when a baseline policy is absent from a shard report",
    )
    rpshm.add_argument(
        "--allow-report-failures",
        action="store_true",
        help="Do not fail the merged history gate when a shard benchmark report itself failed",
    )
    rpshm.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 2 when a shard or merged history regression gate fails",
    )

    # route policy scenario CI manifest
    rpsci = subparsers.add_parser(
        "route-policy-scenario-ci-manifest",
        help="Generate a CI matrix manifest from a route policy scenario shard plan",
    )
    rpsci.add_argument("--shard-plan", required=True, help="Route policy scenario shard plan JSON")
    rpsci.add_argument("--manifest-id", default=None, help="Optional CI manifest id")
    rpsci.add_argument(
        "--report-dir",
        default="outputs/route_policy_scenarios/shard_reports",
        help="Base directory for per-shard benchmark reports",
    )
    rpsci.add_argument(
        "--run-output-dir",
        default="outputs/route_policy_scenarios/shard_runs",
        help="Base directory for per-shard run JSON outputs",
    )
    rpsci.add_argument(
        "--history-output-dir",
        default="outputs/route_policy_scenarios/shard_histories",
        help="Base directory for per-shard history JSON outputs",
    )
    rpsci.add_argument("--merge-id", default="route-policy-scenario-shard-merge", help="Shard merge id")
    rpsci.add_argument(
        "--merge-output",
        default="outputs/route_policy_scenarios/scenario_shard_merge.json",
        help="Merged shard report JSON path",
    )
    rpsci.add_argument(
        "--merge-history-output",
        default="outputs/route_policy_scenarios/shard_history.json",
        help="Merged benchmark history JSON path",
    )
    rpsci.add_argument("--merge-markdown-output", default=None, help="Optional merged shard report Markdown path")
    rpsci.add_argument(
        "--merge-history-markdown-output",
        default=None,
        help="Optional merged benchmark history Markdown path",
    )
    rpsci.add_argument("--cache-key-prefix", default="route-policy-scenario", help="Prefix for CI cache keys")
    rpsci.add_argument("--include-markdown", action="store_true", help="Include per-shard Markdown output paths")
    rpsci.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Include --fail-on-regression in generated shard and merge commands",
    )
    rpsci.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_ci_manifest.json",
        help="Scenario CI manifest JSON path",
    )
    rpsci.add_argument("--markdown-output", default=None, help="Optional scenario CI manifest Markdown path")

    # route policy scenario CI workflow
    rpswf = subparsers.add_parser(
        "route-policy-scenario-ci-workflow",
        help="Materialize a GitHub Actions workflow from a route policy scenario CI manifest",
    )
    rpswf.add_argument("--manifest", required=True, help="Route policy scenario CI manifest JSON")
    rpswf.add_argument("--workflow-id", default="route-policy-scenario-shards", help="Workflow materialization id")
    rpswf.add_argument("--workflow-name", default="Route Policy Scenario Shards", help="GitHub Actions workflow name")
    rpswf.add_argument("--runs-on", default="ubuntu-latest", help="GitHub Actions runner label")
    rpswf.add_argument("--python-version", default="3.11", help="Python version used by generated jobs")
    rpswf.add_argument("--install-command", default='pip install -e ".[dev]"', help="Dependency install shell command")
    rpswf.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact root path to upload/download; defaults to the common shard output root",
    )
    rpswf.add_argument("--artifact-retention-days", type=int, default=7, help="Shard artifact retention in days")
    rpswf.add_argument(
        "--no-workflow-dispatch",
        action="store_true",
        help="Do not include the workflow_dispatch trigger",
    )
    rpswf.add_argument("--push-branch", action="append", default=None, help="Add a push trigger branch")
    rpswf.add_argument(
        "--pull-request-branch",
        action="append",
        default=None,
        help="Add a pull_request trigger branch",
    )
    rpswf.add_argument("--fail-fast", action="store_true", help="Enable strategy fail-fast for shard jobs")
    rpswf.add_argument(
        "--workflow-output",
        default="outputs/route_policy_scenarios/scenario_ci_workflow.yml",
        help="Generated GitHub Actions workflow YAML path",
    )
    rpswf.add_argument(
        "--index-output",
        default="outputs/route_policy_scenarios/scenario_ci_workflow.json",
        help="Workflow materialization metadata JSON path",
    )
    rpswf.add_argument("--markdown-output", default=None, help="Optional workflow materialization Markdown path")

    # route policy scenario CI workflow validation
    rpswfv = subparsers.add_parser(
        "route-policy-scenario-ci-workflow-validate",
        help="Validate a generated route policy scenario CI workflow against its manifest",
    )
    rpswfv.add_argument("--manifest", required=True, help="Route policy scenario CI manifest JSON")
    rpswfv.add_argument("--workflow-index", required=True, help="Workflow materialization metadata JSON")
    rpswfv.add_argument("--workflow", default=None, help="Generated GitHub Actions workflow YAML to validate")
    rpswfv.add_argument("--validation-id", default=None, help="Optional validation report id")
    rpswfv.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_ci_workflow_validation.json",
        help="Workflow validation report JSON path",
    )
    rpswfv.add_argument("--markdown-output", default=None, help="Optional workflow validation Markdown path")
    rpswfv.add_argument(
        "--fail-on-validation",
        action="store_true",
        help="Exit with status 2 when workflow validation fails",
    )

    # route policy scenario CI workflow activation
    rpswfa = subparsers.add_parser(
        "route-policy-scenario-ci-workflow-activate",
        help="Activate a generated route policy scenario CI workflow after validation passes",
    )
    rpswfa.add_argument("--workflow-index", required=True, help="Workflow materialization metadata JSON")
    rpswfa.add_argument("--validation-report", required=True, help="Workflow validation report JSON")
    rpswfa.add_argument("--workflow", default=None, help="Generated GitHub Actions workflow YAML to activate")
    rpswfa.add_argument(
        "--active-workflow-output",
        required=True,
        help="Active GitHub Actions workflow path under .github/workflows",
    )
    rpswfa.add_argument("--activation-id", default=None, help="Optional activation report id")
    rpswfa.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_ci_workflow_activation.json",
        help="Workflow activation report JSON path",
    )
    rpswfa.add_argument("--markdown-output", default=None, help="Optional workflow activation Markdown path")
    rpswfa.add_argument("--overwrite", action="store_true", help="Overwrite an existing active workflow file")
    rpswfa.add_argument(
        "--fail-on-activation",
        action="store_true",
        help="Exit with status 2 when workflow activation is blocked",
    )

    # route policy scenario CI review artifact
    rpsrev = subparsers.add_parser(
        "route-policy-scenario-ci-review",
        help="Publish a review artifact for route policy scenario CI workflow changes",
    )
    rpsrev.add_argument("--shard-merge", required=True, help="Scenario shard merge report JSON")
    rpsrev.add_argument("--validation-report", required=True, help="Workflow validation report JSON")
    rpsrev.add_argument("--activation-report", required=True, help="Workflow activation report JSON")
    rpsrev.add_argument("--review-id", default=None, help="Optional CI review artifact id")
    rpsrev.add_argument("--pages-base-url", default=None, help="Optional Pages base URL stored in review metadata")
    rpsrev.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_ci_review.json",
        help="Scenario CI review JSON path",
    )
    rpsrev.add_argument("--markdown-output", default=None, help="Optional scenario CI review Markdown path")
    rpsrev.add_argument("--html-output", default=None, help="Optional scenario CI review HTML path")
    rpsrev.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional directory that receives review.json, review.md, and index.html",
    )
    rpsrev.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Exit with status 2 when the scenario CI review does not pass",
    )
    rpsrev.add_argument(
        "--no-correlation-reports",
        action="store_true",
        help=(
            "Skip embedding real-vs-sim correlation reports gathered from each shard's run JSON "
            "(default: any correlation reports attached via gs-mapper route-policy-scenario-set "
            "--correlation-report flow into the review artifact)"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-translation-mean-meters",
        type=float,
        default=None,
        help=(
            "Optional regression gate: fail the review when any embedded correlation report's "
            "translation_error_mean_meters exceeds this bound"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-translation-p95-meters",
        type=float,
        default=None,
        help=(
            "Optional regression gate: fail the review when any embedded correlation report's "
            "translation_error_p95_meters exceeds this bound"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-translation-max-meters",
        type=float,
        default=None,
        help=(
            "Optional regression gate: fail the review when any embedded correlation report's "
            "translation_error_max_meters exceeds this bound"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-heading-mean-radians",
        type=float,
        default=None,
        help=(
            "Optional regression gate: fail the review when any embedded correlation report's "
            "heading_error_mean_radians (when present) exceeds this bound"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-pair-translation-meters",
        type=float,
        default=None,
        help=(
            "Per-pair distribution gate (paired with --max-correlation-pair-fraction): the "
            "translation_error_meters above which a CorrelatedPosePair counts as exceeding"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-pair-fraction",
        type=float,
        default=None,
        help=(
            "Per-pair distribution gate (paired with --max-correlation-pair-translation-meters): "
            "fail the review when more than this fraction of pairs in any embedded correlation "
            "report exceed the per-pair translation bound (0.05 = 5%%)"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-pair-heading-radians",
        type=float,
        default=None,
        help=(
            "Per-pair heading distribution gate (paired with --max-correlation-heading-pair-fraction): "
            "the heading_error_radians above which a CorrelatedPosePair counts as exceeding"
        ),
    )
    rpsrev.add_argument(
        "--max-correlation-heading-pair-fraction",
        type=float,
        default=None,
        help=(
            "Per-pair heading distribution gate (paired with --max-correlation-pair-heading-radians): "
            "fail the review when more than this fraction of pairs with heading data in any embedded "
            "correlation report exceed the per-pair heading bound (0.05 = 5%%)"
        ),
    )
    rpsrev.add_argument(
        "--correlation-pair-distribution-strata",
        type=int,
        default=None,
        help=(
            "Optional time stratification: split each correlation report's pair list into N windows "
            "and run the per-pair distribution + aggregate-stat gates against each window independently "
            "(failure tag includes the window index)"
        ),
    )
    rpsrev.add_argument(
        "--correlation-pair-distribution-strata-mode",
        choices=("equal-duration", "equal-pair-count", "event-aligned"),
        default="equal-duration",
        help=(
            "Stratification mode for --correlation-pair-distribution-strata: 'equal-duration' splits "
            "by bag_timestamp_seconds (default), 'equal-pair-count' splits by pair index so windows hold "
            "near-equal pair counts even when bag sample density is uneven, 'event-aligned' buckets "
            "pairs by externally-supplied scenario event windows (see --correlation-event-windows)"
        ),
    )
    rpsrev.add_argument(
        "--correlation-event-windows",
        default=None,
        help=(
            "Path to a gs-mapper-correlation-event-windows/v1 JSON file. Required when "
            "--correlation-pair-distribution-strata-mode=event-aligned. The file lists "
            "{name, startTime, endTime, tags, source} windows; pairs whose bag timestamp "
            "falls outside every window are dropped. Missing files or zero-window files "
            "fall back to equal-pair-count and the fallback is recorded in the review "
            "bundle metadata (correlationStratificationFallbacks)."
        ),
    )
    rpsrev.add_argument(
        "--correlation-thresholds-config",
        default=None,
        help=(
            "Optional JSON file with per-bag-topic correlation threshold overrides (shape: "
            "{<bag_source_topic>: {<thresholds>}}). Topics matched here use the override; "
            "other topics fall through to the scalar --max-correlation-* defaults."
        ),
    )
    rpsrev.add_argument(
        "--kind",
        choices=("synthetic", "production"),
        default=None,
        help=(
            "Mark the review as a synthetic smoke fixture or a production benchmark run. "
            "Setting this populates the first-class provenance block on the review artifact "
            "and is required when other --scene-id / --policy-version / etc. flags are set. "
            "Omit to keep backwards-compatible v1 JSON without a provenance block."
        ),
    )
    rpsrev.add_argument(
        "--scene-id",
        default=None,
        help="Provenance: scene id (e.g. matches docs/scenes-list.json entry)",
    )
    rpsrev.add_argument(
        "--scenario-set-id",
        default=None,
        help="Provenance: scenario set / shard plan id",
    )
    rpsrev.add_argument(
        "--matrix-hash",
        default=None,
        help="Provenance: sha256 (or similar) hash of the scenario matrix expansion JSON",
    )
    rpsrev.add_argument(
        "--policy-version",
        default=None,
        help="Provenance: route / imitation policy version (git tag or semver)",
    )
    rpsrev.add_argument(
        "--env-contract-version",
        default=None,
        help="Provenance: env contract version (e.g. gs_sim2real.sim.contract module hash or git tag)",
    )
    rpsrev.add_argument(
        "--correlation-threshold-profile",
        default=None,
        help="Provenance: name of the correlation threshold profile applied (separate from the threshold object itself)",
    )
    rpsrev.add_argument(
        "--asset-source",
        default=None,
        help="Provenance: scene asset source (e.g. bag6, mcd-ntu-day02)",
    )
    rpsrev.add_argument(
        "--git-commit",
        default=None,
        help=(
            "Provenance: explicit git commit. When omitted, the CLI best-effort resolves it from "
            "'git rev-parse HEAD' in the current working directory; leave unset and run outside a "
            "repo to keep the field absent."
        ),
    )
    rpsrev.add_argument(
        "--generated-at",
        default=None,
        help="Provenance: explicit ISO 8601 timestamp. When omitted, the CLI uses datetime.now(UTC).",
    )
    rpsrev.add_argument(
        "--provenance-extra",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Provenance: add a key=value pair to the provenance.extra mapping. Repeat the flag "
            "to add multiple keys (e.g. --provenance-extra runTrigger=nightly --provenance-extra ci=github)"
        ),
    )
    rpsrev.add_argument(
        "--adoption-report",
        default=None,
        help="Optional scenario CI workflow adoption report JSON to embed in the review",
    )
    rpsrev.add_argument(
        "--manual-workflow",
        default=None,
        help="Optional manual-only workflow YAML path (defaults to activation report's active path)",
    )
    rpsrev.add_argument(
        "--adopted-workflow",
        default=None,
        help="Optional adopted workflow YAML path (defaults to adoption report's adopted active path)",
    )

    # route policy dataset -> policy trace events (offline extractor)
    rpd2t = subparsers.add_parser(
        "route-policy-dataset-to-trace",
        help="Extract policy trace events (goal_reached / collision / near_miss / truncated) from a route policy dataset JSON",
    )
    rpd2t.add_argument("--dataset", required=True, help="route policy dataset JSON path")
    rpd2t.add_argument("--output", required=True, help="policy trace JSONL output path")
    rpd2t.add_argument(
        "--segment-duration-seconds",
        type=float,
        default=1.0,
        help="Per-step duration used to synthesize event timestamps (default: 1.0)",
    )
    rpd2t.add_argument(
        "--near-miss-clearance-meters",
        type=float,
        default=None,
        help=(
            "When set, also emit a near_miss event per transition whose nearest obstacle "
            "clearance feature is below this threshold"
        ),
    )
    rpd2t.add_argument(
        "--near-miss-feature-key",
        default="nearest-dynamic-obstacle-clearance-meters",
        help=(
            "next_observation feature key inspected for the --near-miss-clearance-meters gate "
            "(default: nearest-dynamic-obstacle-clearance-meters)"
        ),
    )
    rpd2t.add_argument(
        "--time-offset-seconds",
        type=float,
        default=0.0,
        help="Offset added to every synthesized timestamp (default: 0.0)",
    )

    # policy trace -> correlation event windows
    rpt2ew = subparsers.add_parser(
        "route-policy-trace-to-event-windows",
        help="Convert a policy trace JSONL into a gs-mapper-correlation-event-windows/v1 JSON file",
    )
    rpt2ew.add_argument("--trace", required=True, help="Policy trace JSONL path")
    rpt2ew.add_argument("--output", required=True, help="Correlation event windows JSON output path")
    rpt2ew.add_argument(
        "--half-width-seconds",
        type=float,
        default=0.5,
        help=("Half-width of the synthesized window around each point event (window = [t - hw, t + hw]; default: 0.5)"),
    )
    rpt2ew.add_argument(
        "--time-offset-seconds",
        type=float,
        default=0.0,
        help=(
            "Offset added to timestamp_seconds when no bag_timestamp_seconds is set on the event "
            "(use this to align a sim-time trace into bag-time)"
        ),
    )
    rpt2ew.add_argument(
        "--name-template",
        default="{event_name}-{episode_id}-{step_index}",
        help=(
            "Window name template; format keys: {event_name}, {episode_id}, {episode_index}, {step_index} "
            "(default: '{event_name}-{episode_id}-{step_index}')"
        ),
    )

    # route policy scenario CI workflow trigger promotion
    rpswfp = subparsers.add_parser(
        "route-policy-scenario-ci-workflow-promote",
        help="Gate promotion of scenario CI workflow triggers after review passes",
    )
    rpswfp.add_argument("--review", required=True, help="Scenario CI review JSON")
    rpswfp.add_argument("--review-url", default=None, help="Published review URL attached to the promotion gate")
    rpswfp.add_argument("--promotion-id", default=None, help="Optional workflow promotion report id")
    rpswfp.add_argument(
        "--trigger-mode",
        choices=("pull-request", "push", "push-and-pull-request"),
        default="pull-request",
        help="Repository trigger mode to promote",
    )
    rpswfp.add_argument("--push-branch", action="append", default=None, help="Add a literal push trigger branch")
    rpswfp.add_argument(
        "--pull-request-branch",
        action="append",
        default=None,
        help="Add a literal pull_request trigger branch",
    )
    rpswfp.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_ci_workflow_promotion.json",
        help="Workflow promotion report JSON path",
    )
    rpswfp.add_argument("--markdown-output", default=None, help="Optional workflow promotion Markdown path")
    rpswfp.add_argument(
        "--fail-on-promotion",
        action="store_true",
        help="Exit with status 2 when workflow trigger promotion is blocked",
    )

    # route policy scenario CI workflow trigger adoption
    rpswfad = subparsers.add_parser(
        "route-policy-scenario-ci-workflow-adopt",
        help="Re-materialize and activate a trigger-enabled workflow after promotion passes",
    )
    rpswfad.add_argument("--manifest", required=True, help="Scenario CI manifest JSON")
    rpswfad.add_argument(
        "--workflow-index",
        required=True,
        help="Manual-only workflow materialization metadata JSON",
    )
    rpswfad.add_argument("--promotion", required=True, help="Workflow promotion report JSON")
    rpswfad.add_argument(
        "--adopted-workflow-output",
        required=True,
        help="Generated trigger-enabled workflow YAML path (staged source)",
    )
    rpswfad.add_argument(
        "--adopted-active-workflow-output",
        required=True,
        help="Adopted active GitHub Actions workflow path under .github/workflows",
    )
    rpswfad.add_argument("--adoption-id", default=None, help="Optional adoption report id")
    rpswfad.add_argument(
        "--output",
        default="outputs/route_policy_scenarios/scenario_ci_workflow_adoption.json",
        help="Workflow adoption report JSON path",
    )
    rpswfad.add_argument("--markdown-output", default=None, help="Optional workflow adoption Markdown path")
    rpswfad.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing adopted active workflow file",
    )
    rpswfad.add_argument(
        "--fail-on-adoption",
        action="store_true",
        help="Exit with status 2 when workflow trigger adoption is blocked",
    )

    # experiment labs — specs drive a nested `experiment` subparser plus
    # hidden top-level aliases for back-compat.
    experiment_specs: list[tuple[str, str, str]] = [
        (
            "localization-alignment",
            "experiment-localization-alignment",
            "Compare multiple localization alignment strategies and optionally refresh experiment docs",
        ),
        (
            "render-backend-selection",
            "experiment-render-backend-selection",
            "Compare render backend-selection policies and optionally refresh experiment docs",
        ),
        (
            "outdoor-training-features",
            "experiment-outdoor-training-features",
            "Compare outdoor depth/appearance/pose/sky training feature bundles",
        ),
        (
            "localization-import",
            "experiment-localization-import",
            "Compare localization estimate import policies and optionally refresh experiment docs",
        ),
        (
            "query-transport-selection",
            "experiment-query-transport-selection",
            "Compare query transport policies and optionally refresh experiment docs",
        ),
        (
            "query-request-import",
            "experiment-query-request-import",
            "Compare query request import policies and optionally refresh experiment docs",
        ),
        (
            "live-localization-stream-import",
            "experiment-live-localization-stream-import",
            "Compare live localization stream import policies and optionally refresh experiment docs",
        ),
        (
            "route-capture-import",
            "experiment-route-capture-import",
            "Compare route capture bundle import policies and optionally refresh experiment docs",
        ),
        (
            "sim2real-websocket-protocol",
            "experiment-sim2real-websocket-protocol",
            "Compare sim2real websocket message protocol policies and optionally refresh experiment docs",
        ),
        (
            "localization-review-bundle-import",
            "experiment-localization-review-bundle-import",
            "Compare localization review bundle import policies and optionally refresh experiment docs",
        ),
        (
            "query-cancellation-policy",
            "experiment-query-cancellation-policy",
            "Compare query cancellation policies and optionally refresh experiment docs",
        ),
        (
            "query-coalescing-policy",
            "experiment-query-coalescing-policy",
            "Compare query dedupe/coalescing policies and optionally refresh experiment docs",
        ),
        (
            "query-error-mapping",
            "experiment-query-error-mapping",
            "Compare query error mapping policies and optionally refresh experiment docs",
        ),
        (
            "query-queue-policy",
            "experiment-query-queue-policy",
            "Compare query queue policies and optionally refresh experiment docs",
        ),
        (
            "query-source-identity",
            "experiment-query-source-identity",
            "Compare query source identity policies and optionally refresh experiment docs",
        ),
        (
            "query-timeout-policy",
            "experiment-query-timeout-policy",
            "Compare query timeout policies and optionally refresh experiment docs",
        ),
        (
            "query-response-build",
            "experiment-query-response-build",
            "Compare query response build policies and optionally refresh experiment docs",
        ),
    ]

    def _add_experiment_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repetitions", type=int, default=200, help="Runtime benchmark repetitions per fixture")
        p.add_argument(
            "--write-docs",
            action="store_true",
            help="Refresh docs/experiments.md, docs/experiments.generated.md, docs/decisions.md, and docs/interfaces.md",
        )
        p.add_argument(
            "--docs-dir",
            default="docs",
            help="Directory where experiment process docs are written when --write-docs is set",
        )
        p.add_argument("--output", default=None, help="Optional path for the full experiment report JSON")

    exp_parent = subparsers.add_parser(
        "experiment",
        help="Experiment labs: A/B strategies and refresh docs/experiments.md",
    )
    exp_sub = exp_parent.add_subparsers(dest="experiment_command", metavar="<lab>", help="Available experiment labs")

    for short, _legacy, help_text in experiment_specs:
        nested = exp_sub.add_parser(short, help=help_text)
        _add_experiment_flags(nested)
    # Back-compat for the flat `experiment-foo` aliases is handled via
    # argv rewriting in `main()` so the main --help stays focused on core tools.

    return parser


LEGACY_EXPERIMENT_ALIASES: dict[str, tuple[str, str]] = {
    "experiment-localization-alignment": ("experiment", "localization-alignment"),
    "experiment-render-backend-selection": ("experiment", "render-backend-selection"),
    "experiment-outdoor-training-features": ("experiment", "outdoor-training-features"),
    "experiment-localization-import": ("experiment", "localization-import"),
    "experiment-query-transport-selection": ("experiment", "query-transport-selection"),
    "experiment-query-request-import": ("experiment", "query-request-import"),
    "experiment-live-localization-stream-import": ("experiment", "live-localization-stream-import"),
    "experiment-route-capture-import": ("experiment", "route-capture-import"),
    "experiment-sim2real-websocket-protocol": ("experiment", "sim2real-websocket-protocol"),
    "experiment-localization-review-bundle-import": ("experiment", "localization-review-bundle-import"),
    "experiment-query-cancellation-policy": ("experiment", "query-cancellation-policy"),
    "experiment-query-coalescing-policy": ("experiment", "query-coalescing-policy"),
    "experiment-query-error-mapping": ("experiment", "query-error-mapping"),
    "experiment-query-queue-policy": ("experiment", "query-queue-policy"),
    "experiment-query-source-identity": ("experiment", "query-source-identity"),
    "experiment-query-timeout-policy": ("experiment", "query-timeout-policy"),
    "experiment-query-response-build": ("experiment", "query-response-build"),
}


def _rewrite_legacy_experiment_argv(argv: list[str]) -> list[str]:
    """Rewrite `gs-mapper experiment-foo ...` -> `gs-mapper experiment foo ...`.

    Keeps old scripts + the READMEs from PR #67 working while the main
    --help listing stays focused on core tools.
    """
    if not argv:
        return argv
    legacy = argv[0]
    mapped = LEGACY_EXPERIMENT_ALIASES.get(legacy)
    if mapped is None:
        return argv
    import warnings

    warnings.warn(
        f"`gs-mapper {legacy}` is deprecated; use `gs-mapper {mapped[0]} {mapped[1]}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return [mapped[0], mapped[1], *argv[1:]]


def cmd_download(args: argparse.Namespace) -> None:
    """Handle the download subcommand."""
    if args.sample_images:
        from gs_sim2real.common.download import download_sample_images

        output_dir = Path(args.output) if args.output else Path("data/sample")
        download_sample_images(output_dir)
        return

    if args.dataset is None:
        print("Error: --dataset is required (unless using --sample-images).")
        sys.exit(1)

    from gs_sim2real.common.download import download_dataset

    output_dir = Path(args.output) if args.output else None
    download_dataset(
        name=args.dataset,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Handle the preprocess subcommand."""
    images_path = Path(args.images)
    output_dir = Path(args.output)

    if args.method == "waymo":
        from gs_sim2real.datasets.waymo import WaymoLoader

        loader = WaymoLoader(data_dir=str(images_path))
        images_out = loader.extract_frames(
            output_dir=str(output_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        # Convert to COLMAP format if camera_params.json exists
        params_path = output_dir / "camera_params.json"
        if params_path.exists():
            sparse_dir = loader.to_colmap_format(
                camera_params_path=str(params_path),
                output_dir=str(output_dir),
            )
            print(f"Waymo frames extracted to: {images_out}")
            print(f"COLMAP sparse model at: {sparse_dir}")
        else:
            print(f"Waymo frames loaded from: {images_out}")
        if args.extract_lidar_depth:
            depth_dir = loader.extract_lidar_depth(
                output_dir=str(output_dir),
                camera=args.camera,
                max_frames=args.max_frames,
                every_n=args.every_n,
            )
            print(f"Waymo LiDAR depth extracted to: {depth_dir}")
        if args.extract_dynamic_masks:
            masks_dir = loader.extract_dynamic_masks(
                output_dir=str(output_dir),
                camera=args.camera,
                max_frames=args.max_frames,
                every_n=args.every_n,
            )
            print(f"Waymo dynamic masks extracted to: {masks_dir}")
        return

    if args.method == "mcd":
        if args.list_topics:
            from gs_sim2real.datasets.mcd import MCDLoader

            loader = MCDLoader(data_dir=str(images_path))
            topics = loader.list_topics()
            if not topics:
                print(f"No rosbag topics found under: {images_path}")
                return
            print("MCD rosbag topics:")
            for topic in topics:
                preferred = " default" if topic["is_preferred_default"] else ""
                print(f"  [{topic['role']}] {topic['topic']} ({topic['msgtype']}, {topic['msgcount']} msgs){preferred}")
            return
        _run_mcd_preprocess_to_colmap(images_path, Path(output_dir), args, run_colmap=False)
        return

    if args.method == "frames":
        from gs_sim2real.preprocess.extract_frames import (
            extract_frames,
            extract_frames_from_dir,
        )

        if images_path.is_file():
            extract_frames(
                video_path=images_path,
                output_dir=output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        elif images_path.is_dir():
            extract_frames_from_dir(
                input_dir=images_path,
                output_dir=output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        else:
            print(f"Error: '{images_path}' is not a file or directory.")
            sys.exit(1)
    elif args.method == "lidar-slam":
        sparse_dir = _run_lidar_slam_preprocess_to_colmap(images_path, Path(output_dir), args)
        print(f"LiDAR SLAM import complete: {sparse_dir}")
    elif args.method == "external-slam":
        sparse_dir = _run_external_slam_preprocess_to_colmap(images_path, Path(output_dir), args)
        if sparse_dir is None:
            if getattr(args, "external_slam_manifest_format", "text") != "json":
                print("External SLAM dry run complete.")
        else:
            print(f"External SLAM import complete: {sparse_dir}")
    elif args.method in _POSE_FREE_METHOD_MAP:
        from gs_sim2real.preprocess.pose_free import run_pose_free

        run_pose_free(
            image_dir=images_path,
            output_dir=output_dir,
            **_pose_free_kwargs_from_args(args),
        )
    else:
        from gs_sim2real.preprocess.colmap import run_colmap

        run_colmap(
            image_dir=images_path,
            output_dir=output_dir,
            matching=args.matching,
            use_gpu=not args.no_gpu,
            colmap_path=args.colmap_path,
        )


def _preflight_gsplat_train_data(data_dir: Path, skip: bool) -> None:
    """Fail fast if COLMAP sparse model is missing or incomplete (gsplat)."""
    if skip:
        return
    from gs_sim2real.preprocess.colmap_ready import require_colmap_sparse_model

    require_colmap_sparse_model(data_dir)


def _resolve_photos_to_splat_quality(args: argparse.Namespace) -> argparse.Namespace:
    """Apply photos-to-splat quality presets while preserving explicit overrides."""
    values = vars(args).copy()
    preset_name = values.get("quality", "draft")
    preset = _PHOTOS_TO_SPLAT_QUALITY_PRESETS[preset_name]
    for key, value in preset.items():
        if values.get(key) == _PHOTOS_TO_SPLAT_DEFAULTS[key]:
            values[key] = value
    return argparse.Namespace(**values)


_POSE_FREE_METHOD_MAP = {
    "pose-free": "dust3r",
    "dust3r": "dust3r",
    "mast3r": "mast3r",
    "vggt": "vggt",
    "simple": "simple",
}


def _pose_free_kwargs_from_args(args: argparse.Namespace, *, method: str | None = None) -> dict:
    """Build PoseFreeProcessor kwargs from CLI flags."""
    cli_method = method or getattr(args, "method", None) or getattr(args, "preprocess", "dust3r")
    resolved = _POSE_FREE_METHOD_MAP.get(cli_method, cli_method)
    kwargs: dict = {
        "method": resolved,
        "num_frames": getattr(args, "num_frames", 30),
        "scene_graph": getattr(args, "scene_graph", "complete"),
        "align_iters": getattr(args, "align_iters", 300),
        "mast3r_subsample": getattr(args, "mast3r_subsample", 8),
    }
    if resolved == "mast3r":
        if getattr(args, "mast3r_checkpoint", None):
            kwargs["checkpoint"] = Path(args.mast3r_checkpoint)
        if getattr(args, "mast3r_root", None):
            kwargs["mast3r_root"] = Path(args.mast3r_root)
    elif resolved == "vggt":
        if getattr(args, "vggt_checkpoint", None):
            kwargs["checkpoint"] = args.vggt_checkpoint
        if getattr(args, "vggt_root", None):
            kwargs["vggt_root"] = Path(args.vggt_root)
    else:
        if getattr(args, "dust3r_checkpoint", None):
            kwargs["checkpoint"] = Path(args.dust3r_checkpoint)
        if getattr(args, "dust3r_root", None):
            kwargs["dust3r_root"] = Path(args.dust3r_root)
    return kwargs


def _find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root that contains ``docs/splat.html``."""
    candidate = (start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "docs" / "splat.html").is_file():
            return path
    return candidate


def _open_local_splat_viewer(splat_path: Path, *, port: int = 8000) -> None:
    """Serve the repo over HTTP and open ``docs/splat.html`` for a local ``.splat`` file."""
    import os
    import threading
    import webbrowser
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import quote

    repo_root = _find_repo_root()
    splat_path = splat_path.resolve()
    docs_dir = repo_root / "docs"
    rel_splat = os.path.relpath(splat_path, docs_dir).replace(os.sep, "/")

    class RepoHTTPRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(repo_root), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), RepoHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    viewer_url = f"http://127.0.0.1:{port}/docs/splat.html?url={quote(rel_splat, safe='/')}"
    print(f"\nLocal viewer: {viewer_url}")
    print("Leave this process running while you inspect the map in the browser.")
    webbrowser.open(viewer_url)


def cmd_video_to_splat(args: argparse.Namespace) -> None:
    """Handle the video-to-splat / map subcommands.

    One-shot pipeline: video file -> frame extraction -> pose-free sparse -> gsplat training
    -> antimatter15 ``.splat`` binary, optionally opened in the local docs viewer.
    """
    from gs_sim2real.preprocess.extract_frames import VIDEO_EXTENSIONS, extract_frames, plan_video_frame_sampling

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        sys.exit(2)

    suffix = video_path.suffix.lower()
    if suffix and suffix not in VIDEO_EXTENSIONS:
        print(f"Warning: {suffix} is not a known video extension; trying OpenCV anyway.")

    output_dir = Path(args.output) if args.output else Path("outputs") / f"{video_path.stem}_splat"
    frames_dir = output_dir / video_path.stem
    sampling_target = args.num_frames if args.num_frames > 0 else 32
    plan = plan_video_frame_sampling(video_path, target_frames=sampling_target)

    print("=" * 60)
    print(f"Step 1/4: Extract frames from {video_path.name}")
    print("=" * 60)
    print(
        f"Sampling plan: every_n={plan.every_n}, max_frames={plan.max_frames}, "
        f"video_frames={plan.total_frames}, fps={plan.video_fps:.1f}"
    )
    extracted = extract_frames(
        video_path,
        frames_dir,
        every_n=plan.every_n,
        max_frames=plan.max_frames,
    )
    if not extracted:
        print("Error: no frames were extracted from the video.")
        sys.exit(2)

    photos_args = argparse.Namespace(**vars(args))
    photos_args.images = str(frames_dir)
    photos_args.output = str(output_dir)
    if photos_args.num_frames <= 0:
        photos_args.num_frames = len(extracted)
    else:
        photos_args.num_frames = min(photos_args.num_frames, len(extracted))

    cmd_photos_to_splat(photos_args)

    splat_path = output_dir / f"{video_path.stem}.splat"
    if getattr(args, "open_viewer", False):
        if splat_path.is_file():
            _open_local_splat_viewer(splat_path, port=args.viewer_port)
        else:
            print(f"Warning: expected splat at {splat_path} but file is missing; skipping viewer open.")


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the train subcommand."""
    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Load config override if provided
    config = None
    if args.config:
        from gs_sim2real.common.config import load_config

        config = load_config(args.config)

    if args.method == "gsplat":
        from gs_sim2real.train.gsplat_trainer import train_gsplat

        _preflight_gsplat_train_data(data_dir, getattr(args, "skip_data_check", False))
        ply_path = train_gsplat(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
            num_iterations=args.iterations,
        )
        print(f"\nTrained model saved to: {ply_path}")
    else:
        from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

        output = train_nerfstudio(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
        )
        print(f"\nNerfstudio output at: {output}")


def cmd_large_scale_3dgs_smoke_data(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-smoke-data subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSSmokeDataOptions,
        format_large_scale_3dgs_smoke_data_text,
        write_large_scale_3dgs_smoke_data,
    )

    manifest = write_large_scale_3dgs_smoke_data(
        LargeScale3DGSSmokeDataOptions(
            output_dir=Path(args.output),
            axes=args.axes,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            tile_size=args.tile_size,
            images_per_tile=args.images_per_tile,
            points_per_tile=args.points_per_tile,
            image_size=args.image_size,
        )
    )

    if args.format == "json":
        print(json.dumps(manifest, indent=2))
    else:
        print(format_large_scale_3dgs_smoke_data_text(manifest))


def cmd_large_scale_3dgs_discover(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-discover subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSDiscoveryOptions,
        build_large_scale_3dgs_discovery,
        format_large_scale_3dgs_discovery_text,
        parse_large_scale_3dgs_tile_sizes,
        write_large_scale_3dgs_discovery,
    )

    options = LargeScale3DGSDiscoveryOptions(
        root_dir=Path(args.root),
        output_path=Path(args.output) if args.output else None,
        axes=args.axes,
        tile_sizes=parse_large_scale_3dgs_tile_sizes(args.tile_sizes),
        target_images_per_chunk=args.target_images_per_chunk,
        pilot_chunks=args.pilot_chunks,
        route_start_image=args.route_start_image,
        max_depth=args.max_depth,
        max_results=args.max_results,
        include_chunk_models=args.include_chunk_models,
    )
    report = build_large_scale_3dgs_discovery(options)
    report_path = write_large_scale_3dgs_discovery(report, options.output_path)

    if args.format == "json":
        print(json.dumps({**report, "reportPath": str(report_path)}, indent=2))
    else:
        print(format_large_scale_3dgs_discovery_text(report, report_path))


def cmd_large_scale_3dgs_bootstrap(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-bootstrap subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSBootstrapOptions,
        build_large_scale_3dgs_bootstrap,
        format_large_scale_3dgs_bootstrap_text,
        parse_large_scale_3dgs_tile_sizes,
        write_large_scale_3dgs_bootstrap,
    )

    options = LargeScale3DGSBootstrapOptions(
        root_dir=Path(args.root),
        output_dir=Path(args.output) if args.output else None,
        report_path=Path(args.report) if args.report else None,
        axes=args.axes,
        tile_sizes=parse_large_scale_3dgs_tile_sizes(args.tile_sizes),
        overlap=args.overlap,
        min_images=args.min_images,
        target_images_per_chunk=args.target_images_per_chunk,
        pilot_chunks=args.pilot_chunks,
        route_start_image=args.route_start_image,
        iterations=args.iterations,
        config=args.config,
        write_plan=args.write_plan,
        link_mode=args.link_mode,
        max_depth=args.max_depth,
        max_results=args.max_results,
        include_chunk_models=args.include_chunk_models,
    )
    report = build_large_scale_3dgs_bootstrap(options)
    report_path = write_large_scale_3dgs_bootstrap(report, options.report_path)

    if args.format == "json":
        print(json.dumps({**report, "reportPath": str(report_path)}, indent=2))
    else:
        print(format_large_scale_3dgs_bootstrap_text(report, report_path))


def cmd_large_scale_3dgs_preflight(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-preflight subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSPreflightOptions,
        build_large_scale_3dgs_preflight,
        format_large_scale_3dgs_preflight_text,
        parse_large_scale_3dgs_tile_sizes,
        write_large_scale_3dgs_preflight,
    )

    options = LargeScale3DGSPreflightOptions(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        axes=args.axes,
        tile_sizes=parse_large_scale_3dgs_tile_sizes(args.tile_sizes),
        overlap=args.overlap,
        min_images=args.min_images,
        target_images_per_chunk=args.target_images_per_chunk,
        iterations=args.iterations,
        config=args.config,
        write_plan=args.write_plan,
        write_pilot=args.write_pilot,
        pilot_chunks=args.pilot_chunks,
        route_start_image=args.route_start_image,
        link_mode=args.link_mode,
    )
    report = build_large_scale_3dgs_preflight(options)
    report_path = write_large_scale_3dgs_preflight(report, Path(args.output))

    if args.format == "json":
        print(json.dumps({**report, "reportPath": str(report_path)}, indent=2))
    else:
        print(format_large_scale_3dgs_preflight_text(report, report_path))


def cmd_large_scale_3dgs_pilot(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-pilot subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSPilotOptions,
        build_large_scale_3dgs_pilot,
        format_large_scale_3dgs_pilot_text,
        format_large_scale_3dgs_shell,
        write_large_scale_3dgs_pilot,
    )

    options = LargeScale3DGSPilotOptions(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        axes=args.axes,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_images=args.min_images,
        pilot_chunks=args.pilot_chunks,
        route_start_image=args.route_start_image,
        target_images_per_chunk=args.target_images_per_chunk,
        iterations=args.iterations,
        config=args.config,
        link_mode=args.link_mode,
        export_max_points=args.export_max_points,
        splat_min_opacity=args.splat_min_opacity,
        splat_max_scale=args.splat_max_scale,
        splat_max_scale_percentile=args.splat_max_scale_percentile,
    )
    report = build_large_scale_3dgs_pilot(options)
    report_path, plan_path = write_large_scale_3dgs_pilot(report, Path(args.output))

    if args.format == "json":
        printable = {key: value for key, value in report.items() if key != "plan"}
        print(json.dumps({**printable, "reportPath": str(report_path), "planPath": str(plan_path)}, indent=2))
    elif args.format == "shell":
        print(format_large_scale_3dgs_shell(report["plan"]), end="")
    else:
        print(format_large_scale_3dgs_pilot_text(report, report_path, plan_path))


def cmd_large_scale_3dgs_plan(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-plan subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSOptions,
        build_large_scale_3dgs_plan,
        format_large_scale_3dgs_shell,
        format_large_scale_3dgs_text,
        write_large_scale_3dgs_plan,
    )

    options = LargeScale3DGSOptions(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        tile_size=args.tile_size,
        overlap=args.overlap,
        axes=args.axes,
        min_images=args.min_images,
        iterations=args.iterations,
        config=args.config,
        export_max_points=args.export_max_points,
        splat_min_opacity=args.splat_min_opacity,
        splat_max_scale=args.splat_max_scale,
        splat_max_scale_percentile=args.splat_max_scale_percentile,
        materialize=args.materialize,
        link_mode=args.link_mode,
    )
    plan = build_large_scale_3dgs_plan(options)
    plan_path = write_large_scale_3dgs_plan(plan, Path(args.output))

    if args.format == "json":
        print(json.dumps({**plan, "planPath": str(plan_path)}, indent=2))
    elif args.format == "shell":
        print(format_large_scale_3dgs_shell(plan), end="")
    else:
        print(format_large_scale_3dgs_text(plan, plan_path))


def cmd_large_scale_3dgs_run(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-run subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSRunOptions,
        format_large_scale_3dgs_run_text,
        run_large_scale_3dgs_plan,
    )

    report = run_large_scale_3dgs_plan(
        LargeScale3DGSRunOptions(
            plan_path=Path(args.plan),
            report_path=Path(args.report) if args.report else None,
            max_chunks=args.max_chunks,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            fail_fast=not args.no_fail_fast,
        )
    )

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(format_large_scale_3dgs_run_text(report))


def cmd_large_scale_3dgs_catalog(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-catalog subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSCatalogOptions,
        build_large_scale_3dgs_catalog,
        build_large_scale_3dgs_web_runbook,
        format_large_scale_3dgs_catalog_text,
        write_large_scale_3dgs_catalog,
    )

    options = LargeScale3DGSCatalogOptions(
        plan_path=Path(args.plan),
        output_path=Path(args.output) if args.output else None,
        run_report_path=Path(args.run_report) if args.run_report else None,
        scene_id=args.scene_id,
        label=args.label,
        public_root=Path(args.public_root) if args.public_root else None,
        public_url_prefix=args.public_url_prefix,
        link_mode=args.link_mode,
        require_splats=args.require_splats,
        web_app_dir=Path(args.web_app_dir) if args.web_app_dir else None,
        site_url=args.site_url,
        tile_preload=args.tile_preload,
        route_path=args.route,
        route_playback=args.route_playback,
        route_playback_ms=args.route_playback_ms,
        route_playback_loop=args.route_playback_loop,
    )
    catalog = build_large_scale_3dgs_catalog(options)
    catalog_path = write_large_scale_3dgs_catalog(catalog, options)

    if args.format == "json":
        print(
            json.dumps(
                {
                    **catalog,
                    "catalogPath": str(catalog_path),
                    "webRunbook": build_large_scale_3dgs_web_runbook(catalog_path, options),
                },
                indent=2,
            )
        )
    else:
        print(format_large_scale_3dgs_catalog_text(catalog, catalog_path, options))


def cmd_large_scale_3dgs_route(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-route subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSRouteOptions,
        build_large_scale_3dgs_route,
        format_large_scale_3dgs_route_text,
        write_large_scale_3dgs_route,
    )

    options = LargeScale3DGSRouteOptions(
        catalog_path=Path(args.catalog),
        output_path=Path(args.output) if args.output else None,
        label=args.label,
        description=args.description,
        fragment_id=args.fragment_id,
        fragment_label=args.fragment_label,
        frame_id=args.frame_id,
        asset_label=args.asset_label,
        zone_map_url=args.zone_map_url,
        world_splat_url=args.world_splat_url,
        collider_mesh_url=args.collider_mesh_url,
        default_y=args.default_y,
        order=args.order,
        include_missing_splats=args.include_missing_splats,
    )
    route = build_large_scale_3dgs_route(options)
    route_path = write_large_scale_3dgs_route(route, options)

    if args.format == "json":
        print(json.dumps({**route, "routePath": str(route_path)}, indent=2))
    else:
        print(format_large_scale_3dgs_route_text(route, route_path))


def cmd_large_scale_3dgs_promote(args: argparse.Namespace) -> None:
    """Handle the large-scale-3dgs-promote subcommand."""
    from gs_sim2real.train.large_scale_3dgs import (
        LargeScale3DGSPromoteOptions,
        build_large_scale_3dgs_promotion,
        format_large_scale_3dgs_promotion_text,
        write_large_scale_3dgs_promotion,
    )

    options = LargeScale3DGSPromoteOptions(
        bootstrap_path=Path(args.bootstrap) if args.bootstrap else None,
        plan_path=Path(args.plan) if args.plan else None,
        run_report_path=Path(args.run_report) if args.run_report else None,
        report_path=Path(args.report) if args.report else None,
        public_root=Path(args.public_root),
        catalog_path=Path(args.catalog) if args.catalog else None,
        route_path=Path(args.route) if args.route else None,
        scene_id=args.scene_id,
        label=args.label,
        public_url_prefix=args.public_url_prefix,
        link_mode=args.link_mode,
        require_splats=not args.allow_missing_splats,
        use_full_plan=args.full_plan,
        write_route=not args.no_route,
        web_app_dir=Path(args.web_app_dir) if args.web_app_dir else None,
        site_url=args.site_url,
        tile_preload=args.tile_preload,
        route_playback=not args.no_route_playback,
        route_playback_ms=args.route_playback_ms,
        route_playback_loop=not args.no_route_playback_loop,
        route_label=args.route_label,
        route_description=args.route_description,
        fragment_id=args.fragment_id,
        fragment_label=args.fragment_label,
        frame_id=args.frame_id,
        asset_label=args.asset_label,
        zone_map_url=args.zone_map_url,
        world_splat_url=args.world_splat_url,
        collider_mesh_url=args.collider_mesh_url,
        default_y=args.default_y,
        route_order=args.route_order,
        include_missing_splats_in_route=args.include_missing_splats_in_route,
    )
    report = build_large_scale_3dgs_promotion(options)
    report_path = write_large_scale_3dgs_promotion(report, options.report_path)

    if args.format == "json":
        print(json.dumps({**report, "reportPath": str(report_path)}, indent=2))
    else:
        print(format_large_scale_3dgs_promotion_text(report, report_path))


def cmd_view(args: argparse.Namespace) -> None:
    """Handle the view subcommand."""
    from gs_sim2real.viewer.web_viewer import GaussianViewer

    viewer = GaussianViewer(host=args.host, port=args.port)

    if args.colmap:
        viewer.view_colmap(args.model)
    else:
        viewer.view_ply(args.model)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export subcommand."""
    if args.format == "json":
        from gs_sim2real.viewer.web_export import ply_to_json

        result = ply_to_json(args.model, args.output, max_points=args.max_points)
    elif args.format == "binary":
        from gs_sim2real.viewer.web_export import ply_to_binary

        result = ply_to_binary(args.model, args.output, max_points=args.max_points)
    elif args.format == "splat":
        from gs_sim2real.viewer.web_export import ply_to_splat

        result = ply_to_splat(
            args.model,
            args.output,
            max_points=args.max_points,
            normalize_target_extent=args.splat_normalize_extent,
            min_opacity=args.splat_min_opacity,
            max_scale=args.splat_max_scale,
            max_scale_percentile=args.splat_max_scale_percentile,
        )
    else:
        from gs_sim2real.viewer.web_export import ply_to_scene_bundle

        result = ply_to_scene_bundle(
            args.model,
            args.output,
            asset_format=args.bundle_asset_format,
            scene_id=args.scene_id,
            label=args.label,
            description=args.description,
            max_points=args.max_points,
        )

    print(f"Exported to: {result}")


def cmd_splat_tile_catalog(args: argparse.Namespace) -> None:
    """Handle the splat-tile-catalog subcommand."""
    from gs_sim2real.viewer.web_export import splat_to_tile_catalog

    catalog = splat_to_tile_catalog(
        args.input,
        args.output,
        public_root=args.public_root,
        scene_id=args.scene_id,
        label=args.label,
        tile_size=args.tile_size,
        overlap=args.overlap,
        axes=args.axes,
        min_splats=args.min_splats,
        public_url_prefix=args.public_url_prefix,
    )

    if args.json:
        print(json.dumps(catalog, indent=2))
    else:
        summary = catalog["summary"]
        print("Splat tile catalog")
        print(f"  scene: {catalog['sceneId']} / {catalog['label']}")
        print(f"  input: {catalog['planPath'].removesuffix(':splat-tiling')}")
        print(f"  tiles: {summary['readyTileCount']} ready / {summary['inputSplatCount']} input splats")
        print(f"  catalog: {args.output}")


def cmd_photos_to_splat(args: argparse.Namespace) -> None:
    """Handle the photos-to-splat subcommand.

    One-shot pipeline: image directory -> pose-free sparse (DUSt3R by default)
    -> gsplat training -> antimatter15 .splat binary written to ``<output>/<name>.splat``.
    The .splat can be dropped into ``docs/assets/...`` or served through the
    Pages ``splat.html?url=...`` URL directly.
    """
    from gs_sim2real.preprocess.pose_free import PoseFreeProcessor
    from gs_sim2real.train.gsplat_trainer import train_gsplat
    from gs_sim2real.viewer.web_export import ply_to_splat

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Error: --images must be a directory (got {images_dir})")
        sys.exit(2)

    output_dir = Path(args.output)
    quality_args = _resolve_photos_to_splat_quality(args)
    sparse_dir = output_dir / "sparse_input"
    train_dir = output_dir / "train"
    splat_path = output_dir / f"{images_dir.name}.splat"

    print("=" * 60)
    print(f"Step 1/3: Pose-free preprocess ({quality_args.preprocess}, quality={quality_args.quality})")
    print("=" * 60)
    if quality_args.quality in {"clean", "hero"} and quality_args.preprocess == "dust3r":
        print(
            "Tip: --preprocess mast3r usually produces cleaner pose-free outdoor maps "
            "than DUSt3R when the checkpoint is available."
        )
    processor_kwargs = _pose_free_kwargs_from_args(quality_args, method=quality_args.preprocess)
    processor = PoseFreeProcessor(**processor_kwargs)
    processor.estimate_poses(images_dir, sparse_dir)

    print("\n" + "=" * 60)
    print(f"Step 2/3: gsplat training ({quality_args.iterations} iterations)")
    print("=" * 60)
    config = None
    if quality_args.config:
        from gs_sim2real.common.config import load_config

        config = load_config(quality_args.config)
    _preflight_gsplat_train_data(sparse_dir, getattr(quality_args, "skip_data_check", False))
    ply_path = train_gsplat(
        data_dir=sparse_dir,
        output_dir=train_dir,
        config=config,
        num_iterations=quality_args.iterations,
    )

    print("\n" + "=" * 60)
    print("Step 3/3: Exporting to antimatter15 .splat format")
    print("=" * 60)
    splat_path.parent.mkdir(parents=True, exist_ok=True)
    ply_to_splat(
        ply_path,
        splat_path,
        max_points=quality_args.splat_max_points,
        normalize_target_extent=quality_args.splat_normalize_extent,
        min_opacity=quality_args.splat_min_opacity,
        max_scale=quality_args.splat_max_scale,
        max_scale_percentile=quality_args.splat_max_scale_percentile,
    )
    print(f"\nDone. Open locally: docs/splat.html?url={splat_path}")
    print(f"Splat file: {splat_path}")


def cmd_splat_filter(args: argparse.Namespace) -> None:
    """Handle the splat-filter subcommand."""
    from gs_sim2real.viewer.web_export import filter_splat_file

    report = filter_splat_file(
        args.input,
        args.output,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
        max_scale_percentile=args.max_scale_percentile,
        max_points=args.max_points,
    )
    print(f"Filtered splat: {args.output}")
    adaptive = f"{report.adaptive_max_scale:.4f}" if report.adaptive_max_scale is not None else "off"
    print(
        f"Kept {report.output_count:,}/{report.input_count:,} splats "
        f"({report.kept_ratio:.1%}); adaptive_max_scale={adaptive}"
    )


def cmd_splat_inspect(args: argparse.Namespace) -> None:
    """Handle the splat-inspect subcommand."""
    from gs_sim2real.viewer.web_export import inspect_splat_file

    report = inspect_splat_file(args.input, low_opacity_threshold=args.low_opacity_threshold)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return

    print(f"Splat: {args.input}")
    print(f"Gaussians: {report.input_count:,} ({report.size_mb:.2f} MB)")
    print(
        "Opacity: "
        f"min={report.opacity_min:.3f} "
        f"p10={report.opacity_p10:.3f} "
        f"p50={report.opacity_p50:.3f} "
        f"p90={report.opacity_p90:.3f} "
        f"max={report.opacity_max:.3f}"
    )
    print(
        "Scale max: "
        f"p50={report.scale_p50:.4f} "
        f"p95={report.scale_p95:.4f} "
        f"p98={report.scale_p98:.4f} "
        f"p99={report.scale_p99:.4f} "
        f"max={report.scale_max:.4f} "
        f"tail={report.scale_tail_ratio:.1f}x"
    )
    print(
        f"Low-opacity splats (<{report.low_opacity_threshold:g}): "
        f"{report.low_opacity_count:,} ({report.low_opacity_ratio:.1%})"
    )
    if report.low_opacity_ratio >= 0.05 or report.scale_tail_ratio >= 4.0:
        print("Suggested cleanup:")
        print(
            "  gs-mapper splat-filter "
            f"--input {args.input} --output <clean.splat> "
            f"--min-opacity {report.low_opacity_threshold:g} --max-scale-percentile 98"
        )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the benchmark subcommand."""
    from gs_sim2real.benchmark import Benchmark

    bench = Benchmark(data_dir=args.data, output_dir=args.output)

    if args.method in ("gsplat", "both"):
        print("Running gsplat benchmark...")
        bench.run_gsplat(
            num_iterations=args.iterations,
            dataset_name=args.dataset_name,
            skip_data_check=getattr(args, "skip_data_check", False),
        )

    if args.method in ("nerfstudio", "both"):
        print("Running nerfstudio benchmark...")
        bench.run_nerfstudio(num_iterations=args.iterations, dataset_name=args.dataset_name)

    print("\n" + bench.compare())
    bench.save_results()
    print(f"\nResults saved to: {Path(args.output) / 'benchmark_results.json'}")


def _run_waymo_preprocess(
    source_dir: Path,
    colmap_dir: Path,
    args: argparse.Namespace,
):
    """Extract Waymo frames and prepare COLMAP-format inputs for training."""
    from gs_sim2real.datasets.waymo import WaymoLoader
    from gs_sim2real.preprocess.colmap import run_colmap

    loader = WaymoLoader(data_dir=str(source_dir))
    images_out = loader.extract_frames(
        output_dir=str(colmap_dir),
        camera=args.camera,
        max_frames=args.max_frames,
        every_n=args.every_n,
    )

    params_path = colmap_dir / "camera_params.json"
    if params_path.exists():
        sparse_dir = loader.to_colmap_format(
            camera_params_path=str(params_path),
            output_dir=str(colmap_dir),
        )
        print(f"Waymo frames extracted to: {images_out}")
        print(f"COLMAP sparse model at: {sparse_dir}")
    else:
        sparse_dir = run_colmap(
            image_dir=images_out,
            output_dir=colmap_dir,
            matching=getattr(args, "matching", "exhaustive"),
            use_gpu=not getattr(args, "no_gpu", False),
            colmap_path=getattr(args, "colmap_path", "colmap"),
        )
        print(f"Waymo frames loaded from: {images_out}")
        print(f"COLMAP sparse model at: {sparse_dir}")

    if getattr(args, "extract_lidar_depth", False):
        depth_dir = loader.extract_lidar_depth(
            output_dir=str(colmap_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        print(f"Waymo LiDAR depth extracted to: {depth_dir}")

    if getattr(args, "extract_dynamic_masks", False):
        masks_dir = loader.extract_dynamic_masks(
            output_dir=str(colmap_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        print(f"Waymo dynamic masks extracted to: {masks_dir}")

    return sparse_dir


def _parse_topic_arg(value: str | None) -> str | list[str] | None:
    """Parse a CLI topic argument without importing MCD internals at startup."""
    from gs_sim2real.preprocess.mcd import parse_topic_arg

    return parse_topic_arg(value)


def _run_mcd_preprocess_to_colmap(
    source_dir: Path,
    colmap_dir: Path,
    args: argparse.Namespace,
    *,
    run_colmap: bool = True,
):
    """Delegate MCD preprocessing to the isolated MCD preprocess module."""
    from gs_sim2real.preprocess.mcd import MCDPreprocessOptions, run_mcd_preprocess_to_colmap

    options = MCDPreprocessOptions.from_namespace(args)
    options.run_colmap = run_colmap
    return run_mcd_preprocess_to_colmap(source_dir, colmap_dir, options)


def _run_lidar_slam_preprocess_to_colmap(
    images_dir: Path,
    colmap_dir: Path,
    args: argparse.Namespace,
):
    """Import a trajectory through the existing generic trajectory importer."""
    from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

    trajectory = getattr(args, "trajectory", None)
    if not trajectory:
        print("Error: --trajectory is required for lidar-slam method.")
        sys.exit(1)
    return import_lidar_slam(
        trajectory_path=trajectory,
        image_dir=images_dir,
        output_dir=colmap_dir,
        trajectory_format=getattr(args, "trajectory_format", "tum"),
        pointcloud_path=getattr(args, "pointcloud", None),
        pinhole_calib_path=getattr(args, "pinhole_calib", None),
        nmea_time_offset_sec=getattr(args, "nmea_time_offset_sec", 0.0),
    )


def _run_external_slam_preprocess_to_colmap(
    images_dir: Path,
    colmap_dir: Path,
    args: argparse.Namespace,
):
    """Import artifacts exported by MASt3R-SLAM/VGGT-SLAM/LoGeR/Pi3-like front-ends."""
    from gs_sim2real.preprocess import external_slam as external_slam_module

    try:
        if getattr(args, "external_slam_dry_run", False):
            try:
                manifest = external_slam_module.build_external_slam_artifact_manifest(
                    image_dir=images_dir,
                    system=getattr(args, "external_slam_system", "generic"),
                    artifact_dir=getattr(args, "external_slam_output", None),
                    trajectory_path=getattr(args, "trajectory", None),
                    trajectory_format=getattr(args, "trajectory_format", None),
                    pointcloud_path=getattr(args, "pointcloud", None),
                    pinhole_calib_path=getattr(args, "pinhole_calib", None),
                )
            except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
                manifest = external_slam_module.build_external_slam_artifact_error_manifest(
                    error=exc,
                    image_dir=images_dir,
                    system=getattr(args, "external_slam_system", "generic"),
                    artifact_dir=getattr(args, "external_slam_output", None),
                    trajectory_path=getattr(args, "trajectory", None),
                    trajectory_format=getattr(args, "trajectory_format", None),
                    pointcloud_path=getattr(args, "pointcloud", None),
                    pinhole_calib_path=getattr(args, "pinhole_calib", None),
                )
                if getattr(args, "external_slam_manifest_format", "text") == "json":
                    print(external_slam_module.render_external_slam_artifact_manifest_json(manifest), end="")
                else:
                    print(external_slam_module.render_external_slam_artifact_manifest_text(manifest), end="")
                raise SystemExit(2 if getattr(args, "external_slam_fail_on_dry_run_gate", False) else 1) from exc
            gate = external_slam_module.evaluate_external_slam_manifest_gate(
                manifest,
                external_slam_module.ExternalSLAMManifestGatePolicy(
                    min_aligned_frames=getattr(args, "external_slam_min_aligned_frames", 2),
                    allow_dropped_images=getattr(args, "external_slam_allow_dropped_images", False),
                    require_pointcloud=getattr(args, "external_slam_require_pointcloud", False),
                    min_point_count=getattr(args, "external_slam_min_point_count", 0),
                ),
            )
            manifest["gate"] = gate
            if getattr(args, "external_slam_manifest_format", "text") == "json":
                print(external_slam_module.render_external_slam_artifact_manifest_json(manifest), end="")
            else:
                print(external_slam_module.render_external_slam_artifact_manifest_text(manifest), end="")
                print(external_slam_module.render_external_slam_manifest_gate_text(gate), end="")
            if getattr(args, "external_slam_fail_on_dry_run_gate", False) and not gate["passed"]:
                raise SystemExit(2)
            return None
        return external_slam_module.import_external_slam(
            image_dir=images_dir,
            output_dir=colmap_dir,
            system=getattr(args, "external_slam_system", "generic"),
            artifact_dir=getattr(args, "external_slam_output", None),
            trajectory_path=getattr(args, "trajectory", None),
            trajectory_format=getattr(args, "trajectory_format", None),
            pointcloud_path=getattr(args, "pointcloud", None),
            pinhole_calib_path=getattr(args, "pinhole_calib", None),
            nmea_time_offset_sec=getattr(args, "nmea_time_offset_sec", 0.0),
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the run subcommand (full pipeline)."""
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    colmap_dir = output_dir / "colmap"
    train_dir = output_dir / "train"
    config = None

    if args.config:
        from gs_sim2real.common.config import load_config

        config = load_config(args.config)

    # Step 1: Preprocess
    if not args.skip_preprocess:
        preprocess_method = args.preprocess_method
        print("=" * 60)
        print(f"Step 1: Preprocessing ({preprocess_method})")
        print("=" * 60)

        if preprocess_method == "lidar-slam":
            _run_lidar_slam_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method == "external-slam":
            _run_external_slam_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method == "waymo":
            _run_waymo_preprocess(images_dir, colmap_dir, args)
        elif preprocess_method == "mcd":
            _run_mcd_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method in _POSE_FREE_METHOD_MAP:
            from gs_sim2real.preprocess.pose_free import run_pose_free

            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                **_pose_free_kwargs_from_args(args, method=preprocess_method),
            )
        else:
            from gs_sim2real.preprocess.colmap import run_colmap

            run_colmap(
                image_dir=images_dir,
                output_dir=colmap_dir,
                matching=args.matching,
                use_gpu=not args.no_gpu,
                colmap_path=args.colmap_path,
            )
    else:
        print("Skipping preprocessing (--skip-preprocess)")

    # Step 2: Train
    print("\n" + "=" * 60)
    print("Step 2: Training")
    print("=" * 60)

    ply_path = None
    if args.method == "gsplat":
        from gs_sim2real.train.gsplat_trainer import train_gsplat

        _preflight_gsplat_train_data(colmap_dir, getattr(args, "skip_data_check", False))
        ply_path = train_gsplat(
            data_dir=colmap_dir,
            output_dir=train_dir,
            config=config,
            num_iterations=args.iterations,
        )
    else:
        from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

        train_nerfstudio(
            data_dir=colmap_dir,
            output_dir=train_dir,
        )

    # Step 3: View
    if not args.no_viewer and ply_path is not None:
        print("\n" + "=" * 60)
        print("Step 3: Viewer")
        print("=" * 60)
        from gs_sim2real.viewer.web_viewer import launch_viewer

        launch_viewer(ply_path, port=args.port)

    print("\nPipeline complete!")


def cmd_demo(args: argparse.Namespace) -> None:
    """Handle the demo subcommand (images -> splat -> Dynamic Map Viewer teleop)."""
    import subprocess

    ply_path = None

    if args.ply:
        # Use an existing PLY directly
        ply_path = Path(args.ply)
        if not ply_path.exists():
            print(f"Error: PLY file not found: {ply_path}")
            sys.exit(1)
        print(f"Using existing PLY: {ply_path}")
    elif args.images:
        images_dir = Path(args.images)
        output_dir = Path(args.output)
        colmap_dir = output_dir / "colmap"
        train_dir = output_dir / "train"
        config = None

        if args.config:
            from gs_sim2real.common.config import load_config

            config = load_config(args.config)

        # Step 1: Preprocess
        preprocess_method = args.preprocess_method
        print("=" * 60)
        print(f"Step 1/3: Preprocessing ({preprocess_method})")
        print("=" * 60)

        if preprocess_method == "lidar-slam":
            _run_lidar_slam_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method == "external-slam":
            _run_external_slam_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method == "waymo":
            _run_waymo_preprocess(images_dir, colmap_dir, args)
        elif preprocess_method == "mcd":
            _run_mcd_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method in _POSE_FREE_METHOD_MAP:
            from gs_sim2real.preprocess.pose_free import run_pose_free

            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                **_pose_free_kwargs_from_args(args, method=preprocess_method),
            )
        else:
            from gs_sim2real.preprocess.colmap import run_colmap as _run_colmap

            _run_colmap(
                image_dir=images_dir,
                output_dir=colmap_dir,
                matching=args.matching,
                use_gpu=not args.no_gpu,
                colmap_path=args.colmap_path,
            )

        # Step 2: Train
        print("\n" + "=" * 60)
        print("Step 2/3: Training")
        print("=" * 60)

        if args.method == "gsplat":
            from gs_sim2real.train.gsplat_trainer import train_gsplat

            _preflight_gsplat_train_data(colmap_dir, getattr(args, "skip_data_check", False))
            ply_path = train_gsplat(
                data_dir=colmap_dir,
                output_dir=train_dir,
                config=config,
                num_iterations=args.iterations,
            )
        else:
            from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

            train_nerfstudio(data_dir=colmap_dir, output_dir=train_dir)
            ply_path = train_dir / "point_cloud.ply"
    else:
        print("Error: --images or --ply is required.")
        sys.exit(1)

    # Step 3: Stage into Dynamic Map Viewer
    print("\n" + "=" * 60)
    print("Step 3/3: Staging for Dynamic Map Viewer")
    print("=" * 60)

    from gs_sim2real.demo.stage_for_dreamwalker import stage_ply

    result = stage_ply(ply_path, fragment=args.fragment)
    print(f"Splat staged: {result['splat_dest']}")
    print(f"Manifest updated: {result['manifest']}")
    print(f"Launch URL: {result['launch_url']}")

    # Launch Vite dev server
    if not args.no_launch:
        dreamwalker_dir = Path(result["manifest"]).parent.parent.parent
        print(f"\nStarting Dynamic Map Viewer dev server in {dreamwalker_dir} ...")
        print("Open your browser at:", result["launch_url"])
        print("Controls: WASD = move, Mouse = look, R = toggle robot mode")
        subprocess.run(["npm", "run", "dev"], cwd=dreamwalker_dir)
    else:
        print("\nTo launch manually:")
        print(f"  cd {Path(result['manifest']).parent.parent.parent}")
        print("  npm run dev")
        print(f"  Open: {result['launch_url']}")


def cmd_robotics_node(args: argparse.Namespace) -> None:
    """Handle the robotics-node subcommand."""
    from gs_sim2real.robotics.ros2_bridge_node import run_cli

    run_cli(args)


def cmd_sim2real_server(args: argparse.Namespace) -> None:
    """Handle the sim2real-server subcommand."""
    from gs_sim2real.robotics.gsplat_render_server import run_cli

    run_cli(args)


def cmd_sim2real_query(args: argparse.Namespace) -> None:
    """Handle the sim2real-query subcommand."""
    from gs_sim2real.robotics.render_query_client import run_cli

    run_cli(args)


def cmd_sim2real_benchmark_images(args: argparse.Namespace) -> None:
    """Handle the sim2real-benchmark-images subcommand."""
    from gs_sim2real.robotics.localization_image_benchmark import run_cli

    run_cli(args)


def cmd_route_policy_benchmark(args: argparse.Namespace) -> None:
    """Handle the route-policy-benchmark subcommand."""
    from gs_sim2real.sim.policy_benchmark import run_cli

    run_cli(args)


def cmd_route_policy_benchmark_history(args: argparse.Namespace) -> None:
    """Handle the route-policy-benchmark-history subcommand."""
    from gs_sim2real.sim.policy_benchmark_history import run_cli

    run_cli(args)


def cmd_route_policy_scenario_set(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-set subcommand."""
    from gs_sim2real.sim.policy_scenario_set import run_cli

    run_cli(args)


def cmd_route_policy_scenario_matrix(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-matrix subcommand."""
    from gs_sim2real.sim.policy_scenario_matrix import run_cli

    run_cli(args)


def cmd_route_policy_scenario_shards(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-shards subcommand."""
    from gs_sim2real.sim.policy_scenario_sharding import run_shard_plan_cli

    run_shard_plan_cli(args)


def cmd_route_policy_scenario_shard_merge(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-shard-merge subcommand."""
    from gs_sim2real.sim.policy_scenario_sharding import run_shard_merge_cli

    run_shard_merge_cli(args)


def cmd_route_policy_scenario_ci_manifest(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-manifest subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_manifest import run_cli

    run_cli(args)


def cmd_route_policy_scenario_ci_workflow(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-workflow subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_workflow import run_cli

    run_cli(args)


def cmd_route_policy_scenario_ci_workflow_validate(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-workflow-validate subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_workflow import run_validation_cli

    run_validation_cli(args)


def cmd_route_policy_scenario_ci_workflow_activate(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-workflow-activate subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_activation import run_activation_cli

    run_activation_cli(args)


def cmd_route_policy_scenario_ci_review(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-review subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_review import run_review_cli

    run_review_cli(args)


def cmd_route_policy_dataset_to_trace(args: argparse.Namespace) -> None:
    """Handle the route-policy-dataset-to-trace subcommand."""
    from gs_sim2real.sim.policy_trace import run_dataset_to_trace_cli

    run_dataset_to_trace_cli(args)


def cmd_route_policy_trace_to_event_windows(args: argparse.Namespace) -> None:
    """Handle the route-policy-trace-to-event-windows subcommand."""
    from gs_sim2real.sim.policy_trace import run_trace_to_event_windows_cli

    run_trace_to_event_windows_cli(args)


def cmd_route_policy_scenario_ci_workflow_promote(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-workflow-promote subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_promotion import run_promotion_cli

    run_promotion_cli(args)


def cmd_route_policy_scenario_ci_workflow_adopt(args: argparse.Namespace) -> None:
    """Handle the route-policy-scenario-ci-workflow-adopt subcommand."""
    from gs_sim2real.sim.policy_scenario_ci_adoption import run_adoption_cli

    run_adoption_cli(args)


def cmd_experiment(args: argparse.Namespace) -> None:
    """Handle the nested `experiment` subcommand by deferring to the legacy handler."""
    handler_map = {
        "localization-alignment": cmd_experiment_localization_alignment,
        "render-backend-selection": cmd_experiment_render_backend_selection,
        "outdoor-training-features": cmd_experiment_outdoor_training_features,
        "localization-import": cmd_experiment_localization_import,
        "query-transport-selection": cmd_experiment_query_transport_selection,
        "query-request-import": cmd_experiment_query_request_import,
        "live-localization-stream-import": cmd_experiment_live_localization_stream_import,
        "route-capture-import": cmd_experiment_route_capture_import,
        "sim2real-websocket-protocol": cmd_experiment_sim2real_websocket_protocol,
        "localization-review-bundle-import": cmd_experiment_localization_review_bundle_import,
        "query-cancellation-policy": cmd_experiment_query_cancellation_policy,
        "query-coalescing-policy": cmd_experiment_query_coalescing_policy,
        "query-error-mapping": cmd_experiment_query_error_mapping,
        "query-queue-policy": cmd_experiment_query_queue_policy,
        "query-source-identity": cmd_experiment_query_source_identity,
        "query-timeout-policy": cmd_experiment_query_timeout_policy,
        "query-response-build": cmd_experiment_query_response_build,
    }
    subcmd = getattr(args, "experiment_command", None)
    if subcmd is None:
        print("Error: specify an experiment lab. Run `gs-mapper experiment --help`.", file=sys.stderr)
        sys.exit(2)
    handler = handler_map.get(subcmd)
    if handler is None:
        print(f"Unknown experiment lab: {subcmd}", file=sys.stderr)
        sys.exit(1)
    handler(args)


def cmd_experiment_localization_alignment(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-alignment subcommand."""
    from gs_sim2real.experiments.localization_alignment_lab import run_cli

    run_cli(args)


def cmd_experiment_render_backend_selection(args: argparse.Namespace) -> None:
    """Handle the experiment-render-backend-selection subcommand."""
    from gs_sim2real.experiments.render_backend_selection_lab import run_cli

    run_cli(args)


def cmd_experiment_outdoor_training_features(args: argparse.Namespace) -> None:
    """Handle the experiment-outdoor-training-features subcommand."""
    from gs_sim2real.experiments.outdoor_training_features_lab import run_cli

    run_cli(args)


def cmd_experiment_localization_import(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-import subcommand."""
    from gs_sim2real.experiments.localization_estimate_import_lab import run_cli

    run_cli(args)


def cmd_experiment_query_transport_selection(args: argparse.Namespace) -> None:
    """Handle the experiment-query-transport-selection subcommand."""
    from gs_sim2real.experiments.query_transport_selection_lab import run_cli

    run_cli(args)


def cmd_experiment_query_request_import(args: argparse.Namespace) -> None:
    """Handle the experiment-query-request-import subcommand."""
    from gs_sim2real.experiments.query_request_import_lab import run_cli

    run_cli(args)


def cmd_experiment_live_localization_stream_import(args: argparse.Namespace) -> None:
    """Handle the experiment-live-localization-stream-import subcommand."""
    from gs_sim2real.experiments.live_localization_stream_import_lab import run_cli

    run_cli(args)


def cmd_experiment_route_capture_import(args: argparse.Namespace) -> None:
    """Handle the experiment-route-capture-import subcommand."""
    from gs_sim2real.experiments.route_capture_bundle_import_lab import run_cli

    run_cli(args)


def cmd_experiment_sim2real_websocket_protocol(args: argparse.Namespace) -> None:
    """Handle the experiment-sim2real-websocket-protocol subcommand."""
    from gs_sim2real.experiments.sim2real_websocket_protocol_lab import run_cli

    run_cli(args)


def cmd_experiment_localization_review_bundle_import(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-review-bundle-import subcommand."""
    from gs_sim2real.experiments.localization_review_bundle_import_lab import run_cli

    run_cli(args)


def cmd_experiment_query_cancellation_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-cancellation-policy subcommand."""
    from gs_sim2real.experiments.query_cancellation_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_coalescing_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-coalescing-policy subcommand."""
    from gs_sim2real.experiments.query_coalescing_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_error_mapping(args: argparse.Namespace) -> None:
    """Handle the experiment-query-error-mapping subcommand."""
    from gs_sim2real.experiments.query_error_mapping_lab import run_cli

    run_cli(args)


def cmd_experiment_query_queue_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-queue-policy subcommand."""
    from gs_sim2real.experiments.query_queue_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_source_identity(args: argparse.Namespace) -> None:
    """Handle the experiment-query-source-identity subcommand."""
    from gs_sim2real.experiments.query_source_identity_lab import run_cli

    run_cli(args)


def cmd_experiment_query_timeout_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-timeout-policy subcommand."""
    from gs_sim2real.experiments.query_timeout_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_response_build(args: argparse.Namespace) -> None:
    """Handle the experiment-query-response-build subcommand."""
    from gs_sim2real.experiments.query_response_build_lab import run_cli

    run_cli(args)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the GS Mapper CLI."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    rewritten = _rewrite_legacy_experiment_argv(raw_argv)
    args = parser.parse_args(rewritten)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "large-scale-3dgs-discover": cmd_large_scale_3dgs_discover,
        "large-scale-3dgs-bootstrap": cmd_large_scale_3dgs_bootstrap,
        "large-scale-3dgs-smoke-data": cmd_large_scale_3dgs_smoke_data,
        "large-scale-3dgs-preflight": cmd_large_scale_3dgs_preflight,
        "large-scale-3dgs-pilot": cmd_large_scale_3dgs_pilot,
        "large-scale-3dgs-plan": cmd_large_scale_3dgs_plan,
        "large-scale-3dgs-run": cmd_large_scale_3dgs_run,
        "large-scale-3dgs-catalog": cmd_large_scale_3dgs_catalog,
        "large-scale-3dgs-route": cmd_large_scale_3dgs_route,
        "large-scale-3dgs-promote": cmd_large_scale_3dgs_promote,
        "view": cmd_view,
        "export": cmd_export,
        "photos-to-splat": cmd_photos_to_splat,
        "video-to-splat": cmd_video_to_splat,
        "map": cmd_video_to_splat,
        "splat-filter": cmd_splat_filter,
        "splat-inspect": cmd_splat_inspect,
        "splat-tile-catalog": cmd_splat_tile_catalog,
        "benchmark": cmd_benchmark,
        "run": cmd_run,
        "demo": cmd_demo,
        "robotics-node": cmd_robotics_node,
        "sim2real-server": cmd_sim2real_server,
        "sim2real-query": cmd_sim2real_query,
        "sim2real-benchmark-images": cmd_sim2real_benchmark_images,
        "route-policy-benchmark": cmd_route_policy_benchmark,
        "route-policy-benchmark-history": cmd_route_policy_benchmark_history,
        "route-policy-dataset-to-trace": cmd_route_policy_dataset_to_trace,
        "route-policy-scenario-ci-manifest": cmd_route_policy_scenario_ci_manifest,
        "route-policy-scenario-ci-review": cmd_route_policy_scenario_ci_review,
        "route-policy-scenario-ci-workflow-activate": cmd_route_policy_scenario_ci_workflow_activate,
        "route-policy-scenario-ci-workflow-adopt": cmd_route_policy_scenario_ci_workflow_adopt,
        "route-policy-scenario-ci-workflow": cmd_route_policy_scenario_ci_workflow,
        "route-policy-scenario-ci-workflow-promote": cmd_route_policy_scenario_ci_workflow_promote,
        "route-policy-scenario-ci-workflow-validate": cmd_route_policy_scenario_ci_workflow_validate,
        "route-policy-scenario-matrix": cmd_route_policy_scenario_matrix,
        "route-policy-scenario-shard-merge": cmd_route_policy_scenario_shard_merge,
        "route-policy-scenario-shards": cmd_route_policy_scenario_shards,
        "route-policy-scenario-set": cmd_route_policy_scenario_set,
        "route-policy-trace-to-event-windows": cmd_route_policy_trace_to_event_windows,
        "experiment": cmd_experiment,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
