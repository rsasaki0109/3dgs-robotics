"""Video-to-frames extraction utility.

This module extracts frames from video files at a configurable frame rate
or interval. Extracted frames are saved as individual image files for
downstream COLMAP or direct 3DGS processing.

Supported video formats: mp4, avi, mkv, mov (via OpenCV VideoCapture).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


@dataclass(frozen=True)
class VideoFrameSamplingPlan:
    """Frame interval settings for evenly sampling a video."""

    every_n: int
    max_frames: int | None
    total_frames: int
    video_fps: float


def probe_video_metadata(video_path: Path | str) -> tuple[int, float]:
    """Return ``(total_frames, video_fps)`` for a readable video file."""
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for frame extraction. Install it with: pip install opencv-python")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if video_fps <= 0:
        logger.warning("Could not determine video FPS for %s, assuming 30.", video_path.name)
        video_fps = 30.0

    return max(total_frames, 0), video_fps


def plan_video_frame_sampling(
    video_path: Path | str,
    target_frames: int = 32,
) -> VideoFrameSamplingPlan:
    """Choose ``every_n`` / ``max_frames`` to sample a video into roughly ``target_frames`` frames."""
    if target_frames < 0:
        raise ValueError("target_frames must be >= 0")

    total_frames, video_fps = probe_video_metadata(video_path)

    if target_frames == 0:
        every_n = 1
        max_frames = None
    elif total_frames <= 0:
        every_n = 1
        max_frames = target_frames
    elif total_frames <= target_frames:
        every_n = 1
        max_frames = total_frames
    else:
        every_n = max(1, total_frames // target_frames)
        max_frames = target_frames

    return VideoFrameSamplingPlan(
        every_n=every_n,
        max_frames=max_frames,
        total_frames=total_frames,
        video_fps=video_fps,
    )


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    fps: float | None = None,
    every_n: int | None = None,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[Path]:
    """Extract frames from a video file.

    Specify either ``fps`` (target frames per second) or ``every_n``
    (extract every N-th frame). If neither is given, defaults to fps=2.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be saved.
        fps: Target extraction rate in frames per second.
        every_n: Extract every N-th frame from the video.
        max_frames: Maximum number of frames to extract.
        resize: Optional (width, height) to resize frames.

    Returns:
        List of paths to the extracted frame images.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If both fps and every_n are specified.
        RuntimeError: If OpenCV cannot open the video.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for frame extraction. Install it with: pip install opencv-python")

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if fps is not None and every_n is not None:
        raise ValueError("Specify either 'fps' or 'every_n', not both.")

    # Default to 2 fps if neither is specified
    if fps is None and every_n is None:
        fps = 2.0

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        logger.warning("Could not determine video FPS, assuming 30.")
        video_fps = 30.0

    # Calculate frame interval
    if every_n is not None:
        frame_interval = every_n
    else:
        # fps mode
        frame_interval = max(1, int(round(video_fps / fps)))

    logger.info(
        "Video: %s, FPS=%.1f, total_frames=%d, extracting every %d frames",
        video_path.name,
        video_fps,
        total_frames,
        frame_interval,
    )
    print(
        f"Extracting frames from {video_path.name} "
        f"(video FPS={video_fps:.1f}, total={total_frames}, interval={frame_interval})"
    )

    try:
        from tqdm import tqdm

        progress = tqdm(total=min(total_frames, (max_frames or total_frames) * frame_interval), desc="Extracting")
    except ImportError:
        progress = None

    extracted_paths: list[Path] = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize)

            out_path = output_dir / f"frame_{saved_count:05d}.png"
            cv2.imwrite(str(out_path), frame)
            extracted_paths.append(out_path)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_idx += 1
        if progress is not None:
            progress.update(1)

    cap.release()
    if progress is not None:
        progress.close()

    print(f"Extracted {saved_count} frames to {output_dir}")
    return extracted_paths


def extract_frames_from_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    fps: float = 2.0,
    max_frames: int = 100,
    resize: tuple[int, int] | None = None,
) -> dict[str, list[Path]]:
    """Process all videos in a directory.

    Args:
        input_dir: Directory containing video files.
        output_dir: Root directory for extracted frames. Each video gets a subdirectory.
        fps: Target extraction rate in frames per second.
        max_frames: Maximum number of frames per video.
        resize: Optional (width, height) to resize frames.

    Returns:
        Dictionary mapping video filenames to lists of extracted frame paths.

    Raises:
        FileNotFoundError: If input_dir does not exist.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    video_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)

    if not video_files:
        print(f"No video files found in {input_dir}")
        return {}

    print(f"Found {len(video_files)} video(s) in {input_dir}")

    results: dict[str, list[Path]] = {}
    for video_path in video_files:
        video_output = output_dir / video_path.stem
        frames = extract_frames(
            video_path,
            video_output,
            fps=fps,
            max_frames=max_frames,
            resize=resize,
        )
        results[video_path.name] = frames

    return results
