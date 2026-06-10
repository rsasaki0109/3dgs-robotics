"""Generic rosbag image frame source — no ROS runtime required.

Reads camera frames out of ROS 1 ``.bag`` files and ROS 2 ``rosbag2``
recordings (``.db3`` / ``.mcap``, bare files or directories with
``metadata.yaml``) using the pure-Python ``rosbags`` library. This is the
shared layer behind the MCD dataset loader, the live-mapping bag replay
(``scripts/run_live_mapping_demo.py --bag``), and the ``map <bag>`` CLI
entrance.

Only ``sensor_msgs`` ``Image`` / ``CompressedImage`` topics are supported;
exotic transports (e.g. theora) raise an explicit error instead of being
silently skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_MSGTYPES = frozenset(
    {
        "sensor_msgs/msg/Image",
        "sensor_msgs/Image",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    }
)
COMPRESSED_IMAGE_MSGTYPES = frozenset(
    {
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    }
)

ROSBAG_FILE_SUFFIXES = frozenset({".bag", ".db3", ".mcap"})


# --------------------------------------------------------------------- readers


def get_anyreader():
    """Return the rosbags AnyReader class (raises ImportError when missing)."""
    from rosbags.highlevel import AnyReader

    return AnyReader


def reader_kwargs(bag_paths: list[Path]) -> dict[str, Any]:
    """Build AnyReader kwargs (default typestore for bags lacking definitions)."""
    if not bag_paths:
        return {}

    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        return {}

    if any(path.suffix == ".bag" for path in bag_paths):
        return {"default_typestore": get_typestore(Stores.ROS1_NOETIC)}
    if any(path.is_dir() or path.suffix in {".db3", ".mcap"} for path in bag_paths):
        return {"default_typestore": get_typestore(Stores.ROS2_HUMBLE)}
    return {}


def create_reader(reader_cls: type[Any], bag_paths: list[Path]):
    """Instantiate AnyReader, retrying without kwargs for shims that reject them."""
    kwargs = reader_kwargs(bag_paths)
    if not kwargs:
        return reader_cls(bag_paths)
    try:
        return reader_cls(bag_paths, **kwargs)
    except TypeError as exc:
        if "default_typestore" not in str(exc):
            raise
        return reader_cls(bag_paths)


def find_bag_paths(path: str | Path) -> list[Path]:
    """Find rosbag1 files or rosbag2 recordings under ``path``.

    Accepts a ``.bag`` file, a bare ``.db3`` / ``.mcap`` file, a rosbag2
    directory (with ``metadata.yaml``), or a directory tree containing any of
    those.
    """
    path = Path(path)
    if path.is_file() and path.suffix in {".db3", ".mcap"}:
        # Prefer the enclosing rosbag2 directory when metadata is available.
        if (path.parent / "metadata.yaml").exists():
            return [path.parent]
        return [path]
    if path.is_file() and path.suffix == ".bag":
        return [path]
    if path.is_dir() and (path / "metadata.yaml").exists():
        return [path]
    if path.is_dir():
        bag2_dirs = sorted({p.parent for p in path.rglob("metadata.yaml")})
        if bag2_dirs:
            return bag2_dirs
        bag1_paths = sorted(path.rglob("*.bag"))
        if bag1_paths:
            return bag1_paths
        return sorted(list(path.rglob("*.mcap")) + list(path.rglob("*.db3")))
    return []


def is_rosbag_path(path: str | Path) -> bool:
    """Return True when ``path`` looks like a rosbag input (file or directory)."""
    path = Path(path)
    if path.is_file():
        return path.suffix in ROSBAG_FILE_SUFFIXES
    if path.is_dir():
        return bool(find_bag_paths(path))
    return False


# ------------------------------------------------------------ topic selection


def normalize_requested_topics(
    requested_topic: str | list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Normalize a requested topic argument (string is comma-separable) into a list."""
    if requested_topic is None:
        return None
    if isinstance(requested_topic, (list, tuple)):
        topics = [str(topic).strip() for topic in requested_topic if str(topic).strip()]
        return topics or None
    topics = [topic.strip() for topic in str(requested_topic).split(",") if topic.strip()]
    return topics or None


def select_connection(
    topics: dict[str, Any],
    requested_topic: str | None,
    preferred_topics: tuple[str, ...],
    allowed_msgtypes: frozenset[str],
):
    """Pick the first matching connection for the requested or preferred topics."""
    if requested_topic:
        info = topics.get(requested_topic)
        if info is None:
            raise ValueError(f"Requested topic not found: {requested_topic}")
        for connection in info.connections:
            if connection.msgtype in allowed_msgtypes:
                return connection
        raise ValueError(f"Requested topic {requested_topic} has unsupported type(s).")

    for topic_name in preferred_topics:
        info = topics.get(topic_name)
        if info is None:
            continue
        for connection in info.connections:
            if connection.msgtype in allowed_msgtypes:
                return connection

    for info in topics.values():
        for connection in info.connections:
            if connection.msgtype in allowed_msgtypes:
                return connection
    return None


def select_connections(
    topics: dict[str, Any],
    requested_topics: list[str] | None,
    preferred_topics: tuple[str, ...],
    allowed_msgtypes: frozenset[str],
) -> list[Any]:
    """Pick one or more matching connections."""
    if requested_topics:
        selected = []
        for requested_topic in requested_topics:
            connection = select_connection(
                topics,
                requested_topic=requested_topic,
                preferred_topics=preferred_topics,
                allowed_msgtypes=allowed_msgtypes,
            )
            if connection is None:
                raise ValueError(f"Requested topic not found: {requested_topic}")
            selected.append(connection)
        return selected

    connection = select_connection(
        topics,
        requested_topic=None,
        preferred_topics=preferred_topics,
        allowed_msgtypes=allowed_msgtypes,
    )
    return [] if connection is None else [connection]


def sanitize_topic_name(topic_name: str) -> str:
    """Convert a ROS topic name into a filesystem-friendly folder label."""
    parts = [part for part in topic_name.split("/") if part]
    if not parts:
        return "images"
    return "__".join(parts)


@dataclass(frozen=True)
class BagImageTopic:
    """An image-bearing topic discovered in a bag."""

    topic: str
    msgtype: str
    msgcount: int


def list_image_topics(bag_path: str | Path) -> list[BagImageTopic]:
    """List Image/CompressedImage topics with message counts."""
    bag_paths = find_bag_paths(bag_path)
    if not bag_paths:
        raise FileNotFoundError(f"No rosbag found at {bag_path}")
    reader_cls = get_anyreader()
    found: list[BagImageTopic] = []
    with create_reader(reader_cls, bag_paths) as reader:
        for topic_name, info in sorted(reader.topics.items()):
            msgtype = info.msgtype or "unknown"
            if msgtype in IMAGE_MSGTYPES:
                found.append(BagImageTopic(topic_name, msgtype, int(info.msgcount)))
    return found


def resolve_image_topic(bag_path: str | Path, requested_topic: str | None = None) -> BagImageTopic:
    """Resolve which image topic to read.

    With ``requested_topic`` set, validates it carries a supported image type.
    Otherwise a single image topic is auto-selected; zero or multiple topics
    raise an actionable error listing the candidates for ``--image-topic``.
    """
    candidates = list_image_topics(bag_path)
    if requested_topic:
        for candidate in candidates:
            if candidate.topic == requested_topic:
                return candidate
        listing = _format_topic_listing(candidates)
        raise ValueError(
            f"Topic {requested_topic} is not a supported image topic in {bag_path}.\n"
            f"Image topics in this bag:\n{listing}"
        )
    if not candidates:
        raise ValueError(
            f"No sensor_msgs Image/CompressedImage topic found in {bag_path}. "
            "Only raw and jpeg/png-compressed image topics are supported."
        )
    if len(candidates) > 1:
        listing = _format_topic_listing(candidates)
        raise ValueError(f"Multiple image topics found in {bag_path}; pick one with --image-topic:\n{listing}")
    return candidates[0]


def _format_topic_listing(candidates: list[BagImageTopic]) -> str:
    if not candidates:
        return "  (none)"
    return "\n".join(f"  {c.topic}  [{c.msgtype}, {c.msgcount} msgs]" for c in candidates)


# ------------------------------------------------------------------- decoding


def decode_image_message(msg: Any, msgtype: str) -> tuple[np.ndarray | None, str]:
    """Decode a ROS Image/CompressedImage message into an OpenCV image."""
    if msgtype in COMPRESSED_IMAGE_MSGTYPES:
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, ".jpg"
        if img.ndim == 2:
            return img, ".png"
        return img, ".jpg"

    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(getattr(msg, "step", 0))
    data = bytes(msg.data)

    if encoding == "rgb8":
        image = reshape_image_buffer(data, height, width, np.uint8, 3, step)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), ".jpg"
    if encoding == "bgr8":
        return reshape_image_buffer(data, height, width, np.uint8, 3, step), ".jpg"
    if encoding in {"mono8", "8uc1"}:
        return reshape_image_buffer(data, height, width, np.uint8, 1, step), ".png"
    if encoding in {"mono16", "16uc1"}:
        image16 = reshape_image_buffer(data, height, width, np.uint16, 1, step)
        scale = 255.0 / max(1.0, float(np.max(image16)))
        return cv2.convertScaleAbs(image16, alpha=scale), ".png"

    # Fall back to raw bytes interpreted as BGR8, which is a common case.
    if len(data) == height * width * 3:
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), ".jpg"
    return None, ".png"


def reshape_image_buffer(
    data: bytes,
    height: int,
    width: int,
    dtype: type[np.generic],
    channels: int,
    step: int,
) -> np.ndarray:
    """Reshape image bytes using the ROS step size when present."""
    itemsize = np.dtype(dtype).itemsize
    row_items = width * channels if step <= 0 else step // itemsize
    array = np.frombuffer(data, dtype=dtype).reshape(height, row_items)
    array = array[:, : width * channels]
    if channels == 1:
        return array.reshape(height, width)
    return array.reshape(height, width, channels)


# ----------------------------------------------------------------- iteration


@dataclass
class BagFrame:
    """One decoded camera frame from a bag."""

    index: int
    timestamp_sec: float
    image_bgr: np.ndarray
    topic: str
    extension: str


def iter_bag_frames(
    bag_path: str | Path,
    image_topic: str | None = None,
    *,
    every_n: int = 1,
    max_frames: int | None = None,
    start_offset_sec: float = 0.0,
) -> Iterator[BagFrame]:
    """Yield decoded frames from one image topic in bag-timestamp order.

    ``every_n`` subsamples messages, ``max_frames`` caps the yield count, and
    ``start_offset_sec`` skips the topic's initial seconds. Undecodable
    messages are skipped with a warning.
    """
    selected = resolve_image_topic(bag_path, image_topic)
    bag_paths = find_bag_paths(bag_path)
    reader_cls = get_anyreader()
    yielded = 0
    seen = 0
    first_ts: float | None = None
    with create_reader(reader_cls, bag_paths) as reader:
        connections = [
            c
            for info in reader.topics.values()
            for c in info.connections
            if c.topic == selected.topic and c.msgtype in IMAGE_MSGTYPES
        ]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            ts_sec = float(timestamp_ns) * 1e-9
            if first_ts is None:
                first_ts = ts_sec
            if start_offset_sec > 0.0 and ts_sec - first_ts < start_offset_sec:
                continue
            idx = seen
            seen += 1
            if idx % max(every_n, 1) != 0:
                continue
            if max_frames is not None and yielded >= max_frames:
                break
            msg = reader.deserialize(rawdata, connection.msgtype)
            image, extension = decode_image_message(msg, connection.msgtype)
            if image is None:
                logger.warning("skipping undecodable %s message at %.3fs", connection.msgtype, ts_sec)
                continue
            yield BagFrame(
                index=yielded,
                timestamp_sec=ts_sec,
                image_bgr=image,
                topic=connection.topic,
                extension=extension,
            )
            yielded += 1


def plan_bag_frame_sampling(msgcount: int, target_frames: int) -> int:
    """Return the ``every_n`` stride that lands near ``target_frames`` frames."""
    if target_frames <= 0 or msgcount <= target_frames:
        return 1
    return max(1, msgcount // target_frames)


def extract_bag_frames(
    bag_path: str | Path,
    output_dir: str | Path,
    image_topic: str | None = None,
    *,
    target_frames: int = 32,
    every_n: int | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract frames from a bag into ``output_dir`` and return the written paths.

    Without an explicit ``every_n``, the stride is derived from the selected
    topic's message count so roughly ``target_frames`` frames come out (same
    auto-sampling idea as the video one-liner).
    """
    selected = resolve_image_topic(bag_path, image_topic)
    if every_n is None:
        every_n = plan_bag_frame_sampling(selected.msgcount, target_frames)
    if max_frames is None and target_frames > 0:
        max_frames = target_frames

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for frame in iter_bag_frames(
        bag_path,
        selected.topic,
        every_n=every_n,
        max_frames=max_frames,
    ):
        path = out / f"frame_{frame.index:06d}{frame.extension}"
        cv2.imwrite(str(path), frame.image_bgr)
        written.append(path)
    logger.info(
        "extracted %d frames from %s (%s, every_n=%d)",
        len(written),
        bag_path,
        selected.topic,
        every_n,
    )
    return written
