"""MCD rosbag loader for outdoor 3DGS preprocessing.

Provides utilities to extract camera frames and selected sensor streams from
MCD rosbag recordings using ``rosbags``. The loader also falls back to
pre-extracted images when bag files are unavailable.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gs_sim2real.datasets import rosbag_frames

logger = logging.getLogger(__name__)


def _apply_antenna_offset_base_link(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    offset_base: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract ``R_enu_base @ offset_base`` from antenna positions in ENU (base_link x=forward, y=left, z=up).

    Heading for each sample is inferred from successive ENU horizontal differences (GNSS course proxy).
    """
    ox, oy, oz = (float(offset_base[0]), float(offset_base[1]), float(offset_base[2]))
    v = np.array([ox, oy, oz], dtype=np.float64)
    n = len(east)
    out_e = east.astype(np.float64).copy()
    out_n = north.astype(np.float64).copy()
    out_u = up.astype(np.float64).copy()

    for i in range(n):
        if i > 0:
            d_e = float(east[i] - east[i - 1])
            d_n = float(north[i] - north[i - 1])
        elif n > 1:
            d_e = float(east[1] - east[0])
            d_n = float(north[1] - north[0])
        else:
            d_e, d_n = 1.0, 0.0
        norm = float(np.hypot(d_e, d_n))
        if norm < 1e-6:
            d_e, d_n = 1.0, 0.0
            norm = 1.0
        f = np.array([d_e / norm, d_n / norm, 0.0], dtype=np.float64)
        left = np.array([-f[1], f[0], 0.0], dtype=np.float64)
        fz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        r_enu_base = np.column_stack([f, left, fz])
        delta = r_enu_base @ v
        out_e[i] -= delta[0]
        out_n[i] -= delta[1]
        out_u[i] -= delta[2]

    return out_e, out_n, out_u


class MCDLoader:
    """Load and process MCD rosbag recordings for 3DGS reconstruction."""

    IMAGE_MSGTYPES = rosbag_frames.IMAGE_MSGTYPES
    COMPRESSED_IMAGE_MSGTYPES = rosbag_frames.COMPRESSED_IMAGE_MSGTYPES
    POINTCLOUD_MSGTYPES = frozenset(
        {
            "sensor_msgs/msg/PointCloud2",
            "sensor_msgs/PointCloud2",
        }
    )
    IMU_MSGTYPES = frozenset(
        {
            "sensor_msgs/msg/Imu",
            "sensor_msgs/Imu",
        }
    )
    NAVSAT_MSGTYPES = frozenset(
        {
            "sensor_msgs/msg/NavSatFix",
            "sensor_msgs/NavSatFix",
        }
    )
    CAMERA_INFO_MSGTYPES = frozenset(
        {
            "sensor_msgs/msg/CameraInfo",
            "sensor_msgs/CameraInfo",
        }
    )
    TF_MESSAGE_MSGTYPES = frozenset(
        {
            "tf2_msgs/msg/TFMessage",
            "tf2_msgs/TFMessage",
            "tf/msg/tfMessage",
            "tf/tfMessage",
        }
    )

    DEFAULT_IMAGE_TOPICS = (
        "/d455t/color/image_raw",
        "/d455b/color/image_raw",
        "/d435i/color/image_raw",
        "/d455t/infra1/image_rect_raw",
        "/d455b/infra1/image_rect_raw",
        "/d435i/infra1/image_rect_raw",
    )
    DEFAULT_LIDAR_TOPICS = (
        "/os_cloud_node/points",
        "/livox/lidar",
    )
    DEFAULT_IMU_TOPICS = (
        "/vn200/imu",
        "/vn100/imu",
        "/os_cloud_node/imu",
        "/d455t/imu",
        "/d455b/imu",
        "/d435i/imu",
    )
    DEFAULT_GNSS_TOPICS = (
        "/vn200/GPS",
        "/vn100/GPS",
        "/vn200/gps",
        "/vn100/gps",
        "/gnss/fix",
        "/gps/fix",
        "/fix",
    )
    DEFAULT_TF_STATIC_TOPICS = ("/tf_static",)

    POINTFIELD_DTYPES = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        8: np.float64,
    }

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def extract_frames(
        self,
        output_dir: str | Path,
        image_topic: str | list[str] | tuple[str, ...] | None = None,
        max_frames: int = 100,
        every_n: int = 1,
        save_image_timestamps: bool = False,
        start_offset_sec: float = 0.0,
    ) -> str:
        """Extract camera frames from an MCD rosbag or pre-extracted images."""
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            logger.info("No rosbag found in %s; falling back to pre-extracted images", self.data_dir)
            return self._load_pre_extracted(output_path, max_frames)

        try:
            reader_cls = self._get_anyreader()
        except ImportError:
            logger.warning("rosbags not available; falling back to pre-extracted images")
            return self._load_pre_extracted(output_path, max_frames)

        requested_topics = self._normalize_requested_topics(image_topic)
        with self._create_reader(reader_cls, bag_paths) as reader:
            connections = self._select_connections(
                reader.topics,
                requested_topics=requested_topics,
                preferred_topics=self.DEFAULT_IMAGE_TOPICS,
                allowed_msgtypes=self.IMAGE_MSGTYPES,
            )
            if not connections:
                raise FileNotFoundError("No supported image topic found in the provided MCD rosbag.")

            multi_topic = len(connections) > 1
            write_dirs = {
                connection.topic: (
                    images_dir / self._sanitize_topic_name(connection.topic) if multi_topic else images_dir
                )
                for connection in connections
            }
            for path in write_dirs.values():
                path.mkdir(parents=True, exist_ok=True)

            extracted_counts = {connection.topic: 0 for connection in connections}
            seen_counts = {connection.topic: 0 for connection in connections}
            first_timestamps = {connection.topic: None for connection in connections}
            timestamp_rows: list[tuple[str, int]] = []

            for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                topic_name = connection.topic
                ts_sec = float(timestamp_ns) * 1e-9
                if first_timestamps[topic_name] is None:
                    first_timestamps[topic_name] = ts_sec
                if start_offset_sec > 0.0 and ts_sec - first_timestamps[topic_name] < start_offset_sec:
                    continue
                idx = seen_counts[topic_name]
                seen_counts[topic_name] += 1
                if idx % every_n != 0:
                    continue
                if extracted_counts[topic_name] >= max_frames:
                    if all(count >= max_frames for count in extracted_counts.values()):
                        break
                    continue

                msg = reader.deserialize(rawdata, connection.msgtype)
                image, extension = self._decode_image_message(msg, connection.msgtype)
                if image is None:
                    continue

                output_image = write_dirs[topic_name] / f"frame_{extracted_counts[topic_name]:06d}{extension}"
                cv2.imwrite(str(output_image), image)
                if save_image_timestamps:
                    rel = output_image.relative_to(images_dir).as_posix()
                    timestamp_rows.append((rel, int(timestamp_ns)))
                extracted_counts[topic_name] += 1

            if save_image_timestamps and timestamp_rows:
                ts_path = images_dir / "image_timestamps.csv"
                with open(ts_path, "w", newline="") as tf:
                    writer = csv.writer(tf)
                    writer.writerow(["filename", "timestamp_ns"])
                    writer.writerows(timestamp_rows)
                logger.info("Wrote %d image timestamps to %s", len(timestamp_rows), ts_path)

        for topic_name, count in extracted_counts.items():
            logger.info("Extracted %d frames from MCD topic %s", count, topic_name)
        return str(images_dir)

    def list_topics(self) -> list[dict[str, str | int | bool]]:
        """List bag topics with msg types, counts, and inferred sensor role."""
        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            return []

        reader_cls = self._get_anyreader()
        with self._create_reader(reader_cls, bag_paths) as reader:
            topics = []
            for topic_name, info in sorted(reader.topics.items()):
                msgtype = info.msgtype or "unknown"
                topics.append(
                    {
                        "topic": topic_name,
                        "msgtype": msgtype,
                        "msgcount": int(info.msgcount),
                        "role": self._infer_topic_role(topic_name, msgtype),
                        "is_preferred_default": self._is_preferred_default_topic(topic_name),
                    }
                )
            return topics

    def extract_lidar(
        self,
        output_dir: str | Path,
        lidar_topic: str | None = None,
        max_frames: int = 100,
        every_n: int = 1,
        save_timestamps: bool = False,
        start_offset_sec: float = 0.0,
    ) -> str:
        """Extract LiDAR point clouds from an MCD rosbag as ``.npy`` arrays."""
        output_path = Path(output_dir)
        lidar_dir = output_path / "lidar"
        lidar_dir.mkdir(parents=True, exist_ok=True)

        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            logger.warning("No rosbag found in %s; cannot extract LiDAR", self.data_dir)
            return str(lidar_dir)

        try:
            reader_cls = self._get_anyreader()
        except ImportError:
            logger.warning("rosbags not available; cannot extract LiDAR")
            return str(lidar_dir)
        count = 0
        timestamp_rows: list[tuple[str, int]] = []
        with self._create_reader(reader_cls, bag_paths) as reader:
            connection = self._select_connection(
                reader.topics,
                requested_topic=lidar_topic,
                preferred_topics=self.DEFAULT_LIDAR_TOPICS,
                allowed_msgtypes=self.POINTCLOUD_MSGTYPES,
            )
            if connection is None:
                raise FileNotFoundError("No supported PointCloud2 topic found in the provided MCD rosbag.")

            seen_count = 0
            first_ts: float | None = None
            for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
                ts_sec = float(timestamp_ns) * 1e-9
                if first_ts is None:
                    first_ts = ts_sec
                if start_offset_sec > 0.0 and ts_sec - first_ts < start_offset_sec:
                    continue
                idx = seen_count
                seen_count += 1
                if idx % every_n != 0:
                    continue
                if count >= max_frames:
                    break

                msg = reader.deserialize(rawdata, connection.msgtype)
                points = self._pointcloud2_to_numpy(msg)
                filename = f"frame_{count:06d}.npy"
                np.save(lidar_dir / filename, points)
                if save_timestamps:
                    timestamp_rows.append((filename, int(timestamp_ns)))
                count += 1

        if save_timestamps and timestamp_rows:
            ts_path = lidar_dir / "timestamps.csv"
            with open(ts_path, "w", newline="") as tf:
                writer = csv.writer(tf)
                writer.writerow(["filename", "timestamp_ns"])
                writer.writerows(timestamp_rows)
            logger.info("Wrote %d LiDAR timestamps to %s", len(timestamp_rows), ts_path)

        logger.info("Extracted %d LiDAR frames from MCD topic %s", count, connection.topic)
        return str(lidar_dir)

    def merge_lidar_frames_to_world(
        self,
        lidar_dir: str | Path,
        trajectory_path: str | Path,
        output_path: str | Path,
        T_base_lidar: np.ndarray | None = None,
        max_points: int = 200_000,
        rng_seed: int = 42,
    ) -> str:
        """Merge per-frame LiDAR NPYs into a single world-frame ``.npy`` using a TUM ENU trajectory."""
        from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor

        lidar_path = Path(lidar_dir)
        traj_path = Path(trajectory_path)
        out_path = Path(output_path)

        npy_files = sorted(lidar_path.glob("frame_*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No LiDAR frames found in {lidar_path}")

        ts_csv = lidar_path / "timestamps.csv"
        stamp_map: dict[str, int] = {}
        if ts_csv.exists():
            with open(ts_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stamp_map[str(row["filename"])] = int(row["timestamp_ns"])

        timestamps, poses = LiDARSLAMProcessor._load_tum_trajectory(LiDARSLAMProcessor(), traj_path)
        if len(timestamps) < 1:
            raise ValueError(f"Trajectory {traj_path} has no poses")

        ts_array = np.array(timestamps, dtype=np.float64)
        poses_arr = [np.asarray(p, dtype=np.float64) for p in poses]

        rotations = self._infer_trajectory_rotations(ts_array, poses_arr)

        if stamp_map:
            frame_times = np.array(
                [float(stamp_map.get(p.name, 0)) * 1e-9 for p in npy_files],
                dtype=np.float64,
            )
        else:
            logger.warning(
                "No timestamps.csv in %s; falling back to sequential time mapping across trajectory range",
                lidar_path,
            )
            if len(npy_files) == 1:
                frame_times = np.array([ts_array[0]], dtype=np.float64)
            else:
                frame_times = np.linspace(ts_array[0], ts_array[-1], num=len(npy_files))

        chunks: list[np.ndarray] = []
        for idx, npy_file in enumerate(npy_files):
            pts = np.load(npy_file)
            if pts.ndim != 2 or pts.shape[1] < 3:
                continue
            xyz = pts[:, :3].astype(np.float64, copy=False)
            mask = np.isfinite(xyz).all(axis=1)
            xyz = xyz[mask]
            if xyz.shape[0] == 0:
                continue

            nearest = int(np.argmin(np.abs(ts_array - frame_times[idx])))
            T_world_base = np.eye(4, dtype=np.float64)
            T_world_base[:3, :3] = rotations[nearest]
            T_world_base[:3, 3] = poses_arr[nearest][:3, 3]
            T_world_lidar = T_world_base if T_base_lidar is None else T_world_base @ np.asarray(T_base_lidar)

            R = T_world_lidar[:3, :3]
            t = T_world_lidar[:3, 3]
            transformed = xyz @ R.T + t
            chunks.append(transformed.astype(np.float32, copy=False))

        if not chunks:
            raise ValueError(f"No valid LiDAR points after filtering in {lidar_path}")

        all_pts = np.concatenate(chunks, axis=0)
        finite = np.isfinite(all_pts).all(axis=1)
        all_pts = all_pts[finite]

        if all_pts.shape[0] > max_points:
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(all_pts.shape[0], size=max_points, replace=False)
            all_pts = all_pts[idx]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, all_pts.astype(np.float32, copy=False))
        logger.info("Merged %d LiDAR frames -> %d points at %s", len(npy_files), all_pts.shape[0], out_path)
        return str(out_path)

    @staticmethod
    def _infer_trajectory_rotations(timestamps: np.ndarray, poses: list[np.ndarray]) -> list[np.ndarray]:
        """Return per-pose rotations, replacing identity orientations with yaw inferred from ENU motion."""
        del timestamps
        n = len(poses)
        positions = np.array([p[:3, 3] for p in poses], dtype=np.float64)
        rotations: list[np.ndarray] = []
        is_identity = [bool(np.allclose(p[:3, :3], np.eye(3), atol=1e-6)) for p in poses]
        if not any(is_identity):
            return [p[:3, :3].copy() for p in poses]

        yaws = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if 0 < i < n - 1:
                de = positions[i + 1, 0] - positions[i - 1, 0]
                dn = positions[i + 1, 1] - positions[i - 1, 1]
            elif i == 0 and n > 1:
                de = positions[1, 0] - positions[0, 0]
                dn = positions[1, 1] - positions[0, 1]
            elif i == n - 1 and n > 1:
                de = positions[-1, 0] - positions[-2, 0]
                dn = positions[-1, 1] - positions[-2, 1]
            else:
                de, dn = 1.0, 0.0
            if float(np.hypot(de, dn)) < 1e-9:
                yaws[i] = np.nan
            else:
                yaws[i] = float(np.arctan2(dn, de))

        if np.all(np.isnan(yaws)):
            yaws[:] = 0.0
        else:
            valid = ~np.isnan(yaws)
            first_valid = int(np.argmax(valid))
            last_valid = n - 1 - int(np.argmax(valid[::-1]))
            for i in range(n):
                if np.isnan(yaws[i]):
                    if i < first_valid:
                        yaws[i] = yaws[first_valid]
                    elif i > last_valid:
                        yaws[i] = yaws[last_valid]
                    else:
                        yaws[i] = yaws[first_valid]

        for i, pose in enumerate(poses):
            if is_identity[i]:
                c, s = float(np.cos(yaws[i])), float(np.sin(yaws[i]))
                R = np.array(
                    [
                        [c, -s, 0.0],
                        [s, c, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                rotations.append(R)
            else:
                rotations.append(pose[:3, :3].copy())
        return rotations

    def colorize_lidar_world_from_images(
        self,
        lidar_world_xyz: np.ndarray,
        images_root: str | Path,
        trajectory_path: str | Path,
        cameras: list[dict],
        hybrid_tf: Any | None = None,
        base_frame: str = "base_link",
        stamp_tolerance_sec: float = 0.25,
    ) -> np.ndarray:
        """Return per-point uint8 RGB sampled from images that see each world point.

        For each camera / each image, projects world points into the camera via
        the vehicle trajectory (TUM) + per-image TF extrinsics (HybridTfLookup)
        or a constant ``T_base_cam`` fallback per camera. Accumulates the
        bilinearly-sampled RGB at valid projections and averages, falling back
        to 128 grey for points no camera ever saw.

        Args:
            lidar_world_xyz: (N, 3) ENU world positions.
            images_root: directory with ``<subdir>/frame_*.jpg`` and
                ``image_timestamps.csv``.
            trajectory_path: vehicle-frame TUM file (position + identity or
                inferred orientation).
            cameras: list of dicts with ``subdir``, ``camera_frame``,
                ``pinhole=(fx, fy, cx, cy, w, h)`` and optional ``T_base_cam``.
            hybrid_tf: optional :class:`HybridTfLookup`; if given, per-image
                ``base_frame → camera_frame`` is queried at the frame timestamp.
            base_frame: TF parent (default ``base_link``).
            stamp_tolerance_sec: skip frames whose trajectory match is farther
                than this in seconds.
        """
        from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor

        xyz = np.asarray(lidar_world_xyz, dtype=np.float64)
        n_pts = xyz.shape[0]
        if n_pts == 0:
            return np.full((0, 3), 128, dtype=np.uint8)

        images_root_p = Path(images_root)
        ts_csv = images_root_p / "image_timestamps.csv"
        if not ts_csv.is_file():
            logger.warning("No image_timestamps.csv in %s; returning grey colors", images_root_p)
            return np.full((n_pts, 3), 128, dtype=np.uint8)
        stamp_map: dict[str, float] = {}
        with open(ts_csv, newline="") as f:
            for row in csv.DictReader(f):
                stamp_map[row["filename"].strip()] = float(row["timestamp_ns"]) * 1e-9

        proc = LiDARSLAMProcessor()
        timestamps, poses = proc._load_tum_trajectory(Path(trajectory_path))
        if len(timestamps) < 1:
            logger.warning("Empty trajectory %s; returning grey colors", trajectory_path)
            return np.full((n_pts, 3), 128, dtype=np.uint8)
        rotations = self._infer_trajectory_rotations(np.asarray(timestamps), poses)
        ts_array = np.asarray(timestamps, dtype=np.float64)
        poses_arr = [np.asarray(p, dtype=np.float64) for p in poses]

        xyz_hom = np.concatenate([xyz, np.ones((n_pts, 1), dtype=np.float64)], axis=1)

        color_sum = np.zeros((n_pts, 3), dtype=np.float64)
        color_count = np.zeros((n_pts,), dtype=np.int32)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        total_visits = 0
        for cam in cameras:
            subdir = str(cam.get("subdir") or "")
            pinhole = cam.get("pinhole")
            if not pinhole:
                continue
            fx, fy, cx, cy, w_img, h_img = pinhole
            camera_frame = str(cam.get("camera_frame") or "").strip()
            T_base_cam_const = cam.get("T_base_cam")
            if T_base_cam_const is not None:
                T_base_cam_const = np.asarray(T_base_cam_const, dtype=np.float64)

            folder = images_root_p / subdir if subdir else images_root_p
            if not folder.is_dir():
                continue
            cam_images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
            for img_path in cam_images:
                rel = img_path.relative_to(images_root_p).as_posix()
                if rel not in stamp_map:
                    continue
                img_ts = stamp_map[rel]
                match_idx = int(np.argmin(np.abs(ts_array - img_ts)))
                if abs(ts_array[match_idx] - img_ts) > stamp_tolerance_sec:
                    continue
                T_world_base = np.eye(4, dtype=np.float64)
                T_world_base[:3, :3] = rotations[match_idx]
                T_world_base[:3, 3] = poses_arr[match_idx][:3, 3]
                T_base_cam = None
                if hybrid_tf is not None and camera_frame:
                    T_base_cam = hybrid_tf.lookup(base_frame, camera_frame, int(round(img_ts * 1e9)))
                if T_base_cam is None:
                    T_base_cam = T_base_cam_const
                if T_base_cam is None:
                    continue
                T_world_cam = T_world_base @ np.asarray(T_base_cam, dtype=np.float64)
                T_cam_world = np.linalg.inv(T_world_cam)
                cam_pts = xyz_hom @ T_cam_world.T  # (N, 4)
                x = cam_pts[:, 0]
                y = cam_pts[:, 1]
                z = cam_pts[:, 2]
                valid_z = z > 1e-3
                if not np.any(valid_z):
                    continue
                u = fx * x / np.where(valid_z, z, 1.0) + cx
                v = fy * y / np.where(valid_z, z, 1.0) + cy
                in_bounds = valid_z & (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
                if not np.any(in_bounds):
                    continue
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                if img_bgr.shape[0] != int(h_img) or img_bgr.shape[1] != int(w_img):
                    img_bgr = cv2.resize(img_bgr, (int(w_img), int(h_img)))
                idx = np.nonzero(in_bounds)[0]
                ui = u[idx].astype(np.int32)
                vi = v[idx].astype(np.int32)
                rgb = img_bgr[vi, ui][:, ::-1]  # BGR -> RGB
                color_sum[idx] += rgb.astype(np.float64)
                color_count[idx] += 1
                total_visits += int(idx.size)

        logger.info(
            "colorize_lidar_world: %d points, %d projections, %d covered",
            n_pts,
            total_visits,
            int((color_count > 0).sum()),
        )

        rgb_out = np.full((n_pts, 3), 128, dtype=np.uint8)
        covered = color_count > 0
        if np.any(covered):
            avg = (color_sum[covered] / color_count[covered, None]).clip(0, 255).astype(np.uint8)
            rgb_out[covered] = avg
        return rgb_out

    def export_lidar_depth_per_image(
        self,
        lidar_world_xyz: np.ndarray,
        images_root: str | Path,
        trajectory_path: str | Path,
        cameras: list[dict],
        output_dir: str | Path,
        hybrid_tf: Any | None = None,
        base_frame: str = "base_link",
        stamp_tolerance_sec: float = 0.25,
    ) -> int:
        """Project world LiDAR points into each training image and save sparse depth maps.

        For each camera frame, computes ``T_cam_world`` (via HybridTfLookup or a
        constant ``T_base_cam``), projects every world LiDAR point, and keeps
        the closest depth per pixel. The depth map is stored as float32
        ``(H, W)`` numpy under ``<output_dir>/<subdir>/<stem>.npy`` with zeros
        where no LiDAR point projected (i.e. ``depth > 0`` marks valid pixels).

        Returns the number of images that received a valid depth map.
        """
        from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor

        xyz = np.asarray(lidar_world_xyz, dtype=np.float64)
        if xyz.shape[0] == 0:
            return 0

        images_root_p = Path(images_root)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        ts_csv = images_root_p / "image_timestamps.csv"
        if not ts_csv.is_file():
            logger.warning("No image_timestamps.csv in %s; cannot export per-image depth", images_root_p)
            return 0
        stamp_map: dict[str, float] = {}
        with open(ts_csv, newline="") as f:
            for row in csv.DictReader(f):
                stamp_map[row["filename"].strip()] = float(row["timestamp_ns"]) * 1e-9

        proc = LiDARSLAMProcessor()
        timestamps, poses = proc._load_tum_trajectory(Path(trajectory_path))
        if not timestamps:
            return 0
        rotations = self._infer_trajectory_rotations(np.asarray(timestamps), poses)
        ts_array = np.asarray(timestamps, dtype=np.float64)
        poses_arr = [np.asarray(p, dtype=np.float64) for p in poses]
        xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        count_written = 0
        for cam in cameras:
            subdir = str(cam.get("subdir") or "")
            pinhole = cam.get("pinhole")
            if not pinhole:
                continue
            fx, fy, cx, cy, w_img, h_img = pinhole
            w_img = int(w_img)
            h_img = int(h_img)
            camera_frame = str(cam.get("camera_frame") or "").strip()
            T_base_cam_const = cam.get("T_base_cam")
            if T_base_cam_const is not None:
                T_base_cam_const = np.asarray(T_base_cam_const, dtype=np.float64)

            folder = images_root_p / subdir if subdir else images_root_p
            if not folder.is_dir():
                continue
            out_sub = out_root / subdir if subdir else out_root
            out_sub.mkdir(parents=True, exist_ok=True)

            cam_images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
            for img_path in cam_images:
                rel = img_path.relative_to(images_root_p).as_posix()
                if rel not in stamp_map:
                    continue
                img_ts = stamp_map[rel]
                match_idx = int(np.argmin(np.abs(ts_array - img_ts)))
                if abs(ts_array[match_idx] - img_ts) > stamp_tolerance_sec:
                    continue
                T_world_base = np.eye(4, dtype=np.float64)
                T_world_base[:3, :3] = rotations[match_idx]
                T_world_base[:3, 3] = poses_arr[match_idx][:3, 3]
                T_base_cam = None
                if hybrid_tf is not None and camera_frame:
                    T_base_cam = hybrid_tf.lookup(base_frame, camera_frame, int(round(img_ts * 1e9)))
                if T_base_cam is None:
                    T_base_cam = T_base_cam_const
                if T_base_cam is None:
                    continue
                T_cam_world = np.linalg.inv(T_world_base @ np.asarray(T_base_cam, dtype=np.float64))
                cam_pts = xyz_hom @ T_cam_world.T
                z = cam_pts[:, 2]
                valid_z = z > 1e-3
                if not np.any(valid_z):
                    continue
                u = fx * cam_pts[:, 0] / np.where(valid_z, z, 1.0) + cx
                v = fy * cam_pts[:, 1] / np.where(valid_z, z, 1.0) + cy
                in_bounds = valid_z & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
                if not np.any(in_bounds):
                    continue
                ui = u[in_bounds].astype(np.int32)
                vi = v[in_bounds].astype(np.int32)
                zi = z[in_bounds].astype(np.float32)
                depth = np.zeros((h_img, w_img), dtype=np.float32)
                # Keep the closest depth per pixel.
                flat_idx = vi * w_img + ui
                order = np.argsort(-zi)  # write furthest first so nearest overwrites
                depth.reshape(-1)[flat_idx[order]] = zi[order]
                out_path = out_sub / f"{img_path.stem}.npy"
                np.save(out_path, depth)
                count_written += 1

        logger.info("export_lidar_depth_per_image: wrote %d depth maps under %s", count_written, out_root)
        return count_written

    def extract_imu(
        self,
        output_dir: str | Path,
        imu_topic: str | None = None,
        max_messages: int | None = None,
    ) -> str:
        """Extract IMU measurements from an MCD rosbag into ``imu.csv``."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        imu_path = output_path / "imu.csv"

        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            logger.warning("No rosbag found in %s; cannot extract IMU", self.data_dir)
            return str(imu_path)

        try:
            reader_cls = self._get_anyreader()
        except ImportError:
            logger.warning("rosbags not available; cannot extract IMU")
            return str(imu_path)
        with self._create_reader(reader_cls, bag_paths) as reader, open(imu_path, "w", newline="") as f:
            connection = self._select_connection(
                reader.topics,
                requested_topic=imu_topic,
                preferred_topics=self.DEFAULT_IMU_TOPICS,
                allowed_msgtypes=self.IMU_MSGTYPES,
            )
            if connection is None:
                raise FileNotFoundError("No supported IMU topic found in the provided MCD rosbag.")

            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_ns",
                    "timestamp_sec",
                    "orientation_x",
                    "orientation_y",
                    "orientation_z",
                    "orientation_w",
                    "angular_velocity_x",
                    "angular_velocity_y",
                    "angular_velocity_z",
                    "linear_acceleration_x",
                    "linear_acceleration_y",
                    "linear_acceleration_z",
                ]
            )

            count = 0
            for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
                if max_messages is not None and count >= max_messages:
                    break
                msg = reader.deserialize(rawdata, connection.msgtype)
                writer.writerow(
                    [
                        int(timestamp_ns),
                        float(timestamp_ns) * 1e-9,
                        float(msg.orientation.x),
                        float(msg.orientation.y),
                        float(msg.orientation.z),
                        float(msg.orientation.w),
                        float(msg.angular_velocity.x),
                        float(msg.angular_velocity.y),
                        float(msg.angular_velocity.z),
                        float(msg.linear_acceleration.x),
                        float(msg.linear_acceleration.y),
                        float(msg.linear_acceleration.z),
                    ]
                )
                count += 1

        logger.info("Extracted %d IMU measurements from MCD topic %s", count, connection.topic)
        return str(imu_path)

    def extract_navsat_trajectory(
        self,
        output_dir: str | Path,
        gnss_topic: str | None = None,
        max_poses: int | None = None,
        T_base_cam: np.ndarray | None = None,
        vehicle_frame_only: bool = False,
        antenna_offset_enu: tuple[float, float, float] | None = None,
        antenna_offset_base: tuple[float, float, float] | None = None,
        reference_origin: tuple[float, float, float] | None = None,
        imu_csv_path: str | Path | None = None,
        flatten_altitude: bool = False,
        start_offset_sec: float = 0.0,
    ) -> str:
        """Write ``sensor_msgs/NavSatFix`` samples to a TUM trajectory file (local ENU).

        The first valid fix defines the ENU origin. Without ``T_base_cam``, orientation is identity
        (vehicle / antenna frame). With ``T_base_cam`` (base_link→camera from ``/tf_static``),
        each line is the **camera** pose (camera-to-world) in ENU.

        ``antenna_offset_enu`` subtracts a fixed (East, North, Up) vector (metres).

        ``antenna_offset_base`` is the vector from ``base_link`` origin to the GNSS antenna in **base_link**
        (x forward, y left, z up). It is rotated into ENU using a per-sample heading from successive ENU
        positions, then subtracted. Do not combine with ``antenna_offset_enu``.

        ``reference_origin = (lat, lon, alt)`` overrides the automatic "first fix" origin so multiple
        bags can be expressed in the same ENU frame. Also writes ``pose/origin_wgs84.json`` with the
        resolved origin for downstream consumers.

        ``imu_csv_path`` points at the CSV written by :meth:`extract_imu`. When
        provided, the orientation column is used as each TUM entry's quaternion
        (linearly interpolated to the GNSS timestamp) instead of the default identity.
        If the orientation column is unavailable or all identity, non-zero
        ``angular_velocity_z`` is integrated as a yaw-only fallback. This improves
        robustness for stationary segments and sharp turns.

        Some public MCD bags publish placeholder NavSatFix messages at
        latitude=longitude=0. Treat those as missing fixes so pose-seeded
        preprocessing fails instead of producing a static ENU trajectory.

        ``flatten_altitude`` projects all fixes to the median valid altitude
        before ENU conversion. This is useful for VectorNav bags whose first
        seconds contain an implausible altitude warm-up spike while horizontal
        latitude/longitude are already usable.

        ``start_offset_sec`` skips the selected GNSS topic's initial seconds.
        Use the same offset for image/LiDAR extraction when trimming sensor
        warm-up.
        """
        from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor

        if antenna_offset_enu is not None and antenna_offset_base is not None:
            raise ValueError("Use only one of antenna_offset_enu or antenna_offset_base, not both.")

        output_path = Path(output_dir)
        pose_dir = output_path / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        tum_path = pose_dir / "gnss_trajectory.tum"

        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            raise FileNotFoundError(f"No rosbag found in {self.data_dir}; cannot extract NavSatFix trajectory.")

        reader_cls = self._get_anyreader()
        valid_rows: list[tuple[float, float, float, float]] = []
        connection = None

        with self._create_reader(reader_cls, bag_paths) as reader:
            connection = self._select_connection(
                reader.topics,
                requested_topic=gnss_topic,
                preferred_topics=self.DEFAULT_GNSS_TOPICS,
                allowed_msgtypes=self.NAVSAT_MSGTYPES,
            )
            if connection is None:
                raise FileNotFoundError("No sensor_msgs/NavSatFix topic found in the provided MCD rosbag.")

            first_ts: float | None = None
            for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
                ts = float(timestamp_ns) * 1e-9
                if first_ts is None:
                    first_ts = ts
                if start_offset_sec > 0.0 and ts - first_ts < start_offset_sec:
                    continue
                if max_poses is not None and len(valid_rows) >= max_poses:
                    break
                msg = reader.deserialize(rawdata, connection.msgtype)
                lat = float(getattr(msg, "latitude", float("nan")))
                lon = float(getattr(msg, "longitude", float("nan")))
                alt = float(getattr(msg, "altitude", 0.0))
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                if abs(lat) < 1e-12 and abs(lon) < 1e-12:
                    continue
                st = getattr(msg, "status", None)
                if st is not None:
                    code = getattr(st, "status", st)
                    try:
                        if int(code) < 0:
                            continue
                    except (TypeError, ValueError):
                        pass

                valid_rows.append((ts, lat, lon, alt))

        count = len(valid_rows)
        if count < 2:
            raise ValueError(f"Need at least 2 NavSatFix samples for a trajectory, got {count} from {connection.topic}")

        if reference_origin is not None:
            ref_lat, ref_lon, ref_alt = (float(x) for x in reference_origin)
        else:
            ref_lat, ref_lon, ref_alt = valid_rows[0][1], valid_rows[0][2], valid_rows[0][3]
            if flatten_altitude:
                ref_alt = float(np.median([row[3] for row in valid_rows]))

        rows_ts: list[float] = []
        rows_e: list[float] = []
        rows_n: list[float] = []
        rows_u: list[float] = []
        for ts, lat, lon, alt in valid_rows:
            alt_for_enu = ref_alt if flatten_altitude else alt
            east, north, up = LiDARSLAMProcessor._wgs84_to_enu(lat, lon, alt_for_enu, ref_lat, ref_lon, ref_alt)
            rows_ts.append(ts)
            rows_e.append(float(east))
            rows_n.append(float(north))
            rows_u.append(float(up))

        east_a = np.array(rows_e, dtype=np.float64)
        north_a = np.array(rows_n, dtype=np.float64)
        up_a = np.array(rows_u, dtype=np.float64)

        if antenna_offset_enu is not None:
            east_a -= float(antenna_offset_enu[0])
            north_a -= float(antenna_offset_enu[1])
            up_a -= float(antenna_offset_enu[2])
        if antenna_offset_base is not None:
            east_a, north_a, up_a = _apply_antenna_offset_base_link(east_a, north_a, up_a, antenna_offset_base)

        imu_samples = _load_imu_orientation_csv(imu_csv_path) if imu_csv_path else None
        imu_yaw_samples = None
        if imu_samples is not None:
            logger.info("extract_navsat_trajectory: interpolating %d IMU orientations", imu_samples.shape[0])
        elif imu_csv_path:
            imu_yaw_samples = _load_imu_angular_velocity_yaw_csv(imu_csv_path)
            if imu_yaw_samples is not None:
                logger.info(
                    "extract_navsat_trajectory: integrating %d IMU angular-velocity samples",
                    imu_yaw_samples.shape[0],
                )

        with open(tum_path, "w") as f:
            for i, ts in enumerate(rows_ts):
                east = float(east_a[i])
                north = float(north_a[i])
                up = float(up_a[i])
                imu_quat = _interp_imu_quaternion(imu_samples, ts) if imu_samples is not None else None
                if imu_quat is None and imu_yaw_samples is not None:
                    imu_quat = _interp_imu_yaw_quaternion(imu_yaw_samples, ts)
                if vehicle_frame_only:
                    if imu_quat is not None:
                        qx, qy, qz, qw = imu_quat
                        f.write(f"{ts} {east} {north} {up} {qx} {qy} {qz} {qw}\n")
                    else:
                        f.write(f"{ts} {east} {north} {up} 0 0 0 1\n")
                elif T_base_cam is None:
                    if imu_quat is not None:
                        qx, qy, qz, qw = imu_quat
                        f.write(f"{ts} {east} {north} {up} {qx} {qy} {qz} {qw}\n")
                    else:
                        f.write(f"{ts} {east} {north} {up} 0 0 0 1\n")
                else:
                    T_world_base = np.eye(4, dtype=np.float64)
                    if imu_quat is not None:
                        T_world_base[:3, :3] = _quat_to_rotmat(imu_quat)
                    T_world_base[0, 3] = east
                    T_world_base[1, 3] = north
                    T_world_base[2, 3] = up
                    T_world_cam = T_world_base @ T_base_cam
                    twc = T_world_cam[:3, 3]
                    R_wc = T_world_cam[:3, :3]
                    qw, qx, qy, qz = LiDARSLAMProcessor._rotation_to_quaternion(R_wc)
                    f.write(f"{ts} {twc[0]} {twc[1]} {twc[2]} {qx} {qy} {qz} {qw}\n")

        logger.info("Wrote %d GNSS poses to %s", count, tum_path)

        origin_path = pose_dir / "origin_wgs84.json"
        import json as _json

        origin_path.write_text(
            _json.dumps(
                {
                    "ref_lat": float(ref_lat),
                    "ref_lon": float(ref_lon),
                    "ref_alt": float(ref_alt),
                    "source": "reference_origin" if reference_origin is not None else "first_fix",
                    "altitude_mode": "flattened_median" if flatten_altitude else "navsat_altitude",
                    "start_offset_sec": float(start_offset_sec),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(tum_path)

    @staticmethod
    def load_navsat_origin(pose_dir: str | Path) -> tuple[float, float, float] | None:
        """Load ``origin_wgs84.json`` written by :meth:`extract_navsat_trajectory` (returns None if missing)."""
        import json as _json

        p = Path(pose_dir) / "origin_wgs84.json"
        if not p.is_file():
            return None
        data = _json.loads(p.read_text(encoding="utf-8"))
        return float(data["ref_lat"]), float(data["ref_lon"]), float(data["ref_alt"])

    def build_tf_map(self, include_dynamic_tf: bool = False):
        """Load ``TFMessage`` transforms into a :class:`~gs_sim2real.datasets.ros_tf.StaticTfMap`.

        Reads ``/tf_static`` first, then optionally ``/tf`` (later messages override earlier ones
        for the same child frame).
        """
        from gs_sim2real.datasets.ros_tf import StaticTfMap, geometry_transform_to_matrix

        bag_paths = self._find_bag_paths(self.data_dir)
        tf_map = StaticTfMap()
        if not bag_paths:
            return tf_map

        topic_names = list(self.DEFAULT_TF_STATIC_TOPICS)
        if include_dynamic_tf:
            topic_names.append("/tf")

        reader_cls = self._get_anyreader()
        for topic_name in topic_names:
            with self._create_reader(reader_cls, bag_paths) as reader:
                info = reader.topics.get(topic_name)
                if info is None:
                    continue
                connections: list[Any] = []
                for c in info.connections:
                    if c.msgtype in self.TF_MESSAGE_MSGTYPES:
                        connections.append(c)
                if not connections:
                    continue

                for connection, _, rawdata in reader.messages(connections=connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    transforms = getattr(msg, "transforms", None)
                    if not transforms:
                        continue
                    for ts in transforms:
                        parent = str(getattr(getattr(ts, "header", None), "frame_id", "") or "")
                        child = str(getattr(ts, "child_frame_id", "") or "")
                        if not parent or not child:
                            continue
                        T = geometry_transform_to_matrix(ts.transform)
                        tf_map.add(parent, child, T)

        logger.info(
            "Loaded %d TF edges from rosbag (dynamic_tf=%s)",
            len(tf_map),
            include_dynamic_tf,
        )
        return tf_map

    def build_static_tf_map(self):
        """Backward-compatible alias for ``build_tf_map(include_dynamic_tf=False)``."""
        return self.build_tf_map(include_dynamic_tf=False)

    def collect_tf_dynamic_edges(self):
        """Read ``/tf`` and return time-stamped edges for per-message extrinsics (see :class:`TimestampedTfEdges`)."""
        from gs_sim2real.datasets.ros_tf import TimestampedTfEdges, geometry_transform_to_matrix

        edges = TimestampedTfEdges()
        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            edges.finalize()
            return edges

        reader_cls = self._get_anyreader()
        with self._create_reader(reader_cls, bag_paths) as reader:
            info = reader.topics.get("/tf")
            if info is None:
                logger.warning("No /tf topic in bag; timestamped TF lookup will be unavailable")
                edges.finalize()
                return edges

            connections: list[Any] = []
            for c in info.connections:
                if c.msgtype in self.TF_MESSAGE_MSGTYPES:
                    connections.append(c)
            if not connections:
                edges.finalize()
                return edges

            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                transforms = getattr(msg, "transforms", None)
                if not transforms:
                    continue
                for ts_msg in transforms:
                    parent = str(getattr(getattr(ts_msg, "header", None), "frame_id", "") or "")
                    child = str(getattr(ts_msg, "child_frame_id", "") or "")
                    if not parent or not child:
                        continue
                    hdr = getattr(ts_msg, "header", None)
                    st = getattr(hdr, "stamp", None) if hdr is not None else None
                    if st is None:
                        stamp_ns = 0
                    else:
                        sec = int(getattr(st, "sec", 0))
                        nsec = int(getattr(st, "nanosec", getattr(st, "nsec", 0)))
                        stamp_ns = sec * 1_000_000_000 + nsec
                    T = geometry_transform_to_matrix(ts_msg.transform)
                    edges.add(stamp_ns, parent, child, T)

        edges.finalize()
        logger.info("Collected %d /tf transform samples", len(edges))
        return edges

    def extract_camera_info(
        self,
        output_dir: str | Path,
        image_topics: str | list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        """Save one ``sensor_msgs/CameraInfo`` message per image topic as JSON (PINHOLE intrinsics).

        Infers ``.../camera_info`` from each image topic's parent path (e.g. ``.../raw_image`` → ``.../camera_info``).
        """
        output_path = Path(output_dir)
        calib_dir = output_path / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)

        topics = self._normalize_requested_topics(image_topics)
        if not topics:
            return []

        bag_paths = self._find_bag_paths(self.data_dir)
        if not bag_paths:
            raise FileNotFoundError(f"No rosbag found in {self.data_dir}; cannot extract CameraInfo.")

        reader_cls = self._get_anyreader()
        written: list[str] = []
        with self._create_reader(reader_cls, bag_paths) as reader:
            for image_topic in topics:
                info_topic = self._infer_camera_info_topic(str(image_topic))
                try:
                    connection = self._select_connection(
                        reader.topics,
                        requested_topic=info_topic,
                        preferred_topics=tuple(),
                        allowed_msgtypes=self.CAMERA_INFO_MSGTYPES,
                    )
                except ValueError:
                    logger.warning("No CameraInfo topic %s for image topic %s", info_topic, image_topic)
                    continue
                if connection is None:
                    logger.warning("No CameraInfo at %s for image topic %s", info_topic, image_topic)
                    continue
                for _, _, rawdata in reader.messages(connections=[connection]):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    calib = self._camera_info_to_pinhole_dict(msg)
                    label = self._sanitize_topic_name(str(image_topic))
                    out_path = calib_dir / f"{label}.json"
                    with open(out_path, "w") as f:
                        json.dump(calib, f, indent=2)
                    written.append(str(out_path))
                    logger.info("Wrote camera calibration for %s to %s", image_topic, out_path)
                    break

        return written

    @staticmethod
    def _infer_camera_info_topic(image_topic: str) -> str:
        """Map an image topic to a sibling ``camera_info`` topic."""
        parts = [p for p in image_topic.split("/") if p]
        if not parts:
            return "/camera_info"
        parent = "/" + "/".join(parts[:-1])
        return f"{parent}/camera_info" if parent else "/camera_info"

    @staticmethod
    def _camera_info_to_pinhole_dict(msg: Any) -> dict[str, float | int | str]:
        """Convert CameraInfo K matrix to PINHOLE parameters plus ``header.frame_id``."""
        w = int(getattr(msg, "width", 0))
        h = int(getattr(msg, "height", 0))
        k = getattr(msg, "k", None)
        if k is None or len(k) < 9:
            raise ValueError("CameraInfo.k must have at least 9 elements")
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        hdr = getattr(msg, "header", None)
        fid = str(getattr(hdr, "frame_id", "") or "") if hdr is not None else ""
        return {"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "frame_id": fid}

    @staticmethod
    def _get_anyreader():
        """Return the rosbags AnyReader class."""
        return rosbag_frames.get_anyreader()

    @classmethod
    def _create_reader(cls, reader_cls: type[Any], bag_paths: list[Path]):
        """Instantiate AnyReader with a default typestore for bags lacking definitions."""
        return rosbag_frames.create_reader(reader_cls, bag_paths)

    @staticmethod
    def _get_reader_kwargs(bag_paths: list[Path]) -> dict[str, Any]:
        """Build AnyReader kwargs based on rosbag format."""
        return rosbag_frames.reader_kwargs(bag_paths)

    @staticmethod
    def _find_bag_paths(data_dir: Path) -> list[Path]:
        """Find rosbag1 files or rosbag2 directories under the input path."""
        return rosbag_frames.find_bag_paths(data_dir)

    def _load_pre_extracted(self, output_dir: Path, max_frames: int) -> str:
        """Load pre-extracted images from ``images/`` or the dataset root."""
        del max_frames
        candidates = [
            output_dir / "images",
            self.data_dir / "images",
            self.data_dir,
        ]
        for images_dir in candidates:
            images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            if images:
                logger.info("Found %d pre-extracted MCD images", len(images))
                return str(images_dir)

        raise FileNotFoundError(
            f"No images found in {self.data_dir}. Either provide rosbag files or pre-extracted images."
        )

    @classmethod
    def _infer_topic_role(cls, topic_name: str, msgtype: str) -> str:
        """Infer whether a topic is best treated as image, lidar, imu, or other."""
        if msgtype in cls.IMAGE_MSGTYPES:
            return "image"
        if msgtype in cls.POINTCLOUD_MSGTYPES:
            return "lidar"
        if msgtype in cls.IMU_MSGTYPES:
            return "imu"
        if msgtype in cls.NAVSAT_MSGTYPES:
            return "gnss"
        if msgtype in cls.CAMERA_INFO_MSGTYPES:
            return "calibration"
        if msgtype in cls.TF_MESSAGE_MSGTYPES:
            return "tf"
        if "image" in topic_name:
            return "image"
        if "point" in topic_name or "lidar" in topic_name:
            return "lidar"
        if "imu" in topic_name:
            return "imu"
        return "other"

    @classmethod
    def _is_preferred_default_topic(cls, topic_name: str) -> bool:
        """Return whether the topic matches one of the built-in MCD defaults."""
        return topic_name in {
            *cls.DEFAULT_IMAGE_TOPICS,
            *cls.DEFAULT_LIDAR_TOPICS,
            *cls.DEFAULT_IMU_TOPICS,
            *cls.DEFAULT_GNSS_TOPICS,
        }

    @staticmethod
    def _select_connection(
        topics: dict[str, Any],
        requested_topic: str | None,
        preferred_topics: tuple[str, ...],
        allowed_msgtypes: frozenset[str],
    ):
        """Pick the first matching connection for the requested or preferred topics."""
        return rosbag_frames.select_connection(topics, requested_topic, preferred_topics, allowed_msgtypes)

    @staticmethod
    def _select_connections(
        topics: dict[str, Any],
        requested_topics: list[str] | None,
        preferred_topics: tuple[str, ...],
        allowed_msgtypes: frozenset[str],
    ) -> list[Any]:
        """Pick one or more matching connections."""
        return rosbag_frames.select_connections(topics, requested_topics, preferred_topics, allowed_msgtypes)

    @staticmethod
    def _normalize_requested_topics(
        requested_topic: str | list[str] | tuple[str, ...] | None,
    ) -> list[str] | None:
        """Normalize a requested topic argument into a list."""
        return rosbag_frames.normalize_requested_topics(requested_topic)

    @staticmethod
    def _sanitize_topic_name(topic_name: str) -> str:
        """Convert a ROS topic name into a filesystem-friendly folder label."""
        return rosbag_frames.sanitize_topic_name(topic_name)

    @classmethod
    def _decode_image_message(cls, msg: Any, msgtype: str) -> tuple[np.ndarray | None, str]:
        """Decode a ROS image message into an OpenCV image."""
        return rosbag_frames.decode_image_message(msg, msgtype)

    @staticmethod
    def _reshape_image_buffer(
        data: bytes,
        height: int,
        width: int,
        dtype: type[np.generic],
        channels: int,
        step: int,
    ) -> np.ndarray:
        """Reshape image bytes using the ROS step size when present."""
        return rosbag_frames.reshape_image_buffer(data, height, width, dtype, channels, step)

    @classmethod
    def _pointcloud2_to_numpy(cls, msg: Any) -> np.ndarray:
        """Convert a ``sensor_msgs/PointCloud2`` message to an ``Nx4`` float array."""
        formats = []
        names = []
        offsets = []
        endian = ">" if getattr(msg, "is_bigendian", False) else "<"

        for field in msg.fields:
            dtype = cls.POINTFIELD_DTYPES.get(int(field.datatype))
            if dtype is None:
                continue
            np_dtype = np.dtype(dtype).newbyteorder(endian)
            names.append(str(field.name))
            offsets.append(int(field.offset))
            formats.append((np_dtype, int(field.count)) if int(field.count) > 1 else np_dtype)

        dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": int(msg.point_step),
            }
        )
        count = int(msg.width) * int(msg.height)
        raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=count)

        if not {"x", "y", "z"}.issubset(raw.dtype.names or ()):
            raise ValueError("PointCloud2 message does not contain x/y/z fields.")

        x = raw["x"].astype(np.float32, copy=False)
        y = raw["y"].astype(np.float32, copy=False)
        z = raw["z"].astype(np.float32, copy=False)
        intensity = np.zeros_like(x, dtype=np.float32)
        for key in ("intensity", "reflectivity", "i"):
            if key in raw.dtype.names:
                values = raw[key]
                intensity = (
                    values[:, 0].astype(np.float32, copy=False)
                    if values.ndim > 1
                    else values.astype(np.float32, copy=False)
                )
                break

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        return np.stack([x[mask], y[mask], z[mask], intensity[mask]], axis=-1)


def _load_imu_orientation_csv(path: str | Path) -> np.ndarray | None:
    """Load (timestamp_sec, qx, qy, qz, qw) rows from the CSV :meth:`MCDLoader.extract_imu` writes.

    Returns a (N, 5) ``float64`` array sorted by timestamp or ``None`` if
    the file is missing / empty / has only identity quaternions (useful when
    an IMU topic actually carries no orientation).
    """
    p = Path(path)
    if not p.is_file():
        return None
    rows: list[tuple[float, float, float, float, float]] = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp_sec"])
                qx = float(row["orientation_x"])
                qy = float(row["orientation_y"])
                qz = float(row["orientation_z"])
                qw = float(row["orientation_w"])
            except (KeyError, ValueError):
                continue
            quat = np.array([qx, qy, qz, qw], dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if not (np.isfinite(ts) and np.all(np.isfinite(quat))) or norm < 1e-9:
                continue
            quat /= norm
            rows.append((ts, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
    if not rows:
        return None
    arr = np.asarray(sorted(rows, key=lambda r: r[0]), dtype=np.float64)
    # Reject bags whose IMU topic reports identity all the way through. A
    # constant non-identity attitude is still useful, e.g. stationary rig
    # warm-up or a fixed mount orientation.
    identity_rows = (np.linalg.norm(arr[:, 1:4], axis=1) < 1e-6) & (np.abs(np.abs(arr[:, 4]) - 1.0) < 1e-6)
    if bool(np.all(identity_rows)):
        return None
    return arr


def _load_imu_angular_velocity_yaw_csv(path: str | Path) -> np.ndarray | None:
    """Load IMU z-rate samples and integrate them into (timestamp_sec, yaw_rad)."""
    p = Path(path)
    if not p.is_file():
        return None
    rows: list[tuple[float, float]] = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp_sec"])
                wz = float(row["angular_velocity_z"])
            except (KeyError, ValueError):
                continue
            if not (np.isfinite(ts) and np.isfinite(wz)):
                continue
            rows.append((ts, wz))
    if len(rows) < 2:
        return None

    times: list[float] = []
    rates: list[float] = []
    for ts, wz in sorted(rows, key=lambda r: r[0]):
        if times and ts == times[-1]:
            rates[-1] = wz
        else:
            times.append(ts)
            rates.append(wz)
    if len(times) < 2 or max(abs(wz) for wz in rates) < 1e-9:
        return None

    times_arr = np.asarray(times, dtype=np.float64)
    rates_arr = np.asarray(rates, dtype=np.float64)
    yaw = np.zeros_like(times_arr)
    for i in range(1, len(times_arr)):
        dt = float(times_arr[i] - times_arr[i - 1])
        if dt <= 0.0:
            yaw[i] = yaw[i - 1]
        else:
            yaw[i] = yaw[i - 1] + 0.5 * (rates_arr[i - 1] + rates_arr[i]) * dt

    return np.column_stack([times_arr, yaw])


def _interp_imu_quaternion(imu_samples: np.ndarray, ts: float) -> tuple[float, float, float, float] | None:
    """Interpolate a (qx, qy, qz, qw) quaternion at ``ts``."""
    if imu_samples is None or imu_samples.shape[0] == 0:
        return None
    times = imu_samples[:, 0]
    if ts <= times[0]:
        quat = imu_samples[0, 1:]
    elif ts >= times[-1]:
        quat = imu_samples[-1, 1:]
    else:
        idx = int(np.searchsorted(times, ts))
        lo = imu_samples[idx - 1]
        hi = imu_samples[idx]
        dt = hi[0] - lo[0]
        if dt <= 0:
            quat = lo[1:]
        else:
            alpha = (ts - lo[0]) / dt
            q_lo = lo[1:]
            q_hi = hi[1:]
            if float(np.dot(q_lo, q_hi)) < 0.0:
                q_hi = -q_hi
            quat = (1.0 - alpha) * q_lo + alpha * q_hi
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return None
    quat = quat / norm
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _interp_imu_yaw_quaternion(yaw_samples: np.ndarray, ts: float) -> tuple[float, float, float, float] | None:
    """Interpolate yaw samples and return a z-axis quaternion."""
    if yaw_samples is None or yaw_samples.shape[0] == 0:
        return None
    times = yaw_samples[:, 0]
    if ts <= times[0]:
        yaw = float(yaw_samples[0, 1])
    elif ts >= times[-1]:
        yaw = float(yaw_samples[-1, 1])
    else:
        idx = int(np.searchsorted(times, ts))
        lo = yaw_samples[idx - 1]
        hi = yaw_samples[idx]
        dt = float(hi[0] - lo[0])
        if dt <= 0.0:
            yaw = float(lo[1])
        else:
            alpha = float((ts - lo[0]) / dt)
            yaw = float((1.0 - alpha) * lo[1] + alpha * hi[1])
    if not np.isfinite(yaw):
        return None
    half = 0.5 * yaw
    return 0.0, 0.0, float(np.sin(half)), float(np.cos(half))


def _quat_to_rotmat(quat: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a (qx, qy, qz, qw) quaternion into a 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
