"""Tests for the generic rosbag image frame source (datasets/rosbag_frames.py).

These write small synthetic rosbag2 recordings with the pure-Python
``rosbags`` library — no ROS 2 installation is involved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags")
pytest.importorskip("cv2")

import cv2  # noqa: E402

from gs_sim2real.datasets import rosbag_frames  # noqa: E402

BASE_STAMP_NS = 1_700_000_000_000_000_000
FRAME_DT_NS = 500_000_000  # 2 Hz


def _typestore():
    from rosbags.typesys import Stores, get_typestore

    return get_typestore(Stores.ROS2_HUMBLE)


def _write_synthetic_bag(
    bag_dir: Path,
    *,
    raw_topics: tuple[str, ...] = ("/cam/image_raw",),
    compressed_topics: tuple[str, ...] = (),
    frames: int = 6,
) -> Path:
    """Write a rosbag2 directory with bgr8 Image and/or jpeg CompressedImage topics.

    Raw frames encode their index in the pixel value (``index * 20``) so tests
    can assert ordering after decode.
    """
    from rosbags.rosbag2 import Writer

    typestore = _typestore()
    image_type = typestore.types["sensor_msgs/msg/Image"]
    compressed_type = typestore.types["sensor_msgs/msg/CompressedImage"]
    header_type = typestore.types["std_msgs/msg/Header"]
    time_type = typestore.types["builtin_interfaces/msg/Time"]

    with Writer(bag_dir, version=8) as writer:
        for topic in raw_topics:
            connection = writer.add_connection(topic, image_type.__msgtype__, typestore=typestore)
            for index in range(frames):
                pixels = np.full((8, 12, 3), index * 20, dtype=np.uint8)
                msg = image_type(
                    header=header_type(stamp=time_type(sec=index, nanosec=0), frame_id="cam"),
                    height=8,
                    width=12,
                    encoding="bgr8",
                    is_bigendian=0,
                    step=36,
                    data=pixels.reshape(-1),
                )
                writer.write(
                    connection,
                    BASE_STAMP_NS + index * FRAME_DT_NS,
                    typestore.serialize_cdr(msg, image_type.__msgtype__),
                )
        for topic in compressed_topics:
            connection = writer.add_connection(topic, compressed_type.__msgtype__, typestore=typestore)
            for index in range(frames):
                pixels = np.full((8, 12, 3), index * 20, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", pixels)
                assert ok
                msg = compressed_type(
                    header=header_type(stamp=time_type(sec=index, nanosec=0), frame_id="cam"),
                    format="jpeg",
                    data=np.frombuffer(encoded.tobytes(), dtype=np.uint8),
                )
                writer.write(
                    connection,
                    BASE_STAMP_NS + index * FRAME_DT_NS,
                    typestore.serialize_cdr(msg, compressed_type.__msgtype__),
                )
    return bag_dir


class TestBagDiscovery:
    def test_find_bag_paths_accepts_rosbag2_directory(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag")
        assert rosbag_frames.find_bag_paths(bag) == [bag]

    def test_find_bag_paths_prefers_parent_dir_for_db3_with_metadata(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag")
        db3 = next(bag.glob("*.db3"))
        assert rosbag_frames.find_bag_paths(db3) == [bag]

    def test_find_bag_paths_accepts_bare_db3(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag")
        bare = tmp_path / "bare.db3"
        bare.write_bytes(next(bag.glob("*.db3")).read_bytes())
        assert rosbag_frames.find_bag_paths(bare) == [bare]

    def test_find_bag_paths_searches_directory_tree(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "logs" / "drive_bag")
        assert rosbag_frames.find_bag_paths(tmp_path) == [bag]

    def test_is_rosbag_path_rejects_image_folder_and_video(self, tmp_path: Path) -> None:
        images = tmp_path / "frames"
        images.mkdir()
        (images / "frame_000000.jpg").write_bytes(b"\xff\xd8\xff")
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"")
        assert not rosbag_frames.is_rosbag_path(images)
        assert not rosbag_frames.is_rosbag_path(video)
        assert not rosbag_frames.is_rosbag_path(tmp_path / "missing")

    def test_is_rosbag_path_accepts_bag_inputs(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag")
        assert rosbag_frames.is_rosbag_path(bag)
        assert rosbag_frames.is_rosbag_path(next(bag.glob("*.db3")))


class TestTopicResolution:
    def test_list_image_topics_reports_count(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=4)
        topics = rosbag_frames.list_image_topics(bag)
        assert topics == [rosbag_frames.BagImageTopic("/cam/image_raw", "sensor_msgs/msg/Image", 4)]

    def test_resolve_auto_selects_single_topic(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag")
        selected = rosbag_frames.resolve_image_topic(bag)
        assert selected.topic == "/cam/image_raw"

    def test_resolve_ambiguous_lists_candidates(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(
            tmp_path / "drive_bag",
            raw_topics=("/cam_front/image_raw",),
            compressed_topics=("/cam_rear/image_raw/compressed",),
        )
        with pytest.raises(ValueError, match="--image-topic") as excinfo:
            rosbag_frames.resolve_image_topic(bag)
        assert "/cam_front/image_raw" in str(excinfo.value)
        assert "/cam_rear/image_raw/compressed" in str(excinfo.value)

    def test_resolve_no_image_topic_is_actionable(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag", raw_topics=(), frames=0)
        with pytest.raises(ValueError, match="No sensor_msgs"):
            rosbag_frames.resolve_image_topic(bag)

    def test_resolve_rejects_unknown_requested_topic(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag")
        with pytest.raises(ValueError, match="not a supported image topic"):
            rosbag_frames.resolve_image_topic(bag, "/lidar/points")


class TestFrameIteration:
    def test_iter_decodes_bgr8_frames_in_order(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=5)
        frames = list(rosbag_frames.iter_bag_frames(bag))
        assert [f.index for f in frames] == [0, 1, 2, 3, 4]
        for i, frame in enumerate(frames):
            assert frame.image_bgr.shape == (8, 12, 3)
            assert int(frame.image_bgr[0, 0, 0]) == i * 20
            assert frame.timestamp_sec == pytest.approx((BASE_STAMP_NS + i * FRAME_DT_NS) * 1e-9)

    def test_iter_decodes_compressed_frames(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(
            tmp_path / "drive_bag",
            raw_topics=(),
            compressed_topics=("/cam/image_raw/compressed",),
            frames=3,
        )
        frames = list(rosbag_frames.iter_bag_frames(bag))
        assert len(frames) == 3
        assert frames[0].image_bgr.shape == (8, 12, 3)

    def test_iter_supports_stride_and_cap(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=6)
        frames = list(rosbag_frames.iter_bag_frames(bag, every_n=2, max_frames=2))
        assert [int(f.image_bgr[0, 0, 0]) for f in frames] == [0, 40]

    def test_plan_bag_frame_sampling(self) -> None:
        assert rosbag_frames.plan_bag_frame_sampling(6, 32) == 1
        assert rosbag_frames.plan_bag_frame_sampling(100, 32) == 3
        assert rosbag_frames.plan_bag_frame_sampling(100, 0) == 1

    def test_extract_bag_frames_targets_frame_count(self, tmp_path: Path) -> None:
        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=6)
        out = tmp_path / "frames"
        written = rosbag_frames.extract_bag_frames(bag, out, target_frames=3)
        assert [p.name for p in written] == ["frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg"]
        first = cv2.imread(str(written[0]))
        assert first is not None and first.shape == (8, 12, 3)


class TestLiveMappingReplayWiring:
    def test_demo_parser_accepts_bag_flags(self) -> None:
        from scripts.run_live_mapping_demo import build_parser

        args = build_parser().parse_args(["--bag", "drive_bag", "--image-topic", "/cam/image_raw", "--rate", "4"])
        assert args.bag == "drive_bag"
        assert args.image_topic == "/cam/image_raw"
        assert args.rate == 4.0

    def test_demo_parser_rejects_images_and_bag_together(self) -> None:
        from scripts.run_live_mapping_demo import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--images", "frames/", "--bag", "drive_bag"])

    def test_iter_rosbag_replay_uses_relative_bag_timestamps(self, tmp_path: Path) -> None:
        from scripts.run_live_mapping_demo import iter_rosbag_replay

        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=4)
        fed = list(iter_rosbag_replay(str(bag), "/cam/image_raw", rate=1.0, realtime=False))
        assert [round(ts, 3) for _, ts, _ in fed] == [0.0, 0.5, 1.0, 1.5]
        assert all(sleep == 0.0 for _, _, sleep in fed)

    def test_iter_rosbag_replay_realtime_paces_by_bag_dt(self, tmp_path: Path) -> None:
        from scripts.run_live_mapping_demo import iter_rosbag_replay

        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=3)
        fed = list(iter_rosbag_replay(str(bag), "/cam/image_raw", rate=2.0, realtime=True))
        assert [round(sleep, 3) for _, _, sleep in fed] == [0.0, 0.25, 0.25]


class TestMapSubcommandBagInput:
    def test_map_extracts_bag_frames_and_delegates(self, tmp_path: Path, monkeypatch) -> None:
        from gs_sim2real.cli import main

        bag = _write_synthetic_bag(tmp_path / "drive_bag", frames=6)
        output_dir = tmp_path / "out"
        delegated: list[argparse.Namespace] = []
        monkeypatch.setattr("gs_sim2real.cli.cmd_photos_to_splat", lambda args: delegated.append(args))

        main(
            [
                "map",
                str(bag),
                "--output",
                str(output_dir),
                "--preprocess",
                "simple",
                "--num-frames",
                "4",
                "--no-open-viewer",
            ]
        )

        frames_dir = output_dir / "drive_bag"
        assert sorted(p.name for p in frames_dir.glob("frame_*.jpg")) == [
            "frame_000000.jpg",
            "frame_000001.jpg",
            "frame_000002.jpg",
            "frame_000003.jpg",
        ]
        assert len(delegated) == 1
        assert delegated[0].images == str(frames_dir)
        assert delegated[0].num_frames == 4

    def test_map_ambiguous_topic_exits_with_candidates(self, tmp_path: Path, monkeypatch, capsys) -> None:
        from gs_sim2real.cli import main

        bag = _write_synthetic_bag(
            tmp_path / "drive_bag",
            raw_topics=("/cam_front/image_raw", "/cam_rear/image_raw"),
            frames=2,
        )
        monkeypatch.setattr("gs_sim2real.cli.cmd_photos_to_splat", lambda args: None)
        with pytest.raises(SystemExit):
            main(["map", str(bag), "--no-open-viewer"])
        out = capsys.readouterr().out
        assert "--image-topic" in out
        assert "/cam_front/image_raw" in out

    def test_map_image_topic_flag_disambiguates(self, tmp_path: Path, monkeypatch) -> None:
        from gs_sim2real.cli import main

        bag = _write_synthetic_bag(
            tmp_path / "drive_bag",
            raw_topics=("/cam_front/image_raw", "/cam_rear/image_raw"),
            frames=3,
        )
        delegated: list[argparse.Namespace] = []
        monkeypatch.setattr("gs_sim2real.cli.cmd_photos_to_splat", lambda args: delegated.append(args))

        main(
            [
                "map",
                str(bag),
                "--image-topic",
                "/cam_rear/image_raw",
                "--output",
                str(tmp_path / "out"),
                "--no-open-viewer",
            ]
        )
        assert len(delegated) == 1
