"""Tests for the rclpy-free pieces of the live mapper ROS 2 node."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gs_sim2real.robotics.live_mapper_node import (
    _stamp_to_seconds,
    build_parser,
    decode_compressed_image,
    decode_raw_image,
)


@dataclass
class _FakeStamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class _FakeHeader:
    stamp: _FakeStamp = field(default_factory=_FakeStamp)
    frame_id: str = "camera"


@dataclass
class _FakeImageMsg:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes
    header: _FakeHeader = field(default_factory=_FakeHeader)


@dataclass
class _FakeCompressedMsg:
    data: bytes
    format: str = "jpeg"
    header: _FakeHeader = field(default_factory=_FakeHeader)


def _bgr_frame(h: int = 24, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


class TestDecodeCompressedImage:
    def test_jpeg_roundtrip(self):
        frame = _bgr_frame()
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        decoded = decode_compressed_image(_FakeCompressedMsg(data=encoded.tobytes()))
        assert decoded is not None
        assert decoded.shape == frame.shape

    def test_garbage_returns_none(self):
        assert decode_compressed_image(_FakeCompressedMsg(data=b"not an image")) is None


class TestDecodeRawImage:
    def test_bgr8_passthrough(self):
        frame = _bgr_frame()
        msg = _FakeImageMsg(
            height=frame.shape[0],
            width=frame.shape[1],
            encoding="bgr8",
            step=frame.shape[1] * 3,
            data=frame.tobytes(),
        )
        decoded = decode_raw_image(msg)
        assert decoded is not None
        np.testing.assert_array_equal(decoded, frame)

    def test_rgb8_is_converted_to_bgr(self):
        frame = _bgr_frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        msg = _FakeImageMsg(
            height=frame.shape[0],
            width=frame.shape[1],
            encoding="rgb8",
            step=frame.shape[1] * 3,
            data=rgb.tobytes(),
        )
        decoded = decode_raw_image(msg)
        np.testing.assert_array_equal(decoded, frame)

    def test_mono8_becomes_three_channel(self):
        gray = cv2.cvtColor(_bgr_frame(), cv2.COLOR_BGR2GRAY)
        msg = _FakeImageMsg(
            height=gray.shape[0],
            width=gray.shape[1],
            encoding="mono8",
            step=gray.shape[1],
            data=gray.tobytes(),
        )
        decoded = decode_raw_image(msg)
        assert decoded is not None
        assert decoded.shape == (*gray.shape, 3)

    def test_row_padding_in_step_is_stripped(self):
        frame = _bgr_frame()
        padded = np.concatenate(
            [frame.reshape(frame.shape[0], -1), np.zeros((frame.shape[0], 8), dtype=np.uint8)], axis=1
        )
        msg = _FakeImageMsg(
            height=frame.shape[0],
            width=frame.shape[1],
            encoding="bgr8",
            step=frame.shape[1] * 3 + 8,
            data=padded.tobytes(),
        )
        decoded = decode_raw_image(msg)
        np.testing.assert_array_equal(decoded, frame)

    def test_unsupported_encoding_returns_none(self):
        msg = _FakeImageMsg(height=2, width=2, encoding="bayer_rggb8", step=2, data=b"\x00" * 4)
        assert decode_raw_image(msg) is None


class TestStampToSeconds:
    def test_uses_header_stamp(self):
        msg = _FakeImageMsg(height=1, width=1, encoding="mono8", step=1, data=b"\x00")
        msg.header.stamp = _FakeStamp(sec=10, nanosec=500_000_000)
        assert _stamp_to_seconds(msg, fallback=99.0) == 10.5

    def test_zero_stamp_falls_back(self):
        msg = _FakeImageMsg(height=1, width=1, encoding="mono8", step=1, data=b"\x00")
        assert _stamp_to_seconds(msg, fallback=99.0) == 99.0


def test_parser_defaults_are_sane():
    args = build_parser().parse_args([])
    assert args.image_topic.endswith("/compressed")
    assert args.method == "dust3r"
    assert args.port == 8765
    assert args.rebuild_min_new >= 1
