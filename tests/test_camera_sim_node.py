"""Tests for the ROS 2 GS camera simulator node (ROS-free pieces)."""

from __future__ import annotations

import numpy as np
import pytest

from gs_sim2real.robotics.camera_sim_node import (
    build_parser,
    camera_intrinsics_from_colmap,
    render_optical,
    replay_poses,
    scale_intrinsics,
    validate_args,
    vertical_fov_degrees,
)
from gs_sim2real.robotics.gsplat_render_server import (
    CameraPose,
    HeadlessSplatRenderer,
    build_view_matrix_world_to_camera,
    compute_camera_intrinsics,
)
from gs_sim2real.robotics.localize import viewmat_to_colmap
from gs_sim2real.train.large_scale_3dgs import ColmapImageRecord


def _record(name: str, qvec, tvec) -> ColmapImageRecord:
    return ColmapImageRecord(
        image_id=1,
        camera_id=1,
        name=name,
        qvec=tuple(float(v) for v in qvec),
        tvec=tuple(float(v) for v in tvec),
        center=(0.0, 0.0, 0.0),
        metadata_line="",
        points2d_line="",
    )


class TestParserAndValidation:
    def test_defaults(self):
        args = build_parser().parse_args(["--map", "session"])
        assert args.pose_topic == "/gs_camera_sim/pose"
        assert args.image_topic.endswith("/compressed")
        assert args.depth_topic == ""
        assert not args.replay
        validate_args(args)

    def test_requires_exactly_one_input(self):
        with pytest.raises(SystemExit):
            validate_args(build_parser().parse_args([]))
        with pytest.raises(SystemExit):
            validate_args(build_parser().parse_args(["--map", "session", "--ply", "scene.ply"]))

    def test_replay_needs_a_session(self):
        with pytest.raises(SystemExit):
            validate_args(build_parser().parse_args(["--ply", "scene.ply", "--replay"]))

    def test_rejects_non_positive_fps(self):
        with pytest.raises(SystemExit):
            validate_args(build_parser().parse_args(["--map", "session", "--fps", "0"]))


class TestCameraIntrinsicsFromColmap:
    def test_pinhole(self):
        cam = {"model": "PINHOLE", "width": 64, "height": 48, "params": [50.0, 52.0, 32.0, 24.0]}
        assert camera_intrinsics_from_colmap(cam) == (64, 48, 50.0, 52.0, 32.0, 24.0)

    def test_simple_pinhole(self):
        cam = {"model": "SIMPLE_PINHOLE", "width": 64, "height": 48, "params": [50.0, 32.0, 24.0]}
        assert camera_intrinsics_from_colmap(cam) == (64, 48, 50.0, 50.0, 32.0, 24.0)

    def test_unknown_model_falls_back_to_center(self):
        cam = {"model": "MYSTERY", "width": 64, "height": 48, "params": [50.0]}
        width, height, fx, fy, cx, cy = camera_intrinsics_from_colmap(cam)
        assert (fx, fy) == (50.0, 50.0)
        assert (cx, cy) == (32.0, 24.0)


class TestIntrinsicsHelpers:
    def test_scale_intrinsics_half_size(self):
        fx, fy, cx, cy = scale_intrinsics((100.0, 100.0, 63.5, 47.5), from_size=(128, 96), to_size=(64, 48))
        assert fx == pytest.approx(50.0)
        assert fy == pytest.approx(50.0)
        assert cx == pytest.approx(31.5)
        assert cy == pytest.approx(23.5)

    def test_vertical_fov_round_trips_compute_camera_intrinsics(self):
        _, fy, _, _ = compute_camera_intrinsics(640, 480, 60.0)
        assert vertical_fov_degrees(fy, 480) == pytest.approx(60.0)


class TestReplayPoses:
    def test_identity_pose(self):
        poses = replay_poses([_record("kf_000000.jpg", (1, 0, 0, 0), (1.0, 2.0, 3.0))])
        name, pose = poses[0]
        assert name == "kf_000000.jpg"
        assert pose.position == pytest.approx((-1.0, -2.0, -3.0))
        assert pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_round_trips_to_colmap_viewmat(self):
        qvec = (0.8, 0.1, -0.3, 0.5)
        tvec = (0.4, -1.2, 2.5)
        ((_, pose),) = replay_poses([_record("kf_000001.jpg", qvec, tvec)])
        viewmat = build_view_matrix_world_to_camera(pose)
        qvec_back, tvec_back, _ = viewmat_to_colmap(np.asarray(viewmat, dtype=np.float64))
        norm = np.linalg.norm(qvec)
        expected = np.asarray(qvec) / norm
        if np.dot(expected, qvec_back) < 0:
            expected = -expected
        assert np.asarray(qvec_back) == pytest.approx(expected, abs=1e-5)
        assert np.asarray(tvec_back) == pytest.approx(np.asarray(tvec), abs=1e-5)


class _FakeRenderer:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.calls: list[dict] = []

    def render_rgbd(self, pose, **kwargs):
        self.calls.append(kwargs)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[0, 0] = (255, 0, 0)
        depth = np.full((4, 4), 9.0, dtype=np.float32)
        depth[0, 0] = 1.0
        return rgb, depth


class TestRenderOptical:
    def _render(self, backend: str):
        renderer = _FakeRenderer(backend)
        rgb, depth = render_optical(
            renderer,
            CameraPose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)),
            width=4,
            height=4,
            intrinsics=(2.0, 2.0, 1.5, 1.5),
            near_clip=0.01,
            far_clip=100.0,
            point_radius=1,
        )
        return renderer, rgb, depth

    def test_gsplat_backend_is_passed_through(self):
        renderer, rgb, depth = self._render("gsplat")
        assert tuple(rgb[0, 0]) == (255, 0, 0)
        assert depth[0, 0] == pytest.approx(1.0)
        assert renderer.calls[0]["intrinsics"] == (2.0, 2.0, 1.5, 1.5)

    def test_simple_backend_output_is_mirrored_vertically(self):
        _, rgb, depth = self._render("simple")
        assert tuple(rgb[-1, 0]) == (255, 0, 0)
        assert depth[-1, 0] == pytest.approx(1.0)


def _write_point_ply(path, points):
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property float red",
        "property float green",
        "property float blue",
        "end_header",
    ]
    for x, y, z, r, g, b in points:
        lines.append(f"{x} {y} {z} {r} {g} {b}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestSimpleBackendOpticalProjection:
    def test_intrinsics_override_and_optical_flip(self, tmp_path):
        """A point below the optical axis (+y) must land below the image center."""
        ply = tmp_path / "scene.ply"
        _write_point_ply(ply, [(0.0, 0.5, 2.0, 1.0, 1.0, 1.0)])
        renderer = HeadlessSplatRenderer(ply, backend="simple")
        intrinsics = (8.0, 8.0, 7.5, 7.5)
        rgb, depth = render_optical(
            renderer,
            CameraPose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)),
            width=16,
            height=16,
            intrinsics=intrinsics,
            near_clip=0.1,
            far_clip=50.0,
            point_radius=0,
        )
        lit = np.argwhere(rgb.sum(axis=2) > 0)
        assert len(lit) == 1
        row, col = lit[0]
        assert col == 8  # (0 / 2) * 8 + 7.5 -> rounds to 8
        assert row > 8  # +y in the optical convention is downwards in the image
        assert depth[row, col] == pytest.approx(2.0)
