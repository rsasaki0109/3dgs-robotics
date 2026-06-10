"""CPU-only helpers of scripts/build_robotics_demo_gif.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.robotics.gsplat_render_server import CameraPose

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gif_module():
    path = REPO_ROOT / "scripts" / "build_robotics_demo_gif.py"
    spec = importlib.util.spec_from_file_location("build_robotics_demo_gif", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pose(position, orientation=(0.0, 0.0, 0.0, 1.0)) -> CameraPose:
    return CameraPose(position=tuple(position), orientation=tuple(orientation))


class TestInterpolatePoses:
    def test_inserts_steps_and_keeps_keyframes(self, gif_module):
        poses = [("kf_0", _pose((0, 0, 0))), ("kf_1", _pose((0, 0, 2)))]
        out = gif_module.interpolate_poses(poses, steps_between=1)
        assert [label for label, _, is_kf in out if is_kf] == ["kf_0", "kf_1"]
        assert len(out) == 3
        _, mid, is_kf = out[1]
        assert not is_kf
        assert mid.position == pytest.approx((0.0, 0.0, 1.0))

    def test_zero_steps_passthrough(self, gif_module):
        poses = [("kf_0", _pose((0, 0, 0))), ("kf_1", _pose((1, 0, 0)))]
        out = gif_module.interpolate_poses(poses, steps_between=0)
        assert len(out) == 2
        assert all(is_kf for _, _, is_kf in out)

    def test_orientation_is_slerped(self, gif_module):
        quarter = (0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4))  # 90 deg about y
        poses = [("kf_0", _pose((0, 0, 0))), ("kf_1", _pose((0, 0, 0), quarter))]
        out = gif_module.interpolate_poses(poses, steps_between=1)
        _, mid, _ = out[1]
        expected = (0.0, np.sin(np.pi / 8), 0.0, np.cos(np.pi / 8))  # 45 deg
        assert np.asarray(mid.orientation) == pytest.approx(np.asarray(expected), abs=1e-6)


class TestGridView:
    def test_world_to_pixel_roundtrip(self, gif_module):
        view = gif_module.GridView(
            image=np.zeros((10, 20, 3), dtype=np.uint8),
            basis2=np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
            origin=(-1.0, -2.0),
            resolution=0.5,
        )
        # world (x, y, z) -> grid (z, -x); point z=0.5, x=-0.5 -> grid (0.5, 0.5)
        col, row = view.world_to_pixel(np.array([-0.5, 0.0, 0.5]))
        assert (col, row) == (3, 5)

    def test_world_to_pixel_clamps(self, gif_module):
        view = gif_module.GridView(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            basis2=np.eye(3)[:2],
            origin=(0.0, 0.0),
            resolution=1.0,
        )
        assert view.world_to_pixel(np.array([100.0, 100.0, 0.0])) == (3, 3)
        assert view.world_to_pixel(np.array([-100.0, -100.0, 0.0])) == (0, 0)


class TestComposeFrame:
    def test_frame_dimensions_and_annotations(self, gif_module):
        view = gif_module.GridView(
            image=np.full((30, 60, 3), 60, dtype=np.uint8),
            basis2=np.eye(3)[:2],
            origin=(0.0, 0.0),
            resolution=0.1,
        )
        camera_rgb = np.zeros((20, 40, 3), dtype=np.uint8)
        frame = gif_module.compose_frame(
            camera_rgb,
            view,
            gt_centers=np.array([[0.5, 0.5, 0.0], [5.0, 2.0, 0.0]]),
            estimates=[np.array([1.0, 1.0, 0.0])],
            current=np.array([5.0, 2.0, 0.0]),
            width=120,
        )
        assert frame.width == 120
        # camera panel scaled to 120x60, grid to 120x60, two 26 px captions
        assert frame.height == 60 + 60 + 52
        pixels = np.asarray(frame)
        assert (pixels == np.asarray(gif_module.ESTIMATE_DOT)).all(axis=2).any()
        assert (pixels == np.asarray(gif_module.GT_TRAIL)).all(axis=2).any()
