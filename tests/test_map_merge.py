"""Tests for merging two 3DGS maps (collaborative mapping)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.robotics.gauge_alignment import apply_to_points, invert, rotation_to_quat
from gs_sim2real.robotics.map_merge import (
    RawGaussianPly,
    duplicate_mask,
    merge_raw_gaussians,
    merge_sessions,
    read_raw_gaussian_ply,
    transform_raw_gaussians,
    write_raw_gaussian_ply,
)

_PROPS = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "f_rest_0",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def _raw(positions: np.ndarray) -> RawGaussianPly:
    count = len(positions)
    data = np.zeros((count, len(_PROPS)), dtype=np.float32)
    data[:, 0:3] = positions
    data[:, 5] = 1.0  # nz
    data[:, 6:9] = 0.5  # f_dc
    data[:, 9] = 0.25  # f_rest_0
    data[:, 10] = 2.0  # opacity (logit)
    data[:, 11:14] = -3.0  # log scales
    data[:, 14] = 1.0  # rot_0 (w) = identity quaternion
    return RawGaussianPly(properties=list(_PROPS), data=data)


def _write_ascii_ply(path: Path, raw: RawGaussianPly) -> None:
    header = ["ply", "format ascii 1.0", f"element vertex {len(raw)}"]
    header += [f"property float {name}" for name in raw.properties]
    header.append("end_header")
    rows = [" ".join(f"{value:.8f}" for value in row) for row in raw.data]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _sim3_example():
    angle = np.pi / 2  # 90 deg about +z
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    del angle
    return (2.0, rotation, np.array([1.0, -2.0, 0.5]))


class TestRawPlyIo:
    def test_binary_roundtrip(self, tmp_path):
        raw = _raw(np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]))
        path = write_raw_gaussian_ply(tmp_path / "out.ply", raw)
        back = read_raw_gaussian_ply(path)
        assert back.properties == raw.properties
        assert back.data == pytest.approx(raw.data, abs=1e-6)

    def test_ascii_read(self, tmp_path):
        raw = _raw(np.array([[0.5, -0.25, 4.0]]))
        _write_ascii_ply(tmp_path / "in.ply", raw)
        back = read_raw_gaussian_ply(tmp_path / "in.ply")
        assert back.data == pytest.approx(raw.data, abs=1e-5)


class TestTransformRawGaussians:
    def test_positions_scales_quats_normals(self):
        transform = _sim3_example()
        raw = _raw(np.array([[1.0, 0.0, 0.0]]))
        moved = transform_raw_gaussians(raw, transform)

        expected = apply_to_points(transform, np.array([[1.0, 0.0, 0.0]]))[0]
        assert moved.columns(["x", "y", "z"])[0] == pytest.approx(expected, abs=1e-5)
        # normal (0,0,1) rotated about z stays (0,0,1)
        assert moved.columns(["nx", "ny", "nz"])[0] == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
        # identity quaternion composes to the transform's own rotation
        expected_quat = rotation_to_quat(transform[1])
        assert moved.columns(["rot_0", "rot_1", "rot_2", "rot_3"])[0] == pytest.approx(expected_quat, abs=1e-5)
        # log scales shift by log(2)
        assert moved.column("scale_0")[0] == pytest.approx(-3.0 + np.log(2.0), abs=1e-5)
        # appearance untouched
        assert moved.column("f_dc_0")[0] == pytest.approx(0.5)
        assert raw.column("x")[0] == pytest.approx(1.0)  # source not mutated


class TestDuplicateMask:
    def test_flags_nearby_points_only(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[0.05, 0.0, 0.0], [5.0, 0.0, 0.0]])
        mask = duplicate_mask(a, b, radius=0.1)
        assert mask.tolist() == [True, False]

    def test_zero_radius_disables(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        assert duplicate_mask(a, b, radius=0.0).tolist() == [False]


class TestMergeRawGaussians:
    def test_concatenates_in_a_gauge(self):
        transform = _sim3_example()
        raw_a = _raw(np.array([[0.0, 0.0, 0.0]]))
        # B lives in its own gauge: pre-image of a known A-gauge point
        target = np.array([[3.0, 3.0, 3.0]])
        raw_b = _raw(apply_to_points(invert(transform), target))
        merged, dropped = merge_raw_gaussians(raw_a, raw_b, transform)
        assert len(merged) == 2 and dropped == 0
        assert merged.columns(["x", "y", "z"])[1] == pytest.approx(target[0], abs=1e-4)

    def test_dedup_drops_overlap(self):
        from gs_sim2real.robotics.gauge_alignment import identity_sim3

        raw_a = _raw(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        raw_b = _raw(np.array([[0.0, 0.0, 0.01], [9.0, 9.0, 9.0]]))
        merged, dropped = merge_raw_gaussians(raw_a, raw_b, identity_sim3(), dedup_radius=0.05)
        assert dropped == 1
        assert len(merged) == 3

    def test_dc_only_zeroes_b_rest_coeffs(self):
        from gs_sim2real.robotics.gauge_alignment import identity_sim3

        raw_a = _raw(np.array([[0.0, 0.0, 0.0]]))
        raw_b = _raw(np.array([[5.0, 5.0, 5.0]]))
        merged, _ = merge_raw_gaussians(raw_a, raw_b, identity_sim3(), dc_only_b=True)
        assert merged.column("f_rest_0")[0] == pytest.approx(0.25)  # A untouched
        assert merged.column("f_rest_0")[1] == pytest.approx(0.0)  # B zeroed

    def test_mismatched_layouts_raise(self):
        from gs_sim2real.robotics.gauge_alignment import identity_sim3

        raw_a = _raw(np.array([[0.0, 0.0, 0.0]]))
        raw_b = RawGaussianPly(properties=["x", "y", "z"], data=np.zeros((1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="property layouts"):
            merge_raw_gaussians(raw_a, raw_b, identity_sim3())


def _write_session(session: Path, raw: RawGaussianPly, centers, rotations, names) -> None:
    from tests.test_change_detection import _write_images_txt

    (session / "keyframes").mkdir(parents=True)
    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"lastSuccessfulRound": {"round": 1}}), encoding="utf-8")
    sparse = session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n", encoding="utf-8")
    _write_images_txt(sparse / "images.txt", names, centers, rotations)
    train = session / "rounds" / "round_001" / "train"
    train.mkdir(parents=True)
    write_raw_gaussian_ply(train / "point_cloud.ply", raw)


class TestMergeSessions:
    def _make_pair(self, tmp_path):
        transform = _sim3_example()
        inverse = invert(transform)
        names = [f"kf_{i:06d}.jpg" for i in range(4)]
        centers_a = np.stack([np.zeros(4), np.full(4, -1.5), np.arange(4.0)], axis=1)
        rotations_a = [np.eye(3)] * 4
        points_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
        points_b_world = np.array([[0.5, -0.5, 1.0], [2.0, 0.0, 3.0]])

        session_a = tmp_path / "session_a"
        _write_session(session_a, _raw(points_a), centers_a, rotations_a, names)
        session_b = tmp_path / "session_b"
        _write_session(
            session_b,
            _raw(apply_to_points(inverse, points_b_world)),
            apply_to_points(inverse, centers_a),
            [inverse[1] @ r for r in rotations_a],
            names,
        )
        return session_a, session_b, points_b_world

    def test_merge_across_gauges(self, tmp_path):
        session_a, session_b, points_b_world = self._make_pair(tmp_path)
        stats = merge_sessions(session_a, session_b, tmp_path / "merged.ply", align="shared")
        assert stats["merged"] == 4 and stats["deduplicated"] == 0
        assert stats["alignment"]["scale"] == pytest.approx(2.0, rel=1e-3)
        merged = read_raw_gaussian_ply(tmp_path / "merged.ply")
        assert merged.columns(["x", "y", "z"])[2] == pytest.approx(points_b_world[0], abs=1e-3)

    def test_cli_merge_maps(self, tmp_path, capsys):
        from gs_sim2real import cli

        session_a, session_b, _ = self._make_pair(tmp_path)
        args = cli.build_parser().parse_args(
            [
                "merge-maps",
                "--map-a",
                str(session_a),
                "--map-b",
                str(session_b),
                "--align",
                "shared",
                "--output",
                str(tmp_path / "merged.ply"),
            ]
        )
        cli.cmd_merge_maps(args)
        out = capsys.readouterr().out
        assert "Merged 2 (A) + 2 (B)" in out
        assert (tmp_path / "merged.ply").is_file()
