"""Tests for the nav2 occupancy-grid export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.robotics.occupancy_grid import (
    GridParams,
    build_occupancy_grid,
    camera_axes,
    export_occupancy_grid,
    write_map_files,
)

# Synthetic scene in the optical convention of a forward-driving camera:
# identity camera orientation means y is down, so world up is (0, -1, 0).
# The ground plane is y=0, cameras drive along +z at y=-1.5, and a wall
# stands at x=2 between z=1 and z=3.
_UP = np.array([0.0, -1.0, 0.0])
_IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


def _synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    xs = np.linspace(-2.0, 2.0, 21)
    zs = np.linspace(0.0, 4.0, 21)
    gx, gz = np.meshgrid(xs, zs)
    ground = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)

    wy = np.linspace(-0.4, -2.0, 9)
    wz = np.linspace(1.0, 3.0, 9)
    gy, gz2 = np.meshgrid(wy, wz)
    wall = np.stack([np.full(gy.size, 2.0), gy.ravel(), gz2.ravel()], axis=1)

    points = np.vstack([ground, wall])
    opacities = np.ones(len(points))
    camera_centers = np.stack([np.zeros(5), np.full(5, -1.5), np.arange(5.0)], axis=1)
    qvecs = [_IDENTITY_Q] * 5
    return points, opacities, camera_centers, qvecs


def _cell_value(grid, world_point) -> int:
    xy = np.asarray(world_point, dtype=np.float64) @ grid.basis[:2].T
    col = int(np.floor((xy[0] - grid.origin[0]) / grid.resolution))
    row = int(np.floor((xy[1] - grid.origin[1]) / grid.resolution))
    return int(grid.data[row, col])


class TestCameraAxes:
    def test_identity_orientation_is_y_down_z_forward(self):
        up, forward = camera_axes(_IDENTITY_Q)
        assert up == pytest.approx((0.0, -1.0, 0.0))
        assert forward == pytest.approx((0.0, 0.0, 1.0))


class TestBuildOccupancyGrid:
    def test_ground_frame_estimation(self):
        grid = build_occupancy_grid(*_synthetic_scene())
        assert grid.up == pytest.approx(_UP, abs=1e-9)
        assert grid.camera_height == pytest.approx(1.5, abs=0.05)
        assert grid.ground_height == pytest.approx(0.0, abs=0.05)
        assert grid.resolution == pytest.approx(grid.camera_height / 20.0)

    def test_wall_occupied_trajectory_free_far_corner_unknown(self):
        points, opacities, centers, qvecs = _synthetic_scene()
        grid = build_occupancy_grid(points, opacities, centers, qvecs)
        assert _cell_value(grid, (2.0, -1.0, 2.0)) == 100  # wall
        assert _cell_value(grid, (0.0, -1.5, 2.0)) == 0  # camera path
        assert _cell_value(grid, (0.0, -1.5, 2.5)) == 0  # swept corridor between cameras
        assert _cell_value(grid, (-2.0, 0.0, 2.0)) == 0  # ground evidence
        # inside the padding margin but away from any evidence -> unknown
        assert _cell_value(grid, (-2.0 - 0.9 * grid.camera_height, -1.0, 2.0)) == -1

    def test_transparent_gaussians_are_ignored(self):
        points, opacities, centers, qvecs = _synthetic_scene()
        opacities[:] = 0.05  # everything below min_opacity
        grid = build_occupancy_grid(points, opacities, centers, qvecs)
        assert grid.occupied_cells == 0

    def test_obstacle_band_excludes_overhangs(self):
        points, opacities, centers, qvecs = _synthetic_scene()
        params = GridParams(obstacle_band=(0.2, 0.5))  # wall tops out above 0.5 * 1.5
        grid = build_occupancy_grid(points, opacities, centers, qvecs, params=params)
        low = build_occupancy_grid(points, opacities, centers, qvecs)
        assert 0 < grid.occupied_cells <= low.occupied_cells

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            build_occupancy_grid(np.zeros((0, 3)), np.zeros(0), np.zeros((1, 3)), [_IDENTITY_Q])


class TestWriteMapFiles:
    def test_writes_pgm_yaml_json(self, tmp_path):
        grid = build_occupancy_grid(*_synthetic_scene())
        yaml_path, pgm_path, json_path = write_map_files(grid, tmp_path / "map.yaml")
        assert yaml_path.is_file() and pgm_path.is_file() and json_path.is_file()

        height, width = grid.data.shape
        payload = pgm_path.read_bytes()
        assert payload.startswith(f"P5\n{width} {height}\n255\n".encode("ascii"))
        image = np.frombuffer(payload.split(b"\n255\n", 1)[1], dtype=np.uint8).reshape(height, width)
        # row 0 of the PGM is max grid-y (flipped relative to grid.data)
        assert np.array_equal(image[-1] == 0, grid.data[0] == 100)

        text = yaml_path.read_text(encoding="utf-8")
        assert f"image: {pgm_path.name}" in text
        assert "mode: trinary" in text
        assert f"resolution: {grid.resolution:.6f}" in text

        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        assert sidecar["camera_height"] == pytest.approx(grid.camera_height)
        assert np.asarray(sidecar["basis"]).shape == (3, 3)

    def test_rejects_non_yaml_output(self, tmp_path):
        grid = build_occupancy_grid(*_synthetic_scene())
        with pytest.raises(ValueError):
            write_map_files(grid, tmp_path / "map.pgm")


def _write_fake_session(session: Path) -> None:
    (session / "keyframes").mkdir(parents=True)
    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"lastSuccessfulRound": {"round": 1}}), encoding="utf-8")
    sparse = session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n", encoding="utf-8")
    lines = []
    for i in range(4):
        # identity orientation, COLMAP center = -tvec -> cameras at y=-1.5 along +z
        lines.append(f"{i + 1} 1 0 0 0 0 1.5 {-float(i)} 1 kf_{i:06d}.jpg\n\n")
    (sparse / "images.txt").write_text("".join(lines), encoding="utf-8")

    points, _, _, _ = _synthetic_scene()
    train = session / "rounds" / "round_001" / "train"
    train.mkdir(parents=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x} {y} {z}" for x, y, z in points]
    (train / "point_cloud.ply").write_text("\n".join(header + body) + "\n", encoding="utf-8")


class TestExportAndCli:
    def test_export_occupancy_grid_from_session(self, tmp_path):
        session = tmp_path / "session"
        _write_fake_session(session)
        grid, yaml_path = export_occupancy_grid(session, tmp_path / "nav2" / "map.yaml")
        assert yaml_path.is_file()
        assert grid.camera_height == pytest.approx(1.5, abs=0.05)
        assert grid.occupied_cells > 0

    def test_cli_export_grid(self, tmp_path, capsys):
        from gs_sim2real import cli

        session = tmp_path / "session"
        _write_fake_session(session)
        args = cli.build_parser().parse_args(
            ["export-grid", "--map", str(session), "--output", str(tmp_path / "map.yaml")]
        )
        cli.cmd_export_grid(args)
        out = capsys.readouterr().out
        assert (tmp_path / "map.pgm").is_file()
        assert "nav2_map_server" in out

    def test_cli_rejects_bad_band(self, tmp_path):
        from gs_sim2real import cli

        args = cli.build_parser().parse_args(
            ["export-grid", "--map", "x", "--output", "map.yaml", "--obstacle-band", "2.0,0.2"]
        )
        with pytest.raises(SystemExit):
            cli.cmd_export_grid(args)
