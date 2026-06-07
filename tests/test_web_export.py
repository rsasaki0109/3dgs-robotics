"""Tests for PLY to web format conversion."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from gs_sim2real.viewer.web_export import (
    filter_splat_file,
    inspect_splat_file,
    ply_to_splat,
    ply_to_binary,
    ply_to_json,
    ply_to_scene_bundle,
    points_to_scene_bundle,
    splat_to_tile_catalog,
)


def _write_ascii_ply(path: Path, positions: list[list[float]], colors: list[list[int]]) -> Path:
    """Write a minimal ASCII PLY file with positions and RGB colors."""
    n = len(positions)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    lines = []
    for pos, col in zip(positions, colors):
        lines.append(f"{pos[0]} {pos[1]} {pos[2]} {col[0]} {col[1]} {col[2]}")

    path.write_text(header + "\n".join(lines) + "\n")
    return path


def _pack_splat_record(
    xyz: tuple[float, float, float],
    scale: tuple[float, float, float],
    alpha: int,
) -> bytes:
    return struct.pack("<ffffffBBBBBBBB", *xyz, *scale, 255, 255, 255, alpha, 128, 128, 128, 128)


class _GaussianPly:
    def __init__(self) -> None:
        self.positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.scales = np.log(
            np.asarray(
                [
                    [0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.2],
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                ],
                dtype=np.float32,
            )
        )
        self.rotations = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
        self.opacities = np.asarray([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        self.colors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )


class TestPlyToJson:
    """Tests for ply_to_json conversion."""

    def test_creates_valid_json(self, tmp_path: Path) -> None:
        """ply_to_json creates a valid JSON file with correct structure."""
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)
        out_path = tmp_path / "out.json"

        result = ply_to_json(str(ply_path), str(out_path))

        assert Path(result).exists()
        with open(result) as f:
            data = json.load(f)

        assert "positions" in data
        assert "colors" in data
        assert "count" in data
        assert "bounds" in data
        assert data["count"] == 3
        assert len(data["positions"]) == 9  # 3 points * 3 coords
        assert len(data["colors"]) == 9
        assert "min" in data["bounds"]
        assert "max" in data["bounds"]
        assert len(data["bounds"]["min"]) == 3
        assert len(data["bounds"]["max"]) == 3

    def test_subsampling(self, tmp_path: Path) -> None:
        """ply_to_json subsamples when max_points is smaller than point count."""
        positions = [[float(i), float(i), float(i)] for i in range(20)]
        colors = [[128, 128, 128]] * 20
        ply_path = _write_ascii_ply(tmp_path / "many.ply", positions, colors)
        out_path = tmp_path / "sub.json"

        result = ply_to_json(str(ply_path), str(out_path), max_points=5)

        with open(result) as f:
            data = json.load(f)

        assert data["count"] == 5
        assert len(data["positions"]) == 15  # 5 * 3
        assert len(data["colors"]) == 15

    def test_bounds_correct(self, tmp_path: Path) -> None:
        """Bounds min/max match the actual point positions."""
        positions = [[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]
        colors = [[255, 255, 255], [0, 0, 0]]
        ply_path = _write_ascii_ply(tmp_path / "bounds.ply", positions, colors)
        out_path = tmp_path / "bounds.json"

        ply_to_json(str(ply_path), str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        np.testing.assert_allclose(data["bounds"]["min"], [0.0, -1.0, -5.0], atol=1e-5)
        np.testing.assert_allclose(data["bounds"]["max"], [3.0, 4.0, 2.0], atol=1e-5)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """ply_to_json creates parent directories for the output path."""
        positions = [[1.0, 2.0, 3.0]]
        colors = [[128, 128, 128]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)
        out_path = tmp_path / "nested" / "dir" / "out.json"

        result = ply_to_json(str(ply_path), str(out_path))
        assert Path(result).exists()


class TestPlyToBinary:
    """Tests for ply_to_binary conversion."""

    def test_creates_binary_with_correct_header(self, tmp_path: Path) -> None:
        """ply_to_binary creates a binary file with correct num_points and bounds."""
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        colors = [[255, 0, 0], [0, 255, 0]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)
        out_path = tmp_path / "out.bin"

        result = ply_to_binary(str(ply_path), str(out_path))
        assert Path(result).exists()

        with open(result, "rb") as f:
            # Read num_points
            (num_points,) = struct.unpack("<I", f.read(4))
            assert num_points == 2

            # Read bounds (6 floats: min_x, min_y, min_z, max_x, max_y, max_z)
            bounds = struct.unpack("<6f", f.read(24))
            np.testing.assert_allclose(bounds[:3], [1.0, 2.0, 3.0], atol=1e-5)
            np.testing.assert_allclose(bounds[3:], [4.0, 5.0, 6.0], atol=1e-5)

            # Read point data: 2 points * 6 floats each
            point_data = struct.unpack(f"<{2 * 6}f", f.read(2 * 24))
            # First point: x, y, z, r, g, b
            np.testing.assert_allclose(point_data[0:3], [1.0, 2.0, 3.0], atol=1e-5)
            # Colors are normalized 0-1
            np.testing.assert_allclose(point_data[3], 1.0, atol=1e-2)  # red=255 -> 1.0

    def test_file_size(self, tmp_path: Path) -> None:
        """Binary file has expected size: 4 + 24 + N*24 bytes."""
        n = 10
        positions = [[float(i), float(i), float(i)] for i in range(n)]
        colors = [[128, 128, 128]] * n
        ply_path = _write_ascii_ply(tmp_path / "size.ply", positions, colors)
        out_path = tmp_path / "size.bin"

        result = ply_to_binary(str(ply_path), str(out_path))
        expected_size = 4 + 24 + n * 24
        assert Path(result).stat().st_size == expected_size

    def test_subsampling(self, tmp_path: Path) -> None:
        """ply_to_binary subsamples when max_points is smaller than point count."""
        positions = [[float(i), float(i), float(i)] for i in range(20)]
        colors = [[128, 128, 128]] * 20
        ply_path = _write_ascii_ply(tmp_path / "many.ply", positions, colors)
        out_path = tmp_path / "sub.bin"

        result = ply_to_binary(str(ply_path), str(out_path), max_points=5)

        with open(result, "rb") as f:
            (num_points,) = struct.unpack("<I", f.read(4))
            assert num_points == 5

        expected_size = 4 + 24 + 5 * 24
        assert Path(result).stat().st_size == expected_size


class TestPlyToSplat:
    """Tests for antimatter15 .splat export."""

    def test_adaptive_scale_percentile_filters_giant_gaussians(self, tmp_path: Path, monkeypatch) -> None:
        """Scale percentile filtering should remove giant blurry gaussians without fixed metric assumptions."""
        import gs_sim2real.viewer.web_viewer as web_viewer

        monkeypatch.setattr(web_viewer, "load_ply", lambda _: _GaussianPly())
        out = tmp_path / "scene.splat"

        result = ply_to_splat("ignored.ply", out, max_scale_percentile=50.0)

        assert result == str(out)
        assert out.stat().st_size == 2 * 32

    def test_adaptive_scale_percentile_rejects_invalid_range(self, tmp_path: Path, monkeypatch) -> None:
        import pytest
        import gs_sim2real.viewer.web_viewer as web_viewer

        monkeypatch.setattr(web_viewer, "load_ply", lambda _: _GaussianPly())
        with pytest.raises(ValueError, match="max_scale_percentile"):
            ply_to_splat("ignored.ply", tmp_path / "scene.splat", max_scale_percentile=0.0)


class TestFilterSplatFile:
    """Tests for direct browser .splat cleanup."""

    def test_filters_low_opacity_and_large_scales(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.splat"
        src.write_bytes(
            b"".join(
                [
                    _pack_splat_record((0, 0, 0), (0.1, 0.1, 0.1), 255),
                    _pack_splat_record((1, 0, 0), (0.2, 0.2, 0.2), 8),
                    _pack_splat_record((2, 0, 0), (5.0, 5.0, 5.0), 255),
                    _pack_splat_record((3, 0, 0), (0.3, 0.3, 0.3), 255),
                ]
            )
        )
        out = tmp_path / "clean.splat"

        report = filter_splat_file(src, out, min_opacity=0.05, max_scale_percentile=75.0)

        assert report.input_count == 4
        assert report.output_count == 2
        assert report.kept_ratio == 0.5
        assert out.stat().st_size == 2 * 32

    def test_rejects_non_splat_size(self, tmp_path: Path) -> None:
        import pytest

        src = tmp_path / "bad.splat"
        src.write_bytes(b"not a splat")
        with pytest.raises(ValueError, match="32-byte"):
            filter_splat_file(src, tmp_path / "out.splat")


class TestInspectSplatFile:
    """Tests for direct browser .splat health reports."""

    def test_reports_opacity_and_scale_stats(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.splat"
        src.write_bytes(
            b"".join(
                [
                    _pack_splat_record((0, 0, 0), (0.1, 0.1, 0.1), 255),
                    _pack_splat_record((1, 0, 0), (0.2, 0.2, 0.2), 4),
                    _pack_splat_record((2, 0, 0), (5.0, 5.0, 5.0), 128),
                    _pack_splat_record((3, 0, 0), (0.3, 0.3, 0.3), 64),
                ]
            )
        )

        report = inspect_splat_file(src, low_opacity_threshold=0.05)

        assert report.input_count == 4
        assert report.size_bytes == 4 * 32
        assert report.low_opacity_count == 1
        assert report.low_opacity_ratio == 0.25
        assert report.opacity_min == 4 / 255
        assert report.opacity_max == 1.0
        assert report.scale_max == 5.0
        assert report.scale_tail_ratio >= 1.0
        assert report.as_dict()["input_count"] == 4

    def test_rejects_empty_splat(self, tmp_path: Path) -> None:
        import pytest

        src = tmp_path / "empty.splat"
        src.write_bytes(b"")
        with pytest.raises(ValueError, match="does not contain any splats"):
            inspect_splat_file(src)


class TestSplatToTileCatalog:
    """Tests for tiling an existing browser .splat into dynamic-map assets."""

    def test_writes_tile_splats_and_catalog(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.splat"
        src.write_bytes(
            b"".join(
                [
                    _pack_splat_record((0, 0, 0), (0.1, 0.1, 0.1), 255),
                    _pack_splat_record((4, 0, 0), (0.1, 0.1, 0.1), 255),
                    _pack_splat_record((9, 0, 0), (0.1, 0.1, 0.1), 255),
                    _pack_splat_record((12, 0, 0), (0.1, 0.1, 0.1), 255),
                ]
            )
        )
        catalog_path = tmp_path / "public" / "manifests" / "demo-catalog.json"

        catalog = splat_to_tile_catalog(
            src,
            catalog_path,
            public_root=tmp_path / "public",
            scene_id="Demo Tiles",
            label="Demo Tiles",
            tile_size=10,
            overlap=1,
            axes="xz",
            min_splats=1,
        )

        assert catalog["type"] == "large-scale-3dgs-tile-catalog"
        assert catalog["sceneId"] == "demo-tiles"
        assert catalog["summary"]["inputSplatCount"] == 4
        assert catalog["summary"]["tileCount"] == 2
        assert [tile["coreSplatCount"] for tile in catalog["tiles"]] == [3, 1]
        assert [tile["splatCount"] for tile in catalog["tiles"]] == [3, 2]
        assert catalog["tiles"][0]["splatUrl"] == "/splats/demo-tiles/tile_x000_z000.splat"
        assert Path(catalog["tiles"][0]["publicPath"]).stat().st_size == 3 * 32
        assert Path(catalog["tiles"][1]["publicPath"]).stat().st_size == 2 * 32
        assert json.loads(catalog_path.read_text(encoding="utf-8"))["summary"]["tileCount"] == 2


class TestPlyToSceneBundle:
    """Tests for GitHub Pages scene bundle export."""

    def test_creates_scene_manifest_and_binary_asset(self, tmp_path: Path) -> None:
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 2.5, 1.0]]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ply_path = _write_ascii_ply(tmp_path / "bundle.ply", positions, colors)
        output_dir = tmp_path / "bundle"

        manifest_path = ply_to_scene_bundle(
            str(ply_path),
            str(output_dir),
            asset_format="binary",
            scene_id="demo-room",
            label="Demo Room",
            description="Tiny binary demo scene",
        )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["type"] == "web-scene-manifest"
        assert manifest["sceneId"] == "demo-room"
        assert manifest["label"] == "Demo Room"
        assert manifest["asset"]["format"] == "binary"
        assert manifest["asset"]["href"] == "demo-room.points.bin"
        assert (output_dir / "demo-room.points.bin").exists()
        assert manifest["camera"]["target"] == [4.0, 3.5, 3.5]

    def test_creates_scene_manifest_and_json_asset(self, tmp_path: Path) -> None:
        positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        colors = [[255, 255, 255], [0, 0, 0]]
        ply_path = _write_ascii_ply(tmp_path / "bundle-json.ply", positions, colors)
        output_dir = tmp_path / "bundle-json"

        manifest_path = ply_to_scene_bundle(
            str(ply_path),
            str(output_dir),
            asset_format="json",
            scene_id="json-scene",
        )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        asset = json.loads((output_dir / "json-scene.points.json").read_text(encoding="utf-8"))
        assert manifest["asset"]["format"] == "json"
        assert asset["count"] == 2


class TestPointsToSceneBundle:
    """Tests for direct points/colors scene bundle export."""

    def test_creates_manifest_from_arrays(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "array-bundle"
        positions = np.asarray(
            [
                [-1.0, -0.5, -2.0],
                [0.0, 0.25, -1.5],
                [1.0, 0.75, -1.0],
            ],
            dtype=np.float32,
        )
        colors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        camera = {
            "position": [0.0, 0.0, 4.0],
            "target": [0.0, 0.0, -1.5],
            "up": [0.0, 1.0, 0.0],
        }

        manifest_path = points_to_scene_bundle(
            positions,
            colors,
            str(output_dir),
            asset_format="binary",
            scene_id="array-scene",
            label="Array Scene",
            description="Direct array export",
            camera=camera,
        )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["sceneId"] == "array-scene"
        assert manifest["label"] == "Array Scene"
        assert manifest["count"] == 3
        assert manifest["camera"] == camera
        assert (output_dir / "array-scene.points.bin").exists()
