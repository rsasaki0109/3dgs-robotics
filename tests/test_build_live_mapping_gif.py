"""Gauge-alignment math of scripts/build_live_mapping_gif.py (CPU only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gif_module():
    path = REPO_ROOT / "scripts" / "build_live_mapping_gif.py"
    spec = importlib.util.spec_from_file_location("build_live_mapping_gif", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_parse_images_txt_reads_every_image_line(gif_module, tmp_path) -> None:
    """The COLMAP writer emits an empty POINTS2D line per image; none may be skipped."""
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1.0 0.0 0.0 0.0 0.5 0.0 0.0 1 kf_000000.jpg\n"
        "\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.5 0.0 1 kf_000002.jpg\n"
        "\n"
        "3 1.0 0.0 0.0 0.0 0.0 0.0 0.5 2 kf_000004.jpg\n"
        "\n",
        encoding="utf-8",
    )
    names, centers, rotations = gif_module._parse_images_txt(images_txt)
    assert names == ["kf_000000.jpg", "kf_000002.jpg", "kf_000004.jpg"]
    assert centers.shape == (3, 3)
    assert rotations.shape == (3, 3, 3)
    # identity quaternion: center = -t
    np.testing.assert_allclose(centers[0], [-0.5, 0.0, 0.0], atol=1e-12)


def test_similarity_from_poses_recovers_known_transform(gif_module) -> None:
    rng = np.random.default_rng(7)
    rotation_true = _random_rotation(rng)
    scale_true, t_true = 1.7, np.array([0.4, -2.0, 3.1])
    src_centers = rng.normal(size=(2, 3))  # two shared cameras must be enough
    src_rotations = np.stack([_random_rotation(rng) for _ in range(2)])
    dst_centers = src_centers @ rotation_true.T * scale_true + t_true
    dst_rotations = np.einsum("ij,njk->nik", rotation_true, src_rotations)

    scale, rotation, translation = gif_module.similarity_from_poses(
        src_centers, src_rotations, dst_centers, dst_rotations
    )
    np.testing.assert_allclose(scale, scale_true, rtol=1e-9)
    np.testing.assert_allclose(rotation, rotation_true, atol=1e-9)
    np.testing.assert_allclose(translation, t_true, atol=1e-9)


def test_align_to_anchor_chains_through_intermediate_round(gif_module) -> None:
    """Round 1 shares no cameras with round 3 directly; the chain must still align it."""
    rng = np.random.default_rng(3)
    world = rng.normal(size=(6, 3)) * 4.0
    world_rot = np.stack([_random_rotation(rng) for _ in range(6)])
    names = [f"kf_{i:06d}.jpg" for i in range(6)]

    def world_to_gauge(points: np.ndarray, gauge: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        scale, rotation, translation = gauge
        return (points - translation) @ rotation / scale

    def make_round(index: int, ids: list[int], gauge: tuple[float, np.ndarray, np.ndarray]):
        _scale, rotation, _translation = gauge
        centers = world_to_gauge(world[ids], gauge)
        rotations = np.einsum("ij,njk->nik", rotation.T, world_rot[ids])
        return gif_module.RoundData(index, Path("unused.ply"), [names[i] for i in ids], centers, rotations)

    gauges = [
        (float(rng.uniform(0.5, 2.0)), _random_rotation(rng), rng.normal(size=3)) for _ in range(3)
    ]
    rounds = [
        make_round(1, [0, 1], gauges[0]),
        make_round(2, [0, 1, 2, 3], gauges[1]),
        make_round(3, [2, 3, 4, 5], gauges[2]),
    ]
    transforms = gif_module.align_to_anchor(rounds)

    scale_last, rot_last, t_last = transforms[-1]
    np.testing.assert_allclose(scale_last, 1.0)
    np.testing.assert_allclose(rot_last, np.eye(3))
    np.testing.assert_allclose(t_last, np.zeros(3))

    # round 1's cameras, pushed through the chain, must land in round 3's gauge
    scale, rotation, translation = transforms[0]
    aligned = rounds[0].centers @ rotation.T * scale + translation
    expected = world_to_gauge(world[[0, 1]], gauges[2])
    np.testing.assert_allclose(aligned, expected, atol=1e-8)
