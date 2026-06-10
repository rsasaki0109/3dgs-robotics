"""Runtime Sim3 gauge alignment: gauge_alignment.py + the Step-1 live-mapping wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.robotics.gauge_alignment import (
    RoundPoses,
    SessionGaugeChain,
    apply_to_points,
    compose,
    invert,
    quat_to_rotation,
    read_gauge_transform,
    rotation_to_quat,
    write_gauge_transform,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quat_to_rotation(q)


def _world_to_gauge(points: np.ndarray, gauge) -> np.ndarray:
    scale, rotation, translation = gauge
    return (points - translation) @ rotation / scale


def _write_images_txt(path: Path, names: list[str], centers: np.ndarray, rotations_wc: np.ndarray) -> None:
    """Write COLMAP image lines for world-from-camera rotations + camera centers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for i, name in enumerate(names):
        r_cw = rotations_wc[i].T
        qw, qx, qy, qz = rotation_to_quat(r_cw)
        t = -r_cw @ centers[i]
        lines.append(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestSim3Primitives:
    def test_compose_with_inverse_is_identity(self) -> None:
        rng = np.random.default_rng(11)
        transform = (1.7, _random_rotation(rng), rng.normal(size=3))
        scale, rotation, translation = compose(transform, invert(transform))
        np.testing.assert_allclose(scale, 1.0, rtol=1e-12)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-12)

    def test_apply_to_points_matches_definition(self) -> None:
        rng = np.random.default_rng(5)
        scale, rotation, translation = 0.5, _random_rotation(rng), rng.normal(size=3)
        points = rng.normal(size=(7, 3))
        expected = scale * (rotation @ points.T).T + translation
        np.testing.assert_allclose(apply_to_points((scale, rotation, translation), points), expected, atol=1e-12)

    def test_quaternion_roundtrip(self) -> None:
        rng = np.random.default_rng(3)
        rotation = _random_rotation(rng)
        np.testing.assert_allclose(quat_to_rotation(rotation_to_quat(rotation)), rotation, atol=1e-9)


class TestSessionGaugeChain:
    def test_chains_rounds_onto_first_round_gauge(self) -> None:
        rng = np.random.default_rng(21)
        world = rng.normal(size=(6, 3)) * 4.0
        world_rot = np.stack([_random_rotation(rng) for _ in range(6)])
        names = [f"kf_{i:06d}.jpg" for i in range(6)]
        gauges = [(float(rng.uniform(0.5, 2.0)), _random_rotation(rng), rng.normal(size=3)) for _ in range(3)]
        ids_per_round = [[0, 1, 2], [1, 2, 3, 4], [3, 4, 5]]

        chain = SessionGaugeChain()
        results = []
        for gauge, ids in zip(gauges, ids_per_round):
            poses = RoundPoses(
                names=[names[i] for i in ids],
                centers=_world_to_gauge(world[ids], gauge),
                rotations=np.einsum("ij,njk->nik", gauge[1].T, world_rot[ids]),
            )
            results.append((poses, *chain.update(poses)))

        # first round defines the session gauge
        _, transform0, rebased0, shared0 = results[0]
        assert not rebased0 and shared0 == 0
        np.testing.assert_allclose(transform0[0], 1.0)

        # round 3 never shares a camera with round 1 directly, yet the chain
        # must land its centers in round 1's gauge
        poses3, transform3, rebased3, shared3 = results[2]
        assert not rebased3 and shared3 == 2
        aligned = apply_to_points(transform3, poses3.centers)
        expected = _world_to_gauge(world[ids_per_round[2]], gauges[0])
        np.testing.assert_allclose(aligned, expected, atol=1e-8)

    def test_rebases_when_rounds_share_too_few_cameras(self) -> None:
        rng = np.random.default_rng(8)

        def poses(names: list[str]) -> RoundPoses:
            n = len(names)
            return RoundPoses(
                names=names,
                centers=rng.normal(size=(n, 3)),
                rotations=np.stack([_random_rotation(rng) for _ in range(n)]),
            )

        chain = SessionGaugeChain()
        chain.update(poses(["a.jpg", "b.jpg", "c.jpg"]))
        transform, rebased, shared = chain.update(poses(["x.jpg", "y.jpg", "z.jpg"]))
        assert rebased and shared == 0
        np.testing.assert_allclose(transform[0], 1.0)
        np.testing.assert_allclose(transform[1], np.eye(3))

    def test_gauge_transform_json_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        transform = (2.5, _random_rotation(rng), rng.normal(size=3))
        write_gauge_transform(tmp_path, transform, rebased=False, shared_cameras=7)
        loaded, rebased = read_gauge_transform(tmp_path)
        assert not rebased
        np.testing.assert_allclose(loaded[0], transform[0])
        np.testing.assert_allclose(loaded[1], transform[1])
        np.testing.assert_allclose(loaded[2], transform[2])
        assert read_gauge_transform(tmp_path / "missing") is None


class TestSplatRebuilderGaugeWiring:
    def _make_round(self, round_dir: Path, names: list[str], centers: np.ndarray, rotations: np.ndarray) -> None:
        _write_images_txt(round_dir / "sparse_input" / "sparse" / "0" / "images.txt", names, centers, rotations)

    def test_align_round_gauge_chains_and_persists(self, tmp_path: Path) -> None:
        from gs_sim2real.robotics.live_mapping import LiveMapperConfig, SplatRebuilder

        rng = np.random.default_rng(31)
        world = rng.normal(size=(4, 3)) * 3.0
        world_rot = np.stack([_random_rotation(rng) for _ in range(4)])
        names = [f"kf_{i:06d}.jpg" for i in range(4)]
        gauge2 = (1.8, _random_rotation(rng), rng.normal(size=3))

        round1 = tmp_path / "rounds" / "round_001"
        round2 = tmp_path / "rounds" / "round_002"
        self._make_round(round1, names[:3], world[:3], world_rot[:3])
        self._make_round(
            round2, names, _world_to_gauge(world, gauge2), np.einsum("ij,njk->nik", gauge2[1].T, world_rot)
        )

        builder = SplatRebuilder(LiveMapperConfig(workdir=tmp_path))
        transform1, rebased1 = builder._align_round_gauge(round1)
        assert not rebased1
        np.testing.assert_allclose(transform1[0], 1.0)

        transform2, rebased2 = builder._align_round_gauge(round2)
        assert not rebased2
        aligned = apply_to_points(transform2, _world_to_gauge(world, gauge2))
        np.testing.assert_allclose(aligned, world, atol=1e-8)

        loaded, _ = read_gauge_transform(round2)
        np.testing.assert_allclose(loaded[0], transform2[0])
        assert (round1 / "gauge_transform.json").is_file()

    def test_align_round_gauge_rebases_on_missing_poses(self, tmp_path: Path) -> None:
        from gs_sim2real.robotics.live_mapping import LiveMapperConfig, SplatRebuilder

        round_dir = tmp_path / "rounds" / "round_001"
        round_dir.mkdir(parents=True)
        builder = SplatRebuilder(LiveMapperConfig(workdir=tmp_path))
        transform, rebased = builder._align_round_gauge(round_dir)
        assert rebased
        np.testing.assert_allclose(transform[0], 1.0)
        _, rebased_flag = read_gauge_transform(round_dir)
        assert rebased_flag


class _TwoGaussianPly:
    def __init__(self) -> None:
        self.positions = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        self.scales = np.log(np.asarray([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1]], dtype=np.float32))
        self.rotations = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32)
        self.opacities = np.asarray([4.0, 4.0], dtype=np.float32)
        self.colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _read_splat(path: Path) -> np.ndarray:
    dtype = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    return np.frombuffer(path.read_bytes(), dtype=dtype)


class TestPlyToSplatSimilarity:
    @pytest.fixture(autouse=True)
    def _patch_ply(self, monkeypatch):
        import gs_sim2real.viewer.web_viewer as web_viewer

        monkeypatch.setattr(web_viewer, "load_ply", lambda _: _TwoGaussianPly())

    def test_similarity_transform_moves_positions_scales_and_quats(self, tmp_path: Path) -> None:
        from gs_sim2real.viewer.web_export import ply_to_splat

        # 90 deg about +z, scale 2, shift +x
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transform = (2.0, rotation, np.array([1.0, 0.0, 0.0]))
        out = tmp_path / "scene.splat"
        ply_to_splat("ignored.ply", out, similarity_transform=transform)

        records = _read_splat(out)
        positions = np.sort(records["pos"], axis=0)
        expected = np.sort(np.asarray([[1.0, 2.0, 0.0], [-3.0, 0.0, 0.0]], dtype=np.float32), axis=0)
        np.testing.assert_allclose(positions, expected, atol=1e-5)
        np.testing.assert_allclose(np.sort(records["scale"][:, 0]), [0.2, 0.4], atol=1e-5)

        quat = (records["rot"][0].astype(np.float64) - 128.0) / 128.0
        half = np.sqrt(0.5)
        np.testing.assert_allclose(quat, [half, 0.0, 0.0, half], atol=2.0 / 128.0)

    def test_normalize_params_apply_fixed_frame(self, tmp_path: Path) -> None:
        from gs_sim2real.viewer.web_export import ply_to_splat

        out = tmp_path / "scene.splat"
        ply_to_splat("ignored.ply", out, normalize_params=(np.array([1.0, 0.0, 0.0]), 2.0))
        records = _read_splat(out)
        positions = {tuple(np.round(p, 5)) for p in records["pos"]}
        assert positions == {(0.0, 0.0, 0.0), (-0.5, 1.0, 0.0)}
        np.testing.assert_allclose(np.sort(records["scale"][:, 0]), [0.05, 0.1], atol=1e-5)

    def test_normalize_target_extent_and_params_are_exclusive(self, tmp_path: Path) -> None:
        from gs_sim2real.viewer.web_export import ply_to_splat

        with pytest.raises(ValueError, match="not both"):
            ply_to_splat(
                "ignored.ply",
                tmp_path / "scene.splat",
                normalize_target_extent=17.0,
                normalize_params=(np.zeros(3), 1.0),
            )

    def test_compute_splat_normalization_matches_legacy_branch(self) -> None:
        from gs_sim2real.viewer.web_export import compute_splat_normalization

        positions = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        centroid, factor = compute_splat_normalization(positions, target_extent=5.0)
        np.testing.assert_allclose(centroid, [5.0, 0.0, 0.0])
        np.testing.assert_allclose(factor, 2.0)
