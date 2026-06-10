"""Round-level Sim3 pose graph (loop correction over session-gauge transforms)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gs_sim2real.robotics.gauge_alignment import (
    RoundPoses,
    apply_to_points,
    compose,
    quat_to_rotation,
)
from gs_sim2real.robotics.gauge_pose_graph import (
    Sim3Edge,
    build_round_edges,
    optimize_session_transforms,
    rotation_exp,
    rotation_log,
    sim3_from_params,
    sim3_to_params,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quat_to_rotation(q)


def _world_to_gauge(points: np.ndarray, gauge) -> np.ndarray:
    scale, rotation, translation = gauge
    return (points - translation) @ rotation / scale


def _gauge_to_session(gauge) -> tuple[float, np.ndarray, np.ndarray]:
    """The true round->world transform for a gauge created by _world_to_gauge."""
    scale, rotation, translation = gauge
    return scale, rotation, translation  # x_world = s * R @ x_gauge + t


def _make_rounds(rng: np.random.Generator, ids_per_round: list[list[int]], gauges) -> list[RoundPoses]:
    n_world = max(max(ids) for ids in ids_per_round) + 1
    world = rng.normal(size=(n_world, 3)) * 5.0
    world_rot = np.stack([_random_rotation(rng) for _ in range(n_world)])
    names = [f"kf_{i:06d}.jpg" for i in range(n_world)]
    rounds = []
    for gauge, ids in zip(gauges, ids_per_round):
        rounds.append(
            RoundPoses(
                names=[names[i] for i in ids],
                centers=_world_to_gauge(world[ids], gauge),
                rotations=np.einsum("ij,njk->nik", gauge[1].T, world_rot[ids]),
            )
        )
    return rounds


class TestSim3Chart:
    def test_rotation_log_exp_roundtrip(self) -> None:
        rng = np.random.default_rng(4)
        rotation = _random_rotation(rng)
        np.testing.assert_allclose(rotation_exp(rotation_log(rotation)), rotation, atol=1e-9)
        np.testing.assert_allclose(rotation_exp(np.zeros(3)), np.eye(3))

    def test_sim3_params_roundtrip(self) -> None:
        rng = np.random.default_rng(6)
        transform = (1.9, _random_rotation(rng), rng.normal(size=3))
        scale, rotation, translation = sim3_from_params(sim3_to_params(transform))
        np.testing.assert_allclose(scale, transform[0], rtol=1e-12)
        np.testing.assert_allclose(rotation, transform[1], atol=1e-9)
        np.testing.assert_allclose(translation, transform[2], atol=1e-12)


class TestBuildRoundEdges:
    def test_edges_connect_every_overlapping_pair(self) -> None:
        rng = np.random.default_rng(17)
        gauges = [(float(rng.uniform(0.5, 2.0)), _random_rotation(rng), rng.normal(size=3)) for _ in range(3)]
        # strided rounds: early keyframes appear in every round (like the live session)
        rounds = _make_rounds(rng, [[0, 1, 2], [0, 2, 3, 4], [0, 1, 4, 5]], gauges)
        edges = build_round_edges(rounds)
        assert {(e.src, e.dst) for e in edges} == {(0, 1), (0, 2), (1, 2)}
        # edge transform maps src gauge into dst gauge (shared keyframes 0 and 2)
        edge01 = next(e for e in edges if (e.src, e.dst) == (0, 1))
        mapped = apply_to_points(edge01.transform, rounds[0].centers[[0, 2]])
        np.testing.assert_allclose(mapped, rounds[1].centers[[0, 1]], atol=1e-8)
        assert edge01.weight == 2.0

    def test_disjoint_rounds_produce_no_edge(self) -> None:
        rng = np.random.default_rng(18)
        gauges = [(1.0, np.eye(3), np.zeros(3))] * 2
        rounds = _make_rounds(rng, [[0, 1, 2], [3, 4, 5]], gauges)
        assert build_round_edges(rounds) == []


class TestOptimizeSessionTransforms:
    def test_noisy_chain_is_pulled_back_by_loop_edges(self) -> None:
        """Compounding chain noise must shrink once direct (loop) edges constrain it."""
        rng = np.random.default_rng(23)
        n_rounds = 5
        gauges = [(float(rng.uniform(0.7, 1.5)), _random_rotation(rng), rng.normal(size=3)) for _ in range(n_rounds)]
        # every round keeps keyframe 0/1 (stride) and adds its own — dense graph
        ids = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 6, 7]]
        rounds = _make_rounds(rng, ids, gauges)
        edges = build_round_edges(rounds)
        assert len(edges) == n_rounds * (n_rounds - 1) // 2

        # ground truth: session gauge == round 0 gauge
        world_from = [_gauge_to_session(g) for g in gauges]
        session_from_world = (
            1.0 / world_from[0][0],
            world_from[0][1].T,
            -(world_from[0][1].T @ world_from[0][2]) / world_from[0][0],
        )
        truth = [compose(session_from_world, t) for t in world_from]

        # corrupt the initial guesses (as a drifted chain would)
        def jitter(transform, magnitude):
            scale, rotation, translation = transform
            return (
                scale * float(np.exp(rng.normal(0.0, magnitude))),
                rotation_exp(rng.normal(0.0, magnitude, size=3)) @ rotation,
                translation + rng.normal(0.0, magnitude, size=3),
            )

        initial = [truth[0]] + [jitter(t, 0.05 * (k + 1)) for k, t in enumerate(truth[1:])]
        refined = optimize_session_transforms(initial, edges)

        def error(transforms) -> float:
            total = 0.0
            for estimate, true in zip(transforms[1:], truth[1:]):
                total += float(np.linalg.norm(np.asarray(estimate[2]) - np.asarray(true[2])))
                total += float(np.linalg.norm(estimate[1] - true[1]))
                total += abs(np.log(estimate[0] / true[0]))
            return total

        assert error(refined) < 0.01 * error(initial)
        np.testing.assert_allclose(refined[0][1], truth[0][1])  # anchor untouched

    def test_returns_initial_when_nothing_to_solve(self) -> None:
        identity = (1.0, np.eye(3), np.zeros(3))
        assert optimize_session_transforms([identity], []) == [identity]
        edge = Sim3Edge(0, 1, identity, 2.0)
        assert optimize_session_transforms([identity, identity], [edge])[1][0] == 1.0


class TestSplatRebuilderPoseGraph:
    def _write_round(self, round_dir: Path, poses: RoundPoses) -> None:
        from tests.test_gauge_alignment import _write_images_txt

        _write_images_txt(
            round_dir / "sparse_input" / "sparse" / "0" / "images.txt", poses.names, poses.centers, poses.rotations
        )

    def test_third_round_triggers_refinement_and_marks_json(self, tmp_path: Path) -> None:
        from gs_sim2real.robotics.live_mapping import LiveMapperConfig, SplatRebuilder

        rng = np.random.default_rng(41)
        gauges = [(1.0, np.eye(3), np.zeros(3))] + [
            (float(rng.uniform(0.7, 1.5)), _random_rotation(rng), rng.normal(size=3)) for _ in range(2)
        ]
        rounds = _make_rounds(rng, [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5]], gauges)

        builder = SplatRebuilder(LiveMapperConfig(workdir=tmp_path))
        dirs = []
        for index, poses in enumerate(rounds, start=1):
            round_dir = tmp_path / "rounds" / f"round_{index:03d}"
            self._write_round(round_dir, poses)
            dirs.append(round_dir)
            transform, rebased = builder._align_round_gauge(round_dir)
            assert not rebased

        # all three rounds were rewritten by the pose graph
        import json

        for round_dir in dirs:
            data = json.loads((round_dir / "gauge_transform.json").read_text())
            assert data["optimized"] is True

        # round 3 centers, through its refined transform, land in round 1's gauge (== world here)
        data = json.loads((dirs[2] / "gauge_transform.json").read_text())
        refined = (
            float(data["scale"]),
            np.asarray(data["rotation"]),
            np.asarray(data["translation"]),
        )
        aligned = apply_to_points(refined, rounds[2].centers)
        expected = apply_to_points(_gauge_to_session(gauges[2]), rounds[2].centers)
        np.testing.assert_allclose(aligned, expected, atol=1e-6)

    def test_refinement_can_be_disabled(self, tmp_path: Path) -> None:
        from gs_sim2real.robotics.live_mapping import LiveMapperConfig, SplatRebuilder

        rng = np.random.default_rng(42)
        gauges = [(1.0, np.eye(3), np.zeros(3))] * 3
        rounds = _make_rounds(rng, [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5]], gauges)
        builder = SplatRebuilder(LiveMapperConfig(workdir=tmp_path, pose_graph_refinement=False))
        import json

        for index, poses in enumerate(rounds, start=1):
            round_dir = tmp_path / "rounds" / f"round_{index:03d}"
            self._write_round(round_dir, poses)
            builder._align_round_gauge(round_dir)
            data = json.loads((round_dir / "gauge_transform.json").read_text())
            assert data["optimized"] is False
