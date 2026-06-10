"""Tests for open-vocabulary map queries (CLIPSeg stubbed out)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics.language_query import (
    QueryParams,
    cluster_hits,
    project_scores,
    query_map,
    write_query_preview,
)

_IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)
_INTRINSICS = (50.0, 50.0, 32.0, 24.0)  # 64x48 pinhole


def _record(name: str, tvec=(0.0, 0.0, 0.0)) -> SimpleNamespace:
    return SimpleNamespace(name=name, qvec=_IDENTITY_Q, tvec=tuple(tvec), camera_id=1)


class TestProjectScores:
    def test_bright_dark_and_behind(self):
        heatmap = np.zeros((48, 64), dtype=np.float32)
        heatmap[:, 40:] = 1.0  # right half of the image is relevant
        # x=0.5,z=2 -> u = 0.5/2*50+32 = 44.5 (bright); x=-0.5 -> u=19.5 (dark)
        points = np.array([[0.5, 0.0, 2.0], [-0.5, 0.0, 2.0], [0.0, 0.0, -2.0]])
        scores, views = project_scores(points, [_record("kf_000000.jpg")], {"kf_000000.jpg": heatmap}, _INTRINSICS)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert views.tolist() == [1, 1, 0]

    def test_scores_average_across_views(self):
        bright = np.ones((48, 64), dtype=np.float32)
        dark = np.zeros((48, 64), dtype=np.float32)
        points = np.array([[0.0, 0.0, 2.0]])
        scores, views = project_scores(
            points,
            [_record("a.jpg"), _record("b.jpg")],
            {"a.jpg": bright, "b.jpg": dark},
            _INTRINSICS,
        )
        assert views[0] == 2
        assert scores[0] == pytest.approx(0.5)


class TestClusterHits:
    def test_clusters_ranked_and_filtered(self):
        rng_free = np.linspace(0.0, 1.0, 30)
        blob = np.stack([0.5 + 0.01 * rng_free, np.full(30, -0.5), np.full(30, 2.0)], axis=1)
        stray = np.array([[5.0, 5.0, 5.0]])
        points = np.vstack([blob, stray])
        scores = np.concatenate([np.full(30, 0.9), [0.95]])
        views = np.ones(len(points), dtype=np.int64)
        basis = np.eye(3)
        hits = cluster_hits(points, scores, views, basis, voxel_size=0.25, params=QueryParams(min_cluster_gaussians=10))
        assert len(hits) == 1  # the stray point is filtered
        assert hits[0].gaussians == 30
        assert np.asarray(hits[0].centroid) == pytest.approx(blob.mean(axis=0), abs=1e-6)

    def test_no_hits_below_threshold(self):
        points = np.zeros((5, 3))
        hits = cluster_hits(
            points,
            np.full(5, 0.1),
            np.ones(5, dtype=np.int64),
            np.eye(3),
            voxel_size=0.25,
            params=QueryParams(),
        )
        assert hits == []


def _write_fake_session(session: Path) -> np.ndarray:
    """Optical-convention scene: ground y=0, cameras at y=-1.5 along +z, an
    'object' blob at x=0.5, y=-0.5, z=2 (image-right of camera 0)."""
    import cv2

    keyframes = session / "keyframes"
    keyframes.mkdir(parents=True)
    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"lastSuccessfulRound": {"round": 1}}), encoding="utf-8")
    sparse = session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n", encoding="utf-8")
    lines = []
    for i in range(3):
        lines.append(f"{i + 1} 1 0 0 0 0 1.5 {-float(i)} 1 kf_{i:06d}.jpg\n\n")
        cv2.imwrite(str(keyframes / f"kf_{i:06d}.jpg"), np.full((48, 64, 3), 127, dtype=np.uint8))
    (sparse / "images.txt").write_text("".join(lines), encoding="utf-8")

    xs = np.linspace(-2.0, 2.0, 11)
    zs = np.linspace(0.0, 4.0, 11)
    gx, gz = np.meshgrid(xs, zs)
    ground = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    blob = np.stack(
        [0.5 + 0.02 * np.arange(30), np.full(30, -0.5), 2.0 + 0.01 * np.arange(30)],
        axis=1,
    )
    points = np.vstack([ground, blob])
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
    return blob


def _object_heatmap_fn(image_bgr: np.ndarray, prompt: str) -> np.ndarray:
    """Stub CLIPSeg: relevance on the right half of every keyframe."""
    heatmap = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    heatmap[:, image_bgr.shape[1] // 2 :] = 1.0
    return heatmap


class TestQueryMap:
    def test_finds_the_object_blob(self, tmp_path):
        session = tmp_path / "session"
        blob = _write_fake_session(session)
        result, points = query_map(
            session,
            "object",
            params=QueryParams(min_cluster_gaussians=10, score_threshold=0.6),
            heatmap_fn=_object_heatmap_fn,
        )
        assert result.hits
        best = result.hits[0]
        # the blob sits image-right (x > 0) for every camera and must dominate
        assert best.centroid[0] > 0.2
        assert best.gaussians >= 10
        # goal feeds the navigation planner: grid coords = centroid @ basis[:2].T
        expected_goal = np.asarray(best.centroid) @ result.basis[:2].T
        assert np.asarray(best.goal_xy) == pytest.approx(expected_goal, abs=1e-9)
        preview = write_query_preview(result, points, tmp_path / "query.png")
        assert preview.is_file()
        del blob

    def test_cli_query_map(self, tmp_path, capsys, monkeypatch):
        from gs_sim2real import cli

        session = tmp_path / "session"
        _write_fake_session(session)
        monkeypatch.setattr(
            "gs_sim2real.robotics.language_query.clipseg_heatmap_fn",
            lambda device="cuda": _object_heatmap_fn,
        )
        args = cli.build_parser().parse_args(
            [
                "query-map",
                "object",
                "--map",
                str(session),
                "--threshold",
                "0.6",
                "--min-cluster-gaussians",
                "10",
                "--output",
                str(tmp_path / "query.json"),
            ]
        )
        cli.cmd_query_map(args)
        out = capsys.readouterr().out
        assert "hit(s)" in out
        assert "navigate --map" in out.replace("3dgs-robotics navigate --map", "navigate --map")
        assert (tmp_path / "query.json").is_file()
        assert (tmp_path / "query.png").is_file()
