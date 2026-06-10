"""Tests for the browser-viewer overlay export (splat-frame JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.robotics.viewer_overlay import build_overlay, splat_frame_mapper
from tests.test_language_query import _write_fake_session

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_nav_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "reached": True,
                "path_vertices": [[0.0, 0.0], [2.0, 0.0]],
                "goal": [2.0, 0.0],
                "camera_height": 1.5,
            }
        ),
        encoding="utf-8",
    )


def _write_query_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt": "object",
                "camera_height": 1.5,
                "hits": [
                    {
                        "centroid": [0.8, -0.5, 2.1],
                        "extent": [0.6, 0.1, 0.3],
                        "gaussians": 30,
                        "mean_score": 0.9,
                        "goal_xy": [2.1, -0.8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class TestSplatFrameMapper:
    def test_identity_gauge_recovers_normalization(self, tmp_path):
        from gs_sim2real.robotics.localize import resolve_live_map_session
        from gs_sim2real.viewer.web_viewer import load_ply

        session_dir = tmp_path / "session"
        _write_fake_session(session_dir)
        session = resolve_live_map_session(session_dir)
        mapper = splat_frame_mapper(session, target_extent=17.0)

        positions = np.asarray(load_ply(session.round.ply_path).positions, dtype=np.float64)
        moved = mapper.points(positions)
        # no gauge transform on disk -> pure normalization: centered, extent 17
        assert moved.mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-5)  # float32 centroid
        assert float(np.max(moved.max(axis=0) - moved.min(axis=0))) == pytest.approx(17.0, abs=1e-5)
        assert mapper.distance(1.0) == pytest.approx(1.0 / mapper.factor)


class TestBuildOverlay:
    def test_trajectory_nav_and_query(self, tmp_path):
        session_dir = tmp_path / "session"
        _write_fake_session(session_dir)
        _write_nav_json(tmp_path / "nav.json")
        _write_query_json(tmp_path / "query.json")

        stats = build_overlay(
            session_dir,
            tmp_path / "overlay.json",
            nav_json=tmp_path / "nav.json",
            query_json=tmp_path / "query.json",
        )
        assert stats["polylines"] == 2
        assert stats["markers"] == 2  # goal + one query hit

        payload = json.loads((tmp_path / "overlay.json").read_text(encoding="utf-8"))
        assert payload["frame"] == "splat"
        labels = [polyline["label"] for polyline in payload["polylines"]]
        assert labels == ["mapped trajectory", "planned path"]
        trajectory = np.asarray(payload["polylines"][0]["points"])
        assert trajectory.shape == (3, 3)  # one vertex per mapped keyframe
        hit = payload["markers"][1]
        assert hit["label"].startswith("object #1")
        assert hit["radius"] > 0
        # the planned path hugs the road: its splat-frame points stay inside
        # the normalized scene extent
        path = np.asarray(payload["polylines"][1]["points"])
        assert np.abs(path).max() < 17.0

    def test_trajectory_only(self, tmp_path):
        session_dir = tmp_path / "session"
        _write_fake_session(session_dir)
        stats = build_overlay(session_dir, tmp_path / "overlay.json")
        assert stats["polylines"] == 1
        assert stats["markers"] == 0


class TestViewerWiring:
    def test_main_js_supports_overlay_param(self):
        main_js = (_REPO_ROOT / "docs" / "splat-viewer" / "main.js").read_text(encoding="utf-8")
        assert 'params.get("overlay")' in main_js
        assert "drawOverlay(viewProj)" in main_js

    def test_cli_export_overlay(self, tmp_path, capsys):
        from gs_sim2real import cli

        session_dir = tmp_path / "session"
        _write_fake_session(session_dir)
        _write_nav_json(tmp_path / "nav.json")
        args = cli.build_parser().parse_args(
            [
                "export-overlay",
                "--map",
                str(session_dir),
                "--nav",
                str(tmp_path / "nav.json"),
                "--output",
                str(tmp_path / "overlay.json"),
            ]
        )
        cli.cmd_export_overlay(args)
        out = capsys.readouterr().out
        assert "2 polyline(s), 1 marker(s)" in out
        assert "overlay=" in out
        assert (tmp_path / "overlay.json").is_file()
