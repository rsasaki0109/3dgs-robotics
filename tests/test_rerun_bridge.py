"""Tests for rerun replay assembly and logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics import rerun_bridge


def _touch_round(session_dir: Path, index: int) -> Path:
    round_dir = session_dir / "rounds" / f"round_{index:03d}"
    (round_dir / "train").mkdir(parents=True)
    (round_dir / "images").mkdir(parents=True)
    (round_dir / "train" / "point_cloud.ply").touch()
    (round_dir / "scene.splat").touch()
    (round_dir / "images" / "a.jpg").touch()
    (round_dir / "images" / "z.jpg").touch()
    return round_dir


def test_session_timeline_assembles_sorted_rounds_and_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    _touch_round(session_dir, 2)
    _touch_round(session_dir, 1)
    (session_dir / "live").mkdir(parents=True)
    (session_dir / "live" / "loop_candidates.json").write_text(
        json.dumps(
            {
                "loopCandidates": [
                    {"queryIndex": 0, "matchIndex": 1},
                    {"queryIndex": 0, "matchIndex": 99},
                ]
            }
        ),
        encoding="utf-8",
    )

    positions = np.column_stack(
        [
            np.arange(50, dtype=np.float64),
            np.zeros(50, dtype=np.float64),
            np.ones(50, dtype=np.float64),
        ]
    )
    centers_by_round = {
        1: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        2: np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64),
    }

    def fake_session(_session_dir: Path, *, round_index: int | None = None) -> SimpleNamespace:
        return SimpleNamespace(round=SimpleNamespace(round_index=round_index))

    def fake_records(session: SimpleNamespace) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(center=center, qvec=np.array([1.0, 0.0, 0.0, 0.0]))
            for center in centers_by_round[session.round.round_index]
        ]

    monkeypatch.setattr(rerun_bridge, "resolve_live_map_session", fake_session)
    monkeypatch.setattr(rerun_bridge, "load_mapped_records", fake_records)
    monkeypatch.setattr(
        rerun_bridge, "load_ply", lambda *args, **kwargs: SimpleNamespace(positions=positions, colors=None)
    )
    monkeypatch.setattr(rerun_bridge, "read_gauge_transform", lambda *args, **kwargs: None)

    entries, extras = rerun_bridge.session_timeline(session_dir, max_points_per_round=10)

    assert [entry.round_index for entry in entries] == [1, 2]
    assert [len(entry.positions) for entry in entries] == [10, 10]
    assert entries[0].colors.dtype == np.uint8
    assert entries[0].colors.shape == (10, 3)
    assert np.all(entries[0].colors == 128)
    assert entries[0].image_path == session_dir / "rounds" / "round_001" / "images" / "z.jpg"
    assert extras["loop_edges"] == [
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    ]
    assert extras["session"] == str(session_dir)


def test_session_timeline_uses_colors_and_nav_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    round_dir = _touch_round(session_dir, 1)
    nav_json = tmp_path / "nav_result.json"
    nav_json.write_text(json.dumps({"path_vertices": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    colors = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]], dtype=np.float64)
    centers = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float64)
    records = [SimpleNamespace(center=center, qvec=np.array([1.0, 0.0, 0.0, 0.0])) for center in centers]
    transform = (2.0, np.eye(3), np.array([1.0, 0.0, 0.0]))

    monkeypatch.setattr(
        rerun_bridge,
        "resolve_live_map_session",
        lambda *args, **kwargs: SimpleNamespace(round=SimpleNamespace(round_dir=round_dir)),
    )
    monkeypatch.setattr(rerun_bridge, "load_mapped_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(
        rerun_bridge, "load_ply", lambda *args, **kwargs: SimpleNamespace(positions=positions, colors=colors)
    )
    monkeypatch.setattr(rerun_bridge, "read_gauge_transform", lambda *args, **kwargs: (transform, False))
    monkeypatch.setattr(
        rerun_bridge,
        "estimate_ground_frame",
        lambda *args, **kwargs: (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            0.0,
            2.0,
        ),
    )

    entries, extras = rerun_bridge.session_timeline(session_dir, nav_json=nav_json)

    assert entries[0].colors.tolist() == [[0, 127, 255], [255, 63, 0]]
    np.testing.assert_allclose(entries[0].positions, positions * 2.0 + np.array([1.0, 0.0, 0.0]))
    assert extras["nav_points"] is not None
    assert len(extras["nav_points"]) == 2


def test_session_timeline_no_rounds_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no successful rounds"):
        rerun_bridge.session_timeline(tmp_path / "empty")


def test_log_session_records_entities_and_stats(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    logged: list[tuple[str, object]] = []
    times: list[int] = []

    rr = SimpleNamespace(
        init=lambda application_id, spawn=False: calls.update({"init": (application_id, spawn)}),
        save=lambda path: calls.update({"save": path}),
        set_time_sequence=lambda name, index: times.append(index),
        log=lambda path, obj: logged.append((path, obj)),
        Points3D=lambda positions, colors=None: ("Points3D", positions, colors),
        LineStrips3D=lambda strips, colors=None: ("LineStrips3D", strips, colors),
        Image=lambda image: ("Image", image),
    )
    entries = [
        rerun_bridge.RoundEntry(
            round_index=1,
            positions=np.zeros((3, 3), dtype=np.float64),
            colors=np.zeros((3, 3), dtype=np.uint8),
            centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            image_path=None,
        ),
        rerun_bridge.RoundEntry(
            round_index=2,
            positions=np.zeros((4, 3), dtype=np.float64),
            colors=np.zeros((4, 3), dtype=np.uint8),
            centers=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            image_path=None,
        ),
    ]
    extras = {
        "loop_edges": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        "nav_points": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }

    stats = rerun_bridge.log_session(entries, extras, rr=rr, save=tmp_path / "session.rrd", spawn=True)

    assert calls["init"] == ("3dgs-robotics", True)
    assert calls["save"] == str(tmp_path / "session.rrd")
    assert times == [1, 2, 2]
    assert [path for path, _obj in logged] == [
        "world/map",
        "world/map",
        "world/trajectory",
        "world/loops",
        "world/nav/planned_path",
    ]
    assert stats == {
        "rounds": 2,
        "points_logged": 7,
        "loop_edges": 1,
        "nav": True,
        "rrd": str(tmp_path / "session.rrd"),
    }


def test_log_session_time_compat_uses_set_time() -> None:
    calls: list[tuple[str, int]] = []
    logged: list[tuple[str, object]] = []
    rr = SimpleNamespace(
        init=lambda application_id, spawn=False: None,
        set_time=lambda name, *, sequence: calls.append((name, sequence)),
        log=lambda path, obj: logged.append((path, obj)),
        Points3D=lambda positions, colors=None: ("Points3D", positions, colors),
        LineStrips3D=lambda strips, colors=None: ("LineStrips3D", strips, colors),
        Image=lambda image: ("Image", image),
    )
    entries = [
        rerun_bridge.RoundEntry(
            round_index=3,
            positions=np.zeros((1, 3), dtype=np.float64),
            colors=np.zeros((1, 3), dtype=np.uint8),
            centers=np.zeros((0, 3), dtype=np.float64),
            image_path=None,
        )
    ]

    rerun_bridge.log_session(entries, {"loop_edges": [], "nav_points": None}, rr=rr)

    assert calls == [("round", 3), ("round", 3)]


def test_import_rerun_message_contains_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "rerun":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match=r"\[rerun\]"):
        rerun_bridge._import_rerun()
