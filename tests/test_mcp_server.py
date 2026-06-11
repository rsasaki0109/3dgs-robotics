"""Tests for the Talk to Your Map MCP server wiring."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gs_sim2real.robotics import mcp_server


def _write_session(root: Path, name: str = "session_a") -> Path:
    session = root / name
    keyframes = session / "keyframes"
    keyframes.mkdir(parents=True)
    for index in range(3):
        (keyframes / f"kf_{index:06d}.jpg").touch()

    sparse = session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0"
    sparse.mkdir(parents=True)
    train = session / "rounds" / "round_001" / "train"
    train.mkdir(parents=True)
    (train / "point_cloud.ply").touch()
    (sparse / "images.txt").touch()

    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"lastSuccessfulRound": {"round": 1}}), encoding="utf-8")
    (live / "latest.splat").touch()
    return session


def test_list_map_sessions_finds_sessions_and_counts(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "live_mapping"
    session = _write_session(root)

    result = mcp_server.list_map_sessions(str(root))

    assert result == {
        "sessions": [
            {
                "name": "session_a",
                "path": str(session),
                "keyframe_count": 3,
                "round_count": 1,
                "last_successful_round": 1,
                "has_latest_splat": True,
            }
        ]
    }


def test_list_map_sessions_missing_root_returns_hint(tmp_path: Path) -> None:
    result = mcp_server.list_map_sessions(str(tmp_path / "missing"))

    assert result["sessions"] == []
    assert "run live mapping first" in result["hint"]


def test_list_map_sessions_accepts_session_dir(tmp_path: Path) -> None:
    session = _write_session(tmp_path)

    result = mcp_server.list_map_sessions(str(session))

    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["path"] == str(session)


def test_map_info_fixture(tmp_path: Path) -> None:
    session = _write_session(tmp_path)

    result = mcp_server.map_info(str(session))

    assert result["name"] == "session_a"
    assert result["keyframe_count"] == 3
    assert result["rounds"] == [1]
    assert result["resolved_round"] == 1
    assert result["artifacts"] == {
        "point_cloud": str(session / "rounds" / "round_001" / "train" / "point_cloud.ply"),
        "images_txt": str(session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0" / "images.txt"),
    }


def test_map_info_non_session_raises(tmp_path: Path) -> None:
    not_session = tmp_path / "not_session"
    not_session.mkdir()

    with pytest.raises(FileNotFoundError, match="Use list_map_sessions"):
        mcp_server.map_info(str(not_session))


def test_query_map_builds_argv_reads_json_and_caps_hits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []

    def fake_run_cli(args):
        calls.append(list(args))
        out_path = Path(args[args.index("--output") + 1])
        hits = [{"goal_xy": [float(index), float(index + 1)], "rank": index} for index in range(12)]
        out_path.write_text(json.dumps({"prompt": "chair", "hits": hits}), encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.query_map(str(session), "chair", threshold=0.7, round_index=1, device="cpu")

    assert calls == [
        [
            "query-map",
            "chair",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "query_chair_20260102-030405.json"),
            "--threshold",
            "0.7",
            "--device",
            "cpu",
            "--round",
            "1",
        ]
    ]
    assert result["prompt"] == "chair"
    assert len(result["hits"]) == 10
    assert result["hit_count"] == 12
    assert result["preview_png"].endswith("query_chair_20260102-030405.png")
    assert result["navigate_suggestion"]["arguments"]["goal_xy"] == [0.0, 1.0]


def test_navigate_builds_argv_reads_summary_and_drops_path_vertices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []

    def fake_run_cli(args):
        calls.append(list(args))
        out_path = Path(args[args.index("--output") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "reached": True,
                    "steps": 42,
                    "localization_fixes": 2,
                    "cross_track_median": 0.1,
                    "cross_track_max": 0.3,
                    "goal": [1.0, 2.0],
                    "path_vertices": [[0.0, 0.0], [1.0, 2.0]],
                    "note": "distances in the map's reconstruction gauge unless mapped with metric poses",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.navigate(str(session), goal_xy=[1.0, 2.0], gif=True, round_index=1, device="cpu", max_steps=99)

    assert calls == [
        [
            "navigate",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "nav_20260102-030405.json"),
            "--max-steps",
            "99",
            "--device",
            "cpu",
            "--round",
            "1",
            "--goal",
            "1.0,2.0",
            "--gif",
            str(session / "mcp" / "nav_20260102-030405.gif"),
        ]
    ]
    assert result["reached"] is True
    assert result["steps"] == 42
    assert result["trace_png"].endswith("nav_20260102-030405.png")
    assert result["gif"].endswith("nav_20260102-030405.gif")
    assert "path_vertices" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"to": "chair", "goal_xy": [1.0, 2.0]},
    ],
)
def test_navigate_goal_validation(tmp_path: Path, kwargs: dict[str, object]) -> None:
    session = _write_session(tmp_path)

    with pytest.raises(ValueError, match="exactly one"):
        mcp_server.navigate(str(session), **kwargs)


def test_detect_changes_builds_argv_and_summarizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    map_a = _write_session(tmp_path, "a")
    map_b = _write_session(tmp_path, "b")
    calls: list[list[str]] = []

    def fake_run_cli(args):
        calls.append(list(args))
        out_path = Path(args[args.index("--output") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "alignment": {"mode": "shared"},
                    "appeared": [{"id": index} for index in range(7)],
                    "disappeared": [{"id": index} for index in range(6)],
                    "clusters": [{"id": index} for index in range(13)],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.detect_changes(str(map_a), str(map_b), round_a=1, round_b=2, device="cpu")

    assert calls == [
        [
            "detect-changes",
            "--map-a",
            str(map_a),
            "--output",
            str(map_a / "mcp" / "changes_20260102-030405.json"),
            "--device",
            "cpu",
            "--map-b",
            str(map_b),
            "--round-a",
            "1",
            "--round-b",
            "2",
        ]
    ]
    assert result["alignment"] == {"mode": "shared"}
    assert result["appeared_count"] == 7
    assert result["disappeared_count"] == 6
    assert len(result["clusters"]) == 10
    assert result["cluster_count"] == 13


def test_export_overlay_builds_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []

    def fake_run_cli(args):
        calls.append(list(args))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.export_overlay(
        str(session),
        nav_json=str(tmp_path / "nav.json"),
        query_json=str(tmp_path / "query.json"),
        round_index=1,
    )

    assert calls == [
        [
            "export-overlay",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "overlay_20260102-030405.json"),
            "--round",
            "1",
            "--nav",
            str(tmp_path / "nav.json"),
            "--query",
            str(tmp_path / "query.json"),
        ]
    ]
    assert result["overlay_json"] == str(session / "mcp" / "overlay_20260102-030405.json")
    assert "splat.html?" in result["usage_hint"]
    assert "overlay_20260102-030405.json" in result["usage_hint"]


def test_run_cli_failure_includes_stderr_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = "\n".join(f"line {index}" for index in range(30))

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        mcp_server._run_cli(["query-map"])

    message = str(exc_info.value)
    assert "line 10" in message
    assert "line 29" in message
    assert "line 9" not in message


def test_splat_clean_builds_argv_and_returns_stdout_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []
    stdout = "\n".join(f"out {index}" for index in range(12))

    def fake_run_cli(args):
        calls.append(list(args))
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.splat_clean(str(session), "traffic sign", threshold=0.6, round_index=1, device="cpu")

    assert calls == [
        [
            "splat-clean",
            "traffic sign",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "cleaned_traffic-sign_20260102-030405.ply"),
            "--threshold",
            "0.6",
            "--device",
            "cpu",
            "--round",
            "1",
        ]
    ]
    assert result["output_ply"].endswith("cleaned_traffic-sign_20260102-030405.ply")
    assert result["preview_png"].endswith("cleaned_traffic-sign_20260102-030405.png")
    assert result["stdout_tail"] == "\n".join(f"out {index}" for index in range(2, 12))


def test_explore_builds_argv_and_drops_coverage_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []

    def fake_run_cli(args):
        calls.append(list(args))
        out_path = Path(args[args.index("--output") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "coverage_fraction": 0.91,
                    "coverage_target": 0.9,
                    "reachable_free_cells": 100,
                    "observed_free_cells": 91,
                    "goals": [{"goal_xy": [1.0, 2.0], "reached": True, "steps": 12}],
                    "goals_chosen": 1,
                    "total_steps": 12,
                    "distance": 3.4,
                    "localization_fixes": 0,
                    "stop_reason": "coverage-target",
                    "camera_height": 1.0,
                    "coverage_history": [[0, 0.2], [12, 0.91]],
                    "note": "distances in the map's reconstruction gauge unless mapped with metric poses",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.explore(
        str(session),
        sensor_range=5.0,
        coverage_target=0.9,
        max_goals=7,
        gif=True,
        round_index=1,
        device="cpu",
        localize_every=10,
    )

    assert calls == [
        [
            "explore",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "explore_20260102-030405.json"),
            "--device",
            "cpu",
            "--localize-every",
            "10",
            "--round",
            "1",
            "--sensor-range",
            "5.0",
            "--coverage-target",
            "0.9",
            "--max-goals",
            "7",
            "--gif",
            str(session / "mcp" / "explore_20260102-030405.gif"),
        ]
    ]
    assert result["coverage_fraction"] == 0.91
    assert result["goals_chosen"] == 1
    assert "coverage_history" not in result
    assert result["output_json"].endswith("explore_20260102-030405.json")
    assert result["trace_png"].endswith("explore_20260102-030405.png")
    assert result["gif"].endswith("explore_20260102-030405.gif")


def test_merge_maps_builds_argv_and_returns_stdout_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_a = _write_session(tmp_path, "session_a")
    session_b = _write_session(tmp_path, "session_b")
    calls: list[list[str]] = []
    stdout = "\n".join(f"merge out {index}" for index in range(12))

    def fake_run_cli(args):
        calls.append(list(args))
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.merge_maps(
        str(session_a),
        str(session_b),
        dedup_radius=0.25,
        round_a=1,
        round_b=2,
        device="cpu",
    )

    assert calls == [
        [
            "merge-maps",
            "--map-a",
            str(session_a),
            "--map-b",
            str(session_b),
            "--output",
            str(session_a / "mcp" / "merged_20260102-030405.ply"),
            "--device",
            "cpu",
            "--dedup-radius",
            "0.25",
            "--round-a",
            "1",
            "--round-b",
            "2",
        ]
    ]
    assert result["output_ply"].endswith("merged_20260102-030405.ply")
    assert result["stdout_tail"] == "\n".join(f"merge out {index}" for index in range(2, 12))


def test_patrol_builds_argv_and_caps_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []

    def fake_run_cli(args):
        calls.append(list(args))
        out_path = Path(args[args.index("--output") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "stops": [
                        {
                            "label": f"stop {index}",
                            "source": "xy",
                            "goal_xy": [float(index), 2.0],
                            "planned": True,
                            "reached": True,
                            "steps": 3,
                            "capture": None,
                        }
                        for index in range(25)
                    ],
                    "totals": {
                        "total_steps": 75,
                        "distance": 12.5,
                        "reached_count": 25,
                        "waypoint_count": 25,
                        "localization_fixes": 0,
                    },
                    "start_keyframe": 0,
                    "camera_height": 1.0,
                    "note": "distances in the map's reconstruction gauge unless mapped with metric poses",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.patrol(
        str(session),
        to="car;traffic sign",
        num_waypoints=6,
        render=True,
        gif=True,
        round_index=1,
        device="cpu",
        localize_every=10,
    )

    assert calls == [
        [
            "patrol",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "patrol_20260102-030405.json"),
            "--device",
            "cpu",
            "--localize-every",
            "10",
            "--round",
            "1",
            "--to",
            "car;traffic sign",
            "--num-waypoints",
            "6",
            "--render",
            "--gif",
            str(session / "mcp" / "patrol_20260102-030405.gif"),
        ]
    ]
    assert result["totals"]["reached_count"] == 25
    assert result["stops_total"] == 25
    assert len(result["stops"]) == 20
    assert result["output_json"].endswith("patrol_20260102-030405.json")
    assert result["trace_png"].endswith("patrol_20260102-030405.png")
    assert result["gif"].endswith("patrol_20260102-030405.gif")


def test_splat_grab_builds_argv_and_reads_sidecar_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    calls: list[list[str]] = []
    stdout = "\n".join(f"out {index}" for index in range(12))

    def fake_run_cli(args):
        calls.append(list(args))
        sidecar = session / "mcp" / "grabbed_traffic-sign_20260102-030405.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text('{"prompt": "traffic sign", "gaussians": 5}\n', encoding="utf-8")
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.splat_grab(str(session), "traffic sign", threshold=0.6, round_index=1, device="cpu")

    assert calls == [
        [
            "splat-grab",
            "traffic sign",
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "grabbed_traffic-sign_20260102-030405.ply"),
            "--threshold",
            "0.6",
            "--device",
            "cpu",
            "--round",
            "1",
        ]
    ]
    assert result["output_ply"].endswith("grabbed_traffic-sign_20260102-030405.ply")
    assert result["sidecar"].endswith("grabbed_traffic-sign_20260102-030405.json")
    assert result["preview_png"].endswith("grabbed_traffic-sign_20260102-030405.png")
    assert result["summary"]["prompt"] == "traffic sign"
    assert result["summary"]["gaussians"] == 5
    assert result["stdout_tail"] == "\n".join(f"out {index}" for index in range(2, 12))


def test_splat_paste_builds_argv_and_returns_stdout_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _write_session(tmp_path)
    object_ply = tmp_path / "object.ply"
    object_ply.write_text("ply\n", encoding="utf-8")
    calls: list[list[str]] = []
    stdout = "\n".join(f"paste {index}" for index in range(11))

    def fake_run_cli(args):
        calls.append(list(args))
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102-030405")

    result = mcp_server.splat_paste(
        str(session),
        str(object_ply),
        [1.2, 0.3],
        yaw_deg=15.0,
        scale=2.0,
        round_index=3,
    )

    assert calls == [
        [
            "splat-paste",
            str(object_ply),
            "--map",
            str(session),
            "--output",
            str(session / "mcp" / "pasted_20260102-030405.ply"),
            "--at",
            "1.2,0.3",
            "--yaw",
            "15.0",
            "--scale",
            "2.0",
            "--round",
            "3",
        ]
    ]
    assert result["output_ply"].endswith("pasted_20260102-030405.ply")
    assert result["preview_png"].endswith("pasted_20260102-030405.png")
    assert result["stdout_tail"] == "\n".join(f"paste {index}" for index in range(1, 11))


def test_export_isaac_route_mcp_builds_cli_argv_and_returns_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(mcp_server, "_ensure_session", lambda map_dir: Path(map_dir))
    monkeypatch.setattr(mcp_server, "_mcp_out_dir", lambda session: tmp_path / "mcp")
    monkeypatch.setattr(mcp_server, "_timestamp", lambda: "20260102_030405")

    def fake_run_cli(args: list[str]) -> Any:
        captured["args"] = args
        out_path = tmp_path / "mcp" / "route_20260102_030405.usda"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = (
            f"Wrote {out_path} (2 polyline(s), 3 marker(s))\n"
            "Open it: usdview route.usda (the route references scene.usdz)\n"
            "Note: route distances are reconstruction-gauge camera-height units, not meters.\n"
        )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(mcp_server, "_run_cli", fake_run_cli)

    result = mcp_server.export_isaac_route(
        str(session_dir),
        nav_json="nav_result.json",
        query_json="query.json",
        usdz="scene.usdz",
        round_index=7,
    )

    assert captured["args"] == [
        "export-isaac-route",
        "--map",
        str(session_dir),
        "--output",
        str(tmp_path / "mcp" / "route_20260102_030405.usda"),
        "--round",
        "7",
        "--nav",
        "nav_result.json",
        "--query",
        "query.json",
        "--usdz",
        "scene.usdz",
    ]
    assert result["output_usda"] == str(tmp_path / "mcp" / "route_20260102_030405.usda")
    assert result["usdz_reference"] == "scene.usdz"
    assert (
        result["stdout_tail"].splitlines()[-1]
        == "Note: route distances are reconstruction-gauge camera-height units, not meters."
    )
