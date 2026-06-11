"""Tests for collaborative live-map merging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics import live_merge


def _write_round(session: Path, index: int) -> None:
    (session / "rounds" / f"round_{index:03d}").mkdir(parents=True, exist_ok=True)


def test_read_last_round_state_json_and_fallback(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "live").mkdir(parents=True)
    _write_round(session, 1)
    _write_round(session, 3)

    (session / "live" / "state.json").write_text(
        json.dumps({"lastSuccessfulRound": {"round": 2}}),
        encoding="utf-8",
    )
    assert live_merge.read_last_round(session) == 2

    (session / "live" / "state.json").write_text("{not json", encoding="utf-8")
    assert live_merge.read_last_round(session) == 3

    empty = tmp_path / "empty"
    assert live_merge.read_last_round(empty) is None


def test_merge_once_publishes_outputs_and_freezes_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    normalize_calls: list[np.ndarray] = []
    sentinel = (np.array([1.0, 2.0, 3.0], dtype=np.float32), 4.0)

    def fake_merge_sessions(session_a, session_b, output_ply, **kwargs):
        assert Path(session_a) == tmp_path / "a"
        assert Path(session_b) == tmp_path / "b"
        assert kwargs["round_a"] == 4
        assert kwargs["round_b"] == 5
        assert kwargs["align"] == "shared"
        assert kwargs["dedup_radius_camera_heights"] == 0.25
        assert kwargs["localize_config"] is None
        calls.append(Path(output_ply))
        assert Path(output_ply).name == "merged.ply.tmp"
        Path(output_ply).write_bytes(b"fake ply")
        return {
            "output": str(output_ply),
            "gaussians_a": 7,
            "gaussians_b": 9,
            "deduplicated": 2,
            "merged": 14,
            "alignment": {"mode": "shared", "matched_keyframes": 3, "scale": 1.0},
            "dedup_radius": 0.5,
        }

    def fake_ply_to_splat(ply_path, output_path, **kwargs):
        assert Path(ply_path).name == "merged.ply"
        assert Path(output_path).name == "latest.splat.tmp"
        assert kwargs["normalize_params"] is sentinel
        Path(output_path).write_bytes(b"splat")
        return str(output_path)

    def fake_load_ply(path):
        assert Path(path).name == "merged.ply"
        return SimpleNamespace(positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64))

    def fake_compute_splat_normalization(positions, extent):
        assert extent == 2.0
        normalize_calls.append(np.asarray(positions))
        return sentinel

    monkeypatch.setattr(live_merge, "merge_sessions", fake_merge_sessions)
    monkeypatch.setattr(live_merge, "ply_to_splat", fake_ply_to_splat)
    monkeypatch.setattr(live_merge, "load_ply", fake_load_ply)
    monkeypatch.setattr(live_merge, "compute_splat_normalization", fake_compute_splat_normalization)

    config = live_merge.LiveMergeConfig(align="shared", dedup_radius_camera_heights=0.25)
    result = live_merge.merge_once(
        tmp_path / "a",
        tmp_path / "b",
        tmp_path / "out",
        config=config,
        round_a=4,
        round_b=5,
    )

    assert (tmp_path / "out" / "live" / "merged.ply").read_bytes() == b"fake ply"
    assert (tmp_path / "out" / "live" / "latest.splat").read_bytes() == b"splat"
    assert not (tmp_path / "out" / "live" / "merged.ply.tmp").exists()
    assert not (tmp_path / "out" / "live" / "latest.splat.tmp").exists()
    assert result["merged"] == 14
    assert result["round_a"] == 4
    assert result["round_b"] == 5
    assert result["normalize_params"] is sentinel

    result2 = live_merge.merge_once(
        tmp_path / "a",
        tmp_path / "b",
        tmp_path / "out",
        config=config,
        round_a=4,
        round_b=5,
        normalize_params=sentinel,
    )
    assert result2["normalize_params"] is sentinel
    assert len(normalize_calls) == 1
    assert len(calls) == 2


def test_watch_and_merge_records_two_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    for session in (session_a, session_b):
        (session / "live").mkdir(parents=True)
        (session / "live" / "state.json").write_text(
            json.dumps({"lastSuccessfulRound": {"round": 1}}),
            encoding="utf-8",
        )

    pairs: list[tuple[int | None, int | None]] = []
    logs: list[str] = []

    def fake_merge_once(session_a_arg, session_b_arg, output_dir_arg, **kwargs):
        pairs.append((kwargs["round_a"], kwargs["round_b"]))
        return {
            "round_a": kwargs["round_a"],
            "round_b": kwargs["round_b"],
            "gaussians_a": 10,
            "gaussians_b": 20,
            "deduplicated": 3,
            "merged": 27 + len(pairs),
            "alignment": {"mode": "shared", "matched_keyframes": 2, "scale": 1.0},
            "normalize_params": kwargs["normalize_params"] or (np.zeros(3, dtype=np.float32), 1.0),
            "splat": str(Path(output_dir_arg) / "live" / "latest.splat"),
            "preview": None,
        }

    def sleep_fn(_seconds):
        (session_b / "live" / "state.json").write_text(
            json.dumps({"lastSuccessfulRound": {"round": 2}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(live_merge, "merge_once", fake_merge_once)

    result = live_merge.watch_and_merge(
        session_a,
        session_b,
        tmp_path / "out",
        config=live_merge.LiveMergeConfig(interval_s=0.0),
        max_merges=2,
        sleep_fn=sleep_fn,
        log_fn=logs.append,
    )

    assert pairs == [(1, 1), (1, 2)]
    assert len(result) == 2
    state = json.loads((tmp_path / "out" / "live" / "state.json").read_text(encoding="utf-8"))
    assert state["mode"] == "merge-live"
    assert len(state["merges"]) == 2
    assert state["lastMerge"]["roundA"] == 1
    assert state["lastMerge"]["roundB"] == 2
    assert len(logs) == 2
    assert logs[0].startswith("merge 1:")


def test_watch_and_merge_once_requires_both_sessions_to_have_rounds(tmp_path: Path) -> None:
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    _write_round(session_a, 1)

    with pytest.raises(RuntimeError, match=str(session_b)):
        live_merge.watch_and_merge(
            session_a,
            session_b,
            tmp_path / "out",
            config=live_merge.LiveMergeConfig(interval_s=0.0),
            once=True,
            sleep_fn=lambda _seconds: None,
        )


def test_watch_and_merge_skips_failed_pair_until_round_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    for session in (session_a, session_b):
        (session / "live").mkdir(parents=True)
        (session / "live" / "state.json").write_text(
            json.dumps({"lastSuccessfulRound": {"round": 1}}),
            encoding="utf-8",
        )

    attempts: list[tuple[int | None, int | None]] = []
    logs: list[str] = []
    sleep_calls = 0

    def fake_merge_once(_session_a, _session_b, output_dir, **kwargs):
        pair = (kwargs["round_a"], kwargs["round_b"])
        attempts.append(pair)
        if pair == (1, 1):
            raise ValueError("alignment failed")
        return {
            "round_a": pair[0],
            "round_b": pair[1],
            "gaussians_a": 1,
            "gaussians_b": 2,
            "deduplicated": 0,
            "merged": 3,
            "alignment": {"mode": "localize", "matched_keyframes": 0, "scale": 1.1},
            "normalize_params": kwargs["normalize_params"] or (np.zeros(3, dtype=np.float32), 1.0),
            "splat": str(Path(output_dir) / "live" / "latest.splat"),
            "preview": None,
        }

    def sleep_fn(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            (session_b / "live" / "state.json").write_text(
                json.dumps({"lastSuccessfulRound": {"round": 2}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(live_merge, "merge_once", fake_merge_once)

    result = live_merge.watch_and_merge(
        session_a,
        session_b,
        tmp_path / "out",
        config=live_merge.LiveMergeConfig(interval_s=0.0),
        max_merges=1,
        sleep_fn=sleep_fn,
        log_fn=logs.append,
    )

    assert attempts == [(1, 1), (1, 2)]
    assert len(result) == 1
    assert any("failed for A round 1 + B round 1" in line for line in logs)
    assert any(line.startswith("merge 1:") for line in logs)
