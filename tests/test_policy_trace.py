"""Tests for the route policy trace events module (Sprint 3 / PR C)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from gs_sim2real.robotics import (
    CORRELATION_EVENT_WINDOWS_VERSION,
    CorrelationEventWindow,
    load_correlation_event_windows_json,
)
from gs_sim2real.sim import (
    ROUTE_POLICY_TRACE_EVENT_VERSION,
    PolicyTraceEvent,
    RoutePolicyDatasetExport,
    RoutePolicyEpisodeRecord,
    RoutePolicyTransitionRecord,
    convert_policy_trace_events_to_event_windows,
    extract_policy_trace_events_from_dataset,
    load_policy_trace_jsonl,
    policy_trace_event_from_dict,
    write_correlation_event_windows_json,
    write_policy_trace_jsonl,
    write_route_policy_dataset_json,
)
from gs_sim2real.sim.policy_trace import (
    run_dataset_to_trace_cli,
    run_trace_to_event_windows_cli,
)


def _build_transition(
    *,
    episode_id: str,
    scene_id: str,
    episode_index: int,
    step_index: int,
    info: dict[str, object] | None = None,
    next_observation: dict[str, float] | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> RoutePolicyTransitionRecord:
    return RoutePolicyTransitionRecord(
        episode_id=episode_id,
        scene_id=scene_id,
        episode_index=episode_index,
        step_index=step_index,
        observation={"x": 0.0},
        action={"target": [0.0, 0.0, 0.0]},
        reward=0.0,
        next_observation=next_observation or {"x": 0.0},
        terminated=terminated,
        truncated=truncated,
        info=info or {},
    )


def _build_episode(
    *,
    episode_id: str,
    scene_id: str,
    episode_index: int,
    transitions: tuple[RoutePolicyTransitionRecord, ...],
) -> RoutePolicyEpisodeRecord:
    return RoutePolicyEpisodeRecord(
        episode_id=episode_id,
        scene_id=scene_id,
        episode_index=episode_index,
        seed=None,
        initial_observation={"x": 0.0},
        reset_info={"sceneId": scene_id, "episodeIndex": episode_index},
        transitions=transitions,
    )


def test_policy_trace_event_round_trips_through_json() -> None:
    event = PolicyTraceEvent(
        event_name="goal_reached",
        timestamp_seconds=12.5,
        episode_id="scene-episode-0",
        episode_index=0,
        step_index=4,
        bag_timestamp_seconds=1700000012.5,
        tags=("scene:outdoor", "phase:approach"),
        metadata={"stepCount": 5, "totalReward": 1.25},
    )
    payload = event.to_dict()
    restored = policy_trace_event_from_dict(payload)
    assert restored == event
    assert payload["recordType"] == "route-policy-trace-event"
    assert payload["version"] == ROUTE_POLICY_TRACE_EVENT_VERSION


def test_policy_trace_event_rejects_empty_event_name() -> None:
    with pytest.raises(ValueError, match="event_name"):
        PolicyTraceEvent(
            event_name="",
            timestamp_seconds=0.0,
            episode_id="e",
            episode_index=0,
            step_index=0,
        )


def test_policy_trace_event_rejects_non_finite_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_seconds"):
        PolicyTraceEvent(
            event_name="collision",
            timestamp_seconds=math.inf,
            episode_id="e",
            episode_index=0,
            step_index=0,
        )


def test_policy_trace_event_from_dict_rejects_unknown_version() -> None:
    payload = {
        "recordType": "route-policy-trace-event",
        "version": "gs-mapper-route-policy-trace-event/v999",
        "eventName": "collision",
        "timestampSeconds": 0.0,
        "episodeId": "e",
        "episodeIndex": 0,
        "stepIndex": 0,
    }
    with pytest.raises(ValueError, match="version"):
        policy_trace_event_from_dict(payload)


def test_policy_trace_event_to_dict_omits_optional_fields_when_absent() -> None:
    event = PolicyTraceEvent(
        event_name="goal_reached",
        timestamp_seconds=1.0,
        episode_id="scene-episode-0",
        episode_index=0,
        step_index=2,
    )
    payload = event.to_dict()
    assert "bagTimestampSeconds" not in payload
    assert "tags" not in payload
    assert "metadata" not in payload


def test_load_policy_trace_jsonl_round_trips_events(tmp_path: Path) -> None:
    events = (
        PolicyTraceEvent(
            event_name="near_miss",
            timestamp_seconds=2.5,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=1,
            tags=("scene:outdoor",),
            metadata={"clearanceMeters": 0.4, "thresholdMeters": 0.5},
        ),
        PolicyTraceEvent(
            event_name="goal_reached",
            timestamp_seconds=5.0,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=4,
        ),
    )
    path = tmp_path / "trace.jsonl"
    write_policy_trace_jsonl(path, events)
    restored = load_policy_trace_jsonl(path)
    assert restored == events


def test_load_policy_trace_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    payload = json.dumps(
        {
            "eventName": "collision",
            "timestampSeconds": 1.0,
            "episodeId": "e",
            "episodeIndex": 0,
            "stepIndex": 0,
        }
    )
    path.write_text("\n\n" + payload + "\n\n", encoding="utf-8")
    events = load_policy_trace_jsonl(path)
    assert len(events) == 1
    assert events[0].event_name == "collision"


def test_extract_policy_trace_events_from_dataset_emits_goal_reached_terminal() -> None:
    transitions = (
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=0,
        ),
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=1,
            info={"goal_reached": True, "termination_reason": "goal-reached"},
            terminated=True,
        ),
    )
    episode = _build_episode(
        episode_id="scene-episode-0",
        scene_id="outdoor",
        episode_index=0,
        transitions=transitions,
    )
    dataset = RoutePolicyDatasetExport(dataset_id="ds", episodes=(episode,))

    events = extract_policy_trace_events_from_dataset(dataset, segment_duration_seconds=0.5)

    assert len(events) == 1
    terminal = events[0]
    assert terminal.event_name == "goal_reached"
    assert terminal.episode_id == "scene-episode-0"
    assert terminal.step_index == 1
    assert terminal.timestamp_seconds == pytest.approx(0.5 * (1 + 1))
    assert terminal.tags == ("scene:outdoor",)
    assert terminal.metadata["terminationReason"] == "goal-reached"


def test_extract_policy_trace_events_from_dataset_emits_collision_terminal() -> None:
    transitions = (
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=0,
            info={"blocked": True, "termination_reason": "blocked"},
            terminated=True,
        ),
    )
    episode = _build_episode(
        episode_id="scene-episode-0",
        scene_id="outdoor",
        episode_index=0,
        transitions=transitions,
    )
    dataset = RoutePolicyDatasetExport(dataset_id="ds", episodes=(episode,))

    events = extract_policy_trace_events_from_dataset(dataset)

    assert [event.event_name for event in events] == ["collision"]


def test_extract_policy_trace_events_from_dataset_emits_truncated_terminal() -> None:
    transitions = (
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=0,
            info={"termination_reason": "collector-max-steps"},
            truncated=True,
        ),
    )
    episode = _build_episode(
        episode_id="scene-episode-0",
        scene_id="outdoor",
        episode_index=0,
        transitions=transitions,
    )
    dataset = RoutePolicyDatasetExport(dataset_id="ds", episodes=(episode,))

    events = extract_policy_trace_events_from_dataset(dataset)

    assert [event.event_name for event in events] == ["truncated"]


def test_extract_policy_trace_events_from_dataset_emits_near_miss_under_threshold() -> None:
    transitions = (
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=0,
            next_observation={"nearest-dynamic-obstacle-clearance-meters": 0.3},
        ),
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=1,
            next_observation={"nearest-dynamic-obstacle-clearance-meters": 1.0},
        ),
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=2,
            info={"goal_reached": True, "termination_reason": "goal-reached"},
            next_observation={"nearest-dynamic-obstacle-clearance-meters": 0.8},
            terminated=True,
        ),
    )
    episode = _build_episode(
        episode_id="scene-episode-0",
        scene_id="outdoor",
        episode_index=0,
        transitions=transitions,
    )
    dataset = RoutePolicyDatasetExport(dataset_id="ds", episodes=(episode,))

    events = extract_policy_trace_events_from_dataset(
        dataset,
        near_miss_clearance_meters=0.5,
    )

    assert [event.event_name for event in events] == ["near_miss", "goal_reached"]
    near_miss = events[0]
    assert near_miss.step_index == 0
    assert near_miss.metadata["clearanceMeters"] == pytest.approx(0.3)
    assert near_miss.metadata["thresholdMeters"] == pytest.approx(0.5)


def test_extract_policy_trace_events_from_dataset_respects_time_offset_and_episode_offsets() -> None:
    transitions = (
        _build_transition(
            episode_id="scene-episode-0",
            scene_id="outdoor",
            episode_index=0,
            step_index=0,
            info={"goal_reached": True},
            terminated=True,
        ),
    )
    dataset = RoutePolicyDatasetExport(
        dataset_id="ds",
        episodes=(
            _build_episode(
                episode_id="scene-episode-0",
                scene_id="outdoor",
                episode_index=0,
                transitions=transitions,
            ),
        ),
    )

    events = extract_policy_trace_events_from_dataset(
        dataset,
        segment_duration_seconds=2.0,
        time_offset_seconds=100.0,
        base_episode_offsets_seconds={"scene-episode-0": 50.0},
    )

    assert len(events) == 1
    # 100 (global) + 50 (episode) + (step_index + 1) * 2.0 = 152.0
    assert events[0].timestamp_seconds == pytest.approx(152.0)


def test_convert_policy_trace_events_to_event_windows_creates_half_width_windows() -> None:
    events = (
        PolicyTraceEvent(
            event_name="collision",
            timestamp_seconds=10.0,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=3,
            tags=("scene:outdoor",),
        ),
    )

    windows = convert_policy_trace_events_to_event_windows(
        events,
        half_width_seconds=1.5,
        time_offset_seconds=5.0,
    )

    assert len(windows) == 1
    window = windows[0]
    assert isinstance(window, CorrelationEventWindow)
    assert window.start_time == pytest.approx(10.0 + 5.0 - 1.5)
    assert window.end_time == pytest.approx(10.0 + 5.0 + 1.5)
    assert window.source == "policy_trace"
    assert "policy-trace" in window.tags
    assert "event:collision" in window.tags
    assert "scene:outdoor" in window.tags
    assert window.name == "collision-scene-episode-0-3"


def test_convert_policy_trace_events_uses_bag_timestamp_when_available() -> None:
    events = (
        PolicyTraceEvent(
            event_name="goal_reached",
            timestamp_seconds=1.0,
            bag_timestamp_seconds=1700000100.0,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=5,
        ),
    )

    windows = convert_policy_trace_events_to_event_windows(
        events,
        half_width_seconds=0.25,
        time_offset_seconds=999.0,  # ignored when bag_timestamp_seconds is set
    )

    assert windows[0].start_time == pytest.approx(1700000100.0 - 0.25)
    assert windows[0].end_time == pytest.approx(1700000100.0 + 0.25)


def test_convert_policy_trace_events_rejects_non_positive_half_width() -> None:
    with pytest.raises(ValueError, match="half_width_seconds"):
        convert_policy_trace_events_to_event_windows((), half_width_seconds=0.0)


def test_write_correlation_event_windows_json_round_trips_via_loader(tmp_path: Path) -> None:
    events = (
        PolicyTraceEvent(
            event_name="goal_reached",
            timestamp_seconds=2.0,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=1,
        ),
    )
    windows = convert_policy_trace_events_to_event_windows(events, half_width_seconds=0.5)
    output = tmp_path / "windows.json"
    write_correlation_event_windows_json(output, windows)

    loaded = load_correlation_event_windows_json(output)
    assert loaded == windows
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recordType"] == CORRELATION_EVENT_WINDOWS_VERSION


def test_cli_route_policy_dataset_to_trace_subcommand_handler(tmp_path: Path) -> None:
    dataset = RoutePolicyDatasetExport(
        dataset_id="ds",
        episodes=(
            _build_episode(
                episode_id="scene-episode-0",
                scene_id="outdoor",
                episode_index=0,
                transitions=(
                    _build_transition(
                        episode_id="scene-episode-0",
                        scene_id="outdoor",
                        episode_index=0,
                        step_index=0,
                        info={"goal_reached": True, "termination_reason": "goal-reached"},
                        next_observation={
                            "nearest-dynamic-obstacle-clearance-meters": 0.2,
                        },
                        terminated=True,
                    ),
                ),
            ),
        ),
    )
    dataset_path = tmp_path / "dataset.json"
    write_route_policy_dataset_json(dataset_path, dataset)
    trace_path = tmp_path / "trace.jsonl"

    args = argparse.Namespace(
        dataset=str(dataset_path),
        output=str(trace_path),
        segment_duration_seconds=0.5,
        near_miss_clearance_meters=0.5,
        near_miss_feature_key="nearest-dynamic-obstacle-clearance-meters",
        time_offset_seconds=0.0,
    )
    run_dataset_to_trace_cli(args)

    events = load_policy_trace_jsonl(trace_path)
    assert [event.event_name for event in events] == ["near_miss", "goal_reached"]


def test_cli_route_policy_trace_to_event_windows_subcommand_handler(tmp_path: Path) -> None:
    events = (
        PolicyTraceEvent(
            event_name="collision",
            timestamp_seconds=4.0,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=2,
        ),
    )
    trace_path = tmp_path / "trace.jsonl"
    write_policy_trace_jsonl(trace_path, events)
    output_path = tmp_path / "windows.json"

    args = argparse.Namespace(
        trace=str(trace_path),
        output=str(output_path),
        half_width_seconds=0.5,
        time_offset_seconds=0.0,
        name_template="{event_name}-{step_index}",
    )
    run_trace_to_event_windows_cli(args)

    windows = load_correlation_event_windows_json(output_path)
    assert len(windows) == 1
    assert windows[0].name == "collision-2"
    assert windows[0].source == "policy_trace"


def test_cli_route_policy_trace_to_event_windows_argparse_round_trip(
    tmp_path: Path,
) -> None:
    """End-to-end argparse smoke: build_parser() accepts the new subcommand flags."""

    events = (
        PolicyTraceEvent(
            event_name="goal_reached",
            timestamp_seconds=2.0,
            episode_id="scene-episode-0",
            episode_index=0,
            step_index=1,
        ),
    )
    trace_path = tmp_path / "trace.jsonl"
    write_policy_trace_jsonl(trace_path, events)
    output_path = tmp_path / "windows.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gs_sim2real.cli",
            "route-policy-trace-to-event-windows",
            "--trace",
            str(trace_path),
            "--output",
            str(output_path),
            "--half-width-seconds",
            "0.25",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    windows = load_correlation_event_windows_json(output_path)
    assert windows[0].start_time == pytest.approx(2.0 - 0.25)
