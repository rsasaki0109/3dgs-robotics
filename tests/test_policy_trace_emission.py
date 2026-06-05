"""Tests for live route policy trace emission (Sprint 3 / PR C2).

These exercise the live emission path (`RoutePolicyTraceEmitter`,
`JsonlPolicyTraceEventStream`, and the Gym adapter hook) so a benchmark
or imitation rollout can stream `PolicyTraceEvent` records to disk
without the post-hoc `RoutePolicyDatasetExport` round-trip covered by
`test_policy_trace.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    JsonlPolicyTraceEventStream,
    PolicyTraceEmissionConfig,
    PolicyTraceEvent,
    Pose3D,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyTraceEmitter,
    RouteRewardWeights,
    build_simulation_catalog,
    load_policy_trace_jsonl,
)


def test_jsonl_policy_trace_event_stream_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "trace.jsonl"
    event = PolicyTraceEvent(
        event_name="goal_reached",
        timestamp_seconds=1.0,
        episode_id="unit-scene-episode-0",
        episode_index=0,
        step_index=0,
        tags=("scene:unit-scene",),
        metadata={"terminationReason": "goal-reached"},
    )

    with JsonlPolicyTraceEventStream(output_path) as stream:
        stream.emit(event)

    loaded = load_policy_trace_jsonl(output_path)
    assert loaded == (event,)


def test_jsonl_policy_trace_event_stream_flushes_per_event(tmp_path: Path) -> None:
    """Each emit must be readable from disk before close — crash safety."""

    output_path = tmp_path / "trace.jsonl"
    stream = JsonlPolicyTraceEventStream(output_path)
    try:
        stream.emit(
            PolicyTraceEvent(
                event_name="near_miss",
                timestamp_seconds=0.5,
                episode_id="unit-scene-episode-0",
                episode_index=0,
                step_index=0,
            )
        )
        # Reading before close must already see the event (flush guarantee).
        events_mid_stream = load_policy_trace_jsonl(output_path)
        assert len(events_mid_stream) == 1
        stream.emit(
            PolicyTraceEvent(
                event_name="goal_reached",
                timestamp_seconds=1.0,
                episode_id="unit-scene-episode-0",
                episode_index=0,
                step_index=0,
            )
        )
    finally:
        stream.close()

    events = load_policy_trace_jsonl(output_path)
    assert tuple(event.event_name for event in events) == ("near_miss", "goal_reached")


def test_jsonl_policy_trace_event_stream_rejects_emit_after_close(tmp_path: Path) -> None:
    stream = JsonlPolicyTraceEventStream(tmp_path / "trace.jsonl")
    stream.close()
    with pytest.raises(RuntimeError):
        stream.emit(
            PolicyTraceEvent(
                event_name="terminated",
                timestamp_seconds=0.0,
                episode_id="ep",
                episode_index=0,
                step_index=0,
            )
        )


def test_jsonl_policy_trace_event_stream_close_is_idempotent(tmp_path: Path) -> None:
    stream = JsonlPolicyTraceEventStream(tmp_path / "trace.jsonl")
    stream.close()
    stream.close()  # must not raise


def test_jsonl_policy_trace_event_stream_append_mode_preserves_existing(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with JsonlPolicyTraceEventStream(path) as stream:
        stream.emit(
            PolicyTraceEvent(
                event_name="near_miss",
                timestamp_seconds=0.5,
                episode_id="ep-0",
                episode_index=0,
                step_index=0,
            )
        )
    with JsonlPolicyTraceEventStream(path, mode="a") as stream:
        stream.emit(
            PolicyTraceEvent(
                event_name="goal_reached",
                timestamp_seconds=1.0,
                episode_id="ep-0",
                episode_index=0,
                step_index=0,
            )
        )
    events = load_policy_trace_jsonl(path)
    assert tuple(event.event_name for event in events) == ("near_miss", "goal_reached")


def test_policy_trace_emission_config_rejects_non_positive_segment_duration() -> None:
    with pytest.raises(ValueError, match="segment_duration_seconds"):
        PolicyTraceEmissionConfig(segment_duration_seconds=0.0)


def test_policy_trace_emission_config_requires_episode_index_token() -> None:
    with pytest.raises(ValueError, match="episode_id_template"):
        PolicyTraceEmissionConfig(episode_id_template="{scene_id}-static")


def test_route_policy_trace_emitter_requires_begin_episode_before_step() -> None:
    emitter = RoutePolicyTraceEmitter()
    with pytest.raises(RuntimeError, match="begin_episode"):
        emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=0,
            next_observation={},
            blocked=False,
            goal_reached=True,
            truncated=False,
            terminated=True,
        )


def test_route_policy_trace_emitter_emits_terminal_goal_event() -> None:
    emitter = RoutePolicyTraceEmitter()
    emitter.begin_episode(scene_id="unit-scene", episode_index=2)
    emitted = emitter.record_step(
        scene_id="unit-scene",
        episode_index=2,
        step_index=3,
        next_observation={},
        blocked=False,
        goal_reached=True,
        truncated=False,
        terminated=True,
    )
    assert len(emitted) == 1
    event = emitted[0]
    assert event.event_name == "goal_reached"
    assert event.episode_id == "unit-scene-episode-2"
    assert event.episode_index == 2
    assert event.step_index == 3
    # default segment_duration=1.0, time_offset=0.0 -> (step_index+1) * 1.0 = 4.0
    assert event.timestamp_seconds == pytest.approx(4.0)
    assert event.metadata["terminationReason"] == "goal-reached"


def test_route_policy_trace_emitter_emits_terminal_collision_event() -> None:
    emitter = RoutePolicyTraceEmitter()
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    emitted = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=0,
        next_observation={},
        blocked=True,
        goal_reached=False,
        truncated=False,
        terminated=True,
    )
    assert tuple(event.event_name for event in emitted) == ("collision",)
    assert emitted[0].metadata["terminationReason"] == "blocked-route"


def test_route_policy_trace_emitter_emits_terminal_truncated_event() -> None:
    emitter = RoutePolicyTraceEmitter()
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    emitted = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=5,
        next_observation={},
        blocked=False,
        goal_reached=False,
        truncated=True,
        terminated=False,
    )
    assert tuple(event.event_name for event in emitted) == ("truncated",)
    assert emitted[0].metadata["terminationReason"] == "max-steps"


def test_route_policy_trace_emitter_falls_back_to_terminated_label() -> None:
    emitter = RoutePolicyTraceEmitter()
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    emitted = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=0,
        next_observation={},
        blocked=False,
        goal_reached=False,
        truncated=False,
        terminated=True,
    )
    assert tuple(event.event_name for event in emitted) == ("terminated",)


def test_route_policy_trace_emitter_emits_no_event_for_non_terminal_step() -> None:
    emitter = RoutePolicyTraceEmitter()
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    emitted = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=0,
        next_observation={"nearest-dynamic-obstacle-distance-meters": 5.0},
        blocked=False,
        goal_reached=False,
        truncated=False,
        terminated=False,
    )
    assert emitted == ()


def test_route_policy_trace_emitter_near_miss_edge_only_fires_once_per_crossing() -> None:
    """Default near_miss_edge_only=True: emit on descent, suppress while held."""

    emitter = RoutePolicyTraceEmitter(
        config=PolicyTraceEmissionConfig(near_miss_clearance_meters=0.5),
    )
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    # Step 0: clearance above threshold — no event.
    assert (
        emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=0,
            next_observation={"nearest-dynamic-obstacle-distance-meters": 1.0},
            blocked=False,
            goal_reached=False,
            truncated=False,
            terminated=False,
        )
        == ()
    )
    # Step 1: crosses threshold — emit one near_miss.
    fired = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=1,
        next_observation={"nearest-dynamic-obstacle-distance-meters": 0.4},
        blocked=False,
        goal_reached=False,
        truncated=False,
        terminated=False,
    )
    assert tuple(event.event_name for event in fired) == ("near_miss",)
    assert fired[0].metadata["clearanceMeters"] == pytest.approx(0.4)
    assert fired[0].metadata["thresholdMeters"] == pytest.approx(0.5)
    # Step 2: still below threshold — suppressed by edge detector.
    assert (
        emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=2,
            next_observation={"nearest-dynamic-obstacle-distance-meters": 0.3},
            blocked=False,
            goal_reached=False,
            truncated=False,
            terminated=False,
        )
        == ()
    )
    # Step 3: clearance recovers above threshold — no event but state resets.
    assert (
        emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=3,
            next_observation={"nearest-dynamic-obstacle-distance-meters": 0.9},
            blocked=False,
            goal_reached=False,
            truncated=False,
            terminated=False,
        )
        == ()
    )
    # Step 4: crosses again — must fire a fresh event.
    refired = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=4,
        next_observation={"nearest-dynamic-obstacle-distance-meters": 0.2},
        blocked=False,
        goal_reached=False,
        truncated=False,
        terminated=False,
    )
    assert tuple(event.event_name for event in refired) == ("near_miss",)


def test_route_policy_trace_emitter_near_miss_continuous_mode_fires_every_step() -> None:
    emitter = RoutePolicyTraceEmitter(
        config=PolicyTraceEmissionConfig(
            near_miss_clearance_meters=0.5,
            near_miss_edge_only=False,
        ),
    )
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    for step_index in range(3):
        events = emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=step_index,
            next_observation={"nearest-dynamic-obstacle-distance-meters": 0.3},
            blocked=False,
            goal_reached=False,
            truncated=False,
            terminated=False,
        )
        assert tuple(event.event_name for event in events) == ("near_miss",)


def test_route_policy_trace_emitter_combines_near_miss_with_terminal_event() -> None:
    emitter = RoutePolicyTraceEmitter(
        config=PolicyTraceEmissionConfig(near_miss_clearance_meters=0.5),
    )
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    events = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=0,
        next_observation={"nearest-dynamic-obstacle-distance-meters": 0.2},
        blocked=True,
        goal_reached=False,
        truncated=False,
        terminated=True,
    )
    assert tuple(event.event_name for event in events) == ("near_miss", "collision")


def test_route_policy_trace_emitter_writes_to_stream_when_attached(tmp_path: Path) -> None:
    output_path = tmp_path / "live.jsonl"
    with JsonlPolicyTraceEventStream(output_path) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        emitter.begin_episode(scene_id="unit-scene", episode_index=0)
        emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=0,
            next_observation={},
            blocked=False,
            goal_reached=True,
            truncated=False,
            terminated=True,
        )
    events = load_policy_trace_jsonl(output_path)
    assert tuple(event.event_name for event in events) == ("goal_reached",)


def test_route_policy_trace_emitter_close_closes_underlying_stream(tmp_path: Path) -> None:
    stream = JsonlPolicyTraceEventStream(tmp_path / "live.jsonl")
    emitter = RoutePolicyTraceEmitter(stream=stream)
    emitter.close()
    with pytest.raises(RuntimeError):
        stream.emit(
            PolicyTraceEvent(
                event_name="terminated",
                timestamp_seconds=0.0,
                episode_id="ep",
                episode_index=0,
                step_index=0,
            )
        )


def test_route_policy_trace_emitter_resets_near_miss_state_between_episodes() -> None:
    emitter = RoutePolicyTraceEmitter(
        config=PolicyTraceEmissionConfig(near_miss_clearance_meters=0.5),
    )
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    # First episode descent into the threshold.
    assert tuple(
        event.event_name
        for event in emitter.record_step(
            scene_id="unit-scene",
            episode_index=0,
            step_index=0,
            next_observation={"nearest-dynamic-obstacle-distance-meters": 0.3},
            blocked=False,
            goal_reached=False,
            truncated=False,
            terminated=False,
        )
    ) == ("near_miss",)
    # Episode 2 starts: descent should fire again even though we never crossed up.
    emitter.begin_episode(scene_id="unit-scene", episode_index=1)
    events = emitter.record_step(
        scene_id="unit-scene",
        episode_index=1,
        step_index=0,
        next_observation={"nearest-dynamic-obstacle-distance-meters": 0.2},
        blocked=False,
        goal_reached=False,
        truncated=False,
        terminated=False,
    )
    assert tuple(event.event_name for event in events) == ("near_miss",)
    assert events[0].episode_id == "unit-scene-episode-1"


def test_route_policy_trace_emitter_honours_time_offset_in_timestamps() -> None:
    emitter = RoutePolicyTraceEmitter(
        config=PolicyTraceEmissionConfig(
            segment_duration_seconds=2.0,
            time_offset_seconds=100.0,
        ),
    )
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    (event,) = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=4,
        next_observation={},
        blocked=False,
        goal_reached=True,
        truncated=False,
        terminated=True,
    )
    # 100.0 + (4 + 1) * 2.0 = 110.0
    assert event.timestamp_seconds == pytest.approx(110.0)


def test_route_policy_trace_emitter_ignores_non_finite_clearance() -> None:
    emitter = RoutePolicyTraceEmitter(
        config=PolicyTraceEmissionConfig(near_miss_clearance_meters=0.5),
    )
    emitter.begin_episode(scene_id="unit-scene", episode_index=0)
    events = emitter.record_step(
        scene_id="unit-scene",
        episode_index=0,
        step_index=0,
        next_observation={"nearest-dynamic-obstacle-distance-meters": float("nan")},
        blocked=False,
        goal_reached=False,
        truncated=False,
        terminated=False,
    )
    assert events == ()


def test_gym_adapter_streams_terminal_event_to_emitter(tmp_path: Path) -> None:
    output_path = tmp_path / "live.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    with JsonlPolicyTraceEventStream(output_path) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        adapter = RoutePolicyGymAdapter(
            env,
            RoutePolicyEnvConfig(
                scene_id="unit-scene",
                reward_weights=RouteRewardWeights(
                    distance_penalty_per_meter=0.0,
                    step_penalty=0.0,
                ),
            ),
            trace_emitter=emitter,
        )
        adapter.reset(goal=_unit_pose((0.25, 0.0, 0.0)))
        _, _, terminated, truncated, info = adapter.step((0.25, 0.0, 0.0))
        assert terminated is True
        assert truncated is False
        assert info["termination_reason"] == "goal-reached"
    events = load_policy_trace_jsonl(output_path)
    assert len(events) == 1
    event = events[0]
    assert event.event_name == "goal_reached"
    assert event.episode_id == "unit-scene-episode-0"
    assert event.step_index == 0


def test_gym_adapter_streams_collision_event_to_emitter(tmp_path: Path) -> None:
    output_path = tmp_path / "live.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    with JsonlPolicyTraceEventStream(output_path) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        adapter = RoutePolicyGymAdapter(
            env,
            RoutePolicyEnvConfig(
                scene_id="unit-scene",
                reward_weights=RouteRewardWeights(
                    distance_penalty_per_meter=0.0,
                    step_penalty=0.0,
                ),
            ),
            trace_emitter=emitter,
        )
        adapter.reset(goal=_unit_pose((0.5, 0.0, 0.0)))
        adapter.step({"target": {"x": 2.0, "y": 0.0, "z": 0.0}})
    events = load_policy_trace_jsonl(output_path)
    assert tuple(event.event_name for event in events) == ("collision",)


def test_gym_adapter_streams_truncated_event_when_max_steps_reached(tmp_path: Path) -> None:
    output_path = tmp_path / "live.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    with JsonlPolicyTraceEventStream(output_path) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        adapter = RoutePolicyGymAdapter(
            env,
            RoutePolicyEnvConfig(
                scene_id="unit-scene",
                max_steps=1,
                reward_weights=RouteRewardWeights(
                    distance_penalty_per_meter=0.0,
                    step_penalty=0.0,
                ),
            ),
            trace_emitter=emitter,
        )
        adapter.reset(goal=_unit_pose((0.75, 0.0, 0.0)))
        adapter.step({"target": {"x": 0.25, "y": 0.0, "z": 0.0}})
    events = load_policy_trace_jsonl(output_path)
    assert tuple(event.event_name for event in events) == ("truncated",)


def test_gym_adapter_emits_one_terminal_event_per_episode_across_resets(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "live.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    with JsonlPolicyTraceEventStream(output_path) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        adapter = RoutePolicyGymAdapter(
            env,
            RoutePolicyEnvConfig(
                scene_id="unit-scene",
                reward_weights=RouteRewardWeights(
                    distance_penalty_per_meter=0.0,
                    step_penalty=0.0,
                ),
            ),
            trace_emitter=emitter,
        )
        for _ in range(2):
            adapter.reset(goal=_unit_pose((0.25, 0.0, 0.0)))
            adapter.step((0.25, 0.0, 0.0))
    events = load_policy_trace_jsonl(output_path)
    assert len(events) == 2
    assert events[0].episode_id == "unit-scene-episode-0"
    assert events[1].episode_id == "unit-scene-episode-1"


def _build_unit_catalog():
    return build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/unit-scene/unit-scene.splat",
                    "label": "Unit Scene",
                    "summary": "Generic unit scene",
                }
            ]
        },
        docs_root=Path("."),
        site_url="https://example.test/gs/",
    )


def _unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
