"""Tests for trace emitter pass-through in the route replanning helpers (Sprint 3 / PR C5).

PR C4 wired ``RoutePolicyTraceEmitter`` directly into
``route_execution.rollout_route``. The closed-loop replanning helpers
(``replan_after_blocked_rollout`` and ``rollout_route_with_replanning``)
build on top of that primitive and own their own rollout sequence, so the
emitter needs to flow through them too. This module covers the kwargs
plumbing: that the emitter is forwarded to recovery rollouts, that the
initial rollout and each replan are emitted as distinct episodes with
auto-incremented episode indices, that ``execute=False`` does not call
the emitter, and that omitting the emitter preserves the prior behaviour.
"""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    JsonlPolicyTraceEventStream,
    Pose3D,
    RouteCandidate,
    RoutePolicyTraceEmitter,
    build_simulation_catalog,
    load_policy_trace_jsonl,
    replan_after_blocked_rollout,
    rollout_route,
    rollout_route_with_replanning,
)


def test_replan_after_blocked_rollout_forwards_emitter_to_recovery_rollout(
    tmp_path: Path,
) -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("blocked", (start, outside)), action_type="teleport")
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    sink = tmp_path / "trace.jsonl"
    emitter = RoutePolicyTraceEmitter(stream=JsonlPolicyTraceEventStream(sink))

    replan = replan_after_blocked_rollout(
        env,
        scene_id="unit-scene",
        rollout=rollout,
        candidates=(RouteCandidate("recovery", (goal,)),),
        execute=True,
        action_type="teleport",
        trace_emitter=emitter,
        trace_scene_id="recovery-scene",
        trace_episode_index=3,
    )
    emitter.close()

    assert replan.passed is True
    events = load_policy_trace_jsonl(sink)
    assert len(events) == 1
    assert events[0].event_name == "goal_reached"
    assert events[0].episode_id == "recovery-scene-episode-3"
    assert events[0].tags == ("scene:recovery-scene",)


def test_replan_after_blocked_rollout_skips_emitter_when_not_executing() -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("blocked", (start, outside)), action_type="teleport")
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    begin_calls: list[tuple[str, int]] = []

    class CapturingEmitter(RoutePolicyTraceEmitter):
        def begin_episode(self, *, scene_id: str, episode_index: int) -> str:  # type: ignore[override]
            begin_calls.append((scene_id, episode_index))
            return super().begin_episode(scene_id=scene_id, episode_index=episode_index)

    emitter = CapturingEmitter()
    replan = replan_after_blocked_rollout(
        env,
        scene_id="unit-scene",
        rollout=rollout,
        candidates=(RouteCandidate("planned-only", (goal,)),),
        execute=False,
        trace_emitter=emitter,
    )

    assert replan.rollout is None
    assert begin_calls == [], "emitter must remain untouched when execute=False"


def test_rollout_route_with_replanning_emits_one_episode_per_rollout(
    tmp_path: Path,
) -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    sink = tmp_path / "trace.jsonl"
    emitter = RoutePolicyTraceEmitter(stream=JsonlPolicyTraceEventStream(sink))

    closed_loop = rollout_route_with_replanning(
        env,
        scene_id="unit-scene",
        initial_route=RouteCandidate("initial-blocked", (start, outside)),
        replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
        action_type="teleport",
        trace_emitter=emitter,
        trace_scene_id="closed-loop-scene",
    )
    emitter.close()

    assert closed_loop.passed is True
    assert len(closed_loop.rollouts) == 2

    events = load_policy_trace_jsonl(sink)
    assert len(events) == 2, "one terminal event per rollout (initial + replan)"
    assert events[0].event_name == "collision"
    assert events[0].episode_id == "closed-loop-scene-episode-0"
    assert events[1].event_name == "goal_reached"
    assert events[1].episode_id == "closed-loop-scene-episode-1"


def test_rollout_route_with_replanning_auto_increments_from_base_episode_index() -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    captured: list[tuple[str, int]] = []

    class CapturingEmitter(RoutePolicyTraceEmitter):
        def begin_episode(self, *, scene_id: str, episode_index: int) -> str:  # type: ignore[override]
            captured.append((scene_id, episode_index))
            return super().begin_episode(scene_id=scene_id, episode_index=episode_index)

    emitter = CapturingEmitter()
    rollout_route_with_replanning(
        env,
        scene_id="unit-scene",
        initial_route=RouteCandidate("initial-blocked", (start, outside)),
        replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
        action_type="teleport",
        trace_emitter=emitter,
        trace_scene_id="closed-loop-scene",
        trace_base_episode_index=10,
    )

    assert captured == [
        ("closed-loop-scene", 10),
        ("closed-loop-scene", 11),
    ]


def test_rollout_route_with_replanning_skips_replan_emit_when_initial_passes() -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    goal = _pose_like(start, position=(0.25, 0.0, 0.0))
    unused = _pose_like(start, position=(0.5, 0.0, 0.0))

    captured_indices: list[int] = []

    class CapturingEmitter(RoutePolicyTraceEmitter):
        def begin_episode(self, *, scene_id: str, episode_index: int) -> str:  # type: ignore[override]
            captured_indices.append(episode_index)
            return super().begin_episode(scene_id=scene_id, episode_index=episode_index)

    emitter = CapturingEmitter()
    closed_loop = rollout_route_with_replanning(
        env,
        scene_id="unit-scene",
        initial_route=RouteCandidate("clear-initial", (start, goal)),
        replan_candidate_batches=((RouteCandidate("never-used", (unused,)),),),
        action_type="teleport",
        trace_emitter=emitter,
    )

    assert closed_loop.passed is True
    assert len(closed_loop.rollouts) == 1
    assert len(closed_loop.replans) == 0
    assert captured_indices == [0], "only initial rollout should open an episode"


def test_rollout_route_with_replanning_emitter_optional_keeps_old_behavior() -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    closed_loop = rollout_route_with_replanning(
        env,
        scene_id="unit-scene",
        initial_route=RouteCandidate("initial-blocked", (start, outside)),
        replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
        action_type="teleport",
    )

    assert closed_loop.passed is True
    assert len(closed_loop.rollouts) == 2
    assert len(closed_loop.replans) == 1


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


def _pose_like(template: Pose3D, *, position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=template.orientation_xyzw,
        frame_id=template.frame_id,
    )
