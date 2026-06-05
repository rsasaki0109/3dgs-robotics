"""Tests for per-policy trace emitter factory wiring (Sprint 3 / PR C3).

`evaluate_route_policy_baselines` and the public benchmark runners accept
a ``trace_emitter_factory`` so each baseline policy can stream live
``PolicyTraceEvent`` records to its own sink (e.g. a JSONL file). This
covers the plumbing end-to-end: the factory is invoked once per policy,
the produced emitter receives terminal events from the rollouts, the
emitter is closed once that policy finishes, and the factory remains
optional with backwards-compatible behaviour when omitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    JsonlPolicyTraceEventStream,
    PolicyTraceEmissionConfig,
    Pose3D,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyTraceEmitter,
    RouteRewardWeights,
    build_simulation_catalog,
    evaluate_route_policy_baselines,
    load_policy_trace_jsonl,
)


def test_evaluate_route_policy_baselines_invokes_factory_once_per_policy() -> None:
    adapters = (_unit_adapter(),)
    invoked_with: list[str] = []
    emitters: dict[str, RoutePolicyTraceEmitter] = {}

    def factory(policy_name: str) -> RoutePolicyTraceEmitter:
        invoked_with.append(policy_name)
        emitter = RoutePolicyTraceEmitter()
        emitters[policy_name] = emitter
        return emitter

    evaluate_route_policy_baselines(
        adapters,
        {"alpha": _direct_goal_policy, "beta": _direct_goal_policy},
        episode_count=1,
        goals=[_goal((0.25, 0.0, 0.0))],
        trace_emitter_factory=factory,
    )

    assert invoked_with == ["alpha", "beta"]
    assert set(emitters) == {"alpha", "beta"}


def test_evaluate_route_policy_baselines_streams_terminal_events_per_policy(
    tmp_path: Path,
) -> None:
    adapters = (_unit_adapter(),)
    sinks: dict[str, Path] = {
        "alpha": tmp_path / "alpha.jsonl",
        "beta": tmp_path / "beta.jsonl",
    }

    def factory(policy_name: str) -> RoutePolicyTraceEmitter:
        stream = JsonlPolicyTraceEventStream(sinks[policy_name])
        return RoutePolicyTraceEmitter(stream=stream)

    evaluate_route_policy_baselines(
        adapters,
        {"alpha": _direct_goal_policy, "beta": _direct_goal_policy},
        episode_count=2,
        goals=[_goal((0.25, 0.0, 0.0))],
        trace_emitter_factory=factory,
    )

    for policy_name, path in sinks.items():
        events = load_policy_trace_jsonl(path)
        assert len(events) == 2, f"{policy_name} should have one event per episode"
        assert all(event.event_name == "goal_reached" for event in events)
        episode_ids = {event.episode_id for event in events}
        assert len(episode_ids) == 2  # one per episode_index


def test_evaluate_route_policy_baselines_closes_emitter_after_collection() -> None:
    adapters = (_unit_adapter(),)
    closed: list[str] = []

    class TrackingEmitter(RoutePolicyTraceEmitter):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name

        def close(self) -> None:  # type: ignore[override]
            closed.append(self._name)
            super().close()

    evaluate_route_policy_baselines(
        adapters,
        {"alpha": _direct_goal_policy, "beta": _direct_goal_policy},
        episode_count=1,
        goals=[_goal((0.25, 0.0, 0.0))],
        trace_emitter_factory=lambda name: TrackingEmitter(name),
    )

    assert closed == ["alpha", "beta"]


def test_evaluate_route_policy_baselines_closes_emitter_on_collection_error() -> None:
    adapters = (_unit_adapter(),)
    closed: list[str] = []

    def exploding_policy(observation: Mapping[str, float], info: Mapping[str, Any]) -> dict[str, Any]:
        del observation, info
        raise RuntimeError("boom")

    class TrackingEmitter(RoutePolicyTraceEmitter):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name

        def close(self) -> None:  # type: ignore[override]
            closed.append(self._name)
            super().close()

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_route_policy_baselines(
            adapters,
            {"alpha": exploding_policy},
            episode_count=1,
            goals=[_goal((0.25, 0.0, 0.0))],
            trace_emitter_factory=lambda name: TrackingEmitter(name),
        )

    assert closed == ["alpha"]


def test_evaluate_route_policy_baselines_factory_optional_keeps_old_behavior() -> None:
    adapters = (_unit_adapter(),)
    evaluation = evaluate_route_policy_baselines(
        adapters,
        {"alpha": _direct_goal_policy},
        episode_count=1,
        goals=[_goal((0.25, 0.0, 0.0))],
    )
    assert evaluation.results[0].policy_name == "alpha"
    assert evaluation.results[0].dataset.episodes  # rollouts ran end-to-end


def test_evaluate_route_policy_baselines_factory_can_skip_emitter() -> None:
    adapters = (_unit_adapter(),)

    def factory(policy_name: str) -> RoutePolicyTraceEmitter | None:
        if policy_name == "alpha":
            return None
        return RoutePolicyTraceEmitter()

    evaluation = evaluate_route_policy_baselines(
        adapters,
        {"alpha": _direct_goal_policy, "beta": _direct_goal_policy},
        episode_count=1,
        goals=[_goal((0.25, 0.0, 0.0))],
        trace_emitter_factory=factory,
    )
    assert {result.policy_name for result in evaluation.results} == {"alpha", "beta"}


def test_evaluate_route_policy_baselines_respects_per_policy_config(tmp_path: Path) -> None:
    adapters = (_unit_adapter(),)
    sink = tmp_path / "trace.jsonl"

    def factory(policy_name: str) -> RoutePolicyTraceEmitter:
        del policy_name
        stream = JsonlPolicyTraceEventStream(sink)
        return RoutePolicyTraceEmitter(
            stream=stream,
            config=PolicyTraceEmissionConfig(
                episode_id_template="{scene_id}/policy-{episode_index}",
            ),
        )

    evaluate_route_policy_baselines(
        adapters,
        {"alpha": _direct_goal_policy},
        episode_count=1,
        goals=[_goal((0.25, 0.0, 0.0))],
        trace_emitter_factory=factory,
    )

    events = load_policy_trace_jsonl(sink)
    assert len(events) == 1
    assert events[0].episode_id == "unit-scene/policy-0"


def _unit_adapter() -> RoutePolicyGymAdapter:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    return RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            reward_weights=RouteRewardWeights(
                distance_penalty_per_meter=0.0,
                step_penalty=0.0,
            ),
        ),
    )


def _direct_goal_policy(observation: Mapping[str, float], info: Mapping[str, Any]) -> dict[str, Any]:
    del observation
    return {
        "routeId": f"direct-{info.get('episodeIndex', 0)}-{info.get('stepIndex', 0)}",
        "target": info["goal"],
    }


def _goal(position: tuple[float, float, float]) -> Pose3D:
    return _unit_pose(position)


def _unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")


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
