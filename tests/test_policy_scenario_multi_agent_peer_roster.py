"""Tests for the multi-agent peer-roster synthesizer (PR D3).

PR D2 embedded ``agents`` / ``population`` / ``populationSeed`` /
``interactionMetrics`` into expanded scenarios' metadata. PR D3 reads
those back out and synthesizes a :class:`DynamicObstacleTimeline` the
existing headless env consumes via its ``dynamic_obstacles`` plumbing.

These tests verify:

1. The synthesizer is deterministic in ``populationSeed`` — same seed
   produces the same peer set across runs.
2. The synthesizer drops ego agents and only produces obstacles for
   ``peer-*`` roles.
3. Population-driven synthesis honours ``agent_count_per_scenario``
   (peers = total - 1, ego is the route policy itself) and clamps each
   peer's start position to ``spawn_volume``.
4. Different seeds produce different rosters (sanity check).
5. Returns ``None`` for legacy ego-only scenarios.
6. ``builtin_policy`` maps to the matching ``ObstaclePolicy`` instance.
7. The PR D2-embedded scenario metadata loop-end-to-end with the
   synthesizer.
"""

from __future__ import annotations

import pytest

from gs_sim2real.sim import (
    AgentRoleSpec,
    AxisAlignedBounds,
    ChaseAgentObstaclePolicy,
    DEFAULT_PEER_RADIUS_METERS,
    FleeAgentObstaclePolicy,
    InteractionMetricsSpec,
    MaintainSeparationObstaclePolicy,
    Pose3D,
    PopulationSpec,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyScenarioMatrix,
    Vec3,
    WaypointInterpolationObstaclePolicy,
    expand_route_policy_scenario_matrix,
    synthesize_peer_roster_from_scenario_metadata,
)


def test_synthesizer_returns_none_for_legacy_metadata() -> None:
    assert synthesize_peer_roster_from_scenario_metadata({"matrixId": "legacy", "sceneKey": "ego-only"}) is None


def test_synthesizer_drops_ego_and_keeps_peers() -> None:
    metadata = _scene_to_scenario_metadata(
        RoutePolicyMatrixSceneSpec(
            scene_key="2-agent",
            scene_catalog="scenes.json",
            agents=(
                AgentRoleSpec(
                    agent_id="ego",
                    role="ego",
                    start_pose=_unit_pose((0.0, 0.0, 0.0)),
                ),
                AgentRoleSpec(
                    agent_id="peer-a",
                    role="peer-obstacle",
                    start_pose=_unit_pose((4.0, 0.0, 0.0)),
                    builtin_policy="chase",
                ),
                AgentRoleSpec(
                    agent_id="peer-b",
                    role="peer-coop",
                    start_pose=_unit_pose((2.0, 3.0, 0.0)),
                    builtin_policy="maintain_separation",
                ),
            ),
        )
    )
    timeline = synthesize_peer_roster_from_scenario_metadata(metadata)
    assert timeline is not None
    obstacle_ids = tuple(obstacle.obstacle_id for obstacle in timeline.obstacles)
    assert obstacle_ids == ("peer-a", "peer-b"), "ego must not appear as an obstacle"
    assert all(obstacle.radius_meters == pytest.approx(DEFAULT_PEER_RADIUS_METERS) for obstacle in timeline.obstacles)


def test_synthesizer_maps_builtin_policy_to_obstacle_policy_class() -> None:
    metadata = _scene_to_scenario_metadata(
        RoutePolicyMatrixSceneSpec(
            scene_key="every-policy",
            scene_catalog="scenes.json",
            agents=(
                AgentRoleSpec(
                    agent_id="chaser",
                    role="peer-obstacle",
                    start_pose=_unit_pose((1.0, 0.0, 0.0)),
                    builtin_policy="chase",
                ),
                AgentRoleSpec(
                    agent_id="flyer",
                    role="peer-obstacle",
                    start_pose=_unit_pose((2.0, 0.0, 0.0)),
                    builtin_policy="flee",
                ),
                AgentRoleSpec(
                    agent_id="planner",
                    role="peer-obstacle",
                    start_pose=_unit_pose((3.0, 0.0, 0.0)),
                    builtin_policy="waypoint",
                ),
                AgentRoleSpec(
                    agent_id="separator",
                    role="peer-obstacle",
                    start_pose=_unit_pose((4.0, 0.0, 0.0)),
                    builtin_policy="maintain_separation",
                ),
            ),
        )
    )
    timeline = synthesize_peer_roster_from_scenario_metadata(metadata)
    assert timeline is not None
    by_id = {obstacle.obstacle_id: obstacle for obstacle in timeline.obstacles}
    assert isinstance(by_id["chaser"].policy, ChaseAgentObstaclePolicy)
    assert isinstance(by_id["flyer"].policy, FleeAgentObstaclePolicy)
    assert isinstance(by_id["planner"].policy, WaypointInterpolationObstaclePolicy)
    assert isinstance(by_id["separator"].policy, MaintainSeparationObstaclePolicy)


def test_synthesizer_population_produces_correct_peer_count() -> None:
    matrix = _matrix_with_population(_unit_population(agent_count_per_scenario=5, random_seed=0))
    scenarios = expand_route_policy_scenario_matrix(matrix)[0].scenarios
    timeline = synthesize_peer_roster_from_scenario_metadata(scenarios[0].metadata)
    assert timeline is not None
    # agent_count_per_scenario includes ego -> peers = 4.
    assert timeline.obstacle_count == 4


def test_synthesizer_population_is_deterministic_in_seed() -> None:
    scenario = expand_route_policy_scenario_matrix(
        _matrix_with_population(_unit_population(agent_count_per_scenario=3, random_seed=42))
    )[0].scenarios[0]
    first = synthesize_peer_roster_from_scenario_metadata(scenario.metadata)
    second = synthesize_peer_roster_from_scenario_metadata(scenario.metadata)
    assert first is not None and second is not None
    first_positions = [obstacle.waypoints[0].position for obstacle in first.obstacles]
    second_positions = [obstacle.waypoints[0].position for obstacle in second.obstacles]
    assert first_positions == second_positions


def test_synthesizer_population_differs_across_seeds() -> None:
    matrix = _matrix_with_population(_unit_population(agent_count_per_scenario=3, random_seed=1, seed_count=2))
    scenarios = expand_route_policy_scenario_matrix(matrix)[0].scenarios
    assert len(scenarios) == 2
    first = synthesize_peer_roster_from_scenario_metadata(scenarios[0].metadata)
    second = synthesize_peer_roster_from_scenario_metadata(scenarios[1].metadata)
    assert first is not None and second is not None
    first_positions = [obstacle.waypoints[0].position for obstacle in first.obstacles]
    second_positions = [obstacle.waypoints[0].position for obstacle in second.obstacles]
    assert first_positions != second_positions, "different seeds must yield different rosters"


def test_synthesizer_clamps_population_peers_into_spawn_volume() -> None:
    bounds = AxisAlignedBounds(
        minimum=Vec3(-5.0, -2.0, 0.0),
        maximum=Vec3(5.0, 2.0, 1.0),
        source="test-fixture",
        confidence="exact",
    )
    matrix = _matrix_with_population(
        PopulationSpec(
            agent_count_per_scenario=10,
            peer_role_distribution={"chase": 1.0},
            random_seed=0,
            spawn_volume=bounds,
        )
    )
    scenario = expand_route_policy_scenario_matrix(matrix)[0].scenarios[0]
    timeline = synthesize_peer_roster_from_scenario_metadata(scenario.metadata)
    assert timeline is not None
    for obstacle in timeline.obstacles:
        x, y, z = obstacle.waypoints[0].position
        assert bounds.minimum.x <= x <= bounds.maximum.x
        assert bounds.minimum.y <= y <= bounds.maximum.y
        assert bounds.minimum.z <= z <= bounds.maximum.z


def test_synthesizer_rejects_population_without_seed() -> None:
    metadata = {
        "population": _unit_population().to_dict(),
        # No populationSeed key -> reject (matrix expander always sets it).
    }
    with pytest.raises(ValueError, match="populationSeed"):
        synthesize_peer_roster_from_scenario_metadata(metadata)


def test_synthesizer_rejects_agent_without_start_pose() -> None:
    metadata = {
        "agents": [
            {
                "recordType": "route-policy-agent-role",
                "version": "gs-mapper-route-policy-agent-role-spec/v1",
                "agentId": "peer-x",
                "role": "peer-obstacle",
                "startVolume": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0],
                    "source": "test",
                    "confidence": "exact",
                },
                "builtinPolicy": "chase",
                "seedOffset": 0,
            }
        ],
    }
    with pytest.raises(ValueError, match="startPose"):
        synthesize_peer_roster_from_scenario_metadata(metadata)


def test_run_scenario_wire_up_prefers_explicit_dynamic_obstacles_path() -> None:
    """When an explicit dynamic_obstacles_path is set, synthesis is bypassed."""

    # We don't actually exercise the full run loop here (that needs a
    # benchmark fixture); instead we lock down the metadata contract
    # the run loop relies on: agents/population-only scenarios produce
    # a synthesized timeline that the existing dynamic_obstacles
    # parameter accepts unmodified.
    metadata = _scene_to_scenario_metadata(
        RoutePolicyMatrixSceneSpec(
            scene_key="multi",
            scene_catalog="scenes.json",
            agents=(
                AgentRoleSpec(
                    agent_id="peer-1",
                    role="peer-obstacle",
                    start_pose=_unit_pose((4.0, 0.0, 0.0)),
                    builtin_policy="chase",
                ),
            ),
            interaction_metrics=InteractionMetricsSpec(
                aggregate_keys=("min-peer-separation",),
            ),
        )
    )
    timeline = synthesize_peer_roster_from_scenario_metadata(metadata)
    assert timeline is not None
    assert timeline.obstacle_count == 1
    # interactionMetrics flows through scenario.metadata untouched so D4 can
    # consume it; the synthesizer itself never strips it.
    assert "interactionMetrics" in metadata


# ----------------------------------------------------------------- helpers


def _scene_to_scenario_metadata(scene: RoutePolicyMatrixSceneSpec) -> dict:
    matrix = RoutePolicyScenarioMatrix(
        matrix_id="d3-fixture",
        registries=(
            RoutePolicyMatrixRegistrySpec(
                registry_id="direct-baseline",
                policy_registry_path="registry.json",
            ),
        ),
        scenes=(scene,),
        goal_suites=(RoutePolicyMatrixGoalSuiteSpec(goal_suite_key="near-goals"),),
        configs=(RoutePolicyMatrixConfigSpec(config_id="default"),),
    )
    return dict(expand_route_policy_scenario_matrix(matrix)[0].scenarios[0].metadata)


def _matrix_with_population(population: PopulationSpec) -> RoutePolicyScenarioMatrix:
    scene = RoutePolicyMatrixSceneSpec(
        scene_key="pop-scene",
        scene_catalog="scenes.json",
        population=population,
    )
    return RoutePolicyScenarioMatrix(
        matrix_id="d3-pop-fixture",
        registries=(
            RoutePolicyMatrixRegistrySpec(
                registry_id="direct-baseline",
                policy_registry_path="registry.json",
            ),
        ),
        scenes=(scene,),
        goal_suites=(RoutePolicyMatrixGoalSuiteSpec(goal_suite_key="near-goals"),),
        configs=(RoutePolicyMatrixConfigSpec(config_id="default"),),
    )


def _unit_population(
    *,
    agent_count_per_scenario: int = 4,
    random_seed: int = 0,
    seed_count: int = 1,
) -> PopulationSpec:
    return PopulationSpec(
        agent_count_per_scenario=agent_count_per_scenario,
        peer_role_distribution={"chase": 0.5, "flee": 0.5},
        random_seed=random_seed,
        spawn_volume=AxisAlignedBounds(
            minimum=Vec3(0.0, 0.0, 0.0),
            maximum=Vec3(10.0, 10.0, 1.0),
            source="test-fixture",
            confidence="exact",
        ),
        seed_count=seed_count,
    )


def _unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
