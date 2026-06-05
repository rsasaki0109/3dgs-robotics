"""Tests for the matrix loader / expander multi-agent extensions (PR D2).

PR D added the AgentRoleSpec / PopulationSpec / InteractionMetricsSpec
records. PR D2 wires them into ``RoutePolicyMatrixSceneSpec`` so a
scene-axis value can carry optional multi-agent dimensions, and extends
``expand_route_policy_scenario_matrix`` to fan a scene out across the
seed sequence implied by its ``population.seed_count``. These tests
verify the new round-trip, the agents/population mutual-exclusion, the
seed fan-out, and that legacy ego-only matrices keep expanding to the
exact same scenarios as before PR D2.
"""

from __future__ import annotations

import pytest

from gs_sim2real.sim import (
    AgentRoleSpec,
    AxisAlignedBounds,
    InteractionMetricsSpec,
    Pose3D,
    PopulationSpec,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyScenarioMatrix,
    Vec3,
    expand_route_policy_scenario_matrix,
    route_policy_matrix_scene_spec_from_dict,
    route_policy_scenario_matrix_from_dict,
)


# -------------------------------------------------- RoutePolicyMatrixSceneSpec


def test_scene_spec_round_trip_with_agents_and_interaction_metrics() -> None:
    scene = RoutePolicyMatrixSceneSpec(
        scene_key="2-agent-crossing",
        scene_catalog="scenes.json",
        agents=(
            AgentRoleSpec(
                agent_id="ego",
                role="ego",
                start_pose=_unit_pose((0.0, 0.0, 0.0)),
            ),
            AgentRoleSpec(
                agent_id="peer-1",
                role="peer-obstacle",
                start_pose=_unit_pose((5.0, 0.0, 0.0)),
                builtin_policy="chase",
            ),
        ),
        interaction_metrics=InteractionMetricsSpec(
            aggregate_keys=("min-peer-separation",),
            min_separation_meters=0.5,
        ),
    )
    restored = route_policy_matrix_scene_spec_from_dict(scene.to_dict())
    assert restored == scene


def test_scene_spec_round_trip_with_population_only() -> None:
    scene = RoutePolicyMatrixSceneSpec(
        scene_key="dense-population",
        scene_catalog="scenes.json",
        population=_unit_population(seed_count=3),
    )
    restored = route_policy_matrix_scene_spec_from_dict(scene.to_dict())
    assert restored == scene
    assert restored.is_multi_agent is True


def test_scene_spec_rejects_agents_and_population_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoutePolicyMatrixSceneSpec(
            scene_key="conflict",
            scene_catalog="scenes.json",
            agents=(
                AgentRoleSpec(
                    agent_id="peer-1",
                    role="peer-obstacle",
                    start_pose=_unit_pose((0.0, 0.0, 0.0)),
                    builtin_policy="chase",
                ),
            ),
            population=_unit_population(),
        )


def test_scene_spec_rejects_duplicate_agent_ids() -> None:
    with pytest.raises(ValueError, match="unique agent_id"):
        RoutePolicyMatrixSceneSpec(
            scene_key="dupes",
            scene_catalog="scenes.json",
            agents=(
                AgentRoleSpec(
                    agent_id="peer-1",
                    role="peer-obstacle",
                    start_pose=_unit_pose((0.0, 0.0, 0.0)),
                    builtin_policy="chase",
                ),
                AgentRoleSpec(
                    agent_id="peer-1",
                    role="peer-obstacle",
                    start_pose=_unit_pose((1.0, 0.0, 0.0)),
                    builtin_policy="flee",
                ),
            ),
        )


def test_scene_spec_rejects_multiple_ego_agents() -> None:
    with pytest.raises(ValueError, match="at most one ego"):
        RoutePolicyMatrixSceneSpec(
            scene_key="two-egos",
            scene_catalog="scenes.json",
            agents=(
                AgentRoleSpec(
                    agent_id="ego-a",
                    role="ego",
                    start_pose=_unit_pose((0.0, 0.0, 0.0)),
                ),
                AgentRoleSpec(
                    agent_id="ego-b",
                    role="ego",
                    start_pose=_unit_pose((1.0, 0.0, 0.0)),
                ),
            ),
        )


def test_is_multi_agent_reflects_peer_presence() -> None:
    ego_only = RoutePolicyMatrixSceneSpec(
        scene_key="ego",
        scene_catalog="scenes.json",
        agents=(
            AgentRoleSpec(
                agent_id="ego",
                role="ego",
                start_pose=_unit_pose((0.0, 0.0, 0.0)),
            ),
        ),
    )
    assert ego_only.is_multi_agent is False
    with_peer = RoutePolicyMatrixSceneSpec(
        scene_key="with-peer",
        scene_catalog="scenes.json",
        agents=(
            AgentRoleSpec(
                agent_id="peer-1",
                role="peer-obstacle",
                start_pose=_unit_pose((0.0, 0.0, 0.0)),
                builtin_policy="chase",
            ),
        ),
    )
    assert with_peer.is_multi_agent is True


# ------------------------------------------------------------- Expansion


def test_expand_fans_population_across_seeds() -> None:
    matrix = _build_matrix(
        scene=RoutePolicyMatrixSceneSpec(
            scene_key="dense",
            scene_catalog="scenes.json",
            population=_unit_population(random_seed=10, seed_count=3),
            interaction_metrics=InteractionMetricsSpec(
                aggregate_keys=("min-peer-separation",),
            ),
        ),
    )
    sets = expand_route_policy_scenario_matrix(matrix)
    assert len(sets) == 1
    scenarios = sets[0].scenarios
    # 3 seeds × 1 goal_suite × 1 config = 3 scenarios.
    assert len(scenarios) == 3
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    assert scenario_ids == [
        "dense-near-goals-default-seed-10",
        "dense-near-goals-default-seed-11",
        "dense-near-goals-default-seed-12",
    ]
    for scenario, expected_seed in zip(scenarios, (10, 11, 12), strict=True):
        assert scenario.metadata["populationSeed"] == expected_seed
        assert scenario.metadata["population"]["agentCountPerScenario"] == 4
        # interactionMetrics ride along on the expanded scenario.
        assert scenario.metadata["interactionMetrics"]["aggregateKeys"] == ["min-peer-separation"]


def test_expand_keeps_legacy_scene_seedless_when_population_absent() -> None:
    matrix = _build_matrix(
        scene=RoutePolicyMatrixSceneSpec(
            scene_key="ego-only",
            scene_catalog="scenes.json",
        ),
    )
    sets = expand_route_policy_scenario_matrix(matrix)
    scenarios = sets[0].scenarios
    assert len(scenarios) == 1
    # Pre-PR-D2 scenario_id format: no -seed suffix.
    assert scenarios[0].scenario_id == "ego-only-near-goals-default"
    metadata = scenarios[0].metadata
    assert "agents" not in metadata
    assert "population" not in metadata
    assert "interactionMetrics" not in metadata
    assert "populationSeed" not in metadata


def test_scenario_count_per_set_accounts_for_seed_fanout() -> None:
    matrix = _build_matrix(
        scene=RoutePolicyMatrixSceneSpec(
            scene_key="dense",
            scene_catalog="scenes.json",
            population=_unit_population(seed_count=4),
        ),
    )
    # 1 scene × 4 seeds × 1 goal_suite × 1 config = 4 scenarios per set.
    assert matrix.scenario_count_per_set == 4
    sets = expand_route_policy_scenario_matrix(matrix)
    assert len(sets[0].scenarios) == 4


def test_mixed_legacy_and_multi_agent_scenes_in_one_matrix() -> None:
    legacy_scene = RoutePolicyMatrixSceneSpec(
        scene_key="legacy",
        scene_catalog="scenes.json",
    )
    multi_agent_scene = RoutePolicyMatrixSceneSpec(
        scene_key="multi",
        scene_catalog="scenes.json",
        population=_unit_population(random_seed=0, seed_count=2),
    )
    matrix = RoutePolicyScenarioMatrix(
        matrix_id="mixed",
        registries=(_unit_registry(),),
        scenes=(legacy_scene, multi_agent_scene),
        goal_suites=(_unit_goal_suite(),),
        configs=(_unit_config(),),
    )
    # legacy: 1 scenario; multi-agent: 2 scenarios -> 3 total per set.
    assert matrix.scenario_count_per_set == 3
    sets = expand_route_policy_scenario_matrix(matrix)
    scenarios = sets[0].scenarios
    assert [scenario.scenario_id for scenario in scenarios] == [
        "legacy-near-goals-default",
        "multi-near-goals-default-seed-0",
        "multi-near-goals-default-seed-1",
    ]


def test_matrix_round_trip_preserves_scene_multi_agent_fields() -> None:
    matrix = _build_matrix(
        scene=RoutePolicyMatrixSceneSpec(
            scene_key="dense",
            scene_catalog="scenes.json",
            population=_unit_population(seed_count=2),
            interaction_metrics=InteractionMetricsSpec(
                aggregate_keys=("min-peer-separation",),
            ),
        ),
    )
    restored = route_policy_scenario_matrix_from_dict(matrix.to_dict())
    assert restored == matrix


# ----------------------------------------------------------------- helpers


def _build_matrix(scene: RoutePolicyMatrixSceneSpec) -> RoutePolicyScenarioMatrix:
    return RoutePolicyScenarioMatrix(
        matrix_id="pr-d2-fixture",
        registries=(_unit_registry(),),
        scenes=(scene,),
        goal_suites=(_unit_goal_suite(),),
        configs=(_unit_config(),),
    )


def _unit_registry() -> RoutePolicyMatrixRegistrySpec:
    return RoutePolicyMatrixRegistrySpec(
        registry_id="direct-baseline",
        policy_registry_path="registry.json",
    )


def _unit_goal_suite() -> RoutePolicyMatrixGoalSuiteSpec:
    return RoutePolicyMatrixGoalSuiteSpec(goal_suite_key="near-goals")


def _unit_config() -> RoutePolicyMatrixConfigSpec:
    return RoutePolicyMatrixConfigSpec(config_id="default")


def _unit_population(*, random_seed: int = 0, seed_count: int = 1) -> PopulationSpec:
    return PopulationSpec(
        agent_count_per_scenario=4,
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
