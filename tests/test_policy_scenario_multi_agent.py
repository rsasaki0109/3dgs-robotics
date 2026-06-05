"""Tests for the Sprint 4 / PR D multi-agent scenario records.

PR D ships three optional records (``AgentRoleSpec`` / ``PopulationSpec``
/ ``InteractionMetricsSpec``) that future PRs D2 → D6 will plumb into
the matrix expansion, scenario set run loop, shard merge, and review
bundle. These tests verify:

1. The records validate inputs at construction time (mutually-exclusive
   policy fields, role enum, distribution normalisation, histogram
   bins).
2. ``to_dict`` / ``from_dict`` round-trips preserve every field on
   ``__eq__``.
3. Legacy ego-only ``RoutePolicyScenarioMatrix`` JSON (which has no
   ``agents`` / ``population`` / ``interactionMetrics`` keys) still
   round-trips through ``route_policy_scenario_matrix_from_dict``
   unchanged — i.e. the new records are strictly additive.
"""

from __future__ import annotations

import pytest

from gs_sim2real.sim import (
    AGENT_ROLE_SPEC_VERSION,
    AgentRoleSpec,
    AxisAlignedBounds,
    INTERACTION_METRICS_SPEC_VERSION,
    InteractionMetricsSpec,
    POPULATION_SPEC_VERSION,
    Pose3D,
    PopulationSpec,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyScenarioMatrix,
    Vec3,
    agent_role_spec_from_dict,
    interaction_metrics_spec_from_dict,
    population_spec_from_dict,
    route_policy_scenario_matrix_from_dict,
)


# ---------------------------------------------------------------- AgentRoleSpec


def test_agent_role_spec_round_trip_with_start_pose() -> None:
    spec = AgentRoleSpec(
        agent_id="peer-1",
        role="peer-obstacle",
        start_pose=_unit_pose((1.0, 2.0, 0.0)),
        goal_pose=_unit_pose((5.0, 2.0, 0.0)),
        builtin_policy="chase",
        seed_offset=7,
        metadata={"note": "crossing peer"},
    )
    restored = agent_role_spec_from_dict(spec.to_dict())
    assert restored == spec
    assert spec.to_dict()["version"] == AGENT_ROLE_SPEC_VERSION


def test_agent_role_spec_round_trip_with_start_volume_and_policy_ref() -> None:
    spec = AgentRoleSpec(
        agent_id="ego",
        role="ego",
        start_volume=_unit_bounds(),
        policy_ref="registry://imitation-baseline",
    )
    restored = agent_role_spec_from_dict(spec.to_dict())
    assert restored == spec


def test_agent_role_spec_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        AgentRoleSpec(
            agent_id="x",
            role="hostile",
            start_pose=_unit_pose((0.0, 0.0, 0.0)),
        )


def test_agent_role_spec_requires_exactly_one_start_location() -> None:
    with pytest.raises(ValueError, match="start_pose / start_volume"):
        AgentRoleSpec(agent_id="x", role="ego")
    with pytest.raises(ValueError, match="start_pose / start_volume"):
        AgentRoleSpec(
            agent_id="x",
            role="ego",
            start_pose=_unit_pose((0.0, 0.0, 0.0)),
            start_volume=_unit_bounds(),
        )


def test_agent_role_spec_rejects_both_policy_ref_and_builtin_policy() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AgentRoleSpec(
            agent_id="peer-1",
            role="peer-obstacle",
            start_pose=_unit_pose((0.0, 0.0, 0.0)),
            policy_ref="registry://x",
            builtin_policy="chase",
        )


def test_agent_role_spec_rejects_unknown_builtin_policy() -> None:
    with pytest.raises(ValueError, match="builtin_policy"):
        AgentRoleSpec(
            agent_id="peer-1",
            role="peer-obstacle",
            start_pose=_unit_pose((0.0, 0.0, 0.0)),
            builtin_policy="wander",
        )


def test_agent_role_spec_requires_policy_for_peer_role() -> None:
    with pytest.raises(ValueError, match="peer role"):
        AgentRoleSpec(
            agent_id="peer-1",
            role="peer-coop",
            start_pose=_unit_pose((0.0, 0.0, 0.0)),
        )


def test_agent_role_spec_ego_role_does_not_require_policy() -> None:
    # ego defers to the matrix-level registry, so neither policy field
    # is required for the ego role.
    spec = AgentRoleSpec(
        agent_id="ego",
        role="ego",
        start_pose=_unit_pose((0.0, 0.0, 0.0)),
    )
    assert spec.policy_ref is None and spec.builtin_policy is None


# ---------------------------------------------------------------- PopulationSpec


def test_population_spec_round_trip() -> None:
    spec = PopulationSpec(
        agent_count_per_scenario=8,
        peer_role_distribution={"chase": 0.25, "flee": 0.25, "maintain_separation": 0.5},
        random_seed=42,
        spawn_volume=_unit_bounds(),
        homogeneous=False,
        metadata={"note": "4-agent route-conflict population"},
    )
    restored = population_spec_from_dict(spec.to_dict())
    assert restored == spec
    assert spec.to_dict()["version"] == POPULATION_SPEC_VERSION


def test_population_spec_normalises_distribution_key_order() -> None:
    spec = PopulationSpec(
        agent_count_per_scenario=2,
        peer_role_distribution={"flee": 0.5, "chase": 0.5},
        random_seed=0,
        spawn_volume=_unit_bounds(),
    )
    # __post_init__ sorts keys so JSON output is stable.
    assert list(spec.peer_role_distribution) == ["chase", "flee"]


def test_population_spec_rejects_distribution_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        PopulationSpec(
            agent_count_per_scenario=2,
            peer_role_distribution={"chase": 0.5, "flee": 0.3},
            random_seed=0,
            spawn_volume=_unit_bounds(),
        )


def test_population_spec_rejects_unknown_distribution_key() -> None:
    with pytest.raises(ValueError, match="subset"):
        PopulationSpec(
            agent_count_per_scenario=2,
            peer_role_distribution={"wander": 1.0},
            random_seed=0,
            spawn_volume=_unit_bounds(),
        )


def test_population_spec_rejects_zero_agent_count() -> None:
    with pytest.raises(ValueError, match="agent_count"):
        PopulationSpec(
            agent_count_per_scenario=0,
            peer_role_distribution={"chase": 1.0},
            random_seed=0,
            spawn_volume=_unit_bounds(),
        )


def test_population_spec_rejects_weight_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        PopulationSpec(
            agent_count_per_scenario=2,
            peer_role_distribution={"chase": 1.5, "flee": -0.5},
            random_seed=0,
            spawn_volume=_unit_bounds(),
        )


# ----------------------------------------------------------- InteractionMetricsSpec


def test_interaction_metrics_spec_round_trip() -> None:
    spec = InteractionMetricsSpec(
        aggregate_keys=("min-peer-separation", "max-peer-collision-count"),
        min_separation_meters=0.5,
        pairwise_clearance_histogram_bins=(0.0, 0.25, 0.5, 1.0, 2.0),
        require_ego_survives=False,
        metadata={"note": "dense crossing"},
    )
    restored = interaction_metrics_spec_from_dict(spec.to_dict())
    assert restored == spec
    assert spec.to_dict()["version"] == INTERACTION_METRICS_SPEC_VERSION


def test_interaction_metrics_spec_minimal_round_trip() -> None:
    spec = InteractionMetricsSpec(aggregate_keys=("min-peer-separation",))
    restored = interaction_metrics_spec_from_dict(spec.to_dict())
    assert restored == spec
    # Optional fields stay absent from the payload to keep diffs clean.
    payload = spec.to_dict()
    assert "minSeparationMeters" not in payload
    assert "pairwiseClearanceHistogramBins" not in payload


def test_interaction_metrics_spec_rejects_empty_aggregate_keys() -> None:
    with pytest.raises(ValueError, match="aggregate_keys"):
        InteractionMetricsSpec(aggregate_keys=())


def test_interaction_metrics_spec_rejects_duplicate_aggregate_keys() -> None:
    with pytest.raises(ValueError, match="unique"):
        InteractionMetricsSpec(aggregate_keys=("min-peer-separation", "min-peer-separation"))


def test_interaction_metrics_spec_rejects_non_increasing_histogram_bins() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        InteractionMetricsSpec(
            aggregate_keys=("x",),
            pairwise_clearance_histogram_bins=(0.0, 0.5, 0.5, 1.0),
        )


def test_interaction_metrics_spec_rejects_one_bin_edge() -> None:
    with pytest.raises(ValueError, match="at least two"):
        InteractionMetricsSpec(
            aggregate_keys=("x",),
            pairwise_clearance_histogram_bins=(0.0,),
        )


def test_interaction_metrics_spec_rejects_non_positive_min_separation() -> None:
    with pytest.raises(ValueError, match="positive"):
        InteractionMetricsSpec(aggregate_keys=("x",), min_separation_meters=0.0)


# ----------------------------------------------- Legacy ego-only matrix fallback


def test_legacy_ego_only_matrix_round_trips_without_multi_agent_fields() -> None:
    """RoutePolicyScenarioMatrix JSON without agents / population must still load."""

    matrix = RoutePolicyScenarioMatrix(
        matrix_id="legacy-ego-only",
        registries=(
            RoutePolicyMatrixRegistrySpec(
                registry_id="direct-baseline",
                policy_registry_path="registry.json",
            ),
        ),
        scenes=(
            RoutePolicyMatrixSceneSpec(
                scene_key="outdoor-demo",
                scene_catalog="scenes.json",
            ),
        ),
        goal_suites=(RoutePolicyMatrixGoalSuiteSpec(goal_suite_key="near-goals"),),
        configs=(RoutePolicyMatrixConfigSpec(config_id="default"),),
    )
    payload = matrix.to_dict()
    # PR D does not introduce any new top-level keys; pre-PR-D fixtures
    # would not contain "agents" / "population" / "interactionMetrics" and
    # the existing serialiser must not emit them either.
    assert "agents" not in payload
    assert "population" not in payload
    assert "interactionMetrics" not in payload

    restored = route_policy_scenario_matrix_from_dict(payload)
    assert restored == matrix


# ----------------------------------------------------------------- test helpers


def _unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )


def _unit_bounds() -> AxisAlignedBounds:
    return AxisAlignedBounds(
        minimum=Vec3(0.0, 0.0, 0.0),
        maximum=Vec3(10.0, 10.0, 1.0),
        source="test-fixture",
        confidence="exact",
    )
