"""Multi-agent Tier 3 scenario records (Sprint 4 / PR D).

This module owns the three optional contract additions described in
`docs/plan_outdoor_gs.md` §17.5.1:

- :class:`AgentRoleSpec` — one explicit agent declaration (ego or peer).
- :class:`PopulationSpec` — distribution-driven peer roster generation.
- :class:`InteractionMetricsSpec` — multi-agent metric collection
  declaration that the rollout / shard-merge layers consume.

PR D ships *only* the records, their JSON round-trip helpers, and the
validation rules. Integration points (matrix expansion, scenario run
loop, shard merge, review bundle) come in follow-up PRs D2 → D6. The
records are designed to be optional so that legacy ego-only scenario
matrices keep working without touching their JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from statistics import fmean, quantiles
from typing import Any

import random

from .contract import AxisAlignedBounds, Vec3
from .interfaces import Pose3D
from .policy_dynamic_obstacles import (
    DynamicObstacle,
    DynamicObstacleTimeline,
    DynamicObstacleWaypoint,
)
from .policy_obstacle import (
    ChaseAgentObstaclePolicy,
    FleeAgentObstaclePolicy,
    MaintainSeparationObstaclePolicy,
    WaypointInterpolationObstaclePolicy,
)


AGENT_ROLE_SPEC_VERSION = "gs-mapper-route-policy-agent-role-spec/v1"
POPULATION_SPEC_VERSION = "gs-mapper-route-policy-population-spec/v1"
INTERACTION_METRICS_SPEC_VERSION = "gs-mapper-route-policy-interaction-metrics-spec/v1"
INTERACTION_METRICS_AGGREGATE_VERSION = "gs-mapper-route-policy-interaction-metrics-aggregate/v1"

SCENARIO_INTERACTION_METRIC_VALUES_KEY = "interactionMetricsValues"

AGENT_ROLES: frozenset[str] = frozenset({"ego", "peer-obstacle", "peer-coop"})
BUILTIN_POLICIES: frozenset[str] = frozenset({"waypoint", "chase", "flee", "maintain_separation"})


@dataclass(frozen=True, slots=True)
class AgentRoleSpec:
    """One explicit agent (ego or peer) declared at the scenario level.

    Exactly one of ``start_pose`` / ``start_volume`` must be set. For
    peer roles (``peer-obstacle`` / ``peer-coop``), exactly one of
    ``policy_ref`` / ``builtin_policy`` must be set; the ego role
    defers to the matrix-level policy registry so neither is required.
    """

    agent_id: str
    role: str
    start_pose: Pose3D | None = None
    start_volume: AxisAlignedBounds | None = None
    goal_pose: Pose3D | None = None
    policy_ref: str | None = None
    builtin_policy: str | None = None
    seed_offset: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.agent_id):
            raise ValueError("agent_id must not be empty")
        if self.role not in AGENT_ROLES:
            raise ValueError(f"role {self.role!r} must be one of {sorted(AGENT_ROLES)!r}")
        if (self.start_pose is None) == (self.start_volume is None):
            raise ValueError("exactly one of start_pose / start_volume must be set")
        if self.policy_ref is not None and self.builtin_policy is not None:
            raise ValueError("policy_ref and builtin_policy are mutually exclusive")
        if self.builtin_policy is not None and self.builtin_policy not in BUILTIN_POLICIES:
            raise ValueError(f"builtin_policy {self.builtin_policy!r} must be one of {sorted(BUILTIN_POLICIES)!r}")
        if self.role != "ego" and self.policy_ref is None and self.builtin_policy is None:
            raise ValueError(f"peer role {self.role!r} requires policy_ref or builtin_policy")
        if int(self.seed_offset) < 0:
            raise ValueError("seed_offset must be non-negative")
        object.__setattr__(self, "seed_offset", int(self.seed_offset))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-agent-role",
            "version": AGENT_ROLE_SPEC_VERSION,
            "agentId": self.agent_id,
            "role": self.role,
            "seedOffset": int(self.seed_offset),
        }
        if self.start_pose is not None:
            payload["startPose"] = self.start_pose.to_dict()
        if self.start_volume is not None:
            payload["startVolume"] = self.start_volume.to_dict()
        if self.goal_pose is not None:
            payload["goalPose"] = self.goal_pose.to_dict()
        if self.policy_ref is not None:
            payload["policyRef"] = self.policy_ref
        if self.builtin_policy is not None:
            payload["builtinPolicy"] = self.builtin_policy
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class PopulationSpec:
    """Distribution-driven peer roster generator.

    Used when peers should be sampled from a categorical distribution
    over :data:`BUILTIN_POLICIES` instead of declared one-by-one via
    :class:`AgentRoleSpec`. ``peer_role_distribution`` keys must be a
    non-empty subset of :data:`BUILTIN_POLICIES`; values are mixture
    weights that must each lie in ``[0, 1]`` and sum to ``1.0``
    (within a small tolerance).

    ``seed_count`` controls how many distinct seeds the matrix expander
    fans this population out into. Each seed produces an independent
    scenario instance with the seed sequence
    ``[random_seed, random_seed + 1, ..., random_seed + seed_count - 1]``.
    The default ``seed_count = 1`` keeps single-seed (deterministic)
    populations as a no-op for the expander.
    """

    agent_count_per_scenario: int
    peer_role_distribution: Mapping[str, float]
    random_seed: int
    spawn_volume: AxisAlignedBounds
    homogeneous: bool = False
    seed_count: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.agent_count_per_scenario) < 1:
            raise ValueError("agent_count_per_scenario must be >= 1")
        if int(self.random_seed) < 0:
            raise ValueError("random_seed must be non-negative")
        if int(self.seed_count) < 1:
            raise ValueError("seed_count must be >= 1")
        if not self.peer_role_distribution:
            raise ValueError("peer_role_distribution must not be empty")
        unknown = sorted(set(self.peer_role_distribution) - BUILTIN_POLICIES)
        if unknown:
            raise ValueError(
                f"peer_role_distribution keys must be a subset of {sorted(BUILTIN_POLICIES)!r}; got unknown {unknown!r}"
            )
        weights = list(self.peer_role_distribution.values())
        for key, weight in self.peer_role_distribution.items():
            numeric = float(weight)
            if not math.isfinite(numeric):
                raise ValueError(f"peer_role_distribution[{key!r}] must be finite")
            if numeric < 0.0 or numeric > 1.0:
                raise ValueError(f"peer_role_distribution[{key!r}]={numeric} must lie in [0, 1]")
        total = sum(float(weight) for weight in weights)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"peer_role_distribution weights must sum to 1.0 (got {total})")
        normalised = {str(key): float(value) for key, value in sorted(self.peer_role_distribution.items())}
        object.__setattr__(self, "agent_count_per_scenario", int(self.agent_count_per_scenario))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "homogeneous", bool(self.homogeneous))
        object.__setattr__(self, "seed_count", int(self.seed_count))
        object.__setattr__(self, "peer_role_distribution", normalised)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    def seeds(self) -> tuple[int, ...]:
        """Return the per-scenario seed sequence implied by this spec."""

        return tuple(int(self.random_seed) + offset for offset in range(int(self.seed_count)))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-population",
            "version": POPULATION_SPEC_VERSION,
            "agentCountPerScenario": int(self.agent_count_per_scenario),
            "peerRoleDistribution": dict(self.peer_role_distribution),
            "randomSeed": int(self.random_seed),
            "spawnVolume": self.spawn_volume.to_dict(),
            "homogeneous": bool(self.homogeneous),
            "seedCount": int(self.seed_count),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class InteractionMetricsSpec:
    """Multi-agent metric collection declaration.

    ``aggregate_keys`` names the per-rollout metrics the runtime is
    expected to emit; the shard-merge layer then aggregates them across
    scenarios (mean / p95 / max / histogram, decided per key).
    ``pairwise_clearance_histogram_bins`` is the explicit bin schedule
    for any pairwise clearance histogram and must be strictly
    increasing when set.
    """

    aggregate_keys: tuple[str, ...]
    min_separation_meters: float | None = None
    pairwise_clearance_histogram_bins: tuple[float, ...] | None = None
    require_ego_survives: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = tuple(str(key) for key in self.aggregate_keys)
        if not keys:
            raise ValueError("aggregate_keys must not be empty")
        if any(not key for key in keys):
            raise ValueError("aggregate_keys entries must not be empty")
        if len(set(keys)) != len(keys):
            raise ValueError("aggregate_keys entries must be unique")
        object.__setattr__(self, "aggregate_keys", keys)
        if self.min_separation_meters is not None:
            value = float(self.min_separation_meters)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("min_separation_meters must be positive and finite")
            object.__setattr__(self, "min_separation_meters", value)
        if self.pairwise_clearance_histogram_bins is not None:
            bins = tuple(float(bin_value) for bin_value in self.pairwise_clearance_histogram_bins)
            if len(bins) < 2:
                raise ValueError("pairwise_clearance_histogram_bins must contain at least two edges")
            for left, right in zip(bins, bins[1:]):
                if not math.isfinite(left) or not math.isfinite(right):
                    raise ValueError("pairwise_clearance_histogram_bins entries must be finite")
                if right <= left:
                    raise ValueError("pairwise_clearance_histogram_bins must be strictly increasing")
            object.__setattr__(self, "pairwise_clearance_histogram_bins", bins)
        object.__setattr__(self, "require_ego_survives", bool(self.require_ego_survives))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-interaction-metrics",
            "version": INTERACTION_METRICS_SPEC_VERSION,
            "aggregateKeys": list(self.aggregate_keys),
            "requireEgoSurvives": bool(self.require_ego_survives),
        }
        if self.min_separation_meters is not None:
            payload["minSeparationMeters"] = float(self.min_separation_meters)
        if self.pairwise_clearance_histogram_bins is not None:
            payload["pairwiseClearanceHistogramBins"] = list(self.pairwise_clearance_histogram_bins)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def agent_role_spec_from_dict(payload: Mapping[str, Any]) -> AgentRoleSpec:
    """Rebuild :class:`AgentRoleSpec` from a JSON payload."""

    _check_version(payload, AGENT_ROLE_SPEC_VERSION)
    start_pose_payload = payload.get("startPose")
    start_volume_payload = payload.get("startVolume")
    goal_pose_payload = payload.get("goalPose")
    metadata_payload = payload.get("metadata") or {}
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("AgentRoleSpec metadata must be a mapping")
    return AgentRoleSpec(
        agent_id=str(payload["agentId"]),
        role=str(payload["role"]),
        start_pose=None if start_pose_payload is None else _pose_from_dict(start_pose_payload),
        start_volume=None if start_volume_payload is None else _bounds_from_dict(start_volume_payload),
        goal_pose=None if goal_pose_payload is None else _pose_from_dict(goal_pose_payload),
        policy_ref=None if payload.get("policyRef") is None else str(payload["policyRef"]),
        builtin_policy=None if payload.get("builtinPolicy") is None else str(payload["builtinPolicy"]),
        seed_offset=int(payload.get("seedOffset", 0)),
        metadata=dict(metadata_payload),
    )


def population_spec_from_dict(payload: Mapping[str, Any]) -> PopulationSpec:
    """Rebuild :class:`PopulationSpec` from a JSON payload."""

    _check_version(payload, POPULATION_SPEC_VERSION)
    distribution_payload = payload.get("peerRoleDistribution") or {}
    if not isinstance(distribution_payload, Mapping):
        raise ValueError("PopulationSpec peerRoleDistribution must be a mapping")
    spawn_volume_payload = payload.get("spawnVolume")
    if not isinstance(spawn_volume_payload, Mapping):
        raise ValueError("PopulationSpec spawnVolume must be a mapping")
    metadata_payload = payload.get("metadata") or {}
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("PopulationSpec metadata must be a mapping")
    return PopulationSpec(
        agent_count_per_scenario=int(payload["agentCountPerScenario"]),
        peer_role_distribution={str(key): float(value) for key, value in distribution_payload.items()},
        random_seed=int(payload["randomSeed"]),
        spawn_volume=_bounds_from_dict(spawn_volume_payload),
        homogeneous=bool(payload.get("homogeneous", False)),
        seed_count=int(payload.get("seedCount", 1)),
        metadata=dict(metadata_payload),
    )


def interaction_metrics_spec_from_dict(
    payload: Mapping[str, Any],
) -> InteractionMetricsSpec:
    """Rebuild :class:`InteractionMetricsSpec` from a JSON payload."""

    _check_version(payload, INTERACTION_METRICS_SPEC_VERSION)
    aggregate_keys_payload = payload.get("aggregateKeys")
    if not isinstance(aggregate_keys_payload, Sequence) or isinstance(aggregate_keys_payload, (str, bytes, bytearray)):
        raise ValueError("InteractionMetricsSpec aggregateKeys must be a list of strings")
    bins_payload = payload.get("pairwiseClearanceHistogramBins")
    if bins_payload is not None and (
        not isinstance(bins_payload, Sequence) or isinstance(bins_payload, (str, bytes, bytearray))
    ):
        raise ValueError("InteractionMetricsSpec pairwiseClearanceHistogramBins must be a list of floats")
    metadata_payload = payload.get("metadata") or {}
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("InteractionMetricsSpec metadata must be a mapping")
    min_sep = payload.get("minSeparationMeters")
    return InteractionMetricsSpec(
        aggregate_keys=tuple(str(key) for key in aggregate_keys_payload),
        min_separation_meters=None if min_sep is None else float(min_sep),
        pairwise_clearance_histogram_bins=None
        if bins_payload is None
        else tuple(float(bin_value) for bin_value in bins_payload),
        require_ego_survives=bool(payload.get("requireEgoSurvives", True)),
        metadata=dict(metadata_payload),
    )


DEFAULT_PEER_RADIUS_METERS = 0.5
DEFAULT_PEER_SPEED_M_PER_STEP = 0.1


@dataclass(frozen=True, slots=True)
class InteractionMetricKeyStats:
    """Mean / p95 / max for one :class:`InteractionMetricsSpec` aggregate key."""

    mean: float
    p95: float
    maximum: float
    sample_count: int

    def __post_init__(self) -> None:
        for label, value in (("mean", self.mean), ("p95", self.p95), ("maximum", self.maximum)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{label} must be finite")
            object.__setattr__(self, label if label != "maximum" else "maximum", numeric)
        if int(self.sample_count) < 1:
            raise ValueError("sample_count must be >= 1")
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "p95": float(self.p95),
            "max": float(self.maximum),
            "sampleCount": int(self.sample_count),
        }


@dataclass(frozen=True, slots=True)
class InteractionMetricsAggregate:
    """Per-key roll-up of :class:`InteractionMetricsSpec` values across scenarios.

    Produced by :func:`aggregate_interaction_metrics_across_scenarios`
    from the ``interactionMetricsValues`` mapping written into each
    scenario result's metadata by the run loop. ``sample_scenario_count``
    counts how many scenarios contributed at least one numeric value.
    """

    per_key_stats: Mapping[str, InteractionMetricKeyStats]
    sample_scenario_count: int

    def __post_init__(self) -> None:
        if not self.per_key_stats:
            raise ValueError("per_key_stats must not be empty")
        if int(self.sample_scenario_count) < 1:
            raise ValueError("sample_scenario_count must be >= 1")
        # Normalise key ordering so the JSON output is stable.
        ordered = {str(key): self.per_key_stats[key] for key in sorted(self.per_key_stats)}
        object.__setattr__(self, "per_key_stats", ordered)
        object.__setattr__(self, "sample_scenario_count", int(self.sample_scenario_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-interaction-metrics-aggregate",
            "version": INTERACTION_METRICS_AGGREGATE_VERSION,
            "perKeyStats": {key: stats.to_dict() for key, stats in self.per_key_stats.items()},
            "sampleScenarioCount": int(self.sample_scenario_count),
        }


def interaction_metrics_aggregate_from_dict(
    payload: Mapping[str, Any],
) -> InteractionMetricsAggregate:
    """Rebuild :class:`InteractionMetricsAggregate` from a JSON payload."""

    _check_version(payload, INTERACTION_METRICS_AGGREGATE_VERSION)
    stats_payload = payload.get("perKeyStats") or {}
    if not isinstance(stats_payload, Mapping):
        raise ValueError("InteractionMetricsAggregate perKeyStats must be a mapping")
    per_key_stats: dict[str, InteractionMetricKeyStats] = {}
    for raw_key, stats in stats_payload.items():
        if not isinstance(stats, Mapping):
            raise ValueError("InteractionMetricsAggregate perKeyStats entries must be mappings")
        per_key_stats[str(raw_key)] = InteractionMetricKeyStats(
            mean=float(stats["mean"]),
            p95=float(stats["p95"]),
            maximum=float(stats["max"]),
            sample_count=int(stats["sampleCount"]),
        )
    return InteractionMetricsAggregate(
        per_key_stats=per_key_stats,
        sample_scenario_count=int(payload["sampleScenarioCount"]),
    )


def aggregate_interaction_metrics_across_scenarios(
    scenario_metadata_iter: "Sequence[Mapping[str, Any]]",
) -> InteractionMetricsAggregate | None:
    """Aggregate ``interactionMetricsValues`` across scenario results.

    The run loop is expected (in a follow-up PR) to write a
    ``{"interactionMetricsValues": {key: float, ...}}`` mapping into each
    multi-agent scenario result's metadata. This aggregator scans those
    mappings across every scenario in every shard and produces
    per-key mean / p95 / max / sampleCount statistics. Returns ``None``
    when no scenario carries values so legacy ego-only runs stay
    aggregate-free.
    """

    per_key: dict[str, list[float]] = {}
    scenario_count_with_values = 0
    for scenario_metadata in scenario_metadata_iter:
        values = scenario_metadata.get(SCENARIO_INTERACTION_METRIC_VALUES_KEY)
        if not isinstance(values, Mapping) or not values:
            continue
        contributed = False
        for raw_key, raw_value in values.items():
            numeric = _coerce_finite_float(raw_value)
            if numeric is None:
                continue
            per_key.setdefault(str(raw_key), []).append(numeric)
            contributed = True
        if contributed:
            scenario_count_with_values += 1
    if not per_key:
        return None
    per_key_stats: dict[str, InteractionMetricKeyStats] = {}
    for key, samples in per_key.items():
        per_key_stats[key] = InteractionMetricKeyStats(
            mean=fmean(samples),
            p95=_p95(samples),
            maximum=max(samples),
            sample_count=len(samples),
        )
    return InteractionMetricsAggregate(
        per_key_stats=per_key_stats,
        sample_scenario_count=scenario_count_with_values,
    )


def _coerce_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _p95(samples: Sequence[float]) -> float:
    if len(samples) == 1:
        return float(samples[0])
    return float(quantiles(samples, n=100, method="inclusive")[94])


def synthesize_peer_roster_from_scenario_metadata(
    metadata: Mapping[str, Any],
    *,
    timeline_id: str | None = None,
    default_radius_meters: float = DEFAULT_PEER_RADIUS_METERS,
    default_speed_m_per_step: float = DEFAULT_PEER_SPEED_M_PER_STEP,
) -> DynamicObstacleTimeline | None:
    """Build a :class:`DynamicObstacleTimeline` from D2 scenario metadata.

    PR D2 embedded ``agents`` / ``population`` / ``populationSeed`` /
    ``interactionMetrics`` into each expanded scenario's metadata. This
    helper turns those declarations into a peer roster the existing
    headless env can consume via its ``dynamic_obstacles`` plumbing.

    Returns ``None`` when the scenario is legacy ego-only (no multi-agent
    metadata), so the caller can keep its existing JSON-loaded path.
    The synthesised roster is deterministic in ``populationSeed`` —
    re-running the synthesizer with the same metadata produces the same
    peer set.
    """

    agents_payload = metadata.get("agents") or ()
    population_payload = metadata.get("population")
    if not agents_payload and population_payload is None:
        return None

    obstacles: list[DynamicObstacle] = []
    if agents_payload:
        for index, agent_payload in enumerate(agents_payload):
            if not isinstance(agent_payload, Mapping):
                raise ValueError("scenario metadata 'agents' entries must be mappings")
            obstacle = _agent_to_dynamic_obstacle(
                agent_payload,
                fallback_index=index,
                default_radius_meters=default_radius_meters,
                default_speed_m_per_step=default_speed_m_per_step,
            )
            if obstacle is not None:
                obstacles.append(obstacle)
    elif population_payload is not None:
        if not isinstance(population_payload, Mapping):
            raise ValueError("scenario metadata 'population' must be a mapping")
        seed_value = metadata.get("populationSeed")
        if seed_value is None:
            raise ValueError("scenario metadata 'population' requires 'populationSeed' (set by the matrix expander)")
        obstacles.extend(
            _population_to_dynamic_obstacles(
                population_payload,
                seed=int(seed_value),
                default_radius_meters=default_radius_meters,
                default_speed_m_per_step=default_speed_m_per_step,
            )
        )

    if not obstacles:
        return None

    resolved_timeline_id = timeline_id or "multi-agent-synthesized"
    return DynamicObstacleTimeline(
        timeline_id=resolved_timeline_id,
        obstacles=tuple(obstacles),
        metadata={"source": "policy_scenario_multi_agent.synthesize_peer_roster"},
    )


def _agent_to_dynamic_obstacle(
    payload: Mapping[str, Any],
    *,
    fallback_index: int,
    default_radius_meters: float,
    default_speed_m_per_step: float,
) -> DynamicObstacle | None:
    role = str(payload.get("role", ""))
    if role == "ego":
        return None
    agent_id = str(payload.get("agentId") or f"peer-{fallback_index}")
    start_pose_payload = payload.get("startPose")
    if not isinstance(start_pose_payload, Mapping):
        raise ValueError(
            f"agent {agent_id!r} requires startPose for peer-roster synthesis "
            "(start_volume-only peers are deferred to a follow-up PR)"
        )
    start_position = _float_tuple(start_pose_payload.get("position"), 3, "agents.startPose.position")
    start = (start_position[0], start_position[1], start_position[2])
    builtin_policy = payload.get("builtinPolicy")
    policy = _builtin_policy_instance(
        builtin_policy,
        start_position=start,
        default_speed_m_per_step=default_speed_m_per_step,
    )
    return DynamicObstacle(
        obstacle_id=agent_id,
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=start),),
        radius_meters=default_radius_meters,
        policy=policy,
        metadata={"sourceRole": role, "builtinPolicy": str(builtin_policy or "waypoint")},
    )


def _population_to_dynamic_obstacles(
    payload: Mapping[str, Any],
    *,
    seed: int,
    default_radius_meters: float,
    default_speed_m_per_step: float,
) -> list[DynamicObstacle]:
    spec = population_spec_from_dict(payload)
    peer_count = max(0, spec.agent_count_per_scenario - 1)
    if peer_count == 0:
        return []
    rng = random.Random(seed)
    role_keys = tuple(spec.peer_role_distribution.keys())
    role_weights = tuple(spec.peer_role_distribution.values())
    bounds = spec.spawn_volume
    obstacles: list[DynamicObstacle] = []
    chosen_role = rng.choices(role_keys, weights=role_weights, k=1)[0] if spec.homogeneous else None
    for index in range(peer_count):
        role_choice = chosen_role or rng.choices(role_keys, weights=role_weights, k=1)[0]
        start_x = rng.uniform(bounds.minimum.x, bounds.maximum.x)
        start_y = rng.uniform(bounds.minimum.y, bounds.maximum.y)
        start_z = rng.uniform(bounds.minimum.z, bounds.maximum.z)
        start = (float(start_x), float(start_y), float(start_z))
        policy = _builtin_policy_instance(
            role_choice,
            start_position=start,
            default_speed_m_per_step=default_speed_m_per_step,
        )
        obstacles.append(
            DynamicObstacle(
                obstacle_id=f"population-peer-{index}",
                waypoints=(DynamicObstacleWaypoint(step_index=0, position=start),),
                radius_meters=default_radius_meters,
                policy=policy,
                metadata={
                    "sourceRole": "peer-obstacle",
                    "builtinPolicy": role_choice,
                    "populationIndex": index,
                },
            )
        )
    return obstacles


def _builtin_policy_instance(
    builtin_policy: Any,
    *,
    start_position: tuple[float, float, float],
    default_speed_m_per_step: float,
) -> "ChaseAgentObstaclePolicy | FleeAgentObstaclePolicy | MaintainSeparationObstaclePolicy | WaypointInterpolationObstaclePolicy | None":
    if builtin_policy is None:
        return None
    name = str(builtin_policy)
    if name == "waypoint":
        return WaypointInterpolationObstaclePolicy(((0, start_position),))
    if name == "chase":
        return ChaseAgentObstaclePolicy(start_position, default_speed_m_per_step)
    if name == "flee":
        return FleeAgentObstaclePolicy(start_position, default_speed_m_per_step)
    if name == "maintain_separation":
        return MaintainSeparationObstaclePolicy(
            WaypointInterpolationObstaclePolicy(((0, start_position),)),
            min_separation_meters=default_speed_m_per_step,
        )
    raise ValueError(f"unsupported builtin_policy: {name!r}")


def _pose_from_dict(payload: Mapping[str, Any]) -> Pose3D:
    position = _float_tuple(payload.get("position"), 3, "position")
    orientation = _float_tuple(payload.get("orientationXyzw"), 4, "orientationXyzw")
    return Pose3D(
        position=(position[0], position[1], position[2]),
        orientation_xyzw=(orientation[0], orientation[1], orientation[2], orientation[3]),
        frame_id=str(payload.get("frameId", "world")),
        timestamp_seconds=None if payload.get("timestampSeconds") is None else float(payload["timestampSeconds"]),
    )


def _bounds_from_dict(payload: Mapping[str, Any]) -> AxisAlignedBounds:
    minimum = _float_tuple(payload.get("min"), 3, "spawnVolume.min")
    maximum = _float_tuple(payload.get("max"), 3, "spawnVolume.max")
    return AxisAlignedBounds(
        minimum=Vec3(*minimum),
        maximum=Vec3(*maximum),
        source=str(payload.get("source", "unspecified")),
        confidence=str(payload.get("confidence", "unspecified")),
    )


def _float_tuple(value: Any, expected_size: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_size:
        raise ValueError(f"{field_name} must be a list of {expected_size} numbers")
    return tuple(float(component) for component in value)


def _check_version(payload: Mapping[str, Any], expected: str) -> None:
    version = payload.get("version")
    if version is not None and version != expected:
        raise ValueError(f"unsupported version: {version!r} (expected {expected!r})")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON float values must be finite")
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


__all__ = [
    "AGENT_ROLES",
    "AGENT_ROLE_SPEC_VERSION",
    "AgentRoleSpec",
    "BUILTIN_POLICIES",
    "DEFAULT_PEER_RADIUS_METERS",
    "DEFAULT_PEER_SPEED_M_PER_STEP",
    "INTERACTION_METRICS_AGGREGATE_VERSION",
    "INTERACTION_METRICS_SPEC_VERSION",
    "InteractionMetricKeyStats",
    "InteractionMetricsAggregate",
    "InteractionMetricsSpec",
    "POPULATION_SPEC_VERSION",
    "PopulationSpec",
    "SCENARIO_INTERACTION_METRIC_VALUES_KEY",
    "agent_role_spec_from_dict",
    "aggregate_interaction_metrics_across_scenarios",
    "interaction_metrics_aggregate_from_dict",
    "interaction_metrics_spec_from_dict",
    "population_spec_from_dict",
    "synthesize_peer_roster_from_scenario_metadata",
]
