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
from typing import Any

from .contract import AxisAlignedBounds, Vec3
from .interfaces import Pose3D


AGENT_ROLE_SPEC_VERSION = "gs-mapper-route-policy-agent-role-spec/v1"
POPULATION_SPEC_VERSION = "gs-mapper-route-policy-population-spec/v1"
INTERACTION_METRICS_SPEC_VERSION = "gs-mapper-route-policy-interaction-metrics-spec/v1"

AGENT_ROLES: frozenset[str] = frozenset({"ego", "peer-obstacle", "peer-coop"})
BUILTIN_POLICIES: frozenset[str] = frozenset(
    {"waypoint", "chase", "flee", "maintain_separation"}
)


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
            raise ValueError(
                f"role {self.role!r} must be one of {sorted(AGENT_ROLES)!r}"
            )
        if (self.start_pose is None) == (self.start_volume is None):
            raise ValueError(
                "exactly one of start_pose / start_volume must be set"
            )
        if self.policy_ref is not None and self.builtin_policy is not None:
            raise ValueError(
                "policy_ref and builtin_policy are mutually exclusive"
            )
        if self.builtin_policy is not None and self.builtin_policy not in BUILTIN_POLICIES:
            raise ValueError(
                f"builtin_policy {self.builtin_policy!r} must be one of "
                f"{sorted(BUILTIN_POLICIES)!r}"
            )
        if self.role != "ego" and self.policy_ref is None and self.builtin_policy is None:
            raise ValueError(
                f"peer role {self.role!r} requires policy_ref or builtin_policy"
            )
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
                f"peer_role_distribution keys must be a subset of "
                f"{sorted(BUILTIN_POLICIES)!r}; got unknown {unknown!r}"
            )
        weights = list(self.peer_role_distribution.values())
        for key, weight in self.peer_role_distribution.items():
            numeric = float(weight)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"peer_role_distribution[{key!r}] must be finite"
                )
            if numeric < 0.0 or numeric > 1.0:
                raise ValueError(
                    f"peer_role_distribution[{key!r}]={numeric} must lie in [0, 1]"
                )
        total = sum(float(weight) for weight in weights)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"peer_role_distribution weights must sum to 1.0 (got {total})"
            )
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
                raise ValueError(
                    "min_separation_meters must be positive and finite"
                )
            object.__setattr__(self, "min_separation_meters", value)
        if self.pairwise_clearance_histogram_bins is not None:
            bins = tuple(float(bin_value) for bin_value in self.pairwise_clearance_histogram_bins)
            if len(bins) < 2:
                raise ValueError(
                    "pairwise_clearance_histogram_bins must contain at least two edges"
                )
            for left, right in zip(bins, bins[1:]):
                if not math.isfinite(left) or not math.isfinite(right):
                    raise ValueError("pairwise_clearance_histogram_bins entries must be finite")
                if right <= left:
                    raise ValueError(
                        "pairwise_clearance_histogram_bins must be strictly increasing"
                    )
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
            payload["pairwiseClearanceHistogramBins"] = list(
                self.pairwise_clearance_histogram_bins
            )
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
        start_volume=None
        if start_volume_payload is None
        else _bounds_from_dict(start_volume_payload),
        goal_pose=None if goal_pose_payload is None else _pose_from_dict(goal_pose_payload),
        policy_ref=None if payload.get("policyRef") is None else str(payload["policyRef"]),
        builtin_policy=None
        if payload.get("builtinPolicy") is None
        else str(payload["builtinPolicy"]),
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
        peer_role_distribution={
            str(key): float(value) for key, value in distribution_payload.items()
        },
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
    if not isinstance(aggregate_keys_payload, Sequence) or isinstance(
        aggregate_keys_payload, (str, bytes, bytearray)
    ):
        raise ValueError("InteractionMetricsSpec aggregateKeys must be a list of strings")
    bins_payload = payload.get("pairwiseClearanceHistogramBins")
    if bins_payload is not None and (
        not isinstance(bins_payload, Sequence)
        or isinstance(bins_payload, (str, bytes, bytearray))
    ):
        raise ValueError(
            "InteractionMetricsSpec pairwiseClearanceHistogramBins must be a list of floats"
        )
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


def _pose_from_dict(payload: Mapping[str, Any]) -> Pose3D:
    position = _float_tuple(payload.get("position"), 3, "position")
    orientation = _float_tuple(payload.get("orientationXyzw"), 4, "orientationXyzw")
    return Pose3D(
        position=(position[0], position[1], position[2]),
        orientation_xyzw=(orientation[0], orientation[1], orientation[2], orientation[3]),
        frame_id=str(payload.get("frameId", "world")),
        timestamp_seconds=None
        if payload.get("timestampSeconds") is None
        else float(payload["timestampSeconds"]),
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
        raise ValueError(
            f"{field_name} must be a list of {expected_size} numbers"
        )
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
    "INTERACTION_METRICS_SPEC_VERSION",
    "InteractionMetricsSpec",
    "POPULATION_SPEC_VERSION",
    "PopulationSpec",
    "agent_role_spec_from_dict",
    "interaction_metrics_spec_from_dict",
    "population_spec_from_dict",
]
