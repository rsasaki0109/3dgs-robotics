"""Quality gates and baseline reports for route policy rollout datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from .gym_adapter import RoutePolicyGymAdapter
from .policy_dataset import (
    RoutePolicyCallable,
    RoutePolicyDatasetExport,
    RoutePolicyGoal,
    collect_route_policy_dataset,
)
from .policy_trace import RoutePolicyTraceEmitter

RoutePolicyTraceEmitterFactory = Callable[[str], RoutePolicyTraceEmitter | None]


ROUTE_POLICY_QUALITY_VERSION = "gs-mapper-route-policy-quality/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyQualityThresholds:
    """Pass/fail gates for a rollout dataset before it is used for training."""

    min_success_rate: float = 0.8
    max_collision_rate: float = 0.05
    max_truncation_rate: float = 0.1
    min_mean_reward: float | None = None
    min_scene_count: int = 1
    min_episode_count: int = 1
    min_transition_count: int = 1

    def __post_init__(self) -> None:
        _rate(self.min_success_rate, "min_success_rate")
        _rate(self.max_truncation_rate, "max_truncation_rate")
        _non_negative_float(self.max_collision_rate, "max_collision_rate")
        if self.min_mean_reward is not None:
            _finite_float(self.min_mean_reward, "min_mean_reward")
        _non_negative_int(self.min_scene_count, "min_scene_count")
        _non_negative_int(self.min_episode_count, "min_episode_count")
        _non_negative_int(self.min_transition_count, "min_transition_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minSuccessRate": float(self.min_success_rate),
            "maxCollisionRate": float(self.max_collision_rate),
            "maxTruncationRate": float(self.max_truncation_rate),
            "minMeanReward": None if self.min_mean_reward is None else float(self.min_mean_reward),
            "minSceneCount": int(self.min_scene_count),
            "minEpisodeCount": int(self.min_episode_count),
            "minTransitionCount": int(self.min_transition_count),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyQualityReport:
    """Computed QA report for a route policy dataset."""

    dataset_id: str
    metrics: Mapping[str, float]
    scene_coverage: Mapping[str, int]
    termination_counts: Mapping[str, int]
    reward_distribution: Mapping[str, float]
    thresholds: RoutePolicyQualityThresholds
    failed_checks: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_QUALITY_VERSION

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-quality-report",
            "version": self.version,
            "datasetId": self.dataset_id,
            "passed": self.passed,
            "failedChecks": list(self.failed_checks),
            "thresholds": self.thresholds.to_dict(),
            "metrics": _float_mapping(self.metrics),
            "sceneCoverage": {str(key): int(value) for key, value in sorted(self.scene_coverage.items())},
            "terminationCounts": {str(key): int(value) for key, value in sorted(self.termination_counts.items())},
            "rewardDistribution": _float_mapping(self.reward_distribution),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBaselineResult:
    """One policy's rollout dataset and quality report."""

    policy_name: str
    dataset: RoutePolicyDatasetExport
    quality: RoutePolicyQualityReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.quality.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-baseline-result",
            "policyName": self.policy_name,
            "passed": self.passed,
            "dataset": self.dataset.to_dict(),
            "quality": self.quality.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBaselineEvaluation:
    """Comparison report for multiple policies evaluated on the same rollout task."""

    evaluation_id: str
    results: tuple[RoutePolicyBaselineResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_QUALITY_VERSION

    @property
    def best_policy_name(self) -> str | None:
        if not self.results:
            return None
        return max(self.results, key=lambda result: _quality_rank_key(result.quality)).policy_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-baseline-evaluation",
            "version": self.version,
            "evaluationId": self.evaluation_id,
            "policyCount": len(self.results),
            "bestPolicyName": self.best_policy_name,
            "results": [result.to_dict() for result in self.results],
            "metadata": _json_mapping(self.metadata),
        }


def evaluate_route_policy_dataset_quality(
    dataset: RoutePolicyDatasetExport,
    *,
    thresholds: RoutePolicyQualityThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyQualityReport:
    """Score rollout quality and return threshold failures as stable check IDs."""

    resolved_thresholds = thresholds or RoutePolicyQualityThresholds()
    episode_count = len(dataset.episodes)
    transition_count = dataset.transition_count
    scene_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    episode_rewards: list[float] = []
    transition_rewards: list[float] = []
    success_count = 0
    blocked_count = 0
    truncated_count = 0
    terminated_count = 0
    collision_count = 0.0

    for episode in dataset.episodes:
        scene_counts[str(episode.scene_id)] += 1
        summary = episode.summary()
        episode_rewards.append(float(episode.total_reward))
        success_count += int(bool(summary.get("goalReached", False)))
        blocked_count += int(bool(summary.get("blocked", False)))
        truncated_count += int(episode.truncated)
        terminated_count += int(episode.terminated)
        termination_counts[_termination_bucket(summary, episode.terminated, episode.truncated)] += 1
        for transition in episode.transitions:
            transition_rewards.append(float(transition.reward))
            collision_count += _transition_collision_count(transition.info, transition.next_observation)

    metrics = {
        "episode-count": float(episode_count),
        "transition-count": float(transition_count),
        "scene-count": float(len(scene_counts)),
        "success-rate": _safe_rate(success_count, episode_count),
        "blocked-rate": _safe_rate(blocked_count, episode_count),
        "termination-rate": _safe_rate(terminated_count, episode_count),
        "truncation-rate": _safe_rate(truncated_count, episode_count),
        "collision-count": float(collision_count),
        "collision-rate": _safe_rate(collision_count, transition_count),
        "mean-episode-length": _safe_rate(transition_count, episode_count),
        "mean-reward": _mean(episode_rewards),
        "mean-transition-reward": _mean(transition_rewards),
    }
    reward_distribution = _series_distribution(episode_rewards)
    failed_checks = _failed_quality_checks(
        metrics=metrics,
        thresholds=resolved_thresholds,
    )
    return RoutePolicyQualityReport(
        dataset_id=dataset.dataset_id,
        metrics=metrics,
        scene_coverage=dict(sorted(scene_counts.items())),
        termination_counts=dict(sorted(termination_counts.items())),
        reward_distribution=reward_distribution,
        thresholds=resolved_thresholds,
        failed_checks=failed_checks,
        metadata=_json_mapping(metadata or {}),
    )


def evaluate_route_policy_baselines(
    adapters: Sequence[RoutePolicyGymAdapter],
    policies: Mapping[str, RoutePolicyCallable],
    *,
    episode_count: int,
    evaluation_id: str = "route-policy-baselines",
    seed_start: int = 0,
    goals: Sequence[RoutePolicyGoal] | None = None,
    max_steps: int | None = None,
    thresholds: RoutePolicyQualityThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
    trace_emitter_factory: RoutePolicyTraceEmitterFactory | None = None,
) -> RoutePolicyBaselineEvaluation:
    """Collect and score comparable rollout datasets for named policy baselines.

    Pass ``trace_emitter_factory`` to stream live :class:`PolicyTraceEvent`s
    per policy: the factory is called once with each ``policy_name`` and the
    returned emitter is attached to every cloned adapter for that policy. The
    emitter is closed automatically after the policy's rollout collection
    completes, so file-backed JSONL streams are flushed even on errors.
    """

    if not policies:
        raise ValueError("policies must contain at least one policy")
    results: list[RoutePolicyBaselineResult] = []
    for index, (policy_name, policy) in enumerate(policies.items()):
        dataset_id = f"{evaluation_id}-{index:02d}-{_slug(policy_name)}"
        emitter = trace_emitter_factory(policy_name) if trace_emitter_factory else None
        try:
            dataset = collect_route_policy_dataset(
                _fresh_route_policy_adapters(adapters, emitter=emitter),
                policy,
                episode_count=episode_count,
                dataset_id=dataset_id,
                seed_start=seed_start,
                goals=goals,
                max_steps=max_steps,
                metadata={
                    **dict(metadata or {}),
                    "baselinePolicy": policy_name,
                },
            )
        finally:
            if emitter is not None:
                emitter.close()
        quality = evaluate_route_policy_dataset_quality(
            dataset,
            thresholds=thresholds,
            metadata={"baselinePolicy": policy_name},
        )
        results.append(
            RoutePolicyBaselineResult(
                policy_name=policy_name,
                dataset=dataset,
                quality=quality,
                metadata={"collectorIndex": index},
            )
        )
    return RoutePolicyBaselineEvaluation(
        evaluation_id=evaluation_id,
        results=tuple(results),
        metadata=_json_mapping(metadata or {}),
    )


def render_route_policy_quality_markdown(report: RoutePolicyQualityReport) -> str:
    """Render a compact report suitable for CI logs or pull request comments."""

    metrics = report.metrics
    lines = [
        f"# Route Policy Quality: {report.dataset_id}",
        f"- Status: {'PASS' if report.passed else 'FAIL'}",
        f"- Episodes: {int(metrics.get('episode-count', 0.0))}",
        f"- Transitions: {int(metrics.get('transition-count', 0.0))}",
        f"- Scenes: {int(metrics.get('scene-count', 0.0))}",
        f"- Success rate: {_percent(metrics.get('success-rate', 0.0))}",
        f"- Collision rate: {_format_float(metrics.get('collision-rate', 0.0))}",
        f"- Truncation rate: {_percent(metrics.get('truncation-rate', 0.0))}",
        f"- Mean reward: {_format_float(metrics.get('mean-reward', 0.0))}",
    ]
    if report.failed_checks:
        lines.append(f"- Failed checks: {', '.join(report.failed_checks)}")
    return "\n".join(lines) + "\n"


def _failed_quality_checks(
    *,
    metrics: Mapping[str, float],
    thresholds: RoutePolicyQualityThresholds,
) -> tuple[str, ...]:
    checks: list[str] = []
    if metrics["episode-count"] < thresholds.min_episode_count:
        checks.append("min-episode-count")
    if metrics["transition-count"] < thresholds.min_transition_count:
        checks.append("min-transition-count")
    if metrics["scene-count"] < thresholds.min_scene_count:
        checks.append("min-scene-count")
    if metrics["success-rate"] < thresholds.min_success_rate:
        checks.append("min-success-rate")
    if metrics["collision-rate"] > thresholds.max_collision_rate:
        checks.append("max-collision-rate")
    if metrics["truncation-rate"] > thresholds.max_truncation_rate:
        checks.append("max-truncation-rate")
    if thresholds.min_mean_reward is not None and metrics["mean-reward"] < thresholds.min_mean_reward:
        checks.append("min-mean-reward")
    return tuple(checks)


def _fresh_route_policy_adapters(
    adapters: Sequence[RoutePolicyGymAdapter],
    *,
    emitter: RoutePolicyTraceEmitter | None = None,
) -> tuple[RoutePolicyGymAdapter, ...]:
    return tuple(
        RoutePolicyGymAdapter(adapter.environment, adapter.config, trace_emitter=emitter) for adapter in adapters
    )


def _transition_collision_count(info: Mapping[str, Any], next_observation: Mapping[str, float]) -> float:
    policy_sample = _mapping(info.get("policySample"))
    sample_observation = _mapping(policy_sample.get("observation"))
    sample_features = _mapping(sample_observation.get("features"))
    sample_collision_count = _optional_metric(sample_features, "collision-count")
    if sample_collision_count is not None:
        return sample_collision_count

    next_collision_count = _optional_metric(next_observation, "route-collision-count")
    if next_collision_count is not None:
        return next_collision_count

    rollout = _mapping(info.get("rollout"))
    rollout_metrics = _mapping(rollout.get("metrics"))
    rollout_collision_count = _optional_metric(rollout_metrics, "collision-count")
    return rollout_collision_count if rollout_collision_count is not None else 0.0


def _termination_bucket(summary: Mapping[str, Any], terminated: bool, truncated: bool) -> str:
    reason = summary.get("terminationReason")
    if reason is not None:
        return str(reason)
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "none"


def _quality_rank_key(report: RoutePolicyQualityReport) -> tuple[float, ...]:
    metrics = report.metrics
    return (
        1.0 if report.passed else 0.0,
        metrics.get("success-rate", 0.0),
        -metrics.get("collision-rate", 0.0),
        -metrics.get("truncation-rate", 0.0),
        metrics.get("mean-reward", 0.0),
        -metrics.get("mean-episode-length", 0.0),
    )


def _series_distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": float(len(ordered)),
        "min": ordered[0],
        "p50": _quantile(ordered, 0.5),
        "p90": _quantile(ordered, 0.9),
        "max": ordered[-1],
        "mean": _mean(ordered),
    }


def _quantile(ordered_values: Sequence[float], quantile: float) -> float:
    if not ordered_values:
        return 0.0
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    index = (len(ordered_values) - 1) * quantile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(ordered_values[lower])
    lower_value = float(ordered_values[lower])
    upper_value = float(ordered_values[upper])
    return lower_value + (upper_value - lower_value) * (index - lower)


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / float(len(values))


def _optional_metric(values: Mapping[str, Any], key: str) -> float | None:
    if key not in values:
        return None
    value = float(values[key])
    if not math.isfinite(value):
        raise ValueError(f"metric {key!r} must be finite")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _float_mapping(value: Mapping[str, float]) -> dict[str, float]:
    return {str(key): _finite_float(item, str(key)) for key, item in sorted(value.items())}


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON float values must be finite")
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in str(value))
    parts = [part for part in slug.split("-") if part]
    return "-".join(parts) or "policy"


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.4g}"


def _rate(value: float, field_name: str) -> float:
    normalized = _finite_float(value, field_name)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return normalized


def _non_negative_float(value: float, field_name: str) -> float:
    normalized = _finite_float(value, field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _finite_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _non_negative_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized
