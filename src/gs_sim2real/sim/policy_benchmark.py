"""Benchmark runner and report artifacts for route policy baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .contract import SimulationCatalog, load_simulation_catalog_from_scene_picker
from .gym_adapter import RoutePolicyGymAdapter, RoutePolicyEnvConfig
from .headless import HeadlessPhysicalAIEnvironment
from .interfaces import Pose3D
from .policy_dataset import RoutePolicyCallable, RoutePolicyGoal
from .policy_imitation import (
    RoutePolicyImitationFitConfig,
    RoutePolicyImitationModel,
    fit_route_policy_imitation_model,
    load_route_policy_imitation_model_json,
    write_route_policy_imitation_model_json,
)
from .policy_quality import (
    RoutePolicyBaselineEvaluation,
    RoutePolicyQualityThresholds,
    RoutePolicyTraceEmitterFactory,
    evaluate_route_policy_baselines,
)
from .policy_replay import (
    RoutePolicyReplaySource,
    build_route_policy_replay_batch,
    build_route_policy_replay_schema,
    load_route_policy_dataset_json,
    load_route_policy_transitions_jsonl,
)


ROUTE_POLICY_BENCHMARK_VERSION = "gs-mapper-route-policy-benchmark/v1"
ROUTE_POLICY_GOAL_SUITE_VERSION = "gs-mapper-route-policy-goal-suite/v1"
ROUTE_POLICY_REGISTRY_VERSION = "gs-mapper-route-policy-registry/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyGoalSpec:
    """One named target pose in a reusable route policy goal suite."""

    goal_id: str
    position: tuple[float, float, float]
    frame_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.goal_id):
            raise ValueError("goal_id must not be empty")
        object.__setattr__(self, "position", _position_tuple(self.position))

    def to_pose(self, *, frame_id: str) -> Pose3D:
        return Pose3D(
            position=self.position,
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=self.frame_id or frame_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goalId": self.goal_id,
            "position": list(self.position),
            "frameId": self.frame_id,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyGoalSuite:
    """Versioned goal suite shared by route policy benchmark runs."""

    suite_id: str
    goals: tuple[RoutePolicyGoalSpec, ...]
    scene_id: str | None = None
    frame_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_GOAL_SUITE_VERSION

    def __post_init__(self) -> None:
        if not str(self.suite_id):
            raise ValueError("suite_id must not be empty")
        if not self.goals:
            raise ValueError("goal suite must contain at least one goal")
        goal_ids = tuple(goal.goal_id for goal in self.goals)
        if len(set(goal_ids)) != len(goal_ids):
            raise ValueError("goal suite must not contain duplicate goal ids")

    @property
    def goal_count(self) -> int:
        return len(self.goals)

    def to_goals(self, *, frame_id: str) -> tuple[Pose3D, ...]:
        resolved_frame_id = self.frame_id or frame_id
        return tuple(goal.to_pose(frame_id=resolved_frame_id) for goal in self.goals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-goal-suite",
            "version": self.version,
            "suiteId": self.suite_id,
            "sceneId": self.scene_id,
            "frameId": self.frame_id,
            "goalCount": self.goal_count,
            "goals": [goal.to_dict() for goal in self.goals],
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyRegistryEntry:
    """One named policy entry in a benchmark registry."""

    policy_name: str
    policy_type: str
    model_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.policy_name):
            raise ValueError("policy_name must not be empty")
        normalized_type = str(self.policy_type).strip().lower()
        object.__setattr__(self, "policy_type", normalized_type)
        if normalized_type == "imitation-model" and not self.model_path:
            raise ValueError("imitation-model registry entries require model_path")
        if normalized_type not in {"imitation-model", "direct-goal"}:
            raise ValueError("policy_type must be 'imitation-model' or 'direct-goal'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policyName": self.policy_name,
            "policyType": self.policy_type,
            "modelPath": self.model_path,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyRegistry:
    """Versioned list of policies to evaluate under the same route policy benchmark."""

    registry_id: str
    policies: tuple[RoutePolicyRegistryEntry, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if not str(self.registry_id):
            raise ValueError("registry_id must not be empty")
        if not self.policies:
            raise ValueError("policy registry must contain at least one policy")
        policy_names = tuple(policy.policy_name for policy in self.policies)
        if len(set(policy_names)) != len(policy_names):
            raise ValueError("policy registry must not contain duplicate policy names")

    @property
    def policy_count(self) -> int:
        return len(self.policies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-registry",
            "version": self.version,
            "registryId": self.registry_id,
            "policyCount": self.policy_count,
            "policies": [policy.to_dict() for policy in self.policies],
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkReport:
    """JSON-friendly benchmark report for one route policy comparison run."""

    benchmark_id: str
    evaluation: RoutePolicyBaselineEvaluation
    model_summary: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_BENCHMARK_VERSION

    @property
    def best_policy_name(self) -> str | None:
        return self.evaluation.best_policy_name

    @property
    def passed(self) -> bool:
        return bool(self.evaluation.results) and all(result.passed for result in self.evaluation.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-benchmark-report",
            "version": self.version,
            "benchmarkId": self.benchmark_id,
            "passed": self.passed,
            "bestPolicyName": self.best_policy_name,
            "summary": _evaluation_summary(self.evaluation),
            "modelSummary": _json_mapping(self.model_summary),
            "evaluation": self.evaluation.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }


def run_route_policy_imitation_benchmark(
    adapters: Sequence[RoutePolicyGymAdapter],
    model: RoutePolicyImitationModel,
    *,
    episode_count: int,
    benchmark_id: str = "route-policy-benchmark",
    policy_name: str = "imitation",
    include_direct_baseline: bool = False,
    seed_start: int = 0,
    goals: Sequence[RoutePolicyGoal] | None = None,
    max_steps: int | None = None,
    thresholds: RoutePolicyQualityThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
    trace_emitter_factory: RoutePolicyTraceEmitterFactory | None = None,
) -> RoutePolicyBenchmarkReport:
    """Evaluate a fitted imitation model, optionally beside a direct-goal baseline."""

    policies: dict[str, RoutePolicyCallable] = {}
    if include_direct_baseline:
        policies["direct"] = _direct_goal_policy
    policies[policy_name] = model.as_policy()
    evaluation = evaluate_route_policy_baselines(
        adapters,
        policies,
        episode_count=episode_count,
        evaluation_id=benchmark_id,
        seed_start=seed_start,
        goals=goals,
        max_steps=max_steps,
        thresholds=thresholds,
        metadata={
            "benchmarkId": benchmark_id,
            "policyName": policy_name,
            "includeDirectBaseline": include_direct_baseline,
            **_json_mapping(metadata or {}),
        },
        trace_emitter_factory=trace_emitter_factory,
    )
    return RoutePolicyBenchmarkReport(
        benchmark_id=benchmark_id,
        evaluation=evaluation,
        model_summary=_model_summary(model),
        metadata={
            "policyName": policy_name,
            "includeDirectBaseline": include_direct_baseline,
            **_json_mapping(metadata or {}),
        },
    )


def run_route_policy_registry_benchmark(
    adapters: Sequence[RoutePolicyGymAdapter],
    registry: RoutePolicyRegistry,
    *,
    episode_count: int,
    benchmark_id: str = "route-policy-benchmark",
    registry_base_path: str | Path | None = None,
    include_direct_baseline: bool = False,
    seed_start: int = 0,
    goals: Sequence[RoutePolicyGoal] | None = None,
    max_steps: int | None = None,
    thresholds: RoutePolicyQualityThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
    trace_emitter_factory: RoutePolicyTraceEmitterFactory | None = None,
) -> RoutePolicyBenchmarkReport:
    """Evaluate every policy in a registry under one fixed route policy task."""

    policies, policy_summaries = _resolve_registry_policies(registry, registry_base_path=registry_base_path)
    if include_direct_baseline and "direct" not in policies:
        policies = {"direct": _direct_goal_policy, **policies}
        policy_summaries = (
            {"policyName": "direct", "policyType": "direct-goal", "source": "builtin"},
            *policy_summaries,
        )
    evaluation = evaluate_route_policy_baselines(
        adapters,
        policies,
        episode_count=episode_count,
        evaluation_id=benchmark_id,
        seed_start=seed_start,
        goals=goals,
        max_steps=max_steps,
        thresholds=thresholds,
        metadata={
            "benchmarkId": benchmark_id,
            "registryId": registry.registry_id,
            "includeDirectBaseline": include_direct_baseline,
            **_json_mapping(metadata or {}),
        },
        trace_emitter_factory=trace_emitter_factory,
    )
    return RoutePolicyBenchmarkReport(
        benchmark_id=benchmark_id,
        evaluation=evaluation,
        model_summary={
            "registry": registry.to_dict(),
            "policies": list(policy_summaries),
        },
        metadata={
            "registryId": registry.registry_id,
            "includeDirectBaseline": include_direct_baseline,
            **_json_mapping(metadata or {}),
        },
    )


def write_route_policy_benchmark_report_json(path: str | Path, report: RoutePolicyBenchmarkReport) -> Path:
    """Write a route policy benchmark report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_route_policy_goal_suite_json(path: str | Path, suite: RoutePolicyGoalSuite) -> Path:
    """Write a versioned route policy goal suite JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(suite.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_goal_suite_json(path: str | Path) -> RoutePolicyGoalSuite:
    """Load a route policy goal suite JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_goal_suite_from_dict(_mapping(payload, "goalSuite"))


def route_policy_goal_suite_from_dict(payload: Mapping[str, Any]) -> RoutePolicyGoalSuite:
    """Rebuild a goal suite from its JSON payload."""

    _record_type(payload, "route-policy-goal-suite")
    version = str(payload.get("version", ROUTE_POLICY_GOAL_SUITE_VERSION))
    if version != ROUTE_POLICY_GOAL_SUITE_VERSION:
        raise ValueError(f"unsupported route policy goal suite version: {version}")
    goals = tuple(
        RoutePolicyGoalSpec(
            goal_id=str(goal["goalId"]),
            position=_position_tuple(_sequence(goal.get("position", ()), "position")),
            frame_id=None if goal.get("frameId") is None else str(goal["frameId"]),
            metadata=_json_mapping(_mapping(goal.get("metadata", {}), "metadata")),
        )
        for goal in (_mapping(item, "goal") for item in _sequence(payload.get("goals", ()), "goals"))
    )
    expected_goal_count = payload.get("goalCount")
    if expected_goal_count is not None and int(expected_goal_count) != len(goals):
        raise ValueError("goalCount does not match loaded goals")
    return RoutePolicyGoalSuite(
        suite_id=str(payload["suiteId"]),
        goals=goals,
        scene_id=None if payload.get("sceneId") is None else str(payload["sceneId"]),
        frame_id=None if payload.get("frameId") is None else str(payload["frameId"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def write_route_policy_registry_json(path: str | Path, registry: RoutePolicyRegistry) -> Path:
    """Write a versioned route policy registry JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_registry_json(path: str | Path) -> RoutePolicyRegistry:
    """Load a route policy registry JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_registry_from_dict(_mapping(payload, "registry"))


def route_policy_registry_from_dict(payload: Mapping[str, Any]) -> RoutePolicyRegistry:
    """Rebuild a policy registry from its JSON payload."""

    _record_type(payload, "route-policy-registry")
    version = str(payload.get("version", ROUTE_POLICY_REGISTRY_VERSION))
    if version != ROUTE_POLICY_REGISTRY_VERSION:
        raise ValueError(f"unsupported route policy registry version: {version}")
    policies = tuple(
        RoutePolicyRegistryEntry(
            policy_name=str(policy["policyName"]),
            policy_type=str(policy["policyType"]),
            model_path=None if policy.get("modelPath") is None else str(policy["modelPath"]),
            metadata=_json_mapping(_mapping(policy.get("metadata", {}), "metadata")),
        )
        for policy in (_mapping(item, "policy") for item in _sequence(payload.get("policies", ()), "policies"))
    )
    expected_policy_count = payload.get("policyCount")
    if expected_policy_count is not None and int(expected_policy_count) != len(policies):
        raise ValueError("policyCount does not match loaded policies")
    return RoutePolicyRegistry(
        registry_id=str(payload["registryId"]),
        policies=policies,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_benchmark_markdown(report: RoutePolicyBenchmarkReport) -> str:
    """Render a compact Markdown summary for CLI logs or CI artifacts."""

    lines = [
        f"# Route Policy Benchmark: {report.benchmark_id}",
        f"- Status: {'PASS' if report.passed else 'FAIL'}",
        f"- Best policy: {report.best_policy_name or 'n/a'}",
        f"- Policies: {len(report.evaluation.results)}",
        "",
        "| Policy | Pass | Success | Collision | Truncation | Mean reward |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report.evaluation.results:
        metrics = result.quality.metrics
        lines.append(
            "| "
            f"{result.policy_name} | "
            f"{'yes' if result.passed else 'no'} | "
            f"{_percent(metrics.get('success-rate', 0.0))} | "
            f"{_format_float(metrics.get('collision-rate', 0.0))} | "
            f"{_percent(metrics.get('truncation-rate', 0.0))} | "
            f"{_format_float(metrics.get('mean-reward', 0.0))} |"
        )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy benchmark CLI."""

    catalog = load_simulation_catalog_from_scene_picker(
        getattr(args, "scene_catalog"),
        site_url=getattr(args, "site_url", "https://rsasaki0109.github.io/3dgs-robotics/"),
    )
    goal_suite_path = getattr(args, "goal_suite", None)
    goal_suite = load_route_policy_goal_suite_json(goal_suite_path) if goal_suite_path else None
    scene_id = _resolve_scene_id(catalog, getattr(args, "scene_id", None), goal_suite=goal_suite)
    adapter = RoutePolicyGymAdapter(
        HeadlessPhysicalAIEnvironment(catalog),
        RoutePolicyEnvConfig(
            scene_id=scene_id,
            max_steps=getattr(args, "max_steps", None) or RoutePolicyEnvConfig().max_steps,
        ),
    )
    goals = _goals_from_args(catalog, scene_id, getattr(args, "goal", None), goal_suite=goal_suite)
    registry_path = getattr(args, "policy_registry", None)
    metadata = {
        "sceneCatalog": str(getattr(args, "scene_catalog")),
        "sceneId": scene_id,
        "goalSuite": str(goal_suite_path) if goal_suite_path else None,
    }
    if registry_path:
        if getattr(args, "model_output", None):
            raise ValueError("--model-output cannot be used with --policy-registry")
        registry = load_route_policy_registry_json(registry_path)
        report = run_route_policy_registry_benchmark(
            (adapter,),
            registry,
            episode_count=int(getattr(args, "episode_count")),
            benchmark_id=str(getattr(args, "benchmark_id")),
            registry_base_path=Path(registry_path).parent,
            include_direct_baseline=bool(getattr(args, "include_direct_baseline", False)),
            seed_start=int(getattr(args, "seed_start")),
            goals=goals,
            max_steps=getattr(args, "max_steps", None),
            thresholds=_thresholds_from_args(args),
            metadata={
                **metadata,
                "registryPath": str(registry_path),
            },
        )
    else:
        model, model_source = _load_or_fit_model(args)
        if getattr(args, "model_output", None):
            write_route_policy_imitation_model_json(args.model_output, model)
        report = run_route_policy_imitation_benchmark(
            (adapter,),
            model,
            episode_count=int(getattr(args, "episode_count")),
            benchmark_id=str(getattr(args, "benchmark_id")),
            policy_name=str(getattr(args, "policy_name")),
            include_direct_baseline=bool(getattr(args, "include_direct_baseline", False)),
            seed_start=int(getattr(args, "seed_start")),
            goals=goals,
            max_steps=getattr(args, "max_steps", None),
            thresholds=_thresholds_from_args(args),
            metadata={
                **metadata,
                "modelSource": model_source,
            },
        )
    write_route_policy_benchmark_report_json(getattr(args, "output"), report)
    markdown = render_route_policy_benchmark_markdown(report)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Report saved to: {getattr(args, 'output')}")
    if getattr(args, "model_output", None):
        print(f"Model saved to: {getattr(args, 'model_output')}")


def _load_or_fit_model(args: Any) -> tuple[RoutePolicyImitationModel, str]:
    model_path = getattr(args, "model", None)
    dataset_json = getattr(args, "dataset_json", None)
    transitions_jsonl = getattr(args, "transitions_jsonl", None)
    training_sources = tuple(item for item in (dataset_json, transitions_jsonl) if item)
    if model_path and training_sources:
        raise ValueError("--model cannot be combined with --dataset-json or --transitions-jsonl")
    if model_path:
        return load_route_policy_imitation_model_json(model_path), str(model_path)
    if len(training_sources) != 1:
        raise ValueError("provide exactly one of --model, --dataset-json, or --transitions-jsonl")

    if dataset_json:
        source: RoutePolicyReplaySource = load_route_policy_dataset_json(dataset_json)
        source_label = str(dataset_json)
    else:
        assert transitions_jsonl is not None
        source = load_route_policy_transitions_jsonl(transitions_jsonl)
        source_label = str(transitions_jsonl)

    action_keys = getattr(args, "action_keys", None)
    schema = (
        build_route_policy_replay_schema(source, action_keys=tuple(action_keys))
        if action_keys is not None
        else build_route_policy_replay_schema(source)
    )
    batch = build_route_policy_replay_batch(source, schema=schema)
    model = fit_route_policy_imitation_model(
        batch,
        config=RoutePolicyImitationFitConfig(neighbor_count=int(getattr(args, "neighbor_count"))),
        metadata={"source": source_label},
    )
    return model, source_label


def _resolve_scene_id(
    catalog: SimulationCatalog,
    scene_id: str | None,
    *,
    goal_suite: RoutePolicyGoalSuite | None = None,
) -> str:
    if scene_id:
        catalog.scene_by_id(scene_id)
        return str(scene_id)
    if goal_suite is not None and goal_suite.scene_id:
        catalog.scene_by_id(goal_suite.scene_id)
        return goal_suite.scene_id
    if "outdoor-demo" in catalog.scene_ids():
        return "outdoor-demo"
    scene_ids = catalog.scene_ids()
    if not scene_ids:
        raise ValueError("simulation catalog must contain at least one scene")
    return scene_ids[0]


def _goals_from_args(
    catalog: SimulationCatalog,
    scene_id: str,
    goal_rows: Sequence[Sequence[float]] | None,
    *,
    goal_suite: RoutePolicyGoalSuite | None = None,
) -> tuple[Pose3D, ...] | None:
    if goal_suite is not None and goal_rows:
        raise ValueError("--goal cannot be combined with --goal-suite")
    if not goal_rows:
        if goal_suite is None:
            return None
        frame_id = catalog.scene_by_id(scene_id).coordinate_frame.frame_id
        return goal_suite.to_goals(frame_id=frame_id)
    frame_id = catalog.scene_by_id(scene_id).coordinate_frame.frame_id
    return tuple(
        Pose3D(
            position=_position_tuple(row),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=frame_id,
        )
        for row in goal_rows
    )


def _thresholds_from_args(args: Any) -> RoutePolicyQualityThresholds | None:
    values = {
        "min_success_rate": getattr(args, "min_success_rate", None),
        "max_collision_rate": getattr(args, "max_collision_rate", None),
        "max_truncation_rate": getattr(args, "max_truncation_rate", None),
        "min_episode_count": getattr(args, "min_episode_count", None),
        "min_transition_count": getattr(args, "min_transition_count", None),
    }
    if all(value is None for value in values.values()):
        return None
    defaults = RoutePolicyQualityThresholds()
    return RoutePolicyQualityThresholds(
        min_success_rate=defaults.min_success_rate
        if values["min_success_rate"] is None
        else float(values["min_success_rate"]),
        max_collision_rate=defaults.max_collision_rate
        if values["max_collision_rate"] is None
        else float(values["max_collision_rate"]),
        max_truncation_rate=defaults.max_truncation_rate
        if values["max_truncation_rate"] is None
        else float(values["max_truncation_rate"]),
        min_scene_count=defaults.min_scene_count,
        min_episode_count=defaults.min_episode_count
        if values["min_episode_count"] is None
        else int(values["min_episode_count"]),
        min_transition_count=defaults.min_transition_count
        if values["min_transition_count"] is None
        else int(values["min_transition_count"]),
    )


def _resolve_registry_policies(
    registry: RoutePolicyRegistry,
    *,
    registry_base_path: str | Path | None,
) -> tuple[dict[str, RoutePolicyCallable], tuple[dict[str, Any], ...]]:
    policies: dict[str, RoutePolicyCallable] = {}
    summaries: list[dict[str, Any]] = []
    base_path = Path(registry_base_path) if registry_base_path is not None else Path(".")
    for entry in registry.policies:
        if entry.policy_type == "direct-goal":
            policies[entry.policy_name] = _direct_goal_policy
            summaries.append(
                {
                    "policyName": entry.policy_name,
                    "policyType": entry.policy_type,
                    "source": "builtin",
                    "metadata": _json_mapping(entry.metadata),
                }
            )
            continue
        assert entry.model_path is not None
        model_path = _resolve_registry_path(base_path, entry.model_path)
        model = load_route_policy_imitation_model_json(model_path)
        policies[entry.policy_name] = model.as_policy()
        summaries.append(
            {
                "policyName": entry.policy_name,
                "policyType": entry.policy_type,
                "modelPath": model_path.as_posix(),
                "sampleCount": model.sample_count,
                "schema": model.schema.to_dict(),
                "config": model.config.to_dict(),
                "metadata": _json_mapping(entry.metadata),
            }
        )
    return policies, tuple(summaries)


def _resolve_registry_path(base_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_path / path


def _direct_goal_policy(observation: Mapping[str, float], info: Mapping[str, Any]) -> dict[str, Any]:
    del observation
    return {
        "routeId": f"direct-{info.get('episodeIndex', 0)}-{info.get('stepIndex', 0)}",
        "target": info["goal"],
    }


def _model_summary(model: RoutePolicyImitationModel) -> dict[str, Any]:
    return {
        "version": model.version,
        "sampleCount": model.sample_count,
        "schema": model.schema.to_dict(),
        "config": model.config.to_dict(),
        "metadata": _json_mapping(model.metadata),
    }


def _evaluation_summary(evaluation: RoutePolicyBaselineEvaluation) -> dict[str, Any]:
    return {
        "evaluationId": evaluation.evaluation_id,
        "bestPolicyName": evaluation.best_policy_name,
        "policyCount": len(evaluation.results),
        "policies": [
            {
                "policyName": result.policy_name,
                "passed": result.passed,
                "metrics": _float_mapping(result.quality.metrics),
                "failedChecks": list(result.quality.failed_checks),
            }
            for result in evaluation.results
        ],
    }


def _position_tuple(row: Sequence[float]) -> tuple[float, float, float]:
    if len(row) != 3:
        raise ValueError("goal must contain exactly three coordinates")
    return (float(row[0]), float(row[1]), float(row[2]))


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be a sequence")


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _float_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


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
        return float(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")
