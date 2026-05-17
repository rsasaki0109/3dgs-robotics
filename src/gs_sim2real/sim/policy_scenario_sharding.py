"""Scenario-set sharding and merge reports for route policy benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from os import path as os_path
from pathlib import Path
import re
from typing import Any

from .policy_benchmark_history import (
    RoutePolicyBenchmarkHistoryReport,
    RoutePolicyBenchmarkRegressionThresholds,
    build_route_policy_benchmark_history,
    render_route_policy_benchmark_history_markdown,
    route_policy_benchmark_history_from_dict,
    write_route_policy_benchmark_history_json,
)
from .policy_scenario_matrix import (
    RoutePolicyScenarioMatrixExpansionReport,
    load_route_policy_scenario_matrix_expansion_json,
)
from .policy_scenario_multi_agent import (
    InteractionMetricsAggregate,
    aggregate_interaction_metrics_across_scenarios,
    interaction_metrics_aggregate_from_dict,
)
from .policy_scenario_set import (
    RoutePolicyScenarioSet,
    RoutePolicyScenarioSetRunReport,
    RoutePolicyScenarioSpec,
    load_route_policy_scenario_set_run_json,
    route_policy_scenario_set_from_dict,
    write_route_policy_scenario_set_json,
)


ROUTE_POLICY_SCENARIO_SHARD_PLAN_VERSION = "gs-mapper-route-policy-scenario-shard-plan/v1"
ROUTE_POLICY_SCENARIO_SHARD_MERGE_VERSION = "gs-mapper-route-policy-scenario-shard-merge/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioShardSpec:
    """One executable shard scenario-set in a scenario shard plan."""

    shard_id: str
    source_scenario_set_id: str
    scenario_ids: tuple[str, ...]
    scenario_set_path: str | None = None
    source_scenario_set_path: str | None = None
    policy_registry_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.shard_id):
            raise ValueError("shard_id must not be empty")
        if not str(self.source_scenario_set_id):
            raise ValueError("source_scenario_set_id must not be empty")
        if not self.scenario_ids:
            raise ValueError("scenario shard must contain at least one scenario id")
        scenario_ids = tuple(str(scenario_id) for scenario_id in self.scenario_ids)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario shard must not contain duplicate scenario ids")
        object.__setattr__(self, "scenario_ids", scenario_ids)

    @property
    def scenario_count(self) -> int:
        return len(self.scenario_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-shard",
            "shardId": self.shard_id,
            "sourceScenarioSetId": self.source_scenario_set_id,
            "sourceScenarioSetPath": self.source_scenario_set_path,
            "scenarioSetPath": self.scenario_set_path,
            "policyRegistryPath": self.policy_registry_path,
            "scenarioCount": self.scenario_count,
            "scenarioIds": list(self.scenario_ids),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioShardPlan:
    """Versioned shard index plus embedded shard scenario-set definitions."""

    shard_plan_id: str
    shards: tuple[RoutePolicyScenarioShardSpec, ...]
    scenario_sets: tuple[RoutePolicyScenarioSet, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_SHARD_PLAN_VERSION

    def __post_init__(self) -> None:
        if not str(self.shard_plan_id):
            raise ValueError("shard_plan_id must not be empty")
        if not self.shards:
            raise ValueError("shard plan must contain at least one shard")
        if len(self.shards) != len(self.scenario_sets):
            raise ValueError("shard plan shards and scenario_sets must have the same length")
        shard_ids = tuple(shard.shard_id for shard in self.shards)
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError("shard plan must not contain duplicate shard ids")
        scenario_set_ids = tuple(scenario_set.scenario_set_id for scenario_set in self.scenario_sets)
        if shard_ids != scenario_set_ids:
            raise ValueError("shard ids must match embedded scenario-set ids in order")

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    @property
    def source_scenario_set_count(self) -> int:
        return len({shard.source_scenario_set_id for shard in self.shards})

    @property
    def scenario_count(self) -> int:
        return sum(shard.scenario_count for shard in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-shard-plan",
            "version": self.version,
            "shardPlanId": self.shard_plan_id,
            "shardCount": self.shard_count,
            "sourceScenarioSetCount": self.source_scenario_set_count,
            "scenarioCount": self.scenario_count,
            "shards": [shard.to_dict() for shard in self.shards],
            "scenarioSets": [scenario_set.to_dict() for scenario_set in self.scenario_sets],
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioShardRunSummary:
    """Compact summary of one independently executed shard run."""

    shard_id: str
    scenario_set_id: str
    passed: bool
    scenario_count: int
    report_paths: tuple[str, ...]
    run_path: str | None = None
    history_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.shard_id):
            raise ValueError("shard_id must not be empty")
        if not str(self.scenario_set_id):
            raise ValueError("scenario_set_id must not be empty")
        _positive_int(self.scenario_count, "scenario_count")
        report_paths = tuple(str(path) for path in self.report_paths)
        if not report_paths:
            raise ValueError("shard run summary must contain at least one report path")
        object.__setattr__(self, "report_paths", report_paths)

    @property
    def report_count(self) -> int:
        return len(self.report_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-shard-run-summary",
            "shardId": self.shard_id,
            "scenarioSetId": self.scenario_set_id,
            "passed": bool(self.passed),
            "scenarioCount": self.scenario_count,
            "reportCount": self.report_count,
            "runPath": self.run_path,
            "historyPath": self.history_path,
            "reportPaths": list(self.report_paths),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioShardMergeReport:
    """Merged shard run report with a global benchmark history gate.

    Sprint 4 / PR D4 adds the optional
    :class:`InteractionMetricsAggregate` attached after multi-agent
    runs. The aggregate is computed from each scenario result's
    ``interactionMetricsValues`` metadata mapping (populated by the run
    loop when peers are present); legacy ego-only runs produce no
    values and the aggregate stays ``None``.
    """

    merge_id: str
    shard_runs: tuple[RoutePolicyScenarioShardRunSummary, ...]
    history: RoutePolicyBenchmarkHistoryReport
    history_path: str | None = None
    history_markdown_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    interaction_metrics_aggregate: InteractionMetricsAggregate | None = None
    version: str = ROUTE_POLICY_SCENARIO_SHARD_MERGE_VERSION

    def __post_init__(self) -> None:
        if not str(self.merge_id):
            raise ValueError("merge_id must not be empty")
        if not self.shard_runs:
            raise ValueError("shard merge report must contain at least one shard run")

    @property
    def passed(self) -> bool:
        return self.history.passed and all(shard_run.passed for shard_run in self.shard_runs)

    @property
    def shard_count(self) -> int:
        return len(self.shard_runs)

    @property
    def scenario_count(self) -> int:
        return sum(shard_run.scenario_count for shard_run in self.shard_runs)

    @property
    def report_count(self) -> int:
        return sum(shard_run.report_count for shard_run in self.shard_runs)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-scenario-shard-merge",
            "version": self.version,
            "mergeId": self.merge_id,
            "passed": self.passed,
            "shardCount": self.shard_count,
            "scenarioCount": self.scenario_count,
            "reportCount": self.report_count,
            "historyPath": self.history_path,
            "historyMarkdownPath": self.history_markdown_path,
            "shardRuns": [shard_run.to_dict() for shard_run in self.shard_runs],
            "history": self.history.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }
        if self.interaction_metrics_aggregate is not None:
            payload["interactionMetricsAggregate"] = self.interaction_metrics_aggregate.to_dict()
        return payload


def split_route_policy_scenario_set_into_shards(
    scenario_set: RoutePolicyScenarioSet,
    *,
    max_scenarios_per_shard: int,
    shard_id_prefix: str | None = None,
) -> tuple[RoutePolicyScenarioSet, ...]:
    """Split one scenario set into smaller scenario-set artifacts."""

    _positive_int(max_scenarios_per_shard, "max_scenarios_per_shard")
    chunks = tuple(
        scenario_set.scenarios[index : index + max_scenarios_per_shard]
        for index in range(0, scenario_set.scenario_count, max_scenarios_per_shard)
    )
    prefix = shard_id_prefix or scenario_set.scenario_set_id
    width = max(3, len(str(len(chunks))))
    shards: list[RoutePolicyScenarioSet] = []
    for index, scenarios in enumerate(chunks, start=1):
        shard_id = f"{prefix}-shard-{index:0{width}d}"
        shards.append(
            RoutePolicyScenarioSet(
                scenario_set_id=shard_id,
                policy_registry_path=scenario_set.policy_registry_path,
                episode_count=scenario_set.episode_count,
                seed_start=scenario_set.seed_start,
                max_steps=scenario_set.max_steps,
                include_direct_baseline=scenario_set.include_direct_baseline,
                site_url=scenario_set.site_url,
                scenarios=scenarios,
                metadata={
                    **_json_mapping(scenario_set.metadata),
                    "sourceScenarioSetId": scenario_set.scenario_set_id,
                    "sourceScenarioIds": [scenario.scenario_id for scenario in scenarios],
                    "shardIndex": index,
                    "shardCount": len(chunks),
                    "maxScenariosPerShard": int(max_scenarios_per_shard),
                },
            )
        )
    return tuple(shards)


def build_route_policy_scenario_shard_plan(
    scenario_sets: Sequence[RoutePolicyScenarioSet],
    *,
    shard_plan_id: str = "route-policy-scenario-shards",
    max_scenarios_per_shard: int,
    source_scenario_set_paths: Sequence[str | Path | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioShardPlan:
    """Build an in-memory shard plan from scenario sets without writing files."""

    _positive_int(max_scenarios_per_shard, "max_scenarios_per_shard")
    if not scenario_sets:
        raise ValueError("scenario_sets must contain at least one scenario set")
    source_paths: tuple[str | None, ...]
    if source_scenario_set_paths is None:
        source_paths = tuple(None for _ in scenario_sets)
    else:
        source_paths = tuple(None if path is None else str(path) for path in source_scenario_set_paths)
    if len(source_paths) != len(scenario_sets):
        raise ValueError("source_scenario_set_paths must match scenario_sets length")

    shard_sets: list[RoutePolicyScenarioSet] = []
    shard_specs: list[RoutePolicyScenarioShardSpec] = []
    for scenario_set, source_path in zip(scenario_sets, source_paths, strict=True):
        for shard_set in split_route_policy_scenario_set_into_shards(
            scenario_set,
            max_scenarios_per_shard=max_scenarios_per_shard,
        ):
            shard_sets.append(shard_set)
            shard_specs.append(
                RoutePolicyScenarioShardSpec(
                    shard_id=shard_set.scenario_set_id,
                    source_scenario_set_id=scenario_set.scenario_set_id,
                    source_scenario_set_path=source_path,
                    scenario_set_path=None,
                    policy_registry_path=shard_set.policy_registry_path,
                    scenario_ids=tuple(scenario.scenario_id for scenario in shard_set.scenarios),
                    metadata=shard_set.metadata,
                )
            )

    return RoutePolicyScenarioShardPlan(
        shard_plan_id=shard_plan_id,
        shards=tuple(shard_specs),
        scenario_sets=tuple(shard_sets),
        metadata={
            "maxScenariosPerShard": int(max_scenarios_per_shard),
            **_json_mapping(metadata or {}),
        },
    )


def write_route_policy_scenario_shards_from_expansion(
    expansion: RoutePolicyScenarioMatrixExpansionReport,
    output_dir: str | Path,
    *,
    max_scenarios_per_shard: int,
    shard_plan_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioShardPlan:
    """Write shard scenario-set JSON files from a matrix expansion."""

    _positive_int(max_scenarios_per_shard, "max_scenarios_per_shard")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plan_id = shard_plan_id or f"{expansion.matrix_id}-shards"
    shard_sets: list[RoutePolicyScenarioSet] = []
    shard_specs: list[RoutePolicyScenarioShardSpec] = []

    for output, scenario_set in zip(expansion.outputs, expansion.scenario_sets, strict=True):
        source_path = output.scenario_set_path
        source_base = Path(source_path).parent if source_path is not None else Path(".")
        for shard_set in split_route_policy_scenario_set_into_shards(
            scenario_set,
            max_scenarios_per_shard=max_scenarios_per_shard,
        ):
            rebased_shard_set = _rebase_scenario_set_paths(
                shard_set,
                source_base=source_base,
                target_base=directory,
            )
            shard_path = directory / f"{_slug(rebased_shard_set.scenario_set_id)}.json"
            write_route_policy_scenario_set_json(shard_path, rebased_shard_set)
            shard_sets.append(rebased_shard_set)
            shard_specs.append(
                RoutePolicyScenarioShardSpec(
                    shard_id=rebased_shard_set.scenario_set_id,
                    source_scenario_set_id=scenario_set.scenario_set_id,
                    source_scenario_set_path=source_path,
                    scenario_set_path=shard_path.as_posix(),
                    policy_registry_path=rebased_shard_set.policy_registry_path,
                    scenario_ids=tuple(scenario.scenario_id for scenario in rebased_shard_set.scenarios),
                    metadata=rebased_shard_set.metadata,
                )
            )

    return RoutePolicyScenarioShardPlan(
        shard_plan_id=plan_id,
        shards=tuple(shard_specs),
        scenario_sets=tuple(shard_sets),
        metadata={
            "matrixId": expansion.matrix_id,
            "outputDir": directory.as_posix(),
            "maxScenariosPerShard": int(max_scenarios_per_shard),
            **_json_mapping(metadata or {}),
        },
    )


def merge_route_policy_scenario_shard_runs(
    shard_runs: Sequence[RoutePolicyScenarioSetRunReport],
    *,
    merge_id: str = "route-policy-scenario-shard-merge",
    run_paths: Sequence[str | Path | None] | None = None,
    baseline_report: str | Path | None = None,
    history_output: str | Path | None = None,
    history_markdown_output: str | Path | None = None,
    history_thresholds: RoutePolicyBenchmarkRegressionThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioShardMergeReport:
    """Merge independent shard run reports into one global history gate."""

    if not shard_runs:
        raise ValueError("shard_runs must contain at least one run report")
    resolved_run_paths: tuple[str | None, ...]
    if run_paths is None:
        resolved_run_paths = tuple(None for _ in shard_runs)
    else:
        resolved_run_paths = tuple(None if path is None else str(path) for path in run_paths)
    if len(resolved_run_paths) != len(shard_runs):
        raise ValueError("run_paths must match shard_runs length")

    summaries: list[RoutePolicyScenarioShardRunSummary] = []
    report_paths: list[str] = []
    scenario_metadatas: list[Mapping[str, Any]] = []
    for run, run_path in zip(shard_runs, resolved_run_paths, strict=True):
        shard_report_paths = tuple(result.report_path for result in run.scenario_results)
        report_paths.extend(shard_report_paths)
        scenario_metadatas.extend(result.metadata for result in run.scenario_results)
        summaries.append(
            RoutePolicyScenarioShardRunSummary(
                shard_id=run.scenario_set_id,
                scenario_set_id=run.scenario_set_id,
                run_path=run_path,
                passed=run.passed,
                scenario_count=run.scenario_count,
                history_path=run.history_path,
                report_paths=shard_report_paths,
                metadata=run.metadata,
            )
        )
    interaction_metrics_aggregate = aggregate_interaction_metrics_across_scenarios(
        scenario_metadatas
    )

    history = build_route_policy_benchmark_history(
        tuple(report_paths),
        baseline_report=baseline_report,
        history_id=f"{merge_id}-history",
        thresholds=history_thresholds,
        metadata={
            "mergeId": merge_id,
            "shardCount": len(summaries),
            "baselineReport": None if baseline_report is None else str(baseline_report),
        },
    )
    if history_output is not None:
        write_route_policy_benchmark_history_json(history_output, history)
    if history_markdown_output is not None:
        markdown_path = Path(history_markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_route_policy_benchmark_history_markdown(history), encoding="utf-8")

    return RoutePolicyScenarioShardMergeReport(
        merge_id=merge_id,
        shard_runs=tuple(summaries),
        history=history,
        history_path=None if history_output is None else str(history_output),
        history_markdown_path=None if history_markdown_output is None else str(history_markdown_output),
        metadata=_json_mapping(metadata or {}),
        interaction_metrics_aggregate=interaction_metrics_aggregate,
    )


def merge_route_policy_scenario_shard_run_jsons(
    run_paths: Sequence[str | Path],
    *,
    merge_id: str = "route-policy-scenario-shard-merge",
    baseline_report: str | Path | None = None,
    history_output: str | Path | None = None,
    history_markdown_output: str | Path | None = None,
    history_thresholds: RoutePolicyBenchmarkRegressionThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioShardMergeReport:
    """Load shard run JSON files and merge them into one report."""

    if not run_paths:
        raise ValueError("run_paths must contain at least one run report")
    paths = tuple(Path(path) for path in run_paths)
    runs = tuple(load_route_policy_scenario_set_run_json(path) for path in paths)
    return merge_route_policy_scenario_shard_runs(
        runs,
        merge_id=merge_id,
        run_paths=paths,
        baseline_report=baseline_report,
        history_output=history_output,
        history_markdown_output=history_markdown_output,
        history_thresholds=history_thresholds,
        metadata=metadata,
    )


def write_route_policy_scenario_shard_plan_json(
    path: str | Path,
    plan: RoutePolicyScenarioShardPlan,
) -> Path:
    """Write a route policy scenario shard plan JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_shard_plan_json(path: str | Path) -> RoutePolicyScenarioShardPlan:
    """Load a route policy scenario shard plan JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_shard_plan_from_dict(_mapping(payload, "shardPlan"))


def write_route_policy_scenario_shard_merge_json(
    path: str | Path,
    merge_report: RoutePolicyScenarioShardMergeReport,
) -> Path:
    """Write a route policy scenario shard merge report JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merge_report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_shard_merge_json(path: str | Path) -> RoutePolicyScenarioShardMergeReport:
    """Load a route policy scenario shard merge JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_shard_merge_from_dict(_mapping(payload, "shardMerge"))


def route_policy_scenario_shard_spec_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioShardSpec:
    """Rebuild one shard spec from JSON."""

    _record_type(payload, "route-policy-scenario-shard")
    return RoutePolicyScenarioShardSpec(
        shard_id=str(payload["shardId"]),
        source_scenario_set_id=str(payload["sourceScenarioSetId"]),
        source_scenario_set_path=None
        if payload.get("sourceScenarioSetPath") is None
        else str(payload["sourceScenarioSetPath"]),
        scenario_set_path=None if payload.get("scenarioSetPath") is None else str(payload["scenarioSetPath"]),
        policy_registry_path=None if payload.get("policyRegistryPath") is None else str(payload["policyRegistryPath"]),
        scenario_ids=tuple(str(item) for item in _sequence(payload.get("scenarioIds", ()), "scenarioIds")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_shard_plan_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioShardPlan:
    """Rebuild a shard plan from JSON."""

    _record_type(payload, "route-policy-scenario-shard-plan")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_SHARD_PLAN_VERSION))
    if version != ROUTE_POLICY_SCENARIO_SHARD_PLAN_VERSION:
        raise ValueError(f"unsupported route policy scenario shard plan version: {version}")
    shards = tuple(
        route_policy_scenario_shard_spec_from_dict(_mapping(item, "shard"))
        for item in _sequence(payload.get("shards", ()), "shards")
    )
    scenario_sets = tuple(
        route_policy_scenario_set_from_dict(_mapping(item, "scenarioSet"))
        for item in _sequence(payload.get("scenarioSets", ()), "scenarioSets")
    )
    expected_shard_count = payload.get("shardCount")
    if expected_shard_count is not None and int(expected_shard_count) != len(shards):
        raise ValueError("shardCount does not match loaded shards")
    expected_scenario_count = payload.get("scenarioCount")
    if expected_scenario_count is not None:
        loaded_scenario_count = sum(shard.scenario_count for shard in shards)
        if int(expected_scenario_count) != loaded_scenario_count:
            raise ValueError("scenarioCount does not match loaded shards")
    return RoutePolicyScenarioShardPlan(
        shard_plan_id=str(payload["shardPlanId"]),
        shards=shards,
        scenario_sets=scenario_sets,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_scenario_shard_run_summary_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioShardRunSummary:
    """Rebuild one shard run summary from JSON."""

    _record_type(payload, "route-policy-scenario-shard-run-summary")
    return RoutePolicyScenarioShardRunSummary(
        shard_id=str(payload["shardId"]),
        scenario_set_id=str(payload["scenarioSetId"]),
        passed=bool(payload.get("passed", False)),
        scenario_count=int(payload["scenarioCount"]),
        run_path=None if payload.get("runPath") is None else str(payload["runPath"]),
        history_path=None if payload.get("historyPath") is None else str(payload["historyPath"]),
        report_paths=tuple(str(item) for item in _sequence(payload.get("reportPaths", ()), "reportPaths")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_shard_merge_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioShardMergeReport:
    """Rebuild a shard merge report from JSON."""

    _record_type(payload, "route-policy-scenario-shard-merge")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_SHARD_MERGE_VERSION))
    if version != ROUTE_POLICY_SCENARIO_SHARD_MERGE_VERSION:
        raise ValueError(f"unsupported route policy scenario shard merge version: {version}")
    aggregate_payload = payload.get("interactionMetricsAggregate")
    return RoutePolicyScenarioShardMergeReport(
        merge_id=str(payload["mergeId"]),
        shard_runs=tuple(
            route_policy_scenario_shard_run_summary_from_dict(_mapping(item, "shardRun"))
            for item in _sequence(payload.get("shardRuns", ()), "shardRuns")
        ),
        history=route_policy_benchmark_history_from_dict(_mapping(payload["history"], "history")),
        history_path=None if payload.get("historyPath") is None else str(payload["historyPath"]),
        history_markdown_path=None
        if payload.get("historyMarkdownPath") is None
        else str(payload["historyMarkdownPath"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        interaction_metrics_aggregate=None
        if aggregate_payload is None
        else interaction_metrics_aggregate_from_dict(
            _mapping(aggregate_payload, "interactionMetricsAggregate")
        ),
        version=version,
    )


def render_route_policy_scenario_shard_plan_markdown(plan: RoutePolicyScenarioShardPlan) -> str:
    """Render a compact Markdown summary for a shard plan."""

    lines = [
        f"# Route Policy Scenario Shards: {plan.shard_plan_id}",
        f"- Shards: {plan.shard_count}",
        f"- Source scenario sets: {plan.source_scenario_set_count}",
        f"- Scenarios: {plan.scenario_count}",
        "",
        "| Shard | Source set | Scenarios | Path |",
        "| --- | --- | ---: | --- |",
    ]
    for shard in plan.shards:
        lines.append(
            "| "
            f"{shard.shard_id} | "
            f"{shard.source_scenario_set_id} | "
            f"{shard.scenario_count} | "
            f"{shard.scenario_set_path or 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def render_route_policy_scenario_shard_merge_markdown(report: RoutePolicyScenarioShardMergeReport) -> str:
    """Render a compact Markdown summary for a shard merge report."""

    lines = [
        f"# Route Policy Scenario Shard Merge: {report.merge_id}",
        f"- Status: {'PASS' if report.passed else 'FAIL'}",
        f"- Shards: {report.shard_count}",
        f"- Scenarios: {report.scenario_count}",
        f"- Reports: {report.report_count}",
        f"- History: {'PASS' if report.history.passed else 'FAIL'}",
        "",
        "| Shard | Pass | Scenarios | Reports | Run |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for shard_run in report.shard_runs:
        lines.append(
            "| "
            f"{shard_run.shard_id} | "
            f"{'yes' if shard_run.passed else 'no'} | "
            f"{shard_run.scenario_count} | "
            f"{shard_run.report_count} | "
            f"{shard_run.run_path or 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def run_shard_plan_cli(args: Any) -> None:
    """Run the route policy scenario-shards CLI."""

    expansion = load_route_policy_scenario_matrix_expansion_json(getattr(args, "expansion"))
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        getattr(args, "output_dir"),
        max_scenarios_per_shard=int(getattr(args, "max_scenarios_per_shard")),
        shard_plan_id=getattr(args, "shard_plan_id", None),
    )
    write_route_policy_scenario_shard_plan_json(getattr(args, "index_output"), plan)
    markdown = render_route_policy_scenario_shard_plan_markdown(plan)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario shard plan saved to: {getattr(args, 'index_output')}")


def run_shard_merge_cli(args: Any) -> None:
    """Run the route policy scenario-shard-merge CLI."""

    thresholds = RoutePolicyBenchmarkRegressionThresholds(
        max_success_rate_drop=float(getattr(args, "max_success_rate_drop")),
        max_collision_rate_increase=float(getattr(args, "max_collision_rate_increase")),
        max_truncation_rate_increase=float(getattr(args, "max_truncation_rate_increase")),
        max_mean_reward_drop=getattr(args, "max_mean_reward_drop", None),
        require_baseline_policies=not bool(getattr(args, "allow_missing_policies", False)),
        fail_on_report_failure=not bool(getattr(args, "allow_report_failures", False)),
    )
    report = merge_route_policy_scenario_shard_run_jsons(
        tuple(getattr(args, "run")),
        merge_id=str(getattr(args, "merge_id")),
        baseline_report=getattr(args, "baseline_report", None),
        history_output=getattr(args, "history_output"),
        history_markdown_output=getattr(args, "history_markdown_output", None),
        history_thresholds=thresholds,
        metadata={"baselineReport": getattr(args, "baseline_report", None)},
    )
    write_route_policy_scenario_shard_merge_json(getattr(args, "output"), report)
    markdown = render_route_policy_scenario_shard_merge_markdown(report)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario shard merge saved to: {getattr(args, 'output')}")
    if bool(getattr(args, "fail_on_regression", False)) and not report.passed:
        raise SystemExit(2)


def _rebase_scenario_set_paths(
    scenario_set: RoutePolicyScenarioSet,
    *,
    source_base: Path,
    target_base: Path,
) -> RoutePolicyScenarioSet:
    return RoutePolicyScenarioSet(
        scenario_set_id=scenario_set.scenario_set_id,
        policy_registry_path=_rebase_optional_path(scenario_set.policy_registry_path, source_base, target_base),
        episode_count=scenario_set.episode_count,
        seed_start=scenario_set.seed_start,
        max_steps=scenario_set.max_steps,
        include_direct_baseline=scenario_set.include_direct_baseline,
        site_url=scenario_set.site_url,
        scenarios=tuple(
            RoutePolicyScenarioSpec(
                scenario_id=scenario.scenario_id,
                scene_catalog=_rebase_path(scenario.scene_catalog, source_base, target_base),
                scene_id=scenario.scene_id,
                goal_suite_path=_rebase_optional_path(scenario.goal_suite_path, source_base, target_base),
                episode_count=scenario.episode_count,
                seed_start=scenario.seed_start,
                max_steps=scenario.max_steps,
                site_url=scenario.site_url,
                thresholds=scenario.thresholds,
                metadata=scenario.metadata,
            )
            for scenario in scenario_set.scenarios
        ),
        metadata=scenario_set.metadata,
        version=scenario_set.version,
    )


def _rebase_optional_path(path_value: str | None, source_base: Path, target_base: Path) -> str | None:
    if path_value is None:
        return None
    return _rebase_path(path_value, source_base, target_base)


def _rebase_path(path_value: str, source_base: Path, target_base: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return path.as_posix()
    resolved = (source_base / path).resolve()
    return Path(os_path.relpath(resolved, start=target_base.resolve())).as_posix()


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


def _positive_int(value: int, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be positive")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "unnamed"
