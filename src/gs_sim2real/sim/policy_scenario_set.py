"""Scenario-set execution for route policy benchmark registries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .contract import DEFAULT_SITE_URL, SimulationCatalog, load_simulation_catalog_from_scene_picker
from .gym_adapter import RoutePolicyEnvConfig, RoutePolicyGymAdapter
from .headless import HeadlessPhysicalAIEnvironment
from .interfaces import Pose3D
from .policy_benchmark import (
    RoutePolicyGoalSuite,
    RoutePolicyRegistry,
    load_route_policy_goal_suite_json,
    load_route_policy_registry_json,
    render_route_policy_benchmark_markdown,
    run_route_policy_registry_benchmark,
    write_route_policy_benchmark_report_json,
)
from .policy_benchmark_history import (
    RoutePolicyBenchmarkHistoryReport,
    RoutePolicyBenchmarkRegressionThresholds,
    build_route_policy_benchmark_history,
    render_route_policy_benchmark_history_markdown,
    route_policy_benchmark_history_from_dict,
    write_route_policy_benchmark_history_json,
)
from .policy_dynamic_obstacles import load_route_policy_dynamic_obstacle_timeline_json
from .policy_scenario_multi_agent import (
    SCENARIO_INTERACTION_METRIC_VALUES_KEY,
    synthesize_peer_roster_from_scenario_metadata,
)
from .policy_quality import RoutePolicyQualityThresholds
from .policy_sensor_noise import load_route_policy_sensor_noise_profile_json
from .raw_sensor_noise import load_raw_sensor_noise_profile_json
from ..robotics.rosbag_correlation import (
    RealVsSimCorrelationReport,
    load_real_vs_sim_correlation_report_json,
    real_vs_sim_correlation_report_from_dict,
)


ROUTE_POLICY_SCENARIO_SET_VERSION = "gs-mapper-route-policy-scenario-set/v1"
ROUTE_POLICY_SCENARIO_SET_RUN_VERSION = "gs-mapper-route-policy-scenario-set-run/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioSpec:
    """One scene/goal/config variant in a route policy scenario set."""

    scenario_id: str
    scene_catalog: str
    scene_id: str | None = None
    goal_suite_path: str | None = None
    episode_count: int | None = None
    seed_start: int | None = None
    max_steps: int | None = None
    site_url: str | None = None
    thresholds: RoutePolicyQualityThresholds | None = None
    sensor_noise_profile_path: str | None = None
    raw_sensor_noise_profile_path: str | None = None
    dynamic_obstacles_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.scenario_id):
            raise ValueError("scenario_id must not be empty")
        if not str(self.scene_catalog):
            raise ValueError("scene_catalog must not be empty")
        if self.episode_count is not None:
            _positive_int(self.episode_count, "episode_count")
        if self.seed_start is not None:
            _non_negative_int(self.seed_start, "seed_start")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "max_steps")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario",
            "scenarioId": self.scenario_id,
            "sceneCatalog": self.scene_catalog,
            "sceneId": self.scene_id,
            "goalSuitePath": self.goal_suite_path,
            "episodeCount": self.episode_count,
            "seedStart": self.seed_start,
            "maxSteps": self.max_steps,
            "siteUrl": self.site_url,
            "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
            "sensorNoiseProfilePath": self.sensor_noise_profile_path,
            "rawSensorNoiseProfilePath": self.raw_sensor_noise_profile_path,
            "dynamicObstaclesPath": self.dynamic_obstacles_path,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioSet:
    """Versioned set of route policy benchmark scenarios sharing one policy registry."""

    scenario_set_id: str
    scenarios: tuple[RoutePolicyScenarioSpec, ...]
    policy_registry_path: str | None = None
    episode_count: int = 16
    seed_start: int = 100
    max_steps: int | None = None
    include_direct_baseline: bool = False
    site_url: str = DEFAULT_SITE_URL
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_SET_VERSION

    def __post_init__(self) -> None:
        if not str(self.scenario_set_id):
            raise ValueError("scenario_set_id must not be empty")
        if not self.scenarios:
            raise ValueError("scenario set must contain at least one scenario")
        scenario_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario set must not contain duplicate scenario ids")
        _positive_int(self.episode_count, "episode_count")
        _non_negative_int(self.seed_start, "seed_start")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "max_steps")

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-set",
            "version": self.version,
            "scenarioSetId": self.scenario_set_id,
            "scenarioCount": self.scenario_count,
            "policyRegistryPath": self.policy_registry_path,
            "episodeCount": self.episode_count,
            "seedStart": self.seed_start,
            "maxSteps": self.max_steps,
            "includeDirectBaseline": self.include_direct_baseline,
            "siteUrl": self.site_url,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioRunResult:
    """Artifact paths and compact status for one executed scenario."""

    scenario_id: str
    benchmark_id: str
    report_path: str
    markdown_path: str | None
    passed: bool
    best_policy_name: str | None
    scene_catalog: str
    scene_id: str
    goal_suite_path: str | None
    episode_count: int
    seed_start: int
    max_steps: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-run-result",
            "scenarioId": self.scenario_id,
            "benchmarkId": self.benchmark_id,
            "reportPath": self.report_path,
            "markdownPath": self.markdown_path,
            "passed": self.passed,
            "bestPolicyName": self.best_policy_name,
            "sceneCatalog": self.scene_catalog,
            "sceneId": self.scene_id,
            "goalSuitePath": self.goal_suite_path,
            "episodeCount": self.episode_count,
            "seedStart": self.seed_start,
            "maxSteps": self.max_steps,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioSetRunReport:
    """Top-level report for a scenario-set execution plus its history gate.

    Optional ``correlation_reports`` carry pre-computed real-vs-sim
    correlation reports (one per source rosbag) so scenario CI artifacts
    can show drift between the headless rollout and the recorded bag
    next to the benchmark history. Each entry is paired with the JSON
    path it was loaded from in ``correlation_report_paths`` so the
    review bundle can hyperlink back to the full report.
    """

    scenario_set_id: str
    scenario_results: tuple[RoutePolicyScenarioRunResult, ...]
    history: RoutePolicyBenchmarkHistoryReport
    policy_registry_path: str
    history_path: str | None = None
    history_markdown_path: str | None = None
    correlation_reports: tuple[RealVsSimCorrelationReport, ...] = ()
    correlation_report_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_SET_RUN_VERSION

    def __post_init__(self) -> None:
        reports = tuple(self.correlation_reports)
        paths = tuple(str(item) for item in self.correlation_report_paths)
        if paths and len(paths) != len(reports):
            raise ValueError("correlation_report_paths must have the same length as correlation_reports when provided")
        object.__setattr__(self, "correlation_reports", reports)
        object.__setattr__(self, "correlation_report_paths", paths)

    @property
    def passed(self) -> bool:
        return self.history.passed and all(result.passed for result in self.scenario_results)

    @property
    def scenario_count(self) -> int:
        return len(self.scenario_results)

    @property
    def correlation_report_count(self) -> int:
        return len(self.correlation_reports)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-scenario-set-run",
            "version": self.version,
            "scenarioSetId": self.scenario_set_id,
            "passed": self.passed,
            "scenarioCount": self.scenario_count,
            "policyRegistryPath": self.policy_registry_path,
            "historyPath": self.history_path,
            "historyMarkdownPath": self.history_markdown_path,
            "scenarioResults": [result.to_dict() for result in self.scenario_results],
            "history": self.history.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }
        if self.correlation_reports:
            payload["correlationReports"] = [report.to_dict() for report in self.correlation_reports]
            if self.correlation_report_paths:
                payload["correlationReportPaths"] = list(self.correlation_report_paths)
        return payload


def run_route_policy_scenario_set(
    scenario_set: RoutePolicyScenarioSet,
    registry: RoutePolicyRegistry,
    *,
    report_dir: str | Path,
    scenario_set_base_path: str | Path | None = None,
    registry_base_path: str | Path | None = None,
    policy_registry_path: str | Path | None = None,
    baseline_report: str | Path | None = None,
    history_output: str | Path | None = None,
    history_markdown_output: str | Path | None = None,
    history_thresholds: RoutePolicyBenchmarkRegressionThresholds | None = None,
    write_markdown: bool = True,
    correlation_report_paths: Sequence[str | Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioSetRunReport:
    """Run one policy registry across every scenario in a scenario-set artifact.

    ``correlation_report_paths`` is an optional sequence of pre-computed
    real-vs-sim correlation report JSON paths (typically produced by
    ``scripts/run_rosbag_correlation.py``). Each is loaded and attached
    to the resulting :class:`RoutePolicyScenarioSetRunReport` so the
    scenario-set run JSON / Markdown / review bundle can show the
    headless-vs-bag drift alongside the benchmark history.
    """

    base_path = Path(scenario_set_base_path) if scenario_set_base_path is not None else Path(".")
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_results: list[RoutePolicyScenarioRunResult] = []
    report_paths: list[Path] = []
    for scenario in scenario_set.scenarios:
        result = _run_scenario(
            scenario_set,
            scenario,
            registry,
            base_path=base_path,
            registry_base_path=registry_base_path,
            output_dir=output_dir,
            write_markdown=write_markdown,
        )
        scenario_results.append(result)
        report_paths.append(Path(result.report_path))

    history = build_route_policy_benchmark_history(
        tuple(report_paths),
        baseline_report=baseline_report,
        history_id=f"{scenario_set.scenario_set_id}-history",
        thresholds=history_thresholds,
        metadata={
            "scenarioSetId": scenario_set.scenario_set_id,
            "policyRegistryPath": None if policy_registry_path is None else str(policy_registry_path),
        },
    )
    if history_output is not None:
        write_route_policy_benchmark_history_json(history_output, history)
    if history_markdown_output is not None:
        markdown_path = Path(history_markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_route_policy_benchmark_history_markdown(history), encoding="utf-8")
    correlation_paths_resolved: tuple[str, ...] = ()
    correlation_reports: tuple[RealVsSimCorrelationReport, ...] = ()
    if correlation_report_paths:
        loaded = []
        kept_paths: list[str] = []
        for raw_path in correlation_report_paths:
            resolved = _resolve_path(base_path, raw_path)
            loaded.append(load_real_vs_sim_correlation_report_json(resolved))
            kept_paths.append(str(resolved))
        correlation_reports = tuple(loaded)
        correlation_paths_resolved = tuple(kept_paths)
    return RoutePolicyScenarioSetRunReport(
        scenario_set_id=scenario_set.scenario_set_id,
        scenario_results=tuple(scenario_results),
        history=history,
        policy_registry_path="" if policy_registry_path is None else str(policy_registry_path),
        history_path=None if history_output is None else str(history_output),
        history_markdown_path=None if history_markdown_output is None else str(history_markdown_output),
        correlation_reports=correlation_reports,
        correlation_report_paths=correlation_paths_resolved,
        metadata={
            "registryId": registry.registry_id,
            **_json_mapping(metadata or {}),
        },
    )


def write_route_policy_scenario_set_json(path: str | Path, scenario_set: RoutePolicyScenarioSet) -> Path:
    """Write a route policy scenario-set JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scenario_set.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_set_json(path: str | Path) -> RoutePolicyScenarioSet:
    """Load a route policy scenario-set JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_set_from_dict(_mapping(payload, "scenarioSet"))


def write_route_policy_scenario_set_run_json(path: str | Path, report: RoutePolicyScenarioSetRunReport) -> Path:
    """Write a scenario-set run report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_set_run_json(path: str | Path) -> RoutePolicyScenarioSetRunReport:
    """Load a route policy scenario-set run JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_set_run_from_dict(_mapping(payload, "scenarioSetRun"))


def route_policy_scenario_set_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioSet:
    """Rebuild a scenario set from its JSON payload."""

    _record_type(payload, "route-policy-scenario-set")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_SET_VERSION))
    if version != ROUTE_POLICY_SCENARIO_SET_VERSION:
        raise ValueError(f"unsupported route policy scenario-set version: {version}")
    scenarios = tuple(
        route_policy_scenario_spec_from_dict(_mapping(item, "scenario"))
        for item in _sequence(payload.get("scenarios", ()), "scenarios")
    )
    expected_scenario_count = payload.get("scenarioCount")
    if expected_scenario_count is not None and int(expected_scenario_count) != len(scenarios):
        raise ValueError("scenarioCount does not match loaded scenarios")
    return RoutePolicyScenarioSet(
        scenario_set_id=str(payload["scenarioSetId"]),
        scenarios=scenarios,
        policy_registry_path=None if payload.get("policyRegistryPath") is None else str(payload["policyRegistryPath"]),
        episode_count=int(payload.get("episodeCount", 16)),
        seed_start=int(payload.get("seedStart", 100)),
        max_steps=None if payload.get("maxSteps") is None else int(payload["maxSteps"]),
        include_direct_baseline=bool(payload.get("includeDirectBaseline", False)),
        site_url=DEFAULT_SITE_URL if payload.get("siteUrl") is None else str(payload["siteUrl"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_scenario_spec_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioSpec:
    """Rebuild one scenario spec from JSON."""

    _record_type(payload, "route-policy-scenario")
    thresholds_payload = payload.get("thresholds")
    return RoutePolicyScenarioSpec(
        scenario_id=str(payload["scenarioId"]),
        scene_catalog=str(payload["sceneCatalog"]),
        scene_id=None if payload.get("sceneId") is None else str(payload["sceneId"]),
        goal_suite_path=None if payload.get("goalSuitePath") is None else str(payload["goalSuitePath"]),
        episode_count=None if payload.get("episodeCount") is None else int(payload["episodeCount"]),
        seed_start=None if payload.get("seedStart") is None else int(payload["seedStart"]),
        max_steps=None if payload.get("maxSteps") is None else int(payload["maxSteps"]),
        site_url=None if payload.get("siteUrl") is None else str(payload["siteUrl"]),
        thresholds=None
        if thresholds_payload is None
        else _quality_thresholds_from_dict(_mapping(thresholds_payload, "thresholds")),
        sensor_noise_profile_path=None
        if payload.get("sensorNoiseProfilePath") is None
        else str(payload["sensorNoiseProfilePath"]),
        raw_sensor_noise_profile_path=None
        if payload.get("rawSensorNoiseProfilePath") is None
        else str(payload["rawSensorNoiseProfilePath"]),
        dynamic_obstacles_path=None
        if payload.get("dynamicObstaclesPath") is None
        else str(payload["dynamicObstaclesPath"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_set_run_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioSetRunReport:
    """Rebuild a scenario-set run report from JSON."""

    _record_type(payload, "route-policy-scenario-set-run")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_SET_RUN_VERSION))
    if version != ROUTE_POLICY_SCENARIO_SET_RUN_VERSION:
        raise ValueError(f"unsupported route policy scenario-set run version: {version}")
    correlation_reports = tuple(
        real_vs_sim_correlation_report_from_dict(_mapping(item, "correlationReport"))
        for item in _sequence(payload.get("correlationReports", ()), "correlationReports")
    )
    correlation_report_paths = tuple(
        str(item) for item in _sequence(payload.get("correlationReportPaths", ()), "correlationReportPaths")
    )
    return RoutePolicyScenarioSetRunReport(
        scenario_set_id=str(payload["scenarioSetId"]),
        scenario_results=tuple(
            route_policy_scenario_run_result_from_dict(_mapping(item, "scenarioResult"))
            for item in _sequence(payload.get("scenarioResults", ()), "scenarioResults")
        ),
        history=route_policy_benchmark_history_from_dict(_mapping(payload["history"], "history")),
        policy_registry_path=str(payload["policyRegistryPath"]),
        history_path=None if payload.get("historyPath") is None else str(payload["historyPath"]),
        history_markdown_path=None
        if payload.get("historyMarkdownPath") is None
        else str(payload["historyMarkdownPath"]),
        correlation_reports=correlation_reports,
        correlation_report_paths=correlation_report_paths,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_scenario_run_result_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioRunResult:
    """Rebuild one scenario run result from JSON."""

    _record_type(payload, "route-policy-scenario-run-result")
    return RoutePolicyScenarioRunResult(
        scenario_id=str(payload["scenarioId"]),
        benchmark_id=str(payload["benchmarkId"]),
        report_path=str(payload["reportPath"]),
        markdown_path=None if payload.get("markdownPath") is None else str(payload["markdownPath"]),
        passed=bool(payload.get("passed", False)),
        best_policy_name=None if payload.get("bestPolicyName") is None else str(payload["bestPolicyName"]),
        scene_catalog=str(payload["sceneCatalog"]),
        scene_id=str(payload["sceneId"]),
        goal_suite_path=None if payload.get("goalSuitePath") is None else str(payload["goalSuitePath"]),
        episode_count=int(payload["episodeCount"]),
        seed_start=int(payload["seedStart"]),
        max_steps=None if payload.get("maxSteps") is None else int(payload["maxSteps"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def render_route_policy_scenario_set_markdown(report: RoutePolicyScenarioSetRunReport) -> str:
    """Render a compact Markdown summary for a scenario-set run."""

    lines = [
        f"# Route Policy Scenario Set: {report.scenario_set_id}",
        f"- Status: {'PASS' if report.passed else 'FAIL'}",
        f"- Scenarios: {report.scenario_count}",
        f"- History: {'PASS' if report.history.passed else 'FAIL'}",
        "",
        "| Scenario | Pass | Best policy | Scene | Episodes | Report |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in report.scenario_results:
        lines.append(
            "| "
            f"{result.scenario_id} | "
            f"{'yes' if result.passed else 'no'} | "
            f"{result.best_policy_name or 'n/a'} | "
            f"{result.scene_id} | "
            f"{result.episode_count} | "
            f"{result.report_path} |"
        )
    if report.correlation_reports:
        lines.extend(
            [
                "",
                "## Real-vs-sim correlation",
                "",
                "| Bag topic | Matched pairs | Translation mean (m) | Translation p95 (m) | Translation max (m) | Heading mean (rad) | Report |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for index, correlation in enumerate(report.correlation_reports):
            bag = correlation.bag_source
            heading = (
                f"{correlation.heading_error_mean_radians:.4f}"
                if correlation.heading_error_mean_radians is not None
                else "n/a"
            )
            path_cell = (
                report.correlation_report_paths[index] if index < len(report.correlation_report_paths) else "n/a"
            )
            lines.append(
                "| "
                f"`{bag.source_topic}` | "
                f"{correlation.matched_pair_count} | "
                f"{correlation.translation_error_mean_meters:.4f} | "
                f"{correlation.translation_error_p95_meters:.4f} | "
                f"{correlation.translation_error_max_meters:.4f} | "
                f"{heading} | "
                f"{path_cell} |"
            )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy scenario-set CLI."""

    scenario_set_path = Path(getattr(args, "scenario_set"))
    scenario_set = load_route_policy_scenario_set_json(scenario_set_path)
    scenario_set_base_path = scenario_set_path.parent
    registry_path = _resolve_policy_registry_path(args, scenario_set, scenario_set_base_path)
    registry = load_route_policy_registry_json(registry_path)
    history_thresholds = RoutePolicyBenchmarkRegressionThresholds(
        max_success_rate_drop=float(getattr(args, "max_success_rate_drop")),
        max_collision_rate_increase=float(getattr(args, "max_collision_rate_increase")),
        max_truncation_rate_increase=float(getattr(args, "max_truncation_rate_increase")),
        max_mean_reward_drop=getattr(args, "max_mean_reward_drop", None),
        require_baseline_policies=not bool(getattr(args, "allow_missing_policies", False)),
        fail_on_report_failure=not bool(getattr(args, "allow_report_failures", False)),
    )
    correlation_report_paths = tuple(getattr(args, "correlation_report", None) or ())
    report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=getattr(args, "report_dir"),
        scenario_set_base_path=scenario_set_base_path,
        registry_base_path=registry_path.parent,
        policy_registry_path=registry_path,
        baseline_report=getattr(args, "baseline_report", None),
        history_output=getattr(args, "history_output"),
        history_markdown_output=getattr(args, "history_markdown_output", None),
        history_thresholds=history_thresholds,
        write_markdown=not bool(getattr(args, "no_markdown", False)),
        correlation_report_paths=correlation_report_paths,
        metadata={
            "scenarioSetPath": str(scenario_set_path),
            "policyRegistryPath": str(registry_path),
        },
    )
    write_route_policy_scenario_set_run_json(getattr(args, "output"), report)
    summary = render_route_policy_scenario_set_markdown(report)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Scenario-set run saved to: {getattr(args, 'output')}")
    if bool(getattr(args, "fail_on_regression", False)) and not report.passed:
        raise SystemExit(2)


def _run_scenario(
    scenario_set: RoutePolicyScenarioSet,
    scenario: RoutePolicyScenarioSpec,
    registry: RoutePolicyRegistry,
    *,
    base_path: Path,
    registry_base_path: str | Path | None,
    output_dir: Path,
    write_markdown: bool,
) -> RoutePolicyScenarioRunResult:
    scene_catalog_path = _resolve_path(base_path, scenario.scene_catalog)
    goal_suite_path = None if scenario.goal_suite_path is None else _resolve_path(base_path, scenario.goal_suite_path)
    site_url = scenario.site_url or scenario_set.site_url
    catalog = load_simulation_catalog_from_scene_picker(scene_catalog_path, site_url=site_url)
    goal_suite = load_route_policy_goal_suite_json(goal_suite_path) if goal_suite_path is not None else None
    scene_id = _resolve_scene_id(catalog, scenario.scene_id, goal_suite=goal_suite)
    max_steps = scenario.max_steps if scenario.max_steps is not None else scenario_set.max_steps
    sensor_noise_profile_path = (
        None
        if scenario.sensor_noise_profile_path is None
        else _resolve_path(base_path, scenario.sensor_noise_profile_path)
    )
    sensor_noise_profile = (
        None
        if sensor_noise_profile_path is None
        else load_route_policy_sensor_noise_profile_json(sensor_noise_profile_path)
    )
    raw_sensor_noise_profile_path = (
        None
        if scenario.raw_sensor_noise_profile_path is None
        else _resolve_path(base_path, scenario.raw_sensor_noise_profile_path)
    )
    raw_sensor_noise_profile = (
        None
        if raw_sensor_noise_profile_path is None
        else load_raw_sensor_noise_profile_json(raw_sensor_noise_profile_path)
    )
    dynamic_obstacles_path = (
        None if scenario.dynamic_obstacles_path is None else _resolve_path(base_path, scenario.dynamic_obstacles_path)
    )
    dynamic_obstacles = (
        None
        if dynamic_obstacles_path is None
        else load_route_policy_dynamic_obstacle_timeline_json(dynamic_obstacles_path)
    )
    if dynamic_obstacles is None:
        # Sprint 4 / PR D3: when a scenario carries multi-agent metadata
        # (embedded by the matrix expander in PR D2 via agents /
        # population / populationSeed), synthesize a peer roster on the
        # fly so the headless env's existing dynamic_obstacles plumbing
        # drives the peers. An explicit dynamic_obstacles_path still
        # wins, so legacy scenario JSONs are untouched.
        dynamic_obstacles = synthesize_peer_roster_from_scenario_metadata(
            scenario.metadata,
            timeline_id=f"{scenario.scenario_id}-peer-roster",
        )
    adapter = RoutePolicyGymAdapter(
        HeadlessPhysicalAIEnvironment(
            catalog,
            dynamic_obstacles=dynamic_obstacles,
            raw_sensor_noise_profile=raw_sensor_noise_profile,
        ),
        RoutePolicyEnvConfig(
            scene_id=scene_id,
            max_steps=max_steps or RoutePolicyEnvConfig().max_steps,
            sensor_noise_profile=sensor_noise_profile,
        ),
    )
    goals = _goals_from_goal_suite(catalog, scene_id, goal_suite)
    episode_count = scenario.episode_count if scenario.episode_count is not None else scenario_set.episode_count
    seed_start = scenario.seed_start if scenario.seed_start is not None else scenario_set.seed_start
    benchmark_id = f"{scenario_set.scenario_set_id}-{scenario.scenario_id}"
    report = run_route_policy_registry_benchmark(
        (adapter,),
        registry,
        episode_count=episode_count,
        benchmark_id=benchmark_id,
        registry_base_path=registry_base_path,
        include_direct_baseline=scenario_set.include_direct_baseline,
        seed_start=seed_start,
        goals=goals,
        max_steps=max_steps,
        thresholds=scenario.thresholds,
        metadata={
            "scenarioSetId": scenario_set.scenario_set_id,
            "scenarioId": scenario.scenario_id,
            "sceneCatalog": scene_catalog_path.as_posix(),
            "sceneId": scene_id,
            "goalSuitePath": None if goal_suite_path is None else goal_suite_path.as_posix(),
            **_json_mapping(scenario.metadata),
        },
    )
    slug = _slug(scenario.scenario_id)
    report_path = output_dir / f"{slug}.json"
    write_route_policy_benchmark_report_json(report_path, report)
    markdown_path = output_dir / f"{slug}.md" if write_markdown else None
    if markdown_path is not None:
        markdown_path.write_text(render_route_policy_benchmark_markdown(report), encoding="utf-8")
    result_metadata = _json_mapping(scenario.metadata)
    if dynamic_obstacles is not None:
        interaction_values: dict[str, float] = {
            "peer-count": float(dynamic_obstacles.obstacle_count),
        }
        min_peer_separation = _min_peer_separation_from_report(report)
        if min_peer_separation is not None:
            interaction_values["min-peer-separation-meters"] = min_peer_separation
        result_metadata[SCENARIO_INTERACTION_METRIC_VALUES_KEY] = interaction_values
    return RoutePolicyScenarioRunResult(
        scenario_id=scenario.scenario_id,
        benchmark_id=benchmark_id,
        report_path=report_path.as_posix(),
        markdown_path=None if markdown_path is None else markdown_path.as_posix(),
        passed=report.passed,
        best_policy_name=report.best_policy_name,
        scene_catalog=scene_catalog_path.as_posix(),
        scene_id=scene_id,
        goal_suite_path=None if goal_suite_path is None else goal_suite_path.as_posix(),
        episode_count=episode_count,
        seed_start=seed_start,
        max_steps=max_steps,
        metadata=result_metadata,
    )


NEAREST_PEER_CLEARANCE_FEATURE_KEY = "nearest-dynamic-obstacle-distance-meters"


def _min_peer_separation_from_report(report: Any) -> float | None:
    """Return the worst ego-to-peer clearance observed across the run.

    Walks every transition in every baseline result and tracks the
    minimum of the ``nearest-dynamic-obstacle-distance-meters`` feature
    surfaced by :class:`RoutePolicyGymAdapter`. Returns ``None`` when no
    transition surfaced the metric (e.g. an ego-only run that slipped
    through without any peers, or a rollout that terminated before the
    obstacle block was populated).
    """

    worst: float | None = None
    for result in report.evaluation.results:
        for episode in result.dataset.episodes:
            for transition in episode.transitions:
                value = transition.observation.get(NEAREST_PEER_CLEARANCE_FEATURE_KEY)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric != numeric or numeric in (float("inf"), float("-inf")):
                    continue
                if worst is None or numeric < worst:
                    worst = numeric
    return worst


def _resolve_scene_id(
    catalog: SimulationCatalog,
    scene_id: str | None,
    *,
    goal_suite: RoutePolicyGoalSuite | None,
) -> str:
    if scene_id:
        catalog.scene_by_id(scene_id)
        return scene_id
    if goal_suite is not None and goal_suite.scene_id:
        catalog.scene_by_id(goal_suite.scene_id)
        return goal_suite.scene_id
    if "outdoor-demo" in catalog.scene_ids():
        return "outdoor-demo"
    scene_ids = catalog.scene_ids()
    if not scene_ids:
        raise ValueError("simulation catalog must contain at least one scene")
    return scene_ids[0]


def _goals_from_goal_suite(
    catalog: SimulationCatalog,
    scene_id: str,
    goal_suite: RoutePolicyGoalSuite | None,
) -> tuple[Pose3D, ...] | None:
    if goal_suite is None:
        return None
    frame_id = catalog.scene_by_id(scene_id).coordinate_frame.frame_id
    return goal_suite.to_goals(frame_id=frame_id)


def _resolve_policy_registry_path(args: Any, scenario_set: RoutePolicyScenarioSet, base_path: Path) -> Path:
    registry_arg = getattr(args, "policy_registry", None)
    if registry_arg:
        return Path(registry_arg)
    if scenario_set.policy_registry_path:
        return _resolve_path(base_path, scenario_set.policy_registry_path)
    raise ValueError("provide --policy-registry or policyRegistryPath in the scenario set")


def _quality_thresholds_from_dict(payload: Mapping[str, Any]) -> RoutePolicyQualityThresholds:
    return RoutePolicyQualityThresholds(
        min_success_rate=float(payload.get("minSuccessRate", 0.8)),
        max_collision_rate=float(payload.get("maxCollisionRate", 0.05)),
        max_truncation_rate=float(payload.get("maxTruncationRate", 0.1)),
        min_mean_reward=None if payload.get("minMeanReward") is None else float(payload["minMeanReward"]),
        min_scene_count=int(payload.get("minSceneCount", 1)),
        min_episode_count=int(payload.get("minEpisodeCount", 1)),
        min_transition_count=int(payload.get("minTransitionCount", 1)),
    )


def _resolve_path(base_path: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_path / path


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


def _non_negative_int(value: int, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "unnamed"
