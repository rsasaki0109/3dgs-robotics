"""CI matrix manifests for route policy scenario shard execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .policy_scenario_sharding import (
    RoutePolicyScenarioShardPlan,
    load_route_policy_scenario_shard_plan_json,
)


ROUTE_POLICY_SCENARIO_CI_MANIFEST_VERSION = "gs-mapper-route-policy-scenario-ci-manifest/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIShardJob:
    """One CI job specification for running a shard scenario set."""

    job_id: str
    shard_id: str
    source_scenario_set_id: str
    scenario_set_path: str
    scenario_count: int
    report_dir: str
    run_output: str
    history_output: str
    cache_key: str
    expected_report_paths: tuple[str, ...]
    policy_registry_path: str | None = None
    markdown_output: str | None = None
    history_markdown_output: str | None = None
    merge_job_id: str | None = None
    command: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.job_id):
            raise ValueError("job_id must not be empty")
        if not str(self.shard_id):
            raise ValueError("shard_id must not be empty")
        if not str(self.source_scenario_set_id):
            raise ValueError("source_scenario_set_id must not be empty")
        if not str(self.scenario_set_path):
            raise ValueError("scenario_set_path must not be empty")
        _positive_int(self.scenario_count, "scenario_count")
        if not str(self.report_dir):
            raise ValueError("report_dir must not be empty")
        if not str(self.run_output):
            raise ValueError("run_output must not be empty")
        if not str(self.history_output):
            raise ValueError("history_output must not be empty")
        if not str(self.cache_key):
            raise ValueError("cache_key must not be empty")
        report_paths = tuple(str(path) for path in self.expected_report_paths)
        if not report_paths:
            raise ValueError("expected_report_paths must not be empty")
        command = tuple(str(part) for part in self.command)
        if not command:
            raise ValueError("command must not be empty")
        object.__setattr__(self, "expected_report_paths", report_paths)
        object.__setattr__(self, "command", command)

    def to_matrix_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "shardId": self.shard_id,
            "scenarioSetPath": self.scenario_set_path,
            "scenarioCount": self.scenario_count,
            "reportDir": self.report_dir,
            "runOutput": self.run_output,
            "historyOutput": self.history_output,
            "cacheKey": self.cache_key,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-shard-job",
            "jobId": self.job_id,
            "shardId": self.shard_id,
            "sourceScenarioSetId": self.source_scenario_set_id,
            "scenarioSetPath": self.scenario_set_path,
            "policyRegistryPath": self.policy_registry_path,
            "scenarioCount": self.scenario_count,
            "reportDir": self.report_dir,
            "runOutput": self.run_output,
            "markdownOutput": self.markdown_output,
            "historyOutput": self.history_output,
            "historyMarkdownOutput": self.history_markdown_output,
            "cacheKey": self.cache_key,
            "mergeJobId": self.merge_job_id,
            "expectedReportPaths": list(self.expected_report_paths),
            "command": list(self.command),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIMergeJob:
    """CI job specification for merging completed shard runs."""

    job_id: str
    merge_id: str
    run_inputs: tuple[str, ...]
    output: str
    history_output: str
    cache_key: str
    depends_on: tuple[str, ...]
    markdown_output: str | None = None
    history_markdown_output: str | None = None
    command: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.job_id):
            raise ValueError("job_id must not be empty")
        if not str(self.merge_id):
            raise ValueError("merge_id must not be empty")
        run_inputs = tuple(str(path) for path in self.run_inputs)
        if not run_inputs:
            raise ValueError("run_inputs must not be empty")
        depends_on = tuple(str(job_id) for job_id in self.depends_on)
        if not depends_on:
            raise ValueError("depends_on must not be empty")
        if not str(self.output):
            raise ValueError("output must not be empty")
        if not str(self.history_output):
            raise ValueError("history_output must not be empty")
        if not str(self.cache_key):
            raise ValueError("cache_key must not be empty")
        command = tuple(str(part) for part in self.command)
        if not command:
            raise ValueError("command must not be empty")
        object.__setattr__(self, "run_inputs", run_inputs)
        object.__setattr__(self, "depends_on", depends_on)
        object.__setattr__(self, "command", command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-merge-job",
            "jobId": self.job_id,
            "mergeId": self.merge_id,
            "runInputs": list(self.run_inputs),
            "output": self.output,
            "markdownOutput": self.markdown_output,
            "historyOutput": self.history_output,
            "historyMarkdownOutput": self.history_markdown_output,
            "cacheKey": self.cache_key,
            "dependsOn": list(self.depends_on),
            "command": list(self.command),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIManifest:
    """Versioned CI matrix manifest for scenario shard execution."""

    manifest_id: str
    shard_plan_id: str
    shard_jobs: tuple[RoutePolicyScenarioCIShardJob, ...]
    merge_job: RoutePolicyScenarioCIMergeJob
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.shard_plan_id):
            raise ValueError("shard_plan_id must not be empty")
        if not self.shard_jobs:
            raise ValueError("CI manifest must contain at least one shard job")
        job_ids = tuple(job.job_id for job in self.shard_jobs)
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("CI manifest must not contain duplicate shard job ids")
        shard_ids = tuple(job.shard_id for job in self.shard_jobs)
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError("CI manifest must not contain duplicate shard ids")
        missing_dependencies = set(self.merge_job.depends_on).difference(job_ids)
        if missing_dependencies:
            raise ValueError(f"merge job depends on unknown shard jobs: {sorted(missing_dependencies)}")

    @property
    def shard_job_count(self) -> int:
        return len(self.shard_jobs)

    @property
    def scenario_count(self) -> int:
        return sum(job.scenario_count for job in self.shard_jobs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-manifest",
            "version": self.version,
            "manifestId": self.manifest_id,
            "shardPlanId": self.shard_plan_id,
            "shardJobCount": self.shard_job_count,
            "scenarioCount": self.scenario_count,
            "matrix": {"include": [job.to_matrix_dict() for job in self.shard_jobs]},
            "shardJobs": [job.to_dict() for job in self.shard_jobs],
            "mergeJob": self.merge_job.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }


def build_route_policy_scenario_ci_manifest(
    shard_plan: RoutePolicyScenarioShardPlan,
    *,
    manifest_id: str | None = None,
    report_dir: str | Path = "outputs/route_policy_scenarios/shard_reports",
    run_output_dir: str | Path = "outputs/route_policy_scenarios/shard_runs",
    history_output_dir: str | Path = "outputs/route_policy_scenarios/shard_histories",
    merge_id: str = "route-policy-scenario-shard-merge",
    merge_output: str | Path = "outputs/route_policy_scenarios/scenario_shard_merge.json",
    merge_history_output: str | Path = "outputs/route_policy_scenarios/shard_history.json",
    include_markdown: bool = False,
    merge_markdown_output: str | Path | None = None,
    merge_history_markdown_output: str | Path | None = None,
    cache_key_prefix: str = "route-policy-scenario",
    fail_on_regression: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioCIManifest:
    """Build a CI manifest from a shard plan."""

    if not shard_plan.shards:
        raise ValueError("shard_plan must contain at least one shard")
    resolved_manifest_id = manifest_id or f"{shard_plan.shard_plan_id}-ci"
    merge_job_id = _slug(merge_id)
    report_base = Path(report_dir)
    run_base = Path(run_output_dir)
    history_base = Path(history_output_dir)
    cache_plan_slug = _slug(shard_plan.shard_plan_id)

    shard_jobs: list[RoutePolicyScenarioCIShardJob] = []
    for shard in shard_plan.shards:
        if shard.scenario_set_path is None:
            raise ValueError(f"shard {shard.shard_id} must have scenarioSetPath for CI manifests")
        shard_slug = _slug(shard.shard_id)
        job_id = f"scenario-{shard_slug}"
        shard_report_dir = (report_base / shard_slug).as_posix()
        run_output = (run_base / f"{shard_slug}.json").as_posix()
        markdown_output = (run_base / f"{shard_slug}.md").as_posix() if include_markdown else None
        history_output = (history_base / f"{shard_slug}.json").as_posix()
        history_markdown_output = (history_base / f"{shard_slug}.md").as_posix() if include_markdown else None
        expected_report_paths = tuple(
            (Path(shard_report_dir) / f"{_slug(scenario_id)}.json").as_posix() for scenario_id in shard.scenario_ids
        )
        command = _scenario_set_command(
            scenario_set_path=shard.scenario_set_path,
            report_dir=shard_report_dir,
            run_output=run_output,
            markdown_output=markdown_output,
            history_output=history_output,
            history_markdown_output=history_markdown_output,
            fail_on_regression=fail_on_regression,
        )
        shard_jobs.append(
            RoutePolicyScenarioCIShardJob(
                job_id=job_id,
                shard_id=shard.shard_id,
                source_scenario_set_id=shard.source_scenario_set_id,
                scenario_set_path=shard.scenario_set_path,
                policy_registry_path=shard.policy_registry_path,
                scenario_count=shard.scenario_count,
                report_dir=shard_report_dir,
                run_output=run_output,
                markdown_output=markdown_output,
                history_output=history_output,
                history_markdown_output=history_markdown_output,
                cache_key=f"{cache_key_prefix}-{cache_plan_slug}-{shard_slug}",
                merge_job_id=merge_job_id,
                expected_report_paths=expected_report_paths,
                command=command,
                metadata=shard.metadata,
            )
        )

    merge_job = RoutePolicyScenarioCIMergeJob(
        job_id=merge_job_id,
        merge_id=merge_id,
        run_inputs=tuple(job.run_output for job in shard_jobs),
        output=Path(merge_output).as_posix(),
        markdown_output=None if merge_markdown_output is None else Path(merge_markdown_output).as_posix(),
        history_output=Path(merge_history_output).as_posix(),
        history_markdown_output=None
        if merge_history_markdown_output is None
        else Path(merge_history_markdown_output).as_posix(),
        cache_key=f"{cache_key_prefix}-{cache_plan_slug}-merge",
        depends_on=tuple(job.job_id for job in shard_jobs),
        command=_shard_merge_command(
            run_inputs=tuple(job.run_output for job in shard_jobs),
            merge_id=merge_id,
            output=Path(merge_output).as_posix(),
            markdown_output=None if merge_markdown_output is None else Path(merge_markdown_output).as_posix(),
            history_output=Path(merge_history_output).as_posix(),
            history_markdown_output=None
            if merge_history_markdown_output is None
            else Path(merge_history_markdown_output).as_posix(),
            fail_on_regression=fail_on_regression,
        ),
        metadata={"shardPlanId": shard_plan.shard_plan_id},
    )
    return RoutePolicyScenarioCIManifest(
        manifest_id=resolved_manifest_id,
        shard_plan_id=shard_plan.shard_plan_id,
        shard_jobs=tuple(shard_jobs),
        merge_job=merge_job,
        metadata={
            "cacheKeyPrefix": cache_key_prefix,
            "includeMarkdown": include_markdown,
            "failOnRegression": fail_on_regression,
            **_json_mapping(metadata or {}),
        },
    )


def write_route_policy_scenario_ci_manifest_json(
    path: str | Path,
    manifest: RoutePolicyScenarioCIManifest,
) -> Path:
    """Write a route policy scenario CI manifest JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_ci_manifest_json(path: str | Path) -> RoutePolicyScenarioCIManifest:
    """Load a route policy scenario CI manifest JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_ci_manifest_from_dict(_mapping(payload, "ciManifest"))


def route_policy_scenario_ci_shard_job_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioCIShardJob:
    """Rebuild one CI shard job from JSON."""

    _record_type(payload, "route-policy-scenario-ci-shard-job")
    return RoutePolicyScenarioCIShardJob(
        job_id=str(payload["jobId"]),
        shard_id=str(payload["shardId"]),
        source_scenario_set_id=str(payload["sourceScenarioSetId"]),
        scenario_set_path=str(payload["scenarioSetPath"]),
        policy_registry_path=None if payload.get("policyRegistryPath") is None else str(payload["policyRegistryPath"]),
        scenario_count=int(payload["scenarioCount"]),
        report_dir=str(payload["reportDir"]),
        run_output=str(payload["runOutput"]),
        markdown_output=None if payload.get("markdownOutput") is None else str(payload["markdownOutput"]),
        history_output=str(payload["historyOutput"]),
        history_markdown_output=None
        if payload.get("historyMarkdownOutput") is None
        else str(payload["historyMarkdownOutput"]),
        cache_key=str(payload["cacheKey"]),
        merge_job_id=None if payload.get("mergeJobId") is None else str(payload["mergeJobId"]),
        expected_report_paths=tuple(
            str(path) for path in _sequence(payload.get("expectedReportPaths", ()), "expectedReportPaths")
        ),
        command=tuple(str(part) for part in _sequence(payload.get("command", ()), "command")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_merge_job_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioCIMergeJob:
    """Rebuild a CI merge job from JSON."""

    _record_type(payload, "route-policy-scenario-ci-merge-job")
    return RoutePolicyScenarioCIMergeJob(
        job_id=str(payload["jobId"]),
        merge_id=str(payload["mergeId"]),
        run_inputs=tuple(str(path) for path in _sequence(payload.get("runInputs", ()), "runInputs")),
        output=str(payload["output"]),
        markdown_output=None if payload.get("markdownOutput") is None else str(payload["markdownOutput"]),
        history_output=str(payload["historyOutput"]),
        history_markdown_output=None
        if payload.get("historyMarkdownOutput") is None
        else str(payload["historyMarkdownOutput"]),
        cache_key=str(payload["cacheKey"]),
        depends_on=tuple(str(job_id) for job_id in _sequence(payload.get("dependsOn", ()), "dependsOn")),
        command=tuple(str(part) for part in _sequence(payload.get("command", ()), "command")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_manifest_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioCIManifest:
    """Rebuild a CI manifest from JSON."""

    _record_type(payload, "route-policy-scenario-ci-manifest")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_MANIFEST_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_MANIFEST_VERSION:
        raise ValueError(f"unsupported route policy scenario CI manifest version: {version}")
    shard_jobs = tuple(
        route_policy_scenario_ci_shard_job_from_dict(_mapping(item, "shardJob"))
        for item in _sequence(payload.get("shardJobs", ()), "shardJobs")
    )
    expected_job_count = payload.get("shardJobCount")
    if expected_job_count is not None and int(expected_job_count) != len(shard_jobs):
        raise ValueError("shardJobCount does not match loaded jobs")
    expected_scenario_count = payload.get("scenarioCount")
    if expected_scenario_count is not None:
        loaded_scenario_count = sum(job.scenario_count for job in shard_jobs)
        if int(expected_scenario_count) != loaded_scenario_count:
            raise ValueError("scenarioCount does not match loaded jobs")
    return RoutePolicyScenarioCIManifest(
        manifest_id=str(payload["manifestId"]),
        shard_plan_id=str(payload["shardPlanId"]),
        shard_jobs=shard_jobs,
        merge_job=route_policy_scenario_ci_merge_job_from_dict(_mapping(payload["mergeJob"], "mergeJob")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_scenario_ci_manifest_markdown(manifest: RoutePolicyScenarioCIManifest) -> str:
    """Render a compact Markdown summary for a scenario CI manifest."""

    lines = [
        f"# Route Policy Scenario CI Manifest: {manifest.manifest_id}",
        f"- Shard plan: {manifest.shard_plan_id}",
        f"- Shard jobs: {manifest.shard_job_count}",
        f"- Scenarios: {manifest.scenario_count}",
        f"- Merge job: {manifest.merge_job.job_id}",
        "",
        "| Job | Shard | Scenarios | Run output | Cache key |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for job in manifest.shard_jobs:
        lines.append(f"| {job.job_id} | {job.shard_id} | {job.scenario_count} | {job.run_output} | {job.cache_key} |")
    lines.extend(
        [
            "",
            "## Merge",
            "",
            "| Job | Depends on | Output | History |",
            "| --- | ---: | --- | --- |",
            "| "
            f"{manifest.merge_job.job_id} | "
            f"{len(manifest.merge_job.depends_on)} | "
            f"{manifest.merge_job.output} | "
            f"{manifest.merge_job.history_output} |",
        ]
    )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy scenario-ci-manifest CLI."""

    shard_plan = load_route_policy_scenario_shard_plan_json(getattr(args, "shard_plan"))
    manifest = build_route_policy_scenario_ci_manifest(
        shard_plan,
        manifest_id=getattr(args, "manifest_id", None),
        report_dir=getattr(args, "report_dir"),
        run_output_dir=getattr(args, "run_output_dir"),
        history_output_dir=getattr(args, "history_output_dir"),
        merge_id=str(getattr(args, "merge_id")),
        merge_output=getattr(args, "merge_output"),
        merge_history_output=getattr(args, "merge_history_output"),
        include_markdown=bool(getattr(args, "include_markdown", False)),
        merge_markdown_output=getattr(args, "merge_markdown_output", None),
        merge_history_markdown_output=getattr(args, "merge_history_markdown_output", None),
        cache_key_prefix=str(getattr(args, "cache_key_prefix")),
        fail_on_regression=bool(getattr(args, "fail_on_regression", False)),
        metadata={"shardPlanPath": getattr(args, "shard_plan")},
    )
    write_route_policy_scenario_ci_manifest_json(getattr(args, "output"), manifest)
    markdown = render_route_policy_scenario_ci_manifest_markdown(manifest)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario CI manifest saved to: {getattr(args, 'output')}")


def _scenario_set_command(
    *,
    scenario_set_path: str,
    report_dir: str,
    run_output: str,
    markdown_output: str | None,
    history_output: str,
    history_markdown_output: str | None,
    fail_on_regression: bool,
) -> tuple[str, ...]:
    command = [
        "3dgs-robotics",
        "route-policy-scenario-set",
        "--scenario-set",
        scenario_set_path,
        "--report-dir",
        report_dir,
        "--output",
        run_output,
        "--history-output",
        history_output,
    ]
    if markdown_output is not None:
        command.extend(("--markdown-output", markdown_output))
    if history_markdown_output is not None:
        command.extend(("--history-markdown-output", history_markdown_output))
    if fail_on_regression:
        command.append("--fail-on-regression")
    return tuple(command)


def _shard_merge_command(
    *,
    run_inputs: Sequence[str],
    merge_id: str,
    output: str,
    markdown_output: str | None,
    history_output: str,
    history_markdown_output: str | None,
    fail_on_regression: bool,
) -> tuple[str, ...]:
    command = ["3dgs-robotics", "route-policy-scenario-shard-merge"]
    for run_input in run_inputs:
        command.extend(("--run", run_input))
    command.extend(("--merge-id", merge_id, "--history-output", history_output, "--output", output))
    if markdown_output is not None:
        command.extend(("--markdown-output", markdown_output))
    if history_markdown_output is not None:
        command.extend(("--history-markdown-output", history_markdown_output))
    if fail_on_regression:
        command.append("--fail-on-regression")
    return tuple(command)


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
