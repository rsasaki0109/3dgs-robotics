"""Review artifacts for generated scenario CI workflow changes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
from html import escape
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .policy_scenario_ci_activation import (
    RoutePolicyScenarioCIWorkflowActivationReport,
    load_route_policy_scenario_ci_workflow_activation_json,
)
from .policy_scenario_ci_workflow import (
    RoutePolicyScenarioCIWorkflowValidationReport,
    load_route_policy_scenario_ci_workflow_validation_json,
)
from .policy_scenario_sharding import (
    RoutePolicyScenarioShardMergeReport,
    RoutePolicyScenarioShardRunSummary,
    load_route_policy_scenario_shard_merge_json,
)
from .policy_scenario_set import load_route_policy_scenario_set_run_json
from ..robotics.rosbag_correlation import (
    CorrelationEventWindow,
    RealVsSimCorrelationReport,
    RealVsSimCorrelationThresholds,
    RealVsSimCorrelationWindowStats,
    compute_per_window_correlation_stats,
    correlation_threshold_overrides_from_dict,
    correlation_threshold_overrides_to_dict,
    evaluate_real_vs_sim_correlation_thresholds,
    load_correlation_event_windows_json,
    real_vs_sim_correlation_report_from_dict,
    real_vs_sim_correlation_thresholds_from_dict,
    real_vs_sim_correlation_window_stats_from_dict,
)


ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION = "gs-mapper-route-policy-scenario-ci-review/v1"
ROUTE_POLICY_SCENARIO_CI_REVIEW_PROVENANCE_VERSION = "gs-mapper-route-policy-scenario-ci-review-provenance/v1"

_REVIEW_PROVENANCE_KINDS: frozenset[str] = frozenset({"synthetic", "production"})


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewProvenance:
    """First-class provenance attached to a scenario CI review artifact.

    Distinguishes synthetic smoke fixture bundles (e.g. the public sample at
    ``docs/reviews/smoke-route-policy-ci/``) from production benchmark runs
    so the Pages reviews index can flag each entry. All non-required fields
    default to ``None`` so callers can fill in only what they know — useful
    when a CI generator does not yet have policy / env contract versions.
    """

    kind: str  # "synthetic" | "production"
    generated_at: str  # ISO 8601, e.g. "2026-05-15T12:34:56+00:00"
    git_commit: str | None = None
    scene_id: str | None = None
    scenario_set_id: str | None = None
    matrix_hash: str | None = None
    policy_version: str | None = None
    env_contract_version: str | None = None
    correlation_threshold_profile: str | None = None
    asset_source: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_REVIEW_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in _REVIEW_PROVENANCE_KINDS:
            raise ValueError(f"kind must be one of {sorted(_REVIEW_PROVENANCE_KINDS)}; got {kind!r}")
        object.__setattr__(self, "kind", kind)
        generated_at = str(self.generated_at)
        if not generated_at:
            raise ValueError("generated_at must not be empty")
        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "extra", _json_mapping(self.extra))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-scenario-ci-review-provenance",
            "version": self.version,
            "kind": self.kind,
            "generatedAt": self.generated_at,
        }
        if self.git_commit is not None:
            payload["gitCommit"] = self.git_commit
        if self.scene_id is not None:
            payload["sceneId"] = self.scene_id
        if self.scenario_set_id is not None:
            payload["scenarioSetId"] = self.scenario_set_id
        if self.matrix_hash is not None:
            payload["matrixHash"] = self.matrix_hash
        if self.policy_version is not None:
            payload["policyVersion"] = self.policy_version
        if self.env_contract_version is not None:
            payload["envContractVersion"] = self.env_contract_version
        if self.correlation_threshold_profile is not None:
            payload["correlationThresholdProfile"] = self.correlation_threshold_profile
        if self.asset_source is not None:
            payload["assetSource"] = self.asset_source
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewShard:
    """Review-friendly summary of one executed scenario shard."""

    shard_id: str
    passed: bool
    scenario_count: int
    report_count: int
    run_path: str | None = None
    history_path: str | None = None
    report_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.shard_id):
            raise ValueError("shard_id must not be empty")
        _positive_int(self.scenario_count, "scenario_count")
        _non_negative_int(self.report_count, "report_count")
        object.__setattr__(self, "report_paths", tuple(str(path) for path in self.report_paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-review-shard",
            "shardId": self.shard_id,
            "passed": bool(self.passed),
            "scenarioCount": self.scenario_count,
            "reportCount": self.report_count,
            "runPath": self.run_path,
            "historyPath": self.history_path,
            "reportPaths": list(self.report_paths),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewAdoption:
    """Review-friendly view of a promotion-backed adoption outcome."""

    adoption_id: str
    adopted: bool
    trigger_mode: str
    adopted_active_workflow_path: str
    adopted_source_workflow_path: str
    push_branches: tuple[str, ...] = ()
    pull_request_branches: tuple[str, ...] = ()
    workflow_diff: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.adoption_id):
            raise ValueError("adoption_id must not be empty")
        if not str(self.trigger_mode):
            raise ValueError("trigger_mode must not be empty")
        if not str(self.adopted_active_workflow_path):
            raise ValueError("adopted_active_workflow_path must not be empty")
        if not str(self.adopted_source_workflow_path):
            raise ValueError("adopted_source_workflow_path must not be empty")
        object.__setattr__(self, "push_branches", tuple(str(branch) for branch in self.push_branches))
        object.__setattr__(
            self,
            "pull_request_branches",
            tuple(str(branch) for branch in self.pull_request_branches),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-review-adoption",
            "adoptionId": self.adoption_id,
            "adopted": bool(self.adopted),
            "triggerMode": self.trigger_mode,
            "adoptedActiveWorkflowPath": self.adopted_active_workflow_path,
            "adoptedSourceWorkflowPath": self.adopted_source_workflow_path,
            "pushBranches": list(self.push_branches),
            "pullRequestBranches": list(self.pull_request_branches),
            "workflowDiff": self.workflow_diff,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewArtifact:
    """Static review artifact for scenario CI workflow publication."""

    review_id: str
    merge_id: str
    workflow_id: str
    manifest_id: str
    validation_id: str
    activation_id: str
    validation_passed: bool
    activation_activated: bool
    shard_merge_passed: bool
    history_passed: bool
    active_workflow_path: str
    source_workflow_path: str
    shards: tuple[RoutePolicyScenarioCIReviewShard, ...]
    history_failed_checks: tuple[str, ...] = ()
    adoption: RoutePolicyScenarioCIReviewAdoption | None = None
    provenance: RoutePolicyScenarioCIReviewProvenance | None = None
    correlation_reports: tuple[RealVsSimCorrelationReport, ...] = ()
    correlation_report_paths: tuple[str, ...] = ()
    correlation_thresholds: RealVsSimCorrelationThresholds | None = None
    correlation_threshold_overrides: Mapping[str, RealVsSimCorrelationThresholds] = field(default_factory=dict)
    correlation_failed_reports: tuple[tuple[int, str, tuple[str, ...]], ...] = ()
    correlation_per_window_stats: tuple[tuple[int, str, tuple[RealVsSimCorrelationWindowStats, ...]], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION

    def __post_init__(self) -> None:
        if not str(self.review_id):
            raise ValueError("review_id must not be empty")
        if not str(self.merge_id):
            raise ValueError("merge_id must not be empty")
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.validation_id):
            raise ValueError("validation_id must not be empty")
        if not str(self.activation_id):
            raise ValueError("activation_id must not be empty")
        if not str(self.active_workflow_path):
            raise ValueError("active_workflow_path must not be empty")
        if not str(self.source_workflow_path):
            raise ValueError("source_workflow_path must not be empty")
        if not self.shards:
            raise ValueError("review artifact must contain at least one shard")
        object.__setattr__(self, "history_failed_checks", tuple(str(check) for check in self.history_failed_checks))
        reports = tuple(self.correlation_reports)
        paths = tuple(str(item) for item in self.correlation_report_paths)
        if paths and len(paths) != len(reports):
            raise ValueError("correlation_report_paths must have the same length as correlation_reports when provided")
        object.__setattr__(self, "correlation_reports", reports)
        object.__setattr__(self, "correlation_report_paths", paths)
        normalised_failures: tuple[tuple[int, str, tuple[str, ...]], ...] = tuple(
            (int(index), str(topic), tuple(str(check) for check in checks))
            for index, topic, checks in self.correlation_failed_reports
        )
        object.__setattr__(self, "correlation_failed_reports", normalised_failures)
        normalised_overrides: dict[str, RealVsSimCorrelationThresholds] = {
            str(topic): thresholds
            for topic, thresholds in self.correlation_threshold_overrides.items()
            if not thresholds.is_empty
        }
        object.__setattr__(self, "correlation_threshold_overrides", normalised_overrides)
        normalised_window_stats: tuple[tuple[int, str, tuple[RealVsSimCorrelationWindowStats, ...]], ...] = tuple(
            (int(report_index), str(topic), tuple(stats))
            for report_index, topic, stats in self.correlation_per_window_stats
        )
        object.__setattr__(self, "correlation_per_window_stats", normalised_window_stats)

    @property
    def correlation_passed(self) -> bool:
        return not self.correlation_failed_reports

    @property
    def passed(self) -> bool:
        return (
            self.validation_passed
            and self.activation_activated
            and self.shard_merge_passed
            and self.history_passed
            and self.correlation_passed
        )

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    @property
    def scenario_count(self) -> int:
        return sum(shard.scenario_count for shard in self.shards)

    @property
    def report_count(self) -> int:
        return sum(shard.report_count for shard in self.shards)

    @property
    def failed_shards(self) -> tuple[str, ...]:
        return tuple(shard.shard_id for shard in self.shards if not shard.passed)

    @property
    def correlation_report_count(self) -> int:
        return len(self.correlation_reports)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-scenario-ci-review",
            "version": self.version,
            "reviewId": self.review_id,
            "passed": self.passed,
            "mergeId": self.merge_id,
            "workflowId": self.workflow_id,
            "manifestId": self.manifest_id,
            "validationId": self.validation_id,
            "activationId": self.activation_id,
            "validationPassed": bool(self.validation_passed),
            "activationActivated": bool(self.activation_activated),
            "shardMergePassed": bool(self.shard_merge_passed),
            "historyPassed": bool(self.history_passed),
            "historyFailedChecks": list(self.history_failed_checks),
            "activeWorkflowPath": self.active_workflow_path,
            "sourceWorkflowPath": self.source_workflow_path,
            "shardCount": self.shard_count,
            "scenarioCount": self.scenario_count,
            "reportCount": self.report_count,
            "failedShards": list(self.failed_shards),
            "shards": [shard.to_dict() for shard in self.shards],
            "adoption": None if self.adoption is None else self.adoption.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        if self.correlation_reports:
            payload["correlationReports"] = [report.to_dict() for report in self.correlation_reports]
            if self.correlation_report_paths:
                payload["correlationReportPaths"] = list(self.correlation_report_paths)
        gate_active = (self.correlation_thresholds is not None and not self.correlation_thresholds.is_empty) or bool(
            self.correlation_threshold_overrides
        )
        if gate_active:
            if self.correlation_thresholds is not None and not self.correlation_thresholds.is_empty:
                payload["correlationThresholds"] = self.correlation_thresholds.to_dict()
            if self.correlation_threshold_overrides:
                payload["correlationThresholdOverrides"] = correlation_threshold_overrides_to_dict(
                    self.correlation_threshold_overrides
                )
            payload["correlationPassed"] = self.correlation_passed
            if self.correlation_failed_reports:
                payload["correlationFailedReports"] = [
                    {
                        "index": int(index),
                        "bagSourceTopic": str(topic),
                        "failedChecks": list(checks),
                    }
                    for index, topic, checks in self.correlation_failed_reports
                ]
        if self.correlation_per_window_stats:
            payload["correlationPerWindowStats"] = [
                {
                    "index": int(report_index),
                    "bagSourceTopic": str(topic),
                    "windows": [stat.to_dict() for stat in stats],
                }
                for report_index, topic, stats in self.correlation_per_window_stats
            ]
        return payload


def build_route_policy_scenario_ci_review_artifact(
    merge_report: RoutePolicyScenarioShardMergeReport,
    validation_report: RoutePolicyScenarioCIWorkflowValidationReport,
    activation_report: RoutePolicyScenarioCIWorkflowActivationReport,
    *,
    review_id: str | None = None,
    pages_base_url: str | None = None,
    adoption: RoutePolicyScenarioCIReviewAdoption | None = None,
    correlation_reports: Sequence[RealVsSimCorrelationReport] = (),
    correlation_report_paths: Sequence[str | Path] = (),
    correlation_thresholds: RealVsSimCorrelationThresholds | None = None,
    correlation_threshold_overrides: Mapping[str, RealVsSimCorrelationThresholds] | None = None,
    provenance: RoutePolicyScenarioCIReviewProvenance | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioCIReviewArtifact:
    """Build a compact review artifact for a scenario CI workflow change.

    When ``adoption`` is provided, the adopted trigger mode, branches, and
    unified diff between the manual and adopted YAMLs ride along on the
    artifact so static Pages consumers can inspect the promotion-backed
    adoption without checking out the branch. The ``passed`` gate is
    unaffected — adoption presentation is purely additive.

    When ``correlation_reports`` is provided (typically aggregated from the
    shard runs' :class:`RoutePolicyScenarioSetRunReport.correlation_reports`),
    each entry is embedded into the review artifact and surfaced in the
    Markdown / HTML so reviewers can see headless-vs-bag drift alongside
    the workflow gate. ``correlation_report_paths`` carries the on-disk
    paths so the rendered tables can hyperlink back to the full JSON.
    Both default to empty so existing scenario-set fixtures keep their
    exact review JSON unchanged.
    """

    resolved_review_id = review_id or f"{activation_report.workflow_id}-review"
    if validation_report.workflow_id != activation_report.workflow_id:
        raise ValueError("validation and activation reports must reference the same workflow")
    if validation_report.manifest_id != activation_report.manifest_id:
        raise ValueError("validation and activation reports must reference the same manifest")
    correlation_failed_reports: tuple[tuple[int, str, tuple[str, ...]], ...] = ()
    correlation_per_window_stats: tuple[tuple[int, str, tuple[RealVsSimCorrelationWindowStats, ...]], ...] = ()
    overrides_map: dict[str, RealVsSimCorrelationThresholds] = {
        str(topic): thresholds
        for topic, thresholds in (correlation_threshold_overrides or {}).items()
        if not thresholds.is_empty
    }
    default_thresholds = correlation_thresholds if correlation_thresholds is not None else None
    gate_active = (default_thresholds is not None and not default_thresholds.is_empty) or bool(overrides_map)
    stratification_fallbacks: list[Mapping[str, Any]] = []
    if gate_active:
        failed: list[tuple[int, str, tuple[str, ...]]] = []
        per_window: list[tuple[int, str, tuple[RealVsSimCorrelationWindowStats, ...]]] = []
        event_windows_cache: dict[str, tuple[CorrelationEventWindow, ...]] = {}
        for index, report in enumerate(correlation_reports):
            applied = overrides_map.get(report.bag_source.source_topic, default_thresholds)
            if applied is None or applied.is_empty:
                continue
            event_windows = _resolve_event_windows_or_record_fallback(
                applied,
                report=report,
                cache=event_windows_cache,
                fallbacks=stratification_fallbacks,
            )
            _, failed_checks = evaluate_real_vs_sim_correlation_thresholds(report, applied, event_windows=event_windows)
            if failed_checks:
                failed.append((index, report.bag_source.source_topic, failed_checks))
            if applied.pair_distribution_strata is not None and applied.pair_distribution_strata > 1:
                window_stats = compute_per_window_correlation_stats(
                    report,
                    strata=int(applied.pair_distribution_strata),
                    mode=applied.pair_distribution_strata_mode,
                    event_windows=event_windows,
                )
                if window_stats:
                    per_window.append((index, report.bag_source.source_topic, window_stats))
        correlation_failed_reports = tuple(failed)
        correlation_per_window_stats = tuple(per_window)
    return RoutePolicyScenarioCIReviewArtifact(
        review_id=resolved_review_id,
        merge_id=merge_report.merge_id,
        workflow_id=activation_report.workflow_id,
        manifest_id=activation_report.manifest_id,
        validation_id=validation_report.validation_id,
        activation_id=activation_report.activation_id,
        validation_passed=validation_report.passed,
        activation_activated=activation_report.activated,
        shard_merge_passed=merge_report.passed,
        history_passed=merge_report.history.passed,
        active_workflow_path=activation_report.active_workflow_path,
        source_workflow_path=activation_report.source_workflow_path,
        history_failed_checks=merge_report.history.failed_checks,
        shards=tuple(_review_shard_from_run(shard_run) for shard_run in merge_report.shard_runs),
        adoption=adoption,
        provenance=provenance,
        correlation_reports=tuple(correlation_reports),
        correlation_report_paths=tuple(str(item) for item in correlation_report_paths),
        correlation_thresholds=correlation_thresholds,
        correlation_threshold_overrides=overrides_map,
        correlation_failed_reports=correlation_failed_reports,
        correlation_per_window_stats=correlation_per_window_stats,
        metadata={
            "pagesBaseUrl": pages_base_url,
            "historyPath": merge_report.history_path,
            "historyMarkdownPath": merge_report.history_markdown_path,
            "validationFailedChecks": list(validation_report.failed_checks),
            "activationFailedChecks": list(activation_report.failed_checks),
            **(
                {"correlationStratificationFallbacks": list(stratification_fallbacks)}
                if stratification_fallbacks
                else {}
            ),
            **_json_mapping(metadata or {}),
        },
    )


def _resolve_event_windows_or_record_fallback(
    applied: RealVsSimCorrelationThresholds,
    *,
    report: RealVsSimCorrelationReport,
    cache: dict[str, tuple[CorrelationEventWindow, ...]],
    fallbacks: list[Mapping[str, Any]],
) -> tuple[CorrelationEventWindow, ...] | None:
    """Load event windows for an event-aligned threshold or note the fallback.

    Returns the loaded event windows tuple when the applied thresholds use
    ``event-aligned`` mode and the file resolves to at least one window.
    When the file is missing, unreadable, or contains zero windows,
    records a structured fallback entry in ``fallbacks`` (so the review
    bundle metadata makes the slippage visible) and emits a stderr
    warning. Reports whose applied mode is not ``event-aligned`` short-
    circuit and return ``None`` without recording anything.

    The cache key includes both the bag topic and the path so per-topic
    overrides each load their own file once.
    """

    if applied.pair_distribution_strata_mode != "event-aligned":
        return None
    path = applied.event_windows_path
    if not path:
        _record_event_aligned_fallback(
            fallbacks,
            bag_topic=report.bag_source.source_topic,
            path=None,
            reason="event-aligned mode requested but event_windows_path is empty",
        )
        return None
    cache_key = f"{report.bag_source.source_topic}::{path}"
    if cache_key in cache:
        cached = cache[cache_key]
        if cached:
            return cached
        return None
    try:
        loaded = load_correlation_event_windows_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        cache[cache_key] = ()
        _record_event_aligned_fallback(
            fallbacks,
            bag_topic=report.bag_source.source_topic,
            path=path,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return None
    if not loaded:
        cache[cache_key] = ()
        _record_event_aligned_fallback(
            fallbacks,
            bag_topic=report.bag_source.source_topic,
            path=path,
            reason="event windows file is empty",
        )
        return None
    cache[cache_key] = loaded
    return loaded


def _record_event_aligned_fallback(
    fallbacks: list[Mapping[str, Any]],
    *,
    bag_topic: str,
    path: str | None,
    reason: str,
) -> None:
    """Append one fallback record to the artifact metadata and warn on stderr."""

    fallbacks.append(
        {
            "bagSourceTopic": bag_topic,
            "requested": "event-aligned",
            "applied": "equal-pair-count",
            "eventWindowsPath": path,
            "reason": reason,
        }
    )
    sys.stderr.write(
        f"warning: event-aligned correlation stratification fell back to "
        f"equal-pair-count for topic {bag_topic!r}: {reason}\n"
    )


def collect_correlation_reports_from_shard_runs(
    merge_report: RoutePolicyScenarioShardMergeReport,
) -> tuple[tuple[RealVsSimCorrelationReport, ...], tuple[str, ...]]:
    """Gather correlation reports embedded in each shard's run JSON.

    Every :class:`RoutePolicyScenarioShardRunSummary` whose ``run_path`` is
    set and loadable is parsed back as a
    :class:`RoutePolicyScenarioSetRunReport`, and its embedded
    :class:`RealVsSimCorrelationReport` entries are collected in shard
    order. Shard summaries without a ``run_path`` (or whose ``run_path``
    cannot be opened — e.g. unit-test fixtures that point at fictional
    paths to keep the merge report self-contained) are skipped silently
    so the review CLI keeps working without correlation reports
    available, mirroring the upstream pre-#121 behaviour.
    """

    reports: list[RealVsSimCorrelationReport] = []
    paths: list[str] = []
    for shard_run in merge_report.shard_runs:
        if shard_run.run_path is None:
            continue
        try:
            loaded = load_route_policy_scenario_set_run_json(shard_run.run_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        for index, report in enumerate(loaded.correlation_reports):
            reports.append(report)
            if index < len(loaded.correlation_report_paths):
                paths.append(loaded.correlation_report_paths[index])
    if len(paths) != len(reports):
        # Fall back to no paths when only some shards published them — the
        # review artifact accepts an empty paths tuple in that case.
        return tuple(reports), ()
    return tuple(reports), tuple(paths)


def build_route_policy_scenario_ci_review_adoption(
    *,
    adoption_id: str,
    adopted: bool,
    trigger_mode: str,
    adopted_active_workflow_path: str | Path,
    adopted_source_workflow_path: str | Path,
    manual_workflow_text: str,
    adopted_workflow_text: str,
    push_branches: Sequence[str] = (),
    pull_request_branches: Sequence[str] = (),
    diff_context_lines: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioCIReviewAdoption:
    """Build a review-friendly adoption summary including a unified YAML diff."""

    diff_lines = list(
        difflib.unified_diff(
            manual_workflow_text.splitlines(keepends=True),
            adopted_workflow_text.splitlines(keepends=True),
            fromfile="manual",
            tofile="adopted",
            n=max(0, int(diff_context_lines)),
        )
    )
    workflow_diff = "".join(diff_lines) if diff_lines else None
    return RoutePolicyScenarioCIReviewAdoption(
        adoption_id=adoption_id,
        adopted=adopted,
        trigger_mode=trigger_mode,
        adopted_active_workflow_path=Path(adopted_active_workflow_path).as_posix(),
        adopted_source_workflow_path=Path(adopted_source_workflow_path).as_posix(),
        push_branches=tuple(push_branches),
        pull_request_branches=tuple(pull_request_branches),
        workflow_diff=workflow_diff,
        metadata=_json_mapping(metadata or {}),
    )


def write_route_policy_scenario_ci_review_json(
    path: str | Path,
    artifact: RoutePolicyScenarioCIReviewArtifact,
) -> Path:
    """Write a scenario CI review artifact as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_ci_review_json(path: str | Path) -> RoutePolicyScenarioCIReviewArtifact:
    """Load a scenario CI review JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_ci_review_from_dict(_mapping(payload, "ciReview"))


def write_route_policy_scenario_ci_review_bundle(
    output_dir: str | Path,
    artifact: RoutePolicyScenarioCIReviewArtifact,
    *,
    json_name: str = "review.json",
    markdown_name: str = "review.md",
    html_name: str = "index.html",
) -> dict[str, str]:
    """Write JSON, Markdown, and HTML files suitable for static Pages hosting."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = write_route_policy_scenario_ci_review_json(directory / json_name, artifact)
    markdown_path = directory / markdown_name
    markdown_path.write_text(render_route_policy_scenario_ci_review_markdown(artifact), encoding="utf-8")
    html_path = directory / html_name
    html_path.write_text(render_route_policy_scenario_ci_review_html(artifact), encoding="utf-8")
    return {
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        "html": html_path.as_posix(),
    }


def route_policy_scenario_ci_review_shard_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewShard:
    """Rebuild one CI review shard from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review-shard")
    return RoutePolicyScenarioCIReviewShard(
        shard_id=str(payload["shardId"]),
        passed=bool(payload.get("passed", False)),
        scenario_count=int(payload["scenarioCount"]),
        report_count=int(payload["reportCount"]),
        run_path=None if payload.get("runPath") is None else str(payload["runPath"]),
        history_path=None if payload.get("historyPath") is None else str(payload["historyPath"]),
        report_paths=tuple(str(item) for item in _sequence(payload.get("reportPaths", ()), "reportPaths")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_review_adoption_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewAdoption:
    """Rebuild a CI review adoption block from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review-adoption")
    return RoutePolicyScenarioCIReviewAdoption(
        adoption_id=str(payload["adoptionId"]),
        adopted=bool(payload.get("adopted", False)),
        trigger_mode=str(payload["triggerMode"]),
        adopted_active_workflow_path=str(payload["adoptedActiveWorkflowPath"]),
        adopted_source_workflow_path=str(payload["adoptedSourceWorkflowPath"]),
        push_branches=tuple(str(item) for item in _sequence(payload.get("pushBranches", ()), "pushBranches")),
        pull_request_branches=tuple(
            str(item) for item in _sequence(payload.get("pullRequestBranches", ()), "pullRequestBranches")
        ),
        workflow_diff=None if payload.get("workflowDiff") is None else str(payload["workflowDiff"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_review_provenance_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewProvenance:
    """Rebuild a scenario CI review provenance block from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review-provenance")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_REVIEW_PROVENANCE_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_REVIEW_PROVENANCE_VERSION:
        raise ValueError(f"unsupported route policy scenario CI review provenance version: {version}")
    extra_payload = payload.get("extra")
    extra = _json_mapping(_mapping(extra_payload, "extra")) if isinstance(extra_payload, Mapping) else {}
    return RoutePolicyScenarioCIReviewProvenance(
        kind=str(payload["kind"]),
        generated_at=str(payload["generatedAt"]),
        git_commit=None if payload.get("gitCommit") is None else str(payload["gitCommit"]),
        scene_id=None if payload.get("sceneId") is None else str(payload["sceneId"]),
        scenario_set_id=None if payload.get("scenarioSetId") is None else str(payload["scenarioSetId"]),
        matrix_hash=None if payload.get("matrixHash") is None else str(payload["matrixHash"]),
        policy_version=None if payload.get("policyVersion") is None else str(payload["policyVersion"]),
        env_contract_version=(
            None if payload.get("envContractVersion") is None else str(payload["envContractVersion"])
        ),
        correlation_threshold_profile=(
            None if payload.get("correlationThresholdProfile") is None else str(payload["correlationThresholdProfile"])
        ),
        asset_source=None if payload.get("assetSource") is None else str(payload["assetSource"]),
        extra=extra,
        version=version,
    )


def route_policy_scenario_ci_review_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewArtifact:
    """Rebuild a scenario CI review artifact from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION:
        raise ValueError(f"unsupported route policy scenario CI review version: {version}")
    shards = tuple(
        route_policy_scenario_ci_review_shard_from_dict(_mapping(item, "reviewShard"))
        for item in _sequence(payload.get("shards", ()), "shards")
    )
    expected_shard_count = payload.get("shardCount")
    if expected_shard_count is not None and int(expected_shard_count) != len(shards):
        raise ValueError("shardCount does not match loaded review shards")
    adoption_payload = payload.get("adoption")
    adoption = (
        None
        if adoption_payload is None
        else route_policy_scenario_ci_review_adoption_from_dict(_mapping(adoption_payload, "adoption"))
    )
    provenance_payload = payload.get("provenance")
    provenance = (
        None
        if provenance_payload is None
        else route_policy_scenario_ci_review_provenance_from_dict(_mapping(provenance_payload, "provenance"))
    )
    correlation_reports = tuple(
        real_vs_sim_correlation_report_from_dict(_mapping(item, "correlationReport"))
        for item in _sequence(payload.get("correlationReports", ()), "correlationReports")
    )
    correlation_report_paths = tuple(
        str(item) for item in _sequence(payload.get("correlationReportPaths", ()), "correlationReportPaths")
    )
    thresholds_payload = payload.get("correlationThresholds")
    correlation_thresholds = (
        real_vs_sim_correlation_thresholds_from_dict(_mapping(thresholds_payload, "correlationThresholds"))
        if isinstance(thresholds_payload, Mapping)
        else None
    )
    overrides_payload = payload.get("correlationThresholdOverrides")
    correlation_threshold_overrides = (
        correlation_threshold_overrides_from_dict(_mapping(overrides_payload, "correlationThresholdOverrides"))
        if isinstance(overrides_payload, Mapping)
        else {}
    )
    correlation_failed_reports = tuple(
        (
            int(item.get("index", 0)),
            str(item.get("bagSourceTopic", "")),
            tuple(str(check) for check in _sequence(item.get("failedChecks", ()), "failedChecks")),
        )
        for item in _sequence(payload.get("correlationFailedReports", ()), "correlationFailedReports")
        if isinstance(item, Mapping)
    )
    correlation_per_window_stats = tuple(
        (
            int(item.get("index", 0)),
            str(item.get("bagSourceTopic", "")),
            tuple(
                real_vs_sim_correlation_window_stats_from_dict(_mapping(window, "correlationWindowStat"))
                for window in _sequence(item.get("windows", ()), "windows")
            ),
        )
        for item in _sequence(payload.get("correlationPerWindowStats", ()), "correlationPerWindowStats")
        if isinstance(item, Mapping)
    )
    return RoutePolicyScenarioCIReviewArtifact(
        review_id=str(payload["reviewId"]),
        merge_id=str(payload["mergeId"]),
        workflow_id=str(payload["workflowId"]),
        manifest_id=str(payload["manifestId"]),
        validation_id=str(payload["validationId"]),
        activation_id=str(payload["activationId"]),
        validation_passed=bool(payload.get("validationPassed", False)),
        activation_activated=bool(payload.get("activationActivated", False)),
        shard_merge_passed=bool(payload.get("shardMergePassed", False)),
        history_passed=bool(payload.get("historyPassed", False)),
        active_workflow_path=str(payload["activeWorkflowPath"]),
        source_workflow_path=str(payload["sourceWorkflowPath"]),
        history_failed_checks=tuple(
            str(item) for item in _sequence(payload.get("historyFailedChecks", ()), "historyFailedChecks")
        ),
        shards=shards,
        adoption=adoption,
        provenance=provenance,
        correlation_reports=correlation_reports,
        correlation_report_paths=correlation_report_paths,
        correlation_thresholds=correlation_thresholds,
        correlation_threshold_overrides=correlation_threshold_overrides,
        correlation_failed_reports=correlation_failed_reports,
        correlation_per_window_stats=correlation_per_window_stats,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_scenario_ci_review_markdown(artifact: RoutePolicyScenarioCIReviewArtifact) -> str:
    """Render a compact Markdown review artifact."""

    sample_notice = _sample_notice(artifact)
    lines = [
        f"# Route Policy Scenario CI Review: {artifact.review_id}",
        f"- Status: {'PASS' if artifact.passed else 'FAIL'}",
        f"- Workflow: {artifact.workflow_id}",
        f"- Manifest: {artifact.manifest_id}",
        f"- Merge: {artifact.merge_id} ({'PASS' if artifact.shard_merge_passed else 'FAIL'})",
        f"- Validation: {artifact.validation_id} ({'PASS' if artifact.validation_passed else 'FAIL'})",
        f"- Activation: {artifact.activation_id} ({'ACTIVE' if artifact.activation_activated else 'BLOCKED'})",
        f"- Active workflow: {artifact.active_workflow_path}",
        f"- Shards: {artifact.shard_count}",
        f"- Scenarios: {artifact.scenario_count}",
        f"- Reports: {artifact.report_count}",
        "",
    ]
    if sample_notice:
        lines.extend([f"> {sample_notice}", ""])
    provenance_lines = _provenance_markdown_lines(artifact.provenance)
    if provenance_lines:
        lines.extend(provenance_lines)
        lines.append("")
    lines.extend(
        [
            "| Shard | Pass | Scenarios | Reports | Run |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for shard in artifact.shards:
        lines.append(
            "| "
            f"{shard.shard_id} | "
            f"{'yes' if shard.passed else 'no'} | "
            f"{shard.scenario_count} | "
            f"{shard.report_count} | "
            f"{shard.run_path or 'n/a'} |"
        )
    if artifact.history_failed_checks:
        lines.extend(["", "## History Failed Checks", ""])
        lines.extend(f"- {check}" for check in artifact.history_failed_checks)
    if artifact.correlation_reports:
        lines.append("")
        lines.append("## Real-vs-sim correlation")
        gate_default_active = (
            artifact.correlation_thresholds is not None and not artifact.correlation_thresholds.is_empty
        )
        if gate_default_active or artifact.correlation_threshold_overrides:
            descriptor_parts: list[str] = []
            if gate_default_active:
                descriptor_parts.append(f"default: {_describe_correlation_thresholds(artifact.correlation_thresholds)}")
            if artifact.correlation_threshold_overrides:
                descriptor_parts.append(f"per-topic overrides: {len(artifact.correlation_threshold_overrides)}")
            lines.append(
                f"- Gate: {'PASS' if artifact.correlation_passed else 'FAIL'}  ({'; '.join(descriptor_parts)})"
            )
        lines.extend(
            [
                "",
                "| Bag topic | Matched pairs | Translation mean (m) | Translation p95 (m) | Translation max (m) | Heading mean (rad) | Report |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for index, correlation in enumerate(artifact.correlation_reports):
            bag = correlation.bag_source
            heading = (
                f"{correlation.heading_error_mean_radians:.4f}"
                if correlation.heading_error_mean_radians is not None
                else "n/a"
            )
            path_cell = (
                artifact.correlation_report_paths[index] if index < len(artifact.correlation_report_paths) else "n/a"
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
        if artifact.correlation_failed_reports:
            lines.extend(["", "### Correlation gate failures", ""])
            for index, topic, checks in artifact.correlation_failed_reports:
                lines.append(f"- `{topic}` (report #{index}): {', '.join(checks)}")
        if artifact.correlation_per_window_stats:
            lines.extend(["", "### Per-window correlation stats", ""])
            for report_index, topic, windows in artifact.correlation_per_window_stats:
                has_events = any(stat.event_name is not None for stat in windows)
                lines.append(f"- `{topic}` (report #{report_index}):")
                lines.append("")
                if has_events:
                    lines.append(
                        "| Window | Event | Bag time (s) | Pairs | Translation mean (m) | Translation p95 (m) | Translation max (m) | Heading mean (rad) |"
                    )
                    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
                else:
                    lines.append(
                        "| Window | Bag time (s) | Pairs | Translation mean (m) | Translation p95 (m) | Translation max (m) | Heading mean (rad) |"
                    )
                    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
                for stat in windows:
                    heading = (
                        f"{stat.heading_error_mean_radians:.4f}"
                        if stat.heading_error_mean_radians is not None
                        else "n/a"
                    )
                    if has_events:
                        event_cell = _render_event_cell_markdown(stat)
                        lines.append(
                            f"| {stat.window_index} | "
                            f"{event_cell} | "
                            f"{stat.bag_time_start_seconds:.2f} – {stat.bag_time_end_seconds:.2f} | "
                            f"{stat.pair_count} | "
                            f"{stat.translation_error_mean_meters:.4f} | "
                            f"{stat.translation_error_p95_meters:.4f} | "
                            f"{stat.translation_error_max_meters:.4f} | "
                            f"{heading} |"
                        )
                    else:
                        lines.append(
                            f"| {stat.window_index} | "
                            f"{stat.bag_time_start_seconds:.2f} – {stat.bag_time_end_seconds:.2f} | "
                            f"{stat.pair_count} | "
                            f"{stat.translation_error_mean_meters:.4f} | "
                            f"{stat.translation_error_p95_meters:.4f} | "
                            f"{stat.translation_error_max_meters:.4f} | "
                            f"{heading} |"
                        )
                lines.append("")
        if artifact.metadata.get("correlationStratificationFallbacks"):
            fallbacks = artifact.metadata["correlationStratificationFallbacks"]
            lines.extend(["", "### Correlation stratification fallbacks", ""])
            for entry in fallbacks:
                lines.append(
                    f"- `{entry.get('bagSourceTopic', '?')}`: requested "
                    f"`{entry.get('requested')}` → applied `{entry.get('applied')}` "
                    f"(reason: {entry.get('reason')})"
                )
    if artifact.adoption is not None:
        adoption = artifact.adoption
        lines.extend(
            [
                "",
                "## Adopted Workflow",
                "",
                f"- Adoption: {adoption.adoption_id} ({'ADOPTED' if adoption.adopted else 'BLOCKED'})",
                f"- Trigger mode: {adoption.trigger_mode}",
                f"- Adopted active path: {adoption.adopted_active_workflow_path}",
                f"- Adopted source path: {adoption.adopted_source_workflow_path}",
                f"- Push branches: {_display_branches(adoption.push_branches)}",
                f"- Pull request branches: {_display_branches(adoption.pull_request_branches)}",
            ]
        )
        if adoption.workflow_diff:
            lines.extend(
                [
                    "",
                    "```diff",
                    adoption.workflow_diff.rstrip("\n"),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def render_route_policy_scenario_ci_review_html(artifact: RoutePolicyScenarioCIReviewArtifact) -> str:
    """Render a self-contained static HTML review page."""

    status_class = "pass" if artifact.passed else "fail"
    sample_notice = _sample_notice(artifact)
    notice_section = f'<p class="notice">{escape(sample_notice)}</p>' if sample_notice else ""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(shard.shard_id)}</td>"
        f'<td><span class="pill {"pass" if shard.passed else "fail"}">{"PASS" if shard.passed else "FAIL"}</span></td>'
        f"<td>{shard.scenario_count}</td>"
        f"<td>{shard.report_count}</td>"
        f"<td>{_optional_link(shard.run_path)}</td>"
        "</tr>"
        for shard in artifact.shards
    )
    history_failed = "".join(f"<li>{escape(check)}</li>" for check in artifact.history_failed_checks)
    failed_section = (
        f"<section><h2>History Failed Checks</h2><ul>{history_failed}</ul></section>"
        if artifact.history_failed_checks
        else ""
    )
    correlation_section = _render_correlation_section_html(artifact)
    adoption_section = _render_adoption_section_html(artifact.adoption)
    provenance_section = _render_provenance_section_html(artifact.provenance)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(artifact.review_id)} CI Review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f8f4; color: #20231f; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 20px; margin: 32px 0 12px; letter-spacing: 0; }}
    .subtitle {{ color: #5b6259; margin: 0 0 24px; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .metric {{ background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; padding: 14px; }}
    .metric span {{ display: block; color: #5b6259; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    .notice {{ background: #fff8d7; border: 1px solid #e7d68c; border-radius: 8px; color: #514411; padding: 12px 14px; margin: 0 0 24px; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 700; }}
    .pass {{ background: #dcefd8; color: #1e5a2b; }}
    .fail {{ background: #f7d6d2; color: #8a1f16; }}
    .synthetic {{ background: #fff1c1; color: #6a4d00; }}
    .production {{ background: #dcefd8; color: #1e5a2b; }}
    .unknown {{ background: #e3e5df; color: #424940; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e9ede5; text-align: left; vertical-align: top; }}
    th {{ background: #eef2ea; font-size: 13px; color: #424940; }}
    tr:last-child td {{ border-bottom: 0; }}
    a {{ color: #285b9b; }}
    code {{ background: #eef2ea; padding: 2px 5px; border-radius: 4px; }}
    pre.diff {{ background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; padding: 12px; overflow-x: auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.45; }}
    pre.diff .add {{ color: #1e5a2b; }}
    pre.diff .del {{ color: #8a1f16; }}
    pre.diff .hunk {{ color: #5b6259; }}
  </style>
</head>
<body>
  <main>
    <h1>Route Policy Scenario CI Review</h1>
    <p class="subtitle"><span class="pill {status_class}">{"PASS" if artifact.passed else "FAIL"}</span> {escape(artifact.review_id)}</p>
    {notice_section}
    <section class="grid">
      <div class="metric"><span>Workflow</span><strong>{escape(artifact.workflow_id)}</strong></div>
      <div class="metric"><span>Manifest</span><strong>{escape(artifact.manifest_id)}</strong></div>
      <div class="metric"><span>Shards</span><strong>{artifact.shard_count}</strong></div>
      <div class="metric"><span>Scenarios</span><strong>{artifact.scenario_count}</strong></div>
      <div class="metric"><span>Reports</span><strong>{artifact.report_count}</strong></div>
    </section>
    {provenance_section}
    <section>
      <h2>Workflow Gate</h2>
      <p>Validation <span class="pill {"pass" if artifact.validation_passed else "fail"}">{"PASS" if artifact.validation_passed else "FAIL"}</span>
      Activation <span class="pill {"pass" if artifact.activation_activated else "fail"}">{"ACTIVE" if artifact.activation_activated else "BLOCKED"}</span></p>
      <p>Active workflow: <code>{escape(artifact.active_workflow_path)}</code></p>
      <p>Source workflow: <code>{escape(artifact.source_workflow_path)}</code></p>
    </section>
    <section>
      <h2>Shard Runs</h2>
      <table>
        <thead><tr><th>Shard</th><th>Status</th><th>Scenarios</th><th>Reports</th><th>Run</th></tr></thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </section>
    {failed_section}
    {correlation_section}
    {adoption_section}
  </main>
</body>
</html>
"""


def run_review_cli(args: Any) -> None:
    """Run the route policy scenario-ci-review CLI."""

    merge_report = load_route_policy_scenario_shard_merge_json(getattr(args, "shard_merge"))
    validation_report = load_route_policy_scenario_ci_workflow_validation_json(getattr(args, "validation_report"))
    activation_report = load_route_policy_scenario_ci_workflow_activation_json(getattr(args, "activation_report"))
    adoption = _load_review_adoption(
        adoption_report_path=getattr(args, "adoption_report", None),
        manual_active_workflow_path=activation_report.active_workflow_path,
        manual_workflow_override=getattr(args, "manual_workflow", None),
        adopted_workflow_override=getattr(args, "adopted_workflow", None),
    )
    if bool(getattr(args, "no_correlation_reports", False)):
        correlation_reports: tuple[RealVsSimCorrelationReport, ...] = ()
        correlation_report_paths: tuple[str, ...] = ()
    else:
        correlation_reports, correlation_report_paths = collect_correlation_reports_from_shard_runs(merge_report)
    correlation_thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=getattr(args, "max_correlation_translation_mean_meters", None),
        max_translation_error_p95_meters=getattr(args, "max_correlation_translation_p95_meters", None),
        max_translation_error_max_meters=getattr(args, "max_correlation_translation_max_meters", None),
        max_heading_error_mean_radians=getattr(args, "max_correlation_heading_mean_radians", None),
        max_pair_translation_error_meters=getattr(args, "max_correlation_pair_translation_meters", None),
        max_exceeding_translation_pair_fraction=getattr(args, "max_correlation_pair_fraction", None),
        max_pair_heading_error_radians=getattr(args, "max_correlation_pair_heading_radians", None),
        max_exceeding_heading_pair_fraction=getattr(args, "max_correlation_heading_pair_fraction", None),
        pair_distribution_strata=getattr(args, "correlation_pair_distribution_strata", None),
        pair_distribution_strata_mode=getattr(args, "correlation_pair_distribution_strata_mode", "equal-duration"),
        event_windows_path=getattr(args, "correlation_event_windows", None),
    )
    if correlation_thresholds.is_empty:
        correlation_thresholds = None
    overrides_path = getattr(args, "correlation_thresholds_config", None)
    correlation_threshold_overrides: dict[str, RealVsSimCorrelationThresholds] = {}
    if overrides_path:
        from ..robotics.rosbag_correlation import load_correlation_threshold_overrides_json

        correlation_threshold_overrides = load_correlation_threshold_overrides_json(overrides_path)
    provenance = _build_review_provenance_from_cli(args)
    artifact = build_route_policy_scenario_ci_review_artifact(
        merge_report,
        validation_report,
        activation_report,
        review_id=getattr(args, "review_id", None),
        pages_base_url=getattr(args, "pages_base_url", None),
        adoption=adoption,
        correlation_reports=correlation_reports,
        correlation_report_paths=correlation_report_paths,
        correlation_thresholds=correlation_thresholds,
        correlation_threshold_overrides=correlation_threshold_overrides,
        provenance=provenance,
    )
    bundle_dir = getattr(args, "bundle_dir", None)
    if bundle_dir:
        paths = write_route_policy_scenario_ci_review_bundle(bundle_dir, artifact)
        print(f"Scenario CI review bundle saved to: {paths['html']}")
    else:
        write_route_policy_scenario_ci_review_json(getattr(args, "output"), artifact)
        if getattr(args, "markdown_output", None):
            markdown_path = Path(getattr(args, "markdown_output"))
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_route_policy_scenario_ci_review_markdown(artifact), encoding="utf-8")
        if getattr(args, "html_output", None):
            html_path = Path(getattr(args, "html_output"))
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(render_route_policy_scenario_ci_review_html(artifact), encoding="utf-8")
        print(f"Scenario CI review saved to: {getattr(args, 'output')}")
    print(render_route_policy_scenario_ci_review_markdown(artifact), end="")
    if bool(getattr(args, "fail_on_review", False)) and not artifact.passed:
        raise SystemExit(2)


_PROVENANCE_FLAG_LABELS: tuple[tuple[str, str], ...] = (
    ("scene_id", "--scene-id"),
    ("scenario_set_id", "--scenario-set-id"),
    ("matrix_hash", "--matrix-hash"),
    ("policy_version", "--policy-version"),
    ("env_contract_version", "--env-contract-version"),
    ("correlation_threshold_profile", "--correlation-threshold-profile"),
    ("asset_source", "--asset-source"),
    ("git_commit", "--git-commit"),
    ("generated_at", "--generated-at"),
    ("provenance_extra", "--provenance-extra"),
)


def _build_review_provenance_from_cli(args: Any) -> RoutePolicyScenarioCIReviewProvenance | None:
    """Assemble a provenance block from the CLI namespace, or return None.

    Returns None when ``--kind`` is not set AND no other provenance flag was
    provided, preserving the v1 JSON shape for callers that haven't migrated.
    Raises SystemExit(2) when a provenance flag is set without ``--kind``.
    """

    kind = getattr(args, "kind", None)
    provenance_flag_values = {attr: getattr(args, attr, None) for attr, _flag in _PROVENANCE_FLAG_LABELS}
    has_other_provenance = any(value not in (None, "", []) for value in provenance_flag_values.values())
    if kind is None and not has_other_provenance:
        return None
    if kind is None:
        offenders = [
            flag for attr, flag in _PROVENANCE_FLAG_LABELS if provenance_flag_values[attr] not in (None, "", [])
        ]
        sys.stderr.write(
            f"error: --kind {{synthetic,production}} is required when any of {', '.join(offenders)} is set\n"
        )
        raise SystemExit(2)
    generated_at = provenance_flag_values["generated_at"] or _utc_now_iso()
    git_commit = provenance_flag_values["git_commit"]
    if git_commit is None:
        git_commit = _resolve_git_commit_best_effort()
    extras = _parse_provenance_extra(provenance_flag_values["provenance_extra"])
    return RoutePolicyScenarioCIReviewProvenance(
        kind=str(kind),
        generated_at=str(generated_at),
        git_commit=git_commit,
        scene_id=provenance_flag_values["scene_id"],
        scenario_set_id=provenance_flag_values["scenario_set_id"],
        matrix_hash=provenance_flag_values["matrix_hash"],
        policy_version=provenance_flag_values["policy_version"],
        env_contract_version=provenance_flag_values["env_contract_version"],
        correlation_threshold_profile=provenance_flag_values["correlation_threshold_profile"],
        asset_source=provenance_flag_values["asset_source"],
        extra=extras,
    )


def _parse_provenance_extra(raw_pairs: Sequence[str] | None) -> dict[str, str]:
    if not raw_pairs:
        return {}
    result: dict[str, str] = {}
    for pair in raw_pairs:
        if "=" not in pair:
            sys.stderr.write(f"error: --provenance-extra expects KEY=VALUE, got {pair!r}\n")
            raise SystemExit(2)
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            sys.stderr.write(f"error: --provenance-extra key must not be empty, got {pair!r}\n")
            raise SystemExit(2)
        result[key] = value
    return result


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_git_commit_best_effort() -> str | None:
    """Try to read the current git commit hash, returning None on failure."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _load_review_adoption(
    *,
    adoption_report_path: str | Path | None,
    manual_active_workflow_path: str,
    manual_workflow_override: str | Path | None = None,
    adopted_workflow_override: str | Path | None = None,
) -> RoutePolicyScenarioCIReviewAdoption | None:
    if adoption_report_path is None:
        return None
    payload = json.loads(Path(adoption_report_path).read_text(encoding="utf-8"))
    payload = _mapping(payload, "adoptionReport")
    manual_text_path = Path(manual_workflow_override) if manual_workflow_override else Path(manual_active_workflow_path)
    adopted_text_path = (
        Path(adopted_workflow_override) if adopted_workflow_override else Path(payload["adoptedActiveWorkflowPath"])
    )
    manual_text = manual_text_path.read_text(encoding="utf-8")
    adopted_text = adopted_text_path.read_text(encoding="utf-8")
    return build_route_policy_scenario_ci_review_adoption(
        adoption_id=str(payload["adoptionId"]),
        adopted=bool(payload.get("adopted", False)),
        trigger_mode=str(payload["triggerMode"]),
        adopted_active_workflow_path=str(payload["adoptedActiveWorkflowPath"]),
        adopted_source_workflow_path=str(payload["adoptedSourceWorkflowPath"]),
        manual_workflow_text=manual_text,
        adopted_workflow_text=adopted_text,
        push_branches=tuple(str(item) for item in _sequence(payload.get("pushBranches", ()), "pushBranches")),
        pull_request_branches=tuple(
            str(item) for item in _sequence(payload.get("pullRequestBranches", ()), "pullRequestBranches")
        ),
    )


def _describe_correlation_thresholds(thresholds: RealVsSimCorrelationThresholds) -> str:
    """Return a compact human-readable description of the populated thresholds."""

    parts: list[str] = []
    if thresholds.max_translation_error_mean_meters is not None:
        parts.append(f"translation mean ≤ {thresholds.max_translation_error_mean_meters:g} m")
    if thresholds.max_translation_error_p95_meters is not None:
        parts.append(f"translation p95 ≤ {thresholds.max_translation_error_p95_meters:g} m")
    if thresholds.max_translation_error_max_meters is not None:
        parts.append(f"translation max ≤ {thresholds.max_translation_error_max_meters:g} m")
    if thresholds.max_heading_error_mean_radians is not None:
        parts.append(f"heading mean ≤ {thresholds.max_heading_error_mean_radians:g} rad")
    if (
        thresholds.max_pair_translation_error_meters is not None
        and thresholds.max_exceeding_translation_pair_fraction is not None
    ):
        parts.append(
            f"pair distribution: ≤ {thresholds.max_exceeding_translation_pair_fraction:g} fraction "
            f"of pairs above {thresholds.max_pair_translation_error_meters:g} m"
        )
    if (
        thresholds.max_pair_heading_error_radians is not None
        and thresholds.max_exceeding_heading_pair_fraction is not None
    ):
        parts.append(
            f"heading pair distribution: ≤ {thresholds.max_exceeding_heading_pair_fraction:g} fraction "
            f"of pairs above {thresholds.max_pair_heading_error_radians:g} rad"
        )
    if thresholds.pair_distribution_strata is not None and thresholds.pair_distribution_strata > 1:
        parts.append(
            f"pair distribution stratified into {thresholds.pair_distribution_strata} "
            f"{thresholds.pair_distribution_strata_mode} windows"
        )
    return ", ".join(parts) if parts else "no thresholds configured"


def _render_correlation_section_html(artifact: RoutePolicyScenarioCIReviewArtifact) -> str:
    if not artifact.correlation_reports:
        return ""
    rows: list[str] = []
    for index, correlation in enumerate(artifact.correlation_reports):
        bag = correlation.bag_source
        heading = (
            f"{correlation.heading_error_mean_radians:.4f}"
            if correlation.heading_error_mean_radians is not None
            else "n/a"
        )
        path_cell = artifact.correlation_report_paths[index] if index < len(artifact.correlation_report_paths) else None
        rows.append(
            "<tr>"
            f"<td><code>{escape(bag.source_topic)}</code></td>"
            f"<td>{correlation.matched_pair_count}</td>"
            f"<td>{correlation.translation_error_mean_meters:.4f}</td>"
            f"<td>{correlation.translation_error_p95_meters:.4f}</td>"
            f"<td>{correlation.translation_error_max_meters:.4f}</td>"
            f"<td>{heading}</td>"
            f"<td>{_optional_link(path_cell)}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    gate_html = ""
    gate_default_active = artifact.correlation_thresholds is not None and not artifact.correlation_thresholds.is_empty
    if gate_default_active or artifact.correlation_threshold_overrides:
        gate_pill = "pass" if artifact.correlation_passed else "fail"
        gate_label = "PASS" if artifact.correlation_passed else "FAIL"
        descriptor_parts: list[str] = []
        if gate_default_active:
            descriptor_parts.append(f"default: {_describe_correlation_thresholds(artifact.correlation_thresholds)}")
        if artifact.correlation_threshold_overrides:
            descriptor_parts.append(f"per-topic overrides: {len(artifact.correlation_threshold_overrides)}")
        gate_html = (
            f'<p>Gate <span class="pill {gate_pill}">{gate_label}</span> ({escape("; ".join(descriptor_parts))})</p>'
        )
    failures_html = ""
    if artifact.correlation_failed_reports:
        failure_items = "".join(
            f"<li><code>{escape(topic)}</code> (report #{index}): {escape(', '.join(checks))}</li>"
            for index, topic, checks in artifact.correlation_failed_reports
        )
        failures_html = f"<h3>Correlation gate failures</h3><ul>{failure_items}</ul>"
    per_window_html = ""
    if artifact.correlation_per_window_stats:
        groups: list[str] = []
        for report_index, topic, windows in artifact.correlation_per_window_stats:
            has_events = any(stat.event_name is not None for stat in windows)
            window_rows = "\n".join(_render_per_window_row_html(stat, has_events) for stat in windows)
            event_header = "<th>Event</th>" if has_events else ""
            groups.append(
                f"<h4><code>{escape(topic)}</code> (report #{report_index})</h4>"
                "<table>"
                "<thead><tr>"
                f"<th>Window</th>{event_header}<th>Bag time (s)</th><th>Pairs</th>"
                "<th>Translation mean (m)</th><th>Translation p95 (m)</th><th>Translation max (m)</th>"
                "<th>Heading mean (rad)</th>"
                "</tr></thead>"
                f"<tbody>{window_rows}</tbody>"
                "</table>"
            )
        per_window_html = f"<h3>Per-window correlation stats</h3>{''.join(groups)}"
    return (
        "<section>"
        "<h2>Real-vs-sim correlation</h2>"
        f"{gate_html}"
        "<table>"
        "<thead><tr>"
        "<th>Bag topic</th><th>Matched pairs</th>"
        "<th>Translation mean (m)</th><th>Translation p95 (m)</th><th>Translation max (m)</th>"
        "<th>Heading mean (rad)</th><th>Report</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        f"{failures_html}"
        f"{per_window_html}"
        "</section>"
    )


def _render_per_window_row_html(stat: RealVsSimCorrelationWindowStats, has_events: bool) -> str:
    """Render one ``<tr>`` for the per-window correlation stats table."""

    event_cell = ""
    if has_events:
        if stat.event_name is None:
            event_cell = "<td>—</td>"
        else:
            parts = [f"<code>{escape(stat.event_name)}</code>"]
            if stat.event_source and stat.event_source != "external":
                parts.append(f"<span>({escape(stat.event_source)})</span>")
            if stat.event_tags:
                tag_block = " ".join(f"<code>{escape(tag)}</code>" for tag in stat.event_tags)
                parts.append(tag_block)
            event_cell = f"<td>{' '.join(parts)}</td>"
    heading_cell = f"{stat.heading_error_mean_radians:.4f}" if stat.heading_error_mean_radians is not None else "n/a"
    return (
        "<tr>"
        f"<td>{stat.window_index}</td>"
        f"{event_cell}"
        f"<td>{stat.bag_time_start_seconds:.2f} – {stat.bag_time_end_seconds:.2f}</td>"
        f"<td>{stat.pair_count}</td>"
        f"<td>{stat.translation_error_mean_meters:.4f}</td>"
        f"<td>{stat.translation_error_p95_meters:.4f}</td>"
        f"<td>{stat.translation_error_max_meters:.4f}</td>"
        f"<td>{heading_cell}</td>"
        "</tr>"
    )


def _render_event_cell_markdown(stat: RealVsSimCorrelationWindowStats) -> str:
    """Render the Event column cell for a per-window correlation stat row."""

    if stat.event_name is None:
        return "—"
    parts = [f"`{stat.event_name}`"]
    if stat.event_source and stat.event_source != "external":
        parts.append(f"({stat.event_source})")
    if stat.event_tags:
        parts.append(", ".join(f"`{tag}`" for tag in stat.event_tags))
    return " ".join(parts)


def _provenance_markdown_lines(provenance: RoutePolicyScenarioCIReviewProvenance | None) -> list[str]:
    """Render the provenance block as Markdown bullets, or an empty list if absent."""

    if provenance is None:
        return []
    lines: list[str] = ["## Run Provenance", ""]
    lines.append(f"- Kind: {provenance.kind}")
    lines.append(f"- Generated at: {provenance.generated_at}")
    optional_pairs: tuple[tuple[str, str | None], ...] = (
        ("Scene id", provenance.scene_id),
        ("Scenario set id", provenance.scenario_set_id),
        ("Asset source", provenance.asset_source),
        ("Policy version", provenance.policy_version),
        ("Env contract version", provenance.env_contract_version),
        ("Matrix hash", provenance.matrix_hash),
        ("Correlation threshold profile", provenance.correlation_threshold_profile),
        ("Git commit", provenance.git_commit),
    )
    for label, value in optional_pairs:
        if value:
            lines.append(f"- {label}: `{value}`")
    if provenance.extra:
        for key, value in sorted(provenance.extra.items()):
            lines.append(f"- {key}: `{value}`")
    return lines


def _render_provenance_section_html(provenance: RoutePolicyScenarioCIReviewProvenance | None) -> str:
    """Render a self-contained HTML section for the provenance block."""

    if provenance is None:
        return ""
    kind_class = provenance.kind if provenance.kind in {"production", "synthetic"} else "unknown"
    rows: list[str] = [
        f'<tr><th>Kind</th><td><span class="pill {kind_class}">{escape(provenance.kind.upper())}</span></td></tr>',
        f"<tr><th>Generated at</th><td><code>{escape(provenance.generated_at)}</code></td></tr>",
    ]
    optional_pairs: tuple[tuple[str, str | None], ...] = (
        ("Scene id", provenance.scene_id),
        ("Scenario set id", provenance.scenario_set_id),
        ("Asset source", provenance.asset_source),
        ("Policy version", provenance.policy_version),
        ("Env contract version", provenance.env_contract_version),
        ("Matrix hash", provenance.matrix_hash),
        ("Correlation threshold profile", provenance.correlation_threshold_profile),
        ("Git commit", provenance.git_commit),
    )
    for label, value in optional_pairs:
        if value:
            rows.append(f"<tr><th>{escape(label)}</th><td><code>{escape(value)}</code></td></tr>")
    for key, value in sorted(provenance.extra.items()):
        rows.append(f"<tr><th>{escape(str(key))}</th><td><code>{escape(str(value))}</code></td></tr>")
    body = "\n".join(rows)
    return f"<section><h2>Run Provenance</h2><table><tbody>{body}</tbody></table></section>"


def _render_adoption_section_html(adoption: RoutePolicyScenarioCIReviewAdoption | None) -> str:
    if adoption is None:
        return ""
    status_pill = "pass" if adoption.adopted else "fail"
    status_label = "ADOPTED" if adoption.adopted else "BLOCKED"
    push_branches = _display_branches(adoption.push_branches)
    pr_branches = _display_branches(adoption.pull_request_branches)
    diff_block = (
        f'<pre class="diff">{_render_diff_html(adoption.workflow_diff)}</pre>'
        if adoption.workflow_diff
        else "<p>No diff available between manual and adopted workflows.</p>"
    )
    return f"""<section>
      <h2>Adopted Workflow</h2>
      <p><span class="pill {status_pill}">{status_label}</span> {escape(adoption.adoption_id)}</p>
      <p>Trigger mode: <code>{escape(adoption.trigger_mode)}</code></p>
      <p>Adopted active path: <code>{escape(adoption.adopted_active_workflow_path)}</code></p>
      <p>Adopted source path: <code>{escape(adoption.adopted_source_workflow_path)}</code></p>
      <p>Push branches: {escape(push_branches)}</p>
      <p>Pull request branches: {escape(pr_branches)}</p>
      {diff_block}
    </section>"""


def _render_diff_html(diff_text: str) -> str:
    rendered_lines: list[str] = []
    for raw_line in diff_text.splitlines():
        escaped = escape(raw_line)
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            rendered_lines.append(f'<span class="hunk">{escaped}</span>')
        elif raw_line.startswith("@@"):
            rendered_lines.append(f'<span class="hunk">{escaped}</span>')
        elif raw_line.startswith("+"):
            rendered_lines.append(f'<span class="add">{escaped}</span>')
        elif raw_line.startswith("-"):
            rendered_lines.append(f'<span class="del">{escaped}</span>')
        else:
            rendered_lines.append(escaped)
    return "\n".join(rendered_lines)


def _display_branches(branches: Sequence[str]) -> str:
    return ", ".join(branches) if branches else "n/a"


def _review_shard_from_run(shard_run: RoutePolicyScenarioShardRunSummary) -> RoutePolicyScenarioCIReviewShard:
    return RoutePolicyScenarioCIReviewShard(
        shard_id=shard_run.shard_id,
        passed=shard_run.passed,
        scenario_count=shard_run.scenario_count,
        report_count=shard_run.report_count,
        run_path=shard_run.run_path,
        history_path=shard_run.history_path,
        report_paths=shard_run.report_paths,
        metadata=shard_run.metadata,
    )


def _optional_link(path: str | None) -> str:
    if path is None:
        return "n/a"
    escaped = escape(path)
    return f'<a href="{escaped}">{escaped}</a>'


def _sample_notice(artifact: RoutePolicyScenarioCIReviewArtifact) -> str | None:
    notice = artifact.metadata.get("sampleNotice")
    return str(notice) if notice else None


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
