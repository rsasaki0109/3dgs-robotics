"""Tests for the interaction-metrics aggregator (PR D4).

PR D / D2 / D3 added the multi-agent contract layer through the run
loop. PR D4 closes the shard-merge side: when scenario results carry
``interactionMetricsValues`` in their metadata, the merge report
attaches an :class:`InteractionMetricsAggregate` with per-key mean / p95
/ max / sampleCount statistics. Legacy ego-only runs (no values) leave
the aggregate as ``None``.

The actual per-step metric collection is a follow-up that needs env
hooks; these tests fabricate synthetic ``interactionMetricsValues``
mappings to exercise the aggregator in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    INTERACTION_METRICS_AGGREGATE_VERSION,
    InteractionMetricKeyStats,
    InteractionMetricsAggregate,
    SCENARIO_INTERACTION_METRIC_VALUES_KEY,
    aggregate_interaction_metrics_across_scenarios,
    interaction_metrics_aggregate_from_dict,
    load_route_policy_scenario_shard_merge_json,
    merge_route_policy_scenario_shard_runs,
    route_policy_scenario_shard_merge_from_dict,
    write_route_policy_scenario_shard_merge_json,
)


# ---------------------------------------------------- aggregate stat helpers


def test_aggregator_returns_none_for_legacy_scenarios() -> None:
    legacy_metadata = [
        {"scenarioId": "ego-only-1"},
        {"scenarioId": "ego-only-2"},
    ]
    assert aggregate_interaction_metrics_across_scenarios(legacy_metadata) is None


def test_aggregator_returns_none_when_values_present_but_non_numeric() -> None:
    metadata = [
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"min-peer-separation": "n/a"}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"min-peer-separation": None}},
    ]
    assert aggregate_interaction_metrics_across_scenarios(metadata) is None


def test_aggregator_computes_mean_p95_max_per_key() -> None:
    metadata = [
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"a": 1.0, "b": 10.0}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"a": 2.0, "b": 20.0}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"a": 3.0, "b": 30.0}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"a": 4.0, "b": 40.0}},
    ]
    aggregate = aggregate_interaction_metrics_across_scenarios(metadata)
    assert aggregate is not None
    assert aggregate.sample_scenario_count == 4
    a = aggregate.per_key_stats["a"]
    assert a.mean == pytest.approx(2.5)
    assert a.maximum == pytest.approx(4.0)
    assert a.sample_count == 4
    b = aggregate.per_key_stats["b"]
    assert b.mean == pytest.approx(25.0)
    assert b.maximum == pytest.approx(40.0)
    assert b.sample_count == 4


def test_aggregator_single_sample_falls_back_to_value_as_p95() -> None:
    metadata = [{SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 7.0}}]
    aggregate = aggregate_interaction_metrics_across_scenarios(metadata)
    assert aggregate is not None
    stats = aggregate.per_key_stats["k"]
    assert stats.mean == pytest.approx(7.0)
    assert stats.p95 == pytest.approx(7.0)
    assert stats.maximum == pytest.approx(7.0)
    assert stats.sample_count == 1


def test_aggregator_skips_non_finite_samples() -> None:
    metadata = [
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 1.0}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": float("nan")}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": float("inf")}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 3.0}},
    ]
    aggregate = aggregate_interaction_metrics_across_scenarios(metadata)
    assert aggregate is not None
    stats = aggregate.per_key_stats["k"]
    assert stats.sample_count == 2
    assert stats.mean == pytest.approx(2.0)


def test_aggregator_sample_scenario_count_only_counts_contributors() -> None:
    metadata = [
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 1.0}},
        {"scenarioId": "legacy"},  # no values -> not counted
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 2.0}},
        {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {}},  # empty -> not counted
    ]
    aggregate = aggregate_interaction_metrics_across_scenarios(metadata)
    assert aggregate is not None
    assert aggregate.sample_scenario_count == 2


def test_aggregate_round_trips_through_to_dict_from_dict() -> None:
    aggregate = InteractionMetricsAggregate(
        per_key_stats={
            "b-key": InteractionMetricKeyStats(mean=2.5, p95=3.8, maximum=4.0, sample_count=4),
            "a-key": InteractionMetricKeyStats(mean=1.5, p95=1.9, maximum=2.0, sample_count=4),
        },
        sample_scenario_count=4,
    )
    payload = aggregate.to_dict()
    assert payload["version"] == INTERACTION_METRICS_AGGREGATE_VERSION
    # Keys must be sorted in JSON output for stable diffs.
    assert list(payload["perKeyStats"]) == ["a-key", "b-key"]
    restored = interaction_metrics_aggregate_from_dict(payload)
    assert restored == aggregate


# ----------------------------------------------------- merge report wiring


def test_merge_attaches_aggregate_when_values_present(tmp_path: Path) -> None:
    shard_run = _build_shard_run(
        tmp_path,
        scenario_metadatas=[
            {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"min-peer-separation": 0.4}},
            {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"min-peer-separation": 0.6}},
        ],
    )
    merge = merge_route_policy_scenario_shard_runs([shard_run], merge_id="multi-agent-merge")
    assert merge.interaction_metrics_aggregate is not None
    stats = merge.interaction_metrics_aggregate.per_key_stats["min-peer-separation"]
    assert stats.mean == pytest.approx(0.5)
    assert stats.sample_count == 2


def test_merge_omits_aggregate_when_no_values(tmp_path: Path) -> None:
    shard_run = _build_shard_run(
        tmp_path,
        scenario_metadatas=[
            {"scenarioId": "legacy-1"},
            {"scenarioId": "legacy-2"},
        ],
    )
    merge = merge_route_policy_scenario_shard_runs([shard_run], merge_id="legacy-merge")
    assert merge.interaction_metrics_aggregate is None
    assert "interactionMetricsAggregate" not in merge.to_dict()


def test_merge_report_round_trips_with_aggregate(tmp_path: Path) -> None:
    shard_run = _build_shard_run(
        tmp_path,
        scenario_metadatas=[
            {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 1.0}},
            {SCENARIO_INTERACTION_METRIC_VALUES_KEY: {"k": 5.0}},
        ],
    )
    merge = merge_route_policy_scenario_shard_runs([shard_run], merge_id="round-trip-merge")
    output = tmp_path / "merge.json"
    write_route_policy_scenario_shard_merge_json(output, merge)
    reloaded = load_route_policy_scenario_shard_merge_json(output)
    assert reloaded.interaction_metrics_aggregate == merge.interaction_metrics_aggregate
    # to_dict() <-> from_dict() round-trip too.
    restored = route_policy_scenario_shard_merge_from_dict(merge.to_dict())
    assert restored.interaction_metrics_aggregate == merge.interaction_metrics_aggregate


# ----------------------------------------------------------------- helpers


def _build_shard_run(tmp_path: Path, *, scenario_metadatas):
    """Synthesize a RoutePolicyScenarioSetRunReport with the given scenario metadata."""

    import json

    from gs_sim2real.sim import (
        RoutePolicyScenarioRunResult,
        RoutePolicyScenarioSetRunReport,
    )
    from gs_sim2real.sim.policy_benchmark_history import (
        build_route_policy_benchmark_history,
    )

    scenario_results = []
    for index, metadata in enumerate(scenario_metadatas):
        # Write a minimal benchmark report JSON directly so the history
        # aggregator can read it without us instantiating the full
        # RoutePolicyBenchmarkReport dataclass (its shape isn't load-bearing
        # for the aggregator tests we care about here).
        report_path = tmp_path / f"scenario-{index}.json"
        report_path.write_text(
            json.dumps(
                {
                    "recordType": "route-policy-benchmark-report",
                    "version": "gs-mapper-route-policy-benchmark/v1",
                    "benchmarkId": f"shard-test-scenario-{index}",
                    "passed": True,
                    "bestPolicyName": "direct-baseline",
                    "summary": {
                        "evaluationId": f"shard-test-scenario-{index}",
                        "bestPolicyName": "direct-baseline",
                        "policyCount": 1,
                        "policies": [
                            {
                                "policyName": "direct-baseline",
                                "passed": True,
                                "metrics": {
                                    "success-rate": 1.0,
                                    "collision-rate": 0.0,
                                    "truncation-rate": 0.0,
                                    "mean-reward": 1.0,
                                },
                                "failedChecks": [],
                            }
                        ],
                    },
                    "modelSummary": {},
                    "metadata": {"sceneId": "unit-scene"},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        scenario_results.append(
            RoutePolicyScenarioRunResult(
                scenario_id=f"scenario-{index}",
                benchmark_id=f"shard-test-scenario-{index}",
                report_path=report_path.as_posix(),
                markdown_path=None,
                passed=True,
                best_policy_name="direct-baseline",
                scene_catalog="scenes.json",
                scene_id="unit-scene",
                goal_suite_path=None,
                episode_count=1,
                seed_start=0,
                max_steps=10,
                metadata=metadata,
            )
        )
    history = build_route_policy_benchmark_history(
        tuple(result.report_path for result in scenario_results),
        baseline_report=None,
        history_id="shard-history",
    )
    return RoutePolicyScenarioSetRunReport(
        scenario_set_id="shard-test",
        scenario_results=tuple(scenario_results),
        history=history,
        policy_registry_path="registry.json",
    )
