"""Tests for route policy benchmark reports and CLI runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gs_sim2real import cli
from gs_sim2real.cli import build_parser
from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RoutePolicyBenchmarkHistoryReport,
    RoutePolicyBenchmarkPolicySnapshot,
    RoutePolicyBenchmarkRegressionThresholds,
    RoutePolicyBenchmarkSnapshot,
    RoutePolicyGoalSpec,
    RoutePolicyGoalSuite,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyRegistry,
    RoutePolicyRegistryEntry,
    RoutePolicyScenarioCIReviewArtifact,
    RoutePolicyScenarioCIReviewProvenance,
    RoutePolicyScenarioCIManifest,
    RoutePolicyScenarioCIMergeJob,
    RoutePolicyScenarioCIShardJob,
    RoutePolicyScenarioMatrix,
    RoutePolicyScenarioSet,
    RoutePolicyScenarioCIWorkflowConfig,
    RoutePolicyScenarioShardMergeReport,
    RoutePolicyScenarioShardRunSummary,
    RoutePolicyScenarioSpec,
    RouteRewardWeights,
    activate_route_policy_scenario_ci_workflow,
    build_route_policy_benchmark_history,
    build_route_policy_scenario_ci_manifest,
    build_route_policy_scenario_ci_review_artifact,
    build_simulation_catalog,
    collect_route_policy_dataset,
    expand_route_policy_scenario_matrix,
    expand_route_policy_scenario_matrix_to_directory,
    load_route_policy_benchmark_history_json,
    load_route_policy_goal_suite_json,
    load_route_policy_imitation_model_json,
    load_route_policy_registry_json,
    load_route_policy_scenario_ci_manifest_json,
    load_route_policy_scenario_ci_review_json,
    load_route_policy_scenario_ci_workflow_activation_json,
    load_route_policy_scenario_ci_workflow_json,
    load_route_policy_scenario_ci_workflow_promotion_json,
    load_route_policy_scenario_ci_workflow_validation_json,
    load_route_policy_scenario_matrix_expansion_json,
    load_route_policy_scenario_matrix_json,
    load_route_policy_scenario_set_run_json,
    load_route_policy_scenario_shard_merge_json,
    load_route_policy_scenario_shard_plan_json,
    merge_route_policy_scenario_shard_run_jsons,
    promote_route_policy_scenario_ci_workflow,
    render_route_policy_benchmark_history_markdown,
    render_route_policy_benchmark_markdown,
    render_route_policy_scenario_ci_manifest_markdown,
    render_route_policy_scenario_ci_review_html,
    render_route_policy_scenario_ci_review_markdown,
    render_route_policy_scenario_ci_workflow_activation_markdown,
    render_route_policy_scenario_ci_workflow_markdown,
    render_route_policy_scenario_ci_workflow_promotion_markdown,
    render_route_policy_scenario_ci_workflow_validation_markdown,
    render_route_policy_scenario_matrix_markdown,
    render_route_policy_scenario_set_markdown,
    render_route_policy_scenario_shard_merge_markdown,
    render_route_policy_scenario_shard_plan_markdown,
    run_route_policy_imitation_benchmark,
    run_route_policy_registry_benchmark,
    run_route_policy_scenario_set,
    materialize_route_policy_scenario_ci_workflow,
    validate_route_policy_scenario_ci_workflow,
    write_route_policy_benchmark_history_json,
    write_route_policy_goal_suite_json,
    write_route_policy_imitation_model_json,
    write_route_policy_registry_json,
    write_route_policy_scenario_ci_manifest_json,
    write_route_policy_scenario_ci_review_bundle,
    write_route_policy_scenario_ci_review_json,
    write_route_policy_scenario_ci_workflow_activation_json,
    write_route_policy_scenario_ci_workflow_json,
    write_route_policy_scenario_ci_workflow_promotion_json,
    write_route_policy_scenario_ci_workflow_validation_json,
    write_route_policy_scenario_ci_workflow_yaml,
    write_route_policy_scenario_matrix_expansion_json,
    write_route_policy_scenario_matrix_json,
    write_route_policy_scenario_set_json,
    write_route_policy_scenario_set_run_json,
    write_route_policy_scenario_shard_merge_json,
    write_route_policy_scenario_shard_plan_json,
    write_route_policy_scenario_shards_from_expansion,
    write_route_policy_transitions_jsonl,
)
from gs_sim2real.robotics import (
    BagPoseSample,
    BagPoseStream,
    SimPoseSample,
    correlate_against_sim_trajectory,
    write_real_vs_sim_correlation_report_json,
)


def test_run_route_policy_imitation_benchmark_compares_direct_baseline() -> None:
    from gs_sim2real.sim import build_route_policy_replay_batch, build_route_policy_replay_schema
    from gs_sim2real.sim import fit_route_policy_imitation_model

    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-benchmark",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))

    report = run_route_policy_imitation_benchmark(
        (build_adapter(),),
        model,
        episode_count=2,
        benchmark_id="unit-benchmark",
        include_direct_baseline=True,
        goals=goals,
        max_steps=4,
    )

    assert report.passed
    assert report.best_policy_name in {"direct", "imitation"}
    assert [result.policy_name for result in report.evaluation.results] == ["direct", "imitation"]
    assert "Route Policy Benchmark: unit-benchmark" in render_route_policy_benchmark_markdown(report)


def test_route_policy_benchmark_cli_fits_saves_and_writes_reports(tmp_path: Path) -> None:
    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-benchmark-cli",
        goals=goals,
    )
    transitions_path = write_route_policy_transitions_jsonl(tmp_path / "transitions.jsonl", dataset)
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    model_path = tmp_path / "model.json"

    args = build_parser().parse_args(
        [
            "route-policy-benchmark",
            "--transitions-jsonl",
            str(transitions_path),
            "--scene-catalog",
            str(catalog_path),
            "--scene-id",
            "unit-scene",
            "--benchmark-id",
            "unit-cli-benchmark",
            "--episode-count",
            "2",
            "--seed-start",
            "0",
            "--max-steps",
            "4",
            "--goal",
            "0.25",
            "0.0",
            "0.0",
            "--goal",
            "0.5",
            "0.0",
            "0.0",
            "--action-keys",
            *target_position_keys(),
            "--include-direct-baseline",
            "--min-success-rate",
            "1.0",
            "--max-collision-rate",
            "0.0",
            "--max-truncation-rate",
            "0.0",
            "--min-episode-count",
            "2",
            "--min-transition-count",
            "2",
            "--model-output",
            str(model_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_benchmark(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    model = load_route_policy_imitation_model_json(model_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert report["recordType"] == "route-policy-benchmark-report"
    assert report["benchmarkId"] == "unit-cli-benchmark"
    assert report["passed"] is True
    assert report["summary"]["policyCount"] == 2
    assert report["modelSummary"]["sampleCount"] == 2
    assert model.sample_count == 2
    assert "| Policy | Pass | Success |" in markdown


def test_route_policy_registry_benchmark_loads_named_policies_and_goal_suite(tmp_path: Path) -> None:
    from gs_sim2real.sim import build_route_policy_replay_batch, build_route_policy_replay_schema
    from gs_sim2real.sim import fit_route_policy_imitation_model

    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-registry-train",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))
    model_path = write_route_policy_imitation_model_json(tmp_path / "imitation.json", model)
    registry_path = write_route_policy_registry_json(
        tmp_path / "policies.json",
        RoutePolicyRegistry(
            registry_id="unit-policies",
            policies=(
                RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),
                RoutePolicyRegistryEntry(
                    policy_name="imitation-k1",
                    policy_type="imitation-model",
                    model_path=model_path.name,
                ),
            ),
        ),
    )
    suite_path = write_route_policy_goal_suite_json(
        tmp_path / "goals.json",
        RoutePolicyGoalSuite(
            suite_id="unit-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(
                RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),
                RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),
            ),
        ),
    )

    registry = load_route_policy_registry_json(registry_path)
    suite = load_route_policy_goal_suite_json(suite_path)
    report = run_route_policy_registry_benchmark(
        (build_adapter(),),
        registry,
        episode_count=2,
        benchmark_id="unit-registry-benchmark",
        registry_base_path=tmp_path,
        goals=suite.to_goals(frame_id="generic_world"),
        max_steps=4,
    )

    assert report.passed
    assert [result.policy_name for result in report.evaluation.results] == ["direct", "imitation-k1"]
    assert report.to_dict()["modelSummary"]["registry"]["registryId"] == "unit-policies"
    assert report.to_dict()["modelSummary"]["policies"][1]["sampleCount"] == 2


def test_route_policy_benchmark_cli_uses_registry_and_goal_suite(tmp_path: Path) -> None:
    from gs_sim2real.sim import build_route_policy_replay_batch, build_route_policy_replay_schema
    from gs_sim2real.sim import fit_route_policy_imitation_model

    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-registry-cli-train",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))
    write_route_policy_imitation_model_json(tmp_path / "model.json", model)
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-registry",
            policies=(
                RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),
                RoutePolicyRegistryEntry(
                    policy_name="imitation",
                    policy_type="imitation-model",
                    model_path="model.json",
                ),
            ),
        ),
    )
    suite_path = write_route_policy_goal_suite_json(
        tmp_path / "goal-suite.json",
        RoutePolicyGoalSuite(
            suite_id="unit-cli-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(
                RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),
                RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),
            ),
        ),
    )
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    output_path = tmp_path / "registry-report.json"

    args = build_parser().parse_args(
        [
            "route-policy-benchmark",
            "--policy-registry",
            str(registry_path),
            "--goal-suite",
            str(suite_path),
            "--scene-catalog",
            str(catalog_path),
            "--benchmark-id",
            "unit-cli-registry-benchmark",
            "--episode-count",
            "2",
            "--max-steps",
            "4",
            "--output",
            str(output_path),
        ]
    )

    cli.cmd_route_policy_benchmark(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["benchmarkId"] == "unit-cli-registry-benchmark"
    assert report["passed"] is True
    assert report["summary"]["policyCount"] == 2
    assert [policy["policyName"] for policy in report["summary"]["policies"]] == ["direct", "imitation"]
    assert report["modelSummary"]["registry"]["registryId"] == "unit-cli-registry"
    assert report["metadata"]["goalSuite"] == str(suite_path)


def test_route_policy_benchmark_history_flags_baseline_regressions(tmp_path: Path) -> None:
    baseline_path = write_benchmark_report_fixture(
        tmp_path / "baseline-report.json",
        benchmark_id="unit-history-baseline",
        policies={
            "direct": {
                "success-rate": 1.0,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    current_path = write_benchmark_report_fixture(
        tmp_path / "current-report.json",
        benchmark_id="unit-history-current",
        policies={
            "direct": {
                "success-rate": 0.9,
                "collision-rate": 0.03,
                "truncation-rate": 0.0,
                "mean-reward": 1.6,
            }
        },
    )

    history = build_route_policy_benchmark_history(
        (current_path,),
        baseline_report=baseline_path,
        history_id="unit-history",
        thresholds=RoutePolicyBenchmarkRegressionThresholds(
            max_success_rate_drop=0.05,
            max_collision_rate_increase=0.01,
            max_truncation_rate_increase=0.01,
            max_mean_reward_drop=0.25,
        ),
    )
    history_path = write_route_policy_benchmark_history_json(tmp_path / "history.json", history)
    loaded = load_route_policy_benchmark_history_json(history_path)
    failed_checks = set(loaded.failed_checks)

    assert not loaded.passed
    assert "success-rate-regression:current-report:direct" in failed_checks
    assert "collision-rate-regression:current-report:direct" in failed_checks
    assert "mean-reward-regression:current-report:direct" in failed_checks
    assert loaded.to_dict()["aggregate"][0]["metrics"]["success-rate"]["mean"] == 0.9
    assert "Regression Gate" in render_route_policy_benchmark_history_markdown(loaded)


def test_route_policy_benchmark_history_cli_writes_artifacts(tmp_path: Path) -> None:
    baseline_path = write_benchmark_report_fixture(
        tmp_path / "baseline.json",
        benchmark_id="unit-cli-history-baseline",
        policies={
            "direct": {
                "success-rate": 1.0,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    current_path = write_benchmark_report_fixture(
        tmp_path / "current.json",
        benchmark_id="unit-cli-history-current",
        policies={
            "direct": {
                "success-rate": 0.97,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 1.95,
            }
        },
    )
    output_path = tmp_path / "history.json"
    markdown_path = tmp_path / "history.md"

    args = build_parser().parse_args(
        [
            "route-policy-benchmark-history",
            "--report",
            str(current_path),
            "--baseline-report",
            str(baseline_path),
            "--history-id",
            "unit-cli-history",
            "--max-success-rate-drop",
            "0.05",
            "--max-collision-rate-increase",
            "0.01",
            "--max-truncation-rate-increase",
            "0.01",
            "--max-mean-reward-drop",
            "0.10",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_benchmark_history(args)

    history = json.loads(output_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert history["recordType"] == "route-policy-benchmark-history"
    assert history["historyId"] == "unit-cli-history"
    assert history["passed"] is True
    assert history["reportCount"] == 1
    assert history["regressionChecks"][0]["checkId"] == "success-rate-regression:current:direct"
    assert "Route Policy Benchmark History: unit-cli-history" in markdown


def test_route_policy_benchmark_history_cli_exits_on_regression(tmp_path: Path) -> None:
    baseline_path = write_benchmark_report_fixture(
        tmp_path / "baseline.json",
        benchmark_id="unit-cli-gate-baseline",
        policies={
            "direct": {
                "success-rate": 1.0,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    current_path = write_benchmark_report_fixture(
        tmp_path / "current.json",
        benchmark_id="unit-cli-gate-current",
        policies={
            "direct": {
                "success-rate": 0.5,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    output_path = tmp_path / "history.json"
    args = build_parser().parse_args(
        [
            "route-policy-benchmark-history",
            "--report",
            str(current_path),
            "--baseline-report",
            str(baseline_path),
            "--max-success-rate-drop",
            "0.05",
            "--output",
            str(output_path),
            "--fail-on-regression",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_route_policy_benchmark_history(args)

    assert exc_info.value.code == 2
    history = json.loads(output_path.read_text(encoding="utf-8"))
    assert history["passed"] is False


def test_route_policy_scenario_set_runs_registry_across_goal_suites(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-scenario-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    near_goals = write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    far_goals = write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-scenarios",
        policy_registry_path=registry_path.name,
        episode_count=1,
        seed_start=0,
        max_steps=4,
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="near",
                scene_catalog=catalog_path.name,
                goal_suite_path=near_goals.name,
            ),
            RoutePolicyScenarioSpec(
                scenario_id="far",
                scene_catalog=catalog_path.name,
                goal_suite_path=far_goals.name,
            ),
        ),
    )
    registry = load_route_policy_registry_json(registry_path)

    report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "history.json",
        history_markdown_output=tmp_path / "history.md",
    )
    run_path = write_route_policy_scenario_set_run_json(tmp_path / "scenario-run.json", report)
    loaded = load_route_policy_scenario_set_run_json(run_path)

    assert loaded.passed
    assert loaded.scenario_count == 2
    assert loaded.history.to_dict()["reportCount"] == 2
    assert [result.scenario_id for result in loaded.scenario_results] == ["near", "far"]
    assert all(Path(result.report_path).exists() for result in loaded.scenario_results)
    assert (tmp_path / "history.json").exists()
    assert "Route Policy Scenario Set: unit-scenarios" in render_route_policy_scenario_set_markdown(loaded)


def _write_unit_correlation_report_json(path: Path) -> Path:
    """Write a tiny pre-computed real-vs-sim correlation report at ``path``."""

    bag_stream = BagPoseStream(
        samples=(
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )
    sim_samples = (
        SimPoseSample(timestamp_seconds=0.01, position=(0.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    )
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.05)
    return write_real_vs_sim_correlation_report_json(path, report)


def test_route_policy_scenario_set_attaches_correlation_reports(tmp_path: Path) -> None:
    """Pre-computed correlation reports must round-trip through the run report and surface in Markdown."""

    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-correlation-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    near_goals = write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-correlation",
        policy_registry_path=registry_path.name,
        episode_count=1,
        seed_start=0,
        max_steps=4,
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="near",
                scene_catalog=catalog_path.name,
                goal_suite_path=near_goals.name,
            ),
        ),
    )
    registry = load_route_policy_registry_json(registry_path)
    correlation_path = _write_unit_correlation_report_json(tmp_path / "correlation.json")

    report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "history.json",
        correlation_report_paths=[correlation_path.name],
    )

    assert report.correlation_report_count == 1
    assert report.correlation_reports[0].matched_pair_count == 2
    assert report.correlation_report_paths[0] == str(tmp_path / "correlation.json")

    run_path = write_route_policy_scenario_set_run_json(tmp_path / "scenario-run.json", report)
    loaded = load_route_policy_scenario_set_run_json(run_path)
    assert loaded.correlation_report_count == 1
    assert loaded.correlation_reports[0].bag_source.source_topic == "/gnss/fix"
    assert loaded.correlation_reports[0].translation_error_mean_meters == pytest.approx(0.05, rel=1e-6)
    assert loaded.correlation_report_paths == report.correlation_report_paths

    markdown = render_route_policy_scenario_set_markdown(loaded)
    assert "Real-vs-sim correlation" in markdown
    assert "/gnss/fix" in markdown


def test_route_policy_scenario_set_cli_writes_reports_and_history(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-scenario-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    scenario_set_path = write_route_policy_scenario_set_json(
        tmp_path / "scenarios.json",
        RoutePolicyScenarioSet(
            scenario_set_id="unit-cli-scenarios",
            policy_registry_path="registry.json",
            episode_count=1,
            seed_start=0,
            max_steps=4,
            scenarios=(
                RoutePolicyScenarioSpec(
                    scenario_id="near",
                    scene_catalog=catalog_path.name,
                    goal_suite_path="near-goals.json",
                ),
                RoutePolicyScenarioSpec(
                    scenario_id="far",
                    scene_catalog=catalog_path.name,
                    goal_suite_path="far-goals.json",
                ),
            ),
        ),
    )
    output_path = tmp_path / "scenario-run.json"
    markdown_path = tmp_path / "scenario-run.md"
    history_path = tmp_path / "history.json"
    history_markdown_path = tmp_path / "history.md"

    args = build_parser().parse_args(
        [
            "route-policy-scenario-set",
            "--scenario-set",
            str(scenario_set_path),
            "--report-dir",
            str(tmp_path / "reports"),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--history-output",
            str(history_path),
            "--history-markdown-output",
            str(history_markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_set(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert report["recordType"] == "route-policy-scenario-set-run"
    assert report["passed"] is True
    assert report["scenarioSetId"] == "unit-cli-scenarios"
    assert [result["scenarioId"] for result in report["scenarioResults"]] == ["near", "far"]
    assert history["recordType"] == "route-policy-benchmark-history"
    assert history["reportCount"] == 2
    assert "Route Policy Scenario Set: unit-cli-scenarios" in markdown_path.read_text(encoding="utf-8")
    assert "Route Policy Benchmark History" in history_markdown_path.read_text(encoding="utf-8")


def test_route_policy_scenario_matrix_expands_to_registry_scenario_sets(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "direct-registry.json",
        RoutePolicyRegistry(
            registry_id="direct-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_registry_json(
        tmp_path / "baseline-registry.json",
        RoutePolicyRegistry(
            registry_id="baseline-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    matrix = RoutePolicyScenarioMatrix(
        matrix_id="unit-matrix",
        registries=(
            RoutePolicyMatrixRegistrySpec("direct", "direct-registry.json"),
            RoutePolicyMatrixRegistrySpec("baseline", "baseline-registry.json"),
        ),
        scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
        goal_suites=(
            RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),
            RoutePolicyMatrixGoalSuiteSpec("far", "far-goals.json"),
        ),
        configs=(
            RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),
            RoutePolicyMatrixConfigSpec("long", episode_count=2, seed_start=10, max_steps=8),
        ),
        episode_count=3,
        seed_start=100,
        max_steps=6,
    )
    matrix_path = write_route_policy_scenario_matrix_json(tmp_path / "matrix.json", matrix)
    loaded_matrix = load_route_policy_scenario_matrix_json(matrix_path)
    scenario_sets = expand_route_policy_scenario_matrix(loaded_matrix)

    assert [scenario_set.scenario_set_id for scenario_set in scenario_sets] == [
        "unit-matrix-direct",
        "unit-matrix-baseline",
    ]
    assert [scenario_set.scenario_count for scenario_set in scenario_sets] == [4, 4]
    assert [scenario.scenario_id for scenario in scenario_sets[0].scenarios] == [
        "unit-near-short",
        "unit-near-long",
        "unit-far-short",
        "unit-far-long",
    ]

    expansion = expand_route_policy_scenario_matrix_to_directory(
        loaded_matrix,
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    expansion_path = write_route_policy_scenario_matrix_expansion_json(tmp_path / "expansion.json", expansion)
    loaded_expansion = load_route_policy_scenario_matrix_expansion_json(expansion_path)

    assert loaded_expansion.scenario_set_count == 2
    assert loaded_expansion.scenario_count == 8
    assert loaded_expansion.scenario_sets[0].policy_registry_path == "../direct-registry.json"
    assert all(output.scenario_set_path is not None for output in loaded_expansion.outputs)
    assert all(Path(output.scenario_set_path or "").exists() for output in loaded_expansion.outputs)
    assert "Route Policy Scenario Matrix: unit-matrix" in render_route_policy_scenario_matrix_markdown(loaded_expansion)


def test_route_policy_scenario_matrix_cli_writes_generated_sets(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "direct-registry.json",
        RoutePolicyRegistry(
            registry_id="direct-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    matrix_path = write_route_policy_scenario_matrix_json(
        tmp_path / "matrix.json",
        RoutePolicyScenarioMatrix(
            matrix_id="unit-cli-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "direct-registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
    )
    index_path = tmp_path / "matrix-expansion.json"
    markdown_path = tmp_path / "matrix-expansion.md"

    args = build_parser().parse_args(
        [
            "route-policy-scenario-matrix",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(tmp_path / "generated"),
            "--index-output",
            str(index_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_matrix(args)

    expansion = json.loads(index_path.read_text(encoding="utf-8"))

    assert expansion["recordType"] == "route-policy-scenario-matrix-expansion"
    assert expansion["matrixId"] == "unit-cli-matrix"
    assert expansion["scenarioSetCount"] == 1
    assert expansion["scenarioCount"] == 1
    assert Path(expansion["outputs"][0]["scenarioSetPath"]).exists()
    assert "Route Policy Scenario Matrix: unit-cli-matrix" in markdown_path.read_text(encoding="utf-8")


def test_route_policy_scenario_shards_split_and_merge_runs(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-shard-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    matrix = RoutePolicyScenarioMatrix(
        matrix_id="unit-shard-matrix",
        registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
        scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
        goal_suites=(
            RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),
            RoutePolicyMatrixGoalSuiteSpec("far", "far-goals.json"),
        ),
        configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        matrix,
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-shards",
    )
    plan_path = write_route_policy_scenario_shard_plan_json(tmp_path / "shard-plan.json", plan)
    loaded_plan = load_route_policy_scenario_shard_plan_json(plan_path)

    assert loaded_plan.shard_count == 2
    assert loaded_plan.scenario_count == 2
    assert [shard.scenario_count for shard in loaded_plan.shards] == [1, 1]
    assert all(shard.scenario_set_path is not None for shard in loaded_plan.shards)
    assert all(Path(shard.scenario_set_path or "").exists() for shard in loaded_plan.shards)
    assert loaded_plan.scenario_sets[0].policy_registry_path == "../registry.json"
    assert "Route Policy Scenario Shards: unit-shards" in render_route_policy_scenario_shard_plan_markdown(loaded_plan)

    registry = load_route_policy_registry_json(registry_path)
    run_paths: list[Path] = []
    for scenario_set in loaded_plan.scenario_sets:
        run = run_route_policy_scenario_set(
            scenario_set,
            registry,
            report_dir=tmp_path / "reports" / scenario_set.scenario_set_id,
            scenario_set_base_path=tmp_path / "shards",
            registry_base_path=tmp_path,
            policy_registry_path=registry_path,
            history_output=tmp_path / "histories" / f"{scenario_set.scenario_set_id}.json",
            history_markdown_output=tmp_path / "histories" / f"{scenario_set.scenario_set_id}.md",
        )
        run_paths.append(
            write_route_policy_scenario_set_run_json(tmp_path / "runs" / f"{scenario_set.scenario_set_id}.json", run)
        )

    merge = merge_route_policy_scenario_shard_run_jsons(
        tuple(run_paths),
        merge_id="unit-shard-merge",
        history_output=tmp_path / "merged-history.json",
        history_markdown_output=tmp_path / "merged-history.md",
    )
    merge_path = write_route_policy_scenario_shard_merge_json(tmp_path / "shard-merge.json", merge)
    loaded_merge = load_route_policy_scenario_shard_merge_json(merge_path)

    assert loaded_merge.passed
    assert loaded_merge.shard_count == 2
    assert loaded_merge.scenario_count == 2
    assert loaded_merge.history.to_dict()["reportCount"] == 2
    assert (tmp_path / "merged-history.json").exists()
    assert "Route Policy Scenario Shard Merge: unit-shard-merge" in render_route_policy_scenario_shard_merge_markdown(
        loaded_merge
    )


def test_route_policy_scenario_set_honours_sensor_noise_profile(tmp_path: Path) -> None:
    from gs_sim2real.sim import (
        RoutePolicyScenarioSet,
        RoutePolicyScenarioSpec,
        RoutePolicySensorNoiseProfile,
        route_policy_scenario_set_from_dict,
        write_route_policy_scenario_set_json,
        write_route_policy_sensor_noise_profile_json,
    )

    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-noise-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    noise_profile_path = write_route_policy_sensor_noise_profile_json(
        tmp_path / "noise.json",
        RoutePolicySensorNoiseProfile(
            profile_id="scenario-noise",
            pose_position_std_meters=0.20,
            pose_heading_std_radians=0.04,
            goal_position_std_meters=0.15,
        ),
    )

    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-noise-scenarios",
        policy_registry_path=str(registry_path),
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="noise-short",
                scene_catalog=str(catalog_path.name),
                scene_id="unit-scene",
                goal_suite_path="near-goals.json",
                episode_count=1,
                seed_start=0,
                max_steps=4,
                sensor_noise_profile_path=str(noise_profile_path.name),
            ),
        ),
    )
    scenario_set_path = write_route_policy_scenario_set_json(tmp_path / "scenario-set.json", scenario_set)

    # JSON round-trip preserves the sensor noise profile path.
    loaded_set = route_policy_scenario_set_from_dict(json.loads(scenario_set_path.read_text(encoding="utf-8")))
    assert loaded_set.scenarios[0].sensor_noise_profile_path == str(noise_profile_path.name)

    # Running the scenario set picks up the noise profile and records a
    # non-zero noise-induced offset between the true and observed pose.
    registry = load_route_policy_registry_json(registry_path)
    run = run_route_policy_scenario_set(
        loaded_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
    )
    assert run.scenario_count == 1
    # The direct-goal policy still succeeds with mild noise because the
    # per-step goal distance dominates the drift.
    assert run.scenario_results[0].passed is True


def test_route_policy_scenario_set_round_trips_raw_sensor_noise_profile_path(tmp_path: Path) -> None:
    from gs_sim2real.sim import (
        RawSensorNoiseProfile,
        RoutePolicyScenarioSet,
        RoutePolicyScenarioSpec,
        route_policy_scenario_set_from_dict,
        write_raw_sensor_noise_profile_json,
        write_route_policy_scenario_set_json,
    )

    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-raw-noise-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    raw_profile_path = write_raw_sensor_noise_profile_json(
        tmp_path / "raw-noise.json",
        RawSensorNoiseProfile(
            profile_id="scenario-raw-noise",
            rgb_intensity_std=3.0,
            depth_range_std_meters=0.1,
            lidar_range_std_meters=0.05,
        ),
    )

    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-raw-noise-scenarios",
        policy_registry_path=str(registry_path),
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="raw-noise-short",
                scene_catalog=str(catalog_path.name),
                scene_id="unit-scene",
                goal_suite_path="near-goals.json",
                episode_count=1,
                seed_start=0,
                max_steps=4,
                raw_sensor_noise_profile_path=str(raw_profile_path.name),
            ),
        ),
    )
    scenario_set_path = write_route_policy_scenario_set_json(tmp_path / "scenario-set.json", scenario_set)

    loaded_set = route_policy_scenario_set_from_dict(json.loads(scenario_set_path.read_text(encoding="utf-8")))
    assert loaded_set.scenarios[0].raw_sensor_noise_profile_path == str(raw_profile_path.name)

    # Running the scenario picks the profile up without crashing, even though the
    # direct-goal rollout does not actually render an observation — the env still
    # loads and carries the profile end-to-end.
    registry = load_route_policy_registry_json(registry_path)
    run = run_route_policy_scenario_set(
        loaded_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
    )
    assert run.scenario_count == 1
    assert run.scenario_results[0].passed is True


def test_route_policy_scenario_matrix_preserves_raw_sensor_noise_profile_path() -> None:
    from gs_sim2real.sim import (
        RoutePolicyMatrixConfigSpec,
        RoutePolicyMatrixGoalSuiteSpec,
        RoutePolicyMatrixRegistrySpec,
        RoutePolicyMatrixSceneSpec,
        RoutePolicyScenarioMatrix,
        expand_route_policy_scenario_matrix,
        route_policy_matrix_config_spec_from_dict,
    )

    config = RoutePolicyMatrixConfigSpec(
        config_id="raw-noise-config",
        raw_sensor_noise_profile_path="raw-noise.json",
    )
    payload = config.to_dict()
    assert payload["rawSensorNoiseProfilePath"] == "raw-noise.json"
    assert route_policy_matrix_config_spec_from_dict(payload) == config

    # Expansion carries the profile path down to every generated scenario spec.
    matrix = RoutePolicyScenarioMatrix(
        matrix_id="raw-noise-matrix",
        registries=(RoutePolicyMatrixRegistrySpec("direct", "direct-registry.json"),),
        scenes=(RoutePolicyMatrixSceneSpec("unit", "scenes.json", scene_id="unit-scene"),),
        goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
        configs=(config,),
    )
    scenario_sets = expand_route_policy_scenario_matrix(matrix)
    assert scenario_sets[0].scenarios[0].raw_sensor_noise_profile_path == "raw-noise.json"


def test_route_policy_scenario_set_runs_mixed_reactive_dynamic_obstacles(tmp_path: Path) -> None:
    from gs_sim2real.sim import (
        DynamicObstacle,
        DynamicObstacleTimeline,
        DynamicObstacleWaypoint,
        RoutePolicyScenarioSet,
        RoutePolicyScenarioSpec,
        route_policy_scenario_set_from_dict,
        write_route_policy_dynamic_obstacle_timeline_json,
        write_route_policy_scenario_set_json,
    )

    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-mixed-reactive-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.1, 0.0, 0.0)),),
        ),
    )
    # Chase obstacle well outside the 0.1 m step hop so the direct-goal rollout
    # still lands at the goal without a dynamic-obstacle collision. Flee
    # retreats on the side; the static waypoint obstacle only contributes to
    # the obstacle count, so the scenario passes end-to-end.
    timeline_path = write_route_policy_dynamic_obstacle_timeline_json(
        tmp_path / "mixed-reactive.json",
        DynamicObstacleTimeline(
            timeline_id="mixed-reactive",
            obstacles=(
                DynamicObstacle(
                    obstacle_id="chaser",
                    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(3.0, 0.0, 0.0)),),
                    radius_meters=0.05,
                    chase_target_agent=True,
                    chase_speed_m_per_step=0.05,
                ),
                DynamicObstacle(
                    obstacle_id="runner",
                    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 2.0, 0.0)),),
                    radius_meters=0.05,
                    flee_from_agent=True,
                    chase_speed_m_per_step=0.05,
                ),
                DynamicObstacle(
                    obstacle_id="bollard",
                    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, -5.0, 0.0)),),
                    radius_meters=0.05,
                ),
            ),
        ),
    )

    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-mixed-reactive-scenarios",
        policy_registry_path=str(registry_path),
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="mixed-reactive-short",
                scene_catalog=str(catalog_path.name),
                scene_id="unit-scene",
                goal_suite_path="near-goals.json",
                episode_count=1,
                seed_start=0,
                max_steps=4,
                dynamic_obstacles_path=str(timeline_path.name),
            ),
        ),
    )
    scenario_set_path = write_route_policy_scenario_set_json(tmp_path / "scenario-set.json", scenario_set)

    loaded_set = route_policy_scenario_set_from_dict(json.loads(scenario_set_path.read_text(encoding="utf-8")))
    assert loaded_set.scenarios[0].dynamic_obstacles_path == str(timeline_path.name)

    registry = load_route_policy_registry_json(registry_path)
    run = run_route_policy_scenario_set(
        loaded_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
    )
    assert run.scenario_count == 1
    assert run.scenario_results[0].passed is True


def test_route_policy_scenario_shards_cli_writes_plan_and_merge(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-shard-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-cli-shard-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    expansion_path = write_route_policy_scenario_matrix_expansion_json(tmp_path / "expansion.json", expansion)
    plan_path = tmp_path / "shard-plan.json"
    plan_markdown_path = tmp_path / "shard-plan.md"
    plan_args = build_parser().parse_args(
        [
            "route-policy-scenario-shards",
            "--expansion",
            str(expansion_path),
            "--max-scenarios-per-shard",
            "1",
            "--shard-plan-id",
            "unit-cli-shards",
            "--output-dir",
            str(tmp_path / "shards"),
            "--index-output",
            str(plan_path),
            "--markdown-output",
            str(plan_markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_shards(plan_args)
    plan = load_route_policy_scenario_shard_plan_json(plan_path)

    assert plan.shard_plan_id == "unit-cli-shards"
    assert plan.shard_count == 1
    assert "Route Policy Scenario Shards: unit-cli-shards" in plan_markdown_path.read_text(encoding="utf-8")

    registry = load_route_policy_registry_json(registry_path)
    run = run_route_policy_scenario_set(
        plan.scenario_sets[0],
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path / "shards",
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "shard-history.json",
    )
    run_path = write_route_policy_scenario_set_run_json(tmp_path / "shard-run.json", run)
    merge_path = tmp_path / "merge.json"
    merge_markdown_path = tmp_path / "merge.md"
    merged_history_path = tmp_path / "merged-history.json"
    merge_args = build_parser().parse_args(
        [
            "route-policy-scenario-shard-merge",
            "--run",
            str(run_path),
            "--merge-id",
            "unit-cli-shard-merge",
            "--history-output",
            str(merged_history_path),
            "--output",
            str(merge_path),
            "--markdown-output",
            str(merge_markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_shard_merge(merge_args)
    merge = json.loads(merge_path.read_text(encoding="utf-8"))

    assert merge["recordType"] == "route-policy-scenario-shard-merge"
    assert merge["mergeId"] == "unit-cli-shard-merge"
    assert merge["passed"] is True
    assert merge["shardCount"] == 1
    assert merged_history_path.exists()
    assert "Route Policy Scenario Shard Merge: unit-cli-shard-merge" in merge_markdown_path.read_text(encoding="utf-8")


def test_route_policy_scenario_ci_manifest_builds_matrix_jobs(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-ci-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-ci-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(
                RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),
                RoutePolicyMatrixGoalSuiteSpec("far", "far-goals.json"),
            ),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-ci-shards",
    )

    manifest = build_route_policy_scenario_ci_manifest(
        plan,
        manifest_id="unit-ci-manifest",
        report_dir="ci/reports",
        run_output_dir="ci/runs",
        history_output_dir="ci/histories",
        merge_id="unit-ci-merge",
        merge_output="ci/merge.json",
        merge_history_output="ci/history.json",
        merge_markdown_output="ci/merge.md",
        merge_history_markdown_output="ci/history.md",
        include_markdown=True,
        cache_key_prefix="unit-cache",
        fail_on_regression=True,
    )
    manifest_path = write_route_policy_scenario_ci_manifest_json(tmp_path / "ci-manifest.json", manifest)
    loaded = load_route_policy_scenario_ci_manifest_json(manifest_path)
    payload = loaded.to_dict()

    assert loaded.manifest_id == "unit-ci-manifest"
    assert loaded.shard_job_count == 2
    assert loaded.scenario_count == 2
    assert payload["matrix"]["include"][0]["shardId"] == "unit-ci-matrix-direct-shard-001"
    assert loaded.shard_jobs[0].report_dir == "ci/reports/unit-ci-matrix-direct-shard-001"
    assert loaded.shard_jobs[0].run_output == "ci/runs/unit-ci-matrix-direct-shard-001.json"
    assert loaded.shard_jobs[0].history_output == "ci/histories/unit-ci-matrix-direct-shard-001.json"
    assert loaded.shard_jobs[0].expected_report_paths[0].endswith("/unit-near-short.json")
    assert "--fail-on-regression" in loaded.shard_jobs[0].command
    assert loaded.merge_job.depends_on == tuple(job.job_id for job in loaded.shard_jobs)
    assert loaded.merge_job.run_inputs == tuple(job.run_output for job in loaded.shard_jobs)
    assert "Route Policy Scenario CI Manifest: unit-ci-manifest" in render_route_policy_scenario_ci_manifest_markdown(
        loaded
    )


def test_route_policy_scenario_ci_manifest_cli_writes_manifest(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-ci-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-cli-ci-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-cli-ci-shards",
    )
    plan_path = write_route_policy_scenario_shard_plan_json(tmp_path / "shard-plan.json", plan)
    output_path = tmp_path / "ci-manifest.json"
    markdown_path = tmp_path / "ci-manifest.md"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-manifest",
            "--shard-plan",
            str(plan_path),
            "--manifest-id",
            "unit-cli-ci-manifest",
            "--report-dir",
            "ci/reports",
            "--run-output-dir",
            "ci/runs",
            "--history-output-dir",
            "ci/histories",
            "--merge-id",
            "unit-cli-ci-merge",
            "--merge-output",
            "ci/merge.json",
            "--merge-history-output",
            "ci/history.json",
            "--merge-markdown-output",
            "ci/merge.md",
            "--merge-history-markdown-output",
            "ci/history.md",
            "--cache-key-prefix",
            "unit-cache",
            "--include-markdown",
            "--fail-on-regression",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_ci_manifest(args)
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["recordType"] == "route-policy-scenario-ci-manifest"
    assert manifest["manifestId"] == "unit-cli-ci-manifest"
    assert manifest["shardJobCount"] == 1
    assert manifest["matrix"]["include"][0]["cacheKey"].startswith("unit-cache-unit-cli-ci-shards-")
    assert manifest["mergeJob"]["dependsOn"] == [manifest["shardJobs"][0]["jobId"]]
    assert "--fail-on-regression" in manifest["mergeJob"]["command"]
    assert "Route Policy Scenario CI Manifest: unit-cli-ci-manifest" in markdown_path.read_text(encoding="utf-8")


def test_route_policy_scenario_ci_workflow_materializes_github_actions_yaml(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-workflow-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-workflow-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-workflow-shards",
    )
    manifest = build_route_policy_scenario_ci_manifest(
        plan,
        manifest_id="unit-workflow-manifest",
        report_dir="ci/reports",
        run_output_dir="ci/runs",
        history_output_dir="ci/histories",
        merge_id="unit-workflow-merge",
        merge_output="ci/merge.json",
        merge_history_output="ci/history.json",
        cache_key_prefix="unit-cache",
        fail_on_regression=True,
    )

    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id="unit-workflow",
            workflow_name="Unit Scenario Workflow",
            artifact_root="ci",
            push_branches=("main",),
            pull_request_branches=("main",),
            artifact_retention_days=3,
        ),
    )
    workflow_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.yml", materialization)
    materialization_path = write_route_policy_scenario_ci_workflow_json(tmp_path / "workflow.json", materialization)
    loaded = load_route_policy_scenario_ci_workflow_json(materialization_path)
    workflow = workflow_path.read_text(encoding="utf-8")

    assert loaded.workflow_id == "unit-workflow"
    assert loaded.config.artifact_root == "ci"
    assert "name: 'Unit Scenario Workflow'" in workflow
    assert "workflow_dispatch: {}" in workflow
    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "route-policy-scenario-shards:" in workflow
    assert "matrix:" in workflow
    assert "shardId: 'unit-workflow-matrix-direct-shard-001'" in workflow
    assert "run: |" in workflow
    assert "route-policy-scenario-set" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "path: 'ci'" in workflow
    assert "actions/download-artifact@v4" in workflow
    assert "route-policy-scenario-shard-merge" in workflow
    assert "Route Policy Scenario CI Workflow: unit-workflow" in render_route_policy_scenario_ci_workflow_markdown(
        loaded
    )
    validation = validate_route_policy_scenario_ci_workflow(manifest, loaded, workflow_path=workflow_path)
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json", validation
    )
    loaded_validation = load_route_policy_scenario_ci_workflow_validation_json(validation_path)

    assert loaded_validation.passed is True
    assert loaded_validation.failed_checks == ()
    assert "yaml-shard-matrix-count" in {check.check_id for check in loaded_validation.checks}
    assert "Route Policy Scenario CI Workflow Validation: unit-workflow-validation" in (
        render_route_policy_scenario_ci_workflow_validation_markdown(loaded_validation)
    )


def test_route_policy_scenario_ci_workflow_validation_catches_tampered_merge_command(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-validation-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-validation-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-validation-shards",
    )
    manifest = build_route_policy_scenario_ci_manifest(
        plan,
        manifest_id="unit-validation-manifest",
        report_dir="ci/reports",
        run_output_dir="ci/runs",
        history_output_dir="ci/histories",
        merge_id="unit-validation-merge",
        merge_output="ci/merge.json",
        merge_history_output="ci/history.json",
    )
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-validation-workflow", artifact_root="ci"),
    )
    tampered = type(materialization)(
        workflow_id=materialization.workflow_id,
        manifest_id=materialization.manifest_id,
        workflow_name=materialization.workflow_name,
        workflow_yaml=materialization.workflow_yaml.replace("route-policy-scenario-shard-merge", "broken-merge", 1),
        config=materialization.config,
        workflow_path=materialization.workflow_path,
        metadata=materialization.metadata,
        version=materialization.version,
    )
    report = validate_route_policy_scenario_ci_workflow(manifest, tampered)

    assert report.passed is False
    assert "merge-command" in report.failed_checks


def test_route_policy_scenario_ci_workflow_cli_writes_yaml_and_index(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-workflow-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-cli-workflow-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-cli-workflow-shards",
    )
    manifest = build_route_policy_scenario_ci_manifest(
        plan,
        manifest_id="unit-cli-workflow-manifest",
        report_dir="ci/reports",
        run_output_dir="ci/runs",
        history_output_dir="ci/histories",
        merge_id="unit-cli-workflow-merge",
        merge_output="ci/merge.json",
        merge_history_output="ci/history.json",
    )
    manifest_path = write_route_policy_scenario_ci_manifest_json(tmp_path / "ci-manifest.json", manifest)
    workflow_path = tmp_path / "workflow.yml"
    index_path = tmp_path / "workflow.json"
    markdown_path = tmp_path / "workflow.md"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-workflow",
            "--manifest",
            str(manifest_path),
            "--workflow-id",
            "unit-cli-workflow",
            "--workflow-name",
            "Unit CLI Scenario Workflow",
            "--runs-on",
            "ubuntu-latest",
            "--python-version",
            "3.12",
            "--install-command",
            "pip install -e .",
            "--artifact-root",
            "ci",
            "--artifact-retention-days",
            "5",
            "--push-branch",
            "main",
            "--pull-request-branch",
            "main",
            "--fail-fast",
            "--workflow-output",
            str(workflow_path),
            "--index-output",
            str(index_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_ci_workflow(args)
    workflow = workflow_path.read_text(encoding="utf-8")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["recordType"] == "route-policy-scenario-ci-workflow"
    assert index["workflowId"] == "unit-cli-workflow"
    assert index["workflowPath"] == workflow_path.as_posix()
    assert index["config"]["pythonVersion"] == "3.12"
    assert index["config"]["failFast"] is True
    assert "name: 'Unit CLI Scenario Workflow'" in workflow
    assert "fail-fast: true" in workflow
    assert "pip install -e ." in workflow
    assert "Route Policy Scenario CI Workflow: unit-cli-workflow" in markdown_path.read_text(encoding="utf-8")


def test_route_policy_scenario_ci_workflow_validation_cli_writes_report(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-validation-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    expansion = expand_route_policy_scenario_matrix_to_directory(
        RoutePolicyScenarioMatrix(
            matrix_id="unit-cli-validation-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(RoutePolicyMatrixSceneSpec("unit", catalog_path.name, scene_id="unit-scene"),),
            goal_suites=(RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
        tmp_path / "generated",
        matrix_base_path=tmp_path,
    )
    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        tmp_path / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id="unit-cli-validation-shards",
    )
    manifest = build_route_policy_scenario_ci_manifest(
        plan,
        manifest_id="unit-cli-validation-manifest",
        report_dir="ci/reports",
        run_output_dir="ci/runs",
        history_output_dir="ci/histories",
        merge_id="unit-cli-validation-merge",
        merge_output="ci/merge.json",
        merge_history_output="ci/history.json",
    )
    manifest_path = write_route_policy_scenario_ci_manifest_json(tmp_path / "ci-manifest.json", manifest)
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id="unit-cli-validation-workflow",
            workflow_name="Unit CLI Validation Workflow",
            artifact_root="ci",
        ),
    )
    workflow_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.yml", materialization)
    index_path = write_route_policy_scenario_ci_workflow_json(tmp_path / "workflow.json", materialization)
    output_path = tmp_path / "workflow-validation.json"
    markdown_path = tmp_path / "workflow-validation.md"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-workflow-validate",
            "--manifest",
            str(manifest_path),
            "--workflow-index",
            str(index_path),
            "--workflow",
            str(workflow_path),
            "--validation-id",
            "unit-cli-validation",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--fail-on-validation",
        ]
    )

    cli.cmd_route_policy_scenario_ci_workflow_validate(args)
    report = load_route_policy_scenario_ci_workflow_validation_json(output_path)

    assert report.validation_id == "unit-cli-validation"
    assert report.workflow_path == str(workflow_path)
    assert report.passed is True
    assert "Route Policy Scenario CI Workflow Validation: unit-cli-validation" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_route_policy_scenario_ci_workflow_activation_writes_active_workflow(tmp_path: Path) -> None:
    manifest = build_unit_ci_workflow_manifest("unit-activation-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id="unit-activation-workflow",
            workflow_name="Unit Activation Workflow",
            artifact_root="ci",
        ),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(manifest, materialization, workflow_path=source_path)
    active_path = tmp_path / ".github" / "workflows" / "unit-activation.yml"

    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=active_path,
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json",
        activation,
    )
    loaded_activation = load_route_policy_scenario_ci_workflow_activation_json(activation_path)

    assert loaded_activation.activated is True
    assert loaded_activation.failed_checks == ()
    assert loaded_activation.metadata["activationState"] == "activated"
    assert active_path.read_text(encoding="utf-8") == materialization.workflow_yaml
    assert "Route Policy Scenario CI Workflow Activation: unit-activation-workflow-activation" in (
        render_route_policy_scenario_ci_workflow_activation_markdown(loaded_activation)
    )


def test_route_policy_scenario_ci_workflow_activation_blocks_failed_validation(tmp_path: Path) -> None:
    manifest = build_unit_ci_workflow_manifest("unit-blocked-activation-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-blocked-activation-workflow", artifact_root="ci"),
    )
    tampered = type(materialization)(
        workflow_id=materialization.workflow_id,
        manifest_id=materialization.manifest_id,
        workflow_name=materialization.workflow_name,
        workflow_yaml=materialization.workflow_yaml.replace("route-policy-scenario-shard-merge", "broken-merge", 1),
        config=materialization.config,
        workflow_path=materialization.workflow_path,
        metadata=materialization.metadata,
        version=materialization.version,
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", tampered)
    validation = validate_route_policy_scenario_ci_workflow(manifest, tampered, workflow_path=source_path)
    active_path = tmp_path / ".github" / "workflows" / "unit-blocked.yml"

    activation = activate_route_policy_scenario_ci_workflow(
        tampered,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=active_path,
    )

    assert activation.activated is False
    assert "validation-passed" in activation.failed_checks
    assert activation.metadata["activationState"] == "blocked"
    assert not active_path.exists()


def test_route_policy_scenario_ci_workflow_activation_blocks_non_workflow_output(tmp_path: Path) -> None:
    manifest = build_unit_ci_workflow_manifest("unit-invalid-activation-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-invalid-activation-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(manifest, materialization, workflow_path=source_path)
    invalid_active_path = tmp_path / "workflows" / "unit-invalid.yml"

    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=invalid_active_path,
    )

    assert activation.activated is False
    assert "active-path-root" in activation.failed_checks
    assert not invalid_active_path.exists()


def test_route_policy_scenario_ci_workflow_activation_cli_writes_report(tmp_path: Path) -> None:
    manifest = build_unit_ci_workflow_manifest("unit-cli-activation-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id="unit-cli-activation-workflow",
            workflow_name="Unit CLI Activation Workflow",
            artifact_root="ci",
        ),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    index_path = write_route_policy_scenario_ci_workflow_json(tmp_path / "workflow.json", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-cli-activation-validation",
        workflow_path=source_path,
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json",
        validation,
    )
    active_path = tmp_path / ".github" / "workflows" / "unit-cli-activation.yml"
    output_path = tmp_path / "workflow-activation.json"
    markdown_path = tmp_path / "workflow-activation.md"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-workflow-activate",
            "--workflow-index",
            str(index_path),
            "--validation-report",
            str(validation_path),
            "--workflow",
            str(source_path),
            "--active-workflow-output",
            str(active_path),
            "--activation-id",
            "unit-cli-activation",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--fail-on-activation",
        ]
    )

    cli.cmd_route_policy_scenario_ci_workflow_activate(args)
    activation = load_route_policy_scenario_ci_workflow_activation_json(output_path)

    assert activation.activation_id == "unit-cli-activation"
    assert activation.activated is True
    assert active_path.read_text(encoding="utf-8") == materialization.workflow_yaml
    assert "Route Policy Scenario CI Workflow Activation: unit-cli-activation" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_route_policy_scenario_ci_review_provenance_roundtrip() -> None:
    """Provenance roundtrips through to_dict / from_dict cleanly."""

    from gs_sim2real.sim import route_policy_scenario_ci_review_provenance_from_dict

    original = RoutePolicyScenarioCIReviewProvenance(
        kind="production",
        generated_at="2026-05-15T08:30:00+00:00",
        git_commit="d866f1e",
        scene_id="outdoor-demo",
        scenario_set_id="outdoor-demo-shard-merge",
        matrix_hash="sha256:abc123",
        policy_version="v0.2.0",
        env_contract_version="contract-v3",
        correlation_threshold_profile="outdoor-demo-default",
        asset_source="bag6",
        extra={"runTrigger": "nightly"},
    )
    payload = original.to_dict()
    assert payload["recordType"] == "route-policy-scenario-ci-review-provenance"
    assert payload["kind"] == "production"
    assert payload["sceneId"] == "outdoor-demo"
    assert payload["extra"] == {"runTrigger": "nightly"}

    rebuilt = route_policy_scenario_ci_review_provenance_from_dict(payload)
    assert rebuilt == original


def test_route_policy_scenario_ci_review_provenance_rejects_bad_kind() -> None:
    with pytest.raises(ValueError):
        RoutePolicyScenarioCIReviewProvenance(
            kind="dev",  # type: ignore[arg-type]
            generated_at="2026-05-15T08:30:00+00:00",
        )
    with pytest.raises(ValueError):
        RoutePolicyScenarioCIReviewProvenance(
            kind="production",
            generated_at="",
        )


def test_route_policy_scenario_ci_review_artifact_writes_pages_outputs(tmp_path: Path) -> None:
    manifest = build_unit_ci_workflow_manifest("unit-review-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-review-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-review-validation",
        workflow_path=source_path,
    )
    active_path = tmp_path / ".github" / "workflows" / "unit-review.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=active_path,
        activation_id="unit-review-activation",
    )
    review = build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id="unit-review",
        pages_base_url="https://example.test/reviews/unit-review/",
    )
    review_path = write_route_policy_scenario_ci_review_json(tmp_path / "review.json", review)
    loaded = load_route_policy_scenario_ci_review_json(review_path)
    bundle_paths = write_route_policy_scenario_ci_review_bundle(tmp_path / "pages" / "unit-review", loaded)

    assert loaded.passed is True
    assert loaded.shard_count == 1
    assert loaded.scenario_count == 1
    assert loaded.report_count == 1
    assert loaded.metadata["pagesBaseUrl"] == "https://example.test/reviews/unit-review/"
    assert "Route Policy Scenario CI Review: unit-review" in render_route_policy_scenario_ci_review_markdown(loaded)
    assert "<title>unit-review CI Review</title>" in render_route_policy_scenario_ci_review_html(loaded)
    assert Path(bundle_paths["json"]).exists()
    assert Path(bundle_paths["markdown"]).exists()
    assert Path(bundle_paths["html"]).read_text(encoding="utf-8").startswith("<!doctype html>")
    # No provenance flag set → the JSON stays in the v1 shape, with no
    # ``provenance`` key. Renderers must not emit a Run Provenance section.
    assert loaded.provenance is None
    review_payload = json.loads(Path(bundle_paths["json"]).read_text(encoding="utf-8"))
    assert "provenance" not in review_payload
    assert "Run Provenance" not in render_route_policy_scenario_ci_review_markdown(loaded)
    assert "Run Provenance" not in render_route_policy_scenario_ci_review_html(loaded)


def test_route_policy_scenario_ci_review_artifact_carries_production_provenance(tmp_path: Path) -> None:
    """When provenance is provided, it roundtrips and renders into the bundle."""

    manifest = build_unit_ci_workflow_manifest("unit-review-prov-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-review-prov-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-review-prov-validation",
        workflow_path=source_path,
    )
    active_path = tmp_path / ".github" / "workflows" / "unit-review-prov.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=active_path,
        activation_id="unit-review-prov-activation",
    )
    provenance = RoutePolicyScenarioCIReviewProvenance(
        kind="production",
        generated_at="2026-05-15T08:30:00+00:00",
        git_commit="d866f1e",
        scene_id="outdoor-demo",
        policy_version="v0.2.0",
        asset_source="bag6",
        extra={"runTrigger": "nightly"},
    )
    review = build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id="unit-review-prov",
        provenance=provenance,
    )
    review_path = write_route_policy_scenario_ci_review_json(tmp_path / "review.json", review)
    loaded = load_route_policy_scenario_ci_review_json(review_path)

    assert loaded.provenance == provenance
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["kind"] == "production"
    assert payload["provenance"]["sceneId"] == "outdoor-demo"
    assert payload["provenance"]["extra"]["runTrigger"] == "nightly"

    markdown = render_route_policy_scenario_ci_review_markdown(loaded)
    assert "## Run Provenance" in markdown
    assert "Kind: production" in markdown
    assert "Scene id: `outdoor-demo`" in markdown
    assert "Generated at: 2026-05-15T08:30:00+00:00" in markdown

    html = render_route_policy_scenario_ci_review_html(loaded)
    assert "Run Provenance" in html
    assert '<span class="pill production">PRODUCTION</span>' in html


def test_route_policy_scenario_ci_review_cli_writes_bundle(tmp_path: Path) -> None:
    manifest = build_unit_ci_workflow_manifest("unit-cli-review-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-cli-review-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-cli-review-validation",
        workflow_path=source_path,
    )
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=tmp_path / ".github" / "workflows" / "unit-cli-review.yml",
        activation_id="unit-cli-review-activation",
    )
    merge_path = write_route_policy_scenario_shard_merge_json(
        tmp_path / "shard-merge.json", build_unit_ci_shard_merge_report()
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json",
        validation,
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json",
        activation,
    )
    bundle_dir = tmp_path / "pages" / "unit-cli-review"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-review",
            "--shard-merge",
            str(merge_path),
            "--validation-report",
            str(validation_path),
            "--activation-report",
            str(activation_path),
            "--review-id",
            "unit-cli-review",
            "--pages-base-url",
            "https://example.test/reviews/unit-cli-review/",
            "--bundle-dir",
            str(bundle_dir),
            "--fail-on-review",
        ]
    )

    cli.cmd_route_policy_scenario_ci_review(args)
    review = load_route_policy_scenario_ci_review_json(bundle_dir / "review.json")

    assert review.review_id == "unit-cli-review"
    assert review.passed is True
    assert review.adoption is None
    assert (bundle_dir / "review.md").exists()
    html_text = (bundle_dir / "index.html").read_text(encoding="utf-8")
    assert "Route Policy Scenario CI Review" in html_text
    assert "Adopted Workflow" not in html_text


def test_route_policy_scenario_ci_review_cli_embeds_adoption_diff(tmp_path: Path) -> None:
    from gs_sim2real.sim import (
        adopt_route_policy_scenario_ci_workflow,
        write_route_policy_scenario_ci_workflow_adoption_json,
    )

    manifest = build_unit_ci_workflow_manifest("unit-cli-review-adoption-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-cli-review-adoption-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-cli-review-adoption-validation",
        workflow_path=source_path,
    )
    manual_active = tmp_path / ".github" / "workflows" / "unit-cli-review-adoption.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=manual_active,
        activation_id="unit-cli-review-adoption-activation",
    )
    review = build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id="unit-cli-review-adoption-review",
        pages_base_url="https://example.test/reviews/unit-cli-review-adoption/",
    )
    promotion = promote_route_policy_scenario_ci_workflow(
        review,
        trigger_mode="pull-request",
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-cli-review-adoption/",
        promotion_id="unit-cli-review-adoption-promotion",
    )
    assert promotion.promoted is True
    adopted_source = tmp_path / "unit-cli-review-adoption-adopted.generated.yml"
    adopted_active = tmp_path / ".github" / "workflows" / "unit-cli-review-adoption-adopted.yml"
    adoption = adopt_route_policy_scenario_ci_workflow(
        promotion,
        manifest,
        materialization,
        adopted_source_workflow_path=adopted_source,
        adopted_active_workflow_path=adopted_active,
        adoption_id="unit-cli-review-adoption",
    )
    assert adoption.adopted is True

    merge_path = write_route_policy_scenario_shard_merge_json(
        tmp_path / "shard-merge.json", build_unit_ci_shard_merge_report()
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json", validation
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json", activation
    )
    adoption_path = write_route_policy_scenario_ci_workflow_adoption_json(tmp_path / "adoption.json", adoption)
    bundle_dir = tmp_path / "pages" / "unit-cli-review-adoption"

    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-review",
            "--shard-merge",
            str(merge_path),
            "--validation-report",
            str(validation_path),
            "--activation-report",
            str(activation_path),
            "--adoption-report",
            str(adoption_path),
            "--review-id",
            "unit-cli-review-adoption",
            "--pages-base-url",
            "https://example.test/reviews/unit-cli-review-adoption/",
            "--bundle-dir",
            str(bundle_dir),
            "--fail-on-review",
        ]
    )

    cli.cmd_route_policy_scenario_ci_review(args)
    loaded = load_route_policy_scenario_ci_review_json(bundle_dir / "review.json")

    assert loaded.adoption is not None
    assert loaded.adoption.adopted is True
    assert loaded.adoption.trigger_mode == "pull-request"
    assert loaded.adoption.pull_request_branches == ("main",)
    assert loaded.adoption.workflow_diff is not None
    assert "+  pull_request:" in loaded.adoption.workflow_diff

    html_text = (bundle_dir / "index.html").read_text(encoding="utf-8")
    assert "Adopted Workflow" in html_text
    assert '<pre class="diff">' in html_text
    assert '<span class="add">+  pull_request:</span>' in html_text

    markdown_text = (bundle_dir / "review.md").read_text(encoding="utf-8")
    assert "## Adopted Workflow" in markdown_text
    assert "```diff" in markdown_text


def test_route_policy_scenario_ci_review_surfaces_correlation_reports_from_shard_runs(tmp_path: Path) -> None:
    """Correlation reports attached to a shard's run JSON must flow into the review artifact + bundle."""
    # Build a unit scenario-set run with one attached correlation report.
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-correlation-review-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    near_goals = write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-correlation-review",
        policy_registry_path=registry_path.name,
        episode_count=1,
        seed_start=0,
        max_steps=4,
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="near",
                scene_catalog=catalog_path.name,
                goal_suite_path=near_goals.name,
            ),
        ),
    )
    registry = load_route_policy_registry_json(registry_path)

    # Pre-computed correlation report (mirroring the #121 fixture path).
    bag_stream = BagPoseStream(
        samples=(
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )
    sim_samples = (
        SimPoseSample(timestamp_seconds=0.01, position=(0.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    )
    correlation_report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.05)
    correlation_path = write_real_vs_sim_correlation_report_json(tmp_path / "correlation.json", correlation_report)

    run_report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "history.json",
        correlation_report_paths=[correlation_path.name],
    )
    run_path = write_route_policy_scenario_set_run_json(tmp_path / "scenario-run.json", run_report)

    # Build a shard merge report whose shard run_path points at the real file
    # so the review CLI can lift the correlation reports back out.
    history = RoutePolicyBenchmarkHistoryReport(
        history_id="unit-correlation-review-history",
        reports=(
            RoutePolicyBenchmarkSnapshot(
                benchmark_id="unit-correlation-review",
                passed=True,
                best_policy_name="direct",
                policies=(
                    RoutePolicyBenchmarkPolicySnapshot(
                        policy_name="direct",
                        passed=True,
                        metrics={"successRate": 1.0, "collisionRate": 0.0},
                    ),
                ),
                source_path=str(run_report.scenario_results[0].report_path),
            ),
        ),
    )
    merge_report = RoutePolicyScenarioShardMergeReport(
        merge_id="unit-correlation-review-merge",
        shard_runs=(
            RoutePolicyScenarioShardRunSummary(
                shard_id="unit-correlation-shard",
                scenario_set_id="unit-correlation-review",
                passed=True,
                scenario_count=1,
                report_paths=(str(run_report.scenario_results[0].report_path),),
                run_path=str(run_path),
                history_path=None,
            ),
        ),
        history=history,
    )
    merge_path = write_route_policy_scenario_shard_merge_json(tmp_path / "shard-merge.json", merge_report)

    # Minimal validation + activation reports (workflow gate is not the focus here).
    manifest = build_unit_ci_workflow_manifest("unit-correlation-review-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-correlation-review-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-correlation-review-validation",
        workflow_path=source_path,
    )
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=tmp_path / ".github" / "workflows" / "unit-correlation-review.yml",
        activation_id="unit-correlation-review-activation",
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json", validation
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json", activation
    )

    bundle_dir = tmp_path / "pages" / "unit-correlation-review"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-review",
            "--shard-merge",
            str(merge_path),
            "--validation-report",
            str(validation_path),
            "--activation-report",
            str(activation_path),
            "--review-id",
            "unit-correlation-review",
            "--bundle-dir",
            str(bundle_dir),
        ]
    )
    cli.cmd_route_policy_scenario_ci_review(args)
    review = load_route_policy_scenario_ci_review_json(bundle_dir / "review.json")

    assert review.correlation_report_count == 1
    assert review.correlation_reports[0].bag_source.source_topic == "/gnss/fix"
    assert review.correlation_reports[0].translation_error_mean_meters == pytest.approx(0.05, rel=1e-6)

    markdown_text = (bundle_dir / "review.md").read_text(encoding="utf-8")
    assert "## Real-vs-sim correlation" in markdown_text
    assert "/gnss/fix" in markdown_text

    html_text = (bundle_dir / "index.html").read_text(encoding="utf-8")
    assert "Real-vs-sim correlation" in html_text
    assert "/gnss/fix" in html_text


def test_route_policy_scenario_ci_review_correlation_gate_fails_review_when_threshold_exceeded(
    tmp_path: Path,
) -> None:
    """Passing --max-correlation-translation-mean-meters flips the review gate when drift exceeds it."""
    # Reuse the same shard run + correlation setup as the surfacing test.
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-correlation-gate-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    near_goals = write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-correlation-gate",
        policy_registry_path=registry_path.name,
        episode_count=1,
        seed_start=0,
        max_steps=4,
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="near",
                scene_catalog=catalog_path.name,
                goal_suite_path=near_goals.name,
            ),
        ),
    )
    registry = load_route_policy_registry_json(registry_path)

    bag_stream = BagPoseStream(
        samples=(
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )
    # Sim trajectory drifts by 0.5 m on average from the bag, so the gate set
    # at 0.1 m must fail.
    sim_samples = (
        SimPoseSample(timestamp_seconds=0.01, position=(0.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    )
    correlation_report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.05)
    correlation_path = write_real_vs_sim_correlation_report_json(tmp_path / "correlation.json", correlation_report)

    run_report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "history.json",
        correlation_report_paths=[correlation_path.name],
    )
    run_path = write_route_policy_scenario_set_run_json(tmp_path / "scenario-run.json", run_report)

    history = RoutePolicyBenchmarkHistoryReport(
        history_id="unit-correlation-gate-history",
        reports=(
            RoutePolicyBenchmarkSnapshot(
                benchmark_id="unit-correlation-gate",
                passed=True,
                best_policy_name="direct",
                policies=(
                    RoutePolicyBenchmarkPolicySnapshot(
                        policy_name="direct",
                        passed=True,
                        metrics={"successRate": 1.0, "collisionRate": 0.0},
                    ),
                ),
                source_path=str(run_report.scenario_results[0].report_path),
            ),
        ),
    )
    merge_report = RoutePolicyScenarioShardMergeReport(
        merge_id="unit-correlation-gate-merge",
        shard_runs=(
            RoutePolicyScenarioShardRunSummary(
                shard_id="unit-correlation-gate-shard",
                scenario_set_id="unit-correlation-gate",
                passed=True,
                scenario_count=1,
                report_paths=(str(run_report.scenario_results[0].report_path),),
                run_path=str(run_path),
                history_path=None,
            ),
        ),
        history=history,
    )
    merge_path = write_route_policy_scenario_shard_merge_json(tmp_path / "shard-merge.json", merge_report)

    manifest = build_unit_ci_workflow_manifest("unit-correlation-gate-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-correlation-gate-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-correlation-gate-validation",
        workflow_path=source_path,
    )
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=tmp_path / ".github" / "workflows" / "unit-correlation-gate.yml",
        activation_id="unit-correlation-gate-activation",
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json", validation
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json", activation
    )

    bundle_dir = tmp_path / "pages" / "unit-correlation-gate"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-review",
            "--shard-merge",
            str(merge_path),
            "--validation-report",
            str(validation_path),
            "--activation-report",
            str(activation_path),
            "--review-id",
            "unit-correlation-gate",
            "--bundle-dir",
            str(bundle_dir),
            "--max-correlation-translation-mean-meters",
            "0.1",
        ]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_route_policy_scenario_ci_review(type("Args", (), {**vars(args), "fail_on_review": True})())
    assert excinfo.value.code == 2

    review = load_route_policy_scenario_ci_review_json(bundle_dir / "review.json")
    assert review.correlation_passed is False
    assert review.passed is False  # gate fail propagates to overall pass
    failed_checks = {check for _, _, checks in review.correlation_failed_reports for check in checks}
    assert "translation-mean" in failed_checks

    markdown_text = (bundle_dir / "review.md").read_text(encoding="utf-8")
    assert "Correlation gate failures" in markdown_text
    assert "translation-mean" in markdown_text


def test_route_policy_scenario_ci_review_correlation_per_topic_overrides_take_priority(
    tmp_path: Path,
) -> None:
    """A per-topic override should fire even when the scalar default would tolerate the drift."""
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-correlation-override-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    near_goals = write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-correlation-override",
        policy_registry_path=registry_path.name,
        episode_count=1,
        seed_start=0,
        max_steps=4,
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="near",
                scene_catalog=catalog_path.name,
                goal_suite_path=near_goals.name,
            ),
        ),
    )
    registry = load_route_policy_registry_json(registry_path)
    bag_stream = BagPoseStream(
        samples=(
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ),
        frame_id="enu",
        source_topic="/strict/topic",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )
    sim_samples = (
        SimPoseSample(timestamp_seconds=0.01, position=(0.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    )
    correlation_report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.05)
    correlation_path = write_real_vs_sim_correlation_report_json(tmp_path / "correlation.json", correlation_report)

    run_report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "history.json",
        correlation_report_paths=[correlation_path.name],
    )
    run_path = write_route_policy_scenario_set_run_json(tmp_path / "scenario-run.json", run_report)

    history = RoutePolicyBenchmarkHistoryReport(
        history_id="unit-correlation-override-history",
        reports=(
            RoutePolicyBenchmarkSnapshot(
                benchmark_id="unit-correlation-override",
                passed=True,
                best_policy_name="direct",
                policies=(
                    RoutePolicyBenchmarkPolicySnapshot(
                        policy_name="direct",
                        passed=True,
                        metrics={"successRate": 1.0, "collisionRate": 0.0},
                    ),
                ),
                source_path=str(run_report.scenario_results[0].report_path),
            ),
        ),
    )
    merge_report = RoutePolicyScenarioShardMergeReport(
        merge_id="unit-correlation-override-merge",
        shard_runs=(
            RoutePolicyScenarioShardRunSummary(
                shard_id="unit-correlation-override-shard",
                scenario_set_id="unit-correlation-override",
                passed=True,
                scenario_count=1,
                report_paths=(str(run_report.scenario_results[0].report_path),),
                run_path=str(run_path),
                history_path=None,
            ),
        ),
        history=history,
    )
    merge_path = write_route_policy_scenario_shard_merge_json(tmp_path / "shard-merge.json", merge_report)

    manifest = build_unit_ci_workflow_manifest("unit-correlation-override-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id="unit-correlation-override-workflow", artifact_root="ci"
        ),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-correlation-override-validation",
        workflow_path=source_path,
    )
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=tmp_path / ".github" / "workflows" / "unit-correlation-override.yml",
        activation_id="unit-correlation-override-activation",
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json", validation
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json", activation
    )

    overrides_payload = {"/strict/topic": {"maxTranslationErrorMeanMeters": 0.1}}
    overrides_path = tmp_path / "thresholds.json"
    overrides_path.write_text(json.dumps(overrides_payload), encoding="utf-8")

    bundle_dir = tmp_path / "pages" / "unit-correlation-override"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-review",
            "--shard-merge",
            str(merge_path),
            "--validation-report",
            str(validation_path),
            "--activation-report",
            str(activation_path),
            "--review-id",
            "unit-correlation-override",
            "--bundle-dir",
            str(bundle_dir),
            "--max-correlation-translation-mean-meters",
            "10.0",
            "--correlation-thresholds-config",
            str(overrides_path),
        ]
    )
    cli.cmd_route_policy_scenario_ci_review(args)
    review = load_route_policy_scenario_ci_review_json(bundle_dir / "review.json")

    assert review.correlation_passed is False
    assert review.passed is False
    failed_topics = {topic for _, topic, _ in review.correlation_failed_reports}
    assert failed_topics == {"/strict/topic"}
    assert "/strict/topic" in review.correlation_threshold_overrides

    markdown_text = (bundle_dir / "review.md").read_text(encoding="utf-8")
    assert "per-topic overrides: 1" in markdown_text


def test_route_policy_scenario_ci_review_omits_correlation_when_flag_set(tmp_path: Path) -> None:
    """--no-correlation-reports must drop the correlation block even if shard runs carry them."""
    # Reuse the unit shard merge fixture (its run_path points at a fictional file
    # which the collector skips silently anyway).
    merge_path = write_route_policy_scenario_shard_merge_json(
        tmp_path / "shard-merge.json", build_unit_ci_shard_merge_report()
    )
    manifest = build_unit_ci_workflow_manifest("unit-no-correlation-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-no-correlation-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "workflow.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id="unit-no-correlation-validation",
        workflow_path=source_path,
    )
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=tmp_path / ".github" / "workflows" / "unit-no-correlation.yml",
        activation_id="unit-no-correlation-activation",
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        tmp_path / "workflow-validation.json", validation
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        tmp_path / "workflow-activation.json", activation
    )
    bundle_dir = tmp_path / "pages" / "unit-no-correlation"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-review",
            "--shard-merge",
            str(merge_path),
            "--validation-report",
            str(validation_path),
            "--activation-report",
            str(activation_path),
            "--review-id",
            "unit-no-correlation",
            "--bundle-dir",
            str(bundle_dir),
            "--no-correlation-reports",
        ]
    )
    cli.cmd_route_policy_scenario_ci_review(args)
    review = load_route_policy_scenario_ci_review_json(bundle_dir / "review.json")

    assert review.correlation_report_count == 0
    markdown_text = (bundle_dir / "review.md").read_text(encoding="utf-8")
    assert "Real-vs-sim correlation" not in markdown_text


def test_route_policy_scenario_ci_workflow_promotion_passes_review_gate(tmp_path: Path) -> None:
    review = build_unit_ci_review_artifact(
        tmp_path,
        prefix="unit-promotion",
        pages_base_url="https://example.test/reviews/unit-promotion/",
    )
    promotion = promote_route_policy_scenario_ci_workflow(
        review,
        trigger_mode="pull-request",
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-promotion/",
        promotion_id="unit-promotion",
    )
    promotion_path = write_route_policy_scenario_ci_workflow_promotion_json(tmp_path / "promotion.json", promotion)
    loaded = load_route_policy_scenario_ci_workflow_promotion_json(promotion_path)
    markdown = render_route_policy_scenario_ci_workflow_promotion_markdown(loaded)

    assert loaded.promoted is True
    assert loaded.passed is True
    assert loaded.trigger_mode == "pull-request"
    assert loaded.pull_request_branches == ("main",)
    assert loaded.review_url == "https://example.test/reviews/unit-promotion/"
    assert "Route Policy Scenario CI Workflow Promotion: unit-promotion" in markdown
    assert "PROMOTED" in markdown


def test_route_policy_scenario_ci_workflow_promotion_blocks_failed_review(tmp_path: Path) -> None:
    review = build_unit_ci_review_artifact(
        tmp_path,
        prefix="unit-promotion-blocked",
        pages_base_url="https://example.test/reviews/unit-promotion-blocked/",
    )
    failed_review = RoutePolicyScenarioCIReviewArtifact(
        review_id=review.review_id,
        merge_id=review.merge_id,
        workflow_id=review.workflow_id,
        manifest_id=review.manifest_id,
        validation_id=review.validation_id,
        activation_id=review.activation_id,
        validation_passed=review.validation_passed,
        activation_activated=review.activation_activated,
        shard_merge_passed=review.shard_merge_passed,
        history_passed=False,
        active_workflow_path=review.active_workflow_path,
        source_workflow_path=review.source_workflow_path,
        shards=review.shards,
        history_failed_checks=("mean_reward_drop",),
        metadata=review.metadata,
        version=review.version,
    )
    promotion = promote_route_policy_scenario_ci_workflow(
        failed_review,
        trigger_mode="pull-request",
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-promotion-blocked/",
    )

    assert promotion.promoted is False
    assert promotion.passed is False
    assert "review-passed" in promotion.failed_checks
    assert "history-passed" in promotion.failed_checks


def test_route_policy_scenario_ci_workflow_promotion_cli_writes_report(tmp_path: Path) -> None:
    review = build_unit_ci_review_artifact(
        tmp_path,
        prefix="unit-cli-promotion",
        pages_base_url="https://example.test/reviews/unit-cli-promotion/",
    )
    review_path = write_route_policy_scenario_ci_review_json(tmp_path / "review.json", review)
    output_path = tmp_path / "workflow-promotion.json"
    markdown_path = tmp_path / "workflow-promotion.md"
    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-workflow-promote",
            "--review",
            str(review_path),
            "--review-url",
            "https://example.test/reviews/unit-cli-promotion/",
            "--trigger-mode",
            "push-and-pull-request",
            "--push-branch",
            "main",
            "--pull-request-branch",
            "main",
            "--promotion-id",
            "unit-cli-promotion",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--fail-on-promotion",
        ]
    )

    cli.cmd_route_policy_scenario_ci_workflow_promote(args)
    promotion = load_route_policy_scenario_ci_workflow_promotion_json(output_path)

    assert promotion.promotion_id == "unit-cli-promotion"
    assert promotion.promoted is True
    assert promotion.push_branches == ("main",)
    assert promotion.pull_request_branches == ("main",)
    assert "Route Policy Scenario CI Workflow Promotion" in markdown_path.read_text(encoding="utf-8")


def test_route_policy_scenario_ci_workflow_adoption_materializes_trigger_enabled_workflow(
    tmp_path: Path,
) -> None:
    from gs_sim2real.sim import (
        adopt_route_policy_scenario_ci_workflow,
        render_route_policy_scenario_ci_workflow_adoption_markdown,
        write_route_policy_scenario_ci_workflow_adoption_json,
    )

    manifest = build_unit_ci_workflow_manifest("unit-adoption-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-adoption-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(
        tmp_path / "unit-adoption.generated.yml", materialization
    )
    validation = validate_route_policy_scenario_ci_workflow(manifest, materialization, workflow_path=source_path)
    manual_active = tmp_path / ".github" / "workflows" / "unit-adoption.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=manual_active,
        activation_id="unit-adoption-activation",
    )
    review = build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id="unit-adoption-review",
        pages_base_url="https://example.test/reviews/unit-adoption/",
    )
    promotion = promote_route_policy_scenario_ci_workflow(
        review,
        trigger_mode="push-and-pull-request",
        push_branches=("main",),
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-adoption/",
        promotion_id="unit-adoption-promotion",
    )
    assert promotion.promoted is True

    adopted_source = tmp_path / "unit-adoption-adopted.generated.yml"
    adopted_active = tmp_path / ".github" / "workflows" / "unit-adoption-adopted.yml"
    report = adopt_route_policy_scenario_ci_workflow(
        promotion,
        manifest,
        materialization,
        adopted_source_workflow_path=adopted_source,
        adopted_active_workflow_path=adopted_active,
        adoption_id="unit-adoption",
    )

    assert report.adopted is True
    assert report.trigger_mode == "push-and-pull-request"
    assert report.manual_active_workflow_path == manual_active.as_posix()
    assert report.adopted_active_workflow_path == adopted_active.as_posix()
    assert manual_active.read_text(encoding="utf-8") != adopted_active.read_text(encoding="utf-8")
    adopted_text = adopted_active.read_text(encoding="utf-8")
    assert "push:" in adopted_text
    assert "pull_request:" in adopted_text
    assert "workflow_dispatch" in adopted_text

    report_path = write_route_policy_scenario_ci_workflow_adoption_json(tmp_path / "adoption.json", report)
    assert report_path.exists()
    markdown = render_route_policy_scenario_ci_workflow_adoption_markdown(report)
    assert "Route Policy Scenario CI Workflow Adoption: unit-adoption" in markdown
    assert "ADOPTED" in markdown


def test_route_policy_scenario_ci_workflow_adoption_blocks_when_promotion_failed(tmp_path: Path) -> None:
    from gs_sim2real.sim import (
        RoutePolicyScenarioCIWorkflowPromotionCheck,
        RoutePolicyScenarioCIWorkflowPromotionReport,
        adopt_route_policy_scenario_ci_workflow,
    )

    manifest = build_unit_ci_workflow_manifest("unit-blocked-adoption-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-blocked-adoption-workflow", artifact_root="ci"),
    )
    manual_active = tmp_path / ".github" / "workflows" / "unit-blocked-adoption.yml"
    failed_promotion = RoutePolicyScenarioCIWorkflowPromotionReport(
        promotion_id="unit-blocked-adoption-promotion",
        review_id="unit-blocked-adoption-review",
        workflow_id=materialization.workflow_id,
        manifest_id=manifest.manifest_id,
        active_workflow_path=manual_active.as_posix(),
        trigger_mode="pull-request",
        promoted=False,
        checks=(
            RoutePolicyScenarioCIWorkflowPromotionCheck(
                check_id="review-passed",
                passed=False,
                message="scenario CI review did not pass",
            ),
        ),
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-blocked-adoption/",
    )

    adopted_active = tmp_path / ".github" / "workflows" / "unit-blocked-adoption-adopted.yml"
    report = adopt_route_policy_scenario_ci_workflow(
        failed_promotion,
        manifest,
        materialization,
        adopted_source_workflow_path=tmp_path / "unit-blocked-adoption-adopted.generated.yml",
        adopted_active_workflow_path=adopted_active,
    )

    assert report.adopted is False
    assert "promotion-promoted" in report.failed_checks
    assert report.adopted_validation is None
    assert report.adopted_activation is None
    assert not adopted_active.exists()
    assert not (tmp_path / "unit-blocked-adoption-adopted.generated.yml").exists()


def test_route_policy_scenario_ci_workflow_adoption_blocks_when_paths_collide(tmp_path: Path) -> None:
    from gs_sim2real.sim import adopt_route_policy_scenario_ci_workflow

    manifest = build_unit_ci_workflow_manifest("unit-collide-adoption-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id="unit-collide-adoption-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(tmp_path / "unit-collide.generated.yml", materialization)
    validation = validate_route_policy_scenario_ci_workflow(manifest, materialization, workflow_path=source_path)
    manual_active = tmp_path / ".github" / "workflows" / "unit-collide-adoption.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=manual_active,
        activation_id="unit-collide-adoption-activation",
    )
    review = build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id="unit-collide-adoption-review",
        pages_base_url="https://example.test/reviews/unit-collide-adoption/",
    )
    promotion = promote_route_policy_scenario_ci_workflow(
        review,
        trigger_mode="pull-request",
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-collide-adoption/",
        promotion_id="unit-collide-adoption-promotion",
    )
    assert promotion.promoted is True

    # Point the adopted active path at the manual active path so the
    # distinct-from-manual gate fires and no overwrite happens.
    report = adopt_route_policy_scenario_ci_workflow(
        promotion,
        manifest,
        materialization,
        adopted_source_workflow_path=tmp_path / "unit-collide-adopted.generated.yml",
        adopted_active_workflow_path=manual_active,
    )

    assert report.adopted is False
    assert "adopted-path-distinct-from-manual" in report.failed_checks
    # The manual file is left as-is (still the materialization content, not a
    # trigger-enabled replacement).
    manual_text = manual_active.read_text(encoding="utf-8")
    assert "pull_request:" not in manual_text


def test_route_policy_scenario_ci_workflow_adopt_cli_writes_report(tmp_path: Path) -> None:
    from gs_sim2real.sim import (
        write_route_policy_scenario_ci_manifest_json,
        write_route_policy_scenario_ci_workflow_promotion_json,
    )

    manifest = build_unit_ci_workflow_manifest("unit-cli-adoption-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id="unit-cli-adoption-workflow",
            workflow_name="Unit CLI Adoption Workflow",
            artifact_root="ci",
        ),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(
        tmp_path / "unit-cli-adoption.generated.yml", materialization
    )
    validation = validate_route_policy_scenario_ci_workflow(manifest, materialization, workflow_path=source_path)
    manual_active = tmp_path / ".github" / "workflows" / "unit-cli-adoption.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=manual_active,
        activation_id="unit-cli-adoption-activation",
    )
    review = build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id="unit-cli-adoption-review",
        pages_base_url="https://example.test/reviews/unit-cli-adoption/",
    )
    promotion = promote_route_policy_scenario_ci_workflow(
        review,
        trigger_mode="pull-request",
        pull_request_branches=("main",),
        review_url="https://example.test/reviews/unit-cli-adoption/",
        promotion_id="unit-cli-adoption-promotion",
    )
    assert promotion.promoted is True

    manifest_path = write_route_policy_scenario_ci_manifest_json(tmp_path / "ci-manifest.json", manifest)
    workflow_index_path = write_route_policy_scenario_ci_workflow_json(tmp_path / "workflow.json", materialization)
    promotion_path = write_route_policy_scenario_ci_workflow_promotion_json(tmp_path / "promotion.json", promotion)

    adopted_source = tmp_path / "unit-cli-adoption-adopted.generated.yml"
    adopted_active = tmp_path / ".github" / "workflows" / "unit-cli-adoption-adopted.yml"
    output_path = tmp_path / "adoption.json"
    markdown_path = tmp_path / "adoption.md"

    args = build_parser().parse_args(
        [
            "route-policy-scenario-ci-workflow-adopt",
            "--manifest",
            str(manifest_path),
            "--workflow-index",
            str(workflow_index_path),
            "--promotion",
            str(promotion_path),
            "--adopted-workflow-output",
            str(adopted_source),
            "--adopted-active-workflow-output",
            str(adopted_active),
            "--adoption-id",
            "unit-cli-adoption",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--fail-on-adoption",
        ]
    )

    cli.cmd_route_policy_scenario_ci_workflow_adopt(args)
    report_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert report_payload["adoptionId"] == "unit-cli-adoption"
    assert report_payload["adopted"] is True
    assert report_payload["triggerMode"] == "pull-request"
    assert report_payload["pullRequestBranches"] == ["main"]
    assert report_payload["manualActiveWorkflowPath"] == manual_active.as_posix()
    assert report_payload["adoptedActiveWorkflowPath"] == adopted_active.as_posix()
    assert adopted_active.read_text(encoding="utf-8") != manual_active.read_text(encoding="utf-8")
    assert "pull_request:" in adopted_active.read_text(encoding="utf-8")
    assert "Route Policy Scenario CI Workflow Adoption: unit-cli-adoption" in markdown_path.read_text(encoding="utf-8")


def build_unit_ci_review_artifact(
    tmp_path: Path,
    *,
    prefix: str,
    pages_base_url: str,
) -> RoutePolicyScenarioCIReviewArtifact:
    manifest = build_unit_ci_workflow_manifest(f"{prefix}-manifest")
    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(workflow_id=f"{prefix}-workflow", artifact_root="ci"),
    )
    source_path = write_route_policy_scenario_ci_workflow_yaml(
        tmp_path / f"{prefix}.generated.yml",
        materialization,
    )
    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id=f"{prefix}-validation",
        workflow_path=source_path,
    )
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_path,
        active_workflow_path=tmp_path / ".github" / "workflows" / f"{prefix}.yml",
        activation_id=f"{prefix}-activation",
    )
    return build_route_policy_scenario_ci_review_artifact(
        build_unit_ci_shard_merge_report(),
        validation,
        activation,
        review_id=prefix,
        pages_base_url=pages_base_url,
    )


def build_unit_ci_workflow_manifest(manifest_id: str) -> RoutePolicyScenarioCIManifest:
    shard_command = (
        "gs-mapper",
        "route-policy-scenario-set",
        "--scenario-set",
        "shards/unit-shard.json",
        "--report-dir",
        "ci/reports/unit-shard",
        "--output",
        "ci/runs/unit-shard.json",
        "--history-output",
        "ci/histories/unit-shard.json",
    )
    merge_command = (
        "gs-mapper",
        "route-policy-scenario-shard-merge",
        "--run",
        "ci/runs/unit-shard.json",
        "--output",
        "ci/merge.json",
        "--history-output",
        "ci/history.json",
    )
    return RoutePolicyScenarioCIManifest(
        manifest_id=manifest_id,
        shard_plan_id="unit-shard-plan",
        shard_jobs=(
            RoutePolicyScenarioCIShardJob(
                job_id="scenario-unit-shard",
                shard_id="unit-shard",
                source_scenario_set_id="unit-source-scenarios",
                scenario_set_path="shards/unit-shard.json",
                scenario_count=1,
                report_dir="ci/reports/unit-shard",
                run_output="ci/runs/unit-shard.json",
                history_output="ci/histories/unit-shard.json",
                cache_key="unit-cache",
                expected_report_paths=("ci/reports/unit-shard/unit-scenario.json",),
                command=shard_command,
            ),
        ),
        merge_job=RoutePolicyScenarioCIMergeJob(
            job_id="route-policy-scenario-merge",
            merge_id="unit-merge",
            run_inputs=("ci/runs/unit-shard.json",),
            output="ci/merge.json",
            history_output="ci/history.json",
            cache_key="unit-cache-merge",
            depends_on=("scenario-unit-shard",),
            command=merge_command,
        ),
    )


def build_unit_ci_shard_merge_report() -> RoutePolicyScenarioShardMergeReport:
    history = RoutePolicyBenchmarkHistoryReport(
        history_id="unit-ci-review-history",
        reports=(
            RoutePolicyBenchmarkSnapshot(
                benchmark_id="unit-scenario",
                passed=True,
                best_policy_name="direct",
                policies=(
                    RoutePolicyBenchmarkPolicySnapshot(
                        policy_name="direct",
                        passed=True,
                        metrics={"successRate": 1.0, "collisionRate": 0.0},
                    ),
                ),
                source_path="ci/reports/unit-shard/unit-scenario.json",
            ),
        ),
    )
    return RoutePolicyScenarioShardMergeReport(
        merge_id="unit-ci-review-merge",
        shard_runs=(
            RoutePolicyScenarioShardRunSummary(
                shard_id="unit-shard",
                scenario_set_id="unit-shard",
                passed=True,
                scenario_count=1,
                report_paths=("ci/reports/unit-shard/unit-scenario.json",),
                run_path="ci/runs/unit-shard.json",
                history_path="ci/histories/unit-shard.json",
            ),
        ),
        history=history,
        history_path="ci/history.json",
        history_markdown_path="ci/history.md",
    )


def target_position_keys() -> tuple[str, str, str]:
    return ("payload.target.position.0", "payload.target.position.1", "payload.target.position.2")


def build_adapter(*, max_steps: int = 4) -> RoutePolicyGymAdapter:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    return RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            max_steps=max_steps,
            goal_reward=2.0,
            reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
        ),
    )


def direct_goal_policy(observation, info):
    del observation
    return {
        "routeId": f"direct-{info['stepIndex']}",
        "target": info["goal"],
    }


def build_unit_catalog():
    return build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/unit-scene/unit-scene.splat",
                    "label": "Unit Scene",
                    "summary": "Generic unit scene",
                }
            ]
        },
        docs_root=Path("."),
        site_url="https://example.test/gs/",
    )


def write_unit_scene_catalog(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "url": "assets/unit-scene/unit-scene.splat",
                        "label": "Unit Scene",
                        "summary": "Generic unit scene",
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_benchmark_report_fixture(
    path: Path,
    *,
    benchmark_id: str,
    policies: dict[str, dict[str, float]],
    passed: bool = True,
) -> Path:
    policy_rows = [
        {
            "policyName": policy_name,
            "passed": passed,
            "metrics": metrics,
            "failedChecks": [],
        }
        for policy_name, metrics in policies.items()
    ]
    path.write_text(
        json.dumps(
            {
                "recordType": "route-policy-benchmark-report",
                "version": "gs-mapper-route-policy-benchmark/v1",
                "benchmarkId": benchmark_id,
                "passed": passed,
                "bestPolicyName": next(iter(policies)),
                "summary": {
                    "evaluationId": benchmark_id,
                    "bestPolicyName": next(iter(policies)),
                    "policyCount": len(policy_rows),
                    "policies": policy_rows,
                },
                "modelSummary": {},
                "metadata": {"sceneId": "unit-scene"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
