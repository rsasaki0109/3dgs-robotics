#!/usr/bin/env python3
"""Smoke recipe for the route policy scenario CI chain.

Walks ``matrix -> shard plan -> scenario-set runs -> shard merge -> CI
manifest -> workflow materialization -> validation -> activation -> review
bundle -> promotion -> adoption`` inside a tmpdir, using a tiny one-scene /
one-policy fixture. Each gate prints a ``PASS``/``FAIL`` line; the script
exits non-zero at the first failing gate so CI can point at it directly.

Both the manual-only and the adopted, trigger-enabled workflows are written
under ``<tmpdir>/.github/workflows/...`` so the real repo's
``.github/workflows/`` is never touched.

Usage
-----

::

    PYTHONPATH=src python3 scripts/smoke_route_policy_scenario_ci.py
    PYTHONPATH=src python3 scripts/smoke_route_policy_scenario_ci.py --keep

``--keep`` preserves the tmpdir and prints its path (handy for inspecting
generated artifacts by hand).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

from gs_sim2real.sim import (
    AgentRoleSpec,
    InteractionMetricsSpec,
    Pose3D,
    RoutePolicyGoalSpec,
    RoutePolicyGoalSuite,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyRegistry,
    RoutePolicyRegistryEntry,
    RoutePolicyScenarioCIWorkflowConfig,
    RoutePolicyScenarioMatrix,
    activate_route_policy_scenario_ci_workflow,
    adopt_route_policy_scenario_ci_workflow,
    build_route_policy_scenario_ci_manifest,
    build_route_policy_scenario_ci_review_adoption,
    build_route_policy_scenario_ci_review_artifact,
    expand_route_policy_scenario_matrix_to_directory,
    load_route_policy_registry_json,
    materialize_route_policy_scenario_ci_workflow,
    merge_route_policy_scenario_shard_run_jsons,
    promote_route_policy_scenario_ci_workflow,
    run_route_policy_scenario_set,
    validate_route_policy_scenario_ci_workflow,
    write_route_policy_goal_suite_json,
    write_route_policy_registry_json,
    write_route_policy_scenario_ci_manifest_json,
    write_route_policy_scenario_ci_review_bundle,
    write_route_policy_scenario_ci_review_json,
    write_route_policy_scenario_ci_workflow_activation_json,
    write_route_policy_scenario_ci_workflow_adoption_json,
    write_route_policy_scenario_ci_workflow_json,
    write_route_policy_scenario_ci_workflow_promotion_json,
    write_route_policy_scenario_ci_workflow_validation_json,
    write_route_policy_scenario_ci_workflow_yaml,
    write_route_policy_scenario_matrix_expansion_json,
    write_route_policy_scenario_matrix_json,
    write_route_policy_scenario_set_run_json,
    write_route_policy_scenario_shard_merge_json,
    write_route_policy_scenario_shard_plan_json,
    write_route_policy_scenario_shards_from_expansion,
)

SMOKE_PREFIX = "smoke-route-policy-ci"
PAGES_BASE_URL = f"https://example.test/reviews/{SMOKE_PREFIX}/"


class GateFailure(RuntimeError):
    """Raised when a scenario CI gate fails in the smoke chain."""


def _write_scene_catalog(path: Path) -> Path:
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


def _seed_fixture(root: Path) -> tuple[Path, Path]:
    """Write registry / scene catalog / goal suites and the matrix JSON."""

    _write_scene_catalog(root / "scenes.json")
    registry_path = write_route_policy_registry_json(
        root / "registry.json",
        RoutePolicyRegistry(
            registry_id=f"{SMOKE_PREFIX}-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        root / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id=f"{SMOKE_PREFIX}-near",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    write_route_policy_goal_suite_json(
        root / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id=f"{SMOKE_PREFIX}-far",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    # Sprint 4 / PR D6: alongside the original ego-only "unit" scene, the
    # smoke chain now exercises a 2-agent deterministic crossing scene
    # (ego + one chase peer). This produces multi-agent scenarios that
    # flow end-to-end through expansion → shard → merge → review so the
    # `interactionMetricsAggregate` and `multiAgent` review surfaces have
    # production-shaped exercise.
    crossing_scene = RoutePolicyMatrixSceneSpec(
        "crossing",
        "scenes.json",
        scene_id="unit-scene",
        agents=(
            AgentRoleSpec(
                agent_id="ego",
                role="ego",
                start_pose=_unit_pose((0.0, 0.0, 0.0)),
            ),
            AgentRoleSpec(
                agent_id="peer-chase",
                role="peer-obstacle",
                start_pose=_unit_pose((2.0, 2.0, 0.0)),
                builtin_policy="chase",
            ),
        ),
        interaction_metrics=InteractionMetricsSpec(
            aggregate_keys=("peer-count", "min-peer-separation-meters"),
        ),
    )
    matrix_path = write_route_policy_scenario_matrix_json(
        root / "matrix.json",
        RoutePolicyScenarioMatrix(
            matrix_id=f"{SMOKE_PREFIX}-matrix",
            registries=(RoutePolicyMatrixRegistrySpec("direct", "registry.json"),),
            scenes=(
                RoutePolicyMatrixSceneSpec("unit", "scenes.json", scene_id="unit-scene"),
                crossing_scene,
            ),
            goal_suites=(
                RoutePolicyMatrixGoalSuiteSpec("near", "near-goals.json"),
                RoutePolicyMatrixGoalSuiteSpec("far", "far-goals.json"),
            ),
            configs=(RoutePolicyMatrixConfigSpec("short", episode_count=1, seed_start=0, max_steps=4),),
        ),
    )
    return matrix_path, registry_path


def _unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="generic_world",
    )


def _gate(log: Callable[[str], None], name: str, ok: bool, detail: str = "") -> None:
    """Log a gate result and raise if it failed."""

    marker = "PASS" if ok else "FAIL"
    suffix = f" ({detail})" if detail else ""
    log(f"[{marker}] {name}{suffix}")
    if not ok:
        raise GateFailure(f"{name} failed{suffix}")


def run_smoke(root: Path, *, log: Callable[[str], None] = print) -> dict[str, Path]:
    """Run the full scenario-CI chain against ``root`` and return artifact paths."""

    matrix_path, registry_path = _seed_fixture(root)

    expansion = expand_route_policy_scenario_matrix_to_directory(
        _load_matrix(matrix_path),
        root / "generated",
        matrix_base_path=root,
    )
    expansion_path = write_route_policy_scenario_matrix_expansion_json(root / "matrix-expansion.json", expansion)
    log(f"[stage] matrix expansion -> {expansion_path}")

    plan = write_route_policy_scenario_shards_from_expansion(
        expansion,
        root / "shards",
        max_scenarios_per_shard=1,
        shard_plan_id=f"{SMOKE_PREFIX}-shards",
    )
    plan_path = write_route_policy_scenario_shard_plan_json(root / "shard-plan.json", plan)
    log(f"[stage] shard plan -> {plan_path} ({plan.shard_count} shards)")

    registry = load_route_policy_registry_json(registry_path)
    run_paths: list[Path] = []
    for scenario_set in plan.scenario_sets:
        run = run_route_policy_scenario_set(
            scenario_set,
            registry,
            report_dir=root / "reports" / scenario_set.scenario_set_id,
            scenario_set_base_path=root / "shards",
            registry_base_path=root,
            policy_registry_path=registry_path,
            history_output=root / "histories" / f"{scenario_set.scenario_set_id}.json",
        )
        _gate(
            log,
            f"scenario-set run :: {scenario_set.scenario_set_id}",
            run.passed,
        )
        run_paths.append(
            write_route_policy_scenario_set_run_json(root / "runs" / f"{scenario_set.scenario_set_id}.json", run)
        )

    merge = merge_route_policy_scenario_shard_run_jsons(
        tuple(run_paths),
        merge_id=f"{SMOKE_PREFIX}-merge",
        history_output=root / "shard-history.json",
        history_markdown_output=root / "shard-history.md",
    )
    merge_path = write_route_policy_scenario_shard_merge_json(root / "shard-merge.json", merge)
    _gate(log, "shard merge", merge.passed, f"{merge.shard_count} shards")

    manifest = build_route_policy_scenario_ci_manifest(
        plan,
        manifest_id=f"{SMOKE_PREFIX}-manifest",
        report_dir="ci/reports",
        run_output_dir="ci/runs",
        history_output_dir="ci/histories",
        merge_id=f"{SMOKE_PREFIX}-merge",
        merge_output="ci/merge.json",
        merge_history_output="ci/history.json",
        include_markdown=True,
        cache_key_prefix=f"{SMOKE_PREFIX}-cache",
        fail_on_regression=True,
    )
    manifest_path = write_route_policy_scenario_ci_manifest_json(root / "ci-manifest.json", manifest)
    log(f"[stage] CI manifest -> {manifest_path} ({manifest.shard_job_count} shard jobs)")

    materialization = materialize_route_policy_scenario_ci_workflow(
        manifest,
        config=RoutePolicyScenarioCIWorkflowConfig(
            workflow_id=f"{SMOKE_PREFIX}-workflow",
            workflow_name="Smoke Route Policy Scenario CI",
            artifact_root="ci",
        ),
    )
    source_workflow_path = write_route_policy_scenario_ci_workflow_yaml(
        root / "ci-workflow.generated.yml", materialization
    )
    workflow_index_path = write_route_policy_scenario_ci_workflow_json(root / "ci-workflow.json", materialization)
    log(f"[stage] workflow materialization -> {source_workflow_path} (index {workflow_index_path})")

    validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        materialization,
        validation_id=f"{SMOKE_PREFIX}-validation",
        workflow_path=source_workflow_path,
    )
    validation_path = write_route_policy_scenario_ci_workflow_validation_json(
        root / "ci-workflow-validation.json", validation
    )
    _gate(log, "workflow validation", validation.passed)

    active_workflow_path = root / ".github" / "workflows" / f"{SMOKE_PREFIX}.yml"
    activation = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation,
        source_workflow_path=source_workflow_path,
        active_workflow_path=active_workflow_path,
        activation_id=f"{SMOKE_PREFIX}-activation",
    )
    activation_path = write_route_policy_scenario_ci_workflow_activation_json(
        root / "ci-workflow-activation.json", activation
    )
    _gate(
        log,
        "workflow activation",
        activation.activated,
        f"active path {active_workflow_path}",
    )

    review = build_route_policy_scenario_ci_review_artifact(
        merge,
        validation,
        activation,
        review_id=f"{SMOKE_PREFIX}-review",
        pages_base_url=PAGES_BASE_URL,
    )
    review_json_path = write_route_policy_scenario_ci_review_json(root / "ci-review.json", review)
    bundle_paths = write_route_policy_scenario_ci_review_bundle(root / "pages" / SMOKE_PREFIX, review)
    _gate(log, "review artifact", review.passed)
    _gate(
        log,
        "review multi-agent surface",
        review.multi_agent and review.interaction_metrics_aggregate is not None,
        "crossing scene contributed interactionMetricsAggregate" if review.multi_agent else "no aggregate attached",
    )
    log(
        f"[stage] review bundle -> json={bundle_paths['json']} "
        f"md={bundle_paths['markdown']} html={bundle_paths['html']}"
    )

    promotion = promote_route_policy_scenario_ci_workflow(
        review,
        trigger_mode="pull-request",
        pull_request_branches=("main",),
        review_url=PAGES_BASE_URL,
        promotion_id=f"{SMOKE_PREFIX}-promotion",
    )
    promotion_path = write_route_policy_scenario_ci_workflow_promotion_json(
        root / "ci-workflow-promotion.json", promotion
    )
    _gate(log, "workflow promotion", promotion.promoted)

    adopted_source_path = root / "ci-workflow-adopted.generated.yml"
    adopted_active_path = root / ".github" / "workflows" / f"{SMOKE_PREFIX}-adopted.yml"
    adoption = adopt_route_policy_scenario_ci_workflow(
        promotion,
        manifest,
        materialization,
        adopted_source_workflow_path=adopted_source_path,
        adopted_active_workflow_path=adopted_active_path,
        adoption_id=f"{SMOKE_PREFIX}-adoption",
    )
    adoption_path = write_route_policy_scenario_ci_workflow_adoption_json(root / "ci-workflow-adoption.json", adoption)
    _gate(
        log,
        "workflow adoption",
        adoption.adopted,
        f"adopted active path {adopted_active_path}",
    )

    # Re-publish the review bundle with adoption info so reviewers on Pages
    # can diff the manual-only vs. adopted YAMLs without checking out.
    review_adoption = build_route_policy_scenario_ci_review_adoption(
        adoption_id=adoption.adoption_id,
        adopted=adoption.adopted,
        trigger_mode=adoption.trigger_mode,
        adopted_active_workflow_path=adoption.adopted_active_workflow_path,
        adopted_source_workflow_path=adoption.adopted_source_workflow_path,
        manual_workflow_text=active_workflow_path.read_text(encoding="utf-8"),
        adopted_workflow_text=adopted_active_path.read_text(encoding="utf-8"),
        push_branches=adoption.push_branches,
        pull_request_branches=adoption.pull_request_branches,
    )
    review_with_adoption = build_route_policy_scenario_ci_review_artifact(
        merge,
        validation,
        activation,
        review_id=f"{SMOKE_PREFIX}-review",
        pages_base_url=PAGES_BASE_URL,
        adoption=review_adoption,
    )
    review_json_path = write_route_policy_scenario_ci_review_json(root / "ci-review.json", review_with_adoption)
    bundle_paths = write_route_policy_scenario_ci_review_bundle(root / "pages" / SMOKE_PREFIX, review_with_adoption)
    _gate(log, "review adoption bundle", review_with_adoption.adoption is not None)

    return {
        "matrix": matrix_path,
        "expansion": expansion_path,
        "shard_plan": plan_path,
        "shard_merge": merge_path,
        "ci_manifest": manifest_path,
        "workflow_yaml": source_workflow_path,
        "workflow_index": workflow_index_path,
        "workflow_validation": validation_path,
        "workflow_activation": activation_path,
        "active_workflow": active_workflow_path,
        "review": review_json_path,
        "review_bundle_html": Path(bundle_paths["html"]),
        "review_bundle_markdown": Path(bundle_paths["markdown"]),
        "promotion": promotion_path,
        "adoption": adoption_path,
        "adopted_source_workflow": adopted_source_path,
        "adopted_active_workflow": adopted_active_path,
    }


def _load_matrix(path: Path) -> RoutePolicyScenarioMatrix:
    from gs_sim2real.sim import load_route_policy_scenario_matrix_json

    return load_route_policy_scenario_matrix_json(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the tmpdir after the run and print its path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Use this directory instead of a system tmpdir. Implies --keep.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    keep = args.keep or args.root is not None

    if args.root is not None:
        root = args.root
        root.mkdir(parents=True, exist_ok=True)
        tmp_handle: tempfile.TemporaryDirectory[str] | None = None
    elif keep:
        root = Path(tempfile.mkdtemp(prefix=f"{SMOKE_PREFIX}-"))
        tmp_handle = None
    else:
        tmp_handle = tempfile.TemporaryDirectory(prefix=f"{SMOKE_PREFIX}-")
        root = Path(tmp_handle.name)

    try:
        try:
            run_smoke(root)
        except GateFailure as exc:
            print(f"[error] smoke chain halted: {exc}", file=sys.stderr)
            return 2
        print(f"[ok] scenario CI smoke chain passed under {root}")
        if keep:
            print(f"[keep] artifacts retained at {root}")
        return 0
    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
