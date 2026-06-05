"""Tests for the multi-agent surface on the scenario CI review bundle (PR D5).

PR D4 added the ``interaction_metrics_aggregate`` on the shard merge.
PR D5 surfaces it on the review artifact:

- ``interaction_metrics_aggregate`` is a first-class optional field that
  rides through to_dict / from_dict.
- The artifact exposes a ``multi_agent`` boolean derived from whether
  the aggregate is set.
- The Markdown / HTML renderers add a "Multi-agent interaction metrics"
  block when the aggregate is present, and the HTML subtitle carries a
  "Multi-agent" pill so reviewers can see at a glance that the bundle
  comes from a multi-agent run.
- The Pages reviews index surfaces ``multiAgent`` per entry and a
  ``multiAgentCount`` summary so the landing page can filter
  multi-agent runs.

The actual ``interactionMetricsValues`` collection inside the env is
deferred; these tests inject synthetic aggregates the same way D4 tests
do.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from gs_sim2real.sim import (
    InteractionMetricKeyStats,
    InteractionMetricsAggregate,
    route_policy_scenario_ci_review_from_dict,
)
from gs_sim2real.sim.policy_scenario_ci_review import (
    render_route_policy_scenario_ci_review_html,
    render_route_policy_scenario_ci_review_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_SCRIPT = REPO_ROOT / "scripts" / "build_pages_reviews_index.py"


def _load_index_module():
    spec = importlib.util.spec_from_file_location("build_pages_reviews_index", INDEX_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_aggregate() -> InteractionMetricsAggregate:
    return InteractionMetricsAggregate(
        per_key_stats={
            "min-peer-separation": InteractionMetricKeyStats(mean=0.42, p95=0.5, maximum=0.8, sample_count=12),
        },
        sample_scenario_count=12,
    )


def _build_artifact(*, with_aggregate: bool):
    """Construct a synthetic review artifact for renderer tests."""

    from gs_sim2real.sim.policy_scenario_ci_review import (
        RoutePolicyScenarioCIReviewArtifact,
        RoutePolicyScenarioCIReviewShard,
    )

    shards = (
        RoutePolicyScenarioCIReviewShard(
            shard_id="shard-001",
            passed=True,
            scenario_count=4,
            report_count=4,
        ),
    )
    return RoutePolicyScenarioCIReviewArtifact(
        review_id="d5-test-review",
        merge_id="d5-test-merge",
        workflow_id="d5-workflow",
        manifest_id="d5-manifest",
        validation_id="d5-validation",
        activation_id="d5-activation",
        validation_passed=True,
        activation_activated=True,
        shard_merge_passed=True,
        history_passed=True,
        active_workflow_path=".github/workflows/d5-active.yml",
        source_workflow_path=".github/workflows/d5-source.yml",
        shards=shards,
        interaction_metrics_aggregate=_sample_aggregate() if with_aggregate else None,
    )


def test_artifact_round_trips_interaction_metrics_aggregate() -> None:
    artifact = _build_artifact(with_aggregate=True)
    payload = artifact.to_dict()
    assert payload["multiAgent"] is True
    assert "interactionMetricsAggregate" in payload
    restored = route_policy_scenario_ci_review_from_dict(payload)
    assert restored.interaction_metrics_aggregate == artifact.interaction_metrics_aggregate
    assert restored.multi_agent is True


def test_artifact_omits_aggregate_when_legacy() -> None:
    artifact = _build_artifact(with_aggregate=False)
    payload = artifact.to_dict()
    assert "interactionMetricsAggregate" not in payload
    assert "multiAgent" not in payload
    assert artifact.multi_agent is False
    restored = route_policy_scenario_ci_review_from_dict(payload)
    assert restored.interaction_metrics_aggregate is None


def test_markdown_renders_interaction_metrics_block_when_aggregate_present() -> None:
    rendered = render_route_policy_scenario_ci_review_markdown(_build_artifact(with_aggregate=True))
    assert "## Multi-agent interaction metrics" in rendered
    assert "min-peer-separation" in rendered
    assert "0.4200" in rendered  # mean column
    assert "Contributing scenarios: 12" in rendered


def test_markdown_omits_interaction_metrics_block_for_legacy_runs() -> None:
    rendered = render_route_policy_scenario_ci_review_markdown(_build_artifact(with_aggregate=False))
    assert "## Multi-agent interaction metrics" not in rendered


def test_html_carries_multi_agent_pill_and_metrics_section() -> None:
    rendered = render_route_policy_scenario_ci_review_html(_build_artifact(with_aggregate=True))
    assert 'class="pill multi-agent">Multi-agent</span>' in rendered
    assert "Multi-agent interaction metrics" in rendered
    assert "min-peer-separation" in rendered


def test_html_skips_multi_agent_pill_for_legacy_runs() -> None:
    rendered = render_route_policy_scenario_ci_review_html(_build_artifact(with_aggregate=False))
    # The CSS class definition is unconditional; the rendered pill / section
    # should not appear when the aggregate is absent.
    assert 'class="pill multi-agent">Multi-agent</span>' not in rendered
    assert "Multi-agent interaction metrics" not in rendered


def test_pages_reviews_index_surfaces_multi_agent_flag(tmp_path: Path) -> None:
    module = _load_index_module()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    # One multi-agent bundle, one legacy bundle.
    _write_bundle(reviews_dir, "ma-bundle", multi_agent=True)
    _write_bundle(reviews_dir, "legacy-bundle", multi_agent=False)
    module.write_reviews_index(
        reviews_dir,
        html_output=reviews_dir / "index.html",
        json_output=reviews_dir / "index.json",
    )
    payload = json.loads((reviews_dir / "index.json").read_text(encoding="utf-8"))
    assert payload["multiAgentCount"] == 1
    entries_by_id = {entry["reviewId"]: entry for entry in payload["entries"]}
    assert entries_by_id["ma-bundle-review"]["multiAgent"] is True
    assert entries_by_id["legacy-bundle-review"]["multiAgent"] is False


def _write_bundle(reviews_dir: Path, bundle_id: str, *, multi_agent: bool) -> None:
    bundle_dir = reviews_dir / bundle_id
    bundle_dir.mkdir()
    payload: dict = {
        "recordType": "route-policy-scenario-ci-review",
        "reviewId": f"{bundle_id}-review",
        "passed": True,
        "shardCount": 1,
        "scenarioCount": 1,
        "reportCount": 1,
    }
    if multi_agent:
        payload["multiAgent"] = True
    (bundle_dir / "review.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    (bundle_dir / "index.html").write_text("<html></html>", encoding="utf-8")
