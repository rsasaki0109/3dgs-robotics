"""Tests for scripts/publish_production_review_bundle.py.

The publish wrapper is the operator entry point that takes a
``review.json`` from an external production scenario CI run and lands it
under ``docs/reviews/<bundle-id>/`` (plus regenerates the reviews
index). The script never runs a benchmark itself, so these tests stub
the input ``review.json`` from the synthetic smoke fixture with
``provenance.kind`` rewritten to ``production``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "publish_production_review_bundle.py"
SAMPLE_REVIEW_PATH = REPO_ROOT / "docs" / "reviews" / "smoke-route-policy-ci" / "review.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "publish_production_review_bundle", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_production_review_payload(tmp_path: Path) -> Path:
    """Return a tmp ``review.json`` derived from the sample with kind=production."""

    payload = json.loads(SAMPLE_REVIEW_PATH.read_text(encoding="utf-8"))
    provenance = dict(payload.get("provenance") or {})
    provenance["kind"] = "production"
    provenance["assetSource"] = "test-fixture://production"
    provenance["sceneId"] = "outdoor-demo"
    payload["provenance"] = provenance
    # Strip the synthetic sampleBundle / sampleSource markers so the
    # republished bundle does not advertise itself as a sample.
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("sampleBundle", None)
    metadata.pop("sampleSource", None)
    metadata.pop("sampleNotice", None)
    metadata.pop("pagesBaseUrl", None)
    payload["metadata"] = metadata
    out = tmp_path / "production-review.json"
    out.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return out


def test_publish_writes_bundle_dir_and_refreshes_index(tmp_path: Path) -> None:
    module = _load_script_module()
    review_path = _build_production_review_payload(tmp_path)
    docs_dir = tmp_path / "docs"
    (docs_dir / "reviews").mkdir(parents=True)

    bundle_dir = module.publish_production_review_bundle(
        review_json=review_path,
        bundle_id="outdoor-demo-direct-baseline-001",
        docs_dir=docs_dir,
    )

    assert bundle_dir == docs_dir / "reviews" / "outdoor-demo-direct-baseline-001"
    # Bundle artefacts written via write_route_policy_scenario_ci_review_bundle.
    assert (bundle_dir / "review.json").exists()
    assert (bundle_dir / "review.md").exists()
    assert (bundle_dir / "index.html").exists()
    # Index regenerated.
    index_json = json.loads((docs_dir / "reviews" / "index.json").read_text(encoding="utf-8"))
    review_ids = {entry["reviewId"] for entry in index_json["entries"]}
    # The reviewId comes from the original payload, not the bundle directory name.
    assert review_ids, "index must list at least the just-published bundle"
    assert index_json["productionCount"] == 1
    assert index_json["syntheticCount"] == 0


def test_publish_rejects_synthetic_kind(tmp_path: Path) -> None:
    module = _load_script_module()
    docs_dir = tmp_path / "docs"
    (docs_dir / "reviews").mkdir(parents=True)
    review_path = tmp_path / "synthetic-review.json"
    review_path.write_text(
        json.dumps(
            {
                "recordType": "route-policy-scenario-ci-review",
                "reviewId": "synthetic",
                "passed": True,
                "shardCount": 1,
                "scenarioCount": 1,
                "reportCount": 1,
                "provenance": {
                    "recordType": "route-policy-scenario-ci-review-provenance",
                    "version": "gs-mapper-route-policy-scenario-ci-review-provenance/v1",
                    "kind": "synthetic",
                    "generatedAt": "2026-05-15T00:00:00+00:00",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.PublishError, match="provenance.kind"):
        module.publish_production_review_bundle(
            review_json=review_path,
            bundle_id="bogus-production",
            docs_dir=docs_dir,
        )


def test_publish_rejects_invalid_bundle_id(tmp_path: Path) -> None:
    module = _load_script_module()
    review_path = _build_production_review_payload(tmp_path)
    docs_dir = tmp_path / "docs"
    (docs_dir / "reviews").mkdir(parents=True)
    with pytest.raises(module.PublishError, match="kebab-case"):
        module.publish_production_review_bundle(
            review_json=review_path,
            bundle_id="Bad ID with spaces",
            docs_dir=docs_dir,
        )


def test_publish_rejects_missing_review_json(tmp_path: Path) -> None:
    module = _load_script_module()
    docs_dir = tmp_path / "docs"
    (docs_dir / "reviews").mkdir(parents=True)
    with pytest.raises(module.PublishError, match="not found"):
        module.publish_production_review_bundle(
            review_json=tmp_path / "does-not-exist.json",
            bundle_id="x",
            docs_dir=docs_dir,
        )


def test_publish_replaces_existing_bundle(tmp_path: Path) -> None:
    module = _load_script_module()
    review_path = _build_production_review_payload(tmp_path)
    docs_dir = tmp_path / "docs"
    (docs_dir / "reviews").mkdir(parents=True)

    # First publish establishes the bundle.
    first = module.publish_production_review_bundle(
        review_json=review_path,
        bundle_id="outdoor-demo-direct-baseline-001",
        docs_dir=docs_dir,
    )
    stale_marker = first / "stale-file.txt"
    stale_marker.write_text("should be removed on re-publish", encoding="utf-8")

    # Second publish should atomically replace the bundle dir.
    module.publish_production_review_bundle(
        review_json=review_path,
        bundle_id="outdoor-demo-direct-baseline-001",
        docs_dir=docs_dir,
    )
    assert not stale_marker.exists(), "re-publish must drop stale files"
