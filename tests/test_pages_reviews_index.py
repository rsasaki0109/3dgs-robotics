"""Tests for scripts/build_pages_reviews_index.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_pages_reviews_index.py"
SAMPLE_SCRIPT = REPO_ROOT / "scripts" / "build_pages_sample_review_bundle.py"


def _load_script_module(path: Path = SCRIPT, name: str = "build_pages_reviews_index"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_review_bundle(
    root: Path,
    *,
    review_id: str,
    passed: bool = True,
    shard_count: int = 1,
    scenario_count: int = 1,
    report_count: int = 1,
    adoption: dict | None = None,
    write_bundle_html: bool = True,
    provenance: dict | None = None,
    metadata: dict | None = None,
) -> Path:
    bundle = root / review_id
    bundle.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "recordType": "route-policy-scenario-ci-review",
        "reviewId": review_id,
        "passed": bool(passed),
        "shardCount": shard_count,
        "scenarioCount": scenario_count,
        "reportCount": report_count,
    }
    if adoption is not None:
        payload["adoption"] = adoption
    if provenance is not None:
        payload["provenance"] = provenance
    if metadata is not None:
        payload["metadata"] = metadata
    (bundle / "review.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    if write_bundle_html:
        (bundle / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    return bundle


def test_script_file_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111
    assert SAMPLE_SCRIPT.is_file()
    assert SAMPLE_SCRIPT.stat().st_mode & 0o111


def test_collect_review_entries_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    entries = module.collect_review_entries(tmp_path / "missing")

    assert entries == []


def test_collect_review_entries_skips_non_bundle_children(tmp_path: Path) -> None:
    module = _load_script_module()

    # Valid bundle.
    _write_review_bundle(tmp_path, review_id="alpha")
    # Sub-dir without review.json should be ignored.
    (tmp_path / "not-a-bundle").mkdir()
    (tmp_path / "not-a-bundle" / "readme.txt").write_text("ignore me", encoding="utf-8")
    # File at the top level should be ignored.
    (tmp_path / "loose-file.txt").write_text("ignore me", encoding="utf-8")
    # Sub-dir whose review.json is not a review record (wrong recordType).
    (tmp_path / "wrong-type").mkdir()
    (tmp_path / "wrong-type" / "review.json").write_text(json.dumps({"recordType": "something-else"}), encoding="utf-8")

    entries = module.collect_review_entries(tmp_path)

    assert [entry.review_id for entry in entries] == ["alpha"]


def test_write_reviews_index_covers_mixed_bundles(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(
        tmp_path,
        review_id="bravo",
        passed=True,
        shard_count=2,
        scenario_count=4,
        report_count=4,
        adoption={
            "recordType": "route-policy-scenario-ci-review-adoption",
            "adoptionId": "bravo-adoption",
            "adopted": True,
            "triggerMode": "pull-request",
            "adoptedActiveWorkflowPath": ".github/workflows/bravo-adopted.yml",
            "adoptedSourceWorkflowPath": "runs/bravo-adopted.generated.yml",
        },
    )
    _write_review_bundle(
        tmp_path,
        review_id="alpha",
        passed=False,
        shard_count=1,
        scenario_count=1,
        report_count=1,
    )

    html_output = tmp_path / "index.html"
    json_output = tmp_path / "index.json"
    markdown_output = tmp_path / "index.md"

    entries = module.write_reviews_index(
        tmp_path,
        html_output=html_output,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    # Stable alphabetical sort so the index is deterministic.
    assert [entry.review_id for entry in entries] == ["alpha", "bravo"]

    index_payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert index_payload["recordType"] == "gs-mapper-route-policy-scenario-ci-reviews-index/v1"
    assert index_payload["reviewCount"] == 2
    assert index_payload["passCount"] == 1
    assert index_payload["failCount"] == 1
    assert [entry["reviewId"] for entry in index_payload["entries"]] == ["alpha", "bravo"]
    assert index_payload["entries"][1]["adoptionTriggerMode"] == "pull-request"
    assert index_payload["entries"][1]["adoptionAdopted"] is True
    assert index_payload["entries"][0]["adoptionAdopted"] is None

    html_text = html_output.read_text(encoding="utf-8")
    assert "Route Policy Scenario CI Reviews" in html_text
    assert 'href="alpha/index.html"' in html_text
    assert 'href="bravo/index.html"' in html_text
    # Pass / fail pills for both.
    assert '<span class="pill pass">PASS</span>' in html_text
    assert '<span class="pill fail">FAIL</span>' in html_text
    # Adoption pill + trigger mode for bravo.
    assert '<span class="pill pass">ADOPTED</span>' in html_text
    assert "<code>pull-request</code>" in html_text
    # Alpha has no adoption → "none" pill.
    assert '<span class="pill info">none</span>' in html_text

    markdown_text = markdown_output.read_text(encoding="utf-8")
    assert "| [alpha](alpha/index.html) | unknown | FAIL |" in markdown_text
    assert "| [bravo](bravo/index.html) | unknown | PASS |" in markdown_text
    assert "ADOPTED (`pull-request`)" in markdown_text
    # The kind columns default to unknown for bundles without a provenance
    # block and without the legacy ``metadata.sampleBundle`` hint.
    assert "- Production: 0" in markdown_text
    assert "- Synthetic: 0" in markdown_text


def test_main_writes_default_outputs_inside_reviews_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(tmp_path, review_id="charlie")
    rc = module.main(["--reviews-dir", str(tmp_path)])

    assert rc == 0
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "index.json").is_file()
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert payload["reviewCount"] == 1


def test_empty_reviews_dir_still_writes_placeholder_index(tmp_path: Path) -> None:
    module = _load_script_module()

    html_output = tmp_path / "index.html"
    json_output = tmp_path / "index.json"
    entries = module.write_reviews_index(
        tmp_path,
        html_output=html_output,
        json_output=json_output,
    )

    assert entries == []
    assert json.loads(json_output.read_text(encoding="utf-8"))["reviewCount"] == 0
    assert "No review bundles published yet" in html_output.read_text(encoding="utf-8")


def test_collect_review_entries_extracts_provenance_kind(tmp_path: Path) -> None:
    """Bundles with first-class provenance surface kind / generatedAt / sceneId."""

    module = _load_script_module()

    _write_review_bundle(
        tmp_path,
        review_id="prod",
        provenance={
            "recordType": "route-policy-scenario-ci-review-provenance",
            "kind": "production",
            "generatedAt": "2026-05-15T08:30:00+00:00",
            "sceneId": "outdoor-demo",
        },
    )
    _write_review_bundle(
        tmp_path,
        review_id="syn",
        provenance={
            "recordType": "route-policy-scenario-ci-review-provenance",
            "kind": "synthetic",
            "generatedAt": "2026-05-15T00:00:00+00:00",
            "sceneId": "smoke",
        },
    )
    # Legacy synthetic bundle without provenance still falls back via metadata.
    _write_review_bundle(
        tmp_path,
        review_id="legacy",
        metadata={"sampleBundle": True},
    )
    # Bundle without provenance or sample hint is unknown rather than masquerading.
    _write_review_bundle(tmp_path, review_id="mystery")

    entries = {entry.review_id: entry for entry in module.collect_review_entries(tmp_path)}

    assert entries["prod"].kind == "production"
    assert entries["prod"].generated_at == "2026-05-15T08:30:00+00:00"
    assert entries["prod"].scene_id == "outdoor-demo"
    assert entries["syn"].kind == "synthetic"
    assert entries["legacy"].kind == "synthetic"
    assert entries["legacy"].generated_at is None
    assert entries["mystery"].kind == "unknown"

    payload = json.loads(module.render_reviews_index_json(list(entries.values())))
    assert payload["productionCount"] == 1
    assert payload["syntheticCount"] == 2
    assert payload["unknownCount"] == 1


def test_render_reviews_index_html_surfaces_kind_pills(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(
        tmp_path,
        review_id="prod",
        provenance={
            "recordType": "route-policy-scenario-ci-review-provenance",
            "kind": "production",
            "generatedAt": "2026-05-15T08:30:00+00:00",
            "sceneId": "outdoor-demo",
        },
    )
    _write_review_bundle(
        tmp_path,
        review_id="syn",
        provenance={
            "recordType": "route-policy-scenario-ci-review-provenance",
            "kind": "synthetic",
            "generatedAt": "2026-05-15T00:00:00+00:00",
        },
    )

    entries = module.collect_review_entries(tmp_path)
    html = module.render_reviews_index_html(entries)
    assert '<span class="pill production">PRODUCTION</span>' in html
    assert '<span class="pill synthetic">SYNTHETIC</span>' in html
    # Scene and generated columns are present in the table header.
    assert "<th>Kind</th>" in html
    assert "<th>Scene</th>" in html
    assert "<th>Generated</th>" in html
    # YYYY-MM-DD prefix is shown for the generated cell.
    assert ">2026-05-15<" in html


def test_render_reviews_index_unknown_kind_does_not_masquerade_as_production() -> None:
    """A bundle without provenance and without the sample hint must stay unknown."""

    module = _load_script_module()
    kind = module._resolve_entry_kind({}, {})
    assert kind == "unknown"
    kind = module._resolve_entry_kind({"kind": "production"}, {})
    assert kind == "production"
    kind = module._resolve_entry_kind({"kind": "bogus"}, {"sampleBundle": True})
    # Unknown provenance kind plus sampleBundle metadata falls back to synthetic.
    assert kind == "synthetic"


def test_entry_falls_back_to_bundle_dir_when_html_missing(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(tmp_path, review_id="delta", write_bundle_html=False)

    entries = module.collect_review_entries(tmp_path)

    # Without a per-bundle index.html, the href points at the directory so the
    # browser shows the Pages directory listing or a fallback.
    assert entries[0].bundle_html == "delta"


def test_sample_review_bundle_generator_writes_self_contained_pages_bundle(tmp_path: Path) -> None:
    module = _load_script_module(SAMPLE_SCRIPT, "build_pages_sample_review_bundle")

    docs_dir = tmp_path / "docs"
    bundle_dir = module.build_sample_review_bundle(docs_dir)

    assert bundle_dir == docs_dir / "reviews" / "smoke-route-policy-ci"
    payload = json.loads((bundle_dir / "review.json").read_text(encoding="utf-8"))
    assert payload["reviewId"] == "smoke-route-policy-ci-review"
    assert payload["passed"] is True
    assert payload["metadata"]["sampleBundle"] is True
    assert (
        payload["metadata"]["pagesBaseUrl"] == "https://rsasaki0109.github.io/gs-mapper/reviews/smoke-route-policy-ci/"
    )
    assert payload["adoption"]["triggerMode"] == "pull-request"
    assert payload["adoption"]["adopted"] is True
    assert "/tmp/" not in json.dumps(payload)
    for shard in payload["shards"]:
        assert (bundle_dir / shard["runPath"]).is_file()
        assert (bundle_dir / shard["historyPath"]).is_file()
        for report_path in shard["reportPaths"]:
            assert (bundle_dir / report_path).is_file()

    html_text = (bundle_dir / "index.html").read_text(encoding="utf-8")
    assert "Synthetic smoke fixture" in html_text
    assert "ADOPTED" in html_text
    assert "/tmp/" not in html_text
    for path in (bundle_dir / "sample-artifacts").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".yml"}:
            text = path.read_text(encoding="utf-8")
            assert "/tmp/" not in text, path
            assert "https://example.test" not in text, path

    # Synthetic sample bundle carries a first-class provenance block.
    assert payload["provenance"]["kind"] == "synthetic"
    assert payload["provenance"]["sceneId"] == "smoke-route-policy-ci"
    assert payload["provenance"]["generatedAt"]

    index_payload = json.loads((docs_dir / "reviews" / "index.json").read_text(encoding="utf-8"))
    assert index_payload["reviewCount"] == 1
    assert index_payload["entries"][0]["reviewId"] == "smoke-route-policy-ci-review"
    assert index_payload["entries"][0]["adoptionTriggerMode"] == "pull-request"
    assert index_payload["entries"][0]["kind"] == "synthetic"
    assert index_payload["syntheticCount"] == 1
    assert index_payload["productionCount"] == 0


def test_published_docs_reviews_index_includes_smoke_sample_bundle() -> None:
    reviews_dir = REPO_ROOT / "docs" / "reviews"
    payload = json.loads((reviews_dir / "index.json").read_text(encoding="utf-8"))

    entries = {entry["reviewId"]: entry for entry in payload["entries"]}
    sample = entries["smoke-route-policy-ci-review"]
    assert sample["passed"] is True
    assert sample["bundleHtml"] == "smoke-route-policy-ci/index.html"
    assert sample["shardCount"] == 2
    assert sample["scenarioCount"] == 2
    assert sample["reportCount"] == 2
    assert sample["adoptionTriggerMode"] == "pull-request"
    assert sample["adoptionAdopted"] is True

    review_payload = json.loads((reviews_dir / "smoke-route-policy-ci" / "review.json").read_text(encoding="utf-8"))
    assert review_payload["metadata"]["sampleBundle"] is True
    assert review_payload["provenance"]["kind"] == "synthetic"
    assert "/tmp/" not in json.dumps(review_payload)
    assert "https://example.test" not in json.dumps(review_payload)
    html_text = (reviews_dir / "smoke-route-policy-ci" / "index.html").read_text(encoding="utf-8")
    assert "Synthetic smoke fixture" in html_text
    for path in (reviews_dir / "smoke-route-policy-ci" / "sample-artifacts").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".yml"}:
            text = path.read_text(encoding="utf-8")
            assert "/tmp/" not in text, path
            assert "https://example.test" not in text, path
