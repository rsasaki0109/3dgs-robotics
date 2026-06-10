#!/usr/bin/env python3
"""Generate the synthetic Pages review bundle used as the public sample.

The bundle is intentionally derived from ``scripts/smoke_route_policy_scenario_ci.py``
instead of hand-written JSON. That keeps the Pages example aligned with the
real scenario-CI chain while still avoiding claims that it is a production
benchmark run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SRC = REPO / "src"
SMOKE_SCRIPT = REPO / "scripts" / "smoke_route_policy_scenario_ci.py"
DEFAULT_BUNDLE_ID = "smoke-route-policy-ci"
SAMPLE_ARTIFACT_DIR = "sample-artifacts"
DEFAULT_SAMPLE_NOTICE = (
    "Synthetic smoke fixture generated from scripts/smoke_route_policy_scenario_ci.py; "
    "it proves the scenario-CI review bundle contract but is not a production benchmark run."
)
DEFAULT_PAGES_BASE_URL = "https://rsasaki0109.github.io/3dgs-robotics/reviews/smoke-route-policy-ci/"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_sample_review_bundle(
    docs_dir: Path = DOCS,
    *,
    bundle_id: str = DEFAULT_BUNDLE_ID,
    sample_notice: str = DEFAULT_SAMPLE_NOTICE,
    pages_base_url: str = DEFAULT_PAGES_BASE_URL,
    generated_at: str | None = None,
) -> Path:
    """Run the smoke chain and publish a self-contained Pages sample bundle."""

    from gs_sim2real.sim import route_policy_scenario_ci_review_from_dict, write_route_policy_scenario_ci_review_bundle

    smoke = _load_smoke_module()
    reviews_dir = docs_dir / "reviews"
    bundle_dir = reviews_dir / bundle_id
    tmp_bundle_dir = reviews_dir / f".{bundle_id}.tmp"
    if tmp_bundle_dir.exists():
        shutil.rmtree(tmp_bundle_dir)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    resolved_generated_at = generated_at or _resolve_generated_at_pin()

    with tempfile.TemporaryDirectory(prefix=f"{bundle_id}-") as raw_tmp:
        smoke_root = Path(raw_tmp)
        artifacts = smoke.run_smoke(smoke_root, log=lambda _line: None)
        payload = json.loads(artifacts["review"].read_text(encoding="utf-8"))
        payload = _rewrite_smoke_paths(payload, smoke_root, pages_base_url=pages_base_url)
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "pagesBaseUrl": pages_base_url,
                "sampleBundle": True,
                "sampleSource": "scripts/smoke_route_policy_scenario_ci.py",
                "sampleNotice": sample_notice,
            }
        )
        payload["metadata"] = metadata
        payload["provenance"] = {
            "recordType": "route-policy-scenario-ci-review-provenance",
            "version": "gs-mapper-route-policy-scenario-ci-review-provenance/v1",
            "kind": "synthetic",
            "generatedAt": resolved_generated_at,
            "sceneId": "smoke-route-policy-ci",
            "assetSource": "scripts/smoke_route_policy_scenario_ci.py",
            "extra": {
                "sampleBundle": "true",
                "runTrigger": "manual",
            },
        }
        review = route_policy_scenario_ci_review_from_dict(payload)
        write_route_policy_scenario_ci_review_bundle(tmp_bundle_dir, review)
        _copy_smoke_artifacts(smoke_root, tmp_bundle_dir / SAMPLE_ARTIFACT_DIR)
        _rewrite_text_artifacts(tmp_bundle_dir / SAMPLE_ARTIFACT_DIR, smoke_root, pages_base_url=pages_base_url)

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    tmp_bundle_dir.rename(bundle_dir)
    _write_reviews_index(reviews_dir)
    return bundle_dir


def _resolve_generated_at_pin() -> str:
    """Return a stable ISO 8601 timestamp for the synthetic sample.

    The sample bundle is a Pages-committed artifact rather than a live run,
    so it pins ``generatedAt`` to a fixed string instead of ``datetime.now``.
    Without the pin, every regeneration would dirty the committed review.json
    even when nothing else changed. Override via ``--generated-at`` when an
    intentional refresh is needed.
    """

    return "2026-05-15T00:00:00+00:00"


def _load_smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_route_policy_scenario_ci", SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SMOKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_smoke_paths(payload: Any, smoke_root: Path, *, pages_base_url: str) -> Any:
    root = smoke_root.as_posix()

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            rewritten = value.replace(root, "artifacts")
            rewritten = rewritten.replace("artifacts/.github/workflows", "artifacts/workflows")
            rewritten = rewritten.replace("artifacts", SAMPLE_ARTIFACT_DIR)
            return rewritten.replace("https://example.test/reviews/smoke-route-policy-ci/", pages_base_url)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {str(key): rewrite(item) for key, item in value.items()}
        return value

    return rewrite(payload)


def _copy_smoke_artifacts(smoke_root: Path, artifact_dir: Path) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"pages"}
    for child in sorted(smoke_root.iterdir()):
        if child.name in skip_names:
            continue
        if child.name == ".github":
            workflows = child / "workflows"
            if workflows.is_dir():
                shutil.copytree(workflows, artifact_dir / "workflows")
            continue
        destination = artifact_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _rewrite_text_artifacts(artifact_dir: Path, smoke_root: Path, *, pages_base_url: str) -> None:
    root = smoke_root.as_posix()
    for path in artifact_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(root, SAMPLE_ARTIFACT_DIR)
        text = text.replace(f"{SAMPLE_ARTIFACT_DIR}/.github/workflows", f"{SAMPLE_ARTIFACT_DIR}/workflows")
        text = text.replace("https://example.test/reviews/smoke-route-policy-ci/", pages_base_url)
        path.write_text(text, encoding="utf-8")


def _write_reviews_index(reviews_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_pages_reviews_index",
        REPO / "scripts" / "build_pages_reviews_index.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/build_pages_reviews_index.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.write_reviews_index(
        reviews_dir,
        html_output=reviews_dir / "index.html",
        json_output=reviews_dir / "index.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DOCS)
    parser.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)
    parser.add_argument("--sample-notice", default=DEFAULT_SAMPLE_NOTICE)
    parser.add_argument("--pages-base-url", default=DEFAULT_PAGES_BASE_URL)
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Override the pinned synthetic generatedAt timestamp (default: pinned for stable diffs)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle_dir = build_sample_review_bundle(
        args.docs_dir,
        bundle_id=args.bundle_id,
        sample_notice=args.sample_notice,
        pages_base_url=args.pages_base_url,
        generated_at=args.generated_at,
    )
    print(f"wrote {bundle_dir}")
    print(f"wrote {bundle_dir.parent / 'index.html'}")
    print(f"wrote {bundle_dir.parent / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
