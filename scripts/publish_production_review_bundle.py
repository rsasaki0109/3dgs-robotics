#!/usr/bin/env python3
"""Publish an externally generated production review bundle to docs/reviews/.

The sample at ``docs/reviews/smoke-route-policy-ci/`` is built from
``scripts/build_pages_sample_review_bundle.py``, which internally runs the
synthetic smoke chain. There is no equivalent for *production* runs because
the inputs (real bag data, real policy registry, real benchmark output)
live outside the repo. When an operator has finished a production
benchmark and produced a ``review.json`` via ``route-policy-scenario-ci-review``
(typically with ``--bundle-dir runs/...``), this script:

1. Validates that ``provenance.kind == "production"`` so a synthetic bundle
   cannot be silently mis-labeled.
2. Re-emits the bundle (``review.json`` / ``review.md`` / ``index.html``)
   under ``docs/reviews/<bundle-id>/`` via
   ``write_route_policy_scenario_ci_review_bundle`` so the on-disk layout
   matches what ``build_pages_reviews_index.py`` expects.
3. Regenerates ``docs/reviews/index.{html,json}`` so the new bundle shows
   up on the Pages reviews landing without a separate manual step.

The script never invokes the benchmark itself — that runs out-of-tree
on real production assets. Bundling and publishing are kept as a thin,
testable wrapper so the operator step is a single command:

::

    PYTHONPATH=src python3 scripts/publish_production_review_bundle.py \\
        --review-json runs/outdoor-demo/ci-review.json \\
        --bundle-id outdoor-demo-direct-baseline-001
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SRC = REPO / "src"
_BUNDLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class PublishError(RuntimeError):
    """Raised when the bundle cannot be published as production."""


def publish_production_review_bundle(
    *,
    review_json: Path,
    bundle_id: str,
    docs_dir: Path = DOCS,
) -> Path:
    """Publish ``review_json`` as ``docs/reviews/<bundle-id>/`` and refresh index."""

    if not _BUNDLE_ID_RE.match(bundle_id):
        raise PublishError(
            f"bundle id {bundle_id!r} must be lowercase kebab-case (a-z, 0-9, '-'; 1-64 chars; starting with a-z/0-9)"
        )
    if not review_json.is_file():
        raise PublishError(f"review JSON not found: {review_json}")

    from gs_sim2real.sim import (
        route_policy_scenario_ci_review_from_dict,
        write_route_policy_scenario_ci_review_bundle,
    )

    payload = json.loads(review_json.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    kind = str(provenance.get("kind", "")).lower()
    if kind != "production":
        raise PublishError(
            f"refusing to publish: provenance.kind={kind!r} (must be 'production'). "
            "Regenerate the review with `route-policy-scenario-ci-review --kind production ...` "
            "or correct the JSON before publishing."
        )

    review = route_policy_scenario_ci_review_from_dict(payload)
    reviews_dir = docs_dir / "reviews"
    bundle_dir = reviews_dir / bundle_id
    tmp_bundle_dir = reviews_dir / f".{bundle_id}.tmp"
    if tmp_bundle_dir.exists():
        shutil.rmtree(tmp_bundle_dir)
    reviews_dir.mkdir(parents=True, exist_ok=True)
    write_route_policy_scenario_ci_review_bundle(tmp_bundle_dir, review)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    tmp_bundle_dir.rename(bundle_dir)
    _refresh_reviews_index(reviews_dir)
    return bundle_dir


def _refresh_reviews_index(reviews_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_pages_reviews_index",
        REPO / "scripts" / "build_pages_reviews_index.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/build_pages_reviews_index.py")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.write_reviews_index(
        reviews_dir,
        html_output=reviews_dir / "index.html",
        json_output=reviews_dir / "index.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-json",
        type=Path,
        required=True,
        help="Path to a production review.json produced by route-policy-scenario-ci-review",
    )
    parser.add_argument(
        "--bundle-id",
        required=True,
        help="Directory id under docs/reviews/ (lowercase kebab-case)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS,
        help="Docs directory containing reviews/ (default: docs/)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        bundle_dir = publish_production_review_bundle(
            review_json=args.review_json,
            bundle_id=args.bundle_id,
            docs_dir=args.docs_dir,
        )
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Published production review bundle to: {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
