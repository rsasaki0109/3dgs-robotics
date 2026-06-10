#!/usr/bin/env python3
"""Generate the GitHub Pages index of route policy scenario CI reviews.

Scans a directory for sub-directories that hold a ``review.json`` produced by
``write_route_policy_scenario_ci_review_bundle`` (see
``src/gs_sim2real/sim/policy_scenario_ci_review.py``), then writes a single
``index.html`` plus a structured ``index.json`` alongside them so Pages
visitors can discover every published review bundle without knowing the
per-bundle URL in advance.

The generator is deliberately read-only toward each per-bundle directory:
it never rewrites the bundle files. Missing ``review.json`` sub-directories
are silently skipped; an empty reviews directory produces an empty index.

Usage
-----

::

    PYTHONPATH=src python3 scripts/build_pages_reviews_index.py \\
        --reviews-dir docs/reviews \\
        --html-output docs/reviews/index.html \\
        --json-output docs/reviews/index.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "gs-mapper-route-policy-scenario-ci-reviews-index/v1"


_KNOWN_KINDS: frozenset[str] = frozenset({"production", "synthetic"})


@dataclass(frozen=True, slots=True)
class ReviewIndexEntry:
    """One row on the Pages reviews index page."""

    review_id: str
    bundle_dir: str
    bundle_html: str
    passed: bool
    shard_count: int
    scenario_count: int
    report_count: int
    adoption_trigger_mode: str | None
    adoption_adopted: bool | None
    kind: str = "unknown"
    generated_at: str | None = None
    scene_id: str | None = None
    multi_agent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewId": self.review_id,
            "bundleDir": self.bundle_dir,
            "bundleHtml": self.bundle_html,
            "passed": bool(self.passed),
            "shardCount": int(self.shard_count),
            "scenarioCount": int(self.scenario_count),
            "reportCount": int(self.report_count),
            "adoptionTriggerMode": self.adoption_trigger_mode,
            "adoptionAdopted": self.adoption_adopted,
            "kind": self.kind,
            "generatedAt": self.generated_at,
            "sceneId": self.scene_id,
            "multiAgent": bool(self.multi_agent),
        }


def collect_review_entries(reviews_dir: Path) -> list[ReviewIndexEntry]:
    """Return a sorted list of ``ReviewIndexEntry`` for ``reviews_dir``.

    Directories under ``reviews_dir`` that do not contain a ``review.json``
    are silently skipped (e.g. tooling caches, per-bundle asset dirs).
    Entries are sorted by ``review_id`` for deterministic output.
    """

    if not reviews_dir.is_dir():
        return []
    entries: list[ReviewIndexEntry] = []
    for child in sorted(reviews_dir.iterdir()):
        if not child.is_dir():
            continue
        review_json = child / "review.json"
        if not review_json.is_file():
            continue
        payload = json.loads(review_json.read_text(encoding="utf-8"))
        if payload.get("recordType") != "route-policy-scenario-ci-review":
            continue
        adoption = payload.get("adoption") or {}
        provenance = payload.get("provenance") or {}
        metadata = payload.get("metadata") or {}
        kind = _resolve_entry_kind(provenance, metadata)
        bundle_html_path = child / "index.html"
        entries.append(
            ReviewIndexEntry(
                review_id=str(payload.get("reviewId", child.name)),
                bundle_dir=child.name,
                bundle_html=f"{child.name}/index.html" if bundle_html_path.is_file() else child.name,
                passed=bool(payload.get("passed", False)),
                shard_count=int(payload.get("shardCount", 0)),
                scenario_count=int(payload.get("scenarioCount", 0)),
                report_count=int(payload.get("reportCount", 0)),
                adoption_trigger_mode=_optional_str(adoption.get("triggerMode")) if adoption else None,
                adoption_adopted=None if not adoption else bool(adoption.get("adopted", False)),
                kind=kind,
                generated_at=_optional_str(provenance.get("generatedAt")) if provenance else None,
                scene_id=_optional_str(provenance.get("sceneId")) if provenance else None,
                multi_agent=bool(payload.get("multiAgent", False)),
            )
        )
    entries.sort(key=lambda entry: entry.review_id)
    return entries


def render_reviews_index_json(entries: list[ReviewIndexEntry]) -> str:
    """Render the reviews index as stable JSON."""

    payload = {
        "recordType": SCHEMA_VERSION,
        "reviewCount": len(entries),
        "passCount": sum(1 for entry in entries if entry.passed),
        "failCount": sum(1 for entry in entries if not entry.passed),
        "productionCount": sum(1 for entry in entries if entry.kind == "production"),
        "syntheticCount": sum(1 for entry in entries if entry.kind == "synthetic"),
        "unknownCount": sum(1 for entry in entries if entry.kind == "unknown"),
        "multiAgentCount": sum(1 for entry in entries if entry.multi_agent),
        "entries": [entry.to_dict() for entry in entries],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_reviews_index_html(entries: list[ReviewIndexEntry]) -> str:
    """Render the reviews index as a self-contained HTML page."""

    if not entries:
        rows = (
            '<tr><td colspan="9" class="empty">No review bundles published yet. '
            "Run <code>3dgs-robotics route-policy-scenario-ci-review --bundle-dir ...</code> to create one.</td></tr>"
        )
    else:
        rows = "\n".join(_render_entry_row_html(entry) for entry in entries)
    pass_count = sum(1 for entry in entries if entry.passed)
    fail_count = len(entries) - pass_count
    production_count = sum(1 for entry in entries if entry.kind == "production")
    synthetic_count = sum(1 for entry in entries if entry.kind == "synthetic")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Route Policy Scenario CI Reviews</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f8f4; color: #20231f; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; letter-spacing: 0; }}
    .subtitle {{ color: #5b6259; margin: 0 0 24px; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 24px; }}
    .metric {{ background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; padding: 14px; }}
    .metric span {{ display: block; color: #5b6259; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 700; }}
    .pass {{ background: #dcefd8; color: #1e5a2b; }}
    .fail {{ background: #f7d6d2; color: #8a1f16; }}
    .info {{ background: #dfe5f7; color: #27428a; }}
    .production {{ background: #dcefd8; color: #1e5a2b; }}
    .synthetic {{ background: #fff1c1; color: #6a4d00; }}
    .unknown {{ background: #e3e5df; color: #424940; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e9ede5; text-align: left; vertical-align: top; }}
    th {{ background: #eef2ea; font-size: 13px; color: #424940; }}
    tr:last-child td {{ border-bottom: 0; }}
    td.empty {{ color: #5b6259; text-align: center; padding: 24px; }}
    a {{ color: #285b9b; }}
    code {{ background: #eef2ea; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>Route Policy Scenario CI Reviews</h1>
    <p class="subtitle">{len(entries)} published bundle{"s" if len(entries) != 1 else ""}.</p>
    <section class="grid">
      <div class="metric"><span>Total</span><strong>{len(entries)}</strong></div>
      <div class="metric"><span>Passing</span><strong>{pass_count}</strong></div>
      <div class="metric"><span>Failing</span><strong>{fail_count}</strong></div>
      <div class="metric"><span>Production</span><strong>{production_count}</strong></div>
      <div class="metric"><span>Synthetic</span><strong>{synthetic_count}</strong></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Review</th>
          <th>Kind</th>
          <th>Status</th>
          <th>Scene</th>
          <th>Generated</th>
          <th>Shards</th>
          <th>Scenarios</th>
          <th>Reports</th>
          <th>Adoption</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def _render_entry_row_html(entry: ReviewIndexEntry) -> str:
    status_pill = "pass" if entry.passed else "fail"
    status_label = "PASS" if entry.passed else "FAIL"
    adoption_cell = _render_adoption_cell_html(entry)
    kind_cell = f'<span class="pill {entry.kind}">{entry.kind.upper()}</span>'
    scene_cell = f"<code>{escape(entry.scene_id)}</code>" if entry.scene_id else "—"
    generated_cell = _format_generated_at_html(entry.generated_at)
    return (
        "        <tr>"
        f'<td><a href="{escape(entry.bundle_html)}">{escape(entry.review_id)}</a></td>'
        f"<td>{kind_cell}</td>"
        f'<td><span class="pill {status_pill}">{status_label}</span></td>'
        f"<td>{scene_cell}</td>"
        f"<td>{generated_cell}</td>"
        f"<td>{entry.shard_count}</td>"
        f"<td>{entry.scenario_count}</td>"
        f"<td>{entry.report_count}</td>"
        f"<td>{adoption_cell}</td>"
        "</tr>"
    )


def _format_generated_at_html(value: str | None) -> str:
    if not value:
        return "—"
    # Trim to the YYYY-MM-DD prefix when the value looks like ISO 8601 so the
    # column stays narrow; show the full string in a title tooltip.
    display = value[:10] if len(value) >= 10 and value[4] == "-" and value[7] == "-" else value
    return f'<span title="{escape(value)}">{escape(display)}</span>'


def _render_adoption_cell_html(entry: ReviewIndexEntry) -> str:
    if entry.adoption_trigger_mode is None:
        return '<span class="pill info">none</span>'
    adopted_pill = "pass" if entry.adoption_adopted else "fail"
    adopted_label = "ADOPTED" if entry.adoption_adopted else "BLOCKED"
    return (
        f'<span class="pill {adopted_pill}">{adopted_label}</span> <code>{escape(entry.adoption_trigger_mode)}</code>'
    )


def render_reviews_index_markdown(entries: list[ReviewIndexEntry]) -> str:
    """Render the reviews index as Markdown (used by tests + docs previews)."""

    production_count = sum(1 for entry in entries if entry.kind == "production")
    synthetic_count = sum(1 for entry in entries if entry.kind == "synthetic")
    lines = [
        "# Route Policy Scenario CI Reviews",
        f"- Total bundles: {len(entries)}",
        f"- Passing: {sum(1 for entry in entries if entry.passed)}",
        f"- Failing: {sum(1 for entry in entries if not entry.passed)}",
        f"- Production: {production_count}",
        f"- Synthetic: {synthetic_count}",
        "",
        "| Review | Kind | Status | Scene | Generated | Shards | Scenarios | Reports | Adoption |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not entries:
        lines.append("| _(no review bundles published yet)_ | n/a | n/a | n/a | n/a | 0 | 0 | 0 | n/a |")
    for entry in entries:
        status = "PASS" if entry.passed else "FAIL"
        if entry.adoption_trigger_mode is None:
            adoption = "none"
        else:
            adopted_label = "ADOPTED" if entry.adoption_adopted else "BLOCKED"
            adoption = f"{adopted_label} (`{entry.adoption_trigger_mode}`)"
        scene_cell = f"`{entry.scene_id}`" if entry.scene_id else "—"
        generated_cell = (
            entry.generated_at[:10]
            if entry.generated_at and len(entry.generated_at) >= 10
            else (entry.generated_at or "—")
        )
        lines.append(
            f"| [{entry.review_id}]({entry.bundle_html}) | {entry.kind} | {status} | "
            f"{scene_cell} | {generated_cell} | "
            f"{entry.shard_count} | {entry.scenario_count} | {entry.report_count} | {adoption} |"
        )
    return "\n".join(lines) + "\n"


def write_reviews_index(
    reviews_dir: Path,
    *,
    html_output: Path | None,
    json_output: Path | None,
    markdown_output: Path | None = None,
) -> list[ReviewIndexEntry]:
    """Generate the index artifacts for ``reviews_dir`` and return the entries."""

    entries = collect_review_entries(reviews_dir)
    if html_output is not None:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(render_reviews_index_html(entries), encoding="utf-8")
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(render_reviews_index_json(entries), encoding="utf-8")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_reviews_index_markdown(entries), encoding="utf-8")
    return entries


def _resolve_entry_kind(provenance: Any, metadata: Any) -> str:
    """Return ``production`` / ``synthetic`` / ``unknown`` for an index entry.

    First-class ``provenance.kind`` wins. When provenance is absent, fall back
    to the legacy ``metadata.sampleBundle`` hint (used by older synthetic
    bundles before the provenance contract existed). Production has no
    legacy fallback — pre-provenance bundles default to ``unknown`` rather
    than silently masquerading as production.
    """

    if isinstance(provenance, dict):
        kind = provenance.get("kind")
        if isinstance(kind, str) and kind in _KNOWN_KINDS:
            return kind
    if isinstance(metadata, dict) and metadata.get("sampleBundle") is True:
        return "synthetic"
    return "unknown"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        default=Path("docs/reviews"),
        help="Directory containing per-bundle sub-directories (default: docs/reviews)",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Write the reviews index HTML here (default: <reviews-dir>/index.html)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write the reviews index JSON here (default: <reviews-dir>/index.json)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown rendering of the reviews index",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reviews_dir = args.reviews_dir
    html_output = args.html_output or (reviews_dir / "index.html")
    json_output = args.json_output or (reviews_dir / "index.json")
    entries = write_reviews_index(
        reviews_dir,
        html_output=html_output,
        json_output=json_output,
        markdown_output=args.markdown_output,
    )
    print(f"[ok] wrote reviews index with {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} to {html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
