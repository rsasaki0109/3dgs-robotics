#!/usr/bin/env python3
"""Generate static outreach collateral for 3DGS Robotics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.marketing.launch_kit import (  # noqa: E402
    build_default_launch_kit,
    render_launch_kit_html,
    render_launch_kit_json,
    render_launch_kit_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=["all", "html", "markdown", "json"],
        default="all",
        help="Output format to generate",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory for generated launch-kit files when --format=all",
    )
    parser.add_argument(
        "--output",
        help="Single output path for --format html/markdown/json. Writes to stdout when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    kit = build_default_launch_kit()

    if args.format == "all":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write(output_dir / "launch-kit.html", render_launch_kit_html(kit))
        _write(output_dir / "launch-kit.md", render_launch_kit_markdown(kit))
        _write(output_dir / "launch-kit.json", render_launch_kit_json(kit))
        return

    if args.format == "html":
        rendered = render_launch_kit_html(kit)
    elif args.format == "markdown":
        rendered = render_launch_kit_markdown(kit)
    else:
        rendered = render_launch_kit_json(kit)

    if args.output:
        _write(Path(args.output), rendered)
    else:
        print(rendered, end="")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
