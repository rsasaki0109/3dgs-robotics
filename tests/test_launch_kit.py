"""Tests for generated public launch collateral."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from gs_sim2real.marketing.launch_kit import (
    build_default_launch_kit,
    render_launch_kit_html,
    render_launch_kit_json,
    render_launch_kit_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_launch_kit_has_core_links_and_bounded_social_post() -> None:
    kit = build_default_launch_kit()

    link_urls = {link.url for link in kit.links}
    assert "https://rsasaki0109.github.io/3dgs-robotics/splat.html" in link_urls
    assert "https://github.com/rsasaki0109/3dgs-robotics" in link_urls
    assert "vggt-slam" in kit.topics

    snippet_keys = {snippet.key for snippet in kit.snippets}
    assert {destination.snippet_key for destination in kit.destinations} <= snippet_keys
    assert "hacker-news" in {destination.key for destination in kit.destinations}

    short = next(snippet for snippet in kit.snippets if snippet.key == "short-social")
    assert short.max_chars == 280
    assert short.is_within_limit
    assert "MASt3R-SLAM" in short.text
    assert "Live demos:" in short.text


def test_launch_kit_renderers_include_copy_blocks_and_metadata() -> None:
    kit = build_default_launch_kit()
    html = render_launch_kit_html(kit)
    markdown = render_launch_kit_markdown(kit)
    payload = json.loads(render_launch_kit_json(kit))

    assert '<meta property="og:image"' in html
    assert "Where To Post" in html
    assert "data-copy-target" in html
    assert "snippet-short-social" in html
    assert "3DGS Robotics Launch Kit" in markdown
    assert "## Where To Post" in markdown
    assert "```text" in markdown
    assert payload["project"] == "3DGS Robotics"
    assert payload["destinations"][0]["snippetKey"] == "short-social"
    assert payload["snippets"][0]["withinLimit"] is True


def test_generate_launch_kit_script_writes_all_formats(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_launch_kit.py",
            "--format",
            "all",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout == ""
    assert (tmp_path / "launch-kit.html").is_file()
    assert (tmp_path / "launch-kit.md").is_file()
    payload = json.loads((tmp_path / "launch-kit.json").read_text(encoding="utf-8"))
    assert payload["siteUrl"] == "https://rsasaki0109.github.io/3dgs-robotics/"


def test_checked_in_launch_kit_files_match_generator(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_launch_kit.py",
            "--format",
            "all",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    for name in ("launch-kit.html", "launch-kit.md", "launch-kit.json"):
        expected = (tmp_path / name).read_text(encoding="utf-8")
        actual = (REPO_ROOT / "docs" / name).read_text(encoding="utf-8")
        assert actual == expected
