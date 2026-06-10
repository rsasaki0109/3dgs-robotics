"""Generate shareable launch collateral for GS Mapper."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any


DEFAULT_SITE_URL = "https://rsasaki0109.github.io/3dgs-robotics/"
DEFAULT_REPO_URL = "https://github.com/rsasaki0109/3dgs-robotics"
DEFAULT_LIVE_VIEWER_URL = "https://rsasaki0109.github.io/3dgs-robotics/splat.html"
DEFAULT_SPARK_VIEWER_URL = "https://rsasaki0109.github.io/3dgs-robotics/splat_spark.html"
DEFAULT_WEBGPU_VIEWER_URL = "https://rsasaki0109.github.io/3dgs-robotics/splat_webgpu.html"
DEFAULT_PHYSICAL_AI_DOC_URL = "https://github.com/rsasaki0109/3dgs-robotics/blob/main/docs/physical-ai-sim.md"
DEFAULT_HERO_IMAGE = "images/demo-sweep/hero.gif"
DEFAULT_SOCIAL_IMAGE_URL = "https://rsasaki0109.github.io/3dgs-robotics/images/social-card.png"


@dataclass(frozen=True, slots=True)
class LaunchLink:
    """One external link that should be included in launch collateral."""

    label: str
    url: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "url": self.url, "description": self.description}


@dataclass(frozen=True, slots=True)
class LaunchSnippet:
    """Copy block tailored to a distribution channel."""

    key: str
    label: str
    text: str
    max_chars: int | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_within_limit(self) -> bool:
        return self.max_chars is None or self.char_count <= self.max_chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "text": self.text,
            "charCount": self.char_count,
            "maxChars": self.max_chars,
            "withinLimit": self.is_within_limit,
        }


@dataclass(frozen=True, slots=True)
class LaunchDestination:
    """One suggested outreach target and the copy angle to use there."""

    key: str
    label: str
    url: str
    audience: str
    angle: str
    snippet_key: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "label": self.label,
            "url": self.url,
            "audience": self.audience,
            "angle": self.angle,
            "snippetKey": self.snippet_key,
        }


@dataclass(frozen=True, slots=True)
class LaunchKit:
    """Complete launch collateral bundle."""

    project: str
    tagline: str
    description: str
    site_url: str
    repo_url: str
    hero_image: str
    social_image_url: str
    links: tuple[LaunchLink, ...]
    snippets: tuple[LaunchSnippet, ...]
    destinations: tuple[LaunchDestination, ...]
    topics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "tagline": self.tagline,
            "description": self.description,
            "siteUrl": self.site_url,
            "repoUrl": self.repo_url,
            "heroImage": self.hero_image,
            "socialImageUrl": self.social_image_url,
            "links": [link.to_dict() for link in self.links],
            "snippets": [snippet.to_dict() for snippet in self.snippets],
            "destinations": [destination.to_dict() for destination in self.destinations],
            "topics": list(self.topics),
        }


def build_default_launch_kit() -> LaunchKit:
    """Return the canonical GS Mapper launch kit."""

    tagline = "Real robot data to Gaussian-splat Physical AI benchmarks."
    description = (
        "GS Mapper turns photos, robotics logs, and MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR "
        "artifacts into browser-viewable Gaussian splats, scene contracts, route-policy benchmarks, "
        "and CI review bundles for Physical AI evaluation."
    )
    links = (
        LaunchLink(
            "Project page", DEFAULT_SITE_URL, "First-stop page with the GS Mapper pitch and viewer entry points."
        ),
        LaunchLink(
            "Live splat viewer", DEFAULT_LIVE_VIEWER_URL, "WebGL scene picker with nine bundled comparison splats."
        ),
        LaunchLink(
            "Spark mobile / VR viewer", DEFAULT_SPARK_VIEWER_URL, "Spark viewer for mobile and WebXR-capable devices."
        ),
        LaunchLink(
            "WebGPU viewer", DEFAULT_WEBGPU_VIEWER_URL, "GPU-sort viewer for Chrome, Edge, and WebGPU-enabled browsers."
        ),
        LaunchLink(
            "Physical AI simulation contract",
            DEFAULT_PHYSICAL_AI_DOC_URL,
            "Scene contract, sensor noise profiles, dynamic obstacles, and scenario CI reference.",
        ),
        LaunchLink(
            "GitHub repository",
            DEFAULT_REPO_URL,
            "Source, Physical AI benchmark docs, external SLAM import docs, and tests.",
        ),
    )
    snippets = (
        LaunchSnippet(
            key="short-social",
            label="Short social post",
            max_chars=280,
            text=(
                "GS Mapper turns robot logs and MASt3R-SLAM / VGGT-SLAM outputs into browser-viewable "
                "3D Gaussian Splats plus CI-friendly Physical AI policy benchmarks.\n\n"
                f"Live demos: {DEFAULT_LIVE_VIEWER_URL}\n"
                "#3DGS #Robotics #PhysicalAI"
            ),
        ),
        LaunchSnippet(
            key="technical-social",
            label="Technical social post",
            text=(
                "I released GS Mapper: an open-source pipeline for turning real outdoor robot data, "
                "DUSt3R / MASt3R pose-free reconstructions, and MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / "
                "LoGeR artifacts into Gaussian-splat scenes and Physical AI policy benchmark artifacts.\n\n"
                "It ships nine public comparison scenes, external SLAM dry-run manifests, route-policy "
                "benchmarks with pose and raw camera / depth / LiDAR noise profiles, dynamic-obstacle "
                "timelines with multi-agent observation features, scenario matrix expansion, CI "
                "sharding, generated workflow validation, activation guards, and review bundles.\n\n"
                f"{DEFAULT_REPO_URL}"
            ),
        ),
        LaunchSnippet(
            key="community-post",
            label="Community post",
            text=(
                "GS Mapper is an open-source pipeline for turning real robot data into Gaussian-splat "
                "Physical AI evaluation artifacts. It accepts image folders, robotics logs, and external "
                "SLAM artifacts, then produces browser-viewable .splat files, scene contracts, policy "
                "benchmark reports, scenario shards, and review bundles.\n\n"
                "The current demo set compares supervised GNSS + LiDAR, DUSt3R, MASt3R, VGGT-SLAM 2.0, "
                "MASt3R-SLAM, and Pi3X outputs on outdoor robotics scenes. The benchmark stack keeps dataset, "
                "policy registry, scenario matrix, CI manifest, workflow validation, activation, and "
                "review publishing as separate testable artifacts, plus partial-information knobs "
                "(pose / goal / heading noise, raw camera / depth / LiDAR noise, and dynamic "
                f"obstacles with nearest + second-nearest features) for reactive-policy studies.\n\n"
                f"Live demo: {DEFAULT_LIVE_VIEWER_URL}\n"
                f"Repo: {DEFAULT_REPO_URL}"
            ),
        ),
        LaunchSnippet(
            key="awesome-list",
            label="Awesome-list entry",
            text=(
                "- [GS Mapper](https://github.com/rsasaki0109/3dgs-robotics) - Converts photos, robotics logs, "
                "and MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR artifacts into browser-viewable Gaussian splats, "
                "Physical AI route-policy benchmarks, and CI review bundles."
            ),
        ),
        LaunchSnippet(
            key="japanese",
            label="Japanese announcement",
            text=(
                "GS Mapper を公開しました。写真フォルダ、ロボティクスログ、MASt3R-SLAM / VGGT-SLAM 2.0 / "
                "Pi3 / LoGeR の出力を、Web で見られる 3D Gaussian Splatting、Physical AI policy "
                f"benchmark、scenario CI review bundle につなぐ OSS です。\n\nLive demo: {DEFAULT_LIVE_VIEWER_URL}\n"
                f"GitHub: {DEFAULT_REPO_URL}"
            ),
        ),
    )
    destinations = (
        LaunchDestination(
            key="x-twitter",
            label="X / Twitter",
            url="https://x.com/intent/tweet",
            audience="3DGS, SLAM, robotics, and Physical AI builders who want a fast demo link.",
            angle="Lead with the live viewer, then mention policy benchmarks and CI artifacts.",
            snippet_key="short-social",
        ),
        LaunchDestination(
            key="hacker-news",
            label="Hacker News Show HN",
            url="https://news.ycombinator.com/submit",
            audience="Graphics, mapping, robotics, simulation, and developer-tool readers who inspect repos.",
            angle="Frame it as an open-source bridge from real robot data to reproducible evaluation artifacts.",
            snippet_key="community-post",
        ),
        LaunchDestination(
            key="linkedin",
            label="LinkedIn",
            url="https://www.linkedin.com/feed/",
            audience="Robotics, autonomy, geospatial, and simulation engineers.",
            angle="Emphasize policy regression, reviewable simulation artifacts, and SLAM-to-benchmark handoff.",
            snippet_key="technical-social",
        ),
        LaunchDestination(
            key="reddit-communities",
            label="Reddit communities",
            url="https://www.reddit.com/search/?q=3D%20Gaussian%20Splatting%20SLAM&type=communities",
            audience="Subreddits around Gaussian Splatting, photogrammetry, 3D scanning, robotics, and ML systems.",
            angle="Pick one relevant community, check its rules, and post the community copy with screenshots.",
            snippet_key="community-post",
        ),
        LaunchDestination(
            key="github-awesome-lists",
            label="GitHub awesome lists",
            url="https://github.com/search?q=awesome+3d+gaussian+splatting&type=repositories",
            audience="Maintainers of curated 3DGS, SLAM, robotics, simulation, and Physical AI resource lists.",
            angle="Open a small PR with the awesome-list entry and link directly to the live demo and README.",
            snippet_key="awesome-list",
        ),
        LaunchDestination(
            key="japanese-robotics",
            label="Japanese robotics channels",
            url=DEFAULT_SITE_URL,
            audience="Japanese robotics, mapping, autonomy, and computer-vision builders.",
            angle="Use the Japanese announcement and point people to the live splat viewer and benchmark docs.",
            snippet_key="japanese",
        ),
    )
    topics = (
        "3d-gaussian-splatting",
        "3dgs",
        "slam",
        "visual-slam",
        "robotics",
        "physical-ai",
        "simulation",
        "benchmark",
        "autonomous-driving",
        "mast3r",
        "dust3r",
        "vggt-slam",
        "gsplat",
        "scenario-ci",
        "route-policy-benchmark",
        "webgl",
        "webgpu",
    )
    return LaunchKit(
        project="GS Mapper",
        tagline=tagline,
        description=description,
        site_url=DEFAULT_SITE_URL,
        repo_url=DEFAULT_REPO_URL,
        hero_image=DEFAULT_HERO_IMAGE,
        social_image_url=DEFAULT_SOCIAL_IMAGE_URL,
        links=links,
        snippets=snippets,
        destinations=destinations,
        topics=topics,
    )


def render_launch_kit_json(kit: LaunchKit) -> str:
    """Render launch collateral as stable JSON."""

    return json.dumps(kit.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def render_launch_kit_markdown(kit: LaunchKit) -> str:
    """Render launch collateral as a Markdown page."""

    lines = [
        f"# {kit.project} Launch Kit",
        "",
        kit.tagline,
        "",
        kit.description,
        "",
        "## Links",
        "",
    ]
    for link in kit.links:
        lines.append(f"- [{link.label}]({link.url}) - {link.description}")
    lines.extend(["", "## Where To Post", ""])
    for destination in kit.destinations:
        lines.extend(
            [
                f"### {destination.label}",
                "",
                f"- URL: {destination.url}",
                f"- Audience: {destination.audience}",
                f"- Angle: {destination.angle}",
                f"- Copy block: `{destination.snippet_key}`",
                "",
            ]
        )
    lines.extend(["", "## Copy Blocks", ""])
    for snippet in kit.snippets:
        limit = f" ({snippet.char_count}/{snippet.max_chars} chars)" if snippet.max_chars else ""
        lines.extend([f"### {snippet.label}{limit}", "", "```text", snippet.text, "```", ""])
    lines.extend(["## Topics", "", ", ".join(f"`{topic}`" for topic in kit.topics), ""])
    return "\n".join(lines)


def render_launch_kit_html(kit: LaunchKit) -> str:
    """Render launch collateral as a standalone static HTML page."""

    link_items = "\n".join(
        '      <a class="link-row" href="{url}"><strong>{label}</strong><span>{description}</span></a>'.format(
            url=html.escape(link.url, quote=True),
            label=html.escape(link.label),
            description=html.escape(link.description),
        )
        for link in kit.links
    )
    destination_blocks = "\n".join(_render_destination_html(destination) for destination in kit.destinations)
    snippet_blocks = "\n".join(_render_snippet_html(snippet) for snippet in kit.snippets)
    topics = "\n".join(f"      <span>{html.escape(topic)}</span>" for topic in kit.topics)
    title = f"{kit.project} Launch Kit"
    description = kit.description
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description, quote=True)}">
<link rel="canonical" href="{html.escape(kit.site_url, quote=True)}launch-kit.html">
<meta property="og:title" content="{html.escape(title, quote=True)}">
<meta property="og:description" content="{html.escape(description, quote=True)}">
<meta property="og:type" content="website">
<meta property="og:url" content="{html.escape(kit.site_url, quote=True)}launch-kit.html">
<meta property="og:image" content="{html.escape(kit.social_image_url, quote=True)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{html.escape(title, quote=True)}">
<meta name="twitter:description" content="{html.escape(description, quote=True)}">
<meta name="twitter:image" content="{html.escape(kit.social_image_url, quote=True)}">
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #e6edf3; background: #0d1117; line-height: 1.55; }}
a {{ color: inherit; }}
.hero {{ position: relative; min-height: 520px; display: flex; align-items: center; overflow: hidden; padding: 5rem 1.5rem 4rem; border-bottom: 1px solid #30363d; }}
.hero img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.4; }}
.hero::after {{ content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, rgba(13,17,23,0.96), rgba(13,17,23,0.72)); }}
.hero-inner {{ position: relative; z-index: 1; width: min(1080px, 100%); margin: 0 auto; }}
.kicker {{ margin-bottom: 1rem; color: #3fb950; font: 700 0.82rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; text-transform: uppercase; letter-spacing: 0; }}
h1 {{ margin: 0 0 1rem; font-size: clamp(2.6rem, 7vw, 5.6rem); line-height: 0.95; }}
.hero p {{ max-width: 760px; margin: 0; color: rgba(230,237,243,0.88); font-size: 1.12rem; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 2rem; }}
.actions a {{ text-decoration: none; min-height: 42px; display: inline-flex; align-items: center; padding: 0.55rem 0.9rem; border: 1px solid rgba(230,237,243,0.24); border-radius: 6px; background: rgba(13,17,23,0.55); }}
.actions a:first-child {{ background: #3fb950; border-color: #3fb950; color: #06120a; font-weight: 800; }}
main {{ width: min(1180px, 100%); margin: 0 auto; padding: 3.5rem 1.5rem 4rem; }}
.section-head {{ max-width: 760px; margin-bottom: 1.5rem; }}
.section-head h2 {{ margin: 0 0 0.35rem; font-size: 1.7rem; }}
.section-head p {{ margin: 0; color: #8b949e; }}
.link-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.85rem; margin-bottom: 3rem; }}
.link-row {{ display: flex; flex-direction: column; gap: 0.3rem; min-height: 116px; padding: 1rem; border: 1px solid #30363d; border-radius: 8px; background: #161b22; text-decoration: none; }}
.link-row span {{ color: #8b949e; font-size: 0.92rem; }}
.destination-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 1rem; margin-bottom: 3rem; }}
.destination {{ display: flex; flex-direction: column; gap: 0.65rem; min-height: 230px; padding: 1rem; border: 1px solid #30363d; border-radius: 8px; background: #161b22; }}
.destination a {{ font-weight: 800; text-decoration: none; color: #f0f6fc; }}
.destination p {{ margin: 0; color: #8b949e; font-size: 0.92rem; }}
.destination code {{ width: fit-content; padding: 0.18rem 0.4rem; color: #3fb950; border: 1px solid rgba(63,185,80,0.35); border-radius: 4px; background: rgba(63,185,80,0.08); }}
.snippet-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 1rem; }}
.snippet {{ border: 1px solid #30363d; border-radius: 8px; background: #161b22; overflow: hidden; }}
.snippet-header {{ display: flex; justify-content: space-between; gap: 1rem; align-items: center; padding: 0.85rem 1rem; border-bottom: 1px solid #30363d; }}
.snippet-header h3 {{ margin: 0; font-size: 1rem; }}
.count {{ color: #8b949e; font: 0.78rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; }}
textarea {{ width: 100%; min-height: 190px; display: block; border: 0; resize: vertical; padding: 1rem; color: #e6edf3; background: transparent; font: 0.9rem/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; }}
button {{ margin: 0 1rem 1rem; min-height: 36px; padding: 0.45rem 0.75rem; color: #e6edf3; background: transparent; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; }}
button:hover {{ border-color: #58a6ff; }}
.topics {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 2rem; }}
.topics span {{ padding: 0.25rem 0.55rem; color: #58a6ff; border: 1px solid rgba(88,166,255,0.35); border-radius: 999px; background: rgba(88,166,255,0.08); font-size: 0.86rem; }}
footer {{ padding: 2rem 1.5rem; border-top: 1px solid #30363d; color: #8b949e; text-align: center; }}
@media (max-width: 720px) {{ .hero {{ min-height: auto; padding: 3rem 1.1rem 2rem; }} .actions {{ flex-direction: column; }} .actions a {{ justify-content: center; }} main {{ padding: 2.5rem 1rem 3rem; }} }}
</style>
</head>
<body>
<header class="hero">
  <img src="{html.escape(kit.hero_image, quote=True)}" alt="{html.escape(kit.project, quote=True)} demo scenes">
  <div class="hero-inner">
    <div class="kicker">Launch kit</div>
    <h1>{html.escape(kit.project)}</h1>
    <p>{html.escape(kit.description)}</p>
    <div class="actions">
      <a href="{html.escape(DEFAULT_LIVE_VIEWER_URL, quote=True)}">Open live viewer</a>
      <a href="{html.escape(kit.repo_url, quote=True)}">GitHub repository</a>
      <a href="{html.escape(kit.site_url, quote=True)}">Project page</a>
    </div>
  </div>
</header>
<main>
  <section>
    <div class="section-head">
      <h2>Link Pack</h2>
      <p>{html.escape(kit.tagline)}</p>
    </div>
    <div class="link-grid">
{link_items}
    </div>
  </section>
  <section>
    <div class="section-head">
      <h2>Where To Post</h2>
      <p>Suggested launch targets with the audience, angle, and copy block to start from.</p>
    </div>
    <div class="destination-grid">
{destination_blocks}
    </div>
  </section>
  <section>
    <div class="section-head">
      <h2>Copy Blocks</h2>
      <p>Channel-specific copy for launch posts, community threads, and awesome-list pull requests.</p>
    </div>
    <div class="snippet-grid">
{snippet_blocks}
    </div>
    <div class="topics">
{topics}
    </div>
  </section>
</main>
<footer>{html.escape(kit.project)} launch kit generated from the repository source.</footer>
<script>
for (const button of document.querySelectorAll("[data-copy-target]")) {{
  button.addEventListener("click", async () => {{
    const target = document.getElementById(button.dataset.copyTarget);
    await navigator.clipboard.writeText(target.value);
    button.textContent = "Copied";
    setTimeout(() => {{ button.textContent = "Copy"; }}, 1400);
  }});
}}
</script>
</body>
</html>
"""


def _render_destination_html(destination: LaunchDestination) -> str:
    return """      <article class="destination">
        <a href="{url}">{label}</a>
        <p><strong>Audience:</strong> {audience}</p>
        <p><strong>Angle:</strong> {angle}</p>
        <code>{snippet_key}</code>
      </article>""".format(
        url=html.escape(destination.url, quote=True),
        label=html.escape(destination.label),
        audience=html.escape(destination.audience),
        angle=html.escape(destination.angle),
        snippet_key=html.escape(destination.snippet_key),
    )


def _render_snippet_html(snippet: LaunchSnippet) -> str:
    limit = f"{snippet.char_count}/{snippet.max_chars}" if snippet.max_chars else f"{snippet.char_count} chars"
    textarea_id = f"snippet-{snippet.key}"
    return """      <article class="snippet">
        <div class="snippet-header">
          <h3>{label}</h3>
          <span class="count">{limit}</span>
        </div>
        <textarea id="{textarea_id}" readonly>{text}</textarea>
        <button type="button" data-copy-target="{textarea_id}" aria-label="Copy {label}">Copy</button>
      </article>""".format(
        label=html.escape(snippet.label),
        limit=html.escape(limit),
        textarea_id=html.escape(textarea_id, quote=True),
        text=html.escape(snippet.text),
    )
