#!/usr/bin/env python3
"""Gate real large-scale 3DGS inputs before bootstrap/run/promote."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gs_sim2real.train.large_scale_3dgs import (  # noqa: E402
    LargeScale3DGSDiscoveryOptions,
    build_large_scale_3dgs_discovery,
    parse_large_scale_3dgs_tile_sizes,
)


FIXTURE_HINTS = ("smoke", "fixture", "sample", "demo")
OK_STATUSES = {"ready-colmap", "needs-preprocess"}


def _fmt_command(parts: list[str | Path | int | float]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def _slugify(value: str, fallback: str = "large-scale-3dgs") -> str:
    slug = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
            previous_dash = False
        elif not previous_dash:
            slug.append("-")
            previous_dash = True
    normalized = "".join(slug).strip("-")
    return normalized or fallback


def _path_has_fixture_hint(path_value: str | Path) -> bool:
    parts = [part.lower() for part in Path(path_value).parts]
    return any(any(hint in part for hint in FIXTURE_HINTS) for part in parts)


def _axis_extent(scene: dict[str, Any]) -> float:
    spans = scene.get("worldSpan")
    if not isinstance(spans, dict) or not spans:
        return 0.0
    return max(float(value) for value in spans.values())


def _reject_reason_for_scene(
    scene: dict[str, Any],
    *,
    min_images: int,
    min_points: int,
    min_extent_m: float,
    allow_fixtures: bool,
) -> str:
    if scene.get("status") != "ready":
        return str(scene.get("status") or "not-ready")
    if not allow_fixtures and _path_has_fixture_hint(str(scene.get("dataDir") or "")):
        return "fixture-path"
    if int(scene.get("registeredImageCount") or 0) < min_images:
        return "too-few-images"
    if int(scene.get("points3DCount") or 0) < min_points:
        return "too-few-points"
    if _axis_extent(scene) < min_extent_m:
        return "too-small-extent"
    return ""


def _reject_reason_for_bag(
    bag_input: dict[str, Any],
    *,
    min_bag_bytes: int,
    allow_fixtures: bool,
) -> str:
    source = str(bag_input.get("source") or bag_input.get("path") or "")
    if not allow_fixtures and _path_has_fixture_hint(source):
        return "fixture-path"
    if int(bag_input.get("bytes") or 0) < min_bag_bytes:
        return "too-small-bag"
    return ""


def _default_output_dir(root_dir: Path) -> Path:
    return Path("outputs") / f"{_slugify(root_dir.name, 'real-large-3dgs')}_large"


def _default_scene_id(root_dir: Path) -> str:
    return _slugify(root_dir.name, "real-large-3dgs")


def _build_commands(
    *,
    root_dir: Path,
    output_dir: Path,
    public_root: Path,
    scene_id: str,
    label: str,
    axes: str,
    tile_sizes: str,
    pilot_chunks: int,
    route_start_image: int,
    recommendation: dict[str, Any],
    status: str,
) -> dict[str, str]:
    bootstrap = _fmt_command(
        [
            "gs-mapper",
            "large-scale-3dgs-bootstrap",
            "--root",
            root_dir,
            "--output",
            output_dir,
            "--axes",
            axes,
            "--tile-sizes",
            tile_sizes,
            "--pilot-chunks",
            pilot_chunks,
            "--route-start-image",
            route_start_image,
            "--write-plan",
        ]
    )
    pilot_plan = output_dir / "large_scale_3dgs_pilot_plan.json"
    full_plan = output_dir / "large_scale_3dgs_plan.json"
    run_report = output_dir / "large_scale_3dgs_run_report.json"
    pilot_run = _fmt_command(["gs-mapper", "large-scale-3dgs-run", "--plan", pilot_plan])
    full_run = _fmt_command(["gs-mapper", "large-scale-3dgs-run", "--plan", full_plan])
    promote = _fmt_command(
        [
            "gs-mapper",
            "large-scale-3dgs-promote",
            "--plan",
            full_plan,
            "--run-report",
            run_report,
            "--public-root",
            public_root,
            "--scene-id",
            scene_id,
            "--label",
            label,
            "--route-order",
            "snake",
        ]
    )
    validate = _fmt_command(
        [
            "npm",
            "--prefix",
            "apps/dreamwalker-web",
            "run",
            "validate:dynamic-map-catalog",
            "--",
            "public/manifests/" + scene_id + "-tile-catalog.json",
            "--public-root",
            "public",
            "--site-url",
            "/dreamwalker/",
            "--preload-mode",
            "metadata",
            "--route",
            "public/robot-routes/" + scene_id + "-route.json",
            "--route-playback",
            1,
            "--route-playback-ms",
            1200,
            "--route-playback-loop",
            1,
        ]
    )
    commands = {
        "bootstrap": bootstrap,
        "pilotRun": pilot_run,
        "fullRun": full_run,
        "promote": promote,
        "validate": validate,
    }
    if status == "needs-preprocess" and recommendation.get("preprocessCommand"):
        commands["preprocess"] = str(recommendation["preprocessCommand"])
    if recommendation.get("preflightCommand"):
        commands["preflight"] = str(recommendation["preflightCommand"])
    return commands


def build_real_input_report(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = Path(args.root).resolve()
    output_dir = Path(args.output) if args.output else _default_output_dir(root_dir)
    scene_id = args.scene_id or _default_scene_id(root_dir)
    label = args.label or scene_id.replace("-", " ").title()
    tile_sizes = parse_large_scale_3dgs_tile_sizes(args.tile_sizes)
    discovery = build_large_scale_3dgs_discovery(
        LargeScale3DGSDiscoveryOptions(
            root_dir=root_dir,
            axes=args.axes,
            tile_sizes=tile_sizes,
            target_images_per_chunk=args.target_images_per_chunk,
            pilot_chunks=args.pilot_chunks,
            route_start_image=args.route_start_image,
            max_depth=args.max_depth,
            max_results=args.max_results,
            include_chunk_models=args.include_chunk_models,
        )
    )

    accepted_scenes = []
    rejected_scenes = []
    for scene in discovery["colmapScenes"]:
        reason = _reject_reason_for_scene(
            scene,
            min_images=args.min_images,
            min_points=args.min_points,
            min_extent_m=args.min_extent_m,
            allow_fixtures=args.allow_fixtures,
        )
        record = {
            "dataDir": scene.get("dataDir", ""),
            "registeredImageCount": scene.get("registeredImageCount", 0),
            "points3DCount": scene.get("points3DCount", 0),
            "extentM": round(_axis_extent(scene), 3),
        }
        if reason:
            rejected_scenes.append({**record, "reason": reason})
        else:
            accepted_scenes.append(record)

    accepted_bags = []
    rejected_bags = []
    for bag_input in discovery["bagInputs"]:
        reason = _reject_reason_for_bag(
            bag_input,
            min_bag_bytes=args.min_bag_bytes,
            allow_fixtures=args.allow_fixtures,
        )
        record = {
            "source": bag_input.get("source", ""),
            "kind": bag_input.get("kind", ""),
            "bytes": bag_input.get("bytes", 0),
        }
        if reason:
            rejected_bags.append({**record, "reason": reason})
        else:
            accepted_bags.append(record)

    if accepted_scenes:
        status = "ready-colmap"
    elif accepted_bags:
        status = "needs-preprocess"
    elif discovery["colmapScenes"] or discovery["bagInputs"]:
        status = "fixture-only"
    else:
        status = "needs-input"

    commands = _build_commands(
        root_dir=root_dir,
        output_dir=output_dir,
        public_root=Path(args.public_root),
        scene_id=scene_id,
        label=label,
        axes=args.axes,
        tile_sizes=args.tile_sizes,
        pilot_chunks=args.pilot_chunks,
        route_start_image=args.route_start_image,
        recommendation=discovery["recommendation"],
        status=status,
    )
    return {
        "version": 1,
        "type": "large-scale-3dgs-real-input-check",
        "status": status,
        "rootDir": str(root_dir),
        "outputDir": str(output_dir),
        "publicRoot": str(args.public_root),
        "sceneId": scene_id,
        "label": label,
        "thresholds": {
            "minImages": args.min_images,
            "minPoints": args.min_points,
            "minExtentM": args.min_extent_m,
            "minBagBytes": args.min_bag_bytes,
            "allowFixtures": args.allow_fixtures,
        },
        "summary": {
            "acceptedColmapSceneCount": len(accepted_scenes),
            "acceptedBagInputCount": len(accepted_bags),
            "rejectedColmapSceneCount": len(rejected_scenes),
            "rejectedBagInputCount": len(rejected_bags),
            "discoveredSplatGroupCount": len(discovery["splatGroups"]),
        },
        "accepted": {
            "colmapScenes": accepted_scenes,
            "bagInputs": accepted_bags,
        },
        "rejected": {
            "colmapScenes": rejected_scenes,
            "bagInputs": rejected_bags,
        },
        "discovery": discovery,
        "commands": commands,
    }


def format_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Large-scale 3DGS real input check",
        f"  status: {report['status']}",
        f"  root: {report['rootDir']}",
        f"  accepted COLMAP: {summary['acceptedColmapSceneCount']}",
        f"  accepted bags: {summary['acceptedBagInputCount']}",
        f"  rejected COLMAP: {summary['rejectedColmapSceneCount']}",
        f"  rejected bags: {summary['rejectedBagInputCount']}",
    ]
    if report["accepted"]["colmapScenes"]:
        scene = report["accepted"]["colmapScenes"][0]
        lines.append(
            "  primary COLMAP: "
            f"{scene['dataDir']} ({scene['registeredImageCount']} images, "
            f"{scene['points3DCount']} points, extent {scene['extentM']}m)"
        )
    elif report["accepted"]["bagInputs"]:
        bag = report["accepted"]["bagInputs"][0]
        lines.append(f"  primary bag: {bag['source']} ({bag['kind']}, {bag['bytes']} bytes)")

    commands = report["commands"]
    if report["status"] == "needs-preprocess" and commands.get("preprocess"):
        lines.append(f"  next preprocess: {commands['preprocess']}")
    lines.append(f"  next bootstrap: {commands['bootstrap']}")
    lines.append(f"  next pilot run: {commands['pilotRun']}")
    lines.append(f"  next full run: {commands['fullRun']}")
    lines.append(f"  next promote: {commands['promote']}")
    lines.append(f"  next validate: {commands['validate']}")
    return "\n".join(lines)


def format_shell(report: dict[str, Any]) -> str:
    commands = report["commands"]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# status: {report['status']}",
    ]
    if report["status"] not in OK_STATUSES:
        lines.extend(
            [
                f"echo {_fmt_command(['large-scale 3DGS input gate did not pass:', report['status']])} >&2",
                "exit 1",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    if report["status"] == "needs-preprocess" and commands.get("preprocess"):
        lines.extend([commands["preprocess"], ""])
    lines.extend(
        [
            commands["bootstrap"],
            commands["pilotRun"],
            "",
            "# Inspect the pilot output before starting the full route.",
            commands["fullRun"],
            commands["promote"],
            commands["validate"],
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/large-scale-3dgs-real", help="Real input staging root")
    parser.add_argument("--output", default=None, help="Output directory for bootstrap/preflight/run artifacts")
    parser.add_argument("--public-root", default="apps/dreamwalker-web/public", help="Dynamic Map Viewer public root")
    parser.add_argument("--scene-id", default=None, help="Scene id for promoted public assets")
    parser.add_argument("--label", default=None, help="Human-readable promoted scene label")
    parser.add_argument("--axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--tile-sizes", default="20,30,50")
    parser.add_argument("--target-images-per-chunk", type=int, default=48)
    parser.add_argument("--pilot-chunks", type=int, default=6)
    parser.add_argument("--route-start-image", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--include-chunk-models", action="store_true")
    parser.add_argument("--min-images", type=int, default=100, help="Minimum registered images for real COLMAP")
    parser.add_argument("--min-points", type=int, default=1000, help="Minimum sparse points for real COLMAP")
    parser.add_argument("--min-extent-m", type=float, default=20.0, help="Minimum map span along any tiling axis")
    parser.add_argument("--min-bag-bytes", type=int, default=10_000_000, help="Minimum raw bag input size")
    parser.add_argument(
        "--allow-fixtures", action="store_true", help="Accept paths containing smoke/fixture/demo/sample"
    )
    parser.add_argument("--format", choices=["text", "json", "shell"], default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_real_input_report(args)
    except Exception as exc:
        print(f"large-scale 3DGS input check failed: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, indent=2))
    elif args.format == "shell":
        print(format_shell(report), end="")
    else:
        print(format_text(report))
    return 0 if report["status"] in OK_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
