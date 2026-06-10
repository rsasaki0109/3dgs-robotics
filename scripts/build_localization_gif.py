#!/usr/bin/env python3
"""Build a localization trajectory overlay GIF from a live-mapping session.

Localizes keyframes that were not part of the final round rebuild, then
composes a top-down map view with the mapped trajectory and estimated poses.

    python3 scripts/build_localization_gif.py \
        --session outputs/live_demo_kitti0056/session \
        --output docs/images/live-mapping/localization-kitti0056.gif
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gs_sim2real.robotics.localize import (  # noqa: E402
    LocalizeConfig,
    list_non_round_keyframes,
    load_mapped_records,
    localize_queries,
    resolve_live_map_session,
)

FRAME_SIZE = (960, 420)
FRAME_DURATION_MS = 900
LAST_FRAME_HOLD_MS = 2600
INK = (236, 240, 246)
ACCENT = (96, 205, 255)
MAP_TRAIL = (80, 190, 255)
EST_TRAIL = (255, 140, 96)
GT_TRAIL = (120, 220, 160)


def _load_gif_module():
    path = REPO / "scripts" / "build_live_mapping_gif.py"
    spec = importlib.util.spec_from_file_location("build_live_mapping_gif", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_trajectory(draw: ImageDraw.ImageDraw, camera, centers: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    projected = camera.project(centers)
    visible = projected[:, 2] > 0
    points = [(float(p[0]), float(p[1])) for p, v in zip(projected, visible) if v]
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def build_localization_gif(
    session_dir: Path,
    output: Path,
    *,
    size: tuple[int, int] = FRAME_SIZE,
    max_queries: int = 12,
    refine_iters: int = 60,
    device: str = "cuda",
) -> dict:
    gif = _load_gif_module()
    session = resolve_live_map_session(session_dir)
    mapped_records = load_mapped_records(session)
    mapped_centers = np.asarray([record.center for record in mapped_records], dtype=np.float64)

    query_paths = list_non_round_keyframes(session, mapped_records)
    if max_queries > 0:
        stride = max(1, len(query_paths) // max_queries)
        query_paths = query_paths[::stride][:max_queries]
    if not query_paths:
        raise SystemExit("no non-round keyframes available for localization demo")

    summary = localize_queries(
        session.session_dir,
        query_paths,
        round_index=session.round.round_index,
        config=LocalizeConfig(device=device, refine_iters=refine_iters),
    )

    rounds = gif.load_rounds(session.session_dir)
    anchor = rounds[-1]
    transforms = gif.align_to_anchor(rounds)
    anchor_splat = gif.load_ply_aligned(anchor.ply_path, transforms[-1])
    camera = gif.fit_view_camera(anchor, anchor_splat, size)
    map_render = gif.render_splat(anchor_splat, camera)

    est_centers: list[np.ndarray] = []
    gt_centers: list[np.ndarray] = []
    frames: list[Image.Image] = []
    for index, result in enumerate(summary.results, start=1):
        est_centers.append(result.center)
        if result.gt_center is not None:
            gt_centers.append(result.gt_center)
        frame = map_render.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        _draw_trajectory(draw, camera, mapped_centers, MAP_TRAIL, 3)
        if gt_centers:
            _draw_trajectory(draw, camera, np.asarray(gt_centers), GT_TRAIL, 3)
        _draw_trajectory(draw, camera, np.asarray(est_centers), EST_TRAIL, 4)

        rel_errors = [
            r.relative_translation_error
            for r in summary.results[:index]
            if r.relative_translation_error is not None
        ]
        median_rel = float(np.median(rel_errors)) if rel_errors else float("nan")
        banner_h = 58
        draw.rectangle((0, size[1] - banner_h, size[0], size[1]), fill=(8, 12, 18, 225))
        draw.text(
            (16, size[1] - banner_h + 8),
            f"3DGS localization — query {index}/{len(summary.results)}",
            font=_load_font(22),
            fill=INK,
        )
        draw.text(
            (16, size[1] - banner_h + 36),
            f"seed {result.seed_keyframe} · rel err median {median_rel:.2f} keyframe spacings · gauge-relative",
            font=_load_font(15),
            fill=ACCENT,
        )
        frames.append(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [FRAME_DURATION_MS] * len(frames)
    if durations:
        durations[-1] = LAST_FRAME_HOLD_MS
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )

    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(summary.to_json(), indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "sidecar": str(sidecar),
        "queries": len(summary.results),
        "median_neighbor_spacing": summary.median_neighbor_spacing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Live-mapping session directory")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--width", type=int, default=FRAME_SIZE[0])
    parser.add_argument("--height", type=int, default=FRAME_SIZE[1])
    parser.add_argument("--max-queries", type=int, default=12, help="Cap query count for runtime (default: 12)")
    parser.add_argument("--refine-iters", type=int, default=60, help="Adam steps per pyramid level (default: 60)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = build_localization_gif(
        Path(args.session),
        Path(args.output),
        size=(args.width, args.height),
        max_queries=args.max_queries,
        refine_iters=args.refine_iters,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
