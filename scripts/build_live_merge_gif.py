#!/usr/bin/env python3
"""Replay two completed live sessions into collaborative live-merge GIF frames."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gs_sim2real.robotics.live_merge import LiveMergeConfig, merge_once, merge_preview  # noqa: E402
from gs_sim2real.viewer.web_viewer import load_ply  # noqa: E402


def _successful_rounds(session: Path) -> list[int]:
    rounds: list[int] = []
    for round_dir in sorted((session / "rounds").glob("round_*")):
        if not round_dir.is_dir():
            continue
        try:
            index = int(round_dir.name.removeprefix("round_"))
        except ValueError:
            continue
        if (round_dir / "train" / "point_cloud.ply").is_file():
            rounds.append(index)
    return rounds


def _limit_rounds(rounds: list[int], limit: int | None) -> list[int]:
    if limit is None or limit <= 0:
        return rounds
    return rounds[:limit]


def _schedule(rounds_a: list[int], rounds_b: list[int]) -> list[tuple[int, int]]:
    if not rounds_a or not rounds_b:
        raise RuntimeError("both sessions need at least one successful round with train/point_cloud.ply")

    ia = 0
    ib = 0
    events = [(rounds_a[ia], rounds_b[ib])]
    advance_a = True
    while ia + 1 < len(rounds_a) or ib + 1 < len(rounds_b):
        if advance_a and ia + 1 < len(rounds_a):
            ia += 1
        elif not advance_a and ib + 1 < len(rounds_b):
            ib += 1
        elif ia + 1 < len(rounds_a):
            ia += 1
        elif ib + 1 < len(rounds_b):
            ib += 1
        events.append((rounds_a[ia], rounds_b[ib]))
        advance_a = not advance_a
    return events


def _caption(image_path: Path, text: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    pad = 16
    box_h = 44
    draw.rectangle((0, 0, image.width, box_h), fill=(255, 255, 255))
    draw.text((pad, 13), text, fill=(20, 20, 20))
    return image


def build_gif(args: argparse.Namespace) -> Path:
    session_a = Path(args.session_a)
    session_b = Path(args.session_b)
    output = Path(args.output)
    workdir = Path(args.workdir) if args.workdir else output.parent / "live_merge_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    rounds_a = _limit_rounds(_successful_rounds(session_a), args.max_rounds_a)
    rounds_b = _limit_rounds(_successful_rounds(session_b), args.max_rounds_b)
    events = _schedule(rounds_a, rounds_b)

    config = LiveMergeConfig(
        align=args.align,
        dedup_radius_camera_heights=args.dedup_radius,
        device=args.device,
        normalize_extent=2.0,
    )

    final_a, final_b = events[-1]
    final_stats = merge_once(
        session_a,
        session_b,
        workdir,
        config=config,
        round_a=final_a,
        round_b=final_b,
        write_preview=False,
    )
    frozen_normalize = final_stats["normalize_params"]
    final_positions = np.asarray(load_ply(str(workdir / "live" / "merged.ply")).positions, dtype=np.float64)
    _preview_path, bounds = merge_preview(
        final_positions,
        int(final_stats["gaussians_a"]),
        workdir / "final_bounds.png",
        image_width=args.width,
    )

    frames: list[Image.Image] = []
    for index, (round_a, round_b) in enumerate(events, start=1):
        stats = merge_once(
            session_a,
            session_b,
            workdir,
            config=config,
            round_a=round_a,
            round_b=round_b,
            normalize_params=frozen_normalize,
            write_preview=False,
        )
        positions = np.asarray(load_ply(str(workdir / "live" / "merged.ply")).positions, dtype=np.float64)
        frame_path = workdir / f"frame_{index:03d}.png"
        merge_preview(
            positions,
            int(stats["gaussians_a"]),
            frame_path,
            image_width=args.width,
            bounds=bounds,
        )
        caption = f"robot A round {round_a} + robot B round {round_b} - {int(stats['merged']):,} gaussians"
        frames.append(_caption(frame_path, caption))
        print(
            f"{index:03d}: A round {round_a} + B round {round_b} -> "
            f"{int(stats['merged']):,} gaussians ({int(stats['deduplicated']):,} deduplicated)"
        )

    durations = [700] * len(frames)
    if durations:
        durations[-1] = 2500
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay two live sessions and build a collaborative merge GIF")
    parser.add_argument("--session-a", required=True, help="First live-mapping session directory")
    parser.add_argument("--session-b", required=True, help="Second live-mapping session directory")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--workdir", default=None, help="Scratch directory (default: output parent/live_merge_work)")
    parser.add_argument("--align", choices=["auto", "shared", "localize"], default="auto")
    parser.add_argument("--dedup-radius", type=float, default=0.1, help="Dedup radius in camera-height units")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--max-rounds-a", type=int, default=None)
    parser.add_argument("--max-rounds-b", type=int, default=None)
    build_gif(parser.parse_args())


if __name__ == "__main__":
    main()
