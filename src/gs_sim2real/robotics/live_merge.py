"""Watch two live-mapping sessions and publish one merged live map."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gs_sim2real.robotics.map_merge import merge_sessions
from gs_sim2real.viewer.web_export import compute_splat_normalization, ply_to_splat
from gs_sim2real.viewer.web_viewer import load_ply


@dataclass(frozen=True)
class LiveMergeConfig:
    align: str = "auto"
    dedup_radius_camera_heights: float = 0.1
    dc_only_b: bool = False
    device: str = "cuda"
    interval_s: float = 5.0
    normalize_extent: float = 2.0
    splat_max_points: int | None = None
    min_opacity: float = 0.02
    max_scale: float = 2.0


def read_last_round(session_dir: Path) -> int | None:
    """Return the latest successful live-mapping round, if one is visible."""
    session_dir = Path(session_dir)
    state_path = session_dir / "live" / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            last = state.get("lastSuccessfulRound") or {}
            round_index = last.get("round")
            if round_index is not None:
                return int(round_index)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    rounds_dir = session_dir / "rounds"
    candidates: list[int] = []
    for round_dir in rounds_dir.glob("round_*"):
        if not round_dir.is_dir():
            continue
        match = re.fullmatch(r"round_(\d+)", round_dir.name)
        if match is not None:
            candidates.append(int(match.group(1)))
    return max(candidates) if candidates else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_for_json(normalize_params: tuple[np.ndarray, float] | None) -> Any:
    if normalize_params is None:
        return None
    centroid, factor = normalize_params
    return [np.asarray(centroid, dtype=float).tolist(), float(factor)]


def _project_top_down(positions: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"expected positions with shape (N, 3), got {pos.shape}")
    if len(pos) == 0:
        return np.zeros((0, 2), dtype=np.float64), (0, 1)
    drop_axis = int(np.argmin(np.var(pos, axis=0)))
    keep_axes = [axis for axis in range(3) if axis != drop_axis]
    spread = np.var(pos[:, keep_axes], axis=0)
    if spread[1] > spread[0]:  # keep the long axis horizontal
        keep_axes = [keep_axes[1], keep_axes[0]]
    return pos[:, keep_axes], (keep_axes[0], keep_axes[1])


def merge_preview(
    merged_positions: np.ndarray,
    split_index: int,
    output_path: Path,
    *,
    image_width: int = 1600,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Path, tuple[np.ndarray, np.ndarray]]:
    """Write a deterministic top-down preview and return the bounds used.

    Default bounds use inner percentiles so far-flung floater gaussians do
    not shrink the map to a sliver.
    """
    from PIL import Image

    xy, _keep_axes = _project_top_down(merged_positions)
    if bounds is None:
        if len(xy) == 0:
            min_xy = np.zeros(2, dtype=np.float64)
            max_xy = np.ones(2, dtype=np.float64)
        else:
            min_xy = np.percentile(xy, 0.5, axis=0)
            max_xy = np.percentile(xy, 99.5, axis=0)
    else:
        min_xy = np.asarray(bounds[0], dtype=np.float64)
        max_xy = np.asarray(bounds[1], dtype=np.float64)

    span = np.maximum(max_xy - min_xy, 1e-9)
    width = max(int(image_width), 1)
    scale = (width - 1) / span[0]
    height = max(int(np.ceil(span[1] * scale)) + 1, 1)

    def to_pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.clip(((points[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, width - 1)
        rows = np.clip(((points[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)
        return rows, cols

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    split = int(np.clip(split_index, 0, len(xy)))

    if split:
        rows, cols = to_pixels(xy[:split])
        image[rows, cols] = (95, 125, 160)
    if split < len(xy):
        rows, cols = to_pixels(xy[split:])
        image[rows, cols] = (235, 140, 40)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[::-1]).save(output_path)
    return output_path, (min_xy, max_xy)


def _localize_config(config: LiveMergeConfig) -> Any | None:
    if config.align not in ("auto", "localize"):
        return None
    from gs_sim2real.robotics.localize import LocalizeConfig

    return LocalizeConfig(device=config.device, refine_iters=40, pyramid_scales=(0.25, 0.5))


def merge_once(
    session_a: Path,
    session_b: Path,
    output_dir: Path,
    *,
    config: LiveMergeConfig,
    round_a: int | None = None,
    round_b: int | None = None,
    normalize_params: tuple[np.ndarray, float] | None = None,
    write_preview: bool = False,
) -> dict[str, Any]:
    """Merge one selected pair of rounds and publish merged.ply/latest.splat atomically."""
    live_dir = Path(output_dir) / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    merged_ply = live_dir / "merged.ply"
    tmp_ply = merged_ply.with_suffix(".ply.tmp")
    stats = merge_sessions(
        Path(session_a),
        Path(session_b),
        tmp_ply,
        round_a=round_a,
        round_b=round_b,
        align=config.align,
        dedup_radius_camera_heights=config.dedup_radius_camera_heights,
        dc_only_b=config.dc_only_b,
        localize_config=_localize_config(config),
    )
    os.replace(tmp_ply, merged_ply)
    stats["output"] = str(merged_ply)

    used_normalize_params = normalize_params
    merged_positions: np.ndarray | None = None
    if used_normalize_params is None and config.normalize_extent > 0:
        merged_positions = np.asarray(load_ply(str(merged_ply)).positions, dtype=np.float64)
        used_normalize_params = compute_splat_normalization(merged_positions, config.normalize_extent)

    latest_splat = live_dir / "latest.splat"
    tmp_splat = latest_splat.with_suffix(".splat.tmp")
    ply_to_splat(
        merged_ply,
        tmp_splat,
        max_points=config.splat_max_points,
        normalize_params=used_normalize_params,
        min_opacity=config.min_opacity,
        max_scale=config.max_scale,
    )
    os.replace(tmp_splat, latest_splat)

    preview_path: str | None = None
    if write_preview:
        if merged_positions is None:
            merged_positions = np.asarray(load_ply(str(merged_ply)).positions, dtype=np.float64)
        preview, _bounds = merge_preview(
            merged_positions,
            int(stats.get("gaussians_a", 0)),
            live_dir / "merge_preview.png",
        )
        preview_path = str(preview)

    return {
        **stats,
        "round_a": round_a,
        "round_b": round_b,
        "normalize_params": used_normalize_params,
        "splat": str(latest_splat),
        "preview": preview_path,
    }


def _merge_line(index: int, stats: dict[str, Any]) -> str:
    alignment = stats.get("alignment") or {}
    return (
        f"merge {index}: A round {stats.get('round_a')} + B round {stats.get('round_b')} "
        f"-> {int(stats.get('merged', 0)):,} gaussians "
        f"(dropped {int(stats.get('deduplicated', 0)):,} dupes, "
        f"align {alignment.get('mode', 'unknown')} scale {float(alignment.get('scale', 1.0)):.3f})"
    )


def _state_merge(stats: dict[str, Any]) -> dict[str, Any]:
    alignment = stats.get("alignment") or {}
    return {
        "roundA": stats.get("round_a"),
        "roundB": stats.get("round_b"),
        "gaussians": stats.get("merged"),
        "deduplicated": stats.get("deduplicated"),
        "alignment": alignment,
    }


def watch_and_merge(
    session_a: Path,
    session_b: Path,
    output_dir: Path,
    *,
    config: LiveMergeConfig,
    once: bool = False,
    max_merges: int | None = None,
    sleep_fn=time.sleep,
    log_fn=print,
    write_preview: bool = False,
) -> list[dict[str, Any]]:
    """Watch two live sessions and publish a merged live map whenever either advances."""
    session_a = Path(session_a)
    session_b = Path(session_b)
    output_dir = Path(output_dir)
    live_dir = output_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    state_merges: list[dict[str, Any]] = []
    last_merged: tuple[int, int] | None = None
    failed_pairs: set[tuple[int, int]] = set()
    frozen_normalize: tuple[np.ndarray, float] | None = None

    try:
        while True:
            round_a = read_last_round(session_a)
            round_b = read_last_round(session_b)
            if round_a is None or round_b is None:
                if once:
                    missing = session_a if round_a is None else session_b
                    raise RuntimeError(f"{missing} has no successful live-mapping round yet")
            else:
                pair = (round_a, round_b)
                if pair != last_merged and pair not in failed_pairs:
                    try:
                        stats = merge_once(
                            session_a,
                            session_b,
                            output_dir,
                            config=config,
                            round_a=round_a,
                            round_b=round_b,
                            normalize_params=frozen_normalize,
                            write_preview=write_preview,
                        )
                    except (ValueError, FileNotFoundError) as error:
                        if once:
                            raise
                        failed_pairs.add(pair)
                        log_fn(f"merge-live failed for A round {round_a} + B round {round_b}: {error}")
                    else:
                        if frozen_normalize is None:
                            frozen_normalize = stats.get("normalize_params")
                        last_merged = pair
                        results.append(stats)
                        state_merges.append(_state_merge(stats))
                        state = {
                            "mode": "merge-live",
                            "sessionA": str(session_a),
                            "sessionB": str(session_b),
                            "merges": state_merges,
                            "lastMerge": state_merges[-1],
                            "updatedUnix": round(time.time(), 3),
                            "splatUrl": "latest.splat",
                        }
                        _atomic_write_json(live_dir / "state.json", state)
                        log_fn(_merge_line(len(results), stats))
                        if once or (max_merges is not None and len(results) >= max_merges):
                            return results

            if once:
                return results
            sleep_fn(config.interval_s)
    except KeyboardInterrupt:
        return results
