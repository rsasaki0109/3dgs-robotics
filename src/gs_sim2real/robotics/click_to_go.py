"""Serve click-to-go navigation for the browser splat viewer.

The service accepts viewer click rays in the viewer splat frame, maps them
back through the frame chain (viewer splat frame -> round gauge -> grid
plane), runs navigation to the ground-plane hit, exports a fresh overlay, and
serves the generated files back to the viewer.

Coordinates are in round-gauge units, which are camera-height-relative map
units rather than calibrated metric units.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ClickToGoConfig:
    port: int = 8787
    round_index: int | None = None
    localize_every: int = 0
    odom_noise: float = 0.0
    device: str = "cuda"
    baseline_round: int | None = None
    changes_align: str = "auto"


@dataclass
class SceneFrame:
    mapper: Any
    basis: np.ndarray
    ground_height: float
    camera_height: float
    splat_rel: str


def load_scene_frame(session_dir: Path, *, round_index: int | None = None) -> SceneFrame:
    from gs_sim2real.robotics import viewer_overlay
    from gs_sim2real.robotics.localize import load_mapped_records, resolve_live_map_session
    from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame
    from gs_sim2real.viewer.web_viewer import load_ply

    session_dir = Path(session_dir)
    session = resolve_live_map_session(session_dir, round_index=round_index)
    records = load_mapped_records(session)
    centers = np.asarray([record.center for record in records], dtype=np.float64)
    points = np.asarray(load_ply(str(session.round.ply_path)).positions, dtype=np.float64)

    up, forward, ground_height, camera_height = estimate_ground_frame(
        centers,
        [record.qvec for record in records],
        lambda candidate_up: points @ candidate_up,
        ground_percentile=30.0,
    )
    up = _normalize(up, "estimated up vector is degenerate")
    e1 = np.asarray(forward, dtype=np.float64) - float(np.dot(forward, up)) * up
    e1 = _normalize(e1, "estimated forward vector is degenerate")
    e2 = _normalize(np.cross(up, e1), "estimated lateral vector is degenerate")
    basis = np.stack([e1, e2, up])

    splat_path = session.round.round_dir / "scene.splat"
    return SceneFrame(
        mapper=viewer_overlay.splat_frame_mapper(session),
        basis=basis,
        ground_height=float(ground_height),
        camera_height=float(camera_height),
        splat_rel=splat_path.relative_to(session_dir).as_posix(),
    )


def splat_ray_to_goal(
    origin_splat: Sequence[float], direction_splat: Sequence[float], frame: SceneFrame
) -> tuple[float, float]:
    origin = _as_vec3(origin_splat, "origin")
    direction = _as_vec3(direction_splat, "direction")
    mapper = frame.mapper

    origin_round = (
        ((origin * float(mapper.factor) + mapper.centroid) - mapper.translation) @ mapper.rotation / float(mapper.scale)
    )
    direction_round = direction @ mapper.rotation
    direction_round = _normalize(direction_round, "direction is degenerate")

    up = np.asarray(frame.basis[2], dtype=np.float64)
    denom = float(np.dot(up, direction_round))
    if abs(denom) < 1e-9:
        raise ValueError("the click ray does not hit the ground plane - aim at the road")

    t_hit = (float(frame.ground_height) - float(np.dot(up, origin_round))) / denom
    if t_hit <= 0.0:
        raise ValueError("the click ray does not hit the ground plane - aim at the road")

    hit = origin_round + t_hit * direction_round
    return float(np.dot(frame.basis[0], hit)), float(np.dot(frame.basis[1], hit))


def run_navigate_and_overlay(
    session_dir: Path,
    goal_xy: tuple[float, float],
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    session_dir = Path(session_dir)
    output_dir = session_dir / "clickgo"
    output_dir.mkdir(parents=True, exist_ok=True)
    nav_json = output_dir / "nav_result.json"
    overlay_json = output_dir / "overlay.json"

    runner = run_cli or _run_cli
    goal = f"{goal_xy[0]},{goal_xy[1]}"

    nav_args = [
        "navigate",
        "--map",
        str(session_dir),
        "--goal",
        goal,
        "--output",
        str(nav_json),
        "--localize-every",
        str(config.localize_every),
        "--odom-noise",
        str(config.odom_noise),
        "--device",
        config.device,
    ]
    if config.round_index is not None:
        nav_args.extend(["--round", str(config.round_index)])
    runner(nav_args)

    overlay_args = [
        "export-overlay",
        "--map",
        str(session_dir),
        "--output",
        str(overlay_json),
        "--nav",
        str(nav_json),
    ]
    if config.round_index is not None:
        overlay_args.extend(["--round", str(config.round_index)])
    runner(overlay_args)

    nav_data = json.loads(nav_json.read_text(encoding="utf-8"))
    return {
        "reached": bool(nav_data.get("reached", False)),
        "steps": int(nav_data.get("steps", 0)),
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "overlay": "/clickgo/overlay.json",
        "nav_json": "/clickgo/nav_result.json",
    }


def run_query_and_overlay(
    session_dir: Path,
    prompt: str,
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Run an open-vocabulary query and export a hit overlay for the viewer."""
    session_dir = Path(session_dir)
    output_dir = session_dir / "clickgo"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_json = output_dir / "query.json"
    overlay_json = output_dir / "overlay.json"

    runner = run_cli or _run_cli

    query_args = [
        "query-map",
        prompt,
        "--map",
        str(session_dir),
        "--output",
        str(query_json),
        "--device",
        config.device,
    ]
    if config.round_index is not None:
        query_args.extend(["--round", str(config.round_index)])
    runner(query_args)

    overlay_args = [
        "export-overlay",
        "--map",
        str(session_dir),
        "--output",
        str(overlay_json),
        "--query",
        str(query_json),
    ]
    if config.round_index is not None:
        overlay_args.extend(["--round", str(config.round_index)])
    runner(overlay_args)

    query_data = json.loads(query_json.read_text(encoding="utf-8"))
    return {
        "prompt": prompt,
        "hits": len(query_data.get("hits", [])),
        "overlay": "/clickgo/overlay.json",
        "query_json": "/clickgo/query.json",
    }


def _export_aligned_splat(
    session_dir: Path,
    cleaned_ply: Path,
    output_splat: Path,
    *,
    round_index: int | None = None,
) -> int:
    """Export a cleaned round PLY to a ``.splat`` aligned with the served scene.

    ``splat-clean`` removes gaussian rows but keeps the round's full-precision
    gauge, so replaying the session's similarity transform and normalization
    params (the same pair that built ``scene.splat``) lands every surviving
    gaussian exactly where it sat in the original viewer splat. Returns the
    gaussian count actually written (32 bytes per gaussian).
    """
    from gs_sim2real.robotics import viewer_overlay
    from gs_sim2real.robotics.localize import resolve_live_map_session
    from gs_sim2real.viewer.web_export import ply_to_splat

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    mapper = viewer_overlay.splat_frame_mapper(session)
    ply_to_splat(
        cleaned_ply,
        output_splat,
        similarity_transform=(float(mapper.scale), mapper.rotation, mapper.translation),
        normalize_params=(mapper.centroid, float(mapper.factor)),
    )
    return int(Path(output_splat).stat().st_size // 32)


def run_clean_and_swap(
    session_dir: Path,
    prompt: str,
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    export_splat: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Erase prompt-matching gaussians and re-export an aligned viewer splat."""
    session_dir = Path(session_dir)
    output_dir = session_dir / "clickgo"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_ply = output_dir / "cleaned.ply"
    cleaned_splat = output_dir / "cleaned.splat"

    runner = run_cli or _run_cli
    exporter = export_splat or _export_aligned_splat

    clean_args = [
        "splat-clean",
        prompt,
        "--map",
        str(session_dir),
        "--output",
        str(cleaned_ply),
        "--device",
        config.device,
    ]
    if config.round_index is not None:
        clean_args.extend(["--round", str(config.round_index)])
    runner(clean_args)

    gaussians = exporter(session_dir, cleaned_ply, cleaned_splat, round_index=config.round_index)
    return {
        "prompt": prompt,
        "splat": "/clickgo/cleaned.splat",
        "gaussians": int(gaussians),
    }


def run_grab_and_swap(
    session_dir: Path,
    prompt: str,
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    export_splat: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Isolate prompt-matching gaussians and export them as an aligned splat."""
    session_dir = Path(session_dir)
    output_dir = session_dir / "clickgo"
    output_dir.mkdir(parents=True, exist_ok=True)
    grabbed_ply = output_dir / "grabbed.ply"
    grabbed_splat = output_dir / "grabbed.splat"

    runner = run_cli or _run_cli
    exporter = export_splat or _export_aligned_splat

    grab_args = [
        "splat-grab",
        prompt,
        "--map",
        str(session_dir),
        "--output",
        str(grabbed_ply),
        "--device",
        config.device,
    ]
    if config.round_index is not None:
        grab_args.extend(["--round", str(config.round_index)])
    runner(grab_args)

    gaussians = exporter(session_dir, grabbed_ply, grabbed_splat, round_index=config.round_index)
    return {
        "prompt": prompt,
        "splat": "/clickgo/grabbed.splat",
        "gaussians": int(gaussians),
    }


def run_changes_and_overlay(
    session_dir: Path,
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Diff the served round against a baseline round and overlay the changes.

    Map A is the served round (so appeared/disappeared boxes land on the
    current splat); map B is the baseline. ``appeared`` clusters are solid now
    but absent in the baseline, ``disappeared`` clusters are the reverse.
    """
    if config.baseline_round is None:
        raise ValueError("no baseline round configured; restart click-to-go with --baseline-round")

    session_dir = Path(session_dir)
    output_dir = session_dir / "clickgo"
    output_dir.mkdir(parents=True, exist_ok=True)
    changes_json = output_dir / "changes.json"
    overlay_json = output_dir / "overlay.json"

    runner = run_cli or _run_cli

    changes_args = [
        "detect-changes",
        "--map-a",
        str(session_dir),
        "--round-b",
        str(config.baseline_round),
        "--align",
        config.changes_align,
        "--output",
        str(changes_json),
        "--device",
        config.device,
    ]
    if config.round_index is not None:
        changes_args.extend(["--round-a", str(config.round_index)])
    runner(changes_args)

    overlay_args = [
        "export-overlay",
        "--map",
        str(session_dir),
        "--output",
        str(overlay_json),
        "--changes",
        str(changes_json),
    ]
    if config.round_index is not None:
        overlay_args.extend(["--round", str(config.round_index)])
    runner(overlay_args)

    changes_data = json.loads(changes_json.read_text(encoding="utf-8"))
    return {
        "appeared": len(changes_data.get("appeared", [])),
        "disappeared": len(changes_data.get("disappeared", [])),
        "overlay": "/clickgo/overlay.json",
        "changes_json": "/clickgo/changes.json",
    }


# Semantic glow: the colour painted onto gaussians that fall inside a query hit
# box, and the alpha multiplier that fades everything else so the match pops.
GLOW_RGBA = (94, 234, 178, 255)
DIM_ALPHA_SCALE = 0.32


def _overlay_box_aabbs(overlay_json: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read an overlay's wireframe boxes back as splat-frame AABBs (low, high)."""
    overlay = json.loads(Path(overlay_json).read_text(encoding="utf-8"))
    aabbs: list[tuple[np.ndarray, np.ndarray]] = []
    for marker in overlay.get("markers", []):
        box = marker.get("box")
        if not box:
            continue
        corners = np.asarray(box, dtype=np.float64)
        aabbs.append((corners.min(axis=0), corners.max(axis=0)))
    return aabbs


def _highlight_splat_file(scene_splat: Path, output_splat: Path, aabbs: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """Glow the gaussians inside any AABB and dim the rest into a 32-byte splat.

    Works entirely in the served splat frame — the box corners already live
    there — so no gauge replay is needed. Each record is 32 bytes: position
    (float32x3), scale (float32x3), RGBA (uint8x4 at offset 24), rotation
    (uint8x4). Returns the number of gaussians lit.
    """
    raw = np.frombuffer(Path(scene_splat).read_bytes(), dtype=np.uint8)
    count = int(raw.size // 32)
    arr = raw[: count * 32].reshape(count, 32).copy()
    if count == 0 or not aabbs:
        Path(output_splat).write_bytes(arr.tobytes())
        return 0

    positions = np.ascontiguousarray(arr[:, 0:12]).view(np.float32).reshape(count, 3).astype(np.float64)
    inside = np.zeros(count, dtype=bool)
    for low, high in aabbs:
        inside |= np.all((positions >= low) & (positions <= high), axis=1)

    arr[inside, 24:28] = np.asarray(GLOW_RGBA, dtype=np.uint8)
    arr[~inside, 27] = (arr[~inside, 27].astype(np.float64) * DIM_ALPHA_SCALE).astype(np.uint8)

    Path(output_splat).write_bytes(arr.tobytes())
    return int(np.count_nonzero(inside))


def _highlight_aligned_splat(
    session_dir: Path, overlay_json: Path, output_splat: Path, *, round_index: int | None = None
) -> int:
    """Recolor the served ``scene.splat`` so the overlay's hit boxes glow."""
    from gs_sim2real.robotics.localize import resolve_live_map_session

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    scene_splat = session.round.round_dir / "scene.splat"
    return _highlight_splat_file(scene_splat, output_splat, _overlay_box_aabbs(overlay_json))


def run_highlight_and_swap(
    session_dir: Path,
    prompt: str,
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    highlight_splat: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Box prompt matches and recolor them to glow inside the served splat.

    Reuses ``run_query_and_overlay`` to write the hit boxes, then lights the
    gaussians inside those boxes so the viewer can hot-swap a splat where the
    search result glows and everything else fades back.
    """
    session_dir = Path(session_dir)
    query_result = run_query_and_overlay(session_dir, prompt, config, run_cli=run_cli)
    overlay_json = session_dir / "clickgo" / "overlay.json"
    highlighted_splat = session_dir / "clickgo" / "highlighted.splat"

    highlighter = highlight_splat or _highlight_aligned_splat
    highlighted = highlighter(session_dir, overlay_json, highlighted_splat, round_index=config.round_index)
    return {
        "prompt": prompt,
        "hits": int(query_result["hits"]),
        "highlighted": int(highlighted),
        "splat": "/clickgo/highlighted.splat",
        "overlay": "/clickgo/overlay.json",
    }


# Confidence axis: paint every gaussian by how solid the optimizer made it.
# A gaussian's opacity is its own confidence — translucent ones are the filler
# the fit never committed to. The heatmap runs warm (low) -> cool (high), and
# the diagnostic view forces a uniform readable alpha so the colours show.
LOW_CONFIDENCE_OPACITY = 0.3
QUALITY_ALPHA = 210
_CONFIDENCE_STOPS = (
    (0.0, (229, 57, 53)),
    (0.5, (251, 192, 45)),
    (1.0, (38, 198, 158)),
)


def _confidence_color(scores: np.ndarray) -> np.ndarray:
    """Map confidence scores in [0, 1] to a warm-low -> cool-high RGB heatmap."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    rgb = np.zeros((scores.size, 3), dtype=np.float64)
    for (t0, c0), (t1, c1) in zip(_CONFIDENCE_STOPS[:-1], _CONFIDENCE_STOPS[1:]):
        mask = (scores >= t0) & (scores <= t1)
        span = t1 - t0
        frac = (scores[mask] - t0) / span if span else np.zeros(int(mask.sum()))
        rgb[mask] = np.outer(1.0 - frac, c0) + np.outer(frac, c1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _quality_splat_file(scene_splat: Path, output_splat: Path) -> dict[str, Any]:
    """Heatmap the served splat by per-gaussian opacity into a 32-byte splat.

    Reads the alpha byte (offset 27) of each record as a confidence score,
    repaints RGB (offset 24-26) to the heatmap colour, and pins alpha to a
    readable constant. Returns the gaussian count, how many fell below the
    low-confidence opacity cut, and the median opacity.
    """
    raw = np.frombuffer(Path(scene_splat).read_bytes(), dtype=np.uint8)
    count = int(raw.size // 32)
    arr = raw[: count * 32].reshape(count, 32).copy()
    if count == 0:
        Path(output_splat).write_bytes(arr.tobytes())
        return {"gaussians": 0, "low_confidence": 0, "median_opacity": 0.0}

    opacity = arr[:, 27].astype(np.float64) / 255.0
    arr[:, 24:27] = _confidence_color(opacity)
    arr[:, 27] = QUALITY_ALPHA
    Path(output_splat).write_bytes(arr.tobytes())
    return {
        "gaussians": count,
        "low_confidence": int(np.count_nonzero(opacity < LOW_CONFIDENCE_OPACITY)),
        "median_opacity": float(np.median(opacity)),
    }


def _quality_aligned_splat(session_dir: Path, output_splat: Path, *, round_index: int | None = None) -> dict[str, Any]:
    """Heatmap the served ``scene.splat`` by per-gaussian confidence."""
    from gs_sim2real.robotics.localize import resolve_live_map_session

    session = resolve_live_map_session(Path(session_dir), round_index=round_index)
    scene_splat = session.round.round_dir / "scene.splat"
    return _quality_splat_file(scene_splat, output_splat)


def run_quality_and_swap(
    session_dir: Path,
    config: ClickToGoConfig,
    *,
    quality_splat: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recolor the served splat into a per-gaussian confidence heatmap.

    Needs no query or CLI pass — the splat carries its own opacity, so the
    Confidence axis is a pure recolor the viewer can hot-swap in place.
    """
    session_dir = Path(session_dir)
    output_dir = session_dir / "clickgo"
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_splat_path = output_dir / "quality.splat"

    builder = quality_splat or _quality_aligned_splat
    stats = builder(session_dir, quality_splat_path, round_index=config.round_index)
    return {
        "splat": "/clickgo/quality.splat",
        "gaussians": int(stats["gaussians"]),
        "low_confidence": int(stats["low_confidence"]),
        "median_opacity": float(stats["median_opacity"]),
    }


class ClickToGoHandler(SimpleHTTPRequestHandler):
    frame: SceneFrame
    config: ClickToGoConfig
    session_dir: Path
    lock: threading.Lock
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None
    exporter: Callable[..., int] | None = None
    highlighter: Callable[..., int] | None = None
    qualitymapper: Callable[..., dict[str, Any]] | None = None

    def end_headers(self) -> None:  # noqa: N802 - http.server API
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        return

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server API
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/goal":
            self._handle_goal()
        elif self.path == "/query":
            self._handle_query()
        elif self.path == "/clean":
            self._handle_clean()
        elif self.path == "/grab":
            self._handle_grab()
        elif self.path == "/highlight":
            self._handle_highlight()
        elif self.path == "/quality":
            self._handle_quality()
        elif self.path == "/changes":
            self._handle_changes()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_goal(self) -> None:
        try:
            payload = self._read_json_body()
            origin = payload["origin"]
            direction = payload["direction"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"malformed JSON body: {exc}"})
            return

        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "navigation already running"})
            return

        try:
            try:
                goal_xy = splat_ray_to_goal(origin, direction, self.frame)
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return

            try:
                result = run_navigate_and_overlay(self.session_dir, goal_xy, self.config, run_cli=self.runner)
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _handle_query(self) -> None:
        try:
            payload = self._read_json_body()
            prompt = str(payload["prompt"]).strip()
            if not prompt:
                raise ValueError("prompt is empty")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"malformed JSON body: {exc}"})
            return

        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "another request is already running"})
            return

        try:
            try:
                result = run_query_and_overlay(self.session_dir, prompt, self.config, run_cli=self.runner)
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _handle_clean(self) -> None:
        try:
            payload = self._read_json_body()
            prompt = str(payload["prompt"]).strip()
            if not prompt:
                raise ValueError("prompt is empty")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"malformed JSON body: {exc}"})
            return

        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "another request is already running"})
            return

        try:
            try:
                result = run_clean_and_swap(
                    self.session_dir, prompt, self.config, run_cli=self.runner, export_splat=self.exporter
                )
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _handle_grab(self) -> None:
        try:
            payload = self._read_json_body()
            prompt = str(payload["prompt"]).strip()
            if not prompt:
                raise ValueError("prompt is empty")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"malformed JSON body: {exc}"})
            return

        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "another request is already running"})
            return

        try:
            try:
                result = run_grab_and_swap(
                    self.session_dir, prompt, self.config, run_cli=self.runner, export_splat=self.exporter
                )
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _handle_highlight(self) -> None:
        try:
            payload = self._read_json_body()
            prompt = str(payload["prompt"]).strip()
            if not prompt:
                raise ValueError("prompt is empty")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"malformed JSON body: {exc}"})
            return

        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "another request is already running"})
            return

        try:
            try:
                result = run_highlight_and_swap(
                    self.session_dir, prompt, self.config, run_cli=self.runner, highlight_splat=self.highlighter
                )
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _handle_quality(self) -> None:
        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "another request is already running"})
            return

        try:
            try:
                result = run_quality_and_swap(self.session_dir, self.config, quality_splat=self.qualitymapper)
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _handle_changes(self) -> None:
        if not self.lock.acquire(blocking=False):
            self._send_json(409, {"error": "another request is already running"})
            return

        try:
            try:
                result = run_changes_and_overlay(self.session_dir, self.config, run_cli=self.runner)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)
        finally:
            self.lock.release()

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(
    session_dir: Path,
    config: ClickToGoConfig,
    *,
    run_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    export_splat: Callable[..., int] | None = None,
    highlight_splat: Callable[..., int] | None = None,
    quality_splat: Callable[..., dict[str, Any]] | None = None,
) -> ThreadingHTTPServer:
    session_dir = Path(session_dir)
    frame = load_scene_frame(session_dir, round_index=config.round_index)

    class BoundClickToGoHandler(ClickToGoHandler):
        pass

    BoundClickToGoHandler.frame = frame
    BoundClickToGoHandler.config = config
    BoundClickToGoHandler.session_dir = session_dir
    BoundClickToGoHandler.lock = threading.Lock()
    # staticmethod keeps the callable from binding as a method when reached via self.runner
    BoundClickToGoHandler.runner = staticmethod(run_cli) if run_cli is not None else None
    BoundClickToGoHandler.exporter = staticmethod(export_splat) if export_splat is not None else None
    BoundClickToGoHandler.highlighter = staticmethod(highlight_splat) if highlight_splat is not None else None
    BoundClickToGoHandler.qualitymapper = staticmethod(quality_splat) if quality_splat is not None else None

    handler = partial(BoundClickToGoHandler, directory=str(session_dir))
    server = ThreadingHTTPServer(("0.0.0.0", config.port), handler)
    server.scene_frame = frame  # for callers that need the served splat path
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve click-to-go navigation for the splat viewer")
    parser.add_argument("--map", required=True, help="Live-mapping session directory")
    parser.add_argument("--port", type=int, default=8787, help="HTTP port")
    parser.add_argument("--round", type=int, default=None, help="Rebuild round (default: last successful)")
    parser.add_argument("--localize-every", type=int, default=0, help="Localization cadence for navigation")
    parser.add_argument("--odom-noise", type=float, default=0.0, help="Wheel odometry noise for navigation")
    parser.add_argument("--device", default="cuda", help="torch device for rendering and localization")
    parser.add_argument(
        "--baseline-round",
        type=int,
        default=None,
        help="Earlier round to diff against for the Changes button (enables Dynamic diff)",
    )
    parser.add_argument(
        "--changes-align",
        choices=["auto", "shared", "localize", "none"],
        default="auto",
        help="Gauge alignment for detect-changes",
    )
    args = parser.parse_args()

    config = ClickToGoConfig(
        port=args.port,
        round_index=args.round,
        localize_every=args.localize_every,
        odom_noise=args.odom_noise,
        device=args.device,
        baseline_round=args.baseline_round,
        changes_align=args.changes_align,
    )
    server = make_server(Path(args.map), config)
    port = int(server.server_address[1])
    splat_url = f"http://localhost:{port}/{server.scene_frame.splat_rel}"
    viewer_url = (
        f"https://rsasaki0109.github.io/3dgs-robotics/splat.html?url={splat_url}&clickgo=http://localhost:{port}"
    )

    print(f"splat: {splat_url}")
    print(f"viewer: {viewer_url}")
    print("double-click the road in the viewer to drive there")
    print("type a prompt in the search box to box open-vocabulary hits (e.g. car)")
    print("hit Highlight to glow the matching gaussians inside the splat and dim the rest")
    print("hit Confidence to heatmap the map by how solid each gaussian is (warm = low)")
    print("hit Erase to delete the matching objects and reload the cleaned splat in place")
    print("hit Grab to keep only the matching objects, or Reset to restore the full map")
    if config.baseline_round is not None:
        print(f"hit Changes to box what appeared/disappeared since round {config.baseline_round}")
    print("coordinates use round-gauge camera-height units, not calibrated metric units")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _run_cli(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "gs_sim2real.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr_lines = (proc.stderr or "").splitlines()
        stderr_tail = "\n".join(stderr_lines[-20:]) or "(no stderr captured)"
        raise RuntimeError(
            f"`3dgs-robotics {args[0]}` failed. Check the map path, selected round, optional dependencies, "
            f"and device. Last stderr lines:\n{stderr_tail}"
        )
    return proc


def _as_vec3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    return array


def _normalize(value: Sequence[float], message: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(message)
    return array / norm


if __name__ == "__main__":
    main()
