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


class ClickToGoHandler(SimpleHTTPRequestHandler):
    frame: SceneFrame
    config: ClickToGoConfig
    session_dir: Path
    lock: threading.Lock
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None

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
        if self.path != "/goal":
            self._send_json(404, {"error": "not found"})
            return

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
    args = parser.parse_args()

    config = ClickToGoConfig(
        port=args.port,
        round_index=args.round,
        localize_every=args.localize_every,
        odom_noise=args.odom_noise,
        device=args.device,
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
