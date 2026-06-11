"""MCP stdio server for conversational map control ("Talk to Your Map").

Exposes the robotics map tools (query-map / navigate / splat-clean / detect-changes /
export-overlay) to LLM agents over the Model Context Protocol. Heavy tools shell out to
the existing `3dgs-robotics` CLI and read back the JSON it writes — no new
reconstruction logic lives here.

Claude Code example: claude mcp add talk-to-your-map -- 3dgs-robotics-mcp
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_DEFAULT_ROOT = "outputs/live_mapping"


def _round_number(path: Path) -> int | None:
    match = re.fullmatch(r"round_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _round_dirs(session_dir: Path) -> list[Path]:
    rounds_dir = session_dir / "rounds"
    if not rounds_dir.is_dir():
        return []
    return sorted(
        (path for path in rounds_dir.iterdir() if path.is_dir() and _round_number(path) is not None),
        key=lambda path: _round_number(path) or -1,
    )


def _keyframe_count(session_dir: Path) -> int:
    keyframes_dir = session_dir / "keyframes"
    if not keyframes_dir.is_dir():
        return 0
    return sum(1 for path in keyframes_dir.iterdir() if path.is_file())


def _is_session(path: Path) -> bool:
    return (path / "keyframes").is_dir()


def _last_successful_round(session_dir: Path) -> int | None:
    state_path = session_dir / "live" / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        last = state.get("lastSuccessfulRound") or {}
        round_index = last.get("round") or state.get("completedRounds")
        if round_index:
            return int(round_index)

    rounds = _round_dirs(session_dir)
    if not rounds:
        return None
    return _round_number(rounds[-1])


def _resolve_round(session_dir: Path, round_index: int | None = None) -> int:
    if round_index is not None:
        return round_index
    resolved = _last_successful_round(session_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"no trained rounds found under {session_dir / 'rounds'}; run live mapping/training first, "
            "or pass a session directory that contains rounds/round_NNN"
        )
    return resolved


def _artifact_paths(session_dir: Path, round_index: int) -> dict[str, str]:
    round_dir = session_dir / "rounds" / f"round_{round_index:03d}"
    return {
        "point_cloud": str(round_dir / "train" / "point_cloud.ply"),
        "images_txt": str(round_dir / "sparse_input" / "sparse" / "0" / "images.txt"),
    }


def _ensure_session(map_dir: str | Path) -> Path:
    session_dir = Path(map_dir)
    if not _is_session(session_dir):
        raise FileNotFoundError(
            f"{session_dir} is not a live-mapping session: expected a keyframes/ directory. "
            "Use list_map_sessions to discover valid sessions, or pass a session directory directly."
        )
    return session_dir


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _slug(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", prompt.strip().lower()).strip("-")
    return slug[:48] or "query"


def _mcp_out_dir(map_dir: str | Path) -> Path:
    out_dir = Path(map_dir) / "mcp"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview_for(path: Path) -> str:
    return str(path.with_suffix(".png"))


def _tail(text: str, lines: int) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def list_map_sessions(root: str | None = None) -> dict[str, Any]:
    """List live-mapping sessions under root (default: the server's --root).

    A session is any directory containing keyframes/. The root itself may be a session. Distances reported by related
    tools are in the map reconstruction gauge, usually camera-height units, not meters unless mapped with metric poses.
    """
    root_path = Path(root or _DEFAULT_ROOT)
    if not root_path.exists():
        return {
            "sessions": [],
            "hint": f"{root_path} does not exist; run live mapping first or pass the directory containing sessions.",
        }

    candidates = [root_path] if _is_session(root_path) else []
    if not candidates and root_path.is_dir():
        candidates = [path for path in root_path.iterdir() if path.is_dir() and _is_session(path)]

    if not candidates:
        return {
            "sessions": [],
            "hint": f"no live-mapping sessions found under {root_path}; expected directories containing keyframes/.",
        }

    sessions = []
    for session_dir in sorted(candidates, key=lambda path: path.name):
        rounds = _round_dirs(session_dir)
        sessions.append(
            {
                "name": session_dir.name,
                "path": str(session_dir),
                "keyframe_count": _keyframe_count(session_dir),
                "round_count": len(rounds),
                "last_successful_round": _last_successful_round(session_dir),
                "has_latest_splat": (session_dir / "live" / "latest.splat").is_file(),
            }
        )
    return {"sessions": sessions}


def map_info(map_dir: str) -> dict[str, Any]:
    """Return details for one live-mapping session.

    Artifact paths point at the trained point cloud and COLMAP images.txt for the resolved round. Coordinates and
    distances used by downstream tools are in map-gauge camera-height units, not meters.
    """
    session_dir = _ensure_session(map_dir)
    rounds = [index for index in (_round_number(path) for path in _round_dirs(session_dir)) if index is not None]
    resolved_round = _resolve_round(session_dir)
    return {
        "name": session_dir.name,
        "path": str(session_dir),
        "keyframe_count": _keyframe_count(session_dir),
        "rounds": rounds,
        "resolved_round": resolved_round,
        "artifacts": _artifact_paths(session_dir, resolved_round),
        "has_latest_splat": (session_dir / "live" / "latest.splat").is_file(),
    }


def query_map(
    map_dir: str,
    prompt: str,
    threshold: float = 0.4,
    round_index: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Query a 3DGS map with open-vocabulary language and return up to 10 3D hits.

    Hit centroids, extents, and navigation goals are in the map reconstruction gauge, typically camera-height units, not
    meters. If there are no hits, retry with a lower threshold or a more concrete object phrase.
    """
    session_dir = _ensure_session(map_dir)
    out_path = _mcp_out_dir(session_dir) / f"query_{_slug(prompt)}_{_timestamp()}.json"
    args = [
        "query-map",
        prompt,
        "--map",
        str(session_dir),
        "--output",
        str(out_path),
        "--threshold",
        str(threshold),
        "--device",
        device,
    ]
    if round_index is not None:
        args.extend(["--round", str(round_index)])
    _run_cli(args)

    payload = _json(out_path)
    hits = list(payload.get("hits") or [])
    suggestion: Any = None
    if hits:
        goal_xy = hits[0].get("goal_xy")
        if goal_xy is not None:
            suggestion = {
                "tool": "navigate",
                "arguments": {"map_dir": str(session_dir), "goal_xy": goal_xy, "device": device},
            }
    else:
        suggestion = f'No hits for "{prompt}" at threshold {threshold}; try lowering threshold.'

    return {
        "prompt": payload.get("prompt", prompt),
        "hits": hits[:10],
        "hit_count": len(hits),
        "preview_png": _preview_for(out_path),
        "output_json": str(out_path),
        "navigate_suggestion": suggestion,
    }


def navigate(
    map_dir: str,
    to: str | None = None,
    goal_xy: list[float] | None = None,
    goal_keyframe: int | None = None,
    gif: bool = False,
    round_index: int | None = None,
    device: str = "cuda",
    max_steps: int = 3000,
) -> dict[str, Any]:
    """Drive the simulated robot to a language goal, grid x/y point, or keyframe.

    Specify exactly one of to, goal_xy, or goal_keyframe. Goals, path statistics, and cross-track errors are in map-gauge
    camera-height units, not meters unless the reconstruction was built with metric poses.
    """
    goals = [to is not None, goal_xy is not None, goal_keyframe is not None]
    if sum(goals) != 1:
        raise ValueError("specify exactly one navigation goal: to, goal_xy, or goal_keyframe.")

    session_dir = _ensure_session(map_dir)
    out_path = _mcp_out_dir(session_dir) / f"nav_{_timestamp()}.json"
    args = [
        "navigate",
        "--map",
        str(session_dir),
        "--output",
        str(out_path),
        "--max-steps",
        str(max_steps),
        "--device",
        device,
    ]
    if round_index is not None:
        args.extend(["--round", str(round_index)])
    if to is not None:
        args.extend(["--to", to])
    elif goal_xy is not None:
        if len(goal_xy) != 2:
            raise ValueError("goal_xy must contain exactly two numbers: [x, y] in map-gauge grid units.")
        args.extend(["--goal", f"{goal_xy[0]},{goal_xy[1]}"])
    elif goal_keyframe is not None:
        args.extend(["--goal-keyframe", str(goal_keyframe)])

    gif_path = out_path.with_suffix(".gif") if gif else None
    if gif_path is not None:
        args.extend(["--gif", str(gif_path)])

    _run_cli(args)
    payload = _json(out_path)
    summary_keys = (
        "reached",
        "steps",
        "localization_fixes",
        "camera_height",
        "cross_track_median",
        "cross_track_max",
        "cross_track_median_camera_heights",
        "goal",
        "note",
    )
    summary = {key: payload[key] for key in summary_keys if key in payload}
    summary.update({"output_json": str(out_path), "trace_png": _preview_for(out_path)})
    if gif_path is not None:
        summary["gif"] = str(gif_path)
    return summary


def splat_clean(
    map_dir: str,
    prompt: str,
    threshold: float = 0.5,
    round_index: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Erase language-described objects from a splat and return the cleaned PLY path.

    Thresholding and dilation happen in the existing CLI. Distances are in the reconstruction gauge, usually
    camera-height units. If nothing is removed, retry with a lower threshold or a more specific prompt.
    """
    session_dir = _ensure_session(map_dir)
    out_path = _mcp_out_dir(session_dir) / f"cleaned_{_slug(prompt)}_{_timestamp()}.ply"
    args = [
        "splat-clean",
        prompt,
        "--map",
        str(session_dir),
        "--output",
        str(out_path),
        "--threshold",
        str(threshold),
        "--device",
        device,
    ]
    if round_index is not None:
        args.extend(["--round", str(round_index)])
    proc = _run_cli(args)
    return {
        "output_ply": str(out_path),
        "preview_png": _preview_for(out_path),
        "stdout_tail": _tail(proc.stdout, 10),
    }


def detect_changes(
    map_a: str,
    map_b: str | None = None,
    round_a: int | None = None,
    round_b: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Diff two maps or two rounds and summarize appeared/disappeared clusters.

    Cluster centroids and voxel sizes are in the maps' reconstruction gauge, commonly camera-height units, not meters.
    Comparing a session with itself needs round_a/round_b (or a different map_b).
    """
    session_a = _ensure_session(map_a)
    out_path = _mcp_out_dir(session_a) / f"changes_{_timestamp()}.json"
    args = ["detect-changes", "--map-a", str(session_a), "--output", str(out_path), "--device", device]
    if map_b is not None:
        args.extend(["--map-b", str(_ensure_session(map_b))])
    if round_a is not None:
        args.extend(["--round-a", str(round_a)])
    if round_b is not None:
        args.extend(["--round-b", str(round_b)])

    _run_cli(args)
    payload = _json(out_path)
    appeared = list(payload.get("appeared") or [])
    disappeared = list(payload.get("disappeared") or [])
    clusters = list(payload.get("clusters") or appeared + disappeared)
    return {
        "alignment": payload.get("alignment"),
        "appeared_count": len(appeared),
        "disappeared_count": len(disappeared),
        "clusters": clusters[:10],
        "cluster_count": len(clusters),
        "output_json": str(out_path),
        "preview_png": _preview_for(out_path),
    }


def export_overlay(
    map_dir: str,
    nav_json: str | None = None,
    query_json: str | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Export a browser-viewer overlay JSON for splat.html.

    The overlay draws navigation paths, query hits, and the mapped trajectory in map-gauge camera-height units. Serve
    the splat and overlay next to docs/ (e.g. python3 -m http.server) and open splat.html?url=...&overlay=... in a
    browser.
    """
    session_dir = _ensure_session(map_dir)
    out_path = _mcp_out_dir(session_dir) / f"overlay_{_timestamp()}.json"
    args = ["export-overlay", "--map", str(session_dir), "--output", str(out_path)]
    if round_index is not None:
        args.extend(["--round", str(round_index)])
    if nav_json is not None:
        args.extend(["--nav", nav_json])
    if query_json is not None:
        args.extend(["--query", query_json])

    proc = _run_cli(args)
    return {
        "overlay_json": str(out_path),
        "usage_hint": f"splat.html?url=<url of the splat>&overlay=<url of {out_path.name}>",
        "stdout_tail": _tail(proc.stdout, 5),
    }


def explore(
    map_dir: str,
    sensor_range: float = 4.0,
    coverage_target: float = 0.95,
    max_goals: int = 30,
    gif: bool = False,
    round_index: int | None = None,
    device: str = "cuda",
    localize_every: int = 0,
) -> dict[str, Any]:
    """Autonomously explore reachable free space in a static 3DGS occupancy map.

    The robot chooses its own frontier goals until coverage is met or no useful frontier remains; use navigate for a
    specific destination. Distances are in map-gauge camera-height units, not meters unless the reconstruction was built
    with metric poses.
    """
    session_dir = _ensure_session(map_dir)
    out_path = _mcp_out_dir(session_dir) / f"explore_{_timestamp()}.json"
    args = [
        "explore",
        "--map",
        str(session_dir),
        "--output",
        str(out_path),
        "--device",
        device,
        "--localize-every",
        str(localize_every),
    ]
    if round_index is not None:
        args.extend(["--round", str(round_index)])
    if sensor_range != 4.0:
        args.extend(["--sensor-range", str(sensor_range)])
    if coverage_target != 0.95:
        args.extend(["--coverage-target", str(coverage_target)])
    if max_goals != 30:
        args.extend(["--max-goals", str(max_goals)])

    gif_path = out_path.with_suffix(".gif") if gif else None
    if gif_path is not None:
        args.extend(["--gif", str(gif_path)])

    _run_cli(args)
    payload = _json(out_path)
    payload.pop("coverage_history", None)
    summary = dict(payload)
    summary["output_json"] = str(out_path)
    summary["trace_png"] = _preview_for(out_path)
    if gif_path is not None:
        summary["gif"] = str(gif_path)
    return summary


def build_server(root: str = _DEFAULT_ROOT) -> Any:
    """Build the talk-to-your-map FastMCP server and register all tools."""
    global _DEFAULT_ROOT
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise ImportError('Talk to Your Map MCP needs the optional SDK: pip install "3dgs-robotics[mcp]"') from error

    _DEFAULT_ROOT = root
    server = FastMCP("talk-to-your-map")
    for tool in (
        list_map_sessions,
        map_info,
        query_map,
        navigate,
        explore,
        splat_clean,
        detect_changes,
        export_overlay,
    ):
        server.add_tool(tool)
    return server


def main() -> None:
    """Run the MCP server over stdio."""
    parser = argparse.ArgumentParser(description="Talk to Your Map MCP stdio server")
    parser.add_argument("--root", default=_DEFAULT_ROOT, help="Default live-mapping sessions root")
    args = parser.parse_args()
    build_server(root=args.root).run()


if __name__ == "__main__":
    main()
