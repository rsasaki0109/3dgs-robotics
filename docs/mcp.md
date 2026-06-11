# Talk to Your Map — MCP server

`3dgs-robotics-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io)
stdio server that turns a live-mapping session into tools an LLM agent can call.
It is thin wiring over the existing CLI: heavy tools shell out to
`3dgs-robotics query-map / navigate / splat-clean / detect-changes / export-overlay`
and return the JSON those commands already write. No new reconstruction logic.

```
You: "Find the car in the KITTI map, erase it, then drive to where it was."
Agent: query_map("car") -> splat_clean("car") -> navigate(goal_xy=...) -> export_overlay(...)
```

## Setup

```bash
pip install -e ".[mcp]"        # or: pip install "3dgs-robotics[mcp]"
```

**Claude Code**

```bash
claude mcp add talk-to-your-map -- 3dgs-robotics-mcp --root outputs/live_mapping
```

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "talk-to-your-map": {
      "command": "3dgs-robotics-mcp",
      "args": ["--root", "/path/to/outputs/live_mapping"]
    }
  }
}
```

`--root` is the directory scanned by `list_map_sessions` when the agent does not
pass one explicitly. Every other tool takes the session directory as `map_dir`.

## Tools

| Tool | Wraps | Returns |
| --- | --- | --- |
| `list_map_sessions` | in-process scan | sessions under root: keyframe/round counts, last successful round, `live/latest.splat` presence |
| `map_info` | in-process scan | rounds, resolved round, artifact paths for one session |
| `query_map` | `query-map` | up to 10 open-vocabulary 3D hits + a ready-to-use `navigate` suggestion for the best hit |
| `navigate` | `navigate` | nav summary (reached / steps / cross-track stats), trace PNG, optional GIF; goal via `to` (language), `goal_xy`, or `goal_keyframe` |
| `explore` | `explore` | autonomous frontier exploration — the robot picks its own goals; coverage summary, trace PNG, optional GIF |
| `patrol` | `patrol` | multi-stop inspection patrol; pass `detect_changes`' output_json as `from_changes` and the robot drives to each change |
| `splat_clean` | `splat-clean` | cleaned PLY + preview paths |
| `splat_grab` | `splat-grab` | a language-selected object as a standalone splat + gauge sidecar |
| `splat_paste` | `splat-paste` | object placed into a target map (auto gauge scale, grounded, `--yaw`) |
| `merge_maps` | `merge-maps` | one merged PLY in map A's gauge (collaborative mapping) |
| `detect_changes` | `detect-changes` | alignment info, appeared/disappeared counts, top 10 clusters |
| `export_overlay` | `export-overlay` | overlay JSON for the browser viewer (`splat.html?url=...&overlay=...`) |
| `export_isaac_route` | `export-isaac-route` | USD layer with nav paths/goals/hits in the USDZ's frame (Isaac Sim / usdview) |

Outputs land under `<session>/mcp/` with timestamped names, so agent runs never
overwrite each other or the session's own artifacts.

## Notes

- Distances and coordinates are in the map's reconstruction gauge (typically
  camera-height units), **not meters**, unless the session was mapped with
  metric poses — the same caveat as the underlying CLI.
- `query_map`, `navigate`, `splat_clean`, and `detect_changes` need the same
  optional dependencies as their CLI counterparts (`transformers`, gsplat, a
  CUDA device by default; pass `device="cpu"` to trade speed for portability).
- The server itself is pure Python (no torch/rclpy import) and starts instantly;
  the heavy imports happen inside the subprocess per call.
