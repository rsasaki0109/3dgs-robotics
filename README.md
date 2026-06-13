# 3DGS Robotics

[![CI](https://github.com/rsasaki0109/3dgs-robotics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rsasaki0109/3dgs-robotics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/rsasaki0109/3dgs-robotics)](LICENSE)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/rsasaki0109/3dgs-robotics)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rsasaki0109/3dgs-robotics/blob/main/notebooks/photos_to_splat_colab.ipynb)

**Real robot logs -> browser 3D Gaussian Splat maps -> robots and LLM agents that work inside them.**

3DGS Robotics turns photos, videos, rosbags, and external SLAM outputs into
browser `.splat` maps, then gives you a robotics stack on top of the same map:
localization, camera simulation, navigation, autonomous exploration,
open-vocabulary queries, multi-robot merging, Isaac Sim export, and an
[MCP server](docs/mcp.md) so LLM agents can operate the map.

**Try it first:** [build your own splat — zero install (HF Spaces)](https://huggingface.co/spaces/rsasaki0109/3dgs-robotics) |
[live 3DGS demo](https://rsasaki0109.github.io/3dgs-robotics/splat.html) |
[large-scale Dynamic Map Viewer](https://rsasaki0109.github.io/3dgs-robotics/dreamwalker/?tileCatalog=%2Fmanifests%2Foutdoor-production-grid-large-tile-catalog.json&tilePreload=metadata&tilePreloadLimit=4&tileResidentLimit=6&robotRoute=%2Frobot-routes%2Foutdoor-production-grid-large-route.json&robotRoutePlayback=1&robotRoutePlaybackMs=1200&robotRoutePlaybackLoop=1)

[![Dynamic map loading over a real rosbag2 3DGS aerial map (Istanbul Bag6)](docs/images/demo-sweep/map-quality.gif)](https://rsasaki0109.github.io/3dgs-robotics/)

Lead GIF: dynamic map loading on real robot data — the base layer is a true top-down (orthographic gsplat) render of the Istanbul Bag6 rosbag2 pilot scene, and the resident (green) / preload (amber) tile window moves along the camera trajectory recovered from the mapped street, so 30 m map tiles light up as the camera drives. Standalone still: `docs/images/demo-sweep/dynamic-map-material.png`.

```bash
pip install -e ".[dev]"
3dgs-robotics video-to-splat my_drive.mp4 --output outputs/my_drive_splat
3dgs-robotics photos-to-splat --images ./my_photos --output outputs/my_splat
```

## Highlights — the map is something your robot uses

Every clip below runs on real driving data (KITTI / rosbag2 sessions) with the
commands in this repo. Map units are camera-height-relative, not calibrated meters.

| | |
| --- | --- |
| **Autonomous exploration** — `explore` picks its own frontier goals and covers 97.9% of the reachable free space.<br>![Autonomous exploration](docs/images/robotics/explore.gif) | **Active mapping** — the robot decides where to look next and the live map grows there, round after round.<br>![Active mapping](docs/images/robotics/active-mapping.gif) |
| **Two robots, one live map** — `merge-live` fuses two live sessions as new rounds arrive, absorbing a 1.8x scale difference.<br>![Live multi-robot merge](docs/images/robotics/live-merge.gif) | **Erase objects by language** — `splat-clean "car"` removes the leading-car ghost from the splat.<br>![Erase objects by language](docs/images/robotics/splat-clean.gif) |
| **One viewer, four axes** — double-click the road to drive there, search to box or glow objects (Semantic), **Erase**/**Grab** to edit, **Diff vs baseline** to inspect changes (Editable + Dynamic), and **Show confidence** to heatmap reconstruction quality (Confidence) — all live on the splat.<br>![Click-to-go in the browser viewer](docs/images/robotics/click-to-go.png) | **Grab & paste between maps** — `splat-grab "car"` cuts the object out of map A and ground-anchors it into map B.<br>![Grab and paste between maps](docs/images/robotics/splat-grab-paste.png) |

More material: [open-vocabulary map inventory](docs/images/robotics/inventory.png) ·
[patrol over detected changes](docs/images/robotics/patrol-trace.png) ·
[robot results overlaid in the browser viewer](docs/images/robotics/viewer-overlay.png) ·
[closed-loop simulate/localize GIF](#robotics-applications--one-map-four-pillars).

## Quickstart — pick your entry point

### Build a map

| What you start with | Minimum command | Deep-dive section |
| --- | --- | --- |
| **A walkaround video** | `3dgs-robotics video-to-splat my_drive.mp4 --output outputs/my_drive_splat` | [Video to splat](#video-to-splat-one-shot-pose-free) |
| **A folder of photos** | `3dgs-robotics photos-to-splat --images ./my_photos --output outputs/my_splat` | [Bring Your Own Photos](#bring-your-own-photos-one-shot-pose-free) |
| **A rosbag from your robot** | `3dgs-robotics map my_drive_bag/` (one-shot) or `python3 scripts/run_live_mapping_demo.py --bag my_drive_bag/ --port 8765` (live replay) — no ROS install needed | [Live Mapping (ROS 2)](#live-mapping-ros-2--watch-the-map-grow) |
| **A live ROS 2 camera topic** | `3dgs-robotics-live-mapper --image-topic /camera/image_raw/compressed --port 8765` | [Live Mapping (ROS 2)](#live-mapping-ros-2--watch-the-map-grow) |
| **External SLAM artifacts** | `python3 scripts/plan_external_slam_imports.py --format shell` then `3dgs-robotics preprocess --method external-slam ...` | [Import External SLAM Results](#import-external-slam-results) |
| **Just a browser** | HF Spaces / Colab badges above | [Zero-install demos](#zero-install-demos-hf-spaces--colab) |

### Use the map — the robot stack

| What you want | Minimum command | Deep-dive section |
| --- | --- | --- |
| **Drive a robot through your map** | `3dgs-robotics navigate --map outputs/live_mapping --output nav/nav_result.json` | [docs/live-mapping.md](docs/live-mapping.md#autonomous-navigation-in-the-map) |
| **Let the robot explore on its own** | `3dgs-robotics explore --map outputs/live_mapping --output explore/explore_result.json` | [docs/live-mapping.md](docs/live-mapping.md#autonomous-exploration-the-robot-picks-its-own-goals) |
| **A simulated camera in your map** | `3dgs-robotics-camera-sim --map outputs/live_mapping --replay --loop` | [docs/live-mapping.md](docs/live-mapping.md#ros-2-gs-camera-simulator-node) |
| **nav2 needs a map** | `3dgs-robotics export-grid --map outputs/live_mapping --output nav2_map/map.yaml` | [docs/live-mapping.md](docs/live-mapping.md#nav2-occupancy-grid-export) |
| **Isaac Sim as the target** | `3dgs-robotics export-isaac --map outputs/live_mapping --output scene.usdz` (+ `export-isaac-route` bakes the nav path into the stage) | [docs/isaac-sim.md](docs/isaac-sim.md) |
| **Two patrols, what changed?** | `3dgs-robotics detect-changes --map-a run1 --map-b run2 --output changes/changes.json` | [docs/live-mapping.md](docs/live-mapping.md#change-detection-inspection) |
| **Go look at what changed** | `3dgs-robotics patrol --map outputs/live_mapping --from-changes changes/changes.json --output patrol/patrol_result.json` | [docs/live-mapping.md](docs/live-mapping.md#inspection-patrol-drive-to-every-stop--or-to-every-change) |
| **Two robots, one map** | `3dgs-robotics merge-maps --map-a run1 --map-b run2 --output merged/merged.ply` (one-shot) or `merge-live` (as they drive) | [docs/live-mapping.md](docs/live-mapping.md#merging-two-maps-collaborative-mapping) |
| **Show robot results in the browser** | `3dgs-robotics export-overlay --map outputs/live_mapping --nav nav/nav_result.json --output overlay.json` then `splat.html?overlay=...` | [docs/live-mapping.md](docs/live-mapping.md#overlaying-robot-results-in-the-browser-viewer) |
| **One browser viewer for all of it** | `3dgs-robotics-click-to-go --map outputs/live_mapping --port 8787 --baseline-round 1` then open the printed viewer URL: search to box objects, **Erase**/**Grab**/**Reset** to edit, **Diff vs baseline** to see changes, **Show confidence** to heatmap quality, double-click the road to drive | [docs/live-mapping.md](docs/live-mapping.md#click-to-go-double-click-the-map-the-robot-drives-there) |
| **Replay a session in rerun.io** | `3dgs-robotics rerun-replay --map outputs/live_mapping` | [docs/live-mapping.md](docs/live-mapping.md#replay-a-session-into-rerun) |
| **Existing splats for policy evaluation** | `python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json` then `3dgs-robotics route-policy-benchmark ...` | [Physical AI benchmark path](#physical-ai-benchmark-path) |

### Talk to the map — language and agents

| What you ask | Minimum command | Deep-dive section |
| --- | --- | --- |
| **"Where is the car?"** | `3dgs-robotics query-map "car" --map outputs/live_mapping --output query/car.json` | [docs/live-mapping.md](docs/live-mapping.md#open-vocabulary-queries-where-is-the-car) |
| **"What is in this map?"** | `3dgs-robotics inventory --map outputs/live_mapping --output inventory/inventory.json` | [docs/live-mapping.md](docs/live-mapping.md#map-inventory-what-is-in-this-map) |
| **"Drive to the car"** | `3dgs-robotics navigate --map outputs/live_mapping --to "car" --output nav/nav_result.json` | [docs/live-mapping.md](docs/live-mapping.md#autonomous-navigation-in-the-map) |
| **"Remove the car from the map"** | `3dgs-robotics splat-clean "car" --map outputs/live_mapping --output clean/no_car.ply` | [docs/live-mapping.md](docs/live-mapping.md#erasing-objects-by-language-remove-the-car) |
| **"Take the car, put it in map B"** | `3dgs-robotics splat-grab "car" --map mapA --output car.ply` then `splat-paste car.ply --map mapB --at 1.0,0.05 --output scene.ply` | [docs/live-mapping.md](docs/live-mapping.md#grab--paste-objects-between-maps-take-the-car-put-it-there) |
| **Let Claude operate the map** | `claude mcp add talk-to-your-map -- 3dgs-robotics-mcp --root outputs/live_mapping` | [Talk to Your Map](#talk-to-your-map--mcp-server), [docs/mcp.md](docs/mcp.md) |

### One map, four research axes — live in the browser

`3dgs-robotics-click-to-go` serves an interactive viewer that puts all four 3DGS
research directions on the **same** served map, with no reload between them:

- **Semantic** — type a prompt in the search box; open-vocabulary hits come back as 3D boxes drawn over the splat, or **Highlight** recolors the matching gaussians to glow inside the splat and dims the rest.
- **Editable** — **Erase** deletes the matching gaussians, **Grab** keeps only them, **Reset** restores the full map. Each edit re-exports a gauge-aligned splat and hot-swaps it into the viewer in place — same camera, no reload.
- **Dynamic** — **Diff vs baseline** runs `detect-changes` against an earlier round (`--baseline-round`) and boxes what appeared (green) / disappeared (orange).
- **Confidence** — **Show confidence** heatmaps the map by how solid each gaussian is, recoloring it by per-gaussian opacity (warm = low confidence, cool = high) so under-reconstructed regions glow red.

…and double-clicking the road still drives the robot there. A single server backs the lot: `/query`, `/highlight`, `/clean`, `/grab`, `/changes`, `/quality`, and `/goal`.

Supervised rosbag pipelines and large-scale tiling: [Outdoor pipeline quickstart](#outdoor-pipeline-quickstart-autoware-leo-drive).

## Talk to Your Map — MCP server

Your 3DGS map becomes a set of tools for LLM agents. The bundled
[MCP](https://modelcontextprotocol.io) server exposes 14 tools — session
discovery, map info, open-vocabulary query and inventory, language-goal
navigation, autonomous exploration, patrol, object erase/grab/paste, map
merging, change detection, viewer overlays, Isaac route export, and rerun
replay — so you can ask Claude to
*"find the car in the parking-lot map, erase it, then drive to the entrance"*
and it chains `query_map` → `splat_clean` → `navigate` for you.

```bash
pip install "3dgs-robotics[mcp]"
claude mcp add talk-to-your-map -- 3dgs-robotics-mcp --root outputs/live_mapping
```

Works with any MCP client (Claude Code, Claude Desktop, …). Tool reference and
client configs: [docs/mcp.md](docs/mcp.md).

## Live Mapping (ROS 2) — watch the map grow

`3dgs-robotics-live-mapper` rebuilds a draft splat as the robot drives; the polling viewer
swaps `live/latest.splat` in place. Full docs: [docs/live-mapping.md](docs/live-mapping.md).

![Live mapping: the 3DGS map grows as the robot drives](docs/images/live-mapping/live-mapping-grow.gif)

Reproduce the growth GIF with `scripts/run_live_mapping_demo.py` + `scripts/build_live_mapping_gif.py`.
Localize query frames against a finished session: `3dgs-robotics localize --map <session> --non-round-keyframes`
(see [docs/live-mapping.md](docs/live-mapping.md#3dgs-localization)).

```bash
3dgs-robotics-live-mapper --image-topic /camera/image_raw/compressed --port 8765       # live ROS 2 topic
3dgs-robotics-localizer --map outputs/live_mapping --follow-latest                       # PoseStamped + TF in the map
3dgs-robotics-camera-sim --map outputs/live_mapping --replay --loop                      # virtual camera in the map
python3 scripts/run_live_mapping_demo.py --images ./my_drive_frames --fps 2 --port 8765 # image-folder replay
python3 scripts/run_live_mapping_demo.py --bag ./my_drive_bag --port 8765               # rosbag replay, no ROS needed
3dgs-robotics rerun-replay --map outputs/live_mapping                                    # session -> rerun.io timeline (.rrd)
```

## Robotics applications — one map, four pillars

![3DGS robotics closed loop: the camera simulator flies the mapped trajectory while the localizer re-estimates its pose over the nav2 grid](docs/images/robotics/robotics-loop.gif)

Everything in this GIF comes from one live-mapping session and closes the
loop inside this repo: the **GS camera simulator** replays the mapped
trajectory (top), the **3DGS localizer** re-estimates each view's pose from
the rendered pixels alone (orange dots vs the green ground-truth trail), and
the backdrop is the **nav2 occupancy grid** exported from the same map.
Reproduce it with `python3 scripts/build_robotics_demo_gif.py --session <session>`.

| Pillar | Command | Docs |
| --- | --- | --- |
| Localization | `3dgs-robotics-localizer --map <session>` (PoseStamped + TF) | [live-mapping.md](docs/live-mapping.md#ros-2-localizer-node) |
| Simulation | `3dgs-robotics-camera-sim --map <session> --replay` / `3dgs-robotics export-isaac` | [live-mapping.md](docs/live-mapping.md#ros-2-gs-camera-simulator-node), [isaac-sim.md](docs/isaac-sim.md) |
| Navigation | `3dgs-robotics export-grid` (nav2 map) / `navigate --to "car"` (language-directed) / `explore` (self-chosen goals) / active mapping (the map grows where the robot decides) | [live-mapping.md](docs/live-mapping.md#nav2-occupancy-grid-export) |
| Perception | `3dgs-robotics detect-changes` / `query-map "car"` / `splat-clean "car"` / `patrol --from-changes` (go look at what changed) | [live-mapping.md](docs/live-mapping.md#change-detection-inspection) |
| Collaboration | `3dgs-robotics merge-live` (two robots, one live map) / `merge-maps` (one-shot) | [live-mapping.md](docs/live-mapping.md#collaborative-live-mapping-multi-robot-merge-live) |

## Live Demo

| URL | Renderer |
| --- | --- |
| [`/splat.html`](https://rsasaki0109.github.io/3dgs-robotics/splat.html) | Default WebGL2 splat viewer |
| [`/splat_spark.html`](https://rsasaki0109.github.io/3dgs-robotics/splat_spark.html) | Spark 2.0 (mobile / WebXR) |
| [`/splat_webgpu.html`](https://rsasaki0109.github.io/3dgs-robotics/splat_webgpu.html) | WebGPU splat viewer |
| [`/`](https://rsasaki0109.github.io/3dgs-robotics/) | Landing page + Physical AI proof |
| [`/dreamwalker/`](https://rsasaki0109.github.io/3dgs-robotics/dreamwalker/) | Dynamic Map Viewer (regional mosaic + Istanbul pilot) |

Production scenes come from `docs/scenes-list.json` (README table, viewers, GIF scripts).

<details>
<summary><b>Production scene gallery (9 public scenes)</b></summary>

| Scene | Preview | Pipeline |
|-------|---------|----------|
| Autoware 6-bag fused (supervised default) | [![](docs/images/demo-sweep/01_outdoor-demo.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/outdoor-demo.splat) | GNSS + LiDAR-seeded COLMAP, gsplat 30–50k |
| bag6 cam0 — DUSt3R pose-free | [![](docs/images/demo-sweep/02_outdoor-demo-dust3r.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/outdoor-demo-dust3r.splat) | 20 frames, DUSt3R, gsplat 3k |
| MCD tuhh_day_04 — DUSt3R pose-free | [![](docs/images/demo-sweep/03_mcd-tuhh-day04.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/mcd-tuhh-day04.splat) | MCD handheld, DUSt3R, gsplat 3k |
| bag6 cam0 — MAST3R pose-free (metric) | [![](docs/images/demo-sweep/04_bag6-mast3r.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/bag6-mast3r.splat) | 20 frames, MAST3R, gsplat 15k |
| bag6 cam0 — VGGT-SLAM 2.0 (15k) | [![](docs/images/demo-sweep/07_bag6-vggt-slam.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/bag6-vggt-slam-20-15k.splat) | VGGT-SLAM 2.0 artifact import, gsplat 15k |
| bag6 cam0 — MASt3R-SLAM (15k) | [![](docs/images/demo-sweep/08_bag6-mast3r-slam.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/bag6-mast3r-slam-20-15k.splat) | MASt3R-SLAM artifact import, gsplat 15k |
| bag6 cam0 — Pi3X (15k) | [![](docs/images/demo-sweep/09_bag6-pi3x.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/bag6-pi3x-20-15k.splat) | Pi3X VO tensor import, gsplat 15k |
| MCD tuhh_day_04 — MAST3R pose-free (metric) | [![](docs/images/demo-sweep/05_mcd-tuhh-day04-mast3r.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/mcd-tuhh-day04-mast3r.splat) | MCD handheld, MAST3R, gsplat 15k |
| MCD ntu_day_02 — supervised | [![](docs/images/demo-sweep/06_mcd-ntu-day02-supervised.png)](https://rsasaki0109.github.io/3dgs-robotics/splat.html?url=assets/outdoor-demo/mcd-ntu-day02-supervised.splat) | Valid GNSS + LiDAR depth-supervised gsplat |

The Autoware supervised default uses the full multi-bag pose-import stack. The MCD supervised row uses `ntu_day_02` because `tuhh_day_04` publishes all-zero GNSS; that rejected zero-GNSS artifact remains documented in `docs/plan_outdoor_gs.md`.

Preview PNGs: `DISPLAY=:0 python3 scripts/capture_readme_splat_previews.py` · proof GIF: `python3 scripts/build_map_quality_gif.py`

</details>

## Video to splat (one-shot, pose-free)

```bash
3dgs-robotics video-to-splat my_drive.mp4 --output outputs/my_drive_splat
3dgs-robotics map my_drive.mp4 --quality balanced --no-open-viewer   # alias
3dgs-robotics map my_drive_bag/ --image-topic /camera/image_raw      # rosbag input (.bag/.db3/.mcap)
```

Optional DUSt3R clone for the default backend; use `--preprocess vggt` with a local
[facebookresearch/vggt](https://github.com/facebookresearch/vggt) clone for faster feedforward reconstruction (distinct from VGGT-SLAM 2.0 artifact import).

## Bring Your Own Photos (one-shot, pose-free)

```bash
3dgs-robotics photos-to-splat --images ./my_photos --output outputs/my_photos_splat --quality draft
3dgs-robotics splat-inspect --input outputs/my_scene.splat
3dgs-robotics splat-filter --input outputs/my_scene.splat --output outputs/my_scene.clean.splat --min-opacity 0.08 --max-scale-percentile 98
```

## Zero-install demos (HF Spaces / Colab)

- **[Hugging Face Space](https://huggingface.co/spaces/rsasaki0109/3dgs-robotics)** — upload photos or a short video in the browser (`apps/hf-space/`).
- **[Colab notebook](https://colab.research.google.com/github/rsasaki0109/3dgs-robotics/blob/main/notebooks/photos_to_splat_colab.ipynb)** — full `photos-to-splat` on a free T4.

## Import External SLAM Results

```bash
python3 scripts/plan_external_slam_imports.py --format shell
3dgs-robotics preprocess --method external-slam --images data/my_scene/images \
  --external-slam-trajectory outputs/slam/poses.txt \
  --external-slam-points outputs/slam/map.ply --output outputs/my_scene_sparse
```

Profiles: MASt3R-SLAM, VGGT-SLAM 2.0, Pi3/Pi3X, LoGeR. Matrix: `docs/plan_outdoor_gs.md`.

## Physical AI benchmark path

```bash
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json
3dgs-robotics route-policy-benchmark --policy-registry runs/scenarios/outdoor-policies.json \
  --goal-suite runs/scenarios/outdoor-goals.json --scene-catalog docs/scenes-list.json \
  --scene-id outdoor-demo --episode-count 16 \
  --output runs/scenarios/outdoor-policy-benchmark.json \
  --markdown-output runs/scenarios/outdoor-policy-benchmark.md
```

Workflow details: `docs/physical-ai-sim.md`.

## Outdoor pipeline quickstart (Autoware Leo Drive)

```bash
3dgs-robotics download --dataset autoware_leo_drive_bag6 --output data/autoware
3dgs-robotics preprocess --method colmap --data data/autoware --output outputs/autoware_sparse
3dgs-robotics train --data outputs/autoware_sparse --method gsplat --iterations 30000
3dgs-robotics export --model outputs/train/point_cloud.ply --format splat --output outputs/autoware.splat
```

Large-scale tiling, real-input staging, and Dynamic Map Viewer promotion:
`docs/large-scale-3dgs-real-run.md` · `3dgs-robotics large-scale-3dgs-bootstrap --help`

## Installation

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[gsplat]"       # training / live mapping / localization
pip install -e ".[app]"          # Streamlit demo (streamlit run app.py)
```

## CLI reference

```bash
3dgs-robotics photos-to-splat --images ./my_photos --output outputs/my_splat
3dgs-robotics video-to-splat my_drive.mp4 --output outputs/my_drive_splat
3dgs-robotics localize --map outputs/live_mapping/session --query frame.jpg
3dgs-robotics download / preprocess / train / export --help
3dgs-robotics-live-mapper --image-topic /camera/image_raw/compressed --port 8765
3dgs-robotics-mcp --root outputs/live_mapping            # MCP server for LLM agents (docs/mcp.md)
3dgs-robotics-click-to-go --map outputs/live_mapping     # browser click -> navigate
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json
```

More: `docs/plan_outdoor_gs.md`, `docs/physical-ai-sim.md`, [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

Wraps DUSt3R, MASt3R, MASt3R-SLAM, VGGT / VGGT-SLAM 2.0, Pi3/Pi3X, LoGeR, gsplat,
nerfstudio, and browser splat viewers. Upstream licenses still apply.

## License

MIT — see [LICENSE](LICENSE).
