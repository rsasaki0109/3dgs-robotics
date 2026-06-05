# GS Mapper

[![CI](https://github.com/rsasaki0109/gs-mapper/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rsasaki0109/gs-mapper/actions/workflows/ci.yml)
[![Pages](https://github.com/rsasaki0109/gs-mapper/actions/workflows/pages.yml/badge.svg?branch=main)](https://rsasaki0109.github.io/gs-mapper/)
[![License: MIT](https://img.shields.io/github/license/rsasaki0109/gs-mapper)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Last commit](https://img.shields.io/github/last-commit/rsasaki0109/gs-mapper/main)](https://github.com/rsasaki0109/gs-mapper/commits/main)

**Real outdoor robot logs -> browser 3D Gaussian Splats -> Physical AI scenario CI.**

GS Mapper turns photos, rosbags, and external SLAM outputs into browser-viewable
`.splat` scenes. The same scenes can feed route-policy benchmarks and
reviewable scenario CI artifacts.

**Try it first:** [Open live 3DGS demo](https://rsasaki0109.github.io/gs-mapper/splat.html) |
[Mission Control proof](https://rsasaki0109.github.io/gs-mapper/#mission-control-section) |
[Scenario CI reviews](https://rsasaki0109.github.io/gs-mapper/reviews/) |
[Physical AI docs](docs/physical-ai-sim.md)

[![Inside-map FPS render from actual shipped outdoor 3DGS splats](docs/images/demo-sweep/map-quality.gif)](https://rsasaki0109.github.io/gs-mapper/splat.html)

_Proof view: FPS-style renders from the shipped `.splat` binaries, with a top-down trace showing the camera inside the map._

```bash
git clone https://github.com/rsasaki0109/gs-mapper.git
cd gs-mapper
pip install -e ".[dev]"

# Photos -> .splat -> browser viewer
gs-mapper photos-to-splat --images ./my_photos --output outputs/my_splat

# Public 3DGS scenes -> Physical AI scene catalog
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json
```

What ships:

- Nine public outdoor `.splat` scenes from supervised GNSS/LiDAR, DUSt3R,
  MAST3R, VGGT-SLAM 2.0, MASt3R-SLAM, and Pi3X.
- `photos-to-splat` for image-folder to browser `.splat` runs.
- `splat-inspect` and `splat-filter` for cleaning cloudy browser splats.
- Route-policy benchmark and scenario CI tooling for Physical AI review bundles.

Project notes: [Physical AI sim contract](docs/physical-ai-sim.md),
[outdoor pipeline handoff](docs/plan_outdoor_gs.md), [launch kit](docs/launch-kit.md),
release notes [v0.1.0](docs/releases/v0.1.0.md).

## Quickstart — pick your entry point

| What you start with | Minimum command | Deep-dive section |
| --- | --- | --- |
| **A folder of photos** | `gs-mapper photos-to-splat --images ./my_photos --output outputs/my_splat` | [Bring Your Own Photos](#bring-your-own-photos-one-shot-pose-free) |
| **External SLAM artifacts** | `python3 scripts/plan_external_slam_imports.py --format shell` then `gs-mapper preprocess --method external-slam ...` | [Import External SLAM Results](#import-external-slam-results) |
| **Existing splats for policy evaluation** | `python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json` then `gs-mapper route-policy-benchmark ...` | [Physical AI benchmark path](#physical-ai-benchmark-path) |

Full rosbag -> supervised outdoor splat is covered in
[Outdoor pipeline quickstart](#outdoor-pipeline-quickstart-autoware-leo-drive).
Generic command examples are in [CLI reference](#cli-reference).

## Live Demo

Pages hosts multiple viewers over the same production scene list:

| URL | Renderer | Use it for |
| --- | --- | --- |
| [`/splat.html`](https://rsasaki0109.github.io/gs-mapper/splat.html) | `antimatter15/splat` WebGL2 | Default lightweight splat viewer |
| [`/splat_spark.html`](https://rsasaki0109.github.io/gs-mapper/splat_spark.html) | Spark 2.0 WebGL2 | Mobile, LoD, and WebXR-capable devices |
| [`/splat_webgpu.html`](https://rsasaki0109.github.io/gs-mapper/splat_webgpu.html) | WebGPU splat viewer | GPU-sort viewer for modern browsers |
| [`/`](https://rsasaki0109.github.io/gs-mapper/) | Three.js point viewer | Landing page and Physical AI proof |

The scene picker is defined once in [`docs/scenes-list.json`](docs/scenes-list.json)
and reused by the viewers, README previews, and GIF scripts.

| Scene | Preview | Pipeline |
|-------|---------|----------|
| Autoware 6-bag fused (supervised default) | [![](docs/images/demo-sweep/01_outdoor-demo.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/outdoor-demo.splat) | GNSS + `/tf_static` + LiDAR-seeded COLMAP, image-projected RGB init, gsplat 30-50k iter |
| bag6 cam0 — DUSt3R pose-free | [![](docs/images/demo-sweep/02_outdoor-demo-dust3r.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/outdoor-demo-dust3r.splat) | 20 frames -> DUSt3R pointmap + global align -> gsplat 3k iter |
| MCD tuhh_day_04 — DUSt3R pose-free | [![](docs/images/demo-sweep/03_mcd-tuhh-day04.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/mcd-tuhh-day04.splat) | 20 MCD handheld frames -> DUSt3R -> gsplat 3k iter |
| bag6 cam0 — MAST3R pose-free (metric) | [![](docs/images/demo-sweep/04_bag6-mast3r.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/bag6-mast3r.splat) | 20 frames -> MAST3R sparse global alignment -> gsplat 15k iter |
| bag6 cam0 — VGGT-SLAM 2.0 (15k) | [![](docs/images/demo-sweep/07_bag6-vggt-slam.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/bag6-vggt-slam-20-15k.splat) | VGGT-SLAM 2.0 artifact import -> gsplat 15k iter |
| bag6 cam0 — MASt3R-SLAM (15k) | [![](docs/images/demo-sweep/08_bag6-mast3r-slam.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/bag6-mast3r-slam-20-15k.splat) | MASt3R-SLAM artifact import -> gsplat 15k iter |
| bag6 cam0 — Pi3X (15k) | [![](docs/images/demo-sweep/09_bag6-pi3x.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/bag6-pi3x-20-15k.splat) | Pi3X VO tensor export -> external artifact import -> gsplat 15k iter |
| MCD tuhh_day_04 — MAST3R pose-free (metric) | [![](docs/images/demo-sweep/05_mcd-tuhh-day04-mast3r.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/mcd-tuhh-day04-mast3r.splat) | MCD handheld frames -> MAST3R -> gsplat 15k iter |
| MCD ntu_day_02 — supervised | [![](docs/images/demo-sweep/06_mcd-ntu-day02-supervised.png)](https://rsasaki0109.github.io/gs-mapper/splat.html?url=assets/outdoor-demo/mcd-ntu-day02-supervised.splat) | Valid GNSS + ATV calibration + LiDAR-seeded depth-supervised gsplat |

The Autoware supervised default uses the full multi-bag pose-import stack. The MCD supervised row uses `ntu_day_02` because `tuhh_day_04` publishes all-zero GNSS; that rejected zero-GNSS artifact remains documented in `docs/plan_outdoor_gs.md`.

Regenerate preview images after changing a production `.splat`:

```bash
DISPLAY=:0 python3 scripts/capture_readme_splat_previews.py
python3 scripts/build_map_quality_gif.py
```

## Bring Your Own Photos (one-shot, pose-free)

Drop JPG/PNG frames in and get a browser `.splat` out.

```bash
# Optional: clone DUSt3R if you use the DUSt3R backend.
git clone --recursive https://github.com/naver/dust3r /tmp/dust3r

# Quick draft.
gs-mapper photos-to-splat \
  --images ./my_photos \
  --output outputs/my_photos_splat \
  --quality draft

# Cleaner run with stricter export filtering.
gs-mapper photos-to-splat \
  --images ./my_photos \
  --output outputs/my_photos_splat_clean \
  --quality clean \
  --preprocess mast3r
```

Inspect or clean an existing browser `.splat`:

```bash
gs-mapper splat-inspect --input outputs/my_scene.splat

gs-mapper splat-filter \
  --input outputs/my_scene.splat \
  --output outputs/my_scene.clean.splat \
  --min-opacity 0.08 \
  --max-scale-percentile 98
```

Preview locally from the repo root:

```bash
python -m http.server
# open http://localhost:8000/docs/splat.html?url=<path-to-splat>
```

## Import External SLAM Results

Run heavy front-ends outside this repo, then import their trajectory and points:

```bash
python3 scripts/plan_external_slam_imports.py --format shell

gs-mapper preprocess \
  --method external-slam \
  --images data/my_scene/images \
  --external-slam-trajectory outputs/slam/poses.txt \
  --external-slam-points outputs/slam/map.ply \
  --output outputs/my_scene_sparse
```

Supported profiles include MASt3R-SLAM, VGGT-SLAM 2.0, Pi3/Pi3X, and LoGeR.
See [`docs/plan_outdoor_gs.md`](docs/plan_outdoor_gs.md) for the current
external-SLAM matrix.

## Physical AI benchmark path

Use the public splat scenes as versioned simulation inputs:

```bash
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json

gs-mapper route-policy-benchmark \
  --policy-registry runs/scenarios/outdoor-policies.json \
  --goal-suite runs/scenarios/outdoor-goals.json \
  --scene-catalog docs/scenes-list.json \
  --scene-id outdoor-demo \
  --episode-count 16 \
  --output runs/scenarios/outdoor-policy-benchmark.json \
  --markdown-output runs/scenarios/outdoor-policy-benchmark.md
```

For matrix, shard, workflow, activation, promotion, adoption, and review-bundle
details, use [docs/physical-ai-sim.md](docs/physical-ai-sim.md).

## Outdoor pipeline quickstart (Autoware Leo Drive)

For supervised outdoor reconstruction, use dataset configs and the normal
download -> preprocess -> train -> export chain:

```bash
gs-mapper download --dataset autoware_leo_drive_bag6 --output data/autoware
gs-mapper preprocess --method colmap --data data/autoware --output outputs/autoware_sparse
gs-mapper train --data outputs/autoware_sparse --method gsplat --iterations 30000
gs-mapper export --model outputs/train/point_cloud.ply --format splat --output outputs/autoware.splat
```

For MCD GNSS-seeded runs, first verify non-zero GNSS fixes:

```bash
python3 scripts/check_mcd_gnss.py data/mcd/ntu_day_02 --gnss-topic /vn200/GPS
```

## Installation

```bash
pip install -e ".[dev]"

# Optional backends.
pip install -e ".[gsplat]"
pip install -e ".[nerfstudio]"
pip install -e ".[app]"
```

Streamlit demo:

```bash
streamlit run app.py
```

## CLI reference

```bash
# One-shot photos -> splat.
gs-mapper photos-to-splat --images ./my_photos --output outputs/my_splat

# Step-by-step.
gs-mapper download --dataset mcd --output data/mcd
gs-mapper preprocess --method colmap --data data/mcd --output outputs/sparse
gs-mapper train --data outputs/sparse --method gsplat --iterations 30000
gs-mapper export --model outputs/train/point_cloud.ply --format splat --output outputs/scene.splat

# Generate the Pages scene catalog.
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json
```

More command details live in:

- [docs/physical-ai-sim.md](docs/physical-ai-sim.md)
- [docs/plan_outdoor_gs.md](docs/plan_outdoor_gs.md)
- [docs/launch-kit.md](docs/launch-kit.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Credits

GS Mapper wraps and interoperates with DUSt3R, MASt3R, MASt3R-SLAM,
VGGT-SLAM 2.0, Pi3/Pi3X, LoGeR, gsplat, nerfstudio, antimatter15/splat,
Spark, and the WebGPU splat viewer. Dataset and upstream licenses still apply;
check each source before commercial use.

## License

MIT. See [LICENSE](LICENSE).
