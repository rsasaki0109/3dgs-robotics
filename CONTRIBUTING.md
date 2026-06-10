# Contributing to GS Mapper

Thanks for looking at the code. Contributions are welcome — a few practical
notes so your first PR isn't blocked on nit-picks.

## Local setup

```bash
git clone https://github.com/rsasaki0109/3dgs-robotics.git
cd gs-mapper
pip install -e ".[dev]"
```

Optional backends (only if you're touching those code paths):

- **DUSt3R** / **MAST3R** pose-free — `git clone` the upstream repos, set
  `DUST3R_PATH` / `MAST3R_PATH` to the clones, download the checkpoints
  listed in the [Credits](README.md#credits) section.
- **gsplat** training — `pip install -e ".[gsplat]"`. Needs CUDA.
- **nerfstudio** secondary trainer — `pip install -e ".[nerfstudio]"`.
- **COLMAP** — `sudo apt install colmap` (Ubuntu) or `brew install colmap`
  (macOS). Only needed for `gs-mapper preprocess --method colmap`.

## Before you push

```bash
# format + lint
ruff format src/ tests/ scripts/
ruff check src/ tests/ scripts/

# full test suite (skip the e2e directory if it's empty)
pytest tests/ -q --ignore=tests/e2e
```

CI runs the same three commands on every PR against Python 3.10, 3.11, and
3.12. All three suites must go green before a merge. 600+ tests usually
complete in under 90 seconds.

## Branch + PR conventions

- Branch from `main` with a `codex/<short-topic>` prefix (example:
  `codex/scene-picker`). Matches existing branch history and makes it
  obvious the change came through the repo's dev-agent workflow.
- One logical change per PR. Bundling two independent improvements is OK
  if the description explicitly says so.
- `git commit` messages: **do not** add `Co-Authored-By:` trailers unless
  the human maintainer explicitly asks. This repo's commit attribution is
  owner-only.
- `gh pr create --body` bodies: **do not** add "Generated with Claude
  Code" / AI-attribution footers. Same reason — owner-only.
- PR title: imperative, under 72 chars. Example:
  `add MAST3R pose-free backend (metric-aware sibling of DUSt3R)`.

## What goes where

| Kind of change | Landing spot |
|----------------|--------------|
| New CLI subcommand / flag | `src/gs_sim2real/cli.py` + matching handler + a parser test under `tests/test_cli.py` |
| Pose-free backend | `src/gs_sim2real/preprocess/pose_free.py` + a `scripts/run_<name>.py` thin CLI + `tests/test_run_<name>_script.py` smoke |
| External SLAM front-end support | `src/gs_sim2real/preprocess/external_slam_artifacts/` (profile + resolver + manifest) + `tests/test_external_slam*.py` |
| Training / exporter | `src/gs_sim2real/train/` + `src/gs_sim2real/viewer/web_export.py` |
| Physical AI sim contract / env / sensor rig | `src/gs_sim2real/sim/contract.py` + `interfaces.py` + `headless.py` + `rendering.py` + `tests/test_physical_ai_headless_env.py` |
| Sensor noise profile (pose / goal / heading or raw camera / depth / LiDAR) | `src/gs_sim2real/sim/policy_sensor_noise.py` (pose-facing) or `raw_sensor_noise.py` (renderer-facing) + `tests/test_policy_sensor_noise.py` / `tests/test_raw_sensor_noise.py` |
| Dynamic obstacle / multi-agent observation feature | `src/gs_sim2real/sim/policy_dynamic_obstacles.py` + `src/gs_sim2real/sim/gym_adapter.py` obstacle block + `tests/test_policy_dynamic_obstacles.py` |
| Scenario spec / matrix config field | `src/gs_sim2real/sim/policy_scenario_set.py` + `policy_scenario_matrix.py` (add field → JSON round-trip → matrix expansion → shard rebase) + `tests/test_physical_ai_policy_benchmark.py` |
| New bundled demo splat | `docs/assets/outdoor-demo/<name>.splat` + `docs/scenes-list.json` preview entry + README table/thumbnail + `tests/test_pages_assets.py` |
| Viewer change | `docs/splat.html` / `docs/splat_spark.html` / `docs/splat_webgpu.html` + shared `docs/scene-picker.js` when adjusting picker behaviour |
| Outdoor-pipeline current handoff / decision log | `docs/plan_outdoor_gs.md` (full 2026-04 history is linked from there) |
| User-facing quickstart / demo story | `README.md` + `docs/images/demo-sweep/` thumbnails |
| Launch kit copy / topics / links | `src/gs_sim2real/marketing/launch_kit.py` + regenerate `docs/launch-kit.{html,md,json}` via `scripts/generate_launch_kit.py` |

## Bundled demo splats

When you add a new `.splat` under `docs/assets/outdoor-demo/`, three things
need to stay in sync or CI will fail:

1. Register it in `docs/scenes-list.json` with `url`, `preview`, `label`, and
   `summary`.
2. Add the README table row and regenerate the preview PNG with
   `scripts/capture_readme_splat_previews.py`.
3. Keep the pre-populated picker options in `docs/splat.html`,
   `docs/splat_spark.html`, and `docs/splat_webgpu.html` in the same order as
   `docs/scenes-list.json`.
4. Extend `tests/test_pages_assets.py` when the new splat needs a
   scene-specific invariant beyond the shared manifest/picker/README checks.

Shipped splats are capped at the antimatter15 400 000-gauss / 12.8 MB
budget via `gs-mapper export --format splat --max-points 400000` — please
keep new bundled demos under that cap so GitHub Pages bandwidth stays
predictable. The supervised `outdoor-demo.splat` uses a tighter 80 000-gauss
cap because its dense reconstruction already absorbs the supervised signal
and loads faster on mobile.

## Where to start

Good small first PRs:

- A new `.splat` demo from your own photos via
  `gs-mapper photos-to-splat --preprocess mast3r` — follow the "Bundled
  demo splats" checklist above.
- A new pose-free backend (MAST3R was roughly 200 lines in `pose_free.py`
  + the symmetric `scripts/run_mast3r.py`).
- A new `RoutePolicySensorNoiseProfile` or `DynamicObstacleTimeline`
  fixture that plugs into an existing scenario spec — the scenario
  wiring carries the path, the gym adapter surfaces the features, and
  the tests under `tests/test_policy_sensor_noise.py` /
  `tests/test_policy_dynamic_obstacles.py` show the expected shape.
- An external SLAM artifact profile tweak (new candidate filename,
  alias, or trajectory format) in
  `src/gs_sim2real/preprocess/external_slam_artifacts/profiles.py` —
  schema invariants are pinned by `tests/test_external_slam_profiles.py`.
- README / docs clarifications where something is confusing.
- Moving any still-experimental `docs/splat*.html` feature (e.g. LoD
  config, `?cameras=` preset) into the shared `scene-picker.js` hook so
  all three viewers pick it up for free.

## What's intentionally out of scope

See `docs/plan_outdoor_gs.md` "Scope Boundaries" for the "don't touch" list —
renaming the `gs_sim2real` package path, merging the `gs-sim2real` legacy
alias away, large-scale DreamWalker reorgs, antimatter15/splat vendored-code
refactors, etc.

## Reporting issues

Open a GitHub issue with:
- A one-line summary in the title.
- The exact `gs-mapper <subcommand> --help` output if the bug is CLI-
  shaped.
- For pose-free / training bugs: OS, GPU (`nvidia-smi | head -3`),
  PyTorch + CUDA versions, DUSt3R/MAST3R commit hashes if applicable.
- A minimal reproducer (a handful of frames in a `.zip` works for
  pose-free complaints; a 100-line synthetic COLMAP text model works for
  trainer complaints).

Thanks again.
