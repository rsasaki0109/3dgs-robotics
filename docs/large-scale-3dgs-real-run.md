# Large-scale 3DGS Real Input Runbook

This runbook is the handoff from real robot logs or a real COLMAP sparse model
to Dynamic Map Viewer public assets. It intentionally rejects smoke/demo paths by
default so fixture outputs do not get mistaken for a production route.

## 1. Stage Inputs

Use one route root per real capture:

```text
data/large-scale-3dgs-real/
  route_001/
    rosbag2/
      route_001.db3
```

or, after preprocessing:

```text
data/large-scale-3dgs-real/
  route_001_sparse/
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.txt
    images/
```

Keep fixture words such as `smoke`, `fixture`, `sample`, and `demo` out of the
real input path. The preflight gate treats those names as non-production unless
`--allow-fixtures` is passed.

## 2. Gate The Root

Run the real-input gate before launching `bootstrap`:

```bash
python3 scripts/check_large_scale_3dgs_inputs.py \
  data/large-scale-3dgs-real \
  --output outputs/autoware_large \
  --scene-id autoware-large \
  --label "Autoware Large Route" \
  --axes xy \
  --tile-sizes 20,30,50
```

Default real-COLMAP thresholds:

| Field | Default |
| --- | ---: |
| Registered images | 100 |
| Sparse points | 1000 |
| Map extent on a tiling axis | 20 m |
| Raw bag size | 10 MB |

The gate exits `0` for:

- `ready-colmap`: a real COLMAP scene is ready for `large-scale-3dgs-bootstrap`.
- `needs-preprocess`: a real `.bag`, `.db3`, or `.mcap` was found and should be
  converted to COLMAP first.

It exits non-zero for:

- `fixture-only`: inputs were found, but they look like smoke/demo fixtures or
  miss the real-route thresholds.
- `needs-input`: no usable bag or sparse model was found.

For a machine-readable report:

```bash
python3 scripts/check_large_scale_3dgs_inputs.py data/large-scale-3dgs-real --format json
```

For a shell runbook:

```bash
python3 scripts/check_large_scale_3dgs_inputs.py \
  data/large-scale-3dgs-real \
  --output outputs/autoware_large \
  --scene-id autoware-large \
  --format shell > /tmp/autoware-large-3dgs.sh
```

Read the generated shell before executing it. It intentionally pauses in the
comments between pilot and full-route training so the pilot splats can be
inspected first.

## 3. Preprocess If Needed

If the gate reports `needs-preprocess`, run the printed preprocess command. The
standard shape is:

```bash
gs-mapper preprocess \
  --method colmap \
  --data data/large-scale-3dgs-real/route_001/rosbag2 \
  --output outputs/autoware_sparse
```

Then rerun the gate against the same root or against the generated sparse output.

## 4. Bootstrap The Real Route

When the gate reports `ready-colmap`, run the printed bootstrap command:

```bash
gs-mapper large-scale-3dgs-bootstrap \
  --root data/large-scale-3dgs-real \
  --output outputs/autoware_large \
  --axes xy \
  --tile-sizes 20,30,50 \
  --pilot-chunks 6 \
  --route-start-image 0 \
  --write-plan
```

Expected outputs:

```text
outputs/autoware_large/large_scale_3dgs_bootstrap.json
outputs/autoware_large/large_scale_3dgs_preflight.json
outputs/autoware_large/large_scale_3dgs_pilot_plan.json
outputs/autoware_large/large_scale_3dgs_plan.json
```

## 5. Train Pilot First

Train the route-contiguous pilot chunks before launching a full run:

```bash
gs-mapper large-scale-3dgs-run \
  --plan outputs/autoware_large/large_scale_3dgs_pilot_plan.json
```

Inspect the generated splats and run report:

```text
outputs/autoware_large/splats/
outputs/autoware_large/large_scale_3dgs_run_report.json
```

## 6. Train Full Route

After the pilot looks usable:

```bash
gs-mapper large-scale-3dgs-run \
  --plan outputs/autoware_large/large_scale_3dgs_plan.json
```

Resume is enabled by default, so rerunning the command skips existing splats.

## 7. Promote To Dynamic Map Viewer

Stage the generated splats, catalog, route, and launch metadata:

```bash
gs-mapper large-scale-3dgs-promote \
  --plan outputs/autoware_large/large_scale_3dgs_plan.json \
  --run-report outputs/autoware_large/large_scale_3dgs_run_report.json \
  --public-root apps/dreamwalker-web/public \
  --scene-id autoware-large \
  --label "Autoware Large Route" \
  --route-order snake
```

Expected public assets:

```text
apps/dreamwalker-web/public/manifests/autoware-large-tile-catalog.json
apps/dreamwalker-web/public/robot-routes/autoware-large-route.json
apps/dreamwalker-web/public/splats/autoware-large/*.splat
```

## 8. Validate The Viewer Bundle

Run the Dynamic Map Viewer validator:

```bash
npm --prefix apps/dreamwalker-web run validate:dynamic-map-catalog -- \
  public/manifests/autoware-large-tile-catalog.json \
  --public-root public \
  --site-url /dreamwalker/ \
  --preload-mode metadata \
  --route public/robot-routes/autoware-large-route.json \
  --route-playback 1 \
  --route-playback-ms 1200 \
  --route-playback-loop 1
```

A non-rectangular route can emit a grid occupancy warning. That is acceptable
for sparse real routes as long as catalog type, tile files, bounds, route
coverage, and launch URL are `OK`.

## Current Workspace Note

In this checkout, `large-scale-3dgs-discover --root .` currently finds smoke and
fixture COLMAP data plus public splat assets. It does not find a raw production
rosbag or a production COLMAP sparse model. Put real inputs under
`data/large-scale-3dgs-real/` before treating a run as production.
