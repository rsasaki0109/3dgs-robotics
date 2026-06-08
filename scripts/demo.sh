#!/usr/bin/env bash
# End-to-end demo: images -> 3DGS splat -> Dynamic Map Viewer robot teleop
set -euo pipefail
IMAGES="${1:?Usage: ./scripts/demo.sh <image_dir> [iterations]}"
ITERS="${2:-1000}"
exec gs-sim2real demo --images "$IMAGES" --iterations "$ITERS"
