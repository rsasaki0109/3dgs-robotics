#!/usr/bin/env python3
"""Plot live-mapping loop candidates as edges over the keyframe trajectory.

Revisit detection v1 (``live/loop_candidates.json``) records "temporally
distant but visually near" keyframe pairs without correcting the map. This
plot is how detection quality gets judged before the round-level pose graph
(Step 3) consumes the candidates: real loops show as short edges connecting
trajectory segments that pass the same place; false positives show as long
chords across the map.

    python3 scripts/plot_loop_candidates.py \
        --session outputs/live_mapping_demo \
        --output docs/images/live-mapping/loop-candidates.png

The trajectory comes from the last successful round's COLMAP poses, projected
onto its two dominant axes (the round's own gauge — pose-free maps are not
metric). Candidates whose keyframes were not part of that round are snapped to
the nearest mapped keyframe index.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gs_sim2real.robotics.gauge_alignment import parse_images_txt  # noqa: E402

_KF_RE = re.compile(r"kf_(\d+)")


def find_anchor_round(session: Path) -> Path:
    """Last round that produced both poses and a published splat."""
    candidates = []
    for round_dir in sorted(session.glob("rounds/round_*")):
        images_txt = round_dir / "sparse_input" / "sparse" / "0" / "images.txt"
        if images_txt.is_file() and (round_dir / "scene.splat").is_file():
            candidates.append(images_txt)
    if not candidates:
        raise SystemExit(f"no successful rounds with poses under {session}/rounds")
    return candidates[-1]


def load_trajectory(images_txt: Path) -> tuple[np.ndarray, np.ndarray]:
    """(sorted keyframe indices, (N, 2) top-down projected centers)."""
    names, centers, _rotations = parse_images_txt(images_txt)
    indices = []
    for name in names:
        match = _KF_RE.search(name)
        indices.append(int(match.group(1)) if match else -1)
    order = np.argsort(indices)
    indices = np.asarray(indices)[order]
    centers = centers[order]
    centered = centers - centers.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered)  # dominant plane of the drive
    return indices, centered @ vt[:2].T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Live mapping session workdir (with live/ and rounds/)")
    parser.add_argument("--output", default=None, help="Output PNG (default: <session>/live/loop_candidates.png)")
    args = parser.parse_args()

    session = Path(args.session)
    candidates_path = session / "live" / "loop_candidates.json"
    candidates = []
    if candidates_path.is_file():
        candidates = json.loads(candidates_path.read_text(encoding="utf-8")).get("loopCandidates", [])

    images_txt = find_anchor_round(session)
    indices, xy = load_trajectory(images_txt)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xy[:, 0], xy[:, 1], color="#4f9dd0", linewidth=2, zorder=1, label="keyframe trajectory")
    ax.scatter(xy[:, 0], xy[:, 1], s=12, color="#2c6e9e", zorder=2)
    ax.scatter(*xy[0], marker="^", s=90, color="#3cb371", zorder=3, label="start")
    ax.scatter(*xy[-1], marker="s", s=70, color="#d08f4f", zorder=3, label="end")

    def nearest_point(keyframe_index: int) -> np.ndarray:
        return xy[int(np.argmin(np.abs(indices - keyframe_index)))]

    for i, cand in enumerate(candidates):
        a = nearest_point(int(cand["queryIndex"]))
        b = nearest_point(int(cand["matchIndex"]))
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#d04f4f",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label="loop candidate" if i == 0 else None,
        )

    ax.set_title(f"loop candidates: {len(candidates)} (round poses: {images_txt.parent.parent.parent.name})")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("drive axis (round gauge, not metric)")
    ax.set_ylabel("cross axis")
    fig.tight_layout()

    output = Path(args.output) if args.output else session / "live" / "loop_candidates.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    print(json.dumps({"candidates": len(candidates), "keyframesWithPoses": int(len(indices)), "output": str(output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
