"""Smoke test for scripts/robotics_smoke.py.

Runs the fixture path (no trained PLY required) end to end through
``HeadlessSplatRenderer`` and the Dynamic Map Viewer topic map, verifying that the
written artifacts look like a real rendered frame + a well-shaped query
payload.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "robotics_smoke.py"


def test_robotics_smoke_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "robotics_smoke.py should be executable"


def test_robotics_smoke_fixture_end_to_end(tmp_path: Path) -> None:
    """Full fixture run writes rgb.png + depth.npy + payload.json."""
    pytest.importorskip("PIL")
    out_dir = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", "--out", str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    assert "robotics_smoke: rendered" in result.stdout

    rgb_path = out_dir / "rgb.png"
    depth_path = out_dir / "depth.npy"
    payload_path = out_dir / "payload.json"
    assert rgb_path.is_file(), "rgb.png missing"
    assert depth_path.is_file(), "depth.npy missing"
    assert payload_path.is_file(), "payload.json missing"

    with Image.open(rgb_path) as img:
        arr = np.asarray(img)
    assert arr.shape == (120, 160, 3)
    assert (arr.sum(axis=-1) > 0).any(), "RGB is entirely background"

    depth = np.load(depth_path)
    assert depth.shape == (120, 160)
    assert np.isfinite(depth).any(), "depth has no finite values"

    bundle = json.loads(payload_path.read_text(encoding="utf-8"))
    assert bundle["payload"]["type"] == "dreamwalker-render-query/v1"
    assert bundle["payload"]["resolution"] == {"width": 160, "height": 120}
    # Topic map should expose the Dynamic Map Viewer relay topics the bridge subscribes to.
    tm = bundle["topic_map"]
    assert tm["namespace"] == "/dreamwalker"
    assert tm["camera_compressed"].startswith("/dreamwalker/")
    assert tm["depth_image"].startswith("/dreamwalker/")
    assert tm["robot_pose_stamped"].startswith("/dreamwalker/")
    # Non-zero pixels reported matches what we assert from the image itself.
    assert bundle["frame_stats"]["rgb_non_background_pixels"] == int((arr.sum(axis=-1) > 0).sum())


def test_robotics_smoke_rejects_no_input(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path / "x")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--ply" in result.stderr or "--fixture" in result.stderr
