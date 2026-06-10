"""Tests for scripts/check_large_scale_3dgs_inputs.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_large_scale_3dgs_inputs.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_large_scale_3dgs_inputs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sparse_fixture(root: Path, *, image_count: int = 4, point_count: int = 6, spacing: float = 12.0) -> Path:
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 400 400 320 240\n",
        encoding="utf-8",
    )
    image_lines = ["# Image list"]
    for index in range(image_count):
        center_x = spacing * index
        name = f"route_{index:03d}.jpg"
        image_lines.append(f"{index + 1} 1 0 0 0 {-center_x:.6f} 0 0 1 {name}")
        image_lines.append("")
        (images / name).write_bytes(b"jpg")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = ["# Point list"]
    for index in range(point_count):
        point_lines.append(f"{index + 1} {index * spacing:.6f} 0 0 255 0 0 0")
    (sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return root


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111


def test_script_passes_python_syntax_check() -> None:
    result = subprocess.run(["python3", "-m", "py_compile", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_real_colmap_scene_passes_gate(tmp_path: Path, capsys) -> None:
    module = _load_script_module()
    root = tmp_path / "real_route"
    _write_sparse_fixture(root / "autoware_sparse")

    rc = module.main(
        [
            str(root),
            "--min-images",
            "3",
            "--min-points",
            "5",
            "--min-extent-m",
            "20",
            "--scene-id",
            "autoware-real",
            "--format",
            "json",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["status"] == "ready-colmap"
    assert report["summary"]["acceptedColmapSceneCount"] == 1
    assert report["accepted"]["colmapScenes"][0]["registeredImageCount"] == 4
    assert "large-scale-3dgs-bootstrap" in report["commands"]["bootstrap"]
    assert "--scene-id autoware-real" in report["commands"]["promote"]


def test_real_bag_input_requests_preprocess(tmp_path: Path, capsys) -> None:
    module = _load_script_module()
    bag_dir = tmp_path / "real_logs" / "rosbag2"
    bag_dir.mkdir(parents=True)
    (bag_dir / "route.db3").write_bytes(b"x" * 128)

    rc = module.main([str(tmp_path / "real_logs"), "--min-bag-bytes", "64", "--format", "json"])

    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["status"] == "needs-preprocess"
    assert report["summary"]["acceptedBagInputCount"] == 1
    assert report["commands"]["preprocess"].startswith("3dgs-robotics preprocess --method colmap")
    assert "large-scale-3dgs-bootstrap" in report["commands"]["bootstrap"]


def test_smoke_fixture_is_rejected_by_default(tmp_path: Path, capsys) -> None:
    module = _load_script_module()
    root = tmp_path / "smoke_route"
    _write_sparse_fixture(root / "smoke_sparse")

    rc = module.main(
        [
            str(root),
            "--min-images",
            "1",
            "--min-points",
            "1",
            "--min-extent-m",
            "1",
            "--format",
            "json",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report["status"] == "fixture-only"
    assert report["rejected"]["colmapScenes"][0]["reason"] == "fixture-path"


def test_shell_output_contains_ordered_runbook(tmp_path: Path, capsys) -> None:
    module = _load_script_module()
    root = tmp_path / "real_route"
    _write_sparse_fixture(root / "autoware_sparse")

    rc = module.main(
        [
            str(root),
            "--min-images",
            "3",
            "--min-points",
            "5",
            "--min-extent-m",
            "20",
            "--output",
            "outputs/autoware_large",
            "--format",
            "shell",
        ]
    )

    shell = capsys.readouterr().out
    assert rc == 0
    assert shell.startswith("#!/usr/bin/env bash")
    assert "large-scale-3dgs-bootstrap" in shell
    assert "large-scale-3dgs-run --plan outputs/autoware_large/large_scale_3dgs_pilot_plan.json" in shell
    assert "large-scale-3dgs-promote" in shell
