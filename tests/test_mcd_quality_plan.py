"""Tests for reproducible MCD quality-run planning."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from gs_sim2real.experiments.mcd_quality_plan import (
    MCDQualityGatePolicy,
    MCDQualityPlanContext,
    build_mcd_quality_plan,
    collect_mcd_quality_results,
    evaluate_mcd_quality_gates,
    render_quality_benchmark_markdown,
    render_quality_gate_markdown,
    render_quality_report_json,
    render_quality_report_markdown,
    render_plan_json,
    render_plan_markdown,
    render_plan_shell,
)


def test_mcd_quality_plan_contains_baseline_and_quality_candidates() -> None:
    plan = build_mcd_quality_plan()

    names = [run.profile.name for run in plan.runs]

    assert names == [
        "ntu_day02_single_400_depth_long",
        "ntu_day02_single_800_ba",
        "ntu_day02_multi_2cam_300each_ba",
    ]
    assert plan.preflight_command[:3] == ("python3", "scripts/check_mcd_gnss.py", "data/mcd/ntu_day_02")
    assert "--flatten-altitude" in plan.preflight_command
    assert "--start-offset-sec" in plan.preflight_command


def test_single_camera_command_reproduces_known_good_trimmed_baseline() -> None:
    plan = build_mcd_quality_plan()
    baseline = plan.runs[0]

    command = list(baseline.preprocess_command)

    assert command[command.index("--image-topic") + 1] == "/d455b/color/image_raw"
    assert command[command.index("--mcd-camera-frame") + 1] == "d455b_color"
    assert command[command.index("--max-frames") + 1] == "400"
    assert command[command.index("--every-n") + 1] == "14"
    assert command[command.index("--mcd-start-offset-sec") + 1] == "35"
    assert baseline.train_command[baseline.train_command.index("--config") + 1] == "configs/training_depth_long.yaml"
    assert baseline.export_command[baseline.export_command.index("--max-points") + 1] == "400000"


def test_multicamera_profile_uses_topic_list_and_yaml_frames() -> None:
    plan = build_mcd_quality_plan()
    multi = plan.runs[2]
    command = list(multi.preprocess_command)

    assert multi.profile.requires_full_folder is True
    assert command[command.index("--image-topic") + 1] == ("/d455b/color/image_raw,/d435i/color/image_raw")
    assert "--mcd-camera-frame" not in command
    assert multi.train_command[multi.train_command.index("--config") + 1] == "configs/training_ba.yaml"


def test_mcd_quality_plan_renders_json_markdown_and_shell() -> None:
    plan = build_mcd_quality_plan(
        MCDQualityPlanContext(output_root="outputs/q", asset_dir="outputs/q/assets", pythonpath="src")
    )

    payload = json.loads(render_plan_json(plan))
    markdown = render_plan_markdown(plan)
    shell = render_plan_shell(plan)

    assert payload["runs"][0]["preprocessDir"] == "outputs/q/ntu_day02_single_400_depth_long/preprocess"
    assert "MCD Quality Run Plan" in markdown
    assert "Single D455B 800 BA" in markdown
    assert "Two-Camera 300 Each BA" in markdown
    assert "PYTHONPATH=src python3 -m gs_sim2real.cli preprocess" in shell
    assert "outputs/q/assets/ntu_day02_single_400_depth_long.splat" in shell


def test_plan_mcd_quality_runs_script_can_emit_single_profile_json() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_mcd_quality_runs.py",
            "--format",
            "json",
            "--profile",
            "ntu_day02_single_800_ba",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert [run["name"] for run in payload["runs"]] == ["ntu_day02_single_800_ba"]
    assert payload["runs"][0]["iterations"] == 50000


def test_collect_mcd_quality_results_reads_artifacts_and_train_log(tmp_path) -> None:
    context = MCDQualityPlanContext(output_root=str(tmp_path / "runs"), asset_dir=str(tmp_path / "assets"))
    plan = build_mcd_quality_plan(context)
    run = plan.runs[0]
    preprocess = tmp_path / "runs" / run.profile.name / "preprocess"
    train = tmp_path / "runs" / run.profile.name / "train"
    export = tmp_path / "assets" / f"{run.profile.name}.splat"

    (preprocess / "images").mkdir(parents=True)
    (preprocess / "images" / "frame_000000.jpg").write_bytes(b"jpg")
    (preprocess / "images" / "image_timestamps.csv").write_text("filename,timestamp_ns\nframe_000000.jpg,1\n")
    (preprocess / "lidar").mkdir()
    (preprocess / "lidar" / "frame_000000.npy").write_bytes(b"npy")
    (preprocess / "depth").mkdir()
    (preprocess / "depth" / "frame_000000.npy").write_bytes(b"npy")
    (preprocess / "pose").mkdir()
    (preprocess / "pose" / "origin_wgs84.json").write_text("{}\n")
    (preprocess / "lidar_world_rgb.npy").write_bytes(b"npy")
    sparse = preprocess / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("# camera list\n1 PINHOLE 640 480 1 1 1 1\n")
    (sparse / "images.txt").write_text("# image list\n1 1 0 0 0 0 0 0 1 frame_000000.jpg\n\n")
    (sparse / "points3D.txt").write_text("# points\n1 0 0 0 255 255 255 0\n2 1 0 0 255 0 0 0\n")
    train.mkdir(parents=True)
    (train / "point_cloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 123\nend_header\n")
    (train / "train.log").write_text(
        "  [Iter  30000/30000] loss=6.1367 l1=0.1951 ssim_loss=0.5354 n_gaussians=123\n"
        "Training complete in 500.3s\n"
        "Final Gaussians: 123\n"
    )
    export.parent.mkdir(parents=True)
    export.write_bytes(b"\0" * 64)

    report = collect_mcd_quality_results(plan)
    baseline = report["runs"][0]

    assert report["completeCount"] == 1
    assert baseline["complete"] is True
    assert baseline["preprocess"]["imageCount"] == 1
    assert baseline["preprocess"]["depthMapCount"] == 1
    assert baseline["preprocess"]["points3DCount"] == 2
    assert baseline["train"]["trainedGaussians"] == 123
    assert baseline["train"]["finalL1"] == 0.1951
    assert baseline["train"]["trainingSeconds"] == 500.3
    assert baseline["export"]["splatBytes"] == 64
    assert baseline["export"]["splatGaussians"] == 2
    assert baseline["plannedFrames"] == 400
    assert baseline["configPath"] == "configs/training_depth_long.yaml"

    benchmark = render_quality_benchmark_markdown(report)

    assert "# MCD Quality Benchmark" in benchmark
    assert (
        "| Single D455B 400 Depth Long | 1/400 | training_depth_long.yaml | 500.3 s | 123 | "
        "0.1951 | 64 B / 2 gauss | yes |"
    ) in benchmark

    gate = evaluate_mcd_quality_gates(report)
    baseline_gate = gate["runs"][0]
    failed_checks = {check["name"] for check in baseline_gate["checks"] if not check["passed"]}

    assert gate["passed"] is False
    assert baseline_gate["passed"] is False
    assert failed_checks == {"frames"}

    loose_gate = evaluate_mcd_quality_gates(
        report,
        MCDQualityGatePolicy(min_frame_fraction=0.0, min_depth_fraction=0.0, max_final_l1=0.2),
    )

    assert loose_gate["runs"][0]["passed"] is True


def test_collect_mcd_quality_results_renders_markdown_and_json_for_missing_runs(tmp_path) -> None:
    plan = build_mcd_quality_plan(MCDQualityPlanContext(output_root=str(tmp_path / "missing")))

    report = collect_mcd_quality_results(plan)
    markdown = render_quality_report_markdown(report)
    gate_markdown = render_quality_gate_markdown(evaluate_mcd_quality_gates(report))
    payload = json.loads(render_quality_report_json(report))

    assert report["completeCount"] == 0
    assert "0/3 complete" in markdown
    assert "Single D455B 400 Depth Long" in markdown
    assert "Gate: 0/3 runs passed" in gate_markdown
    assert "| Single D455B 400 Depth Long | fail |" in gate_markdown
    assert payload["type"] == "mcd-quality-results-report"


def test_collect_mcd_quality_runs_script_can_emit_markdown(tmp_path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_mcd_quality_runs.py",
            "--output-root",
            str(tmp_path / "missing"),
            "--format",
            "markdown",
            "--profile",
            "ntu_day02_single_400_depth_long",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "# MCD Quality Results" in result.stdout
    assert "Single D455B 400 Depth Long" in result.stdout


def test_collect_mcd_quality_runs_script_can_emit_benchmark(tmp_path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_mcd_quality_runs.py",
            "--output-root",
            str(tmp_path / "missing"),
            "--format",
            "benchmark",
            "--profile",
            "ntu_day02_single_400_depth_long",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "# MCD Quality Benchmark" in result.stdout
    assert "| Single D455B 400 Depth Long | n/a/400 | training_depth_long.yaml |" in result.stdout


def test_collect_mcd_quality_runs_script_can_emit_gate_and_fail(tmp_path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_mcd_quality_runs.py",
            "--output-root",
            str(tmp_path / "missing"),
            "--format",
            "gate",
            "--profile",
            "ntu_day02_single_400_depth_long",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "# MCD Quality Gate" in result.stdout
    assert "Gate: 0/1 runs passed" in result.stdout
    assert "| Single D455B 400 Depth Long | fail |" in result.stdout

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/collect_mcd_quality_runs.py",
            "--output-root",
            str(tmp_path / "missing"),
            "--format",
            "gate",
            "--profile",
            "ntu_day02_single_400_depth_long",
            "--fail-on-gate",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert failed.returncode == 2
    assert "# MCD Quality Gate" in failed.stdout
