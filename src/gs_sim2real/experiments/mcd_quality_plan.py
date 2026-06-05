"""Quality-run planning helpers for supervised MCD sessions."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gs_sim2real.experiments.mcd_quality_gate import (
    MCDQualityGatePolicy,
    evaluate_mcd_quality_gates,
    render_quality_gate_markdown,
)


@dataclass(frozen=True)
class MCDQualityPlanContext:
    """Paths and common sensor settings for MCD quality runs."""

    session_dir: str = "data/mcd/ntu_day_02"
    output_root: str = "outputs/mcd_quality"
    calibration_path: str = "data/mcd/calibration_atv.yaml"
    asset_dir: str = "outputs/mcd_quality/assets"
    python_executable: str = "python3"
    pythonpath: str = "src"
    gnss_topic: str = "/vn200/GPS"
    imu_topic: str = "/vn200/imu"
    lidar_topic: str = "/os_cloud_node/points"
    lidar_frame: str = "os_sensor"
    start_offset_sec: float = 35.0


@dataclass(frozen=True)
class MCDQualityRunProfile:
    """One supervised MCD quality candidate."""

    name: str
    label: str
    intent: str
    image_topics: tuple[str, ...]
    max_frames: int
    every_n: int
    iterations: int
    config_path: str
    camera_frame: str | None = None
    include_dynamic_tf: bool = False
    requires_full_folder: bool = False
    export_max_points: int = 400_000
    splat_normalize_extent: float = 17.0
    splat_min_opacity: float = 0.02
    splat_max_scale: float = 2.0


@dataclass(frozen=True)
class MCDQualityRunPlan:
    """Concrete commands and expected artifacts for one quality run."""

    profile: MCDQualityRunProfile
    preprocess_dir: str
    train_dir: str
    export_path: str
    preprocess_command: tuple[str, ...]
    train_command: tuple[str, ...]
    export_command: tuple[str, ...]
    expected_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class MCDQualityPlan:
    """A reproducible quality-run matrix."""

    context: MCDQualityPlanContext
    preflight_command: tuple[str, ...]
    runs: tuple[MCDQualityRunPlan, ...]


def default_mcd_quality_profiles() -> tuple[MCDQualityRunProfile, ...]:
    """Return the default ntu_day_02 quality-push matrix."""
    return (
        MCDQualityRunProfile(
            name="ntu_day02_single_400_depth_long",
            label="Single D455B 400 Depth Long",
            intent="Reproduce the production MCD supervised baseline before changing quality knobs.",
            image_topics=("/d455b/color/image_raw",),
            camera_frame="d455b_color",
            max_frames=400,
            every_n=14,
            iterations=30_000,
            config_path="configs/training_depth_long.yaml",
        ),
        MCDQualityRunProfile(
            name="ntu_day02_single_800_ba",
            label="Single D455B 800 BA",
            intent="Double temporal coverage and enable depth + appearance + pose refinement on the known-good camera.",
            image_topics=("/d455b/color/image_raw",),
            camera_frame="d455b_color",
            max_frames=800,
            every_n=7,
            iterations=50_000,
            config_path="configs/training_ba.yaml",
        ),
        MCDQualityRunProfile(
            # MCDVIRAL ATV ships only d455b + d435i color cameras (no d455t),
            # so the official multi-camera plan caps at two cameras.
            name="ntu_day02_multi_2cam_300each_ba",
            label="Two-Camera 300 Each BA",
            intent="Use the official ATV calibration to compare D455B + D435I coverage against the single-camera baseline.",
            image_topics=(
                "/d455b/color/image_raw",
                "/d435i/color/image_raw",
            ),
            max_frames=300,
            every_n=14,
            iterations=50_000,
            config_path="configs/training_ba.yaml",
            requires_full_folder=True,
        ),
    )


def build_mcd_quality_plan(
    context: MCDQualityPlanContext | None = None,
    *,
    profiles: Iterable[MCDQualityRunProfile] | None = None,
) -> MCDQualityPlan:
    """Build concrete commands for an MCD quality-run matrix."""
    ctx = context or MCDQualityPlanContext()
    selected_profiles = tuple(profiles or default_mcd_quality_profiles())
    preflight_command = (
        ctx.python_executable,
        "scripts/check_mcd_gnss.py",
        ctx.session_dir,
        "--gnss-topic",
        ctx.gnss_topic,
        "--flatten-altitude",
        "--start-offset-sec",
        _format_number(ctx.start_offset_sec),
    )
    runs = tuple(_build_run_plan(ctx, profile) for profile in selected_profiles)
    return MCDQualityPlan(context=ctx, preflight_command=preflight_command, runs=runs)


def _build_run_plan(ctx: MCDQualityPlanContext, profile: MCDQualityRunProfile) -> MCDQualityRunPlan:
    run_root = Path(ctx.output_root) / profile.name
    preprocess_dir = run_root / "preprocess"
    train_dir = run_root / "train"
    export_path = Path(ctx.asset_dir) / f"{profile.name}.splat"

    preprocess_parts: list[str] = [
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "preprocess",
        "--images",
        ctx.session_dir,
        "--output",
        str(preprocess_dir),
        "--method",
        "mcd",
        "--image-topic",
        ",".join(profile.image_topics),
        "--gnss-topic",
        ctx.gnss_topic,
        "--mcd-static-calibration",
        ctx.calibration_path,
        "--mcd-seed-poses-from-gnss",
        "--mcd-flatten-gnss-altitude",
        "--mcd-start-offset-sec",
        _format_number(ctx.start_offset_sec),
        "--mcd-tf-use-image-stamps",
        "--lidar-topic",
        ctx.lidar_topic,
        "--mcd-lidar-frame",
        ctx.lidar_frame,
        "--imu-topic",
        ctx.imu_topic,
        "--extract-lidar",
        "--extract-imu",
        "--mcd-export-depth",
        "--max-frames",
        str(profile.max_frames),
        "--every-n",
        str(profile.every_n),
        "--matching",
        "sequential",
        "--no-gpu",
    ]
    if profile.camera_frame:
        preprocess_parts.extend(["--mcd-camera-frame", profile.camera_frame])
    if profile.include_dynamic_tf:
        preprocess_parts.append("--mcd-include-tf-dynamic")

    train_command = (
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "train",
        "--data",
        str(preprocess_dir),
        "--output",
        str(train_dir),
        "--method",
        "gsplat",
        "--iterations",
        str(profile.iterations),
        "--config",
        profile.config_path,
    )
    export_command = (
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "export",
        "--model",
        str(train_dir / "point_cloud.ply"),
        "--format",
        "splat",
        "--output",
        str(export_path),
        "--max-points",
        str(profile.export_max_points),
        "--splat-normalize-extent",
        _format_number(profile.splat_normalize_extent),
        "--splat-min-opacity",
        _format_number(profile.splat_min_opacity),
        "--splat-max-scale",
        _format_number(profile.splat_max_scale),
    )

    expected_artifacts = (
        str(preprocess_dir / "images" / "image_timestamps.csv"),
        str(preprocess_dir / "pose" / "origin_wgs84.json"),
        str(preprocess_dir / "lidar_world_rgb.npy"),
        str(preprocess_dir / "depth"),
        str(preprocess_dir / "sparse" / "0" / "cameras.txt"),
        str(preprocess_dir / "sparse" / "0" / "images.txt"),
        str(preprocess_dir / "sparse" / "0" / "points3D.txt"),
        str(train_dir / "point_cloud.ply"),
        str(export_path),
    )
    return MCDQualityRunPlan(
        profile=profile,
        preprocess_dir=str(preprocess_dir),
        train_dir=str(train_dir),
        export_path=str(export_path),
        preprocess_command=tuple(preprocess_parts),
        train_command=train_command,
        export_command=export_command,
        expected_artifacts=expected_artifacts,
    )


def plan_to_dict(plan: MCDQualityPlan) -> dict[str, Any]:
    """Convert a quality plan into JSON-serializable data."""
    return {
        "context": plan.context.__dict__,
        "preflightCommand": list(plan.preflight_command),
        "runs": [
            {
                "name": run.profile.name,
                "label": run.profile.label,
                "intent": run.profile.intent,
                "requiresFullFolder": run.profile.requires_full_folder,
                "imageTopics": list(run.profile.image_topics),
                "maxFrames": run.profile.max_frames,
                "everyN": run.profile.every_n,
                "iterations": run.profile.iterations,
                "configPath": run.profile.config_path,
                "preprocessDir": run.preprocess_dir,
                "trainDir": run.train_dir,
                "exportPath": run.export_path,
                "preprocessCommand": list(run.preprocess_command),
                "trainCommand": list(run.train_command),
                "exportCommand": list(run.export_command),
                "expectedArtifacts": list(run.expected_artifacts),
            }
            for run in plan.runs
        ],
    }


def render_plan_json(plan: MCDQualityPlan) -> str:
    """Render the plan as stable JSON."""
    return json.dumps(plan_to_dict(plan), indent=2, sort_keys=True) + "\n"


def render_plan_markdown(plan: MCDQualityPlan) -> str:
    """Render a quality plan as a compact runbook."""
    lines = [
        "# MCD Quality Run Plan",
        "",
        "## Shared Preflight",
        "",
        "```bash",
        render_shell_command(plan.preflight_command),
        "```",
        "",
        "## Runs",
        "",
    ]
    for run in plan.runs:
        lines.extend(
            [
                f"### {run.profile.label}",
                "",
                run.profile.intent,
                "",
                f"- profile: `{run.profile.name}`",
                f"- frames: `{run.profile.max_frames}` every `{run.profile.every_n}`",
                f"- config: `{run.profile.config_path}`",
                f"- requires full folder: `{str(run.profile.requires_full_folder).lower()}`",
                "",
                "```bash",
                render_shell_command(run.preprocess_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.train_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.export_command, pythonpath=plan.context.pythonpath),
                "```",
                "",
                "Expected artifacts:",
            ]
        )
        lines.extend(f"- `{artifact}`" for artifact in run.expected_artifacts)
        lines.append("")
    return "\n".join(lines)


def render_plan_shell(plan: MCDQualityPlan) -> str:
    """Render all commands as a shell runbook."""
    lines = ["set -euo pipefail", "", "# Shared GNSS preflight", render_shell_command(plan.preflight_command), ""]
    for run in plan.runs:
        lines.extend(
            [
                f"# {run.profile.label}",
                render_shell_command(run.preprocess_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.train_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.export_command, pythonpath=plan.context.pythonpath),
                "",
            ]
        )
    return "\n".join(lines)


def render_shell_command(command: Iterable[str], *, pythonpath: str | None = None) -> str:
    """Quote a command for bash copy/paste."""
    parts = [shlex.quote(str(part)) for part in command]
    if pythonpath:
        parts.insert(0, f"PYTHONPATH={shlex.quote(str(pythonpath))}")
    return " ".join(parts)


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def collect_mcd_quality_results(plan: MCDQualityPlan) -> dict[str, Any]:
    """Collect artifact and metric summaries for a quality-run matrix."""
    run_reports = [collect_mcd_quality_run_result(run) for run in plan.runs]
    complete_count = sum(1 for run in run_reports if run["complete"])
    return {
        "type": "mcd-quality-results-report",
        "context": plan.context.__dict__,
        "runCount": len(run_reports),
        "completeCount": complete_count,
        "runs": run_reports,
    }


def collect_mcd_quality_run_result(run: MCDQualityRunPlan) -> dict[str, Any]:
    """Collect artifact presence and lightweight metrics for one quality run."""
    preprocess_dir = Path(run.preprocess_dir)
    train_dir = Path(run.train_dir)
    export_path = Path(run.export_path)
    sparse_dir = preprocess_dir / "sparse" / "0"
    artifact_status = {path: _artifact_exists(Path(path)) for path in run.expected_artifacts}
    missing_artifacts = [path for path, exists in artifact_status.items() if not exists]
    train_log_path = _find_train_log_path(train_dir)
    train_log_metrics = _parse_train_log(train_log_path) if train_log_path else {}
    ply_path = train_dir / "point_cloud.ply"
    trained_gaussians = _read_ply_vertex_count(ply_path)
    if trained_gaussians is None:
        trained_gaussians = train_log_metrics.get("finalGaussians")

    splat_bytes = export_path.stat().st_size if export_path.is_file() else None
    splat_gaussians = None
    if splat_bytes is not None and splat_bytes % 32 == 0:
        splat_gaussians = splat_bytes // 32

    return {
        "name": run.profile.name,
        "label": run.profile.label,
        "plannedFrames": run.profile.max_frames,
        "everyN": run.profile.every_n,
        "iterations": run.profile.iterations,
        "configPath": run.profile.config_path,
        "imageTopics": list(run.profile.image_topics),
        "complete": not missing_artifacts,
        "missingArtifacts": missing_artifacts,
        "artifactStatus": artifact_status,
        "preprocess": {
            "imageCount": _count_images(preprocess_dir / "images"),
            "lidarFrameCount": _count_files(preprocess_dir / "lidar", "frame_*.npy"),
            "depthMapCount": _count_files(preprocess_dir / "depth", "*.npy"),
            "cameraCount": _count_colmap_rows(sparse_dir / "cameras.txt"),
            "registeredImageCount": _count_colmap_image_rows(sparse_dir / "images.txt"),
            "points3DCount": _count_colmap_rows(sparse_dir / "points3D.txt"),
        },
        "train": {
            "pointCloudPath": str(ply_path),
            "pointCloudExists": ply_path.is_file(),
            "trainedGaussians": trained_gaussians,
            "logPath": str(train_log_path) if train_log_path else None,
            "finalLoss": train_log_metrics.get("finalLoss"),
            "finalL1": train_log_metrics.get("finalL1"),
            "finalSsimLoss": train_log_metrics.get("finalSsimLoss"),
            "trainingSeconds": train_log_metrics.get("trainingSeconds"),
        },
        "export": {
            "splatPath": run.export_path,
            "splatExists": export_path.is_file(),
            "splatBytes": splat_bytes,
            "splatGaussians": splat_gaussians,
        },
    }


def render_quality_report_json(report: dict[str, Any]) -> str:
    """Render collected quality results as stable JSON."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_quality_report_markdown(report: dict[str, Any]) -> str:
    """Render collected quality results as a compact Markdown table."""
    lines = [
        "# MCD Quality Results",
        "",
        f"Runs: {report['completeCount']}/{report['runCount']} complete",
        "",
        "| Run | Complete | Images | Depth | Sparse Pts | Gaussians | L1 | Splat | Missing |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        preprocess = run["preprocess"]
        train = run["train"]
        export = run["export"]
        missing = ", ".join(f"`{Path(path).name}`" for path in run["missingArtifacts"]) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    run["label"],
                    "yes" if run["complete"] else "no",
                    _format_optional_int(preprocess["imageCount"]),
                    _format_optional_int(preprocess["depthMapCount"]),
                    _format_optional_int(preprocess["points3DCount"]),
                    _format_optional_int(train["trainedGaussians"]),
                    _format_optional_float(train["finalL1"], digits=4),
                    _format_optional_bytes(export["splatBytes"]),
                    missing,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_quality_benchmark_markdown(report: dict[str, Any]) -> str:
    """Render collected quality results as a comparison-oriented benchmark table."""
    lines = [
        "# MCD Quality Benchmark",
        "",
        f"Runs: {report['completeCount']}/{report['runCount']} complete",
        "",
        "| Run | Frames | Config | Train time | Trained gauss | Final L1 | Splat | Complete |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        preprocess = run["preprocess"]
        train = run["train"]
        export = run["export"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["label"],
                    _format_actual_over_planned(preprocess["imageCount"], run.get("plannedFrames")),
                    _format_config_name(run.get("configPath")),
                    _format_optional_seconds(train["trainingSeconds"]),
                    _format_optional_int(train["trainedGaussians"]),
                    _format_optional_float(train["finalL1"], digits=4),
                    _format_splat_summary(export["splatBytes"], export["splatGaussians"]),
                    "yes" if run["complete"] else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _artifact_exists(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file()


def _count_files(path: Path, pattern: str) -> int | None:
    if not path.is_dir():
        return None
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def _count_images(path: Path) -> int | None:
    if not path.is_dir():
        return None
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def _count_colmap_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.startswith("#"):
                count += 1
    return count


def _count_colmap_image_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) >= 10:
                count += 1
    return count


def _find_train_log_path(train_dir: Path) -> Path | None:
    for name in ("train.log", "training.log", "gsplat_train.log", "stdout.log"):
        path = train_dir / name
        if path.is_file():
            return path
    logs = sorted(train_dir.glob("*.log")) if train_dir.is_dir() else []
    return logs[0] if logs else None


def _parse_train_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    iter_matches = re.findall(
        r"loss=(?P<loss>[0-9.]+)\s+l1=(?P<l1>[0-9.]+)\s+ssim_loss=(?P<ssim>[0-9.]+)",
        text,
    )
    if iter_matches:
        loss, l1, ssim = iter_matches[-1]
        metrics["finalLoss"] = float(loss)
        metrics["finalL1"] = float(l1)
        metrics["finalSsimLoss"] = float(ssim)
    time_match = re.search(r"Training complete in\s+([0-9.]+)s", text)
    if time_match:
        metrics["trainingSeconds"] = float(time_match.group(1))
    gaussian_match = re.search(r"Final Gaussians:\s+([0-9,]+)", text)
    if gaussian_match:
        metrics["finalGaussians"] = int(gaussian_match.group(1).replace(",", ""))
    return metrics


def _read_ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            for _ in range(256):
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except (OSError, ValueError):
        return None
    return None


def _format_optional_int(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _format_optional_float(value: Any, *, digits: int) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _format_optional_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,} B"


def _format_actual_over_planned(actual: Any, planned: Any) -> str:
    actual_text = _format_optional_int(actual)
    planned_text = _format_optional_int(planned)
    return f"{actual_text}/{planned_text}"


def _format_config_name(value: Any) -> str:
    if value is None:
        return "n/a"
    name = Path(str(value)).name
    return name or "n/a"


def _format_optional_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} s"


def _format_splat_summary(bytes_value: Any, gaussians: Any) -> str:
    if bytes_value is None and gaussians is None:
        return "n/a"
    return f"{_format_compact_bytes(bytes_value)} / {_format_compact_gaussians(gaussians)}"


def _format_compact_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    amount = int(value)
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f} MB"
    if amount >= 1_000:
        return f"{amount / 1_000:.1f} KB"
    return f"{amount} B"


def _format_compact_gaussians(value: Any) -> str:
    if value is None:
        return "n/a"
    count = int(value)
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M gauss"
    if count >= 1_000:
        return f"{count / 1_000:.0f}k gauss"
    return f"{count} gauss"


__all__ = [
    "MCDQualityGatePolicy",
    "MCDQualityPlan",
    "MCDQualityPlanContext",
    "MCDQualityRunPlan",
    "MCDQualityRunProfile",
    "build_mcd_quality_plan",
    "collect_mcd_quality_results",
    "collect_mcd_quality_run_result",
    "default_mcd_quality_profiles",
    "evaluate_mcd_quality_gates",
    "plan_to_dict",
    "render_quality_benchmark_markdown",
    "render_quality_gate_markdown",
    "render_quality_report_json",
    "render_quality_report_markdown",
    "render_plan_json",
    "render_plan_markdown",
    "render_plan_shell",
    "render_shell_command",
]
