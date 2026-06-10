"""Scene-level simulation contract for Physical AI workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SITE_URL = "https://rsasaki0109.github.io/3dgs-robotics/"
CATALOG_VERSION = "gs-mapper-physical-ai-sim/v1"


@dataclass(frozen=True, slots=True)
class Vec3:
    """Small JSON-friendly 3D vector."""

    x: float
    y: float
    z: float

    @classmethod
    def from_sequence(cls, value: Any) -> Vec3:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Vec3 requires exactly three numeric values")
        return cls(float(value[0]), float(value[1]), float(value[2]))

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


@dataclass(frozen=True, slots=True)
class AxisAlignedBounds:
    """World-frame bounds used for initial navigation and sampling contracts."""

    minimum: Vec3
    maximum: Vec3
    source: str
    confidence: str

    @property
    def extent(self) -> Vec3:
        return Vec3(
            self.maximum.x - self.minimum.x,
            self.maximum.y - self.minimum.y,
            self.maximum.z - self.minimum.z,
        )

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.minimum.x + self.maximum.x) / 2,
            (self.minimum.y + self.maximum.y) / 2,
            (self.minimum.z + self.maximum.z) / 2,
        )

    def contains(self, point: Vec3) -> bool:
        return (
            self.minimum.x <= point.x <= self.maximum.x
            and self.minimum.y <= point.y <= self.maximum.y
            and self.minimum.z <= point.z <= self.maximum.z
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": self.minimum.to_list(),
            "max": self.maximum.to_list(),
            "extent": self.extent.to_list(),
            "center": self.center.to_list(),
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Coordinate-frame metadata needed by agents and robotics bridges."""

    frame_id: str
    unit: str = "meter"
    up_axis: str = "y"
    gravity: Vec3 = Vec3(0.0, -9.80665, 0.0)
    handedness: str = "right-handed"
    scale_status: str = "metric"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frameId": self.frame_id,
            "unit": self.unit,
            "upAxis": self.up_axis,
            "gravity": self.gravity.to_list(),
            "handedness": self.handedness,
            "scaleStatus": self.scale_status,
        }


@dataclass(frozen=True, slots=True)
class SensorModel:
    """Sensor output contract for an environment scene."""

    sensor_id: str
    modality: str
    status: str
    outputs: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensorId": self.sensor_id,
            "modality": self.modality,
            "status": self.status,
            "outputs": list(self.outputs),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class SensorRig:
    """Named collection of sensor contracts."""

    rig_id: str
    sensors: tuple[SensorModel, ...]

    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(sensor.sensor_id for sensor in self.sensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rigId": self.rig_id,
            "sensors": [sensor.to_dict() for sensor in self.sensors],
        }


@dataclass(frozen=True, slots=True)
class TrajectoryEpisode:
    """Train/eval trajectory split metadata for a scene."""

    episode_id: str
    split: str
    source: str
    pose_count: int | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodeId": self.episode_id,
            "split": self.split,
            "source": self.source,
            "poseCount": self.pose_count,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """One task an agent can be evaluated on inside a scene."""

    task_id: str
    label: str
    kind: str
    metrics: tuple[str, ...]
    required_sensors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "label": self.label,
            "kind": self.kind,
            "metrics": list(self.metrics),
            "requiredSensors": list(self.required_sensors),
        }


@dataclass(frozen=True, slots=True)
class SceneEnvironment:
    """Physical AI contract for one bundled GS Mapper scene."""

    scene_id: str
    label: str
    summary: str
    asset_url: str
    preview_url: str
    viewer_url: str
    reconstruction_method: str
    scene_family: str
    coordinate_frame: CoordinateFrame
    bounds: AxisAlignedBounds
    sensor_rig: SensorRig
    task_split: tuple[TrajectoryEpisode, ...]
    evaluation_tasks: tuple[EvaluationTask, ...]
    trajectory_extent_meters: float | None
    tags: tuple[str, ...]

    def has_task(self, task_id: str) -> bool:
        return any(task.task_id == task_id for task in self.evaluation_tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "label": self.label,
            "summary": self.summary,
            "assetUrl": self.asset_url,
            "previewUrl": self.preview_url,
            "viewerUrl": self.viewer_url,
            "reconstructionMethod": self.reconstruction_method,
            "sceneFamily": self.scene_family,
            "coordinateFrame": self.coordinate_frame.to_dict(),
            "bounds": self.bounds.to_dict(),
            "sensorRig": self.sensor_rig.to_dict(),
            "taskSplit": [episode.to_dict() for episode in self.task_split],
            "evaluationTasks": [task.to_dict() for task in self.evaluation_tasks],
            "trajectoryExtentMeters": self.trajectory_extent_meters,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class SimulationCatalog:
    """Collection of Physical AI scene contracts."""

    version: str
    source_catalog: str
    scenes: tuple[SceneEnvironment, ...]

    def scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes)

    def scene_by_id(self, scene_id: str) -> SceneEnvironment:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"unknown simulation scene: {scene_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sourceCatalog": self.source_catalog,
            "sceneCount": len(self.scenes),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


@dataclass(frozen=True, slots=True)
class _SceneProfile:
    scene_family: str
    scale_status: str
    trajectory_extent_meters: float | None
    pose_count: int | None
    tags: tuple[str, ...]


SCENE_PROFILES: dict[str, _SceneProfile] = {
    "outdoor-demo": _SceneProfile("autoware", "metric", None, 6120, ("supervised", "gnss", "lidar")),
    "outdoor-demo-dust3r": _SceneProfile("autoware", "relative", 1.02, 20, ("dust3r", "pose-free")),
    "mcd-tuhh-day04": _SceneProfile("mcd", "relative", 1.78, 20, ("dust3r", "pose-free")),
    "bag6-mast3r": _SceneProfile("autoware", "metric", 28.1, 20, ("mast3r", "pose-free", "metric")),
    "bag6-vggt-slam-20-15k": _SceneProfile("autoware", "estimated-metric", 3.23, 20, ("vggt-slam", "external-slam")),
    "bag6-mast3r-slam-20-15k": _SceneProfile(
        "autoware", "estimated-metric", 2.16, 10, ("mast3r-slam", "external-slam")
    ),
    "bag6-pi3x-20-15k": _SceneProfile("autoware", "estimated-metric", 74.4, 20, ("pi3x", "pi3", "external-slam")),
    "mcd-tuhh-day04-mast3r": _SceneProfile("mcd", "metric", 59.0, 20, ("mast3r", "pose-free", "metric")),
    "mcd-ntu-day02-supervised": _SceneProfile("mcd", "metric", 250.0, 400, ("supervised", "gnss", "lidar")),
}


def load_scene_picker_catalog(path: str | Path) -> dict[str, Any]:
    """Load the existing viewer scene picker catalog."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_simulation_catalog_from_scene_picker(
    path: str | Path,
    *,
    site_url: str = DEFAULT_SITE_URL,
) -> SimulationCatalog:
    """Build a simulation catalog from `docs/scenes-list.json`."""

    catalog_path = Path(path)
    return build_simulation_catalog(
        load_scene_picker_catalog(catalog_path),
        docs_root=catalog_path.parent,
        site_url=site_url,
        source_catalog=_catalog_source_label(catalog_path),
    )


def build_simulation_catalog(
    scene_picker_payload: dict[str, Any],
    *,
    docs_root: str | Path | None = None,
    site_url: str = DEFAULT_SITE_URL,
    source_catalog: str = "docs/scenes-list.json",
) -> SimulationCatalog:
    """Convert the public viewer catalog into Physical AI scene contracts."""

    scenes = tuple(
        _build_scene_environment(scene_payload, docs_root=Path(docs_root) if docs_root else None, site_url=site_url)
        for scene_payload in scene_picker_payload.get("scenes", [])
    )
    _require_unique_scene_ids(scenes)
    return SimulationCatalog(version=CATALOG_VERSION, source_catalog=source_catalog, scenes=scenes)


def render_simulation_catalog_json(catalog: SimulationCatalog) -> str:
    """Render a stable JSON simulation catalog."""

    return json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _build_scene_environment(
    scene_payload: dict[str, Any],
    *,
    docs_root: Path | None,
    site_url: str,
) -> SceneEnvironment:
    asset_url = str(scene_payload["url"])
    scene_id = _scene_id_from_asset_url(asset_url)
    profile = SCENE_PROFILES.get(scene_id, _fallback_profile(scene_payload))
    reconstruction_method = _infer_reconstruction_method(scene_payload)
    frame_scale = profile.scale_status.replace("-", "_")
    coordinate_frame = CoordinateFrame(
        frame_id=f"{profile.scene_family}_{frame_scale}_world",
        scale_status=profile.scale_status,
    )
    bounds = _bounds_for_scene(scene_id, profile, docs_root)
    sensor_rig = _default_sensor_rig()
    task_split = _default_task_split(scene_id, profile)
    evaluation_tasks = _default_evaluation_tasks(profile.scale_status)

    return SceneEnvironment(
        scene_id=scene_id,
        label=str(scene_payload["label"]),
        summary=str(scene_payload.get("summary") or ""),
        asset_url=asset_url,
        preview_url=str(scene_payload.get("preview") or ""),
        viewer_url=_viewer_url(site_url, asset_url),
        reconstruction_method=reconstruction_method,
        scene_family=profile.scene_family,
        coordinate_frame=coordinate_frame,
        bounds=bounds,
        sensor_rig=sensor_rig,
        task_split=task_split,
        evaluation_tasks=evaluation_tasks,
        trajectory_extent_meters=profile.trajectory_extent_meters,
        tags=(reconstruction_method, profile.scene_family, profile.scale_status, *profile.tags),
    )


def _scene_id_from_asset_url(asset_url: str) -> str:
    return Path(asset_url).stem


def _viewer_url(site_url: str, asset_url: str) -> str:
    return f"{site_url.rstrip('/')}/splat.html?url={asset_url}"


def _catalog_source_label(path: Path) -> str:
    if path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.as_posix()


def _fallback_profile(scene_payload: dict[str, Any]) -> _SceneProfile:
    label = f"{scene_payload.get('label', '')} {scene_payload.get('summary', '')}".lower()
    scene_family = "mcd" if "mcd" in label else "autoware" if "autoware" in label or "bag6" in label else "generic"
    scale_status = "relative" if "dust3r" in label else "estimated-metric"
    return _SceneProfile(scene_family, scale_status, None, None, ("inferred",))


def _infer_reconstruction_method(scene_payload: dict[str, Any]) -> str:
    text = f"{scene_payload.get('label', '')} {scene_payload.get('summary', '')}".lower()
    if "vggt-slam" in text:
        return "vggt-slam-2.0"
    if "mast3r-slam" in text:
        return "mast3r-slam"
    if "pi3x" in text:
        return "pi3x-vo"
    if "pi3" in text:
        return "pi3-vo"
    if "loger" in text:
        return "loger"
    if "mast3r" in text:
        return "mast3r-pose-free"
    if "dust3r" in text:
        return "dust3r-pose-free"
    if "supervised" in text or "gnss" in text or "lidar" in text:
        return "supervised-gnss-lidar"
    return "unknown"


def _bounds_for_scene(scene_id: str, profile: _SceneProfile, docs_root: Path | None) -> AxisAlignedBounds:
    manifest_bounds = _load_manifest_bounds(scene_id, docs_root)
    if manifest_bounds is not None and scene_id == "outdoor-demo":
        return manifest_bounds

    extent = profile.trajectory_extent_meters
    if extent is not None:
        half = max(1.0, extent / 2)
        return AxisAlignedBounds(
            minimum=Vec3(-half, -2.0, -half),
            maximum=Vec3(half, 2.0, half),
            source="readme-trajectory-extent",
            confidence="estimated",
        )

    return AxisAlignedBounds(
        minimum=Vec3(-1.0, -1.0, -1.0),
        maximum=Vec3(1.0, 1.0, 1.0),
        source="unit-placeholder",
        confidence="placeholder",
    )


def _load_manifest_bounds(scene_id: str, docs_root: Path | None) -> AxisAlignedBounds | None:
    if docs_root is None:
        return None

    manifest_relpath = Path("assets") / scene_id / "scene.json"
    manifest_path = docs_root / manifest_relpath
    if not manifest_path.is_file():
        return None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    bounds_payload = payload.get("bounds") or {}
    if "min" not in bounds_payload or "max" not in bounds_payload:
        return None
    return AxisAlignedBounds(
        minimum=Vec3.from_sequence(bounds_payload["min"]),
        maximum=Vec3.from_sequence(bounds_payload["max"]),
        source=manifest_relpath.as_posix(),
        confidence="declared",
    )


def _default_sensor_rig() -> SensorRig:
    return SensorRig(
        rig_id="physical-ai-default",
        sensors=(
            SensorModel(
                sensor_id="rgb-forward",
                modality="rgb",
                status="ready-via-splat-viewer",
                outputs=("rgb",),
                description="RGB observation rendered from the browser-viewable splat scene.",
            ),
            SensorModel(
                sensor_id="depth-proxy",
                modality="depth",
                status="ready-via-splat-raster",
                outputs=("depth", "validity-mask"),
                description="Depth proxy rendered from the local splat rasterizer as float32 depth plus validity mask.",
            ),
            SensorModel(
                sensor_id="lidar-ray-proxy",
                modality="lidar",
                status="ready-via-depth-rays",
                outputs=("ranges", "points"),
                description="LiDAR-like range and point proxy sampled from the local splat raster depth image.",
            ),
            SensorModel(
                sensor_id="imu-proxy",
                modality="imu",
                status="ready-via-kinematic-finite-diff",
                outputs=("angular-velocity", "linear-acceleration"),
                description=(
                    "IMU proxy synthesised by HeadlessPhysicalAIEnvironment: each step finite-differences "
                    "the agent pose to estimate body-frame angular velocity and linear acceleration, with "
                    "no gravity model. teleport actions and the post-reset state both report the zero "
                    "kinematic state (step_dt_seconds == 0). Raw-sensor noise profiles' IMU σ fields "
                    "perturb the rendered outputs in place."
                ),
            ),
        ),
    )


def _default_task_split(scene_id: str, profile: _SceneProfile) -> tuple[TrajectoryEpisode, ...]:
    train_count, eval_count = _split_pose_count(profile.pose_count)
    return (
        TrajectoryEpisode(
            episode_id=f"{scene_id}-train",
            split="train",
            source="reconstruction-trajectory",
            pose_count=train_count,
            description="Trajectory poses used for reconstruction or artifact import.",
        ),
        TrajectoryEpisode(
            episode_id=f"{scene_id}-eval",
            split="eval",
            source="held-out-viewpoint-contract",
            pose_count=eval_count,
            description="Held-out viewpoints sampled within scene bounds for Physical AI evaluation.",
        ),
    )


def _split_pose_count(pose_count: int | None) -> tuple[int | None, int | None]:
    if pose_count is None:
        return None, None
    if pose_count <= 1:
        return pose_count, 0
    eval_count = max(1, round(pose_count * 0.2))
    return pose_count - eval_count, eval_count


def _default_evaluation_tasks(scale_status: str) -> tuple[EvaluationTask, ...]:
    tasks = [
        EvaluationTask(
            task_id="localization",
            label="Visual localization",
            kind="pose-estimation",
            metrics=("translation-error", "rotation-error", "tracking-success-rate"),
            required_sensors=("rgb-forward",),
        ),
        EvaluationTask(
            task_id="viewpoint-planning",
            label="Viewpoint planning",
            kind="planning",
            metrics=("coverage-gain", "path-length", "render-validity"),
            required_sensors=("rgb-forward", "depth-proxy"),
        ),
    ]
    if scale_status in {"metric", "estimated-metric"}:
        tasks.extend(
            [
                EvaluationTask(
                    task_id="waypoint-navigation",
                    label="Waypoint navigation",
                    kind="navigation",
                    metrics=("goal-success", "collision-rate", "path-efficiency"),
                    required_sensors=("rgb-forward", "lidar-ray-proxy"),
                ),
                EvaluationTask(
                    task_id="mapping-coverage",
                    label="Mapping coverage",
                    kind="mapping",
                    metrics=("surface-coverage", "frontier-count", "revisit-rate"),
                    required_sensors=("rgb-forward", "depth-proxy"),
                ),
            ]
        )
    return tuple(tasks)


def _require_unique_scene_ids(scenes: tuple[SceneEnvironment, ...]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for scene in scenes:
        if scene.scene_id in seen:
            duplicates.append(scene.scene_id)
        seen.add(scene.scene_id)
    if duplicates:
        raise ValueError(f"duplicate simulation scene ids: {', '.join(sorted(duplicates))}")


def slugify(value: str) -> str:
    """Return a stable slug for external callers building scene IDs."""

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
