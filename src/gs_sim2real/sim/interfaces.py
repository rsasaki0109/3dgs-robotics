"""Minimal environment interfaces for Physical AI agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class Pose3D:
    """World-frame rigid-body pose for environment queries."""

    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    frame_id: str = "world"
    timestamp_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frameId": self.frame_id,
            "position": list(self.position),
            "orientationXyzw": list(self.orientation_xyzw),
        }
        if self.timestamp_seconds is not None:
            payload["timestampSeconds"] = self.timestamp_seconds
        return payload


@dataclass(frozen=True, slots=True)
class AgentAction:
    """Action contract for step-based agents."""

    action_type: str
    values: Mapping[str, float]
    duration_seconds: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.action_type,
            "values": dict(self.values),
            "durationSeconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    """Request for rendering one sensor observation at a pose."""

    pose: Pose3D
    sensor_id: str
    outputs: tuple[str, ...] = ("rgb",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose": self.pose.to_dict(),
            "sensorId": self.sensor_id,
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """Metadata wrapper for an observation produced by an environment."""

    sensor_id: str
    pose: Pose3D
    outputs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensorId": self.sensor_id,
            "pose": self.pose.to_dict(),
            "outputs": dict(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class KinematicState:
    """Per-step kinematic estimate of an agent.

    ``linear_velocity_world`` is metres-per-second in the scene world frame.
    ``angular_velocity_body`` is radians-per-second in the agent body frame
    (the conventional gyroscope axes). ``linear_acceleration_body`` is
    metres-per-second-squared in the agent body frame (the conventional
    accelerometer axes); gravity is **not** added — it is a kinematic
    finite-difference accelerometer, not a full inertial-frame model.
    ``step_dt_seconds`` is the duration of the step that produced the
    estimate; ``0.0`` indicates the value is the post-reset zero state and
    no finite-difference is available yet.
    """

    linear_velocity_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    step_dt_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "linearVelocityWorld": list(self.linear_velocity_world),
            "angularVelocityBody": list(self.angular_velocity_body),
            "linearAccelerationBody": list(self.linear_acceleration_body),
            "stepDtSeconds": self.step_dt_seconds,
        }


@dataclass(frozen=True, slots=True)
class CollisionQuery:
    """Collision query result for a pose or footprint."""

    pose: Pose3D
    collides: bool
    reason: str
    clearance_meters: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pose": self.pose.to_dict(),
            "collides": self.collides,
            "reason": self.reason,
        }
        if self.clearance_meters is not None:
            payload["clearanceMeters"] = self.clearance_meters
        return payload


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """Evaluation summary for an agent trajectory."""

    metrics: Mapping[str, float]
    passed: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "notes": list(self.notes),
        }


@runtime_checkable
class PhysicalAIEnvironment(Protocol):
    """Interface expected from concrete 3DGS Robotics simulation backends."""

    def reset(self, scene_id: str, *, seed: int | None = None) -> Mapping[str, Any]:
        """Reset an episode for a scene and return initial state metadata."""

    def step(self, action: AgentAction) -> Mapping[str, Any]:
        """Advance the agent by one action and return transition metadata."""

    def render_observation(self, request: ObservationRequest) -> Observation:
        """Render or synthesize one observation for the requested sensor."""

    def query_collision(self, pose: Pose3D) -> CollisionQuery:
        """Check whether a pose or footprint collides with the environment."""

    def sample_goal(self, scene_id: str, *, seed: int | None = None) -> Pose3D:
        """Sample a goal pose inside the requested scene."""

    def score_trajectory(self, scene_id: str, trajectory: Sequence[Pose3D]) -> TrajectoryScore:
        """Score an agent trajectory against the scene task contract."""
