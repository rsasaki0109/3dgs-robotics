"""Gymnasium-style adapters for route policy workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import base64
import math
from typing import TYPE_CHECKING, Any

from .interfaces import ObservationRequest, PhysicalAIEnvironment, Pose3D
from .policy_feedback import RoutePolicySample, RouteRewardWeights, build_route_policy_sample
from .policy_sensor_noise import (
    RoutePolicySensorNoiseProfile,
    apply_sensor_noise_to_pose,
    sensor_noise_rng,
)
from .route_execution import rollout_route
from .route_planning import RouteCandidate

if TYPE_CHECKING:
    from .policy_trace import RoutePolicyTraceEmitter


RoutePolicyAction = RouteCandidate | Pose3D | Mapping[str, Any] | Sequence[Pose3D] | Sequence[float]


@dataclass(frozen=True, slots=True)
class RoutePolicyEnvConfig:
    """Configuration for a Gymnasium-style route policy adapter."""

    scene_id: str | None = None
    max_steps: int = 32
    goal_tolerance_meters: float = 0.25
    action_type: str = "teleport"
    segment_duration_seconds: float = 1.0
    stop_on_collision: bool = True
    reward_weights: RouteRewardWeights = field(default_factory=RouteRewardWeights)
    goal_reward: float = 1.0
    truncation_penalty: float = 0.0
    route_id_prefix: str = "policy-route"
    sensor_noise_profile: RoutePolicySensorNoiseProfile | None = None

    def __post_init__(self) -> None:
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        _non_negative_float(self.goal_tolerance_meters, "goal_tolerance_meters")
        _positive_float(self.segment_duration_seconds, "segment_duration_seconds")


@dataclass(frozen=True, slots=True)
class RoutePolicyEnvState:
    """Episode state exposed by the route policy adapter."""

    scene_id: str
    episode_index: int
    step_index: int
    pose: Pose3D
    goal: Pose3D
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "episodeIndex": self.episode_index,
            "stepIndex": self.step_index,
            "pose": self.pose.to_dict(),
            "goal": self.goal.to_dict(),
            "done": self.done,
        }


class RoutePolicyGymAdapter:
    """Small Gymnasium-compatible wrapper around a Physical AI environment.

    The class intentionally avoids importing ``gymnasium``. It follows the same
    call shape so callers can train against ``reset``/``step`` today and wrap it
    in a concrete ``gymnasium.Env`` later without changing the simulation core.
    """

    metadata = {"render_modes": ()}

    def __init__(
        self,
        environment: PhysicalAIEnvironment,
        config: RoutePolicyEnvConfig,
        *,
        trace_emitter: RoutePolicyTraceEmitter | None = None,
    ):
        self.environment = environment
        self.config = config
        self._episode_index = -1
        self._state: RoutePolicyEnvState | None = None
        self._reset_seed: int | None = None
        self._previous_obstacle_positions: dict[str, tuple[float, float, float]] = {}
        self._trace_emitter = trace_emitter

    @property
    def trace_emitter(self) -> RoutePolicyTraceEmitter | None:
        return self._trace_emitter

    @property
    def state(self) -> RoutePolicyEnvState:
        if self._state is None:
            raise RuntimeError("adapter has not been reset")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
        scene_id: str | None = None,
        goal: Pose3D | Mapping[str, Any] | Sequence[float] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Reset the wrapped environment and return ``(observation, info)``."""

        reset_options = dict(options or {})
        resolved_scene_id = _resolve_scene_id(scene_id=scene_id, options=reset_options, config=self.config)
        reset_payload = self.environment.reset(resolved_scene_id, seed=seed)
        pose = _pose_from_reset_payload(reset_payload)
        resolved_goal = self._resolve_goal(resolved_scene_id, pose=pose, seed=seed, goal=goal, options=reset_options)
        self._episode_index += 1
        self._reset_seed = None if seed is None else int(seed)
        self._previous_obstacle_positions = {}
        self._state = RoutePolicyEnvState(
            scene_id=resolved_scene_id,
            episode_index=self._episode_index,
            step_index=0,
            pose=pose,
            goal=resolved_goal,
        )
        if self._trace_emitter is not None:
            self._trace_emitter.begin_episode(
                scene_id=resolved_scene_id,
                episode_index=self._episode_index,
            )
        observation = self._observation_features(self.state)
        info = self._info(reset_payload=reset_payload)
        return observation, info

    def step(self, action: RoutePolicyAction) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
        """Execute one route action and return Gymnasium-style step output."""

        state = self.state
        if state.done:
            raise RuntimeError("episode is done; call reset before stepping again")

        route = _route_candidate_from_action(
            action,
            current_pose=state.pose,
            route_id=_route_id(self.config, state),
        )
        rollout = rollout_route(
            self.environment,
            route,
            action_type=self.config.action_type,
            segment_duration_seconds=self.config.segment_duration_seconds,
            stop_on_collision=self.config.stop_on_collision,
        )
        sample = build_route_policy_sample(rollout, weights=self.config.reward_weights)
        next_pose = _pose_after_rollout(rollout, fallback=state.pose)
        next_step_index = state.step_index + 1
        goal_distance = _pose_distance(next_pose, state.goal)
        blocked = rollout.blocked_step_index is not None
        goal_reached = goal_distance <= self.config.goal_tolerance_meters
        terminated = blocked or goal_reached
        truncated = not terminated and next_step_index >= self.config.max_steps
        reward = sample.reward.reward
        if goal_reached:
            reward += self.config.goal_reward
        if truncated:
            reward += self.config.truncation_penalty

        self._state = RoutePolicyEnvState(
            scene_id=state.scene_id,
            episode_index=state.episode_index,
            step_index=next_step_index,
            pose=next_pose,
            goal=state.goal,
            done=terminated or truncated,
        )
        observation = self._observation_features(self.state, sample=sample)
        if self._trace_emitter is not None:
            self._trace_emitter.record_step(
                scene_id=state.scene_id,
                episode_index=state.episode_index,
                step_index=state.step_index,
                next_observation=observation,
                blocked=blocked,
                goal_reached=goal_reached,
                truncated=truncated,
                terminated=terminated,
            )
        info = self._info(
            route=route,
            rollout=rollout.to_dict(),
            policy_sample=sample,
            blocked=blocked,
            goal_reached=goal_reached,
            terminated=terminated,
            truncated=truncated,
            termination_reason=_termination_reason(
                blocked=blocked,
                goal_reached=goal_reached,
                truncated=truncated,
            ),
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Match the Gymnasium ``Env.close`` surface."""

    def _resolve_goal(
        self,
        scene_id: str,
        *,
        pose: Pose3D,
        seed: int | None,
        goal: Pose3D | Mapping[str, Any] | Sequence[float] | None,
        options: Mapping[str, Any],
    ) -> Pose3D:
        option_goal = goal if goal is not None else options.get("goal")
        if option_goal is not None:
            return _pose_from_value(option_goal, template=pose)
        goal_seed = seed + 1 if seed is not None else None
        return self.environment.sample_goal(scene_id, seed=goal_seed)

    def _observation_features(
        self,
        state: RoutePolicyEnvState,
        *,
        sample: RoutePolicySample | None = None,
    ) -> dict[str, float]:
        observed_pose, observed_goal = self._apply_sensor_noise(state)
        max_steps = float(self.config.max_steps)
        goal_distance = _pose_distance(observed_pose, observed_goal)
        features = {
            "episode-index": float(state.episode_index),
            "episode-step-index": float(state.step_index),
            "episode-progress": min(float(state.step_index) / max_steps, 1.0),
            "remaining-step-fraction": max((max_steps - float(state.step_index)) / max_steps, 0.0),
            "goal-distance-meters": goal_distance,
            "goal-tolerance-meters": float(self.config.goal_tolerance_meters),
            "goal-reached": _bool_feature(goal_distance <= self.config.goal_tolerance_meters),
            **_pose_features("pose", observed_pose),
            **_pose_features("goal", observed_goal),
            "goal-delta-x": float(observed_goal.position[0] - observed_pose.position[0]),
            "goal-delta-y": float(observed_goal.position[1] - observed_pose.position[1]),
            "goal-delta-z": float(observed_goal.position[2] - observed_pose.position[2]),
        }
        features.update(self._dynamic_obstacle_features(state, observed_pose))
        features.update(self._imu_features(state))
        if sample is not None:
            features.update(_prefixed("route", sample.observation.features))
        return _finite_features(features)

    def _imu_features(self, state: RoutePolicyEnvState) -> dict[str, float]:
        """Return the IMU observation block, empty when the env has no imu-proxy.

        The sensor is rendered through ``render_observation`` so the same
        ``RawSensorNoiseProfile`` / ``apply_raw_sensor_noise_to_observation``
        seam that perturbs angular-velocity / linear-acceleration outputs
        on the env side flows into the policy feature dict. ``step_dt_seconds``
        is surfaced so policies can gate on the post-reset / post-teleport
        zero state (``imu-step-dt-seconds == 0`` means the finite-difference
        readings are not yet meaningful for this step).
        """

        if not hasattr(self.environment, "render_observation"):
            return {}
        request = ObservationRequest(
            pose=state.pose,
            sensor_id="imu-proxy",
            outputs=("angular-velocity", "linear-acceleration"),
        )
        try:
            observation = self.environment.render_observation(request)
        except (ValueError, KeyError, AttributeError):
            return {}
        outputs = observation.outputs
        angular = _decode_imu_vector_block(outputs.get("angular-velocity"), "angularVelocityBase64")
        linear = _decode_imu_vector_block(outputs.get("linear-acceleration"), "linearAccelerationBase64")
        if angular is None and linear is None:
            return {}
        features: dict[str, float] = {}
        step_dt = outputs.get("stepDtSeconds")
        if isinstance(step_dt, (int, float)):
            features["imu-step-dt-seconds"] = float(step_dt)
        if angular is not None:
            features["imu-angular-velocity-x"] = angular[0]
            features["imu-angular-velocity-y"] = angular[1]
            features["imu-angular-velocity-z"] = angular[2]
        if linear is not None:
            features["imu-linear-acceleration-x"] = linear[0]
            features["imu-linear-acceleration-y"] = linear[1]
            features["imu-linear-acceleration-z"] = linear[2]
        return features

    def _dynamic_obstacle_features(
        self,
        state: RoutePolicyEnvState,
        observed_pose: Pose3D,
    ) -> dict[str, float]:
        """Return the obstacle-awareness block, empty when no timeline is set.

        Distances and bearings are measured from the same ``observed_pose``
        the policy already sees — so a sensor-noise profile shifts obstacle
        observations alongside the pose / goal observations, keeping the
        feature block consistent under partial-information benchmarks.

        Both the nearest and the second-nearest obstacle are surfaced so a
        policy can tell apart a lane with one obstacle in it from a lane with
        two obstacles at similar clearance; the second-nearest block stays
        empty when fewer than two obstacles are present.

        Obstacle positions are resolved through
        :meth:`DynamicObstacleTimeline.step_positions`, threading the
        previous step's resolved positions in as ``previous_positions``. For
        policy-driven obstacles this means the ``ObstaclePolicyContext`` they
        receive carries up-to-date peer locations (so a
        ``MaintainSeparationObstaclePolicy`` actually sees its siblings),
        and a new ``peer-min-separation-meters`` feature surfaces the
        minimum pair-wise clearance between obstacle spheres so a learned
        policy can tell apart a tight cluster from a spread-out swarm.
        Policyless obstacles fall through to the legacy chase / flee /
        waypoint behaviour, so this code is a no-op for v1 scenario JSON.
        """

        timeline = getattr(self.environment, "dynamic_obstacles", None)
        if timeline is None or timeline.obstacle_count == 0:
            self._previous_obstacle_positions = {}
            return {}

        observed_position = tuple(observed_pose.position)
        resolved_positions = timeline.step_positions(
            state.step_index,
            agent_position=observed_position,
            previous_positions=self._previous_obstacle_positions,
        )
        self._previous_obstacle_positions = dict(resolved_positions)

        ranked: list[tuple[float, tuple[float, float, float], float]] = []
        radii_by_id: dict[str, float] = {}
        for obstacle in timeline.obstacles:
            centre = resolved_positions[obstacle.obstacle_id]
            radius = float(obstacle.radius_meters)
            radii_by_id[obstacle.obstacle_id] = radius
            distance = math.dist(observed_position, centre)
            clearance = max(0.0, distance - radius)
            ranked.append((clearance, centre, _obstacle_reactive_mode(obstacle)))
        ranked.sort(key=lambda entry: entry[0])

        features: dict[str, float] = {"dynamic-obstacle-count": float(timeline.obstacle_count)}
        features.update(_obstacle_block("nearest-dynamic-obstacle", ranked[0], observed_pose))
        if len(ranked) >= 2:
            features.update(_obstacle_block("second-nearest-dynamic-obstacle", ranked[1], observed_pose))
            features["peer-min-separation-meters"] = _peer_min_separation_meters(
                timeline.obstacles, resolved_positions, radii_by_id
            )
        return features

    def _apply_sensor_noise(self, state: RoutePolicyEnvState) -> tuple[Pose3D, Pose3D]:
        """Return the observed ``(pose, goal)`` with sensor noise applied."""

        profile = self.config.sensor_noise_profile
        if profile is None or profile.is_noise_free:
            return state.pose, state.goal
        pose_rng = sensor_noise_rng(
            base_seed=self._reset_seed,
            profile_id=profile.profile_id,
            episode_index=state.episode_index,
            step_index=state.step_index,
            kind="pose",
        )
        observed_pose = apply_sensor_noise_to_pose(state.pose, profile, rng=pose_rng)
        goal_rng = sensor_noise_rng(
            base_seed=self._reset_seed,
            profile_id=profile.profile_id,
            episode_index=state.episode_index,
            step_index=state.step_index,
            kind="goal",
        )
        observed_goal = apply_sensor_noise_to_pose(
            state.goal,
            profile,
            rng=goal_rng,
            perturb_heading=False,
            position_std_override=profile.goal_position_std_meters,
        )
        return observed_pose, observed_goal

    def _info(self, **extra: Any) -> dict[str, Any]:
        state = self.state
        info: dict[str, Any] = {
            "sceneId": state.scene_id,
            "episodeIndex": state.episode_index,
            "stepIndex": state.step_index,
            "pose": state.pose.to_dict(),
            "goal": state.goal.to_dict(),
            "goalDistanceMeters": _pose_distance(state.pose, state.goal),
            "done": state.done,
        }
        policy_sample = extra.pop("policy_sample", None)
        if policy_sample is not None:
            info["policySample"] = policy_sample.to_dict()
        route = extra.pop("route", None)
        if route is not None:
            info["route"] = route.to_dict()
        info.update(extra)
        return info


def make_route_policy_env(
    environment: PhysicalAIEnvironment,
    *,
    scene_id: str,
    trace_emitter: RoutePolicyTraceEmitter | None = None,
    **config_overrides: Any,
) -> RoutePolicyGymAdapter:
    """Build a route policy adapter with a concise factory call."""

    return RoutePolicyGymAdapter(
        environment,
        RoutePolicyEnvConfig(scene_id=scene_id, **config_overrides),
        trace_emitter=trace_emitter,
    )


def _resolve_scene_id(
    *,
    scene_id: str | None,
    options: Mapping[str, Any],
    config: RoutePolicyEnvConfig,
) -> str:
    selected = scene_id or options.get("sceneId") or options.get("scene_id") or config.scene_id
    if not selected:
        raise ValueError("scene_id must be provided in config, reset argument, or reset options")
    return str(selected)


def _route_id(config: RoutePolicyEnvConfig, state: RoutePolicyEnvState) -> str:
    return f"{config.route_id_prefix}-{state.episode_index}-{state.step_index}"


def _route_candidate_from_action(
    action: RoutePolicyAction,
    *,
    current_pose: Pose3D,
    route_id: str,
) -> RouteCandidate:
    if isinstance(action, RouteCandidate):
        return _anchored_route(action.route_id, current_pose, action.trajectory)
    if isinstance(action, Pose3D):
        return _anchored_route(route_id, current_pose, (action,))
    if isinstance(action, Mapping):
        return _route_candidate_from_mapping(action, current_pose=current_pose, fallback_route_id=route_id)
    if _is_position_sequence(action):
        return _anchored_route(route_id, current_pose, (_pose_from_value(action, template=current_pose),))
    if _is_sequence(action):
        poses = tuple(_pose_from_value(item, template=current_pose) for item in action)
        return _anchored_route(route_id, current_pose, poses)
    raise TypeError("unsupported route policy action")


def _route_candidate_from_mapping(
    action: Mapping[str, Any],
    *,
    current_pose: Pose3D,
    fallback_route_id: str,
) -> RouteCandidate:
    route_id = str(action.get("routeId") or action.get("route_id") or fallback_route_id)
    if "trajectory" in action:
        poses = tuple(_pose_from_value(item, template=current_pose) for item in _require_sequence(action["trajectory"]))
        return _anchored_route(route_id, current_pose, poses)
    if "waypoints" in action:
        poses = tuple(_pose_from_value(item, template=current_pose) for item in _require_sequence(action["waypoints"]))
        return _anchored_route(route_id, current_pose, poses)
    if "target" in action:
        return _anchored_route(route_id, current_pose, (_pose_from_value(action["target"], template=current_pose),))
    if {"x", "y", "z"}.issubset(action):
        return _anchored_route(route_id, current_pose, (_pose_from_value(action, template=current_pose),))
    raise ValueError("route policy action mapping must contain target, waypoints, trajectory, or x/y/z")


def _anchored_route(route_id: str, current_pose: Pose3D, poses: Sequence[Pose3D]) -> RouteCandidate:
    trajectory = tuple(poses)
    if not trajectory:
        trajectory = (current_pose,)
    elif trajectory[0] != current_pose:
        trajectory = (current_pose, *trajectory)
    return RouteCandidate(route_id, trajectory)


def _pose_from_reset_payload(payload: Mapping[str, Any]) -> Pose3D:
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("reset payload must contain state mapping")
    pose = state.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError("reset payload state must contain pose mapping")
    return _pose_from_value(pose)


def _pose_after_rollout(rollout: Any, *, fallback: Pose3D) -> Pose3D:
    for outcome in reversed(rollout.outcomes):
        transition = outcome.transition
        if not isinstance(transition, Mapping):
            continue
        state = transition.get("state")
        if not isinstance(state, Mapping):
            continue
        pose = state.get("pose")
        if isinstance(pose, Mapping):
            return _pose_from_value(pose, template=fallback)
    return fallback


def _pose_from_value(value: Any, *, template: Pose3D | None = None) -> Pose3D:
    if isinstance(value, Pose3D):
        return value
    if isinstance(value, Mapping):
        if "position" in value:
            position = _float_tuple(value["position"], expected_size=3, field_name="position")
        elif {"x", "y", "z"}.issubset(value):
            position = (float(value["x"]), float(value["y"]), float(value["z"]))
        else:
            raise ValueError("pose mapping must contain position or x/y/z")
        orientation_value = value.get("orientationXyzw", value.get("orientation_xyzw"))
        orientation = (
            _float_tuple(orientation_value, expected_size=4, field_name="orientationXyzw")
            if orientation_value is not None
            else template.orientation_xyzw
            if template is not None
            else (0.0, 0.0, 0.0, 1.0)
        )
        return Pose3D(
            position=position,
            orientation_xyzw=orientation,
            frame_id=str(value.get("frameId", value.get("frame_id", template.frame_id if template else "world"))),
            timestamp_seconds=_optional_float(value.get("timestampSeconds", value.get("timestamp_seconds"))),
        )
    if _is_position_sequence(value):
        position = _float_tuple(value, expected_size=3, field_name="position")
        return Pose3D(
            position=position,
            orientation_xyzw=template.orientation_xyzw if template is not None else (0.0, 0.0, 0.0, 1.0),
            frame_id=template.frame_id if template is not None else "world",
        )
    raise TypeError("value cannot be converted to Pose3D")


def _obstacle_block(
    prefix: str,
    ranked_entry: tuple[float, tuple[float, float, float], float],
    observed_pose: Pose3D,
) -> dict[str, float]:
    clearance, centre, reactive_mode = ranked_entry
    delta_x = float(centre[0] - observed_pose.position[0])
    delta_y = float(centre[1] - observed_pose.position[1])
    planar = math.hypot(delta_x, delta_y)
    bearing = math.atan2(delta_y, delta_x) if planar > 0.0 else 0.0
    return {
        f"{prefix}-distance-meters": float(clearance),
        f"{prefix}-bearing-radians": bearing,
        f"{prefix}-bearing-x": delta_x / planar if planar > 0.0 else 0.0,
        f"{prefix}-bearing-y": delta_y / planar if planar > 0.0 else 0.0,
        f"{prefix}-reactive-mode": float(reactive_mode),
    }


def _obstacle_reactive_mode(obstacle: Any) -> float:
    """Return +1.0 when the obstacle chases the agent, -1.0 when it flees, 0.0 otherwise."""

    if getattr(obstacle, "chase_target_agent", False):
        return 1.0
    if getattr(obstacle, "flee_from_agent", False):
        return -1.0
    return 0.0


def _pose_features(prefix: str, pose: Pose3D) -> dict[str, float]:
    return {
        f"{prefix}-position-x": float(pose.position[0]),
        f"{prefix}-position-y": float(pose.position[1]),
        f"{prefix}-position-z": float(pose.position[2]),
        f"{prefix}-orientation-x": float(pose.orientation_xyzw[0]),
        f"{prefix}-orientation-y": float(pose.orientation_xyzw[1]),
        f"{prefix}-orientation-z": float(pose.orientation_xyzw[2]),
        f"{prefix}-orientation-w": float(pose.orientation_xyzw[3]),
    }


def _termination_reason(*, blocked: bool, goal_reached: bool, truncated: bool) -> str | None:
    if blocked:
        return "blocked-route"
    if goal_reached:
        return "goal-reached"
    if truncated:
        return "max-steps"
    return None


def _prefixed(prefix: str, features: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}-{key}": float(value) for key, value in features.items()}


def _finite_features(features: Mapping[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in sorted((str(key), float(value)) for key, value in features.items())
        if math.isfinite(value)
    }


def _pose_distance(source: Pose3D, target: Pose3D) -> float:
    return math.dist(source.position, target.position)


def _bool_feature(value: bool) -> float:
    return 1.0 if value else 0.0


def _peer_min_separation_meters(
    obstacles: Sequence[Any],
    resolved_positions: Mapping[str, tuple[float, float, float]],
    radii_by_id: Mapping[str, float],
) -> float:
    """Minimum centre-to-centre clearance over all obstacle pairs (sphere-aware).

    Each pair contributes ``max(0, distance(centre_i, centre_j) - radius_i - radius_j)``
    so two overlapping obstacles read as zero rather than a negative value.
    Caller guarantees at least two obstacles when calling this helper, so the
    return is always finite.
    """

    minimum = math.inf
    for index, first in enumerate(obstacles):
        first_id = first.obstacle_id
        first_centre = resolved_positions[first_id]
        first_radius = radii_by_id[first_id]
        for second in obstacles[index + 1 :]:
            second_id = second.obstacle_id
            distance = math.dist(first_centre, resolved_positions[second_id])
            separation = distance - first_radius - radii_by_id[second_id]
            if separation < minimum:
                minimum = separation
    return max(0.0, minimum)


def _decode_imu_vector_block(block: Any, base64_key: str) -> tuple[float, float, float] | None:
    """Decode a float32-le-xyz IMU vector block into a 3-tuple of floats.

    ``imu-proxy`` observations from :class:`HeadlessPhysicalAIEnvironment`
    encode angular-velocity / linear-acceleration as a base64-encoded
    little-endian float32 triple under ``angularVelocityBase64`` /
    ``linearAccelerationBase64`` so the same payload survives raw-sensor
    noise perturbation. Return ``None`` when the block is missing or
    malformed so the caller can drop the feature instead of emitting NaN.
    """

    if not isinstance(block, Mapping):
        return None
    payload = block.get(base64_key)
    if not isinstance(payload, str):
        return None
    try:
        raw = base64.b64decode(payload.encode("ascii"))
    except (ValueError, TypeError):
        return None
    if len(raw) < 12:
        return None
    import numpy as np

    arr = np.frombuffer(raw, dtype="<f4", count=3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _is_position_sequence(value: Any) -> bool:
    if not _is_sequence(value) or len(value) != 3:
        return False
    return all(isinstance(component, (int, float)) for component in value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _require_sequence(value: Any) -> Sequence[Any]:
    if not _is_sequence(value):
        raise TypeError("route action field must be a sequence")
    return value


def _float_tuple(value: Any, *, expected_size: int, field_name: str) -> tuple[float, ...]:
    if not _is_sequence(value) or len(value) != expected_size:
        raise ValueError(f"{field_name} must contain {expected_size} values")
    return tuple(float(component) for component in value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _non_negative_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized
